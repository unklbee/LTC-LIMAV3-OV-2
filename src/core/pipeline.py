# src/core/pipeline.py
"""
High-performance async pipeline for video processing.
This module implements a multi-stage pipeline with zero-copy optimization.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import AsyncIterator, Optional, Callable, Dict, List, Tuple
import numpy as np
import cv2
import time
from contextlib import asynccontextmanager

import structlog

logger = structlog.get_logger()


@dataclass
class FrameData:
    """Container for frame data through pipeline stages"""
    frame_id: int
    timestamp: float
    raw_frame: np.ndarray
    preprocessed: Optional[np.ndarray] = None
    detections: Optional[np.ndarray] = None
    tracks: Optional[List] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class PipelineStats:
    """Pipeline performance statistics"""
    frames_processed: int = 0
    frames_dropped: int = 0
    avg_fps: float = 0.0
    avg_latency: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)


class FramePool:
    """Pool of reusable frame buffers to minimize allocations"""

    def __init__(self, size: int, frame_shape: Tuple[int, int, int]):
        self.pool = Queue(maxsize=size)
        self.frame_shape = frame_shape

        # Pre-allocate buffers
        for _ in range(size):
            buffer = np.empty(frame_shape, dtype=np.uint8)
            self.pool.put(buffer)

    def get(self, timeout: float = 1.0) -> np.ndarray:
        """Get a buffer from pool"""
        try:
            return self.pool.get(timeout=timeout)
        except Empty:
            # Allocate new buffer if pool is empty
            logger.warning("Frame pool exhausted, allocating new buffer")
            return np.empty(self.frame_shape, dtype=np.uint8)

    def put(self, buffer: np.ndarray) -> None:
        """Return buffer to pool"""
        try:
            self.pool.put_nowait(buffer)
        except:
            pass  # Pool is full, let GC handle it


class OptimizedPipeline:
    """
    High-performance async pipeline with following stages:
    1. Decode: Video decoding (GPU accelerated if available)
    2. Preprocess: Frame preprocessing for inference
    3. Inference: Object detection (batched)
    4. Tracking: Multi-object tracking
    5. Counting: Vehicle counting logic
    """

    def __init__(self,
                 detector,
                 tracker,
                 counter,
                 buffer_size: int = 3,
                 batch_size: int = 4,
                 use_gpu_decode: bool = True):

        self.detector = detector
        self.tracker = tracker
        self.counter = counter

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.use_gpu_decode = use_gpu_decode

        self._executor = ThreadPoolExecutor(max_workers=4)
        self._stop_event = threading.Event()
        self._frame_pool: Optional[FramePool] = None
        self._stats = PipelineStats()

        # Pipeline queues
        self._decode_queue: asyncio.Queue = None
        self._preprocess_queue: asyncio.Queue = None
        self._inference_queue: asyncio.Queue = None
        self._tracking_queue: asyncio.Queue = None

        # Performance monitoring
        self._stage_timers: Dict[str, List[float]] = {
            'decode': [],
            'preprocess': [],
            'inference': [],
            'tracking': [],
            'counting': []
        }

    async def start(self, video_source) -> AsyncIterator[FrameData]:
        """Start processing pipeline"""
        logger.info("Starting optimized pipeline",
                    batch_size=self.batch_size,
                    use_gpu=self.use_gpu_decode)

        # Initialize queues
        self._decode_queue = asyncio.Queue(maxsize=self.buffer_size)
        self._preprocess_queue = asyncio.Queue(maxsize=self.buffer_size)
        self._inference_queue = asyncio.Queue(maxsize=self.buffer_size)
        self._tracking_queue = asyncio.Queue(maxsize=self.buffer_size)

        # Get frame dimensions
        ret, sample_frame = video_source.read()
        if not ret:
            raise ValueError("Cannot read from video source")
        video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset

        # Initialize frame pool
        self._frame_pool = FramePool(
            size=self.buffer_size * 3,
            frame_shape=sample_frame.shape
        )

        # Start pipeline stages
        tasks = [
            asyncio.create_task(self._decode_stage(video_source)),
            asyncio.create_task(self._preprocess_stage()),
            asyncio.create_task(self._inference_stage()),
            asyncio.create_task(self._tracking_stage()),
        ]

        try:
            # Yield processed frames
            while not self._stop_event.is_set():
                try:
                    frame_data = await asyncio.wait_for(
                        self._tracking_queue.get(),
                        timeout=1.0
                    )

                    # Update statistics
                    self._update_stats(frame_data)

                    yield frame_data

                except asyncio.TimeoutError:
                    continue

        finally:
            # Cleanup
            self._stop_event.set()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _decode_stage(self, video_source):
        """Stage 1: Video decoding with GPU acceleration if available"""
        frame_id = 0

        while not self._stop_event.is_set():
            start_time = time.perf_counter()

            # Get frame buffer from pool
            buffer = self._frame_pool.get()

            # Decode frame
            ret = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                video_source.read,
                buffer  # Read directly into buffer
            )

            if not ret:
                break

            # Create frame data
            frame_data = FrameData(
                frame_id=frame_id,
                timestamp=time.time(),
                raw_frame=buffer
            )

            # Send to next stage
            try:
                await self._decode_queue.put(frame_data)
            except asyncio.QueueFull:
                self._stats.frames_dropped += 1
                self._frame_pool.put(buffer)  # Return buffer

            # Record timing
            elapsed = time.perf_counter() - start_time
            self._stage_timers['decode'].append(elapsed)

            frame_id += 1

    async def _preprocess_stage(self):
        """Stage 2: Frame preprocessing for inference"""
        batch = []

        while not self._stop_event.is_set():
            try:
                # Collect frames for batch
                frame_data = await asyncio.wait_for(
                    self._decode_queue.get(),
                    timeout=0.1
                )

                start_time = time.perf_counter()

                # Preprocess frame
                preprocessed = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self._preprocess_frame,
                    frame_data.raw_frame
                )

                frame_data.preprocessed = preprocessed
                batch.append(frame_data)

                # Process batch when full or timeout
                if len(batch) >= self.batch_size:
                    await self._preprocess_queue.put(batch)
                    batch = []

                # Record timing
                elapsed = time.perf_counter() - start_time
                self._stage_timers['preprocess'].append(elapsed)

            except asyncio.TimeoutError:
                # Send partial batch
                if batch:
                    await self._preprocess_queue.put(batch)
                    batch = []

    async def _inference_stage(self):
        """Stage 3: Batched inference"""
        while not self._stop_event.is_set():
            try:
                batch = await asyncio.wait_for(
                    self._preprocess_queue.get(),
                    timeout=0.5
                )

                start_time = time.perf_counter()

                # Prepare batch tensor
                batch_tensor = np.stack([f.preprocessed for f in batch])

                # Run batched inference
                detections = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self.detector.detect_batch,
                    batch_tensor
                )

                # Assign detections to frames
                for frame_data, dets in zip(batch, detections):
                    frame_data.detections = dets
                    await self._inference_queue.put(frame_data)

                # Record timing
                elapsed = time.perf_counter() - start_time
                self._stage_timers['inference'].append(elapsed)

            except asyncio.TimeoutError:
                continue

    async def _tracking_stage(self):
        """Stage 4: Multi-object tracking and counting"""
        while not self._stop_event.is_set():
            try:
                frame_data = await asyncio.wait_for(
                    self._inference_queue.get(),
                    timeout=0.5
                )

                start_time = time.perf_counter()

                # Update tracks
                tracks = self.tracker.update(frame_data.detections)
                frame_data.tracks = tracks

                # Update counts
                if self.counter:
                    counts = self.counter.update(tracks)
                    frame_data.metadata['counts'] = counts

                # Send to output
                await self._tracking_queue.put(frame_data)

                # Return frame buffer to pool
                self._frame_pool.put(frame_data.raw_frame)

                # Record timing
                elapsed = time.perf_counter() - start_time
                self._stage_timers['tracking'].append(elapsed)

            except asyncio.TimeoutError:
                continue

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for inference"""
        # Letterbox resize
        h, w = frame.shape[:2]
        target_size = (640, 640)

        # Calculate scaling
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        pad_h = (target_size[0] - new_h) // 2
        pad_w = (target_size[1] - new_w) // 2

        padded = cv2.copyMakeBorder(
            resized,
            pad_h, target_size[0] - new_h - pad_h,
            pad_w, target_size[1] - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )

        # Convert to tensor format (CHW, normalized)
        tensor = padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        return tensor

    def _update_stats(self, frame_data: FrameData):
        """Update pipeline statistics"""
        self._stats.frames_processed += 1

        # Calculate average timings
        for stage, timings in self._stage_timers.items():
            if timings:
                # Keep only recent timings
                if len(timings) > 100:
                    timings = timings[-100:]
                    self._stage_timers[stage] = timings

                avg_time = sum(timings) / len(timings)
                self._stats.stage_timings[stage] = avg_time

        # Calculate FPS
        total_time = sum(self._stats.stage_timings.values())
        if total_time > 0:
            self._stats.avg_fps = 1.0 / total_time

        # Calculate latency
        if hasattr(frame_data, 'timestamp'):
            latency = time.time() - frame_data.timestamp
            self._stats.avg_latency = latency

    def get_stats(self) -> PipelineStats:
        """Get current pipeline statistics"""
        return self._stats

    async def stop(self):
        """Stop pipeline gracefully"""
        logger.info("Stopping pipeline",
                    frames_processed=self._stats.frames_processed)
        self._stop_event.set()

        # Cleanup
        if self._executor:
            self._executor.shutdown(wait=True)


# Pipeline factory function
async def create_pipeline(config: dict) -> OptimizedPipeline:
    """Create and configure pipeline based on config"""
    from src.core.detector import create_detector
    from src.core.tracker import create_tracker
    from src.core.counter import create_counter

    # Create components
    detector = await create_detector(config['detector'])
    tracker = create_tracker(config['tracker'])
    counter = create_counter(config['counter'])

    # Create pipeline
    pipeline = OptimizedPipeline(
        detector=detector,
        tracker=tracker,
        counter=counter,
        buffer_size=config.get('buffer_size', 3),
        batch_size=config.get('batch_size', 4),
        use_gpu_decode=config.get('use_gpu_decode', True)
    )

    return pipeline