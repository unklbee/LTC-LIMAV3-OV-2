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
    """Improved frame pool with dynamic sizing"""

    def __init__(self, initial_size: int = 10, max_size: int = 50):
        self.pool = Queue(maxsize=max_size)
        self.max_size = max_size
        self.allocated_count = 0
        self._lock = threading.Lock()

        # Pre-allocate smaller initial pool
        for _ in range(initial_size):
            # Create empty buffer - will be resized on first use
            buffer = np.empty((480, 640, 3), dtype=np.uint8)
            self.pool.put(buffer)

    def get(self, timeout: float = 1.0) -> np.ndarray:
        """Get buffer from pool with dynamic allocation"""
        try:
            buffer = self.pool.get(timeout=timeout)
            with self._lock:
                self.allocated_count += 1
            return buffer
        except Empty:
            with self._lock:
                if self.allocated_count < self.max_size:
                    # Allocate new buffer
                    buffer = np.empty((480, 640, 3), dtype=np.uint8)
                    self.allocated_count += 1
                    logger.debug(f"Allocated new buffer, total: {self.allocated_count}")
                    return buffer
                else:
                    logger.warning("Max frame pool size reached, waiting...")
                    # Wait a bit longer
                    try:
                        buffer = self.pool.get(timeout=5.0)
                        return buffer
                    except Empty:
                        logger.error("Frame pool completely exhausted")
                        # Emergency allocation
                        return np.empty((480, 640, 3), dtype=np.uint8)

    def put(self, buffer: np.ndarray) -> None:
        """Return buffer to pool"""
        if buffer is None:
            return

        try:
            self.pool.put_nowait(buffer)
            with self._lock:
                self.allocated_count = max(0, self.allocated_count - 1)
        except:
            # Pool is full or other error
            with self._lock:
                self.allocated_count = max(0, self.allocated_count - 1)

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
        """Start processing pipeline - IMPROVED VERSION"""
        logger.info("Starting optimized pipeline",
                    batch_size=self.batch_size,
                    use_gpu=self.use_gpu_decode)

        # Validate video source
        if not video_source.isOpened():
            raise ValueError("Video source is not opened")

        # Initialize queues
        self._decode_queue = asyncio.Queue(maxsize=self.buffer_size)
        self._preprocess_queue = asyncio.Queue(maxsize=self.buffer_size)
        self._inference_queue = asyncio.Queue(maxsize=self.buffer_size)
        self._tracking_queue = asyncio.Queue(maxsize=self.buffer_size)

        # Get frame dimensions safely
        ret, sample_frame = video_source.read()
        if not ret or sample_frame is None:
            raise ValueError("Cannot read sample frame from video source")

        # Reset video to beginning
        video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Initialize improved frame pool
        self._frame_pool = FramePool(
            initial_size=self.buffer_size,
            max_size=self.buffer_size * 4
        )

        # Reset stop event
        self._stop_event = threading.Event()

        # Start pipeline stages
        tasks = [
            asyncio.create_task(self._decode_stage(video_source)),
            asyncio.create_task(self._preprocess_stage()),
            asyncio.create_task(self._inference_stage()),
            asyncio.create_task(self._tracking_stage()),
        ]

        frame_count = 0
        start_time = time.time()

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

                    # Yield frame
                    yield frame_data

                    frame_count += 1

                    # Log progress
                    if frame_count % 100 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        logger.info(f"Pipeline progress: {frame_count} frames, {fps:.1f} FPS")

                except asyncio.TimeoutError:
                    # Check if all stages are still running
                    if all(task.done() for task in tasks):
                        logger.info("All pipeline stages completed")
                        break
                    continue

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
        finally:
            # Cleanup
            self._stop_event.set()
            logger.info("Waiting for pipeline stages to stop...")

            # Cancel tasks and wait
            for task in tasks:
                task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("Pipeline stopped")

    async def _decode_stage(self, video_source):
        """Stage 1: Video decoding - FIXED VERSION"""
        frame_id = 0

        while not self._stop_event.is_set():
            start_time = time.perf_counter()

            # FIXED: OpenCV read() tidak menerima buffer parameter
            ret, frame = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                video_source.read
            )

            if not ret or frame is None:
                logger.info("End of video or read failed")
                break

            # Get buffer from pool dan copy frame ke buffer
            buffer = self._frame_pool.get()

            # Resize buffer jika perlu
            if buffer.shape != frame.shape:
                buffer = np.empty(frame.shape, dtype=np.uint8)

            # Copy frame data ke buffer
            np.copyto(buffer, frame)

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
        """Stage 3: Batched inference - FIXED VERSION"""
        while not self._stop_event.is_set():
            try:
                batch = await asyncio.wait_for(
                    self._preprocess_queue.get(),
                    timeout=0.5
                )

                start_time = time.perf_counter()

                # Prepare batch input
                batch_input = [f.preprocessed for f in batch]

                # FIXED: Check if detector method is async or sync
                if hasattr(self.detector, 'detect_batch'):
                    if asyncio.iscoroutinefunction(self.detector.detect_batch):
                        # Async method
                        detections = await self.detector.detect_batch(batch_input)
                    else:
                        # Sync method - run in executor
                        detections = await asyncio.get_event_loop().run_in_executor(
                            self._executor,
                            self.detector.detect_batch,
                            batch_input
                        )
                else:
                    # Fallback: process individually
                    detections = []
                    for frame_data in batch:
                        if asyncio.iscoroutinefunction(self.detector.detect):
                            det = await self.detector.detect(frame_data.preprocessed)
                        else:
                            det = await asyncio.get_event_loop().run_in_executor(
                                self._executor,
                                self.detector.detect,
                                frame_data.preprocessed
                            )
                        detections.append(det)

                # Assign detections to frames
                for frame_data, dets in zip(batch, detections):
                    frame_data.detections = dets
                    await self._inference_queue.put(frame_data)

                # Record timing
                elapsed = time.perf_counter() - start_time
                self._stage_timers['inference'].append(elapsed)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Inference stage error: {e}")
                continue

    def return_frame_buffer(self, frame_data: FrameData):
        """Return frame buffer to pool after display"""
        if hasattr(frame_data, 'raw_frame') and frame_data.raw_frame is not None:
            self._frame_pool.put(frame_data.raw_frame)

    async def _tracking_stage(self):
        """Stage 4: Multi-object tracking and counting - FIXED"""
        while not self._stop_event.is_set():
            try:
                frame_data = await asyncio.wait_for(
                    self._inference_queue.get(),
                    timeout=0.5
                )

                start_time = time.perf_counter()

                # Update tracks
                if frame_data.detections is not None:
                    tracks = self.tracker.update(frame_data.detections)
                    frame_data.tracks = tracks

                    # Update counts
                    if self.counter and tracks:
                        counts = self.counter.update(tracks)
                        frame_data.metadata['counts'] = counts
                else:
                    frame_data.tracks = []
                    frame_data.metadata['counts'] = {}

                # Send to output
                await self._tracking_queue.put(frame_data)

                # DON'T return frame buffer here - it will be returned after display
                # self._frame_pool.put(frame_data.raw_frame)  # REMOVED

                # Record timing
                elapsed = time.perf_counter() - start_time
                self._stage_timers['tracking'].append(elapsed)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Tracking stage error: {e}")
                continue

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame with better memory management"""
        try:
            if frame is None or frame.size == 0:
                logger.warning("Empty frame in preprocessing")
                return None

            # Get dimensions
            h, w = frame.shape[:2]
            target_size = (640, 640)

            # Calculate scaling
            scale = min(target_size[0] / h, target_size[1] / w)
            new_h, new_w = int(h * scale), int(w * scale)

            # Resize with proper interpolation
            if new_h != h or new_w != w:
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                resized = frame.copy()

            # Pad to target size
            if new_h != target_size[0] or new_w != target_size[1]:
                pad_h = (target_size[0] - new_h) // 2
                pad_w = (target_size[1] - new_w) // 2

                padded = cv2.copyMakeBorder(
                    resized,
                    pad_h, target_size[0] - new_h - pad_h,
                    pad_w, target_size[1] - new_w - pad_w,
                    cv2.BORDER_CONSTANT,
                    value=(114, 114, 114)
                )
            else:
                padded = resized

            # Convert to tensor format (CHW, normalized)
            # Use in-place operations to save memory
            normalized = padded.astype(np.float32, copy=False)
            normalized /= 255.0

            # Transpose to CHW format
            tensor = normalized.transpose(2, 0, 1)

            # Add batch dimension
            batched = tensor[None, ...]

            return batched

        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None

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