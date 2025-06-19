# src/services/video_service.py
"""
Video capture service with support for multiple sources and GPU decoding.
"""

import asyncio
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, AsyncIterator, Tuple, Dict, Any
from pathlib import Path
import threading
from queue import Queue, Empty
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import structlog

logger = structlog.get_logger()


@dataclass
class VideoInfo:
    """Video source information"""
    width: int
    height: int
    fps: float
    total_frames: int
    codec: str
    is_live: bool
    source_type: str  # 'file', 'webcam', 'rtsp', 'http'


class VideoSource(ABC):
    """Abstract base class for video sources"""

    @abstractmethod
    async def open(self) -> bool:
        """Open video source"""
        pass

    @abstractmethod
    async def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame"""
        pass

    @abstractmethod
    async def close(self):
        """Close video source"""
        pass

    @abstractmethod
    def get_info(self) -> VideoInfo:
        """Get video source information"""
        pass

    @abstractmethod
    def seek(self, position: Union[int, float]):
        """Seek to position (frame number or timestamp)"""
        pass


class FileVideoSource(VideoSource):
    """Video file source with GPU decoding support"""

    def __init__(self, file_path: str, use_gpu: bool = True):
        self.file_path = Path(file_path)
        self.use_gpu = use_gpu and self._check_gpu_decode_available()
        self.cap = None
        self.info = None

    def _check_gpu_decode_available(self) -> bool:
        """Check if GPU decoding is available"""
        try:
            # Try to create VideoCapture with GPU backend
            test_cap = cv2.VideoCapture()

            # Check for NVIDIA Video Codec SDK
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                return True

            # Check for Intel QSV
            backends = cv2.videoio_registry.getBackends()
            if cv2.CAP_INTEL_MFX in backends:
                return True

            return False
        except:
            return False

    async def open(self) -> bool:
        """Open video file"""
        try:
            if not self.file_path.exists():
                raise FileNotFoundError(f"Video file not found: {self.file_path}")

            # Select backend based on GPU availability
            if self.use_gpu:
                # Try NVIDIA GPU decoding
                self.cap = cv2.VideoCapture(str(self.file_path), cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            else:
                # CPU decoding
                self.cap = cv2.VideoCapture(str(self.file_path))

            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.file_path}")

            # Get video properties
            self.info = VideoInfo(
                width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=self.cap.get(cv2.CAP_PROP_FPS),
                total_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                codec=self._get_codec_name(),
                is_live=False,
                source_type='file'
            )

            logger.info("Opened video file",
                        path=str(self.file_path),
                        gpu_decode=self.use_gpu,
                        info=self.info)

            return True

        except Exception as e:
            logger.error(f"Failed to open video file: {e}")
            return False

    def _get_codec_name(self) -> str:
        """Get video codec name"""
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        return codec

    async def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame"""
        if self.cap is None:
            return False, None

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        ret, frame = await loop.run_in_executor(None, self.cap.read)

        return ret, frame

    async def close(self):
        """Close video file"""
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_info(self) -> VideoInfo:
        """Get video information"""
        return self.info

    def seek(self, position: Union[int, float]):
        """Seek to position"""
        if self.cap:
            if isinstance(position, int):
                # Frame number
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            else:
                # Timestamp in seconds
                self.cap.set(cv2.CAP_PROP_POS_MSEC, position * 1000)


class WebcamSource(VideoSource):
    """Webcam/USB camera source"""

    def __init__(self, device_id: int = 0, resolution: Optional[Tuple[int, int]] = None,
                 fps: Optional[int] = None):
        self.device_id = device_id
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.info = None

    async def open(self) -> bool:
        """Open webcam"""
        try:
            # Use DirectShow on Windows for better performance
            if cv2.os.name == 'nt':
                self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.device_id)

            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open webcam: {self.device_id}")

            # Set resolution if specified
            if self.resolution:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # Set FPS if specified
            if self.fps:
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Get actual properties
            self.info = VideoInfo(
                width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=self.cap.get(cv2.CAP_PROP_FPS),
                total_frames=-1,  # Unknown for live source
                codec="RAW",
                is_live=True,
                source_type='webcam'
            )

            logger.info("Opened webcam", device_id=self.device_id, info=self.info)

            return True

        except Exception as e:
            logger.error(f"Failed to open webcam: {e}")
            return False

    async def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame"""
        if self.cap is None:
            return False, None

        # For webcam, always get the latest frame
        # Flush buffer by reading multiple times
        for _ in range(2):
            self.cap.grab()

        ret, frame = self.cap.retrieve()
        return ret, frame

    async def close(self):
        """Close webcam"""
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_info(self) -> VideoInfo:
        """Get video information"""
        return self.info

    def seek(self, position: Union[int, float]):
        """Seek not supported for live sources"""
        logger.warning("Seek not supported for webcam source")


class StreamSource(VideoSource):
    """Network stream source (RTSP, HTTP, etc.)"""

    def __init__(self, url: str, protocol: Optional[str] = None,
                 reconnect: bool = True, buffer_size: int = 1):
        self.url = url
        self.protocol = protocol or self._detect_protocol(url)
        self.reconnect = reconnect
        self.buffer_size = buffer_size
        self.cap = None
        self.info = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5

    def _detect_protocol(self, url: str) -> str:
        """Detect stream protocol from URL"""
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()

        if scheme in ['rtsp', 'rtsps']:
            return 'rtsp'
        elif scheme in ['http', 'https']:
            if any(ext in url.lower() for ext in ['.m3u8', '.m3u']):
                return 'hls'
            return 'http'
        elif scheme == 'rtmp':
            return 'rtmp'
        else:
            return 'unknown'

    async def open(self) -> bool:
        """Open network stream"""
        try:
            # Configure GStreamer pipeline for RTSP if available
            if self.protocol == 'rtsp' and self._check_gstreamer():
                pipeline = self._build_gstreamer_pipeline()
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            else:
                # Use FFmpeg backend
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

            # Set timeout for network streams
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open stream: {self.url}")

            # Get stream properties
            self.info = VideoInfo(
                width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=self.cap.get(cv2.CAP_PROP_FPS) or 25.0,  # Default to 25 if unknown
                total_frames=-1,
                codec=self._get_codec_name(),
                is_live=True,
                source_type=self.protocol
            )

            logger.info("Opened stream", url=self.url, protocol=self.protocol, info=self.info)

            self._reconnect_attempts = 0
            return True

        except Exception as e:
            logger.error(f"Failed to open stream: {e}")
            return False

    def _check_gstreamer(self) -> bool:
        """Check if GStreamer is available"""
        backends = cv2.videoio_registry.getBackends()
        return cv2.CAP_GSTREAMER in backends

    def _build_gstreamer_pipeline(self) -> str:
        """Build GStreamer pipeline for RTSP"""
        # Low latency RTSP pipeline
        pipeline = (
            f"rtspsrc location={self.url} latency=0 ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! appsink drop=true sync=false"
        )
        return pipeline

    def _get_codec_name(self) -> str:
        """Get stream codec name"""
        try:
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            return codec
        except:
            return "H264"  # Common for streams

    async def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame with reconnection support"""
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()

        # Handle disconnection
        if not ret and self.reconnect:
            logger.warning("Stream disconnected, attempting reconnect...")

            if await self._reconnect():
                # Try reading again
                ret, frame = self.cap.read()
            else:
                return False, None

        return ret, frame

    async def _reconnect(self) -> bool:
        """Attempt to reconnect to stream"""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return False

        self._reconnect_attempts += 1

        # Close current connection
        if self.cap:
            self.cap.release()

        # Wait before reconnecting
        await asyncio.sleep(min(self._reconnect_attempts * 2, 10))

        # Try to reconnect
        return await self.open()

    async def close(self):
        """Close stream"""
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_info(self) -> VideoInfo:
        """Get stream information"""
        return self.info

    def seek(self, position: Union[int, float]):
        """Seek not supported for live streams"""
        logger.warning("Seek not supported for stream source")


class AsyncVideoCapture:
    """Async wrapper for video capture with buffering"""

    def __init__(self, source: VideoSource, buffer_size: int = 5):
        self.source = source
        self.buffer_size = buffer_size
        self._frame_queue = asyncio.Queue(maxsize=buffer_size)
        self._capture_task = None
        self._stop_event = asyncio.Event()

    async def start(self) -> bool:
        """Start video capture"""
        if not await self.source.open():
            return False

        self._capture_task = asyncio.create_task(self._capture_loop())
        return True

    async def stop(self):
        """Stop video capture"""
        self._stop_event.set()

        if self._capture_task:
            await self._capture_task

        await self.source.close()

    async def _capture_loop(self):
        """Background capture loop"""
        while not self._stop_event.is_set():
            try:
                ret, frame = await self.source.read()

                if not ret:
                    logger.warning("Failed to read frame")
                    await asyncio.sleep(0.01)
                    continue

                # Add frame to queue (drop old frames if full)
                try:
                    self._frame_queue.put_nowait((time.time(), frame))
                except asyncio.QueueFull:
                    # Remove oldest frame and add new one
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait((time.time(), frame))
                    except:
                        pass

            except Exception as e:
                logger.error(f"Capture error: {e}")
                await asyncio.sleep(0.1)

    async def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Read next frame with timestamp"""
        try:
            timestamp, frame = await asyncio.wait_for(
                self._frame_queue.get(),
                timeout=1.0
            )
            return True, frame, timestamp
        except asyncio.TimeoutError:
            return False, None, 0.0

    def get_info(self) -> VideoInfo:
        """Get video information"""
        return self.source.get_info()


class VideoService:
    """Main video service for managing video sources"""

    def __init__(self):
        self.sources: Dict[str, AsyncVideoCapture] = {}

    async def create_source(self, source_config: Dict[str, Any]) -> str:
        """Create video source from configuration"""
        source_type = source_config.get('type', 'file')
        source_id = source_config.get('id', f"source_{len(self.sources)}")

        # Create appropriate source
        if source_type == 'file':
            source = FileVideoSource(
                file_path=source_config['path'],
                use_gpu=source_config.get('use_gpu', True)
            )
        elif source_type == 'webcam':
            source = WebcamSource(
                device_id=source_config.get('device_id', 0),
                resolution=source_config.get('resolution'),
                fps=source_config.get('fps')
            )
        elif source_type in ['rtsp', 'http', 'stream']:
            source = StreamSource(
                url=source_config['url'],
                protocol=source_config.get('protocol'),
                reconnect=source_config.get('reconnect', True)
            )
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        # Create async capture
        capture = AsyncVideoCapture(
            source,
            buffer_size=source_config.get('buffer_size', 5)
        )

        # Start capture
        if await capture.start():
            self.sources[source_id] = capture
            logger.info(f"Created video source: {source_id}")
            return source_id
        else:
            raise RuntimeError(f"Failed to create source: {source_id}")

    async def remove_source(self, source_id: str):
        """Remove video source"""
        if source_id in self.sources:
            await self.sources[source_id].stop()
            del self.sources[source_id]
            logger.info(f"Removed video source: {source_id}")

    async def read_frame(self, source_id: str) -> Tuple[bool, Optional[np.ndarray], float]:
        """Read frame from source"""
        if source_id not in self.sources:
            return False, None, 0.0

        return await self.sources[source_id].read()

    def get_source_info(self, source_id: str) -> Optional[VideoInfo]:
        """Get source information"""
        if source_id in self.sources:
            return self.sources[source_id].get_info()
        return None

    async def cleanup(self):
        """Cleanup all sources"""
        for source_id in list(self.sources.keys()):
            await self.remove_source(source_id)


# Factory function
def create_video_service() -> VideoService:
    """Create video service instance"""
    return VideoService()