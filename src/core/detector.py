# src/core/detector.py
"""
Optimized detector implementation with multiple backend support.
Supports TensorRT (NVIDIA), OpenVINO (Intel), and ONNX Runtime as fallback.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import cv2
import time

import structlog

logger = structlog.get_logger()


@dataclass
class Detection:
    """Single detection result"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class DetectorBackend(ABC):
    """Abstract base class for detector backends"""

    @abstractmethod
    async def initialize(self, model_path: str, **kwargs):
        """Initialize the backend with model"""
        pass

    @abstractmethod
    async def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection on single image"""
        pass

    @abstractmethod
    async def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        """Run detection on batch of images"""
        pass

    @abstractmethod
    def warmup(self, input_shape: Tuple[int, ...]):
        """Warmup the model with dummy input"""
        pass


class TensorRTBackend(DetectorBackend):
    """NVIDIA TensorRT backend for maximum GPU performance"""

    def __init__(self):
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None

    async def initialize(self, model_path: str, **kwargs):
        """Initialize TensorRT engine"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            self.trt = trt
            self.cuda = cuda

            # Create logger
            self.logger = trt.Logger(trt.Logger.WARNING)

            # Check if engine exists
            engine_path = Path(model_path).with_suffix('.engine')

            if engine_path.exists():
                # Load existing engine
                logger.info("Loading TensorRT engine", path=str(engine_path))
                with open(engine_path, 'rb') as f:
                    runtime = trt.Runtime(self.logger)
                    self.engine = runtime.deserialize_cuda_engine(f.read())
            else:
                # Build engine from ONNX
                logger.info("Building TensorRT engine from ONNX", path=model_path)
                self.engine = await self._build_engine(model_path, **kwargs)

                # Save engine
                with open(engine_path, 'wb') as f:
                    f.write(self.engine.serialize())

            # Create execution context
            self.context = self.engine.create_execution_context()

            # Setup I/O bindings
            self._setup_bindings()

            # Create CUDA stream
            self.stream = cuda.Stream()

            logger.info("TensorRT backend initialized successfully")

        except ImportError:
            raise RuntimeError("TensorRT not available. Please install tensorrt package.")

    async def _build_engine(self, onnx_path: str, fp16: bool = True,
                            max_batch_size: int = 8) -> 'trt.ICudaEngine':
        """Build TensorRT engine from ONNX model"""
        builder = self.trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(self.trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = self.trt.OnnxParser(network, self.logger)

        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(f"TensorRT parser error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")

        # Builder config
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB

        # Enable FP16 if supported
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(self.trt.BuilderFlag.FP16)
            logger.info("Enabled FP16 precision")

        # Dynamic shape optimization
        profile = builder.create_optimization_profile()

        # Get input shape from network
        input_tensor = network.get_input(0)
        input_shape = input_tensor.shape

        # Set dynamic batch size
        min_shape = (1, *input_shape[1:])
        opt_shape = (max_batch_size // 2, *input_shape[1:])
        max_shape = (max_batch_size, *input_shape[1:])

        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        # Build engine
        logger.info("Building TensorRT engine (this may take a while)...")
        engine = builder.build_engine(network, config)

        if not engine:
            raise RuntimeError("Failed to build TensorRT engine")

        return engine

    def _setup_bindings(self):
        """Setup input/output bindings"""
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = abs(np.prod(shape))
            dtype = self.trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = self.cuda.pagelocked_empty(size, dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)

            # Store binding info
            binding_info = {
                'name': binding,
                'shape': shape,
                'dtype': dtype,
                'host': host_mem,
                'device': device_mem
            }

            if self.engine.binding_is_input(binding):
                self.inputs.append(binding_info)
            else:
                self.outputs.append(binding_info)

            self.bindings.append(int(device_mem))

    async def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection on single image"""
        # Add batch dimension
        batch = image[np.newaxis, ...]
        results = await self.detect_batch(batch)
        return results[0] if results else []

    async def detect_batch(self, images: np.ndarray) -> List[List[Detection]]:
        """Run detection on batch of images"""
        batch_size = len(images)

        # Set dynamic batch size
        self.context.active_optimization_profile = 0
        self.context.set_binding_shape(0, (batch_size, *self.inputs[0]['shape'][1:]))

        # Prepare input
        input_batch = np.ascontiguousarray(images)

        # Copy input to device
        self.inputs[0]['host'][:] = input_batch.flatten()
        self.cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )

        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # Copy output to host
        for output in self.outputs:
            self.cuda.memcpy_dtoh_async(
                output['host'],
                output['device'],
                self.stream
            )

        # Synchronize
        self.stream.synchronize()

        # Post-process outputs
        raw_outputs = []
        for output in self.outputs:
            shape = self.engine.get_binding_shape(output['name'])
            shape = (batch_size, *shape[1:])
            raw_outputs.append(output['host'].reshape(shape))

        # Convert to detections
        return self._postprocess_batch(raw_outputs, batch_size)

    def _postprocess_batch(self, outputs: List[np.ndarray],
                           batch_size: int) -> List[List[Detection]]:
        """Post-process batch outputs to detections"""
        results = []

        for i in range(batch_size):
            # Extract detections for this image
            # Format: [batch, num_detections, 7] 
            # where 7 = [x1, y1, x2, y2, obj_conf, class_conf, class_id]
            detections = outputs[0][i]

            # Filter by confidence
            mask = detections[:, 4] > 0.001  # obj_conf threshold
            detections = detections[mask]

            # Apply NMS
            if len(detections) > 0:
                detections = self._nms(detections, iou_threshold=0.45)

            # Convert to Detection objects
            det_list = []
            for det in detections:
                det_list.append(Detection(
                    x1=float(det[0]),
                    y1=float(det[1]),
                    x2=float(det[2]),
                    y2=float(det[3]),
                    confidence=float(det[4] * det[5]),  # obj_conf * class_conf
                    class_id=int(det[6])
                ))

            results.append(det_list)

        return results

    def _nms(self, detections: np.ndarray, iou_threshold: float) -> np.ndarray:
        """Apply Non-Maximum Suppression"""
        if len(detections) == 0:
            return detections

        # Sort by confidence
        indices = np.argsort(detections[:, 4])[::-1]
        detections = detections[indices]

        keep = []
        while len(detections) > 0:
            # Keep highest confidence
            keep.append(detections[0])

            if len(detections) == 1:
                break

            # Calculate IoU with remaining
            ious = self._calculate_iou(detections[0, :4], detections[1:, :4])

            # Remove overlapping
            mask = ious < iou_threshold
            detections = detections[1:][mask]

        return np.array(keep)

    def _calculate_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Calculate IoU between one box and multiple boxes"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    def warmup(self, input_shape: Tuple[int, ...]):
        """Warmup the model"""
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        for _ in range(10):
            asyncio.run(self.detect(dummy_input))

        logger.info("TensorRT warmup completed")


class OpenVINOBackend(DetectorBackend):
    """Intel OpenVINO backend for Intel hardware acceleration"""

    def __init__(self):
        self.model = None
        self.compiled_model = None
        self.infer_request = None
        self.input_layer = None
        self.output_layer = None

    async def initialize(self, model_path: str, device: str = "AUTO", **kwargs):
        """Initialize OpenVINO model"""
        try:
            import openvino as ov

            self.core = ov.Core()

            # Check available devices
            available_devices = self.core.available_devices
            logger.info("Available OpenVINO devices", devices=available_devices)

            # Read model
            if model_path.endswith('.xml'):
                self.model = self.core.read_model(model_path)
            elif model_path.endswith('.onnx'):
                self.model = self.core.read_model(model_path)
            else:
                raise ValueError(f"Unsupported model format: {model_path}")

            # Optimize for throughput
            config = {
                "PERFORMANCE_HINT": "THROUGHPUT",
                "CACHE_DIR": "model_cache"
            }

            # Compile model
            self.compiled_model = self.core.compile_model(
                self.model,
                device_name=device,
                config=config
            )

            # Get input/output info
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)

            # Create infer request
            self.infer_request = self.compiled_model.create_infer_request()

            logger.info("OpenVINO backend initialized", device=device)

        except ImportError:
            raise RuntimeError("OpenVINO not available. Please install openvino package.")

    async def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection on single image"""
        # Run inference
        self.infer_request.infer({self.input_layer: image})

        # Get output
        output = self.infer_request.get_output_tensor(0).data

        # Post-process
        return self._postprocess(output)

    async def detect_batch(self, images: np.ndarray) -> List[List[Detection]]:
        """Run detection on batch of images"""
        # OpenVINO handles batching internally
        results = []

        for image in images:
            detections = await self.detect(image)
            results.append(detections)

        return results

    def _postprocess(self, output: np.ndarray) -> List[Detection]:
        """Post-process model output"""
        detections = []

        # Reshape output if needed
        if output.ndim == 3:
            output = output.reshape(-1, output.shape[-1])

        # Filter by confidence
        mask = output[:, 4] > 0.25
        output = output[mask]

        # Apply NMS
        if len(output) > 0:
            # Convert to xyxy format if needed
            boxes = output[:, :4]
            scores = output[:, 4]
            class_ids = output[:, 5].astype(int)

            # NMS using OpenCV
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                score_threshold=0.25,
                nms_threshold=0.45
            )

            if len(indices) > 0:
                indices = indices.flatten()

                for i in indices:
                    detections.append(Detection(
                        x1=float(boxes[i, 0]),
                        y1=float(boxes[i, 1]),
                        x2=float(boxes[i, 2]),
                        y2=float(boxes[i, 3]),
                        confidence=float(scores[i]),
                        class_id=int(class_ids[i])
                    ))

        return detections

    def warmup(self, input_shape: Tuple[int, ...]):
        """Warmup the model"""
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        for _ in range(10):
            self.infer_request.infer({self.input_layer: dummy_input})

        logger.info("OpenVINO warmup completed")


class OptimizedDetector:
    """Main detector class with automatic backend selection"""

    def __init__(self, model_path: str, device: str = "AUTO",
                 conf_threshold: float = 0.25,
                 nms_threshold: float = 0.45):
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.backend = None

        # Performance stats
        self.inference_times = []

    async def initialize(self):
        """Initialize detector with best available backend"""
        # Try backends in order of preference
        backends = []

        if self.device in ["GPU", "AUTO"]:
            # Try TensorRT for NVIDIA GPUs
            if self._check_tensorrt_available():
                backends.append(("TensorRT", TensorRTBackend))

        # Try OpenVINO for Intel hardware
        if self._check_openvino_available():
            backends.append(("OpenVINO", OpenVINOBackend))

        # Try each backend
        for name, backend_class in backends:
            try:
                logger.info(f"Trying {name} backend...")
                self.backend = backend_class()
                await self.backend.initialize(self.model_path, device=self.device)
                logger.info(f"Successfully initialized {name} backend")

                # Warmup
                self.backend.warmup((1, 3, 640, 640))
                break

            except Exception as e:
                logger.warning(f"Failed to initialize {name} backend: {e}")
                continue

        if not self.backend:
            raise RuntimeError("No suitable backend available")

    def _check_tensorrt_available(self) -> bool:
        """Check if TensorRT is available"""
        try:
            import tensorrt
            import pycuda.driver
            return True
        except ImportError:
            return False

    def _check_openvino_available(self) -> bool:
        """Check if OpenVINO is available"""
        try:
            import openvino
            return True
        except ImportError:
            return False

    async def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection on single image"""
        start_time = time.perf_counter()

        # Preprocess
        processed = self._preprocess(image)

        # Run detection
        detections = await self.backend.detect(processed)

        # Filter by confidence
        detections = [d for d in detections if d.confidence >= self.conf_threshold]

        # Record timing
        elapsed = time.perf_counter() - start_time
        self.inference_times.append(elapsed)

        return detections

    async def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        """Run detection on batch of images"""
        start_time = time.perf_counter()

        # Preprocess batch
        processed_batch = np.stack([self._preprocess(img) for img in images])

        # Run detection
        results = await self.backend.detect_batch(processed_batch)

        # Filter by confidence
        filtered_results = []
        for detections in results:
            filtered = [d for d in detections if d.confidence >= self.conf_threshold]
            filtered_results.append(filtered)

        # Record timing
        elapsed = time.perf_counter() - start_time
        self.inference_times.append(elapsed / len(images))  # Per image

        return filtered_results

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        # Resize with letterbox
        target_size = (640, 640)
        h, w = image.shape[:2]

        # Calculate scale
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        padded = np.full((target_size[0], target_size[1], 3), 114, dtype=np.uint8)

        # Calculate padding
        pad_h = (target_size[0] - new_h) // 2
        pad_w = (target_size[1] - new_w) // 2

        # Place resized image
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Convert to tensor format (CHW, normalized)
        tensor = padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        return tensor

    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {}

        times = self.inference_times[-100:]  # Last 100 inferences

        return {
            "avg_inference_time": np.mean(times),
            "min_inference_time": np.min(times),
            "max_inference_time": np.max(times),
            "fps": 1.0 / np.mean(times) if times else 0
        }


# Factory function
async def create_detector(config: dict) -> OptimizedDetector:
    """Create and initialize detector from config"""
    detector = OptimizedDetector(
        model_path=config['model_path'],
        device=config.get('device', 'AUTO'),
        conf_threshold=config.get('conf_threshold', 0.25),
        nms_threshold=config.get('nms_threshold', 0.45)
    )

    await detector.initialize()

    return detector