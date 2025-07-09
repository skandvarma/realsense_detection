"""
CUDA-optimized YOLO model implementation with GPU acceleration.
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import warnings

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from ultralytics import YOLO
    from ultralytics.utils import ops
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

from .base_detector import BaseDetector, Detection, DetectionResult, ModelInfo
from ..utils.logger import get_logger


class YOLODetector(BaseDetector):
    """CUDA-optimized YOLO detector with GPU acceleration and TensorRT support."""

    def __init__(self, model_info: ModelInfo, config: Dict[str, Any]):
        """
        Initialize YOLO detector with CUDA optimization.

        Args:
            model_info: Model configuration and metadata
            config: System configuration dictionary
        """
        super().__init__(model_info, config)

        # CUDA configuration - set memory_fraction first
        self.memory_fraction = 0.8
        self.use_tensorrt = config.get('detection', {}).get('yolo', {}).get('use_tensorrt', False)  # Disable TRT by default
        self.use_fp16 = config.get('detection', {}).get('yolo', {}).get('use_fp16', True)
        self.batch_size = config.get('detection', {}).get('yolo', {}).get('batch_size', 1)
        self.device = self._setup_cuda_device()

        # Model components
        self.model = None
        self.tensorrt_engine = None
        self.cuda_stream = None
        self.gpu_memory_pool = None

        # Preprocessing parameters
        self.input_size = (model_info.input_width, model_info.input_height)
        self.normalize_params = self._get_normalization_params()

        # Performance optimization
        self.warmup_iterations = 5
        self.profiling_enabled = config.get('detection', {}).get('profile_cuda', False)

        # CUDA memory management
        self.max_batch_size = 8
        self.memory_fraction = 0.8

        self.logger.info(f"YOLODetector initialized on device: {self.device}")
        if self.use_fp16:
            self.logger.info("Mixed precision (FP16) enabled")
        if self.use_tensorrt and TENSORRT_AVAILABLE:
            self.logger.info("TensorRT optimization enabled")

    def _setup_cuda_device(self) -> torch.device:
        """Setup CUDA device with optimal configuration."""
        if torch.cuda.is_available():
            # Select best GPU
            device_count = torch.cuda.device_count()
            if device_count > 1:
                # Choose GPU with most memory
                best_gpu = 0
                max_memory = 0
                for i in range(device_count):
                    memory = torch.cuda.get_device_properties(i).total_memory
                    if memory > max_memory:
                        max_memory = memory
                        best_gpu = i
                device = torch.device(f'cuda:{best_gpu}')
            else:
                device = torch.device('cuda:0')

            # Set memory fraction safely
            try:
                if hasattr(self, 'memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
                else:
                    torch.cuda.set_per_process_memory_fraction(0.8)
            except Exception as e:
                self.logger.warning(f"Failed to set CUDA memory fraction: {e}")

            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            self.logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
            self.logger.info(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

            return device
        else:
            self.logger.warning("CUDA not available, falling back to CPU")
            return torch.device('cpu')

    def _get_normalization_params(self) -> Dict[str, List[float]]:
        """Get normalization parameters for YOLO models."""
        # Standard ImageNet normalization for YOLO
        return {
            'mean': [0.0, 0.0, 0.0],  # YOLO models typically don't use mean subtraction
            'std': [255.0, 255.0, 255.0]  # Normalize to 0-1 range
        }

    def load_model(self) -> bool:
        """Load YOLO model with CUDA optimization."""
        try:
            if not ULTRALYTICS_AVAILABLE:
                raise ImportError("Ultralytics not available. Please install: pip install ultralytics")

            # Load model
            self.logger.info(f"Loading YOLO model: {self.model_info.weights_path}")
            self.model = YOLO(self.model_info.weights_path)

            # Move to GPU
            if self.device.type == 'cuda':
                self.model.to(self.device)

                # Enable mixed precision
                if self.use_fp16:
                    self.model.model.half()

                # Create CUDA stream for async operations
                self.cuda_stream = torch.cuda.Stream()

                # Setup TensorRT optimization (disabled by default for stability)
                if self.use_tensorrt and TENSORRT_AVAILABLE:
                    try:
                        self._setup_tensorrt()
                    except Exception as e:
                        self.logger.warning(f"TensorRT setup failed, continuing without: {e}")
                        self.use_tensorrt = False

            # Update model info
            self.model_info.inference_device = str(self.device)
            self.model_info.fp16_enabled = self.use_fp16

            # Warmup (simplified)
            try:
                self._warmup_model()
            except Exception as e:
                self.logger.warning(f"Model warmup failed, continuing: {e}")

            self.is_loaded = True
            self.logger.info("YOLO model loaded successfully with CUDA optimization")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.last_error = str(e)
            return False

    def _setup_tensorrt(self):
        """Setup TensorRT optimization for inference acceleration."""
        try:
            if not TENSORRT_AVAILABLE:
                self.logger.warning("TensorRT not available, skipping optimization")
                return

            # Convert model to TensorRT
            self.logger.info("Setting up TensorRT optimization...")

            # Export to ONNX first
            dummy_input = torch.randn(1, 3, *self.input_size, device=self.device)
            onnx_path = self.model_info.weights_path.replace('.pt', '_tensorrt.onnx')

            torch.onnx.export(
                self.model.model,
                dummy_input,
                onnx_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                opset_version=11
            )

            # Build TensorRT engine
            engine_path = onnx_path.replace('.onnx', '.trt')
            self._build_tensorrt_engine(onnx_path, engine_path)

            self.logger.info("TensorRT optimization completed")

        except Exception as e:
            self.logger.warning(f"TensorRT setup failed: {e}")
            self.use_tensorrt = False

    def _build_tensorrt_engine(self, onnx_path: str, engine_path: str):
        """Build TensorRT engine from ONNX model."""
        if not TENSORRT_AVAILABLE:
            return

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()

        # Set memory pool
        config.max_workspace_size = 1 << 28  # 256MB

        # Enable FP16 if available
        if self.use_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Create network
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        # Parse ONNX model
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                self.logger.error("Failed to parse ONNX model")
                return

        # Build engine
        engine = builder.build_engine(network, config)

        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())

        self.tensorrt_engine = engine

    def _warmup_model(self):
        """Warmup model for optimal performance."""
        self.logger.info("Warming up YOLO model...")

        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with torch.cuda.stream(self.cuda_stream) if self.device.type == 'cuda' else torch.no_grad():
            for i in range(self.warmup_iterations):
                try:
                    _ = self.detect(dummy_image)
                except:
                    pass  # Ignore warmup errors

        # Clear cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        self.logger.info("Model warmup completed")

    def detect(self, image: np.ndarray, **kwargs) -> DetectionResult:
        """
        Perform CUDA-accelerated object detection.

        Args:
            image: Input image as numpy array
            **kwargs: Additional detection parameters
                - confidence_threshold: Override default confidence
                - iou_threshold: Override default IoU threshold
                - max_detections: Override max detections

        Returns:
            DetectionResult with detected objects
        """
        start_time = time.time()

        # Validate input
        if not self.validate_image(image):
            return DetectionResult(
                detections=[],
                frame_id=kwargs.get('frame_id', 0),
                timestamp=start_time,
                success=False,
                error_message="Invalid input image",
                model_name=self.model_info.model_name
            )

        if not self.is_loaded:
            return DetectionResult(
                detections=[],
                frame_id=kwargs.get('frame_id', 0),
                timestamp=start_time,
                success=False,
                error_message="Model not loaded",
                model_name=self.model_info.model_name
            )

        try:
            # Extract parameters
            confidence_threshold = kwargs.get('confidence_threshold', self.model_info.confidence_threshold)
            iou_threshold = kwargs.get('iou_threshold', self.model_info.iou_threshold)
            max_detections = kwargs.get('max_detections', self.model_info.max_detections)

            # Preprocessing
            preprocess_start = time.time()
            processed_image, scale_factors = self.preprocess_image(image)
            preprocess_time = time.time() - preprocess_start

            # Inference
            inference_start = time.time()
            with torch.cuda.stream(self.cuda_stream) if self.device.type == 'cuda' else torch.no_grad():
                if self.use_fp16 and self.device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        results = self.model(processed_image, conf=confidence_threshold,
                                           iou=iou_threshold, max_det=max_detections, verbose=False)
                else:
                    results = self.model(processed_image, conf=confidence_threshold,
                                       iou=iou_threshold, max_det=max_detections, verbose=False)

            inference_time = time.time() - inference_start

            # Postprocessing
            postprocess_start = time.time()
            detections = self.postprocess_outputs(results, image.shape[:2], scale_factors)
            postprocess_time = time.time() - postprocess_start

            # Update performance stats
            total_time = time.time() - start_time
            self.update_performance_stats(inference_time, len(detections), True)

            return DetectionResult(
                detections=detections,
                frame_id=kwargs.get('frame_id', 0),
                timestamp=start_time,
                preprocessing_time=preprocess_time,
                inference_time=inference_time,
                postprocessing_time=postprocess_time,
                model_name=self.model_info.model_name,
                model_version=self.model_info.model_version,
                frame_width=image.shape[1],
                frame_height=image.shape[0],
                fps=1.0 / total_time if total_time > 0 else 0.0,
                success=True
            )

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            self.update_performance_stats(0, 0, False)

            return DetectionResult(
                detections=[],
                frame_id=kwargs.get('frame_id', 0),
                timestamp=start_time,
                success=False,
                error_message=str(e),
                model_name=self.model_info.model_name
            )

    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[float, float]]:
        """
        CUDA-accelerated image preprocessing.

        Args:
            image: Input image (H, W, C)

        Returns:
            Tuple of (processed_tensor, scale_factors)
        """
        original_height, original_width = image.shape[:2]
        target_width, target_height = self.input_size

        # Calculate scale factors
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        scale = min(scale_x, scale_y)

        # Resize image maintaining aspect ratio (letterbox)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Create letterbox image
        letterbox = np.full((target_height, target_width, 3), 114, dtype=np.uint8)

        # Center the resized image
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        letterbox[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        # Convert to tensor and move to GPU
        if self.device.type == 'cuda':
            # Direct GPU upload for efficiency
            tensor = torch.from_numpy(letterbox).to(self.device, non_blocking=True)
        else:
            tensor = torch.from_numpy(letterbox)

        # Reorder dimensions and normalize (H, W, C) -> (C, H, W)
        tensor = tensor.permute(2, 0, 1).float()
        tensor = tensor / 255.0  # Normalize to 0-1

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor, (scale_x, scale_y)

    def postprocess_outputs(self, results, original_shape: Tuple[int, int],
                          scale_factors: Tuple[float, float]) -> List[Detection]:
        """
        GPU-optimized postprocessing of YOLO outputs.

        Args:
            results: YOLO model results
            original_shape: Original image shape (height, width)
            scale_factors: Scaling factors used in preprocessing

        Returns:
            List of Detection objects
        """
        detections = []

        try:
            if not results or len(results) == 0:
                return detections

            result = results[0]  # First (and only) result for single image

            if result.boxes is None or len(result.boxes) == 0:
                return detections

            # Get detection data
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            scale_x, scale_y = scale_factors
            original_height, original_width = original_shape

            # Process each detection
            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                try:
                    # Scale back to original image coordinates
                    x1, y1, x2, y2 = box

                    # Account for letterbox padding
                    target_width, target_height = self.input_size
                    scale = min(scale_x, scale_y)

                    # Remove letterbox padding
                    pad_x = (target_width - original_width * scale) / 2
                    pad_y = (target_height - original_height * scale) / 2

                    x1 = (x1 - pad_x) / scale
                    y1 = (y1 - pad_y) / scale
                    x2 = (x2 - pad_x) / scale
                    y2 = (y2 - pad_y) / scale

                    # Clip to image boundaries
                    x1 = max(0, min(x1, original_width))
                    y1 = max(0, min(y1, original_height))
                    x2 = max(0, min(x2, original_width))
                    y2 = max(0, min(y2, original_height))

                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Get class name
                    if class_id < len(self.model_info.class_names):
                        class_name = self.model_info.class_names[class_id]
                    else:
                        class_name = f"class_{class_id}"

                    # Create detection
                    detection = Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(conf),
                        class_id=int(class_id),
                        class_name=class_name,
                        detection_id=i,
                        metadata={
                            'model_type': 'yolo',
                            'model_name': self.model_info.model_name,
                            'device': str(self.device),
                            'fp16': self.use_fp16
                        }
                    )

                    detections.append(detection)

                except Exception as e:
                    self.logger.warning(f"Error processing detection {i}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Postprocessing failed: {e}")

        return detections

    def detect_batch(self, images: List[np.ndarray], **kwargs) -> List[DetectionResult]:
        """
        CUDA-accelerated batch detection for multiple images.

        Args:
            images: List of input images
            **kwargs: Detection parameters

        Returns:
            List of DetectionResult objects
        """
        if not images:
            return []

        # Limit batch size for memory management
        batch_size = min(len(images), self.max_batch_size)
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = self._process_batch(batch, **kwargs)
            results.extend(batch_results)

        return results

    def _process_batch(self, batch: List[np.ndarray], **kwargs) -> List[DetectionResult]:
        """Process a batch of images efficiently."""
        start_time = time.time()

        try:
            # Preprocess batch
            processed_tensors = []
            scale_factors_list = []

            for image in batch:
                if self.validate_image(image):
                    tensor, scale_factors = self.preprocess_image(image)
                    processed_tensors.append(tensor)
                    scale_factors_list.append(scale_factors)

            if not processed_tensors:
                return [DetectionResult(detections=[], frame_id=0, timestamp=start_time,
                                      success=False, error_message="No valid images in batch")]

            # Stack tensors for batch processing
            batch_tensor = torch.cat(processed_tensors, dim=0)

            # Batch inference
            with torch.cuda.stream(self.cuda_stream) if self.device.type == 'cuda' else torch.no_grad():
                if self.use_fp16 and self.device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        results = self.model(batch_tensor, verbose=False)
                else:
                    results = self.model(batch_tensor, verbose=False)

            # Process results
            detection_results = []
            for i, (result, image, scale_factors) in enumerate(zip(results, batch, scale_factors_list)):
                detections = self.postprocess_outputs([result], image.shape[:2], scale_factors)

                detection_result = DetectionResult(
                    detections=detections,
                    frame_id=kwargs.get('frame_id', i),
                    timestamp=start_time,
                    model_name=self.model_info.model_name,
                    frame_width=image.shape[1],
                    frame_height=image.shape[0],
                    success=True
                )

                detection_results.append(detection_result)

            return detection_results

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return [DetectionResult(detections=[], frame_id=0, timestamp=start_time,
                                  success=False, error_message=str(e))]

    def optimize_for_inference(self):
        """Apply additional optimizations for inference."""
        if self.device.type == 'cuda' and self.model:
            try:
                # Compile model for faster inference (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    self.model.model = torch.compile(self.model.model)
                    self.logger.info("Model compiled with torch.compile for optimization")

                # Enable CUDA graph if supported
                if hasattr(torch.cuda, 'CUDAGraph'):
                    self._setup_cuda_graph()

            except Exception as e:
                self.logger.warning(f"Advanced optimization failed: {e}")

    def _setup_cuda_graph(self):
        """Setup CUDA graph for maximum performance."""
        try:
            # Create static input for graph capture
            static_input = torch.randn(1, 3, *self.input_size, device=self.device)

            # Warmup
            for _ in range(3):
                _ = self.model.model(static_input)

            torch.cuda.synchronize()

            # Capture graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = self.model.model(static_input)

            self.cuda_graph = graph
            self.static_input = static_input
            self.static_output = static_output

            self.logger.info("CUDA graph optimization enabled")

        except Exception as e:
            self.logger.warning(f"CUDA graph setup failed: {e}")

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3   # GB
            total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3

            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'utilization_percent': (allocated / total) * 100
            }
        return {}

    def cleanup(self):
        """Clean up CUDA resources and model."""
        try:
            # Clear CUDA cache
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

                if self.cuda_stream:
                    self.cuda_stream.synchronize()

            # Clean up model
            if self.model:
                del self.model

            if self.tensorrt_engine:
                del self.tensorrt_engine

            # Reset GPU memory
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            super().cleanup()
            self.logger.info("YOLO detector cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# Register YOLO detector with factory
def register_yolo_detector():
    """Register YOLO detector with the factory."""
    try:
        from .detector_factory import DetectorFactory
        DetectorFactory.register_detector('yolo', YOLODetector)
    except ImportError:
        pass