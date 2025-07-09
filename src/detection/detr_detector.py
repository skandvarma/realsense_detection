"""
CUDA-accelerated DETR (Detection Transformer) implementation with GPU optimization.
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from transformers import DetrImageProcessor, DetrForObjectDetection
    from transformers import ConditionalDetrImageProcessor, ConditionalDetrForObjectDetection
    from transformers import DeformableDetrImageProcessor, DeformableDetrForObjectDetection
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .base_detector import BaseDetector, Detection, DetectionResult, ModelInfo
from ..utils.logger import get_logger


class DETRDetector(BaseDetector):
    """CUDA-accelerated DETR detector with transformer optimization."""

    def __init__(self, model_info: ModelInfo, config: Dict[str, Any]):
        """
        Initialize DETR detector with CUDA optimization.

        Args:
            model_info: Model configuration and metadata
            config: System configuration dictionary
        """
        super().__init__(model_info, config)

        # CUDA configuration
        self.device = self._setup_cuda_device()
        self.use_fp16 = config.get('detection', {}).get('detr', {}).get('use_fp16', True)
        self.use_tensorrt = config.get('detection', {}).get('detr', {}).get('use_tensorrt', True)
        self.batch_size = config.get('detection', {}).get('detr', {}).get('batch_size', 1)

        # Model components
        self.model = None
        self.processor = None
        self.model_variant = model_info.config.get('variant', 'detr-resnet-50')
        self.model_name_hf = model_info.config.get('model_name', 'facebook/detr-resnet-50')

        # CUDA optimization
        self.cuda_stream = None
        self.use_amp = True  # Automatic Mixed Precision
        self.scaler = None

        # Performance parameters
        self.max_queries = 100  # Maximum number of object queries
        self.warmup_iterations = 3
        self.memory_efficient = True

        # Preprocessing parameters
        self.input_size = (model_info.input_width, model_info.input_height)
        self.normalize_params = self._get_normalization_params()

        # Multi-GPU support
        self.use_multi_gpu = torch.cuda.device_count() > 1

        self.logger.info(f"DETRDetector initialized on device: {self.device}")
        self.logger.info(f"Model variant: {self.model_variant}")
        if self.use_fp16:
            self.logger.info("Mixed precision (FP16) enabled")

    def _setup_cuda_device(self) -> torch.device:
        """Setup CUDA device with optimal configuration for transformers."""
        if torch.cuda.is_available():
            # Select best GPU for transformer models (prioritize memory)
            device_count = torch.cuda.device_count()
            if device_count > 1:
                best_gpu = 0
                max_memory = 0
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    # Prefer newer architectures and more memory
                    memory_score = props.total_memory
                    if props.major >= 7:  # Volta or newer for tensor cores
                        memory_score *= 1.2
                    if memory_score > max_memory:
                        max_memory = memory_score
                        best_gpu = i
                device = torch.device(f'cuda:{best_gpu}')
            else:
                device = torch.device('cuda:0')

            # Transformer-specific optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # Enable tensor core usage for better FP16 performance
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True

            self.logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
            self.logger.info(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

            return device
        else:
            self.logger.warning("CUDA not available, falling back to CPU")
            return torch.device('cpu')

    def _get_normalization_params(self) -> Dict[str, List[float]]:
        """Get normalization parameters for DETR models."""
        # ImageNet normalization used by DETR
        return {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }

    def load_model(self) -> bool:
        """Load DETR model with CUDA optimization."""
        try:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not available. Please install: pip install transformers")

            if not PIL_AVAILABLE:
                raise ImportError("PIL not available. Please install: pip install Pillow")

            # Determine model and processor classes
            model_class, processor_class = self._get_model_classes()

            # Get model name from config
            model_name_hf = self.model_name_hf
            cache_dir = self.model_info.model_path if self.model_info.model_path else None

            self.logger.info(f"Loading DETR model: {model_name_hf}")
            self.logger.info("Note: First-time download may take several minutes...")

            # Load processor
            self.processor = processor_class.from_pretrained(
                model_name_hf,
                cache_dir=cache_dir
            )

            # Load model
            if self.device.type == 'cuda':
                # Load directly to GPU for efficiency
                self.model = model_class.from_pretrained(
                    model_name_hf,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                    device_map='auto' if self.use_multi_gpu else None
                )

                if not self.use_multi_gpu:
                    self.model = self.model.to(self.device)
            else:
                self.model = model_class.from_pretrained(
                    model_name_hf,
                    cache_dir=cache_dir
                )

            # Set model to evaluation mode
            self.model.eval()

            # Enable gradient checkpointing for memory efficiency
            if self.memory_efficient and hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

            # Setup mixed precision
            if self.use_amp and self.device.type == 'cuda':
                self.scaler = torch.cuda.amp.GradScaler()

            # Create CUDA stream
            if self.device.type == 'cuda':
                self.cuda_stream = torch.cuda.Stream()

            # Setup TensorRT optimization
            if self.use_tensorrt and TENSORRT_AVAILABLE and self.device.type == 'cuda':
                self._setup_tensorrt()

            # Update model info
            self.model_info.inference_device = str(self.device)
            self.model_info.fp16_enabled = self.use_fp16

            # Warmup
            self._warmup_model()

            self.is_loaded = True
            self.logger.info("DETR model loaded successfully with CUDA optimization")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load DETR model: {e}")
            self.last_error = str(e)
            return False

    def _get_model_classes(self):
        """Get appropriate model and processor classes based on variant."""
        variant_map = {
            'detr-resnet-50': (DetrForObjectDetection, DetrImageProcessor),
            'detr-resnet-101': (DetrForObjectDetection, DetrImageProcessor),
            'conditional-detr-resnet-50': (ConditionalDetrForObjectDetection, ConditionalDetrImageProcessor),
            'deformable-detr': (DeformableDetrForObjectDetection, DeformableDetrImageProcessor),
        }

        # Default to standard DETR if variant not found
        return variant_map.get(self.model_variant, (DetrForObjectDetection, DetrImageProcessor))

    def _setup_tensorrt(self):
        """Setup TensorRT optimization for DETR model."""
        try:
            if not TENSORRT_AVAILABLE:
                self.logger.warning("TensorRT not available for DETR optimization")
                return

            self.logger.info("Setting up TensorRT optimization for DETR...")

            # Export to ONNX with dynamic shapes for transformers
            dummy_input = torch.randn(1, 3, *self.input_size, device=self.device)
            onnx_path = f"{self.model_info.model_path}/detr_tensorrt.onnx"

            # Export with dynamic batch size
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                input_names=['pixel_values'],
                output_names=['logits', 'pred_boxes'],
                dynamic_axes={
                    'pixel_values': {0: 'batch_size'},
                    'logits': {0: 'batch_size'},
                    'pred_boxes': {0: 'batch_size'}
                },
                opset_version=11,
                do_constant_folding=True
            )

            self.logger.info("DETR TensorRT optimization setup completed")

        except Exception as e:
            self.logger.warning(f"TensorRT setup failed for DETR: {e}")
            self.use_tensorrt = False

    def _warmup_model(self):
        """Warmup DETR model for optimal performance."""
        self.logger.info("Warming up DETR model...")

        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_pil = Image.fromarray(dummy_image)

        with torch.no_grad():
            for i in range(self.warmup_iterations):
                try:
                    # Process with processor
                    inputs = self.processor(images=dummy_pil, return_tensors="pt")

                    # Move to device
                    if self.device.type == 'cuda':
                        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

                    # Forward pass
                    if self.use_amp and self.device.type == 'cuda':
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            _ = self.model(**inputs)
                    else:
                        _ = self.model(**inputs)

                except Exception as e:
                    self.logger.debug(f"Warmup iteration {i} failed: {e}")

        # Clear cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        self.logger.info("DETR model warmup completed")

    def detect(self, image: np.ndarray, **kwargs) -> DetectionResult:
        """
        Perform CUDA-accelerated DETR object detection.

        Args:
            image: Input image as numpy array
            **kwargs: Additional detection parameters

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

            # Preprocessing
            preprocess_start = time.time()
            processed_inputs = self.preprocess_image(image)
            preprocess_time = time.time() - preprocess_start

            # Inference
            inference_start = time.time()
            with torch.cuda.stream(self.cuda_stream) if self.device.type == 'cuda' else torch.no_grad():
                with torch.no_grad():
                    if self.use_amp and self.device.type == 'cuda':
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            outputs = self.model(**processed_inputs)
                    else:
                        outputs = self.model(**processed_inputs)

            inference_time = time.time() - inference_start

            # Postprocessing
            postprocess_start = time.time()
            detections = self.postprocess_outputs(outputs, image.shape[:2], confidence_threshold)
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
            self.logger.error(f"DETR detection failed: {e}")
            self.update_performance_stats(0, 0, False)

            return DetectionResult(
                detections=[],
                frame_id=kwargs.get('frame_id', 0),
                timestamp=start_time,
                success=False,
                error_message=str(e),
                model_name=self.model_info.model_name
            )

    def preprocess_image(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        CUDA-accelerated image preprocessing for DETR.

        Args:
            image: Input image (H, W, C) in BGR format

        Returns:
            Dictionary with processed tensors
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Convert to PIL Image for processor
        pil_image = Image.fromarray(image_rgb)

        # Use HuggingFace processor for optimal preprocessing
        inputs = self.processor(images=pil_image, return_tensors="pt")

        # Move to device efficiently
        if self.device.type == 'cuda':
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        return inputs

    def postprocess_outputs(self, outputs, original_shape: Tuple[int, int],
                          confidence_threshold: float) -> List[Detection]:
        """
        GPU-optimized postprocessing of DETR outputs.

        Args:
            outputs: DETR model outputs
            original_shape: Original image shape (height, width)
            confidence_threshold: Minimum confidence threshold

        Returns:
            List of Detection objects
        """
        detections = []

        try:
            # Get predictions
            logits = outputs.logits  # [batch_size, num_queries, num_classes]
            pred_boxes = outputs.pred_boxes  # [batch_size, num_queries, 4]

            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=-1)

            # Get the maximum probability for each query (excluding background class)
            max_probs, pred_classes = probabilities[..., :-1].max(-1)

            # Filter by confidence
            keep = max_probs > confidence_threshold

            if not keep.any():
                return detections

            # Get valid predictions
            valid_boxes = pred_boxes[0][keep]  # Remove batch dimension
            valid_probs = max_probs[0][keep]
            valid_classes = pred_classes[0][keep]

            # Convert to CPU for processing
            valid_boxes = valid_boxes.cpu().numpy()
            valid_probs = valid_probs.cpu().numpy()
            valid_classes = valid_classes.cpu().numpy()

            original_height, original_width = original_shape

            # Process each detection
            for i, (box, prob, class_id) in enumerate(zip(valid_boxes, valid_probs, valid_classes)):
                try:
                    # DETR outputs normalized coordinates (center_x, center_y, width, height)
                    center_x, center_y, width, height = box

                    # Convert to absolute coordinates
                    center_x *= original_width
                    center_y *= original_height
                    width *= original_width
                    height *= original_height

                    # Convert to (x1, y1, x2, y2) format
                    x1 = center_x - width / 2
                    y1 = center_y - height / 2
                    x2 = center_x + width / 2
                    y2 = center_y + height / 2

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
                        confidence=float(prob),
                        class_id=int(class_id),
                        class_name=class_name,
                        detection_id=i,
                        metadata={
                            'model_type': 'detr',
                            'model_name': self.model_info.model_name,
                            'variant': self.model_variant,
                            'device': str(self.device),
                            'fp16': self.use_fp16,
                            'query_id': i  # Which transformer query produced this detection
                        }
                    )

                    detections.append(detection)

                except Exception as e:
                    self.logger.warning(f"Error processing DETR detection {i}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"DETR postprocessing failed: {e}")

        return detections

    def detect_batch(self, images: List[np.ndarray], **kwargs) -> List[DetectionResult]:
        """
        CUDA-accelerated batch detection for multiple images with DETR.

        Args:
            images: List of input images
            **kwargs: Detection parameters

        Returns:
            List of DetectionResult objects
        """
        if not images:
            return []

        start_time = time.time()

        try:
            # Convert images to PIL format
            pil_images = []
            valid_indices = []

            for i, image in enumerate(images):
                if self.validate_image(image):
                    # Convert BGR to RGB if needed
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        image_rgb = image

                    pil_images.append(Image.fromarray(image_rgb))
                    valid_indices.append(i)

            if not pil_images:
                return [DetectionResult(detections=[], frame_id=0, timestamp=start_time,
                                      success=False, error_message="No valid images in batch")]

            # Batch preprocessing
            inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)

            # Move to device
            if self.device.type == 'cuda':
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

            # Batch inference
            with torch.cuda.stream(self.cuda_stream) if self.device.type == 'cuda' else torch.no_grad():
                with torch.no_grad():
                    if self.use_amp and self.device.type == 'cuda':
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            outputs = self.model(**inputs)
                    else:
                        outputs = self.model(**inputs)

            # Process results for each image
            results = []
            confidence_threshold = kwargs.get('confidence_threshold', self.model_info.confidence_threshold)

            for i, (valid_idx, original_image) in enumerate(zip(valid_indices, images)):
                # Extract outputs for this image
                image_logits = outputs.logits[i:i+1]  # Keep batch dimension
                image_boxes = outputs.pred_boxes[i:i+1]

                # Create outputs object for single image
                from types import SimpleNamespace
                single_outputs = SimpleNamespace()
                single_outputs.logits = image_logits
                single_outputs.pred_boxes = image_boxes

                # Postprocess
                detections = self.postprocess_outputs(single_outputs, original_image.shape[:2], confidence_threshold)

                result = DetectionResult(
                    detections=detections,
                    frame_id=kwargs.get('frame_id', valid_idx),
                    timestamp=start_time,
                    model_name=self.model_info.model_name,
                    frame_width=original_image.shape[1],
                    frame_height=original_image.shape[0],
                    success=True
                )

                results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"DETR batch processing failed: {e}")
            return [DetectionResult(detections=[], frame_id=0, timestamp=start_time,
                                  success=False, error_message=str(e))]

    def optimize_attention_mechanism(self):
        """Apply attention-specific optimizations for transformer models."""
        if self.device.type == 'cuda' and self.model:
            try:
                # Enable flash attention if available (PyTorch 2.0+)
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    # Enable optimized attention
                    torch.backends.cuda.enable_flash_sdp(True)
                    self.logger.info("Flash attention enabled for DETR")

                # Optimize transformer layers
                for module in self.model.modules():
                    if hasattr(module, 'enable_flash_attention'):
                        module.enable_flash_attention()

            except Exception as e:
                self.logger.warning(f"Attention optimization failed: {e}")

    def get_attention_maps(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """
        Extract attention maps from DETR for visualization.

        Args:
            image: Input image

        Returns:
            Attention maps tensor or None if extraction failed
        """
        if not self.is_loaded:
            return None

        try:
            # Preprocess image
            inputs = self.preprocess_image(image)

            # Forward pass with attention output
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)

            # Extract attention weights
            if hasattr(outputs, 'attentions') and outputs.attentions:
                # Return last layer attention maps
                return outputs.attentions[-1]

        except Exception as e:
            self.logger.error(f"Failed to extract attention maps: {e}")

        return None

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage for DETR model."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3   # GB
            total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3

            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'utilization_percent': (allocated / total) * 100,
                'transformer_optimized': True
            }
        return {}

    def cleanup(self):
        """Clean up CUDA resources and DETR model."""
        try:
            # Clear CUDA cache
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

                if self.cuda_stream:
                    self.cuda_stream.synchronize()

            # Clean up model and processor
            if self.model:
                del self.model

            if self.processor:
                del self.processor

            if self.scaler:
                del self.scaler

            # Reset GPU memory
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            super().cleanup()
            self.logger.info("DETR detector cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during DETR cleanup: {e}")


# Register DETR detector with factory
def register_detr_detector():
    """Register DETR detector with the factory."""
    try:
        from .detector_factory import DetectorFactory
        DetectorFactory.register_detector('detr', DETRDetector)
    except ImportError:
        pass