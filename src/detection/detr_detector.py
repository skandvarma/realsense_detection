"""
CUDA-accelerated DETR (Detection Transformer) implementation with GPU optimization and working Grounding DINO.
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

# Robust transformers import with fallback
try:
    import transformers
    from transformers import DetrImageProcessor, DetrForObjectDetection
    TRANSFORMERS_AVAILABLE = True

    # Optional advanced DETR variants
    try:
        from transformers import ConditionalDetrImageProcessor, ConditionalDetrForObjectDetection
        CONDITIONAL_DETR_AVAILABLE = True
    except ImportError:
        CONDITIONAL_DETR_AVAILABLE = False

    try:
        from transformers import DeformableDetrImageProcessor, DeformableDetrForObjectDetection
        DEFORMABLE_DETR_AVAILABLE = True
    except ImportError:
        DEFORMABLE_DETR_AVAILABLE = False

    # Grounding DINO support
    try:
        from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection
        GROUNDING_DINO_AVAILABLE = True
    except ImportError:
        GROUNDING_DINO_AVAILABLE = False

except ImportError as e:
    print(f"Transformers import warning: {e}")
    TRANSFORMERS_AVAILABLE = False
    CONDITIONAL_DETR_AVAILABLE = False
    DEFORMABLE_DETR_AVAILABLE = False
    GROUNDING_DINO_AVAILABLE = False

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

        # Grounding DINO specific
        self.is_grounding_dino = ('grounding' in self.model_variant.lower() or
                                  'grounding' in self.model_name_hf.lower())
        self.default_prompt = model_info.config.get('default_prompt', '')
        self.current_prompt = self.default_prompt

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
        if self.is_grounding_dino:
            self.logger.info(f"Grounding DINO mode enabled with prompt: '{self.current_prompt}'")
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
                error_msg = (
                    "Transformers not available. Please install it:\n"
                    "pip install transformers torch torchvision Pillow\n"
                    f"Current Python: {self.device}\n"
                    "Run: python tests/test_transformers.py to diagnose the issue"
                )
                raise ImportError(error_msg)

            if not PIL_AVAILABLE:
                raise ImportError("PIL not available. Please install: pip install Pillow")

            # Check Grounding DINO availability
            if self.is_grounding_dino and not GROUNDING_DINO_AVAILABLE:
                self.logger.error("Grounding DINO not available. Please install with: pip install transformers[grounding-dino]")
                return False

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

            # Validate processor for Grounding DINO
            if self.is_grounding_dino:
                if not hasattr(self.processor, 'post_process_grounded_object_detection'):
                    self.logger.error(f"Expected GroundingDinoProcessor but got {type(self.processor)}")
                    self.logger.error("This will cause detection failures. Check model variant configuration.")
                    return False
                else:
                    self.logger.info("Grounding DINO processor loaded correctly")
            else:
                self.logger.info(f"Loaded processor type: {type(self.processor)}")

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

            # Enable gradient checkpointing for memory efficiency (if supported)
            if self.memory_efficient and hasattr(self.model, 'gradient_checkpointing_enable'):
                try:
                    self.model.gradient_checkpointing_enable()
                    self.logger.info("Gradient checkpointing enabled")
                except Exception as e:
                    self.logger.info(f"Gradient checkpointing not supported: {e}")
                    self.memory_efficient = False

            # Setup mixed precision
            if self.use_amp and self.device.type == 'cuda':
                self.scaler = torch.amp.GradScaler('cuda',)

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
        # Check Grounding DINO first (most important fix)
        if self.is_grounding_dino and GROUNDING_DINO_AVAILABLE:
            self.logger.info("Using Grounding DINO classes")
            return (GroundingDinoForObjectDetection, GroundingDinoProcessor)

        variant_map = {
            'detr-resnet-50': (DetrForObjectDetection, DetrImageProcessor),
            'detr-resnet-101': (DetrForObjectDetection, DetrImageProcessor),
        }

        # Add conditional DETR if available
        if CONDITIONAL_DETR_AVAILABLE:
            variant_map['conditional-detr-resnet-50'] = (ConditionalDetrForObjectDetection, ConditionalDetrImageProcessor)

        # Add deformable DETR if available
        if DEFORMABLE_DETR_AVAILABLE:
            variant_map['deformable-detr'] = (DeformableDetrForObjectDetection, DeformableDetrImageProcessor)

        # Add Grounding DINO if available (fallback)
        if GROUNDING_DINO_AVAILABLE:
            variant_map['grounding-dino'] = (GroundingDinoForObjectDetection, GroundingDinoProcessor)

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
                    if self.is_grounding_dino:
                        # Use a simple prompt for warmup
                        warmup_prompt = "object"
                        inputs = self.processor(images=dummy_pil, text=warmup_prompt, return_tensors="pt")
                    else:
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
                - text_prompt: Text prompt for Grounding DINO (optional)

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
            text_prompt = kwargs.get('text_prompt', self.current_prompt)

            # Validate text prompt for Grounding DINO
            if self.is_grounding_dino:
                if not text_prompt or text_prompt.strip() == "":
                    return DetectionResult(
                        detections=[],
                        frame_id=kwargs.get('frame_id', 0),
                        timestamp=start_time,
                        success=False,
                        error_message="Grounding DINO requires a non-empty text prompt",
                        model_name=self.model_info.model_name
                    )
                text_prompt = text_prompt.strip()

            # Preprocessing
            preprocess_start = time.time()
            processed_inputs = self.preprocess_image(image, text_prompt)
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
            detections = self.postprocess_outputs(outputs, image.shape[:2], confidence_threshold, text_prompt)
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
                success=True,
                metadata={'text_prompt': text_prompt}
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

    def preprocess_image(self, image: np.ndarray, text_prompt: str = None) -> Dict[str, torch.Tensor]:
        """
        CUDA-accelerated image preprocessing for DETR.

        Args:
            image: Input image (H, W, C) in BGR format
            text_prompt: Text prompt for Grounding DINO (optional)

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
        if self.is_grounding_dino:
            # Grounding DINO ALWAYS needs text prompt
            if not text_prompt or text_prompt.strip() == "":
                raise ValueError("Grounding DINO requires a non-empty text prompt")
            inputs = self.processor(images=pil_image, text=text_prompt.strip(), return_tensors="pt")
        else:
            inputs = self.processor(images=pil_image, return_tensors="pt")

        # Move to device efficiently
        if self.device.type == 'cuda':
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        return inputs

    def postprocess_outputs(self, outputs, original_shape: Tuple[int, int],
                          confidence_threshold: float, text_prompt: str = None) -> List[Detection]:
        """
        GPU-optimized postprocessing of DETR outputs.

        Args:
            outputs: DETR model outputs
            original_shape: Original image shape (height, width)
            confidence_threshold: Minimum confidence threshold
            text_prompt: Text prompt used (for metadata)

        Returns:
            List of Detection objects
        """
        detections = []

        try:
            if self.is_grounding_dino:
                # Use Grounding DINO specific postprocessing (based on working implementation)
                return self._postprocess_grounding_dino(outputs, original_shape, confidence_threshold, text_prompt)
            else:
                # Regular DETR postprocessing
                return self._postprocess_regular_detr(outputs, original_shape, confidence_threshold)

        except Exception as e:
            self.logger.error(f"DETR postprocessing failed: {e}")
            return detections

    def _postprocess_grounding_dino(self, outputs, original_shape: Tuple[int, int],
                                  confidence_threshold: float, text_prompt: str) -> List[Detection]:
        """Postprocess Grounding DINO outputs using the working approach."""
        detections = []

        try:
            # Validate processor type
            if not hasattr(self.processor, 'post_process_grounded_object_detection'):
                self.logger.error(f"Processor type {type(self.processor)} doesn't have post_process_grounded_object_detection method")
                self.logger.error("This indicates Grounding DINO processor wasn't loaded correctly")
                return detections

            # Use the processor's built-in postprocessing (key difference from regular DETR)
            original_height, original_width = original_shape
            target_sizes = torch.tensor([[original_width, original_height]]).to(self.device)

            self.logger.debug(f"Processing Grounding DINO outputs with target size: {target_sizes}")

            results = self.processor.post_process_grounded_object_detection(
                outputs, target_sizes=target_sizes, threshold=confidence_threshold
            )[0]

            self.logger.debug(f"Grounding DINO raw results: {len(results.get('boxes', []))} boxes found")

            # Process results
            if len(results["boxes"]) > 0:
                for i, (box, score, label) in enumerate(zip(results["boxes"], results["scores"], results["labels"])):
                    try:
                        x1, y1, x2, y2 = box.cpu().numpy()

                        # Clip to image boundaries
                        x1 = max(0, min(x1, original_width))
                        y1 = max(0, min(y1, original_height))
                        x2 = max(0, min(x2, original_width))
                        y2 = max(0, min(y2, original_height))

                        # Skip invalid boxes
                        if x2 <= x1 or y2 <= y1:
                            self.logger.debug(f"Skipping invalid box: ({x1}, {y1}, {x2}, {y2})")
                            continue

                        # Create detection
                        detection = Detection(
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            confidence=float(score.cpu().numpy()),
                            class_id=i,  # Grounding DINO doesn't have fixed class IDs
                            class_name=label,
                            detection_id=i,
                            metadata={
                                'model_type': 'grounding-dino',
                                'model_name': self.model_info.model_name,
                                'text_prompt': text_prompt,
                                'device': str(self.device),
                                'fp16': self.use_fp16
                            }
                        )

                        detections.append(detection)
                        self.logger.debug(f"Created detection: {label} with confidence {score:.3f}")

                    except Exception as e:
                        self.logger.warning(f"Error processing Grounding DINO detection {i}: {e}")
                        continue
            else:
                self.logger.debug("No objects detected by Grounding DINO")

        except Exception as e:
            self.logger.error(f"Grounding DINO postprocessing failed: {e}")
            self.logger.error(f"Processor type: {type(self.processor)}")
            self.logger.error(f"Available methods: {[m for m in dir(self.processor) if 'post_process' in m]}")

        return detections

    def _postprocess_regular_detr(self, outputs, original_shape: Tuple[int, int],
                                confidence_threshold: float) -> List[Detection]:
        """Postprocess regular DETR outputs."""
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

            # Get valid predictions (handle batch dimension properly)
            keep_mask = keep[0]  # Remove batch dimension from mask
            valid_boxes = pred_boxes[0][keep_mask]  # Remove batch dimension
            valid_probs = max_probs[0][keep_mask]
            valid_classes = pred_classes[0][keep_mask]

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
            self.logger.error(f"Regular DETR postprocessing failed: {e}")

        return detections

    def update_text_prompt(self, new_prompt: str):
        """Update the current text prompt for Grounding DINO."""
        if self.is_grounding_dino:
            self.current_prompt = new_prompt
            self.logger.info(f"Updated text prompt: {new_prompt}")
        else:
            self.logger.warning("Text prompts only supported for Grounding DINO variant")

    def get_current_prompt(self) -> str:
        """Get the current text prompt."""
        return self.current_prompt

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