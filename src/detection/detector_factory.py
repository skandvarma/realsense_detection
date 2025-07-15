"""
Factory pattern implementation for detection model creation and management.
"""

import time
import importlib
import threading
from typing import Dict, Any, List, Optional, Type, Tuple
from pathlib import Path
import numpy as np
import psutil

from .base_detector import BaseDetector, ModelInfo, DetectionResult
from ..utils.logger import get_logger
from ..utils.config import ConfigManager


class DetectorFactory:
    """Factory class for creating and managing detection models."""

    _registered_detectors: Dict[str, Type[BaseDetector]] = {}
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize detector factory."""
        if not hasattr(self, '_initialized'):
            self.logger = get_logger("DetectorFactory")
            self._model_cache = {}
            self._benchmark_results = {}
            self._initialized = True
            self.logger.info("DetectorFactory initialized")

    @classmethod
    def register_detector(cls, model_type: str, detector_class: Type[BaseDetector]):
        """
        Register a detector class for a specific model type.

        Args:
            model_type: Model type identifier (e.g., 'yolo', 'detr')
            detector_class: Detector class to register
        """
        cls._registered_detectors[model_type] = detector_class
        logger = get_logger("DetectorFactory")
        logger.info(f"Registered detector: {model_type} -> {detector_class.__name__}")

    @classmethod
    def get_registered_detectors(cls) -> Dict[str, Type[BaseDetector]]:
        """Get all registered detector types."""
        return cls._registered_detectors.copy()

    def create_detector(self, config: Dict[str, Any], model_type: Optional[str] = None) -> Optional[BaseDetector]:
        """
        Create a detector instance based on configuration.

        Args:
            config: Complete system configuration
            model_type: Override model type (uses config if None)

        Returns:
            Detector instance or None if creation failed
        """
        try:
            # Determine model type
            if model_type is None:
                model_type = config.get('detection', {}).get('active_model', 'yolo')

            if model_type not in self._registered_detectors:
                self.logger.error(f"Unknown model type: {model_type}")
                return None

            # Extract model configuration
            model_config = config.get('detection', {}).get(model_type, {})
            if not model_config:
                self.logger.error(f"No configuration found for model type: {model_type}")
                return None

            # Create model info
            model_info = self._create_model_info(model_type, model_config)
            if model_info is None:
                return None

            # Validate model files
            if not self._validate_model_files(model_info):
                return None

            # Create detector instance
            detector_class = self._registered_detectors[model_type]
            detector = detector_class(model_info, config)

            # Load model
            if not detector.load_model():
                self.logger.error(f"Failed to load model: {model_info.model_name}")
                return None

            self.logger.info(f"Successfully created detector: {model_type} - {model_info.model_name}")
            return detector

        except Exception as e:
            self.logger.error(f"Error creating detector: {e}")
            return None

    def _create_model_info(self, model_type: str, model_config: Dict[str, Any]) -> Optional[ModelInfo]:
        """Create ModelInfo from configuration."""
        try:
            # Extract basic information
            model_name = model_config.get('variant', f'{model_type}_default')

            if model_type == 'detr':
                # DETR uses HuggingFace models, no local files needed
                model_path = model_config.get('model_path', 'models/detr/')
                weights_path = ""  # Not used for HuggingFace models
                # model_name_hf = model_config.get('model_name', 'facebook/detr-resnet-50')
            else:
                # YOLO uses local files
                model_path = model_config.get('model_path', 'models/')
                weights_file = model_config.get('weights', f'{model_name}.pt')

                # Construct full paths
                if not Path(weights_file).is_absolute():
                    weights_path = str(Path(model_path) / weights_file)
                else:
                    weights_path = weights_file

            # Get input size
            input_size = model_config.get('input_size', [640, 640])
            if len(input_size) != 2:
                self.logger.error(f"Invalid input size: {input_size}")
                return None

            # Create class names list
            class_names = self._get_class_names(model_type, model_config)

            model_info = ModelInfo(
                model_type=model_type,
                model_name=model_name,
                model_path=model_path,
                weights_path=weights_path if model_type != 'detr' else "",
                input_width=input_size[0],
                input_height=input_size[1],
                num_classes=len(class_names),
                class_names=class_names,
                confidence_threshold=model_config.get('confidence_threshold', 0.5),
                iou_threshold=model_config.get('iou_threshold', 0.45),
                max_detections=model_config.get('max_detections', 100),
                config=model_config
            )

            return model_info

        except Exception as e:
            self.logger.error(f"Error creating model info: {e}")
            return None

    def _get_class_names(self, model_type: str, model_config: Dict[str, Any]) -> List[str]:
        """Get class names for the model."""
        # Check if class names are provided in config
        if 'class_names' in model_config:
            return model_config['class_names']

        # Default COCO class names for most models
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

        return coco_classes

    def _validate_model_files(self, model_info: ModelInfo) -> bool:
        """Validate that model files exist."""
        # For HuggingFace models (DETR), skip local file validation
        if model_info.model_type == 'detr':
            return True  # HuggingFace will download automatically

        if not model_info.model_files_exist:
            self.logger.error(f"Model files not found for {model_info.model_name}")
            self.logger.error(f"Weights path: {model_info.weights_path}")
            return False

        return True

    def create_detector_with_fallback(self, config: Dict[str, Any],
                                    primary_type: Optional[str] = None,
                                    fallback_type: Optional[str] = None) -> Optional['DetectorWrapper']:
        """
        Create detector with fallback support.

        Args:
            config: System configuration
            primary_type: Primary detector type
            fallback_type: Fallback detector type

        Returns:
            DetectorWrapper instance or None
        """
        try:
            # Determine detector types
            if primary_type is None:
                primary_type = config.get('detection', {}).get('active_model', 'yolo')

            if fallback_type is None:
                # Choose different fallback
                all_types = list(self._registered_detectors.keys())
                fallback_type = next((t for t in all_types if t != primary_type), None)

            # Create primary detector
            primary_detector = self.create_detector(config, primary_type)

            # Create fallback detector (optional)
            fallback_detector = None
            if fallback_type and fallback_type in self._registered_detectors:
                try:
                    fallback_detector = self.create_detector(config, fallback_type)
                except Exception as e:
                    self.logger.warning(f"Failed to create fallback detector: {e}")

            if primary_detector is None and fallback_detector is None:
                self.logger.error("Failed to create both primary and fallback detectors")
                return None

            return DetectorWrapper(primary_detector, fallback_detector)

        except Exception as e:
            self.logger.error(f"Error creating detector with fallback: {e}")
            return None

    def benchmark_detector(self, detector: BaseDetector, test_images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Benchmark detector performance.

        Args:
            detector: Detector to benchmark
            test_images: List of test images

        Returns:
            Benchmark results dictionary
        """
        self.logger.info(f"Benchmarking detector: {detector.model_info.model_name}")

        # Performance metrics
        inference_times = []
        memory_usage = []
        detection_counts = []

        start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB

        try:
            for i, image in enumerate(test_images):
                if not detector.validate_image(image):
                    continue

                # Measure inference time
                start_time = time.time()
                result = detector.detect(image)
                inference_time = time.time() - start_time

                if result.success:
                    inference_times.append(inference_time)
                    detection_counts.append(result.detection_count)

                    # Measure memory usage
                    current_memory = psutil.virtual_memory().used / 1024 / 1024
                    memory_usage.append(current_memory)

                # Progress logging
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(test_images)} test images")

            # Calculate statistics
            if inference_times:
                results = {
                    'model_name': detector.model_info.model_name,
                    'model_type': detector.model_info.model_type,
                    'test_image_count': len(test_images),
                    'successful_inferences': len(inference_times),
                    'avg_inference_time': np.mean(inference_times),
                    'min_inference_time': np.min(inference_times),
                    'max_inference_time': np.max(inference_times),
                    'std_inference_time': np.std(inference_times),
                    'avg_fps': 1.0 / np.mean(inference_times),
                    'avg_detections_per_frame': np.mean(detection_counts),
                    'avg_memory_usage_mb': np.mean(memory_usage) - start_memory,
                    'max_memory_usage_mb': np.max(memory_usage) - start_memory,
                    'input_size': detector.model_info.input_size,
                    'device': detector.model_info.inference_device
                }

                self.logger.info(f"Benchmark completed for {detector.model_info.model_name}")
                self.logger.info(f"Average FPS: {results['avg_fps']:.2f}")
                self.logger.info(f"Average inference time: {results['avg_inference_time']*1000:.2f}ms")

                return results
            else:
                self.logger.error("No successful inferences during benchmark")
                return {}

        except Exception as e:
            self.logger.error(f"Error during benchmarking: {e}")
            return {}

    def benchmark_all_detectors(self, config: Dict[str, Any],
                              test_images: List[np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark all available detector types.

        Args:
            config: System configuration
            test_images: Test images for benchmarking

        Returns:
            Dictionary of benchmark results for each detector type
        """
        results = {}

        for model_type in self._registered_detectors:
            try:
                self.logger.info(f"Benchmarking {model_type} detector")

                detector = self.create_detector(config, model_type)
                if detector:
                    benchmark_result = self.benchmark_detector(detector, test_images)
                    if benchmark_result:
                        results[model_type] = benchmark_result

                    detector.cleanup()
                else:
                    self.logger.warning(f"Failed to create {model_type} detector for benchmarking")

            except Exception as e:
                self.logger.error(f"Error benchmarking {model_type}: {e}")

        # Store results
        self._benchmark_results = results

        # Log summary
        if results:
            self.logger.info("=== Benchmark Summary ===")
            for model_type, result in results.items():
                self.logger.info(f"{model_type}: {result['avg_fps']:.2f} FPS, "
                               f"{result['avg_inference_time']*1000:.2f}ms avg")

        return results

    def get_recommended_detector(self, requirements: Dict[str, Any] = None) -> Optional[str]:
        """
        Get recommended detector type based on requirements and benchmark results.

        Args:
            requirements: Performance requirements (fps, accuracy, etc.)

        Returns:
            Recommended detector type or None
        """
        if not self._benchmark_results:
            self.logger.warning("No benchmark results available for recommendation")
            return None

        requirements = requirements or {}
        min_fps = requirements.get('min_fps', 0)
        max_memory = requirements.get('max_memory_mb', float('inf'))
        prefer_accuracy = requirements.get('prefer_accuracy', False)

        candidates = []

        for model_type, results in self._benchmark_results.items():
            if (results['avg_fps'] >= min_fps and
                results['avg_memory_usage_mb'] <= max_memory):
                candidates.append((model_type, results))

        if not candidates:
            self.logger.warning("No detectors meet the specified requirements")
            return None

        # Choose best candidate
        if prefer_accuracy:
            # For now, assume higher detection count indicates better accuracy
            best = max(candidates, key=lambda x: x[1]['avg_detections_per_frame'])
        else:
            # Choose fastest
            best = max(candidates, key=lambda x: x[1]['avg_fps'])

        self.logger.info(f"Recommended detector: {best[0]} (FPS: {best[1]['avg_fps']:.2f})")
        return best[0]


class DetectorWrapper:
    """Wrapper for managing primary and fallback detectors."""

    def __init__(self, primary_detector: Optional[BaseDetector],
                 fallback_detector: Optional[BaseDetector] = None):
        """
        Initialize detector wrapper.

        Args:
            primary_detector: Primary detector instance
            fallback_detector: Optional fallback detector instance
        """
        self.primary_detector = primary_detector
        self.fallback_detector = fallback_detector
        self.logger = get_logger("DetectorWrapper")

        # Failure tracking
        self.primary_failures = 0
        self.fallback_failures = 0
        self.failure_threshold = 5
        self.current_detector = 'primary'

        # Performance tracking
        self.primary_performance = []
        self.fallback_performance = []

        if self.primary_detector:
            self.logger.info(f"Primary detector: {self.primary_detector.model_info.model_name}")
        if self.fallback_detector:
            self.logger.info(f"Fallback detector: {self.fallback_detector.model_info.model_name}")

    def detect(self, image: np.ndarray, **kwargs) -> DetectionResult:
        """
        Perform detection with automatic fallback.

        Args:
            image: Input image
            **kwargs: Additional detection parameters

        Returns:
            DetectionResult from active detector
        """
        # Try primary detector first
        if self.current_detector == 'primary' and self.primary_detector:
            try:
                start_time = time.time()
                result = self.primary_detector.detect(image, **kwargs)
                inference_time = time.time() - start_time

                if result.success:
                    self.primary_failures = 0  # Reset failure count
                    self.primary_performance.append(inference_time)
                    return result
                else:
                    self.primary_failures += 1
                    self.logger.warning(f"Primary detector failed: {result.error_message}")

            except Exception as e:
                self.primary_failures += 1
                self.logger.error(f"Primary detector error: {e}")

        # Switch to fallback if primary fails too much
        if (self.primary_failures >= self.failure_threshold and
            self.fallback_detector and self.current_detector == 'primary'):
            self.logger.warning("Switching to fallback detector due to repeated failures")
            self.current_detector = 'fallback'

        # Try fallback detector
        if self.current_detector == 'fallback' and self.fallback_detector:
            try:
                start_time = time.time()
                result = self.fallback_detector.detect(image, **kwargs)
                inference_time = time.time() - start_time

                if result.success:
                    self.fallback_failures = 0
                    self.fallback_performance.append(inference_time)
                    return result
                else:
                    self.fallback_failures += 1
                    self.logger.warning(f"Fallback detector failed: {result.error_message}")

            except Exception as e:
                self.fallback_failures += 1
                self.logger.error(f"Fallback detector error: {e}")

        # Both detectors failed
        self.logger.error("Both primary and fallback detectors failed")
        return DetectionResult(
            detections=[],
            frame_id=0,
            timestamp=time.time(),
            success=False,
            error_message="All detectors failed"
        )

    def get_active_detector(self) -> Optional[BaseDetector]:
        """Get currently active detector."""
        if self.current_detector == 'primary':
            return self.primary_detector
        else:
            return self.fallback_detector

    def force_switch_detector(self):
        """Manually switch between primary and fallback."""
        if self.current_detector == 'primary' and self.fallback_detector:
            self.current_detector = 'fallback'
            self.logger.info("Manually switched to fallback detector")
        elif self.current_detector == 'fallback' and self.primary_detector:
            self.current_detector = 'primary'
            self.primary_failures = 0  # Reset failure count
            self.logger.info("Manually switched to primary detector")

    def get_performance_comparison(self) -> Dict[str, Any]:
        """Get performance comparison between detectors."""
        comparison = {
            'current_detector': self.current_detector,
            'primary_failures': self.primary_failures,
            'fallback_failures': self.fallback_failures
        }

        if self.primary_performance:
            comparison['primary_avg_time'] = np.mean(self.primary_performance)
            comparison['primary_fps'] = 1.0 / np.mean(self.primary_performance)

        if self.fallback_performance:
            comparison['fallback_avg_time'] = np.mean(self.fallback_performance)
            comparison['fallback_fps'] = 1.0 / np.mean(self.fallback_performance)

        return comparison

    def reset_failure_counts(self):
        """Reset failure counters."""
        self.primary_failures = 0
        self.fallback_failures = 0
        self.logger.info("Failure counts reset")

    def cleanup(self):
        """Clean up both detectors."""
        if self.primary_detector:
            self.primary_detector.cleanup()
        if self.fallback_detector:
            self.fallback_detector.cleanup()
        self.logger.info("DetectorWrapper cleaned up")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()