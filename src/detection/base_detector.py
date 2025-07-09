"""
Abstract base classes and interfaces for all detection models.
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.logger import get_logger


@dataclass
class Detection:
    """2D bounding box detection representation."""

    # Bounding box coordinates (x1, y1, x2, y2)
    bbox: Tuple[float, float, float, float]

    # Detection confidence (0.0 to 1.0)
    confidence: float

    # Class information
    class_id: int
    class_name: str

    # Detection ID for tracking
    detection_id: Optional[int] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate detection data after initialization."""
        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        # Validate bounding box
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bounding box: {self.bbox}")

        # Validate class_id
        if self.class_id < 0:
            raise ValueError(f"Class ID must be non-negative, got {self.class_id}")

    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def center(self) -> Tuple[float, float]:
        """Get bounding box center coordinates."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        """Get bounding box width."""
        x1, y1, x2, y2 = self.bbox
        return x2 - x1

    @property
    def height(self) -> float:
        """Get bounding box height."""
        x1, y1, x2, y2 = self.bbox
        return y2 - y1

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Convert bbox to (x, y, width, height) format."""
        x1, y1, x2, y2 = self.bbox
        return (x1, y1, x2 - x1, y2 - y1)

    def to_cxcywh(self) -> Tuple[float, float, float, float]:
        """Convert bbox to (center_x, center_y, width, height) format."""
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return (cx, cy, w, h)

    def iou(self, other: 'Detection') -> float:
        """Calculate Intersection over Union with another detection."""
        x1_a, y1_a, x2_a, y2_a = self.bbox
        x1_b, y1_b, x2_b, y2_b = other.bbox

        # Calculate intersection
        x1_inter = max(x1_a, x1_b)
        y1_inter = max(y1_a, y1_b)
        x2_inter = min(x2_a, x2_b)
        y2_inter = min(y2_a, y2_b)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0

    def scale(self, scale_x: float, scale_y: float) -> 'Detection':
        """Scale detection coordinates."""
        x1, y1, x2, y2 = self.bbox
        scaled_bbox = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)

        return Detection(
            bbox=scaled_bbox,
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
            detection_id=self.detection_id,
            metadata=self.metadata.copy()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary."""
        return {
            'bbox': list(self.bbox),
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'detection_id': self.detection_id,
            'area': self.area,
            'center': list(self.center),
            'metadata': self.metadata
        }


@dataclass
class Detection3D(Detection):
    """3D detection with spatial information."""

    # 3D center point coordinates (meters)
    center_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Distance from camera (meters)
    distance: float = 0.0

    # Depth confidence (0.0 to 1.0)
    depth_confidence: float = 0.0

    # 3D bounding box dimensions (width, height, depth in meters)
    dimensions_3d: Optional[Tuple[float, float, float]] = None

    # 3D bounding box corners (8 points for future expansion)
    bbox_3d_corners: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate 3D detection data."""
        super().__post_init__()

        # Validate depth confidence
        if not 0.0 <= self.depth_confidence <= 1.0:
            raise ValueError(f"Depth confidence must be between 0.0 and 1.0, got {self.depth_confidence}")

        # Validate distance
        if self.distance < 0.0:
            raise ValueError(f"Distance must be non-negative, got {self.distance}")

    @property
    def x_3d(self) -> float:
        """Get 3D X coordinate."""
        return self.center_3d[0]

    @property
    def y_3d(self) -> float:
        """Get 3D Y coordinate."""
        return self.center_3d[1]

    @property
    def z_3d(self) -> float:
        """Get 3D Z coordinate (depth)."""
        return self.center_3d[2]

    @property
    def volume(self) -> float:
        """Calculate 3D volume if dimensions are available."""
        if self.dimensions_3d:
            w, h, d = self.dimensions_3d
            return w * h * d
        return 0.0

    def distance_to(self, other: 'Detection3D') -> float:
        """Calculate 3D Euclidean distance to another 3D detection."""
        dx = self.x_3d - other.x_3d
        dy = self.y_3d - other.y_3d
        dz = self.z_3d - other.z_3d
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert 3D detection to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'center_3d': list(self.center_3d),
            'distance': self.distance,
            'depth_confidence': self.depth_confidence,
            'dimensions_3d': list(self.dimensions_3d) if self.dimensions_3d else None,
            'volume': self.volume,
            'bbox_3d_corners': self.bbox_3d_corners.tolist() if self.bbox_3d_corners is not None else None
        })
        return base_dict


@dataclass
class DetectionResult:
    """Container for complete detection pipeline results."""

    # List of detections
    detections: List[Union[Detection, Detection3D]]

    # Frame information
    frame_id: int
    timestamp: float

    # Timing information (seconds)
    preprocessing_time: float = 0.0
    inference_time: float = 0.0
    postprocessing_time: float = 0.0

    # Model information
    model_name: str = ""
    model_version: str = ""

    # Frame metadata
    frame_width: int = 0
    frame_height: int = 0

    # Performance statistics
    fps: float = 0.0
    memory_usage_mb: float = 0.0

    # Processing status
    success: bool = True
    error_message: str = ""

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_time(self) -> float:
        """Get total processing time."""
        return self.preprocessing_time + self.inference_time + self.postprocessing_time

    @property
    def detection_count(self) -> int:
        """Get number of detections."""
        return len(self.detections)

    @property
    def detection_3d_count(self) -> int:
        """Get number of 3D detections."""
        return sum(1 for det in self.detections if isinstance(det, Detection3D))

    def filter_by_confidence(self, min_confidence: float) -> 'DetectionResult':
        """Filter detections by minimum confidence."""
        filtered_detections = [det for det in self.detections if det.confidence >= min_confidence]

        return DetectionResult(
            detections=filtered_detections,
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            preprocessing_time=self.preprocessing_time,
            inference_time=self.inference_time,
            postprocessing_time=self.postprocessing_time,
            model_name=self.model_name,
            model_version=self.model_version,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            fps=self.fps,
            memory_usage_mb=self.memory_usage_mb,
            success=self.success,
            error_message=self.error_message,
            metadata=self.metadata.copy()
        )

    def filter_by_class(self, class_names: List[str]) -> 'DetectionResult':
        """Filter detections by class names."""
        filtered_detections = [det for det in self.detections if det.class_name in class_names]

        return DetectionResult(
            detections=filtered_detections,
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            preprocessing_time=self.preprocessing_time,
            inference_time=self.inference_time,
            postprocessing_time=self.postprocessing_time,
            model_name=self.model_name,
            model_version=self.model_version,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            fps=self.fps,
            memory_usage_mb=self.memory_usage_mb,
            success=self.success,
            error_message=self.error_message,
            metadata=self.metadata.copy()
        )

    def get_class_counts(self) -> Dict[str, int]:
        """Get count of detections per class."""
        class_counts = {}
        for detection in self.detections:
            class_name = detection.class_name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection result to dictionary."""
        return {
            'detections': [det.to_dict() for det in self.detections],
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'timing': {
                'preprocessing_time': self.preprocessing_time,
                'inference_time': self.inference_time,
                'postprocessing_time': self.postprocessing_time,
                'total_time': self.total_time
            },
            'model_info': {
                'name': self.model_name,
                'version': self.model_version
            },
            'frame_info': {
                'width': self.frame_width,
                'height': self.frame_height
            },
            'performance': {
                'fps': self.fps,
                'memory_usage_mb': self.memory_usage_mb
            },
            'status': {
                'success': self.success,
                'error_message': self.error_message
            },
            'statistics': {
                'detection_count': self.detection_count,
                'detection_3d_count': self.detection_3d_count,
                'class_counts': self.get_class_counts()
            },
            'metadata': self.metadata
        }


@dataclass
class ModelInfo:
    """Model metadata and configuration information."""

    # Model identification
    model_type: str  # 'yolo', 'detr', etc.
    model_name: str
    model_version: str = "unknown"

    # File information
    model_path: str = ""
    weights_path: str = ""
    config_path: str = ""

    # Input specifications
    input_width: int = 640
    input_height: int = 640
    input_channels: int = 3
    input_format: str = "RGB"  # RGB, BGR, etc.

    # Model parameters
    num_classes: int = 80
    class_names: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100

    # Performance characteristics
    inference_device: str = "cpu"  # cpu, cuda, mps
    fp16_enabled: bool = False
    batch_size: int = 1

    # Model capabilities
    supports_3d: bool = False
    supports_tracking: bool = False
    supports_segmentation: bool = False

    # Additional configuration
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def input_size(self) -> Tuple[int, int]:
        """Get input size as tuple."""
        return (self.input_width, self.input_height)

    @property
    def model_files_exist(self) -> bool:
        """Check if all required model files exist."""
        files_to_check = [self.model_path, self.weights_path]
        if self.config_path:
            files_to_check.append(self.config_path)

        return all(Path(f).exists() for f in files_to_check if f)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate model configuration."""
        errors = []

        # Check required fields
        if not self.model_name:
            errors.append("Model name is required")

        if not self.model_type:
            errors.append("Model type is required")

        # Check input dimensions
        if self.input_width <= 0 or self.input_height <= 0:
            errors.append("Input dimensions must be positive")

        if self.input_channels not in [1, 3, 4]:
            errors.append("Input channels must be 1, 3, or 4")

        # Check thresholds
        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append("Confidence threshold must be between 0.0 and 1.0")

        if not 0.0 <= self.iou_threshold <= 1.0:
            errors.append("IoU threshold must be between 0.0 and 1.0")

        # Check class information
        if self.num_classes <= 0:
            errors.append("Number of classes must be positive")

        if self.class_names and len(self.class_names) != self.num_classes:
            errors.append("Class names length must match num_classes")

        # Check file existence
        if not self.model_files_exist:
            errors.append("Required model files do not exist")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert model info to dictionary."""
        return {
            'identification': {
                'model_type': self.model_type,
                'model_name': self.model_name,
                'model_version': self.model_version
            },
            'files': {
                'model_path': self.model_path,
                'weights_path': self.weights_path,
                'config_path': self.config_path
            },
            'input_spec': {
                'width': self.input_width,
                'height': self.input_height,
                'channels': self.input_channels,
                'format': self.input_format
            },
            'parameters': {
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'confidence_threshold': self.confidence_threshold,
                'iou_threshold': self.iou_threshold,
                'max_detections': self.max_detections
            },
            'performance': {
                'inference_device': self.inference_device,
                'fp16_enabled': self.fp16_enabled,
                'batch_size': self.batch_size
            },
            'capabilities': {
                'supports_3d': self.supports_3d,
                'supports_tracking': self.supports_tracking,
                'supports_segmentation': self.supports_segmentation
            },
            'config': self.config
        }


class BaseDetector(ABC):
    """Abstract base class for all detection models."""

    def __init__(self, model_info: ModelInfo, config: Dict[str, Any]):
        """
        Initialize base detector.

        Args:
            model_info: Model configuration and metadata
            config: System configuration dictionary
        """
        self.model_info = model_info
        self.config = config
        self.logger = get_logger(f"Detector_{model_info.model_name}")

        # Performance monitoring
        self.inference_times = []
        self.total_detections = 0
        self.successful_inferences = 0
        self.failed_inferences = 0

        # Model state
        self.is_loaded = False
        self.last_error = ""

        # Validate model info
        is_valid, errors = self.model_info.validate()
        if not is_valid:
            raise ValueError(f"Invalid model configuration: {', '.join(errors)}")

        self.logger.info(f"Initialized {model_info.model_type} detector: {model_info.model_name}")

    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the detection model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    def detect(self, image: np.ndarray, **kwargs) -> DetectionResult:
        """
        Perform object detection on an image.

        Args:
            image: Input image as numpy array
            **kwargs: Additional detection parameters

        Returns:
            DetectionResult containing detected objects
        """
        pass

    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image: Raw input image

        Returns:
            Preprocessed image ready for inference
        """
        pass

    @abstractmethod
    def postprocess_outputs(self, outputs: Any, original_shape: Tuple[int, int]) -> List[Detection]:
        """
        Postprocess model outputs to extract detections.

        Args:
            outputs: Raw model outputs
            original_shape: Original image shape (height, width)

        Returns:
            List of Detection objects
        """
        pass

    def validate_image(self, image: np.ndarray) -> bool:
        """
        Validate input image format and dimensions.

        Args:
            image: Input image to validate

        Returns:
            True if image is valid, False otherwise
        """
        if image is None:
            self.logger.error("Input image is None")
            return False

        if not isinstance(image, np.ndarray):
            self.logger.error("Input image must be numpy array")
            return False

        if len(image.shape) not in [2, 3]:
            self.logger.error(f"Invalid image shape: {image.shape}")
            return False

        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            self.logger.error(f"Invalid number of channels: {image.shape[2]}")
            return False

        return True

    def update_performance_stats(self, inference_time: float, detection_count: int, success: bool):
        """Update performance statistics."""
        if success:
            self.inference_times.append(inference_time)
            self.total_detections += detection_count
            self.successful_inferences += 1
        else:
            self.failed_inferences += 1

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        total_inferences = self.successful_inferences + self.failed_inferences

        stats = {
            'total_inferences': total_inferences,
            'successful_inferences': self.successful_inferences,
            'failed_inferences': self.failed_inferences,
            'success_rate': self.successful_inferences / total_inferences if total_inferences > 0 else 0.0,
            'total_detections': self.total_detections,
            'avg_detections_per_frame': self.total_detections / self.successful_inferences if self.successful_inferences > 0 else 0.0
        }

        if self.inference_times:
            stats.update({
                'avg_inference_time': np.mean(self.inference_times),
                'min_inference_time': np.min(self.inference_times),
                'max_inference_time': np.max(self.inference_times),
                'std_inference_time': np.std(self.inference_times),
                'avg_fps': 1.0 / np.mean(self.inference_times) if np.mean(self.inference_times) > 0 else 0.0
            })

        return stats

    def reset_performance_stats(self):
        """Reset all performance statistics."""
        self.inference_times.clear()
        self.total_detections = 0
        self.successful_inferences = 0
        self.failed_inferences = 0
        self.logger.info("Performance statistics reset")

    def cleanup(self):
        """Clean up model resources."""
        self.is_loaded = False
        self.logger.info(f"Detector {self.model_info.model_name} cleaned up")

    def __enter__(self):
        """Context manager entry."""
        if self.load_model():
            return self
        else:
            raise RuntimeError(f"Failed to load model: {self.model_info.model_name}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()