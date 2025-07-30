# Detection package initialization

# Detection-related constants
MAX_DETECTIONS_PER_FRAME = 1000
MIN_CONFIDENCE_THRESHOLD = 0.0
MAX_CONFIDENCE_THRESHOLD = 1.0

# Detection model types
MODEL_TYPE_YOLO = "yolo"
MODEL_TYPE_DETR = "detr"

# Bounding box formats
BBOX_FORMAT_XYXY = "xyxy"  # x1, y1, x2, y2
BBOX_FORMAT_XYWH = "xywh"  # x, y, width, height
BBOX_FORMAT_CXCYWH = "cxcywh"  # center_x, center_y, width, height

# Detection status codes
DETECTION_SUCCESS = 0
DETECTION_FAILED = 1
DETECTION_NO_OBJECTS = 2
DETECTION_MODEL_ERROR = 3

# Initialize availability flags BEFORE any imports
YOLO_AVAILABLE = False
DETR_AVAILABLE = False

# Import main classes
from .base_detector import Detection, Detection3D, DetectionResult, ModelInfo, BaseDetector
from .detector_factory import DetectorFactory, DetectorWrapper
from .postprocessor import Postprocessor, DetectionTracker, DetectionFusion, TemporalSmoother

# Import and register detector implementations
try:
    from .yolo_detector import YOLODetector, register_yolo_detector
    register_yolo_detector()
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from .detr_detector import DETRDetector, register_detr_detector
    register_detr_detector()
    DETR_AVAILABLE = True
except ImportError:
    DETR_AVAILABLE = False

__all__ = [
    'Detection', 'Detection3D', 'DetectionResult', 'ModelInfo', 'BaseDetector',
    'DetectorFactory', 'DetectorWrapper',
    'Postprocessor', 'DetectionTracker', 'DetectionFusion', 'TemporalSmoother',
    'MODEL_TYPE_YOLO', 'MODEL_TYPE_DETR',
    'BBOX_FORMAT_XYXY', 'BBOX_FORMAT_XYWH', 'BBOX_FORMAT_CXCYWH',
    'DETECTION_SUCCESS', 'DETECTION_FAILED', 'DETECTION_NO_OBJECTS', 'DETECTION_MODEL_ERROR',
    'YOLO_AVAILABLE', 'DETR_AVAILABLE'
]

# Optional: YOLODetector and DETRDetector if available
if YOLO_AVAILABLE:
    __all__.append('YOLODetector')

if DETR_AVAILABLE:
    __all__.append('DETRDetector')