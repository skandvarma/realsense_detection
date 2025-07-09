"""
Model-agnostic postprocessing utilities and 3D integration for object detection.
"""

import time
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple, Union, TYPE_CHECKING
from collections import defaultdict, deque
from dataclasses import dataclass
import threading

from .base_detector import Detection, Detection3D, DetectionResult
from ..utils.logger import get_logger

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from ..camera.depth_processor import DepthProcessor, Point3D, BoundingBox3D


@dataclass
class TrackingInfo:
    """Information for detection tracking across frames."""
    track_id: int
    last_seen_frame: int
    detection_history: List[Detection]
    confidence_history: List[float]
    position_history: List[Tuple[float, float]]
    age: int = 0
    hits: int = 0
    time_since_update: int = 0


class DetectionTracker:
    """Simple detection tracker for ID assignment and temporal smoothing."""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize detection tracker.

        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum hits before confirming track
            iou_threshold: IoU threshold for track association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks: Dict[int, TrackingInfo] = {}
        self.next_track_id = 0
        self.frame_count = 0

        self.logger = get_logger("DetectionTracker")

    def update(self, detections: List[Detection], frame_id: int) -> List[Detection]:
        """
        Update tracks with new detections and assign IDs.

        Args:
            detections: List of detections for current frame
            frame_id: Current frame ID

        Returns:
            List of detections with assigned track IDs
        """
        self.frame_count = frame_id

        # Associate detections with existing tracks
        matched_tracks, unmatched_detections, unmatched_tracks = self._associate_detections(detections)

        # Update matched tracks
        for track_id, detection in matched_tracks:
            track = self.tracks[track_id]
            track.detection_history.append(detection)
            track.confidence_history.append(detection.confidence)
            track.position_history.append(detection.center)
            track.last_seen_frame = frame_id
            track.hits += 1
            track.time_since_update = 0

            # Assign track ID to detection
            detection.detection_id = track_id

        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1

            self.tracks[track_id] = TrackingInfo(
                track_id=track_id,
                last_seen_frame=frame_id,
                detection_history=[detection],
                confidence_history=[detection.confidence],
                position_history=[detection.center],
                hits=1
            )

            detection.detection_id = track_id

        # Update unmatched tracks
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.time_since_update += 1
            track.age += 1

        # Remove old tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_age:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        return detections

    def _associate_detections(self, detections: List[Detection]) -> Tuple[List[Tuple[int, Detection]], List[Detection], List[int]]:
        """Associate detections with existing tracks using IoU."""
        matched_tracks = []
        unmatched_detections = list(detections)
        unmatched_tracks = list(self.tracks.keys())

        if not detections or not self.tracks:
            return matched_tracks, unmatched_detections, unmatched_tracks

        # Calculate IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        track_ids = list(self.tracks.keys())

        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            if track.detection_history:
                last_detection = track.detection_history[-1]
                for j, detection in enumerate(detections):
                    iou_matrix[i, j] = last_detection.iou(detection)

        # Simple greedy assignment
        while True:
            # Find best match
            max_iou = 0
            best_track_idx = -1
            best_det_idx = -1

            for i in range(len(track_ids)):
                for j in range(len(detections)):
                    if iou_matrix[i, j] > max_iou and iou_matrix[i, j] > self.iou_threshold:
                        max_iou = iou_matrix[i, j]
                        best_track_idx = i
                        best_det_idx = j

            if best_track_idx == -1:
                break

            # Make assignment
            track_id = track_ids[best_track_idx]
            detection = detections[best_det_idx]
            matched_tracks.append((track_id, detection))

            # Remove from unmatched lists
            if detection in unmatched_detections:
                unmatched_detections.remove(detection)
            if track_id in unmatched_tracks:
                unmatched_tracks.remove(track_id)

            # Remove from consideration
            iou_matrix[best_track_idx, :] = 0
            iou_matrix[:, best_det_idx] = 0

        return matched_tracks, unmatched_detections, unmatched_tracks

    def get_confirmed_tracks(self) -> List[TrackingInfo]:
        """Get tracks that have been confirmed (min_hits reached)."""
        return [track for track in self.tracks.values() if track.hits >= self.min_hits]


class DetectionFusion:
    """Ensemble methods for fusing detections from multiple models."""

    def __init__(self, iou_threshold: float = 0.5, confidence_weights: Optional[Dict[str, float]] = None):
        """
        Initialize detection fusion.

        Args:
            iou_threshold: IoU threshold for detection matching
            confidence_weights: Weights for different model types
        """
        self.iou_threshold = iou_threshold
        self.confidence_weights = confidence_weights or {'yolo': 1.0, 'detr': 1.0}
        self.logger = get_logger("DetectionFusion")

    def fuse_detections(self, detection_results: List[DetectionResult]) -> List[Detection]:
        """
        Fuse detections from multiple models.

        Args:
            detection_results: List of detection results from different models

        Returns:
            Fused list of detections
        """
        if not detection_results:
            return []

        if len(detection_results) == 1:
            return detection_results[0].detections

        # Collect all detections with model info
        all_detections = []
        for result in detection_results:
            model_type = result.model_name.split('_')[0].lower()  # Extract model type
            weight = self.confidence_weights.get(model_type, 1.0)

            for detection in result.detections:
                # Weight the confidence
                weighted_detection = Detection(
                    bbox=detection.bbox,
                    confidence=detection.confidence * weight,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    detection_id=detection.detection_id,
                    metadata={**detection.metadata, 'model_weight': weight, 'source_model': model_type}
                )
                all_detections.append(weighted_detection)

        # Apply non-maximum suppression across models
        fused_detections = self._cross_model_nms(all_detections)

        return fused_detections

    def _cross_model_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply NMS across detections from multiple models."""
        if not detections:
            return []

        # Group by class
        class_groups = defaultdict(list)
        for det in detections:
            class_groups[det.class_id].append(det)

        final_detections = []

        for class_id, class_detections in class_groups.items():
            # Sort by confidence (descending)
            class_detections.sort(key=lambda x: x.confidence, reverse=True)

            # Apply NMS within class
            keep = []
            suppressed = set()

            for i, det1 in enumerate(class_detections):
                if i in suppressed:
                    continue

                keep.append(det1)

                # Suppress overlapping detections
                for j, det2 in enumerate(class_detections[i+1:], i+1):
                    if j in suppressed:
                        continue

                    if det1.iou(det2) > self.iou_threshold:
                        # Choose detection with higher confidence or from preferred model
                        if self._should_suppress(det1, det2):
                            suppressed.add(i)
                            if det1 in keep:
                                keep.remove(det1)
                            keep.append(det2)
                        else:
                            suppressed.add(j)

            final_detections.extend(keep)

        return final_detections

    def _should_suppress(self, det1: Detection, det2: Detection) -> bool:
        """Determine if det1 should be suppressed in favor of det2."""
        # Prefer higher confidence
        if det2.confidence > det1.confidence * 1.1:  # 10% threshold
            return True

        # Prefer certain model types if confidence is similar
        model1 = det1.metadata.get('source_model', 'unknown')
        model2 = det2.metadata.get('source_model', 'unknown')

        # YOLO generally better for speed-accuracy trade-off
        if model1 == 'detr' and model2 == 'yolo' and abs(det1.confidence - det2.confidence) < 0.1:
            return True

        return False


class TemporalSmoother:
    """Temporal smoothing for detection stability across frames."""

    def __init__(self, window_size: int = 5, confidence_alpha: float = 0.7, position_alpha: float = 0.5):
        """
        Initialize temporal smoother.

        Args:
            window_size: Size of smoothing window
            confidence_alpha: Alpha for confidence smoothing
            position_alpha: Alpha for position smoothing
        """
        self.window_size = window_size
        self.confidence_alpha = confidence_alpha
        self.position_alpha = position_alpha

        self.detection_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.logger = get_logger("TemporalSmoother")

    def smooth_detections(self, detections: List[Detection]) -> List[Detection]:
        """
        Apply temporal smoothing to detections.

        Args:
            detections: Current frame detections

        Returns:
            Smoothed detections
        """
        smoothed_detections = []

        for detection in detections:
            track_id = detection.detection_id
            if track_id is None:
                smoothed_detections.append(detection)
                continue

            # Add to history
            self.detection_history[track_id].append(detection)

            # Apply smoothing if we have enough history
            if len(self.detection_history[track_id]) > 1:
                smoothed_detection = self._smooth_single_detection(track_id, detection)
                smoothed_detections.append(smoothed_detection)
            else:
                smoothed_detections.append(detection)

        return smoothed_detections

    def _smooth_single_detection(self, track_id: int, current_detection: Detection) -> Detection:
        """Smooth a single detection using its history."""
        history = list(self.detection_history[track_id])

        # Smooth confidence
        confidences = [det.confidence for det in history]
        smoothed_confidence = self._exponential_smooth(confidences, self.confidence_alpha)

        # Smooth position
        positions = [det.center for det in history]
        smoothed_center = self._smooth_position(positions, self.position_alpha)

        # Calculate smoothed bbox
        current_width = current_detection.width
        current_height = current_detection.height

        smoothed_bbox = (
            smoothed_center[0] - current_width / 2,
            smoothed_center[1] - current_height / 2,
            smoothed_center[0] + current_width / 2,
            smoothed_center[1] + current_height / 2
        )

        # Create smoothed detection
        return Detection(
            bbox=smoothed_bbox,
            confidence=smoothed_confidence,
            class_id=current_detection.class_id,
            class_name=current_detection.class_name,
            detection_id=current_detection.detection_id,
            metadata={**current_detection.metadata, 'temporally_smoothed': True}
        )

    def _exponential_smooth(self, values: List[float], alpha: float) -> float:
        """Apply exponential smoothing to a series of values."""
        if not values:
            return 0.0

        smoothed = values[0]
        for value in values[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed

        return smoothed

    def _smooth_position(self, positions: List[Tuple[float, float]], alpha: float) -> Tuple[float, float]:
        """Smooth position coordinates."""
        if not positions:
            return (0.0, 0.0)

        x_values = [pos[0] for pos in positions]
        y_values = [pos[1] for pos in positions]

        smoothed_x = self._exponential_smooth(x_values, alpha)
        smoothed_y = self._exponential_smooth(y_values, alpha)

        return (smoothed_x, smoothed_y)


class Postprocessor:
    """Main postprocessor class for standardizing and enhancing detection outputs."""

    def __init__(self, config: Dict[str, Any], depth_processor: Optional['DepthProcessor'] = None):
        """
        Initialize postprocessor.

        Args:
            config: System configuration
            depth_processor: Optional depth processor for 3D integration
        """
        self.config = config
        self.depth_processor = depth_processor
        self.logger = get_logger("Postprocessor")

        # Initialize components
        integration_config = config.get('integration', {})
        tracking_config = integration_config.get('tracking', {})

        self.tracker = DetectionTracker(
            max_age=tracking_config.get('max_age', 30),
            min_hits=tracking_config.get('min_hits', 3),
            iou_threshold=tracking_config.get('distance_threshold', 0.3)
        )

        self.fusion = DetectionFusion(
            iou_threshold=0.5,
            confidence_weights={'yolo': 1.0, 'detr': 0.9}  # Slight preference for YOLO
        )

        self.smoother = TemporalSmoother(
            window_size=5,
            confidence_alpha=0.7,
            position_alpha=0.5
        )

        # 3D integration settings
        self.enable_3d = depth_processor is not None
        self.coordinate_system = integration_config.get('coordinates', {}).get('origin', 'camera')

        # Processing lock for thread safety
        self.processing_lock = threading.Lock()

        self.logger.info("Postprocessor initialized")
        if self.enable_3d:
            self.logger.info("3D integration enabled")

    def process_detection_result(self, result: DetectionResult,
                               depth_frame: Optional[np.ndarray] = None,
                               frame_id: Optional[int] = None) -> DetectionResult:
        """
        Process and enhance detection result with postprocessing pipeline.

        Args:
            result: Raw detection result from model
            depth_frame: Optional depth frame for 3D integration
            frame_id: Frame ID for tracking

        Returns:
            Enhanced detection result
        """
        with self.processing_lock:
            start_time = time.time()

            try:
                # Start with original detections
                detections = result.detections.copy()

                # Apply confidence filtering
                min_confidence = self.config.get('detection', {}).get('confidence_threshold', 0.5)
                detections = [det for det in detections if det.confidence >= min_confidence]

                # Apply class filtering if specified
                target_classes = self.config.get('detection', {}).get('target_classes', [])
                if target_classes:
                    detections = [det for det in detections if det.class_name in target_classes]

                # Apply tracking
                if frame_id is not None:
                    detections = self.tracker.update(detections, frame_id)

                # Apply temporal smoothing
                detections = self.smoother.smooth_detections(detections)

                # Convert to 3D if depth frame is available
                if self.enable_3d and depth_frame is not None:
                    detections = self._convert_to_3d(detections, depth_frame)

                # Update result
                enhanced_result = DetectionResult(
                    detections=detections,
                    frame_id=result.frame_id,
                    timestamp=result.timestamp,
                    preprocessing_time=result.preprocessing_time,
                    inference_time=result.inference_time,
                    postprocessing_time=result.postprocessing_time + (time.time() - start_time),
                    model_name=result.model_name,
                    model_version=result.model_version,
                    frame_width=result.frame_width,
                    frame_height=result.frame_height,
                    fps=result.fps,
                    memory_usage_mb=result.memory_usage_mb,
                    success=result.success,
                    error_message=result.error_message,
                    metadata={
                        **result.metadata,
                        'postprocessed': True,
                        'tracking_enabled': True,
                        '3d_enabled': self.enable_3d,
                        'confirmed_tracks': len(self.tracker.get_confirmed_tracks())
                    }
                )

                return enhanced_result

            except Exception as e:
                self.logger.error(f"Postprocessing failed: {e}")
                # Return original result on failure
                result.metadata['postprocessing_error'] = str(e)
                return result

    def process_multiple_results(self, results: List[DetectionResult],
                               depth_frame: Optional[np.ndarray] = None,
                               frame_id: Optional[int] = None) -> DetectionResult:
        """
        Process and fuse multiple detection results from different models.

        Args:
            results: List of detection results from different models
            depth_frame: Optional depth frame for 3D integration
            frame_id: Frame ID for tracking

        Returns:
            Fused and enhanced detection result
        """
        if not results:
            return DetectionResult(detections=[], frame_id=frame_id or 0, timestamp=time.time(), success=False)

        if len(results) == 1:
            return self.process_detection_result(results[0], depth_frame, frame_id)

        with self.processing_lock:
            start_time = time.time()

            try:
                # Fuse detections from multiple models
                fused_detections = self.fusion.fuse_detections(results)

                # Create combined result
                primary_result = results[0]  # Use first result as template
                fused_result = DetectionResult(
                    detections=fused_detections,
                    frame_id=primary_result.frame_id,
                    timestamp=primary_result.timestamp,
                    preprocessing_time=max(r.preprocessing_time for r in results),
                    inference_time=max(r.inference_time for r in results),
                    postprocessing_time=max(r.postprocessing_time for r in results),
                    model_name=f"fused_{len(results)}_models",
                    frame_width=primary_result.frame_width,
                    frame_height=primary_result.frame_height,
                    success=all(r.success for r in results),
                    metadata={
                        'fused_models': [r.model_name for r in results],
                        'fusion_method': 'cross_model_nms'
                    }
                )

                # Apply standard postprocessing
                enhanced_result = self.process_detection_result(fused_result, depth_frame, frame_id)

                return enhanced_result

            except Exception as e:
                self.logger.error(f"Multi-model processing failed: {e}")
                # Return first result as fallback
                return self.process_detection_result(results[0], depth_frame, frame_id)

    def _convert_to_3d(self, detections: List[Detection], depth_frame: np.ndarray) -> List[Detection3D]:
        """
        Convert 2D detections to 3D using depth information.

        Args:
            detections: List of 2D detections
            depth_frame: Depth frame for 3D conversion

        Returns:
            List of 3D detections
        """
        detections_3d = []

        for detection in detections:
            try:
                # Convert float bbox to int bbox for depth processing
                x1, y1, x2, y2 = detection.bbox
                int_bbox = (int(x1), int(y1), int(x2), int(y2))

                # Get 3D bounding box
                bbox_3d = self.depth_processor.detection_to_3d(
                    int_bbox, depth_frame, sampling_strategy='center_robust'
                )

                if bbox_3d:
                    # Create 3D detection
                    detection_3d = Detection3D(
                        bbox=detection.bbox,
                        confidence=detection.confidence,
                        class_id=detection.class_id,
                        class_name=detection.class_name,
                        detection_id=detection.detection_id,
                        center_3d=(bbox_3d.center.x, bbox_3d.center.y, bbox_3d.center.z),
                        distance=bbox_3d.center.z,  # Z coordinate is distance from camera
                        depth_confidence=bbox_3d.confidence,
                        dimensions_3d=bbox_3d.dimensions,
                        metadata={
                            **detection.metadata,
                            '3d_converted': True,
                            'coordinate_system': self.coordinate_system,
                            'depth_sampling_strategy': 'center_robust'
                        }
                    )

                    detections_3d.append(detection_3d)
                else:
                    # Keep as 2D if 3D conversion failed
                    detection.metadata['3d_conversion_failed'] = True
                    detections_3d.append(detection)

            except Exception as e:
                self.logger.warning(f"3D conversion failed for detection {detection.detection_id}: {e}")
                detection.metadata['3d_conversion_error'] = str(e)
                detections_3d.append(detection)

        return detections_3d

    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get current tracking statistics."""
        confirmed_tracks = self.tracker.get_confirmed_tracks()

        return {
            'total_tracks': len(self.tracker.tracks),
            'confirmed_tracks': len(confirmed_tracks),
            'active_tracks': len([t for t in self.tracker.tracks.values() if t.time_since_update == 0]),
            'next_track_id': self.tracker.next_track_id,
            'frame_count': self.tracker.frame_count
        }

    def reset_tracking(self):
        """Reset tracking state."""
        self.tracker.tracks.clear()
        self.tracker.next_track_id = 0
        self.detection_history.clear()
        self.logger.info("Tracking state reset")

    def update_depth_processor(self, depth_processor: 'DepthProcessor'):
        """Update the depth processor for 3D integration."""
        self.depth_processor = depth_processor
        self.enable_3d = depth_processor is not None
        self.logger.info(f"Depth processor updated, 3D integration: {self.enable_3d}")


# Convenience function for easy postprocessor creation
def create_postprocessor(config: Dict[str, Any], depth_processor: Optional['DepthProcessor'] = None) -> Postprocessor:
    """
    Create a configured postprocessor instance.

    Args:
        config: System configuration
        depth_processor: Optional depth processor

    Returns:
        Configured Postprocessor instance
    """
    return Postprocessor(config, depth_processor)