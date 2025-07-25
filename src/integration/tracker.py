"""
CUDA-accelerated 3D object tracking system with advanced Kalman filtering and trajectory prediction.
Optimized for real-time performance with dozens of simultaneous tracks.
"""

import torch
import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import json

from .cuda_tracking_kernels import CUDATrackingKernels, KernelProfiler
from .gpu_memory_manager import CUDAMemoryManager, MemoryPoolType
from ..detection.base_detector import Detection, Detection3D
from ..camera.depth_processor import Point3D, BoundingBox3D
from ..utils.logger import get_logger, PerformanceMonitor


class TrackState(Enum):
    """Track lifecycle states."""
    TENTATIVE = "tentative"  # New track, not yet confirmed
    CONFIRMED = "confirmed"  # Stable track with enough hits
    LOST = "lost"  # Track lost but still being predicted
    DELETED = "deleted"  # Track marked for deletion


@dataclass
class Track3D:
    """3D track representation with complete state information."""
    track_id: int
    state: TrackState = TrackState.TENTATIVE

    # 3D state vector: [x, y, z, vx, vy, vz, ax, ay, az]
    position: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    velocity: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    acceleration: torch.Tensor = field(default_factory=lambda: torch.zeros(3))

    # State uncertainty (covariance matrix)
    covariance: torch.Tensor = field(default_factory=lambda: torch.eye(9))

    # Detection association
    last_detection: Optional[Detection3D] = None
    detection_history: List[Detection3D] = field(default_factory=list)

    # Timing information
    creation_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    last_detection_time: float = field(default_factory=time.time)

    # Track quality metrics
    confidence: float = 1.0
    hit_count: int = 0
    miss_count: int = 0
    age: int = 0
    time_since_update: int = 0

    # Trajectory analysis
    trajectory_points: deque = field(default_factory=lambda: deque(maxlen=50))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=20))
    acceleration_history: deque = field(default_factory=lambda: deque(maxlen=10))

    # Class and appearance
    class_id: int = -1
    class_name: str = "unknown"
    class_confidence: float = 0.0
    appearance_features: Optional[torch.Tensor] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize tensors on correct device after creation."""
        if hasattr(self, '_device'):
            self.position = self.position.to(self._device)
            self.velocity = self.velocity.to(self._device)
            self.acceleration = self.acceleration.to(self._device)
            self.covariance = self.covariance.to(self._device)

    @property
    def state_vector(self) -> torch.Tensor:
        """Get complete 9D state vector."""
        return torch.cat([self.position, self.velocity, self.acceleration])

    @state_vector.setter
    def state_vector(self, value: torch.Tensor):
        """Set complete state vector."""
        self.position = value[:3]
        self.velocity = value[3:6]
        self.acceleration = value[6:9]

    @property
    def is_confirmed(self) -> bool:
        """Check if track is confirmed."""
        return self.state == TrackState.CONFIRMED

    @property
    def is_active(self) -> bool:
        """Check if track is active (confirmed or tentative)."""
        return self.state in [TrackState.TENTATIVE, TrackState.CONFIRMED]

    @property
    def speed(self) -> float:
        """Get current speed."""
        return torch.norm(self.velocity).item()

    @property
    def direction(self) -> torch.Tensor:
        """Get normalized direction vector."""
        velocity_norm = torch.norm(self.velocity)
        if velocity_norm > 1e-6:
            return self.velocity / velocity_norm
        return torch.zeros_like(self.velocity)

    def update_quality_metrics(self, matched: bool):
        """Update track quality metrics."""
        self.age += 1

        if matched:
            self.hit_count += 1
            self.time_since_update = 0
            self.last_detection_time = time.time()

            # Update confidence based on hit rate
            hit_rate = self.hit_count / max(self.age, 1)
            self.confidence = min(1.0, hit_rate * 1.2)
        else:
            self.miss_count += 1
            self.time_since_update += 1

            # Decay confidence
            self.confidence *= 0.95

    def add_trajectory_point(self, point: torch.Tensor, timestamp: float):
        """Add point to trajectory history."""
        self.trajectory_points.append((point.clone(), timestamp))

        # Update velocity and acceleration history
        if len(self.trajectory_points) >= 2:
            prev_point, prev_time = self.trajectory_points[-2]
            dt = timestamp - prev_time
            if dt > 0:
                velocity = (point - prev_point) / dt
                self.velocity_history.append(velocity.clone())

                if len(self.velocity_history) >= 2:
                    prev_velocity = self.velocity_history[-2]
                    acceleration = (velocity - prev_velocity) / dt
                    self.acceleration_history.append(acceleration.clone())

    def get_predicted_position(self, dt: float) -> torch.Tensor:
        """Get predicted position after time dt."""
        return (self.position +
                self.velocity * dt +
                0.5 * self.acceleration * dt * dt)

    def to_dict(self) -> Dict[str, Any]:
        """Convert track to dictionary for serialization."""
        return {
            'track_id': self.track_id,
            'state': self.state.value,
            'position': self.position.cpu().numpy().tolist(),
            'velocity': self.velocity.cpu().numpy().tolist(),
            'acceleration': self.acceleration.cpu().numpy().tolist(),
            'confidence': self.confidence,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'age': self.age,
            'time_since_update': self.time_since_update,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'class_confidence': self.class_confidence,
            'creation_time': self.creation_time,
            'last_update_time': self.last_update_time,
            'metadata': self.metadata
        }


class TrackingParameters:
    """Configuration parameters for 3D tracking."""

    def __init__(self, config: Dict[str, Any]):
        tracking_config = config.get('integration', {}).get('tracking', {})

        # Track lifecycle parameters
        self.max_age = tracking_config.get('max_age', 30)
        self.min_hits = tracking_config.get('min_hits', 3)
        self.hit_ratio_threshold = tracking_config.get('hit_ratio_threshold', 0.6)

        # Association parameters
        self.distance_threshold = tracking_config.get('distance_threshold', 2.0)
        self.mahalanobis_threshold = tracking_config.get('mahalanobis_threshold', 9.21)  # Chi-squared 3DOF 99%
        self.iou_threshold = tracking_config.get('iou_threshold', 0.3)

        # Kalman filter parameters
        self.process_noise_std = tracking_config.get('process_noise_std', 0.1)
        self.measurement_noise_std = tracking_config.get('measurement_noise_std', 0.5)
        self.initial_position_std = tracking_config.get('initial_position_std', 1.0)
        self.initial_velocity_std = tracking_config.get('initial_velocity_std', 0.5)
        self.initial_acceleration_std = tracking_config.get('initial_acceleration_std', 0.1)

        # Prediction parameters
        self.max_prediction_time = tracking_config.get('max_prediction_time', 1.0)
        self.velocity_smoothing_alpha = tracking_config.get('velocity_smoothing_alpha', 0.7)

        # Performance parameters
        self.max_tracks = tracking_config.get('max_tracks', 100)
        self.batch_processing = tracking_config.get('batch_processing', True)
        self.use_appearance_features = tracking_config.get('use_appearance_features', False)


class Object3DTracker:
    """CUDA-accelerated 3D multi-object tracker with advanced Kalman filtering."""

    def __init__(self, config: Dict[str, Any], memory_manager: Optional[CUDAMemoryManager] = None):
        """
        Initialize 3D object tracker.

        Args:
            config: Configuration dictionary
            memory_manager: Optional GPU memory manager
        """
        self.config = config
        self.params = TrackingParameters(config)
        self.memory_manager = memory_manager
        self.logger = get_logger("Object3DTracker")

        # GPU setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gpu = torch.cuda.is_available()

        # Initialize CUDA kernels
        if self.use_gpu:
            self.kernels = CUDATrackingKernels(
                self.device,
                max_tracks=self.params.max_tracks,
                max_detections=50
            )
            self.profiler = KernelProfiler(self.kernels) if config.get('debug', {}).get('profile_kernels',
                                                                                        False) else None
        else:
            self.kernels = None
            self.profiler = None
            self.logger.warning("CUDA not available, using CPU tracking")

        # Track management
        self.tracks: Dict[int, Track3D] = {}
        self.next_track_id = 0
        self.frame_count = 0

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor(window_size=100)

        # Threading for real-time processing
        self.processing_lock = threading.RLock()
        self.enable_parallel_processing = config.get('integration', {}).get('performance', {}).get('parallel_tracking',
                                                                                                   True)

        # Statistics
        self.stats = {
            'total_tracks_created': 0,
            'total_associations': 0,
            'total_predictions': 0,
            'avg_tracking_time': 0.0,
            'gpu_utilization': 0.0
        }

        self.logger.info(f"Object3DTracker initialized on {self.device}")
        self.logger.info(f"Max tracks: {self.params.max_tracks}, GPU acceleration: {self.use_gpu}")

    def update(self, detections: List[Detection3D], timestamp: float) -> List[Track3D]:
        """
        Update tracker with new detections.

        Args:
            detections: List of 3D detections
            timestamp: Current timestamp

        Returns:
            List of active tracks
        """
        with self.processing_lock:
            start_time = time.time()

            self.frame_count += 1

            # Convert detections to GPU tensors
            detection_positions = self._detections_to_tensors(detections)

            # Prediction step
            self._predict_tracks(timestamp)

            # Data association
            assignments, unmatched_tracks, unmatched_detections = self._associate_detections(
                detection_positions, detections
            )

            # Update matched tracks
            self._update_matched_tracks(assignments, detections, timestamp)

            # Handle unmatched tracks
            self._handle_unmatched_tracks(unmatched_tracks)

            # Create new tracks from unmatched detections
            self._create_new_tracks(unmatched_detections, detections, timestamp)

            # Track lifecycle management
            self._manage_track_lifecycle()

            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_monitor.add_metric('tracking_update_time', processing_time)
            self.stats['avg_tracking_time'] = self.performance_monitor.get_statistics('tracking_update_time').get(
                'mean', 0)

            # Return active tracks
            active_tracks = [track for track in self.tracks.values() if track.is_active]

            self.logger.debug(f"Frame {self.frame_count}: {len(active_tracks)} active tracks, "
                              f"{len(assignments[0])} associations, {processing_time * 1000:.1f}ms")

            return active_tracks

    def _detections_to_tensors(self, detections: List[Detection3D]) -> torch.Tensor:
        """Convert detections to GPU tensors."""
        if not detections:
            return torch.zeros((0, 3), device=self.device)

        positions = []
        for detection in detections:
            if hasattr(detection, 'center_3d') and detection.center_3d != (0, 0, 0):
                positions.append(detection.center_3d)
            else:
                # Fallback to 2D center + estimated depth
                x, y, _, _ = detection.bbox
                center_x = (x + _) / 2
                center_y = (y + _) / 2
                estimated_depth = detection.distance if hasattr(detection, 'distance') else 5.0
                positions.append((center_x * 0.01, center_y * 0.01, estimated_depth))  # Scale 2D to rough 3D

        return torch.tensor(positions, device=self.device, dtype=torch.float32)

    def _predict_tracks(self, timestamp: float):
        """Predict all track states to current timestamp."""
        if not self.tracks:
            return

        active_tracks = [track for track in self.tracks.values() if track.is_active]
        if not active_tracks:
            return

        # Calculate time steps
        dts = torch.tensor([timestamp - track.last_update_time for track in active_tracks],
                           device=self.device, dtype=torch.float32)

        if self.use_gpu and self.kernels:
            # GPU-accelerated batch prediction
            track_states = torch.stack([track.state_vector for track in active_tracks])
            track_covariances = torch.stack([track.covariance for track in active_tracks])

            # Use average dt for batch processing (more efficient)
            avg_dt = torch.mean(dts).item()

            predicted_states, predicted_covariances = self.kernels.predict_tracks_kalman(
                track_states, track_covariances, avg_dt
            )

            # Update track states
            for i, track in enumerate(active_tracks):
                track.state_vector = predicted_states[i]
                track.covariance = predicted_covariances[i]
                track.last_update_time = timestamp
        else:
            # CPU fallback - predict each track individually
            for track in active_tracks:
                dt = timestamp - track.last_update_time
                self._predict_single_track(track, dt)
                track.last_update_time = timestamp

    def _predict_single_track(self, track: Track3D, dt: float):
        """Predict single track state (CPU implementation)."""
        # Simple constant velocity model
        track.position += track.velocity * dt

        # Add process noise to covariance
        process_noise = torch.eye(9, device=self.device) * (self.params.process_noise_std ** 2)
        track.covariance += process_noise * dt

    def _associate_detections(self, detection_positions: torch.Tensor,
                              detections: List[Detection3D]) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
    List[int], List[int]]:
        """Associate detections with existing tracks."""
        active_tracks = [track for track in self.tracks.values() if track.is_active]

        if not active_tracks or len(detections) == 0:
            return (torch.tensor([], dtype=torch.long, device=self.device),
                    torch.tensor([], dtype=torch.long, device=self.device)), \
                list(range(len(active_tracks))), list(range(len(detections)))

        # Get track positions
        track_positions = torch.stack([track.position for track in active_tracks])

        if self.use_gpu and self.kernels:
            # GPU-accelerated association
            distance_matrix = self.kernels.compute_distance_matrix(track_positions, detection_positions)

            # Apply distance threshold
            cost_matrix = distance_matrix.clone()
            cost_matrix[cost_matrix > self.params.distance_threshold] = float('inf')

            # Solve assignment problem
            track_indices, detection_indices = self.kernels.solve_assignment_problem(
                cost_matrix, self.params.distance_threshold
            )
        else:
            # CPU fallback
            track_indices, detection_indices = self._associate_cpu(track_positions, detection_positions)

        # Convert to lists for unmatched identification
        matched_track_indices = track_indices.cpu().numpy().tolist()
        matched_detection_indices = detection_indices.cpu().numpy().tolist()

        unmatched_tracks = [i for i in range(len(active_tracks)) if i not in matched_track_indices]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_detection_indices]

        self.stats['total_associations'] += len(matched_track_indices)

        return (track_indices, detection_indices), unmatched_tracks, unmatched_detections

    def _associate_cpu(self, track_positions: torch.Tensor,
                       detection_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """CPU implementation of detection association."""
        n_tracks = track_positions.shape[0]
        n_detections = detection_positions.shape[0]

        if n_tracks == 0 or n_detections == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        # Compute distance matrix
        distances = torch.cdist(track_positions, detection_positions)

        # Simple greedy assignment
        assignments = []
        used_tracks = set()
        used_detections = set()

        # Sort by distance and assign greedily
        flat_distances = distances.flatten()
        sorted_indices = torch.argsort(flat_distances)

        for flat_idx in sorted_indices:
            track_idx = flat_idx // n_detections
            detection_idx = flat_idx % n_detections

            distance = distances[track_idx, detection_idx].item()

            if (distance <= self.params.distance_threshold and
                    track_idx.item() not in used_tracks and
                    detection_idx.item() not in used_detections):
                assignments.append((track_idx.item(), detection_idx.item()))
                used_tracks.add(track_idx.item())
                used_detections.add(detection_idx.item())

        if assignments:
            track_indices = torch.tensor([a[0] for a in assignments], dtype=torch.long)
            detection_indices = torch.tensor([a[1] for a in assignments], dtype=torch.long)
        else:
            track_indices = torch.tensor([], dtype=torch.long)
            detection_indices = torch.tensor([], dtype=torch.long)

        return track_indices, detection_indices

    def _update_matched_tracks(self, assignments: Tuple[torch.Tensor, torch.Tensor],
                               detections: List[Detection3D], timestamp: float):
        """Update tracks with their matched detections."""
        track_indices, detection_indices = assignments

        if len(track_indices) == 0:
            return

        active_tracks = [track for track in self.tracks.values() if track.is_active]

        for i in range(len(track_indices)):
            track_idx = track_indices[i].item()
            detection_idx = detection_indices[i].item()

            track = active_tracks[track_idx]
            detection = detections[detection_idx]

            # Update track with detection
            self._update_single_track(track, detection, timestamp)

    def _update_single_track(self, track: Track3D, detection: Detection3D, timestamp: float):
        """Update single track with detection."""
        # Extract measurement
        if hasattr(detection, 'center_3d') and detection.center_3d != (0, 0, 0):
            measurement = torch.tensor(detection.center_3d, device=self.device, dtype=torch.float32)
        else:
            # Fallback measurement from 2D detection
            x, y, x2, y2 = detection.bbox
            center_x = (x + x2) / 2
            center_y = (y + y2) / 2
            estimated_depth = detection.distance if hasattr(detection, 'distance') else 5.0
            measurement = torch.tensor([center_x * 0.01, center_y * 0.01, estimated_depth],
                                       device=self.device, dtype=torch.float32)

        if self.use_gpu and self.kernels:
            # GPU Kalman update
            predicted_state = track.state_vector.unsqueeze(0)
            predicted_cov = track.covariance.unsqueeze(0)
            measurement_batch = measurement.unsqueeze(0)

            updated_states, updated_covariances = self.kernels.update_tracks_kalman(
                predicted_state, predicted_cov, measurement_batch
            )

            track.state_vector = updated_states[0]
            track.covariance = updated_covariances[0]
        else:
            # Simple CPU update
            # Weighted average between prediction and measurement
            alpha = 0.7  # Trust measurement more
            track.position = alpha * measurement + (1 - alpha) * track.position

        # Update track metadata
        track.last_detection = detection
        track.detection_history.append(detection)
        if len(track.detection_history) > 10:
            track.detection_history.pop(0)

        # Update quality metrics
        track.update_quality_metrics(matched=True)

        # Update class information
        if detection.confidence > track.class_confidence:
            track.class_id = detection.class_id
            track.class_name = detection.class_name
            track.class_confidence = detection.confidence

        # Add to trajectory
        track.add_trajectory_point(track.position, timestamp)

        # Promote tentative tracks to confirmed
        if (track.state == TrackState.TENTATIVE and
                track.hit_count >= self.params.min_hits):
            track.state = TrackState.CONFIRMED
            self.logger.debug(f"Track {track.track_id} confirmed")

    def _handle_unmatched_tracks(self, unmatched_track_indices: List[int]):
        """Handle tracks that weren't matched with detections."""
        active_tracks = [track for track in self.tracks.values() if track.is_active]

        for track_idx in unmatched_track_indices:
            if track_idx < len(active_tracks):
                track = active_tracks[track_idx]
                track.update_quality_metrics(matched=False)

                # Mark tracks as lost if they've been unmatched too long
                if track.time_since_update > 5:
                    track.state = TrackState.LOST
                    self.logger.debug(f"Track {track.track_id} marked as lost")

    def _create_new_tracks(self, unmatched_detection_indices: List[int],
                           detections: List[Detection3D], timestamp: float):
        """Create new tracks from unmatched detections."""
        for detection_idx in unmatched_detection_indices:
            detection = detections[detection_idx]

            # Only create tracks for high-confidence detections
            if detection.confidence < 0.5:
                continue

            # Create new track
            track = self._initialize_new_track(detection, timestamp)
            self.tracks[track.track_id] = track
            self.stats['total_tracks_created'] += 1

            self.logger.debug(f"Created new track {track.track_id} for {detection.class_name}")

    def _initialize_new_track(self, detection: Detection3D, timestamp: float) -> Track3D:
        """Initialize a new track from detection."""
        track_id = self.next_track_id
        self.next_track_id += 1

        # Initialize position from detection
        if hasattr(detection, 'center_3d') and detection.center_3d != (0, 0, 0):
            initial_position = torch.tensor(detection.center_3d, device=self.device, dtype=torch.float32)
        else:
            # Fallback from 2D detection
            x, y, x2, y2 = detection.bbox
            center_x = (x + x2) / 2
            center_y = (y + y2) / 2
            estimated_depth = detection.distance if hasattr(detection, 'distance') else 5.0
            initial_position = torch.tensor([center_x * 0.01, center_y * 0.01, estimated_depth],
                                            device=self.device, dtype=torch.float32)

        # Initialize covariance matrix
        initial_covariance = torch.eye(9, device=self.device, dtype=torch.float32)
        initial_covariance[:3, :3] *= self.params.initial_position_std ** 2
        initial_covariance[3:6, 3:6] *= self.params.initial_velocity_std ** 2
        initial_covariance[6:9, 6:9] *= self.params.initial_acceleration_std ** 2

        # Create track
        track = Track3D(track_id=track_id)
        track._device = self.device  # Set device for tensor initialization
        track.position = initial_position
        track.velocity = torch.zeros(3, device=self.device, dtype=torch.float32)
        track.acceleration = torch.zeros(3, device=self.device, dtype=torch.float32)
        track.covariance = initial_covariance

        # Set detection information
        track.last_detection = detection
        track.detection_history = [detection]
        track.class_id = detection.class_id
        track.class_name = detection.class_name
        track.class_confidence = detection.confidence

        # Set timing
        track.creation_time = timestamp
        track.last_update_time = timestamp
        track.last_detection_time = timestamp

        # Initialize quality metrics
        track.hit_count = 1
        track.confidence = detection.confidence

        # Add initial trajectory point
        track.add_trajectory_point(track.position, timestamp)

        return track

    def _manage_track_lifecycle(self):
        """Manage track lifecycle (deletion of old tracks)."""
        tracks_to_delete = []

        for track_id, track in self.tracks.items():
            # Delete tracks that are too old or have low confidence
            if (track.age > self.params.max_age or
                    track.time_since_update > 10 or
                    (track.state == TrackState.LOST and track.time_since_update > 5)):
                tracks_to_delete.append(track_id)
                track.state = TrackState.DELETED

        # Remove deleted tracks
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
            self.logger.debug(f"Deleted track {track_id}")

    def predict_tracks(self, prediction_time: float) -> List[Track3D]:
        """
        Predict track positions at future time.

        Args:
            prediction_time: Time to predict to

        Returns:
            List of tracks with predicted positions
        """
        current_time = time.time()
        dt = prediction_time - current_time

        if dt > self.params.max_prediction_time:
            self.logger.warning(f"Prediction time {dt:.2f}s exceeds maximum {self.params.max_prediction_time}s")
            dt = self.params.max_prediction_time

        predicted_tracks = []

        for track in self.tracks.values():
            if not track.is_active:
                continue

            # Create copy for prediction
            predicted_track = Track3D(track_id=track.track_id)
            predicted_track.position = track.get_predicted_position(dt)
            predicted_track.velocity = track.velocity.clone()
            predicted_track.acceleration = track.acceleration.clone()
            predicted_track.confidence = track.confidence * max(0, 1 - dt * 0.1)  # Decay confidence
            predicted_track.class_id = track.class_id
            predicted_track.class_name = track.class_name
            predicted_track.state = track.state

            predicted_tracks.append(predicted_track)

        self.stats['total_predictions'] += len(predicted_tracks)
        return predicted_tracks

    def get_track_trajectories(self, track_ids: Optional[List[int]] = None) -> Dict[
        int, List[Tuple[torch.Tensor, float]]]:
        """
        Get trajectory history for specified tracks.

        Args:
            track_ids: List of track IDs (all tracks if None)

        Returns:
            Dictionary mapping track_id to list of (position, timestamp) tuples
        """
        if track_ids is None:
            track_ids = list(self.tracks.keys())

        trajectories = {}

        for track_id in track_ids:
            if track_id in self.tracks:
                track = self.tracks[track_id]
                trajectories[track_id] = list(track.trajectory_points)

        return trajectories

    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics."""
        active_tracks = [t for t in self.tracks.values() if t.is_active]
        confirmed_tracks = [t for t in active_tracks if t.is_confirmed]

        stats = {
            'frame_count': self.frame_count,
            'total_tracks': len(self.tracks),
            'active_tracks': len(active_tracks),
            'confirmed_tracks': len(confirmed_tracks),
            'tentative_tracks': len(active_tracks) - len(confirmed_tracks),
            **self.stats
        }

        # Add performance metrics
        if self.performance_monitor:
            perf_stats = self.performance_monitor.get_statistics('tracking_update_time')
            if perf_stats:
                stats.update({
                    'avg_processing_time_ms': perf_stats['mean'] * 1000,
                    'max_processing_time_ms': perf_stats['max'] * 1000,
                    'tracking_fps': 1.0 / perf_stats['mean'] if perf_stats['mean'] > 0 else 0
                })

        # Add GPU statistics if available
        if self.kernels:
            kernel_stats = self.kernels.get_kernel_stats()
            stats['gpu_memory_usage_mb'] = kernel_stats['memory_usage_mb']

        return stats

    def reset_tracking(self):
        """Reset tracker state."""
        with self.processing_lock:
            self.tracks.clear()
            self.next_track_id = 0
            self.frame_count = 0
            self.stats = {
                'total_tracks_created': 0,
                'total_associations': 0,
                'total_predictions': 0,
                'avg_tracking_time': 0.0,
                'gpu_utilization': 0.0
            }

            if self.performance_monitor:
                self.performance_monitor = PerformanceMonitor(window_size=100)

            self.logger.info("Tracking state reset")

    def export_tracks(self, output_path: str, format_type: str = 'json'):
        """
        Export track data to file.

        Args:
            output_path: Output file path
            format_type: Export format ('json' or 'csv')
        """
        track_data = {
            'metadata': {
                'frame_count': self.frame_count,
                'total_tracks': len(self.tracks),
                'export_timestamp': time.time(),
                'tracking_parameters': {
                    'max_age': self.params.max_age,
                    'min_hits': self.params.min_hits,
                    'distance_threshold': self.params.distance_threshold
                }
            },
            'tracks': [track.to_dict() for track in self.tracks.values()]
        }

        if format_type == 'json':
            with open(output_path, 'w') as f:
                json.dump(track_data, f, indent=2)
        elif format_type == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header
                writer.writerow(['track_id', 'state', 'x', 'y', 'z', 'vx', 'vy', 'vz',
                                 'confidence', 'hit_count', 'age', 'class_name'])

                # Data
                for track in self.tracks.values():
                    writer.writerow([
                        track.track_id, track.state.value,
                        track.position[0].item(), track.position[1].item(), track.position[2].item(),
                        track.velocity[0].item(), track.velocity[1].item(), track.velocity[2].item(),
                        track.confidence, track.hit_count, track.age, track.class_name
                    ])

        self.logger.info(f"Tracks exported to {output_path}")

    def cleanup(self):
        """Clean up tracker resources."""
        with self.processing_lock:
            self.tracks.clear()

            if self.kernels:
                self.kernels.cleanup()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info("Object3DTracker cleaned up")


# Utility functions for integration
def create_3d_tracker(config: Dict[str, Any],
                      memory_manager: Optional[CUDAMemoryManager] = None) -> Object3DTracker:
    """Create and initialize a 3D tracker."""
    return Object3DTracker(config, memory_manager)


def convert_detection_to_3d(detection: Detection, depth_value: float,
                            camera_intrinsics: Any) -> Detection3D:
    """Convert 2D detection to 3D detection."""
    # This would integrate with the depth processor
    # Simplified implementation for compatibility
    x1, y1, x2, y2 = detection.bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Rough 3D conversion (should use proper camera intrinsics)
    z = depth_value
    x = (center_x - 320) * z / 500  # Rough conversion
    y = (center_y - 240) * z / 500

    detection_3d = Detection3D(
        bbox=detection.bbox,
        confidence=detection.confidence,
        class_id=detection.class_id,
        class_name=detection.class_name,
        detection_id=detection.detection_id,
        center_3d=(x, y, z),
        distance=z,
        depth_confidence=0.8,
        metadata=detection.metadata
    )

    return detection_3d