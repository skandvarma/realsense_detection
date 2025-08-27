"""
Depth data processing and 3D coordinate transformation for RealSense camera.
"""

import numpy as np
import cv2
import pyrealsense2 as rs
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
import scipy.stats as stats
from scipy import ndimage

from ..utils.logger import get_logger


@dataclass
class Point3D:
    """3D point representation."""
    x: float
    y: float
    z: float
    confidence: float = 1.0


@dataclass
class BoundingBox3D:
    """3D bounding box representation."""
    center: Point3D
    min_point: Point3D
    max_point: Point3D
    dimensions: Tuple[float, float, float]  # width, height, depth
    confidence: float = 1.0


class DepthProcessor:
    """Depth data processing and 3D coordinate transformation system."""

    def __init__(self, camera_manager, config: Dict[str, Any]):
        """
        Initialize depth processor.

        Args:
            camera_manager: RealSenseManager instance
            config: Configuration dictionary
        """
        self.camera_manager = camera_manager
        self.config = config
        self.depth_config = config['camera'].get('depth', {})
        self.logger = get_logger("DepthProcessor")

        # Depth processing parameters
        self.min_distance = self.depth_config.get('min_distance', 0.1)  # meters
        self.max_distance = self.depth_config.get('max_distance', 3.0)  # meters
        self.depth_scale = None
        self.depth_intrinsics = None

        # Filtering parameters
        self.median_kernel_size = 5
        self.confidence_threshold = 0.7
        self.outlier_threshold = 2.0  # standard deviations

        # Multi-point sampling configuration
        self.sampling_grid_size = 3  # 3x3 grid for robust depth estimation
        self.sampling_radius = 5     # pixels

        self.logger.info("DepthProcessor initialized")

    def update_camera_parameters(self):
        """Update camera parameters from camera manager."""
        if self.camera_manager.depth_intrinsics:
            self.depth_intrinsics = self.camera_manager.depth_intrinsics
            self.depth_scale = self.camera_manager.depth_scale
            self.logger.debug("Camera parameters updated")
        else:
            self.logger.warning("No depth intrinsics available from camera manager")

    def filter_depth_frame(self, depth_frame: np.ndarray,
                          apply_median: bool = True,
                          apply_distance_filter: bool = True,
                          apply_outlier_removal: bool = True) -> np.ndarray:
        """
        Apply comprehensive filtering to depth frame.

        Args:
            depth_frame: Raw depth frame from camera
            apply_median: Apply median filtering for noise reduction
            apply_distance_filter: Apply min/max distance filtering
            apply_outlier_removal: Remove statistical outliers

        Returns:
            Filtered depth frame
        """
        if depth_frame is None or depth_frame.size == 0:
            return depth_frame

        filtered_depth = depth_frame.copy()

        # Convert to meters if using depth scale
        if self.depth_scale:
            depth_meters = filtered_depth.astype(np.float32) * self.depth_scale
        else:
            depth_meters = filtered_depth.astype(np.float32) / 1000.0  # Assume mm to m conversion

        # Distance-based filtering
        if apply_distance_filter:
            invalid_mask = (depth_meters < self.min_distance) | (depth_meters > self.max_distance)
            depth_meters[invalid_mask] = 0

        # Median filtering for noise reduction
        if apply_median:
            # Only apply to valid (non-zero) regions
            valid_mask = depth_meters > 0
            if np.any(valid_mask):
                filtered_region = cv2.medianBlur(depth_meters.astype(np.uint16), self.median_kernel_size)
                depth_meters[valid_mask] = filtered_region[valid_mask].astype(np.float32) / 1000.0

        # Statistical outlier removal
        if apply_outlier_removal:
            depth_meters = self._remove_outliers(depth_meters)

        # Convert back to original format
        if self.depth_scale:
            filtered_depth = (depth_meters / self.depth_scale).astype(depth_frame.dtype)
        else:
            filtered_depth = (depth_meters * 1000.0).astype(depth_frame.dtype)

        return filtered_depth

    def _remove_outliers(self, depth_data: np.ndarray) -> np.ndarray:
        """Remove statistical outliers from depth data."""
        valid_mask = depth_data > 0

        if not np.any(valid_mask):
            return depth_data

        valid_depths = depth_data[valid_mask]

        # Calculate statistics
        mean_depth = np.mean(valid_depths)
        std_depth = np.std(valid_depths)

        # Mark outliers
        outlier_threshold = self.outlier_threshold * std_depth
        outlier_mask = np.abs(depth_data - mean_depth) > outlier_threshold

        # Remove outliers
        depth_data[outlier_mask & valid_mask] = 0

        return depth_data

    def pixel_to_3d(self, pixel_x: int, pixel_y: int, depth_frame: np.ndarray,
                   use_multi_point: bool = True) -> Optional[Point3D]:
        """
        Convert 2D pixel coordinates to 3D world coordinates.

        Args:
            pixel_x: X coordinate in image
            pixel_y: Y coordinate in image
            depth_frame: Depth frame data
            use_multi_point: Use multi-point sampling for robust estimation

        Returns:
            Point3D object or None if conversion failed
        """
        if self.depth_intrinsics is None:
            self.logger.warning("No depth intrinsics available for 3D conversion")
            return None

        if depth_frame is None or depth_frame.size == 0:
            return None

        h, w = depth_frame.shape

        # Bounds checking
        if not (0 <= pixel_x < w and 0 <= pixel_y < h):
            return None

        if use_multi_point:
            depth_value, confidence = self._get_robust_depth(pixel_x, pixel_y, depth_frame)
        else:
            depth_value = depth_frame[pixel_y, pixel_x]
            confidence = 1.0 if depth_value > 0 else 0.0

        if depth_value <= 0:
            return None

        # Convert depth to meters
        if self.depth_scale:
            depth_meters = depth_value * self.depth_scale
        else:
            depth_meters = depth_value / 1000.0

        # Apply distance constraints
        if not (self.min_distance <= depth_meters <= self.max_distance):
            return None

        # Convert to 3D coordinates using intrinsic parameters
        x_3d = (pixel_x - self.depth_intrinsics.ppx) * depth_meters / self.depth_intrinsics.fx
        y_3d = (pixel_y - self.depth_intrinsics.ppy) * depth_meters / self.depth_intrinsics.fy
        z_3d = depth_meters

        return Point3D(x=x_3d, y=y_3d, z=z_3d, confidence=confidence)

    def _get_robust_depth(self, center_x: int, center_y: int,
                         depth_frame: np.ndarray) -> Tuple[float, float]:
        """
        Get robust depth estimate using multi-point sampling.

        Args:
            center_x: Center X coordinate
            center_y: Center Y coordinate
            depth_frame: Depth frame data

        Returns:
            Tuple of (depth_value, confidence)
        """
        h, w = depth_frame.shape

        # Define sampling grid
        radius = self.sampling_radius
        grid_points = []

        for dy in range(-radius, radius + 1, radius):
            for dx in range(-radius, radius + 1, radius):
                x = center_x + dx
                y = center_y + dy

                if 0 <= x < w and 0 <= y < h:
                    depth_val = depth_frame[y, x]
                    if depth_val > 0:
                        grid_points.append(depth_val)

        if not grid_points:
            return 0.0, 0.0

        # Statistical analysis of depth values
        grid_points = np.array(grid_points)

        if len(grid_points) == 1:
            return float(grid_points[0]), 1.0

        # Remove outliers
        mean_depth = np.mean(grid_points)
        std_depth = np.std(grid_points)

        if std_depth > 0:
            valid_mask = np.abs(grid_points - mean_depth) <= (2 * std_depth)
            valid_points = grid_points[valid_mask]
        else:
            valid_points = grid_points

        if len(valid_points) == 0:
            return float(grid_points[0]), 0.5

        # Calculate robust depth estimate
        robust_depth = np.median(valid_points)

        # Calculate confidence based on consistency
        if len(valid_points) > 1:
            consistency = 1.0 - (np.std(valid_points) / np.mean(valid_points))
            confidence = np.clip(consistency, 0.0, 1.0)
        else:
            confidence = 0.7

        return float(robust_depth), confidence

    def detection_to_3d(self, detection_bbox: Tuple[int, int, int, int],
                       depth_frame: np.ndarray,
                       sampling_strategy: str = 'center_robust') -> Optional[BoundingBox3D]:
        """
        Convert 2D detection bounding box to 3D bounding box.

        Args:
            detection_bbox: (x1, y1, x2, y2) bounding box coordinates
            depth_frame: Depth frame data
            sampling_strategy: Strategy for depth sampling
                - 'center': Use center point only
                - 'center_robust': Use robust center sampling
                - 'grid': Sample multiple points across bbox
                - 'edges': Sample bbox edges

        Returns:
            BoundingBox3D object or None if conversion failed
        """
        if depth_frame is None or depth_frame.size == 0:
            return None

        # Convert float coordinates to integers
        x1, y1, x2, y2 = map(int, detection_bbox)

        # Ensure valid bounding box
        if x2 <= x1 or y2 <= y1:
            return None

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if sampling_strategy == 'center':
            center_3d = self.pixel_to_3d(center_x, center_y, depth_frame, use_multi_point=False)
            if center_3d is None:
                return None

            # Estimate dimensions based on depth and bbox size
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            # Rough estimation of real-world dimensions
            depth_meters = center_3d.z
            pixel_size_at_depth = depth_meters / self.depth_intrinsics.fx  # meters per pixel

            width_3d = bbox_width * pixel_size_at_depth
            height_3d = bbox_height * pixel_size_at_depth
            depth_3d = max(width_3d, height_3d) * 0.5  # Rough depth estimation

            return BoundingBox3D(
                center=center_3d,
                min_point=Point3D(center_3d.x - width_3d/2, center_3d.y - height_3d/2, center_3d.z - depth_3d/2),
                max_point=Point3D(center_3d.x + width_3d/2, center_3d.y + height_3d/2, center_3d.z + depth_3d/2),
                dimensions=(width_3d, height_3d, depth_3d),
                confidence=center_3d.confidence
            )

        elif sampling_strategy == 'center_robust':
            center_3d = self.pixel_to_3d(center_x, center_y, depth_frame, use_multi_point=True)
            if center_3d is None:
                return None

            return self._estimate_3d_bbox_from_center(center_3d, detection_bbox)

        elif sampling_strategy == 'grid':
            return self._grid_sampling_3d_bbox(detection_bbox, depth_frame)

        elif sampling_strategy == 'edges':
            return self._edge_sampling_3d_bbox(detection_bbox, depth_frame)

        else:
            self.logger.warning(f"Unknown sampling strategy: {sampling_strategy}")
            return None

    def _estimate_3d_bbox_from_center(self, center_3d: Point3D,
                                    bbox_2d: Tuple[int, int, int, int]) -> BoundingBox3D:
        """Estimate 3D bounding box from center point and 2D bbox."""
        x1, y1, x2, y2 = bbox_2d

        bbox_width = x2 - x1
        bbox_height = y2 - y1
        depth_meters = center_3d.z

        if self.depth_intrinsics:
            # More accurate calculation using intrinsics
            pixel_size_x = depth_meters / self.depth_intrinsics.fx
            pixel_size_y = depth_meters / self.depth_intrinsics.fy

            width_3d = bbox_width * pixel_size_x
            height_3d = bbox_height * pixel_size_y
        else:
            # Fallback estimation
            pixel_size = depth_meters / 500.0  # Rough approximation
            width_3d = bbox_width * pixel_size
            height_3d = bbox_height * pixel_size

        # Estimate depth based on object size (heuristic)
        depth_3d = min(width_3d, height_3d) * 0.8

        min_point = Point3D(
            center_3d.x - width_3d/2,
            center_3d.y - height_3d/2,
            center_3d.z - depth_3d/2
        )

        max_point = Point3D(
            center_3d.x + width_3d/2,
            center_3d.y + height_3d/2,
            center_3d.z + depth_3d/2
        )

        return BoundingBox3D(
            center=center_3d,
            min_point=min_point,
            max_point=max_point,
            dimensions=(width_3d, height_3d, depth_3d),
            confidence=center_3d.confidence
        )

    def _grid_sampling_3d_bbox(self, bbox_2d: Tuple[int, int, int, int],
                              depth_frame: np.ndarray) -> Optional[BoundingBox3D]:
        """Sample multiple points across bounding box for 3D estimation."""
        # Convert float coordinates to integers
        x1, y1, x2, y2 = map(int, bbox_2d)

        # Sample points in a grid pattern
        sample_points = []
        grid_size = 3

        for i in range(grid_size):
            for j in range(grid_size):
                x = x1 + (x2 - x1) * i // (grid_size - 1)
                y = y1 + (y2 - y1) * j // (grid_size - 1)

                point_3d = self.pixel_to_3d(int(x), int(y), depth_frame)
                if point_3d and point_3d.confidence > 0.5:
                    sample_points.append(point_3d)

        if not sample_points:
            return None

        # Calculate statistics
        x_coords = [p.x for p in sample_points]
        y_coords = [p.y for p in sample_points]
        z_coords = [p.z for p in sample_points]
        confidences = [p.confidence for p in sample_points]

        # Calculate center and bounds
        center = Point3D(
            x=np.mean(x_coords),
            y=np.mean(y_coords),
            z=np.mean(z_coords),
            confidence=np.mean(confidences)
        )

        min_point = Point3D(np.min(x_coords), np.min(y_coords), np.min(z_coords))
        max_point = Point3D(np.max(x_coords), np.max(y_coords), np.max(z_coords))

        dimensions = (
            max_point.x - min_point.x,
            max_point.y - min_point.y,
            max_point.z - min_point.z
        )

        return BoundingBox3D(
            center=center,
            min_point=min_point,
            max_point=max_point,
            dimensions=dimensions,
            confidence=center.confidence
        )

    def _edge_sampling_3d_bbox(self, bbox_2d: Tuple[int, int, int, int],
                              depth_frame: np.ndarray) -> Optional[BoundingBox3D]:
        """Sample points along bounding box edges for 3D estimation."""
        # Convert float coordinates to integers
        x1, y1, x2, y2 = map(int, bbox_2d)

        # Sample points along edges
        edge_points = []
        samples_per_edge = 5

        # Top and bottom edges
        for i in range(samples_per_edge):
            x = x1 + (x2 - x1) * i // (samples_per_edge - 1)

            # Top edge
            point_3d = self.pixel_to_3d(int(x), y1, depth_frame)
            if point_3d and point_3d.confidence > 0.5:
                edge_points.append(point_3d)

            # Bottom edge
            point_3d = self.pixel_to_3d(int(x), y2, depth_frame)
            if point_3d and point_3d.confidence > 0.5:
                edge_points.append(point_3d)

        # Left and right edges
        for i in range(samples_per_edge):
            y = y1 + (y2 - y1) * i // (samples_per_edge - 1)

            # Left edge
            point_3d = self.pixel_to_3d(x1, int(y), depth_frame)
            if point_3d and point_3d.confidence > 0.5:
                edge_points.append(point_3d)

            # Right edge
            point_3d = self.pixel_to_3d(x2, int(y), depth_frame)
            if point_3d and point_3d.confidence > 0.5:
                edge_points.append(point_3d)

        if not edge_points:
            return None

        # Similar processing as grid sampling
        x_coords = [p.x for p in edge_points]
        y_coords = [p.y for p in edge_points]
        z_coords = [p.z for p in edge_points]
        confidences = [p.confidence for p in edge_points]

        center = Point3D(
            x=np.median(x_coords),
            y=np.median(y_coords),
            z=np.median(z_coords),
            confidence=np.mean(confidences)
        )

        min_point = Point3D(np.min(x_coords), np.min(y_coords), np.min(z_coords))
        max_point = Point3D(np.max(x_coords), np.max(y_coords), np.max(z_coords))

        dimensions = (
            max_point.x - min_point.x,
            max_point.y - min_point.y,
            max_point.z - min_point.z
        )

        return BoundingBox3D(
            center=center,
            min_point=min_point,
            max_point=max_point,
            dimensions=dimensions,
            confidence=center.confidence
        )

    def calculate_distance(self, point1: Point3D, point2: Point3D) -> float:
        """Calculate Euclidean distance between two 3D points."""
        return np.sqrt(
            (point1.x - point2.x)**2 +
            (point1.y - point2.y)**2 +
            (point1.z - point2.z)**2
        )

    def get_depth_statistics(self, depth_frame: np.ndarray,
                           region: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, float]:
        """
        Calculate statistical measures for depth data.

        Args:
            depth_frame: Depth frame data
            region: Optional (x1, y1, x2, y2) region of interest

        Returns:
            Dictionary containing depth statistics
        """
        if depth_frame is None or depth_frame.size == 0:
            return {}

        if region:
            x1, y1, x2, y2 = region
            depth_roi = depth_frame[y1:y2, x1:x2]
        else:
            depth_roi = depth_frame

        # Convert to meters
        if self.depth_scale:
            depth_meters = depth_roi.astype(np.float32) * self.depth_scale
        else:
            depth_meters = depth_roi.astype(np.float32) / 1000.0

        # Filter valid depths
        valid_depths = depth_meters[depth_meters > 0]

        if len(valid_depths) == 0:
            return {'valid_pixel_count': 0}

        statistics = {
            'valid_pixel_count': len(valid_depths),
            'invalid_pixel_count': np.sum(depth_meters == 0),
            'mean_depth': float(np.mean(valid_depths)),
            'median_depth': float(np.median(valid_depths)),
            'std_depth': float(np.std(valid_depths)),
            'min_depth': float(np.min(valid_depths)),
            'max_depth': float(np.max(valid_depths)),
            'depth_range': float(np.max(valid_depths) - np.min(valid_depths))
        }

        # Calculate percentiles
        percentiles = [10, 25, 75, 90]
        for p in percentiles:
            statistics[f'depth_p{p}'] = float(np.percentile(valid_depths, p))

        return statistics

    def create_depth_colormap(self, depth_frame: np.ndarray,
                            colormap: int = cv2.COLORMAP_JET,
                            normalize: bool = True) -> np.ndarray:
        """
        Create a colorized visualization of depth data.

        Args:
            depth_frame: Depth frame data
            colormap: OpenCV colormap to use
            normalize: Whether to normalize depth values

        Returns:
            Colorized depth image
        """
        if depth_frame is None or depth_frame.size == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Convert to 8-bit for visualization
        if normalize:
            # Normalize based on valid depth range
            valid_mask = depth_frame > 0
            if np.any(valid_mask):
                min_val = np.min(depth_frame[valid_mask])
                max_val = np.max(depth_frame[valid_mask])

                if max_val > min_val:
                    normalized = ((depth_frame - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    normalized[~valid_mask] = 0
                else:
                    normalized = depth_frame.astype(np.uint8)
            else:
                normalized = np.zeros_like(depth_frame, dtype=np.uint8)
        else:
            normalized = cv2.convertScaleAbs(depth_frame, alpha=0.03)

        # Apply colormap
        colorized = cv2.applyColorMap(normalized, colormap)

        # Make invalid regions black
        invalid_mask = depth_frame == 0
        colorized[invalid_mask] = [0, 0, 0]

        return colorized

    def validate_depth_confidence(self, depth_value: float,
                                region_stats: Dict[str, float]) -> float:
        """
        Assess confidence of a depth measurement based on regional statistics.

        Args:
            depth_value: Depth value to validate
            region_stats: Statistical measures from surrounding region

        Returns:
            Confidence score between 0 and 1
        """
        if depth_value <= 0:
            return 0.0

        confidence = 1.0

        # Check if depth is within reasonable range
        if not (self.min_distance <= depth_value <= self.max_distance):
            confidence *= 0.1

        # Check consistency with regional statistics
        if 'mean_depth' in region_stats and 'std_depth' in region_stats:
            mean_depth = region_stats['mean_depth']
            std_depth = region_stats['std_depth']

            if std_depth > 0:
                z_score = abs(depth_value - mean_depth) / std_depth
                if z_score > 2.0:
                    confidence *= 0.5
                elif z_score > 1.0:
                    confidence *= 0.8

        # Check regional validity
        if 'valid_pixel_count' in region_stats:
            total_pixels = region_stats.get('valid_pixel_count', 0) + region_stats.get('invalid_pixel_count', 0)
            if total_pixels > 0:
                validity_ratio = region_stats['valid_pixel_count'] / total_pixels
                confidence *= validity_ratio

        return np.clip(confidence, 0.0, 1.0)