"""
Real-time CUDA-accelerated visualization system for RealSense detection pipeline.
GPU-optimized display manager with multi-stream rendering and interactive controls.
"""

import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .cuda_visualization_kernels import CUDAVisualizationKernels, VisualizationProfiler
from .gpu_memory_manager import CUDAMemoryManager
from ..detection.base_detector import DetectionResult
from ..utils.logger import get_logger, PerformanceMonitor


class DisplayMode(Enum):
    """Display modes for visualization."""
    RGB_ONLY = "rgb_only"
    DEPTH_ONLY = "depth_only"
    SIDE_BY_SIDE = "side_by_side"
    PICTURE_IN_PICTURE = "picture_in_picture"
    OVERLAY = "overlay"
    TRIPLE_VIEW = "triple_view"  # RGB, Depth, Detection overlay


class WindowLayout(Enum):
    """Window layout configurations."""
    SINGLE_WINDOW = "single_window"
    MULTI_WINDOW = "multi_window"
    FULLSCREEN = "fullscreen"
    CUSTOM_GRID = "custom_grid"


@dataclass
class VisualizationSettings:
    """Visualization configuration settings."""
    # Display settings
    display_mode: DisplayMode = DisplayMode.SIDE_BY_SIDE
    window_layout: WindowLayout = WindowLayout.SINGLE_WINDOW
    target_fps: int = 30
    vsync_enabled: bool = True

    # Detection visualization
    show_bounding_boxes: bool = True
    show_confidence_scores: bool = True
    show_class_labels: bool = True
    show_track_ids: bool = True
    show_3d_positions: bool = True
    show_trajectories: bool = False

    # Depth visualization
    depth_colormap: str = 'jet'
    depth_min: float = 0.1
    depth_max: float = 10.0
    show_depth_legend: bool = True

    # Performance overlay
    show_performance_metrics: bool = True
    show_gpu_metrics: bool = True
    show_frame_info: bool = False

    # Interactive features
    enable_mouse_interaction: bool = True
    enable_keyboard_shortcuts: bool = True

    # Recording settings
    enable_recording: bool = False
    recording_format: str = 'mp4'
    recording_quality: int = 80

    # Advanced settings
    use_gpu_acceleration: bool = True
    enable_anti_aliasing: bool = True
    alpha_blending: float = 0.7


@dataclass
class WindowConfig:
    """Configuration for individual display windows."""
    window_name: str
    position: Tuple[int, int] = (0, 0)
    size: Tuple[int, int] = (640, 480)
    resizable: bool = True
    always_on_top: bool = False
    fullscreen: bool = False
    show_controls: bool = True


class CUDAVisualizer:
    """GPU-accelerated display manager with real-time rendering capabilities."""

    def __init__(self, config: Dict[str, Any], memory_manager: Optional[CUDAMemoryManager] = None):
        """
        Initialize CUDA-accelerated visualizer.

        Args:
            config: Configuration dictionary
            memory_manager: Optional GPU memory manager
        """
        self.config = config
        self.memory_manager = memory_manager
        self.logger = get_logger("CUDAVisualizer")

        # GPU setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gpu = torch.cuda.is_available() and config.get('visualization', {}).get('use_gpu', True)

        # Visualization settings
        vis_config = config.get('integration', {}).get('visualization', {})
        self.settings = self._load_settings(vis_config)

        # Initialize CUDA kernels
        if self.use_gpu:
            max_width = vis_config.get('max_width', 1920)
            max_height = vis_config.get('max_height', 1080)
            self.kernels = CUDAVisualizationKernels(self.device, max_width, max_height)
            self.profiler = VisualizationProfiler(self.kernels) if config.get('debug', {}).get('profile_viz',
                                                                                               False) else None
        else:
            self.kernels = None
            self.profiler = None
            self.logger.warning("GPU acceleration disabled, using CPU visualization")

        # CUDA streams for different rendering stages
        self.cuda_streams = {}
        if self.use_gpu:
            self.cuda_streams = {
                'rgb_processing': torch.cuda.Stream(),
                'depth_processing': torch.cuda.Stream(),
                'overlay_rendering': torch.cuda.Stream(),
                'compositing': torch.cuda.Stream()
            }

        # Window management
        self.windows: Dict[str, WindowConfig] = {}
        self.active_windows: Dict[str, bool] = {}
        self._initialize_windows()

        # Frame buffers and rendering state
        self.frame_buffers = {}
        self.last_frames = {}
        self.rendering_stats = {}

        # Interactive controls
        self.mouse_callbacks: Dict[str, Callable] = {}
        self.keyboard_callbacks: Dict[str, Callable] = {}
        self.interactive_state = {
            'mouse_position': (0, 0),
            'selected_detection': None,
            'zoom_factor': 1.0,
            'pan_offset': (0, 0)
        }

        # Recording system
        self.video_writers = {}
        self.recording_active = False
        self.recording_start_time = 0

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor(window_size=60)
        self.fps_counter = 0
        self.last_fps_time = time.time()

        # Threading for real-time display
        self.display_thread = None
        self.display_queue = queue.Queue(maxsize=10)
        self.stop_display = threading.Event()
        self.display_lock = threading.RLock()

        self.logger.info(f"CUDAVisualizer initialized on {self.device}")
        self.logger.info(f"GPU acceleration: {self.use_gpu}, Display mode: {self.settings.display_mode.value}")

    def _load_settings(self, vis_config: Dict[str, Any]) -> VisualizationSettings:
        """Load visualization settings from configuration."""
        settings = VisualizationSettings()

        # Display settings
        settings.display_mode = DisplayMode(vis_config.get('display_mode', 'side_by_side'))
        settings.window_layout = WindowLayout(vis_config.get('window_layout', 'single_window'))
        settings.target_fps = vis_config.get('target_fps', 30)

        # Detection visualization
        settings.show_bounding_boxes = vis_config.get('show_bboxes', True)
        settings.show_confidence_scores = vis_config.get('show_confidence', True)
        settings.show_class_labels = vis_config.get('show_labels', True)
        settings.show_track_ids = vis_config.get('show_track_ids', True)
        settings.show_3d_positions = vis_config.get('show_3d_positions', True)

        # Depth visualization
        settings.depth_colormap = vis_config.get('depth_colormap', 'jet')
        settings.depth_min = vis_config.get('depth_min', 0.1)
        settings.depth_max = vis_config.get('depth_max', 10.0)

        # Performance overlay
        settings.show_performance_metrics = vis_config.get('show_performance', True)
        settings.show_gpu_metrics = vis_config.get('show_gpu_metrics', True)

        # Interactive features
        settings.enable_mouse_interaction = vis_config.get('mouse_interaction', True)
        settings.enable_keyboard_shortcuts = vis_config.get('keyboard_shortcuts', True)

        return settings

    def _initialize_windows(self):
        """Initialize display windows based on layout configuration."""
        if self.settings.window_layout == WindowLayout.SINGLE_WINDOW:
            self.windows['main'] = WindowConfig(
                window_name='RealSense Detection Pipeline',
                size=(1280, 720),
                resizable=True
            )

        elif self.settings.window_layout == WindowLayout.MULTI_WINDOW:
            self.windows['rgb'] = WindowConfig(
                window_name='RGB Camera',
                position=(0, 0),
                size=(640, 480)
            )
            self.windows['depth'] = WindowConfig(
                window_name='Depth Camera',
                position=(650, 0),
                size=(640, 480)
            )
            self.windows['detections'] = WindowConfig(
                window_name='Detections',
                position=(0, 490),
                size=(1290, 480)
            )

        elif self.settings.window_layout == WindowLayout.FULLSCREEN:
            self.windows['fullscreen'] = WindowConfig(
                window_name='RealSense Pipeline - Fullscreen',
                fullscreen=True,
                size=(1920, 1080)
            )

        # Initialize OpenCV windows
        for window_name, window_config in self.windows.items():
            cv2.namedWindow(window_config.window_name,
                            cv2.WINDOW_NORMAL if window_config.resizable else cv2.WINDOW_AUTOSIZE)

            if not window_config.fullscreen:
                cv2.resizeWindow(window_config.window_name, *window_config.size)
                cv2.moveWindow(window_config.window_name, *window_config.position)
            else:
                cv2.setWindowProperty(window_config.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # Setup mouse callbacks
            if self.settings.enable_mouse_interaction:
                cv2.setMouseCallback(window_config.window_name, self._mouse_callback, window_name)

            self.active_windows[window_name] = True

        self.logger.info(f"Initialized {len(self.windows)} display windows")

    def start_display_thread(self):
        """Start the display thread for real-time rendering."""
        if self.display_thread and self.display_thread.is_alive():
            return

        self.stop_display.clear()
        self.display_thread = threading.Thread(target=self._display_worker, daemon=True)
        self.display_thread.start()
        self.logger.info("Display thread started")

    def stop_display_thread(self):
        """Stop the display thread."""
        if self.display_thread and self.display_thread.is_alive():
            self.stop_display.set()
            self.display_thread.join(timeout=2.0)
        self.logger.info("Display thread stopped")

    def _display_worker(self):
        """Worker thread for real-time display rendering."""
        self.logger.info("Display worker thread started")

        while not self.stop_display.is_set():
            try:
                # Get frame data from queue
                try:
                    frame_data = self.display_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Render frame
                start_time = time.time()
                self._render_frame_internal(frame_data)
                render_time = time.time() - start_time

                # Update performance metrics
                self.performance_monitor.add_metric('display_render_time', render_time)

                # FPS limiting
                target_frame_time = 1.0 / self.settings.target_fps
                if render_time < target_frame_time:
                    time.sleep(target_frame_time - render_time)

            except Exception as e:
                self.logger.error(f"Display worker error: {e}")

        self.logger.info("Display worker thread stopped")

    def render_frame(self, rgb_frame: Optional[np.ndarray] = None,
                     depth_frame: Optional[np.ndarray] = None,
                     detection_result: Optional[DetectionResult] = None,
                     tracks: Optional[List] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Queue frame for rendering in display thread.

        Args:
            rgb_frame: RGB camera frame
            depth_frame: Depth camera frame
            detection_result: Detection results
            tracks: Active tracks
            metadata: Additional metadata
        """
        frame_data = {
            'rgb_frame': rgb_frame,
            'depth_frame': depth_frame,
            'detection_result': detection_result,
            'tracks': tracks,
            'metadata': metadata or {},
            'timestamp': time.time()
        }

        try:
            self.display_queue.put_nowait(frame_data)
        except queue.Full:
            # Drop frame if queue is full
            try:
                self.display_queue.get_nowait()  # Remove oldest frame
                self.display_queue.put_nowait(frame_data)
            except queue.Empty:
                pass

    def _render_frame_internal(self, frame_data: Dict[str, Any]):
        """Internal frame rendering implementation."""
        with self.display_lock:
            rgb_frame = frame_data.get('rgb_frame')
            depth_frame = frame_data.get('depth_frame')
            detection_result = frame_data.get('detection_result')
            tracks = frame_data.get('tracks', [])
            metadata = frame_data.get('metadata', {})

            # Convert frames to GPU tensors if using GPU acceleration
            if self.use_gpu and self.kernels:
                rendered_frames = self._render_gpu(rgb_frame, depth_frame, detection_result, tracks, metadata)
            else:
                rendered_frames = self._render_cpu(rgb_frame, depth_frame, detection_result, tracks, metadata)

            # Display frames in appropriate windows
            self._display_frames(rendered_frames)

            # Update FPS counter
            self._update_fps_counter()

            # Handle recording
            if self.recording_active:
                self._record_frames(rendered_frames)

    def _render_gpu(self, rgb_frame: Optional[np.ndarray],
                    depth_frame: Optional[np.ndarray],
                    detection_result: Optional[DetectionResult],
                    tracks: List[Any],
                    metadata: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """GPU-accelerated frame rendering."""
        rendered_frames = {}

        # Convert input frames to GPU tensors
        rgb_tensor = None
        depth_tensor = None

        if rgb_frame is not None:
            rgb_tensor = torch.from_numpy(rgb_frame).to(self.device)
            if rgb_tensor.shape[2] == 3 and rgb_frame.dtype == np.uint8:
                # Convert RGB to BGR for OpenCV display
                rgb_tensor = torch.flip(rgb_tensor, [2])

        if depth_frame is not None:
            depth_tensor = torch.from_numpy(depth_frame).to(self.device, dtype=torch.float32)

        # Render based on display mode
        if self.settings.display_mode == DisplayMode.RGB_ONLY:
            if rgb_tensor is not None:
                rendered_frames['main'] = self._render_rgb_with_overlays_gpu(
                    rgb_tensor, detection_result, tracks, metadata
                )

        elif self.settings.display_mode == DisplayMode.DEPTH_ONLY:
            if depth_tensor is not None:
                rendered_frames['main'] = self._render_depth_visualization_gpu(depth_tensor)

        elif self.settings.display_mode == DisplayMode.SIDE_BY_SIDE:
            frames_to_combine = []

            if rgb_tensor is not None:
                rgb_with_overlays = self._render_rgb_with_overlays_gpu(
                    rgb_tensor, detection_result, tracks, metadata
                )
                frames_to_combine.append(rgb_with_overlays)

            if depth_tensor is not None:
                depth_colorized = self._render_depth_visualization_gpu(depth_tensor)
                frames_to_combine.append(depth_colorized)

            if len(frames_to_combine) >= 2:
                rendered_frames['main'] = self._combine_frames_side_by_side_gpu(frames_to_combine)
            elif len(frames_to_combine) == 1:
                rendered_frames['main'] = frames_to_combine[0]

        elif self.settings.display_mode == DisplayMode.PICTURE_IN_PICTURE:
            if rgb_tensor is not None and depth_tensor is not None:
                main_frame = self._render_rgb_with_overlays_gpu(
                    rgb_tensor, detection_result, tracks, metadata
                )
                depth_frame_small = self._render_depth_visualization_gpu(depth_tensor)
                rendered_frames['main'] = self._create_picture_in_picture_gpu(main_frame, depth_frame_small)

        elif self.settings.display_mode == DisplayMode.TRIPLE_VIEW:
            if rgb_tensor is not None:
                rgb_clean = rgb_tensor.clone()
                rgb_with_detections = self._render_rgb_with_overlays_gpu(
                    rgb_tensor, detection_result, tracks, metadata
                )

                frames_to_combine = [rgb_clean, rgb_with_detections]

                if depth_tensor is not None:
                    depth_colorized = self._render_depth_visualization_gpu(depth_tensor)
                    frames_to_combine.append(depth_colorized)

                rendered_frames['main'] = self._combine_frames_grid_gpu(frames_to_combine)

        # Multi-window layout
        if self.settings.window_layout == WindowLayout.MULTI_WINDOW:
            if rgb_tensor is not None:
                rendered_frames['rgb'] = self._render_rgb_with_overlays_gpu(
                    rgb_tensor, detection_result, tracks, metadata
                )

            if depth_tensor is not None:
                rendered_frames['depth'] = self._render_depth_visualization_gpu(depth_tensor)

            # Combined detection view
            if rgb_tensor is not None and depth_tensor is not None:
                detection_view = self._create_detection_overview_gpu(
                    rgb_tensor, depth_tensor, detection_result, tracks
                )
                rendered_frames['detections'] = detection_view

        return rendered_frames

    def _render_rgb_with_overlays_gpu(self, rgb_tensor: torch.Tensor,
                                      detection_result: Optional[DetectionResult],
                                      tracks: List[Any],
                                      metadata: Dict[str, Any]) -> torch.Tensor:
        """Render RGB frame with detection overlays using GPU."""
        output = rgb_tensor.clone()

        # Draw detection bounding boxes
        if (self.settings.show_bounding_boxes and detection_result and
                detection_result.detections):

            boxes = []
            colors = []
            labels = []

            for detection in detection_result.detections:
                boxes.append(list(detection.bbox))

                # Get color based on class
                color_idx = detection.class_id % self.kernels.detection_colors.shape[0]
                colors.append(self.kernels.detection_colors[color_idx])

                # Prepare label
                label_parts = []
                if self.settings.show_class_labels:
                    label_parts.append(detection.class_name)
                if self.settings.show_confidence_scores:
                    label_parts.append(f"{detection.confidence:.2f}")
                if self.settings.show_track_ids and detection.detection_id is not None:
                    label_parts.append(f"ID:{detection.detection_id}")

                labels.append(" | ".join(label_parts))

            if boxes:
                boxes_tensor = torch.tensor(boxes, device=self.device, dtype=torch.float32)
                colors_tensor = torch.stack(colors)

                # Draw bounding boxes
                output = self.kernels.draw_bounding_boxes_gpu(output, boxes_tensor, colors_tensor)

                # Draw labels
                for i, (box, label) in enumerate(zip(boxes, labels)):
                    x1, y1, x2, y2 = map(int, box)
                    text_pos = (x1, max(0, y1 - 10))

                    output = self.kernels.render_text_gpu(
                        output, label, text_pos,
                        color=(255, 255, 255),
                        background_color=(0, 0, 0),
                        font_scale=0.6
                    )

        # Draw 3D position indicators
        if self.settings.show_3d_positions and detection_result:
            for detection in detection_result.detections:
                if hasattr(detection, 'center_3d') and detection.center_3d != (0, 0, 0):
                    x1, y1, x2, y2 = detection.bbox
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Draw 3D position text
                    pos_text = f"({detection.center_3d[0]:.1f}, {detection.center_3d[1]:.1f}, {detection.center_3d[2]:.1f})"
                    output = self.kernels.render_text_gpu(
                        output, pos_text, (center_x - 50, center_y + 20),
                        color=(0, 255, 255),
                        font_scale=0.5
                    )

                    # Draw center point
                    centers = torch.tensor([[center_x, center_y]], device=self.device, dtype=torch.float32)
                    radii = torch.tensor([3], device=self.device, dtype=torch.float32)
                    point_colors = torch.tensor([[0, 255, 255]], device=self.device, dtype=torch.uint8)

                    output = self.kernels.draw_circles_gpu(output, centers, radii, point_colors, filled=True)

        # Draw trajectory trails
        if self.settings.show_trajectories and tracks:
            for track in tracks:
                if hasattr(track, 'trajectory_points') and len(track.trajectory_points) > 1:
                    # Draw trajectory line (simplified)
                    points = [point[0] for point in track.trajectory_points[-10:]]  # Last 10 points

                    for i in range(len(points) - 1):
                        # Convert 3D points to 2D screen coordinates (simplified projection)
                        # This would need proper camera intrinsics in a real implementation
                        pass

        # Add performance overlay
        if self.settings.show_performance_metrics:
            perf_metrics = self._get_performance_metrics(metadata)
            output = self.kernels.create_performance_overlay(output, perf_metrics, (10, 10))

        return output

    def _render_depth_visualization_gpu(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        """Render depth frame with colorization using GPU."""
        # Colorize depth frame
        colorized_depth = self.kernels.colorize_depth_frame(
            depth_tensor,
            min_depth=self.settings.depth_min,
            max_depth=self.settings.depth_max,
            colormap_type=self.settings.depth_colormap
        )

        # Add depth legend
        if self.settings.show_depth_legend:
            colorized_depth = self.kernels.create_depth_colormap_legend(
                colorized_depth,
                self.settings.depth_min,
                self.settings.depth_max,
                position=(colorized_depth.shape[1] - 80, 50)
            )

        return colorized_depth

    def _combine_frames_side_by_side_gpu(self, frames: List[torch.Tensor]) -> torch.Tensor:
        """Combine multiple frames side by side using GPU."""
        if len(frames) < 2:
            return frames[0] if frames else torch.zeros((480, 640, 3), device=self.device, dtype=torch.uint8)

        # Ensure all frames have the same height
        target_height = min(frame.shape[0] for frame in frames)

        resized_frames = []
        for frame in frames:
            if frame.shape[0] != target_height:
                # Simple resize (would use proper interpolation in full implementation)
                scale_factor = target_height / frame.shape[0]
                new_width = int(frame.shape[1] * scale_factor)
                resized_frame = F.interpolate(
                    frame.permute(2, 0, 1).unsqueeze(0).float(),
                    size=(target_height, new_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0).byte()
                resized_frames.append(resized_frame)
            else:
                resized_frames.append(frame)

        # Concatenate horizontally
        combined = torch.cat(resized_frames, dim=1)

        return combined

    def _create_picture_in_picture_gpu(self, main_frame: torch.Tensor,
                                       pip_frame: torch.Tensor,
                                       pip_scale: float = 0.25) -> torch.Tensor:
        """Create picture-in-picture view using GPU."""
        output = main_frame.clone()

        # Resize PiP frame
        pip_height = int(main_frame.shape[0] * pip_scale)
        pip_width = int(main_frame.shape[1] * pip_scale)

        pip_resized = F.interpolate(
            pip_frame.permute(2, 0, 1).unsqueeze(0).float(),
            size=(pip_height, pip_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0).byte()

        # Position in top-right corner with margin
        margin = 10
        y_start = margin
        x_start = main_frame.shape[1] - pip_width - margin

        # Draw border
        border_thickness = 2
        border_color = torch.tensor([255, 255, 255], device=self.device, dtype=torch.uint8)

        # Place PiP frame
        y_end = y_start + pip_height
        x_end = x_start + pip_width

        if y_end <= main_frame.shape[0] and x_end <= main_frame.shape[1]:
            # Draw border
            output[y_start - border_thickness:y_start,
            x_start - border_thickness:x_end + border_thickness] = border_color
            output[y_end:y_end + border_thickness, x_start - border_thickness:x_end + border_thickness] = border_color
            output[y_start:y_end, x_start - border_thickness:x_start] = border_color
            output[y_start:y_end, x_end:x_end + border_thickness] = border_color

            # Place PiP content
            output[y_start:y_end, x_start:x_end] = pip_resized

        return output

    def _combine_frames_grid_gpu(self, frames: List[torch.Tensor]) -> torch.Tensor:
        """Combine frames in a grid layout using GPU."""
        if not frames:
            return torch.zeros((480, 640, 3), device=self.device, dtype=torch.uint8)

        num_frames = len(frames)

        if num_frames == 1:
            return frames[0]
        elif num_frames == 2:
            return self._combine_frames_side_by_side_gpu(frames)
        elif num_frames <= 4:
            # 2x2 grid
            rows = 2
            cols = 2
        else:
            # Calculate optimal grid
            rows = int(np.ceil(np.sqrt(num_frames)))
            cols = int(np.ceil(num_frames / rows))

        # Calculate target size for each cell
        max_height = max(frame.shape[0] for frame in frames)
        max_width = max(frame.shape[1] for frame in frames)

        cell_height = max_height // rows
        cell_width = max_width // cols

        # Create output grid
        output_height = cell_height * rows
        output_width = cell_width * cols
        output = torch.zeros((output_height, output_width, 3), device=self.device, dtype=torch.uint8)

        # Place frames in grid
        for i, frame in enumerate(frames):
            if i >= rows * cols:
                break

            row = i // cols
            col = i % cols

            # Resize frame to cell size
            if frame.shape[:2] != (cell_height, cell_width):
                resized_frame = F.interpolate(
                    frame.permute(2, 0, 1).unsqueeze(0).float(),
                    size=(cell_height, cell_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0).byte()
            else:
                resized_frame = frame

            # Place in grid
            y_start = row * cell_height
            y_end = y_start + cell_height
            x_start = col * cell_width
            x_end = x_start + cell_width

            output[y_start:y_end, x_start:x_end] = resized_frame

        return output

    def _create_detection_overview_gpu(self, rgb_tensor: torch.Tensor,
                                       depth_tensor: torch.Tensor,
                                       detection_result: Optional[DetectionResult],
                                       tracks: List[Any]) -> torch.Tensor:
        """Create comprehensive detection overview using GPU."""
        # Create side-by-side view
        rgb_with_overlays = self._render_rgb_with_overlays_gpu(
            rgb_tensor, detection_result, tracks, {}
        )
        depth_colorized = self._render_depth_visualization_gpu(depth_tensor)

        combined = self._combine_frames_side_by_side_gpu([rgb_with_overlays, depth_colorized])

        # Add detection statistics overlay
        if detection_result:
            stats_text = [
                f"Detections: {len(detection_result.detections)}",
                f"Active Tracks: {len(tracks)}",
                f"Model: {detection_result.model_name}",
                f"Inference: {detection_result.inference_time * 1000:.1f}ms"
            ]

            for i, text in enumerate(stats_text):
                combined = self.kernels.render_text_gpu(
                    combined, text, (10, combined.shape[0] - 100 + i * 20),
                    color=(255, 255, 255),
                    background_color=(0, 0, 0),
                    font_scale=0.7
                )

        return combined

    def _render_cpu(self, rgb_frame: Optional[np.ndarray],
                    depth_frame: Optional[np.ndarray],
                    detection_result: Optional[DetectionResult],
                    tracks: List[Any],
                    metadata: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """CPU fallback rendering implementation."""
        rendered_frames = {}

        if rgb_frame is not None:
            output = rgb_frame.copy()

            # Convert RGB to BGR for OpenCV
            if rgb_frame.shape[2] == 3:
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            # Draw detections using OpenCV
            if detection_result and detection_result.detections:
                for detection in detection_result.detections:
                    x1, y1, x2, y2 = map(int, detection.bbox)

                    # Draw bounding box
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

                    # Draw label
                    label = f"{detection.class_name}: {detection.confidence:.2f}"
                    cv2.putText(output, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            rendered_frames['main'] = output

        return rendered_frames

    def _display_frames(self, rendered_frames: Dict[str, Union[torch.Tensor, np.ndarray]]):
        """Display rendered frames in appropriate windows."""
        for window_key, frame in rendered_frames.items():
            if window_key in self.windows and self.active_windows.get(window_key, False):
                window_name = self.windows[window_key].window_name

                # Convert tensor to numpy if needed
                if isinstance(frame, torch.Tensor):
                    display_frame = frame.cpu().numpy()
                else:
                    display_frame = frame

                # Ensure frame is in correct format for OpenCV
                if display_frame.dtype != np.uint8:
                    display_frame = np.clip(display_frame, 0, 255).astype(np.uint8)

                try:
                    cv2.imshow(window_name, display_frame)
                except cv2.error as e:
                    self.logger.warning(f"Display error for window {window_name}: {e}")

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # Key pressed
            self._handle_keyboard_input(key)

    def _get_performance_metrics(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance metrics for display overlay."""
        current_time = time.time()

        metrics = {
            'FPS': self.fps_counter
        }

        # Add GPU metrics if available
        if self.use_gpu and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 ** 2  # MB
            metrics['GPU Memory'] = f"{gpu_memory:.1f}MB"

        # Add detection metrics from metadata
        if 'detection_time' in metadata:
            metrics['Detection'] = f"{metadata['detection_time'] * 1000:.1f}ms"

        if 'tracking_time' in metadata:
            metrics['Tracking'] = f"{metadata['tracking_time'] * 1000:.1f}ms"

        # Add render time
        render_stats = self.performance_monitor.get_statistics('display_render_time')
        if render_stats:
            metrics['Render'] = f"{render_stats['mean'] * 1000:.1f}ms"

        return metrics

    def _update_fps_counter(self):
        """Update FPS counter."""
        current_time = time.time()
        self.fps_counter += 1

        if current_time - self.last_fps_time >= 1.0:
            # Calculate FPS over the last second
            elapsed = current_time - self.last_fps_time
            fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.last_fps_time = current_time

            # Store the calculated FPS
            self.fps_counter = int(fps)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        window_name = param

        self.interactive_state['mouse_position'] = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self._handle_mouse_click(x, y, window_name)
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            self._handle_mouse_drag(x, y, window_name)
        elif event == cv2.EVENT_MOUSEWHEEL:
            self._handle_mouse_wheel(flags, window_name)

    def _handle_mouse_click(self, x: int, y: int, window_name: str):
        """Handle mouse click events."""
        self.logger.debug(f"Mouse click at ({x}, {y}) in window {window_name}")

        # Find clicked detection (simplified)
        # In full implementation, would check if click is within any bounding box
        pass

    def _handle_mouse_drag(self, x: int, y: int, window_name: str):
        """Handle mouse drag events for panning."""
        # Update pan offset
        old_x, old_y = self.interactive_state['mouse_position']
        dx = x - old_x
        dy = y - old_y

        pan_x, pan_y = self.interactive_state['pan_offset']
        self.interactive_state['pan_offset'] = (pan_x + dx, pan_y + dy)

    def _handle_mouse_wheel(self, flags: int, window_name: str):
        """Handle mouse wheel events for zooming."""
        zoom_delta = 0.1
        if flags > 0:  # Scroll up
            self.interactive_state['zoom_factor'] *= (1 + zoom_delta)
        else:  # Scroll down
            self.interactive_state['zoom_factor'] *= (1 - zoom_delta)

        # Clamp zoom factor
        self.interactive_state['zoom_factor'] = np.clip(
            self.interactive_state['zoom_factor'], 0.1, 5.0
        )

    def _handle_keyboard_input(self, key: int):
        """Handle keyboard input."""
        if not self.settings.enable_keyboard_shortcuts:
            return

        if key == ord('q') or key == 27:  # 'q' or ESC
            self._close_all_windows()
        elif key == ord('r'):  # Toggle recording
            self.toggle_recording()
        elif key == ord('s'):  # Save screenshot
            self._save_screenshot()
        elif key == ord('f'):  # Toggle fullscreen
            self._toggle_fullscreen()
        elif key == ord('d'):  # Cycle display mode
            self._cycle_display_mode()
        elif key == ord('p'):  # Toggle performance overlay
            self.settings.show_performance_metrics = not self.settings.show_performance_metrics
        elif key == ord('h'):  # Show help
            self._show_help()

    def _cycle_display_mode(self):
        """Cycle through display modes."""
        modes = list(DisplayMode)
        current_index = modes.index(self.settings.display_mode)
        next_index = (current_index + 1) % len(modes)
        self.settings.display_mode = modes[next_index]

        self.logger.info(f"Display mode changed to: {self.settings.display_mode.value}")

    def _save_screenshot(self):
        """Save current frame as screenshot."""
        timestamp = int(time.time())
        filename = f"screenshot_{timestamp}.jpg"

        # Get current frame from last rendered frames
        if hasattr(self, 'last_frames') and 'main' in self.last_frames:
            frame = self.last_frames['main']
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()

            cv2.imwrite(filename, frame)
            self.logger.info(f"Screenshot saved: {filename}")

    def toggle_recording(self):
        """Toggle video recording."""
        if self.recording_active:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        """Start video recording."""
        if self.recording_active:
            return

        timestamp = int(time.time())

        for window_name in self.windows:
            if self.active_windows.get(window_name, False):
                filename = f"recording_{window_name}_{timestamp}.{self.settings.recording_format}"

                # Get window size
                window_config = self.windows[window_name]
                width, height = window_config.size

                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(filename, fourcc, self.settings.target_fps, (width, height))

                if writer.isOpened():
                    self.video_writers[window_name] = writer
                    self.logger.info(f"Started recording {filename}")

        if self.video_writers:
            self.recording_active = True
            self.recording_start_time = time.time()

    def _stop_recording(self):
        """Stop video recording."""
        for window_name, writer in self.video_writers.items():
            writer.release()

        self.video_writers.clear()
        self.recording_active = False

        duration = time.time() - self.recording_start_time
        self.logger.info(f"Recording stopped. Duration: {duration:.1f}s")

    def _record_frames(self, rendered_frames: Dict[str, Union[torch.Tensor, np.ndarray]]):
        """Record frames to video files."""
        for window_name, frame in rendered_frames.items():
            if window_name in self.video_writers:
                # Convert tensor to numpy if needed
                if isinstance(frame, torch.Tensor):
                    record_frame = frame.cpu().numpy()
                else:
                    record_frame = frame

                # Ensure correct format
                if record_frame.dtype != np.uint8:
                    record_frame = np.clip(record_frame, 0, 255).astype(np.uint8)

                self.video_writers[window_name].write(record_frame)

    def _close_all_windows(self):
        """Close all OpenCV windows."""
        cv2.destroyAllWindows()
        for window_name in self.active_windows:
            self.active_windows[window_name] = False

        # Stop recording if active
        if self.recording_active:
            self._stop_recording()

    def _show_help(self):
        """Display help information."""
        help_text = """
        Keyboard Shortcuts:
        q/ESC - Quit
        r - Toggle recording
        s - Save screenshot
        f - Toggle fullscreen
        d - Cycle display mode
        p - Toggle performance overlay
        h - Show this help
        """
        print(help_text)

    def get_visualization_stats(self) -> Dict[str, Any]:
        """Get comprehensive visualization statistics."""
        stats = {
            'display_mode': self.settings.display_mode.value,
            'window_layout': self.settings.window_layout.value,
            'active_windows': len([w for w in self.active_windows.values() if w]),
            'target_fps': self.settings.target_fps,
            'current_fps': self.fps_counter,
            'recording_active': self.recording_active,
            'gpu_acceleration': self.use_gpu
        }

        # Add performance metrics
        if self.performance_monitor:
            render_stats = self.performance_monitor.get_statistics('display_render_time')
            if render_stats:
                stats.update({
                    'avg_render_time_ms': render_stats['mean'] * 1000,
                    'max_render_time_ms': render_stats['max'] * 1000,
                    'render_fps': 1.0 / render_stats['mean'] if render_stats['mean'] > 0 else 0
                })

        # Add GPU stats if available
        if self.kernels:
            kernel_stats = self.kernels.get_kernel_stats()
            stats.update({
                'gpu_memory_usage_mb': kernel_stats['memory_usage_mb'],
                'gpu_device': kernel_stats['device']
            })

        return stats

    def cleanup(self):
        """Clean up visualizer resources."""
        self.logger.info("Cleaning up CUDA visualizer...")

        # Stop display thread
        self.stop_display_thread()

        # Stop recording
        if self.recording_active:
            self._stop_recording()

        # Close windows
        self._close_all_windows()

        # Clean up GPU resources
        if self.kernels:
            self.kernels.cleanup()

        # Clear CUDA streams
        for stream in self.cuda_streams.values():
            if hasattr(stream, 'synchronize'):
                stream.synchronize()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("CUDA visualizer cleanup completed")

    def __enter__(self):
        """Context manager entry."""
        self.start_display_thread()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Utility functions for visualization
def create_cuda_visualizer(config: Dict[str, Any],
                           memory_manager: Optional[CUDAMemoryManager] = None) -> CUDAVisualizer:
    """Create and initialize CUDA visualizer."""
    return CUDAVisualizer(config, memory_manager)


def render_detection_frame(visualizer: CUDAVisualizer,
                           rgb_frame: np.ndarray,
                           depth_frame: Optional[np.ndarray],
                           detection_result: DetectionResult) -> None:
    """Convenience function to render a detection frame."""
    visualizer.render_frame(rgb_frame, depth_frame, detection_result)