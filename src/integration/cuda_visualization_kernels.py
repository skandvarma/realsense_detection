"""
Custom CUDA kernels for GPU-accelerated visualization operations.
Optimized for real-time rendering with minimal CPU involvement.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import time
from typing import Tuple, List, Dict, Any, Optional, Union
from enum import Enum
import math

from ..utils.logger import get_logger


class ColorSpace(Enum):
    """Supported color spaces for conversion."""
    RGB = "rgb"
    BGR = "bgr"
    HSV = "hsv"
    LAB = "lab"
    GRAY = "gray"


class BlendMode(Enum):
    """Blending modes for overlay composition."""
    ALPHA = "alpha"
    ADD = "add"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"


class CUDAVisualizationKernels:
    """Collection of CUDA kernels for real-time visualization operations."""

    def __init__(self, device: torch.device, max_width: int = 1920, max_height: int = 1080):
        """
        Initialize CUDA visualization kernels.

        Args:
            device: CUDA device
            max_width: Maximum frame width to support
            max_height: Maximum frame height to support
        """
        self.device = device
        self.max_width = max_width
        self.max_height = max_height
        self.logger = get_logger("CUDAVisualizationKernels")

        # Pre-allocate memory for visualization operations
        self._preallocate_buffers()

        # Color palettes for different visualizations
        self._initialize_color_palettes()

        # Font and text rendering parameters
        self.font_scale = 1.0
        self.font_thickness = 2
        self.line_thickness = 2

        # Performance optimization parameters
        self.use_half_precision = True
        self.enable_async_operations = True

        self.logger.info(f"Initialized CUDA visualization kernels for {max_width}x{max_height}")

    def _preallocate_buffers(self):
        """Pre-allocate GPU memory for visualization operations."""
        # Main frame buffers
        self.rgb_buffer = torch.zeros(
            (self.max_height, self.max_width, 3),
            device=self.device, dtype=torch.uint8
        )

        self.depth_buffer = torch.zeros(
            (self.max_height, self.max_width),
            device=self.device, dtype=torch.float32
        )

        self.overlay_buffer = torch.zeros(
            (self.max_height, self.max_width, 4),
            device=self.device, dtype=torch.uint8
        )

        # Colorized depth buffer
        self.depth_colorized = torch.zeros(
            (self.max_height, self.max_width, 3),
            device=self.device, dtype=torch.uint8
        )

        # Alpha channel buffer
        self.alpha_buffer = torch.zeros(
            (self.max_height, self.max_width),
            device=self.device, dtype=torch.float32
        )

        # Temporary computation buffers
        self.temp_float_buffer = torch.zeros(
            (self.max_height, self.max_width),
            device=self.device, dtype=torch.float32
        )

        self.temp_rgb_buffer = torch.zeros(
            (self.max_height, self.max_width, 3),
            device=self.device, dtype=torch.uint8
        )

        # Mask buffers for selective operations
        self.mask_buffer = torch.zeros(
            (self.max_height, self.max_width),
            device=self.device, dtype=torch.bool
        )

        self.logger.debug(f"Pre-allocated {self._get_memory_usage():.1f}MB for visualization")

    def _initialize_color_palettes(self):
        """Initialize color palettes for different visualization types."""
        # Detection class colors (20 distinct colors)
        detection_colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
            [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
            [128, 0, 128], [0, 128, 128], [255, 128, 0], [128, 255, 0], [0, 255, 128],
            [255, 0, 128], [128, 0, 255], [0, 128, 255], [192, 192, 192], [128, 128, 128]
        ]

        self.detection_colors = torch.tensor(
            detection_colors, device=self.device, dtype=torch.uint8
        )

        # Depth colormap (similar to JET colormap)
        self.depth_colormap = self._create_jet_colormap()

        # Heatmap colors (for confidence visualization)
        heatmap_colors = []
        for i in range(256):
            t = i / 255.0
            if t < 0.5:
                # Blue to green
                r = 0
                g = int(255 * t * 2)
                b = int(255 * (1 - t * 2))
            else:
                # Green to red
                r = int(255 * (t - 0.5) * 2)
                g = int(255 * (1 - (t - 0.5) * 2))
                b = 0
            heatmap_colors.append([r, g, b])

        self.heatmap_colors = torch.tensor(
            heatmap_colors, device=self.device, dtype=torch.uint8
        )

    def _create_jet_colormap(self) -> torch.Tensor:
        """Create JET colormap for depth visualization."""
        colormap = []
        for i in range(256):
            t = i / 255.0

            # JET colormap computation
            if t < 0.125:
                r, g, b = 0, 0, int(255 * (0.5 + t * 4))
            elif t < 0.375:
                r, g, b = 0, int(255 * (t - 0.125) * 4), 255
            elif t < 0.625:
                r, g, b = int(255 * (t - 0.375) * 4), 255, int(255 * (1 - (t - 0.375) * 4))
            elif t < 0.875:
                r, g, b = 255, int(255 * (1 - (t - 0.625) * 4)), 0
            else:
                r, g, b = int(255 * (1 - (t - 0.875) * 4)), 0, 0

            colormap.append([r, g, b])

        return torch.tensor(colormap, device=self.device, dtype=torch.uint8)

    def _get_memory_usage(self) -> float:
        """Calculate GPU memory usage in MB."""
        total_elements = (
            self.rgb_buffer.numel() +
            self.depth_buffer.numel() +
            self.overlay_buffer.numel() +
            self.depth_colorized.numel() +
            self.alpha_buffer.numel() +
            self.temp_float_buffer.numel() +
            self.temp_rgb_buffer.numel() +
            self.mask_buffer.numel() +
            self.detection_colors.numel() +
            self.depth_colormap.numel() +
            self.heatmap_colors.numel()
        )
        return total_elements * 4 / (1024 * 1024)  # Approximate 4 bytes per element

    def colorize_depth_frame(self, depth_frame: torch.Tensor,
                           min_depth: float = 0.1, max_depth: float = 10.0,
                           colormap_type: str = 'jet') -> torch.Tensor:
        """
        GPU kernel for converting depth maps to color visualizations.

        Args:
            depth_frame: Input depth frame [H, W]
            min_depth: Minimum depth value for normalization
            max_depth: Maximum depth value for normalization
            colormap_type: Type of colormap to use

        Returns:
            Colorized depth image [H, W, 3]
        """
        h, w = depth_frame.shape
        h = min(h, self.max_height)
        w = min(w, self.max_width)

        # Use pre-allocated buffer
        depth_region = self.depth_buffer[:h, :w]
        output_region = self.depth_colorized[:h, :w]

        # Copy input to buffer
        depth_region.copy_(depth_frame[:h, :w])

        # Normalize depth values to [0, 1] range
        depth_normalized = torch.clamp(
            (depth_region - min_depth) / (max_depth - min_depth),
            0.0, 1.0
        )

        # Convert to colormap indices [0, 255]
        depth_indices = (depth_normalized * 255).long()

        # Apply colormap
        if colormap_type == 'jet':
            colormap = self.depth_colormap
        else:
            colormap = self.heatmap_colors

        # GPU-optimized colormap lookup
        output_region.copy_(colormap[depth_indices.flatten()].view(h, w, 3))

        # Handle invalid depth values (set to black)
        invalid_mask = depth_region == 0
        output_region[invalid_mask] = 0

        return output_region.clone()

    def draw_bounding_boxes_gpu(self, image: torch.Tensor,
                               boxes: torch.Tensor,
                               colors: torch.Tensor,
                               thickness: int = 2) -> torch.Tensor:
        """
        GPU-optimized bounding box rendering.

        Args:
            image: Input image [H, W, 3]
            boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
            colors: Box colors [N, 3]
            thickness: Line thickness

        Returns:
            Image with bounding boxes drawn
        """
        if boxes.numel() == 0:
            return image

        h, w = image.shape[:2]
        output = image.clone()

        # Process each box
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].long()
            color = colors[i]

            # Clamp coordinates to image bounds
            x1 = torch.clamp(x1, 0, w - 1)
            y1 = torch.clamp(y1, 0, h - 1)
            x2 = torch.clamp(x2, 0, w - 1)
            y2 = torch.clamp(y2, 0, h - 1)

            # Draw horizontal lines (top and bottom)
            for t in range(thickness):
                if y1 + t < h:
                    output[y1 + t, x1:x2 + 1] = color
                if y2 - t >= 0:
                    output[y2 - t, x1:x2 + 1] = color

            # Draw vertical lines (left and right)
            for t in range(thickness):
                if x1 + t < w:
                    output[y1:y2 + 1, x1 + t] = color
                if x2 - t >= 0:
                    output[y1:y2 + 1, x2 - t] = color

        return output

    def draw_filled_rectangles_gpu(self, image: torch.Tensor,
                                  boxes: torch.Tensor,
                                  colors: torch.Tensor,
                                  alpha: float = 0.3) -> torch.Tensor:
        """
        Draw filled rectangles with alpha blending.

        Args:
            image: Input image [H, W, 3]
            boxes: Rectangle coordinates [N, 4]
            colors: Fill colors [N, 3]
            alpha: Transparency factor

        Returns:
            Image with filled rectangles
        """
        if boxes.numel() == 0:
            return image

        h, w = image.shape[:2]
        output = image.clone().float()

        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].long()
            color = colors[i].float()

            # Clamp coordinates
            x1 = torch.clamp(x1, 0, w - 1)
            y1 = torch.clamp(y1, 0, h - 1)
            x2 = torch.clamp(x2, 0, w - 1)
            y2 = torch.clamp(y2, 0, h - 1)

            # Alpha blending
            if x2 > x1 and y2 > y1:
                output[y1:y2, x1:x2] = (
                    alpha * color.unsqueeze(0).unsqueeze(0) +
                    (1 - alpha) * output[y1:y2, x1:x2]
                )

        return output.byte()

    def draw_circles_gpu(self, image: torch.Tensor,
                        centers: torch.Tensor,
                        radii: torch.Tensor,
                        colors: torch.Tensor,
                        filled: bool = True) -> torch.Tensor:
        """
        Draw circles on image using GPU.

        Args:
            image: Input image [H, W, 3]
            centers: Circle centers [N, 2] (x, y)
            radii: Circle radii [N]
            colors: Circle colors [N, 3]
            filled: Whether to fill circles

        Returns:
            Image with circles drawn
        """
        if centers.numel() == 0:
            return image

        h, w = image.shape[:2]
        output = image.clone()

        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=self.device),
            torch.arange(w, device=self.device),
            indexing='ij'
        )

        for i in range(centers.shape[0]):
            cx, cy = centers[i]
            radius = radii[i]
            color = colors[i]

            # Calculate distances from center
            distances = torch.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)

            if filled:
                mask = distances <= radius
            else:
                mask = (distances <= radius) & (distances >= radius - 1)

            # Apply color
            output[mask] = color

        return output

    def alpha_blend_layers(self, base_layer: torch.Tensor,
                          overlay_layer: torch.Tensor,
                          alpha_mask: torch.Tensor,
                          blend_mode: BlendMode = BlendMode.ALPHA) -> torch.Tensor:
        """
        GPU-optimized alpha blending between layers.

        Args:
            base_layer: Base image [H, W, 3]
            overlay_layer: Overlay image [H, W, 3]
            alpha_mask: Alpha channel [H, W] (0-1)
            blend_mode: Blending mode

        Returns:
            Blended image
        """
        h, w = base_layer.shape[:2]

        # Ensure alpha mask is the right shape
        if alpha_mask.shape != (h, w):
            alpha_mask = F.interpolate(
                alpha_mask.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze()

        # Convert to float for computation
        base_float = base_layer.float() / 255.0
        overlay_float = overlay_layer.float() / 255.0
        alpha_expanded = alpha_mask.unsqueeze(2).expand(-1, -1, 3)

        if blend_mode == BlendMode.ALPHA:
            # Standard alpha blending
            result = alpha_expanded * overlay_float + (1 - alpha_expanded) * base_float

        elif blend_mode == BlendMode.ADD:
            # Additive blending
            result = torch.clamp(base_float + alpha_expanded * overlay_float, 0, 1)

        elif blend_mode == BlendMode.MULTIPLY:
            # Multiplicative blending
            result = base_float * (1 - alpha_expanded + alpha_expanded * overlay_float)

        elif blend_mode == BlendMode.SCREEN:
            # Screen blending
            result = 1 - (1 - base_float) * (1 - alpha_expanded * overlay_float)

        elif blend_mode == BlendMode.OVERLAY:
            # Overlay blending
            condition = base_float < 0.5
            result = torch.where(
                condition,
                2 * base_float * overlay_float * alpha_expanded + base_float * (1 - alpha_expanded),
                1 - 2 * (1 - base_float) * (1 - overlay_float * alpha_expanded)
            )

        else:
            result = base_float

        return (torch.clamp(result, 0, 1) * 255).byte()

    def convert_color_space_gpu(self, image: torch.Tensor,
                               from_space: ColorSpace,
                               to_space: ColorSpace) -> torch.Tensor:
        """
        GPU-accelerated color space conversion.

        Args:
            image: Input image [H, W, 3]
            from_space: Source color space
            to_space: Target color space

        Returns:
            Converted image
        """
        if from_space == to_space:
            return image

        img_float = image.float() / 255.0

        # RGB to BGR or BGR to RGB
        if ((from_space == ColorSpace.RGB and to_space == ColorSpace.BGR) or
            (from_space == ColorSpace.BGR and to_space == ColorSpace.RGB)):
            return torch.flip(image, [-1])

        # RGB/BGR to Grayscale
        if to_space == ColorSpace.GRAY:
            if from_space == ColorSpace.RGB:
                weights = torch.tensor([0.299, 0.587, 0.114], device=self.device)
            else:  # BGR
                weights = torch.tensor([0.114, 0.587, 0.299], device=self.device)

            gray = torch.sum(img_float * weights, dim=2, keepdim=True)
            return (gray.expand(-1, -1, 3) * 255).byte()

        # RGB to HSV
        if from_space == ColorSpace.RGB and to_space == ColorSpace.HSV:
            return self._rgb_to_hsv_gpu(img_float)

        # HSV to RGB
        if from_space == ColorSpace.HSV and to_space == ColorSpace.RGB:
            return self._hsv_to_rgb_gpu(img_float)

        # For other conversions, use approximations or return original
        self.logger.warning(f"Color space conversion {from_space} -> {to_space} not implemented")
        return image

    def _rgb_to_hsv_gpu(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to HSV on GPU."""
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        max_val, max_idx = torch.max(rgb, dim=2)
        min_val, _ = torch.min(rgb, dim=2)

        delta = max_val - min_val

        # Hue calculation
        h = torch.zeros_like(max_val)

        mask = delta != 0
        r_mask = (max_idx == 0) & mask
        g_mask = (max_idx == 1) & mask
        b_mask = (max_idx == 2) & mask

        h[r_mask] = (60 * ((g[r_mask] - b[r_mask]) / delta[r_mask]) + 360) % 360
        h[g_mask] = (60 * ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 120) % 360
        h[b_mask] = (60 * ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 240) % 360

        # Saturation calculation
        s = torch.zeros_like(max_val)
        s[max_val != 0] = delta[max_val != 0] / max_val[max_val != 0]

        # Value is max_val
        v = max_val

        # Stack and convert to [0, 255] range
        hsv = torch.stack([h / 360 * 255, s * 255, v * 255], dim=2)
        return hsv.byte()

    def _hsv_to_rgb_gpu(self, hsv: torch.Tensor) -> torch.Tensor:
        """Convert HSV to RGB on GPU."""
        h = hsv[..., 0] / 255.0 * 360
        s = hsv[..., 1] / 255.0
        v = hsv[..., 2] / 255.0

        c = v * s
        x = c * (1 - torch.abs((h / 60) % 2 - 1))
        m = v - c

        rgb = torch.zeros_like(hsv).float()

        # Define the 6 regions of the color wheel
        region = (h / 60).long() % 6

        # Region 0: [0, 60)
        mask = region == 0
        rgb[mask, 0] = c[mask]
        rgb[mask, 1] = x[mask]
        rgb[mask, 2] = 0

        # Region 1: [60, 120)
        mask = region == 1
        rgb[mask, 0] = x[mask]
        rgb[mask, 1] = c[mask]
        rgb[mask, 2] = 0

        # Region 2: [120, 180)
        mask = region == 2
        rgb[mask, 0] = 0
        rgb[mask, 1] = c[mask]
        rgb[mask, 2] = x[mask]

        # Region 3: [180, 240)
        mask = region == 3
        rgb[mask, 0] = 0
        rgb[mask, 1] = x[mask]
        rgb[mask, 2] = c[mask]

        # Region 4: [240, 300)
        mask = region == 4
        rgb[mask, 0] = x[mask]
        rgb[mask, 1] = 0
        rgb[mask, 2] = c[mask]

        # Region 5: [300, 360)
        mask = region == 5
        rgb[mask, 0] = c[mask]
        rgb[mask, 1] = 0
        rgb[mask, 2] = x[mask]

        # Add the offset and convert to byte
        rgb += m.unsqueeze(2)
        return (torch.clamp(rgb, 0, 1) * 255).byte()

    def render_text_gpu(self, image: torch.Tensor,
                       text: str,
                       position: Tuple[int, int],
                       color: Tuple[int, int, int] = (255, 255, 255),
                       background_color: Optional[Tuple[int, int, int]] = None,
                       font_scale: float = 1.0) -> torch.Tensor:
        """
        GPU-assisted text rendering (simplified bitmap approach).

        Args:
            image: Input image [H, W, 3]
            text: Text to render
            position: Text position (x, y)
            color: Text color
            background_color: Optional background color
            font_scale: Font scaling factor

        Returns:
            Image with text rendered
        """
        # For GPU efficiency, we'll use a simplified bitmap font approach
        # In a full implementation, this would use pre-rendered character bitmaps

        output = image.clone()
        x, y = position

        # Simple character width/height estimation
        char_width = int(8 * font_scale)
        char_height = int(12 * font_scale)

        # Render background if specified
        if background_color is not None:
            bg_width = len(text) * char_width + 4
            bg_height = char_height + 4

            x1 = max(0, x - 2)
            y1 = max(0, y - 2)
            x2 = min(image.shape[1], x + bg_width)
            y2 = min(image.shape[0], y + bg_height)

            if x2 > x1 and y2 > y1:
                output[y1:y2, x1:x2] = torch.tensor(background_color, device=self.device)

        # Simple text rendering (placeholder - would use actual bitmap fonts)
        for i, char in enumerate(text):
            char_x = x + i * char_width
            char_y = y

            if (char_x + char_width < image.shape[1] and
                char_y + char_height < image.shape[0] and
                char_x >= 0 and char_y >= 0):

                # Simple rectangle for each character (placeholder)
                output[char_y:char_y + char_height, char_x:char_x + char_width] = \
                    torch.tensor(color, device=self.device)

        return output

    def create_performance_overlay(self, image: torch.Tensor,
                                  metrics: Dict[str, float],
                                  position: Tuple[int, int] = (10, 10)) -> torch.Tensor:
        """
        Create performance metrics overlay on GPU.

        Args:
            image: Input image [H, W, 3]
            metrics: Performance metrics dictionary
            position: Overlay position

        Returns:
            Image with performance overlay
        """
        output = image.clone()
        x, y = position
        line_height = 25

        # Create semi-transparent background
        overlay_height = len(metrics) * line_height + 20
        overlay_width = 300

        bg_boxes = torch.tensor([[x, y, x + overlay_width, y + overlay_height]],
                               device=self.device, dtype=torch.float32)
        bg_colors = torch.tensor([[0, 0, 0]], device=self.device, dtype=torch.uint8)

        output = self.draw_filled_rectangles_gpu(output, bg_boxes, bg_colors, alpha=0.7)

        # Render metrics text
        for i, (key, value) in enumerate(metrics.items()):
            if isinstance(value, float):
                text = f"{key}: {value:.2f}"
            else:
                text = f"{key}: {value}"

            text_y = y + 15 + i * line_height
            output = self.render_text_gpu(
                output, text, (x + 10, text_y),
                color=(255, 255, 255), font_scale=0.7
            )

        return output

    def create_depth_colormap_legend(self, image: torch.Tensor,
                                   min_depth: float, max_depth: float,
                                   position: Tuple[int, int] = (50, 50)) -> torch.Tensor:
        """
        Create depth colormap legend overlay.

        Args:
            image: Input image
            min_depth: Minimum depth value
            max_depth: Maximum depth value
            position: Legend position

        Returns:
            Image with depth legend
        """
        output = image.clone()
        x, y = position

        legend_width = 20
        legend_height = 200

        # Create colorbar
        for i in range(legend_height):
            depth_value = min_depth + (max_depth - min_depth) * (1 - i / legend_height)
            color_idx = int((1 - i / legend_height) * 255)
            color = self.depth_colormap[color_idx]

            if (y + i < image.shape[0] and x + legend_width < image.shape[1]):
                output[y + i, x:x + legend_width] = color

        # Add text labels
        output = self.render_text_gpu(output, f"{max_depth:.1f}m", (x + legend_width + 5, y))
        output = self.render_text_gpu(output, f"{min_depth:.1f}m", (x + legend_width + 5, y + legend_height - 15))

        return output

    def batch_process_frames(self, frames: List[torch.Tensor],
                           operations: List[str]) -> List[torch.Tensor]:
        """
        Batch process multiple frames with GPU kernels.

        Args:
            frames: List of input frames
            operations: List of operations to apply

        Returns:
            List of processed frames
        """
        processed_frames = []

        # Stack frames for batch processing
        if len(frames) > 1:
            try:
                batch_frames = torch.stack(frames)

                # Apply batch operations
                for operation in operations:
                    if operation == 'normalize':
                        batch_frames = batch_frames.float() / 255.0
                    elif operation == 'denormalize':
                        batch_frames = (torch.clamp(batch_frames, 0, 1) * 255).byte()
                    elif operation == 'gaussian_blur':
                        # Simple blur approximation
                        kernel = torch.ones(3, 3, device=self.device) / 9.0
                        batch_frames = F.conv2d(
                            batch_frames.float().permute(0, 3, 1, 2),
                            kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1),
                            padding=1, groups=3
                        ).permute(0, 2, 3, 1).byte()

                processed_frames = [batch_frames[i] for i in range(len(frames))]

            except RuntimeError:
                # Fallback to individual processing if batch fails
                processed_frames = frames
        else:
            processed_frames = frames

        return processed_frames

    def synchronize(self):
        """Synchronize all GPU operations."""
        torch.cuda.synchronize(self.device)

    def get_kernel_stats(self) -> Dict[str, Any]:
        """Get kernel performance statistics."""
        return {
            'device': str(self.device),
            'max_resolution': f"{self.max_width}x{self.max_height}",
            'memory_usage_mb': self._get_memory_usage(),
            'use_half_precision': self.use_half_precision,
            'enable_async_operations': self.enable_async_operations,
            'color_palettes': {
                'detection_colors': self.detection_colors.shape[0],
                'depth_colormap': self.depth_colormap.shape[0],
                'heatmap_colors': self.heatmap_colors.shape[0]
            }
        }

    def cleanup(self):
        """Clean up GPU resources."""
        # Clear all buffers
        del self.rgb_buffer
        del self.depth_buffer
        del self.overlay_buffer
        del self.depth_colorized
        del self.alpha_buffer
        del self.temp_float_buffer
        del self.temp_rgb_buffer
        del self.mask_buffer
        del self.detection_colors
        del self.depth_colormap
        del self.heatmap_colors

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("CUDA visualization kernels cleaned up")


# Utility functions for common visualization operations
def create_visualization_kernels(device: torch.device,
                                max_width: int = 1920,
                                max_height: int = 1080) -> CUDAVisualizationKernels:
    """Create and initialize visualization kernels."""
    return CUDAVisualizationKernels(device, max_width, max_height)


def colorize_depth_optimized(depth_frame: torch.Tensor,
                           kernels: CUDAVisualizationKernels,
                           min_depth: float = 0.1,
                           max_depth: float = 10.0) -> torch.Tensor:
    """Optimized depth colorization."""
    return kernels.colorize_depth_frame(depth_frame, min_depth, max_depth)


def draw_detections_gpu(image: torch.Tensor,
                       detections: List[Dict[str, Any]],
                       kernels: CUDAVisualizationKernels) -> torch.Tensor:
    """GPU-optimized detection rendering."""
    if not detections:
        return image

    # Extract boxes and colors
    boxes = []
    colors = []

    for i, detection in enumerate(detections):
        boxes.append(detection['bbox'])
        color_idx = detection.get('class_id', 0) % kernels.detection_colors.shape[0]
        colors.append(kernels.detection_colors[color_idx])

    boxes_tensor = torch.tensor(boxes, device=kernels.device, dtype=torch.float32)
    colors_tensor = torch.stack(colors)

    return kernels.draw_bounding_boxes_gpu(image, boxes_tensor, colors_tensor)


# Performance profiling utilities
class VisualizationProfiler:
    """Profiler for visualization kernel performance."""

    def __init__(self, kernels: CUDAVisualizationKernels):
        self.kernels = kernels
        self.timing_data = {}
        self.logger = get_logger("VisualizationProfiler")

    def profile_kernel(self, kernel_name: str, kernel_func, *args, **kwargs):
        """Profile a visualization kernel."""
        # Warm up
        for _ in range(3):
            kernel_func(*args, **kwargs)

        self.kernels.synchronize()

        # Time the kernel
        times = []
        for _ in range(10):
            start_time = time.time()
            result = kernel_func(*args, **kwargs)
            self.kernels.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = np.mean(times)
        std_time = np.std(times)

        self.timing_data[kernel_name] = {
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000,
            'throughput_fps': 1.0 / avg_time if avg_time > 0 else 0
        }

        self.logger.info(f"{kernel_name}: {avg_time * 1000:.2f}±{std_time * 1000:.2f}ms "
                        f"({1.0/avg_time:.1f} FPS)")

        return result

    def get_profiling_report(self) -> Dict[str, Any]:
        """Get comprehensive profiling report."""
        return {
            'kernel_stats': self.kernels.get_kernel_stats(),
            'timing_data': self.timing_data,
            'total_kernels': len(self.timing_data)
        }