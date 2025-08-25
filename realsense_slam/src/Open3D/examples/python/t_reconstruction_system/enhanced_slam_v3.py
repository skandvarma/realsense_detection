# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/t_reconstruction_system/dense_slam_gui.py

# P.S. This example is used in documentation, so, please ensure the changes are
# synchronized.

import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from config import ConfigParser

import os
import sys
import numpy as np
import threading
import time
import signal
from common import load_rgbd_file_names, save_poses, load_intrinsic, extract_trianglemesh, get_default_dataset, \
    extract_rgbd_frames

# Add project root to Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..'))
sys.path.insert(0, project_root)

# Import D435iCamera from shared camera manager
from src.camera.realsense_manager import D435iCamera


def set_enabled(widget, enable):
    widget.enabled = enable
    for child in widget.get_children():
        child.enabled = enable


class ReconstructionWindow:

    def __init__(self, config, font_id):
        self.config = config

        self.window = gui.Application.instance.create_window(
            'Open3D - Reconstruction', 1280, 800)

        w = self.window
        em = w.theme.font_size

        spacing = int(np.round(0.25 * em))
        vspacing = int(np.round(0.5 * em))

        margins = gui.Margins(vspacing)

        # First panel
        self.panel = gui.Vert(spacing, margins)

        ## Items in fixed props
        self.fixed_prop_grid = gui.VGrid(2, spacing, gui.Margins(em, 0, em, 0))

        ### Depth scale slider
        scale_label = gui.Label('Depth scale')
        self.scale_slider = gui.Slider(gui.Slider.INT)
        self.scale_slider.set_limits(1000, 5000)
        self.scale_slider.int_value = int(config.depth_scale)
        self.fixed_prop_grid.add_child(scale_label)
        self.fixed_prop_grid.add_child(self.scale_slider)

        voxel_size_label = gui.Label('Voxel size')
        self.voxel_size_slider = gui.Slider(gui.Slider.DOUBLE)
        self.voxel_size_slider.set_limits(0.003, 0.01)
        self.voxel_size_slider.double_value = config.voxel_size
        self.fixed_prop_grid.add_child(voxel_size_label)
        self.fixed_prop_grid.add_child(self.voxel_size_slider)

        trunc_multiplier_label = gui.Label('Trunc multiplier')
        self.trunc_multiplier_slider = gui.Slider(gui.Slider.DOUBLE)
        self.trunc_multiplier_slider.set_limits(1.0, 20.0)
        self.trunc_multiplier_slider.double_value = config.trunc_voxel_multiplier
        self.fixed_prop_grid.add_child(trunc_multiplier_label)
        self.fixed_prop_grid.add_child(self.trunc_multiplier_slider)

        est_block_count_label = gui.Label('Est. blocks')
        self.est_block_count_slider = gui.Slider(gui.Slider.INT)
        self.est_block_count_slider.set_limits(40000, 100000)
        self.est_block_count_slider.int_value = config.block_count
        self.fixed_prop_grid.add_child(est_block_count_label)
        self.fixed_prop_grid.add_child(self.est_block_count_slider)

        est_point_count_label = gui.Label('Est. points')
        self.est_point_count_slider = gui.Slider(gui.Slider.INT)
        self.est_point_count_slider.set_limits(500000, 8000000)
        self.est_point_count_slider.int_value = config.est_point_count
        self.fixed_prop_grid.add_child(est_point_count_label)
        self.fixed_prop_grid.add_child(self.est_point_count_slider)

        ## Items in adjustable props
        self.adjustable_prop_grid = gui.VGrid(2, spacing,
                                              gui.Margins(em, 0, em, 0))

        ### Reconstruction interval
        interval_label = gui.Label('Recon. interval')
        self.interval_slider = gui.Slider(gui.Slider.INT)
        self.interval_slider.set_limits(1, 500)
        self.interval_slider.int_value = 50
        self.adjustable_prop_grid.add_child(interval_label)
        self.adjustable_prop_grid.add_child(self.interval_slider)

        ### Depth max slider
        max_label = gui.Label('Depth max')
        self.max_slider = gui.Slider(gui.Slider.DOUBLE)
        self.max_slider.set_limits(3.0, 6.0)
        self.max_slider.double_value = config.depth_max
        self.adjustable_prop_grid.add_child(max_label)
        self.adjustable_prop_grid.add_child(self.max_slider)

        ### Depth diff slider
        diff_label = gui.Label('Depth diff')
        self.diff_slider = gui.Slider(gui.Slider.DOUBLE)
        self.diff_slider.set_limits(0.07, 0.5)
        self.diff_slider.double_value = config.odometry_distance_thr
        self.adjustable_prop_grid.add_child(diff_label)
        self.adjustable_prop_grid.add_child(self.diff_slider)

        ### Update surface?
        update_label = gui.Label('Update surface?')
        self.update_box = gui.Checkbox('')
        self.update_box.checked = True
        self.adjustable_prop_grid.add_child(update_label)
        self.adjustable_prop_grid.add_child(self.update_box)

        ### Ray cast color?
        raycast_label = gui.Label('Raycast color?')
        self.raycast_box = gui.Checkbox('')
        self.raycast_box.checked = True
        self.adjustable_prop_grid.add_child(raycast_label)
        self.adjustable_prop_grid.add_child(self.raycast_box)

        set_enabled(self.fixed_prop_grid, True)

        ## Application control
        b = gui.ToggleSwitch('Resume/Pause')
        b.set_on_clicked(self._on_switch)

        ## Tabs
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()

        ### Input image tab
        tab1 = gui.Vert(0, tab_margins)
        self.input_color_image = gui.ImageWidget()
        self.input_depth_image = gui.ImageWidget()
        tab1.add_child(self.input_color_image)
        tab1.add_fixed(vspacing)
        tab1.add_child(self.input_depth_image)
        tabs.add_tab('Input images', tab1)

        ### Rendered image tab
        tab2 = gui.Vert(0, tab_margins)
        self.raycast_color_image = gui.ImageWidget()
        self.raycast_depth_image = gui.ImageWidget()
        tab2.add_child(self.raycast_color_image)
        tab2.add_fixed(vspacing)
        tab2.add_child(self.raycast_depth_image)
        tabs.add_tab('Raycast images', tab2)

        ### Info tab
        tab3 = gui.Vert(0, tab_margins)
        self.output_info = gui.Label('Output info')
        self.output_info.font_id = font_id
        tab3.add_child(self.output_info)
        tabs.add_tab('Info', tab3)

        ### Occupancy Grid tab
        tab4 = gui.Vert(0, tab_margins)
        self.occupancy_grid_image = gui.ImageWidget()
        self.occupancy_info = gui.Label('Occupancy grid info')
        self.occupancy_info.font_id = font_id
        tab4.add_child(self.occupancy_grid_image)
        tab4.add_fixed(vspacing)
        tab4.add_child(self.occupancy_info)
        tabs.add_tab('Occupancy Grid', tab4)

        self.panel.add_child(gui.Label('Starting settings'))
        self.panel.add_child(self.fixed_prop_grid)
        self.panel.add_fixed(vspacing)
        self.panel.add_child(gui.Label('Reconstruction settings'))
        self.panel.add_child(self.adjustable_prop_grid)
        self.panel.add_child(b)
        self.panel.add_stretch()
        self.panel.add_child(tabs)

        # Scene widget
        self.widget3d = gui.SceneWidget()

        # FPS panel
        self.fps_panel = gui.Vert(spacing, margins)
        self.output_fps = gui.Label('FPS: 0.0')
        self.fps_panel.add_child(self.output_fps)

        # Now add all the complex panels
        w.add_child(self.panel)
        w.add_child(self.widget3d)
        w.add_child(self.fps_panel)

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])

        w.set_on_layout(self._on_layout)
        w.set_on_close(self._on_close)

        self.is_done = False

        self.is_started = False
        self.is_running = False
        self.is_surface_updated = False

        self.idx = 0
        self.poses = []

        # Initialize camera for live streaming
        self.camera = None

        # Occupancy grid parameters
        self.occupancy_grid_resolution = 0.05  # 5cm per pixel
        self.occupancy_grid_size = (400, 400)  # 20m x 20m grid
        self.occupancy_grid_origin = (10.0, 10.0)  # offset from origin
        self.occupancy_height_min = 0.1  # 10cm above ground
        self.occupancy_height_max = 2.0  # 2m height limit
        self.occupancy_update_interval = 30  # update every 30 frames
        self.last_occupancy_update = 0
        self.current_occupancy_grid = None

        # Start running
        threading.Thread(name='UpdateMain', target=self.update_main).start()

    def _on_layout(self, ctx):
        em = ctx.theme.font_size

        panel_width = 20 * em
        rect = self.window.content_rect

        self.panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)

        x = self.panel.frame.get_right()
        self.widget3d.frame = gui.Rect(x, rect.y,
                                       rect.get_right() - x, rect.height)

        fps_panel_width = 7 * em
        fps_panel_height = 2 * em
        self.fps_panel.frame = gui.Rect(rect.get_right() - fps_panel_width,
                                        rect.y, fps_panel_width,
                                        fps_panel_height)

    # Toggle callback: application's main controller
    def _on_switch(self, is_on):
        if not self.is_started:
            gui.Application.instance.post_to_main_thread(
                self.window, self._on_start)
        self.is_running = not self.is_running

    # On start: point cloud buffer and model initialization.
    def _on_start(self):
        max_points = self.est_point_count_slider.int_value

        pcd_placeholder = o3d.t.geometry.PointCloud(
            o3c.Tensor(np.zeros((max_points, 3), dtype=np.float32)))
        pcd_placeholder.point.colors = o3c.Tensor(
            np.zeros((max_points, 3), dtype=np.float32))
        mat = rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.sRGB_color = True
        self.widget3d.scene.scene.add_geometry('points', pcd_placeholder, mat)

        self.model = o3d.t.pipelines.slam.Model(
            self.voxel_size_slider.double_value, 16,
            self.est_block_count_slider.int_value, o3c.Tensor(np.eye(4)),
            o3c.Device(self.config.device))
        self.is_started = True

        set_enabled(self.fixed_prop_grid, False)
        set_enabled(self.adjustable_prop_grid, True)

    def _on_close(self):
        self.is_done = True

        # Stop camera when closing
        if self.camera:
            self.camera.stop()

        if self.is_started:
            print('Saving model to {}...'.format(self.config.path_npz))
            self.model.voxel_grid.save(self.config.path_npz)
            print('Finished.')

            mesh_fname = '.'.join(self.config.path_npz.split('.')[:-1]) + '.ply'
            print('Extracting and saving mesh to {}...'.format(mesh_fname))
            mesh = extract_trianglemesh(self.model.voxel_grid, self.config,
                                        mesh_fname)
            print('Finished.')

            log_fname = '.'.join(self.config.path_npz.split('.')[:-1]) + '.log'
            print('Saving trajectory to {}...'.format(log_fname))
            save_poses(log_fname, self.poses)
            print('Finished.')

            # Save occupancy grid if available
            if self.current_occupancy_grid is not None:
                grid_fname = '.'.join(self.config.path_npz.split('.')[:-1]) + '_occupancy_grid.npy'
                np.save(grid_fname, self.current_occupancy_grid)
                print('Occupancy grid saved to {}...'.format(grid_fname))

        return True

    def world_to_grid(self, world_x, world_y):
        """Convert world coordinates to grid indices."""
        grid_x = int((world_x + self.occupancy_grid_origin[0]) / self.occupancy_grid_resolution)
        grid_y = int((world_y + self.occupancy_grid_origin[1]) / self.occupancy_grid_resolution)
        return grid_x, grid_y

    def is_valid_grid_cell(self, grid_x, grid_y):
        """Check if grid coordinates are within bounds."""
        return (0 <= grid_x < self.occupancy_grid_size[0] and
                0 <= grid_y < self.occupancy_grid_size[1])

    def create_occupancy_grid_from_pointcloud(self, pcd):
        """Create occupancy grid from point cloud """
        if pcd is None or pcd.point.positions.shape[0] == 0:
            return None

        try:
            # Convert tensor point cloud to legacy for VoxelGrid creation
            legacy_pcd = pcd.to_legacy()

            # Filter points by height
            points = np.asarray(legacy_pcd.points)
            height_mask = ((points[:, 2] >= self.occupancy_height_min) &
                           (points[:, 2] <= self.occupancy_height_max))

            if not np.any(height_mask):
                return None

            # Create filtered point cloud
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(points[height_mask])

            # Create voxel grid from filtered point cloud
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
                filtered_pcd,
                voxel_size=self.occupancy_grid_resolution
            )

            # Initialize occupancy grid
            occupancy_grid = np.full(self.occupancy_grid_size, 50, dtype=np.int8)  # 50 = unknown

            # Mark occupied cells
            for voxel in voxel_grid.get_voxels():
                # Get voxel center
                voxel_center = voxel_grid.origin + voxel.grid_index * voxel_grid.voxel_size

                # Convert to grid coordinates
                grid_x, grid_y = self.world_to_grid(voxel_center[0], voxel_center[1])

                if self.is_valid_grid_cell(grid_x, grid_y):
                    occupancy_grid[grid_y, grid_x] = 100  # 100 = occupied

            # Mark free space using ray tracing from recent camera poses
            if len(self.poses) > 0:
                self.mark_free_space_with_rays(occupancy_grid)

            return occupancy_grid

        except Exception as e:
            print(f"Error creating occupancy grid: {e}")
            return None

    def mark_free_space_with_rays(self, occupancy_grid):
        """Mark free space using simple ray tracing from camera poses."""
        # Use last few poses to avoid excessive computation
        recent_poses = self.poses[-5:] if len(self.poses) > 5 else self.poses

        for pose in recent_poses:
            try:
                if pose.shape != (4, 4):
                    continue

                # Get camera position
                cam_pos = pose[:3, 3]
                cam_x, cam_y = cam_pos[0], cam_pos[1]

                # Convert to grid coordinates
                cam_grid_x, cam_grid_y = self.world_to_grid(cam_x, cam_y)

                if not self.is_valid_grid_cell(cam_grid_x, cam_grid_y):
                    continue

                # Simple radial ray tracing
                for angle in np.linspace(0, 2 * np.pi, 16):  # 16 rays around camera
                    dx = np.cos(angle)
                    dy = np.sin(angle)

                    # Trace ray up to 3 meters
                    max_range = int(3.0 / self.occupancy_grid_resolution)
                    for step in range(1, max_range):
                        ray_x = int(cam_grid_x + dx * step)
                        ray_y = int(cam_grid_y + dy * step)

                        if not self.is_valid_grid_cell(ray_x, ray_y):
                            break

                        # If we hit an occupied cell, stop
                        if occupancy_grid[ray_y, ray_x] == 100:
                            break

                        # Mark as free space if currently unknown
                        if occupancy_grid[ray_y, ray_x] == 50:
                            occupancy_grid[ray_y, ray_x] = 0  # 0 = free

            except Exception as e:
                print(f"Error in ray tracing: {e}")
                continue

    def create_occupancy_grid_visualization(self, occupancy_grid):
        """Create RGB visualization of occupancy grid."""
        if occupancy_grid is None:
            return None

        try:
            # Create RGB visualization with explicit contiguous array
            vis_grid = np.zeros((occupancy_grid.shape[0], occupancy_grid.shape[1], 3), dtype=np.uint8, order='C')

            # Color mapping
            vis_grid[occupancy_grid == 0] = [255, 255, 255]  # White for free space
            vis_grid[occupancy_grid == 100] = [0, 0, 0]  # Black for obstacles
            vis_grid[occupancy_grid == 50] = [128, 128, 128]  # Gray for unknown

            # Mark robot position
            if len(self.poses) > 0:
                latest_pose = self.poses[-1]
                if latest_pose.shape == (4, 4):
                    robot_pos = latest_pose[:3, 3]
                    grid_x, grid_y = self.world_to_grid(robot_pos[0], robot_pos[1])

                    if self.is_valid_grid_cell(grid_x, grid_y):
                        # Draw robot as small green square
                        size = 2
                        for dx in range(-size, size + 1):
                            for dy in range(-size, size + 1):
                                rx, ry = grid_x + dx, grid_y + dy
                                if self.is_valid_grid_cell(rx, ry):
                                    vis_grid[ry, rx] = [0, 255, 0]  # Green for robot

            # Flip vertically for correct display and ensure contiguous
            vis_grid = np.flipud(vis_grid)
            vis_grid = np.ascontiguousarray(vis_grid, dtype=np.uint8)

            return vis_grid

        except Exception as e:
            print(f"Error creating occupancy grid visualization: {e}")
            return None

    def _update_occupancy_grid_gui(self, occupancy_grid):
        """Helper method for GUI thread update."""
        self.update_occupancy_grid_display(occupancy_grid)

    def _update_render_gui(self, input_depth, input_color, raycast_depth, raycast_color, pcd, frustum):
        """Helper method for GUI thread render update."""
        self.update_render(input_depth, input_color, raycast_depth, raycast_color, pcd, frustum)

    def update_occupancy_grid_display(self, occupancy_grid):
        """Update occupancy grid visualization in GUI."""
        if occupancy_grid is None:
            return

        try:
            # Create visualization
            vis_grid = self.create_occupancy_grid_visualization(occupancy_grid)

            if vis_grid is not None and vis_grid.size > 0:
                # Ensure array is contiguous for Open3D and has correct dimensions
                if len(vis_grid.shape) == 3 and vis_grid.shape[2] == 3:
                    vis_grid = np.ascontiguousarray(vis_grid, dtype=np.uint8)

                    # Convert to Open3D image
                    o3d_image = o3d.geometry.Image(vis_grid)
                    self.occupancy_grid_image.update_image(o3d_image)
                else:
                    print(f"Invalid visualization grid shape: {vis_grid.shape}")
                    return

                # Update info text
                total_cells = occupancy_grid.size
                free_cells = np.sum(occupancy_grid == 0)
                occupied_cells = np.sum(occupancy_grid == 100)
                unknown_cells = np.sum(occupancy_grid == 50)

                info_text = f'Occupancy Grid ({self.occupancy_grid_size[0]}x{self.occupancy_grid_size[1]})\n\n'
                info_text += f'Resolution: {self.occupancy_grid_resolution:.3f} m/cell\n'
                info_text += f'Coverage: {(total_cells - unknown_cells) / total_cells * 100:.1f}%\n\n'
                info_text += f'Free space: {free_cells} ({free_cells / total_cells * 100:.1f}%)\n'
                info_text += f'Occupied: {occupied_cells} ({occupied_cells / total_cells * 100:.1f}%)\n'
                info_text += f'Unknown: {unknown_cells} ({unknown_cells / total_cells * 100:.1f}%)\n'

                # Robot position info
                if len(self.poses) > 0:
                    latest_pose = self.poses[-1]
                    if latest_pose.shape == (4, 4):
                        robot_pos = latest_pose[:3, 3]
                        grid_x, grid_y = self.world_to_grid(robot_pos[0], robot_pos[1])
                        info_text += f'\nRobot position:\n'
                        info_text += f'World: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})\n'
                        info_text += f'Grid: ({grid_x}, {grid_y})\n'

                self.occupancy_info.text = info_text

        except Exception as e:
            print(f"Error updating occupancy grid display: {e}")

    def init_render(self, depth_ref, color_ref):
        self.input_depth_image.update_image(
            depth_ref.colorize_depth(float(self.scale_slider.int_value),
                                     self.config.depth_min,
                                     self.max_slider.double_value).to_legacy())
        self.input_color_image.update_image(color_ref.to_legacy())

        self.raycast_depth_image.update_image(
            depth_ref.colorize_depth(float(self.scale_slider.int_value),
                                     self.config.depth_min,
                                     self.max_slider.double_value).to_legacy())
        self.raycast_color_image.update_image(color_ref.to_legacy())
        self.window.set_needs_layout()

        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5], [5, 5, 5])
        self.widget3d.setup_camera(60, bbox, [0, 0, 0])
        self.widget3d.look_at([0, 0, 0], [0, -1, -3], [0, -1, 0])

    def update_render(self, input_depth, input_color, raycast_depth,
                      raycast_color, pcd, frustum):
        self.input_depth_image.update_image(
            input_depth.colorize_depth(
                float(self.scale_slider.int_value), self.config.depth_min,
                self.max_slider.double_value).to_legacy())
        self.input_color_image.update_image(input_color.to_legacy())

        self.raycast_depth_image.update_image(
            raycast_depth.colorize_depth(
                float(self.scale_slider.int_value), self.config.depth_min,
                self.max_slider.double_value).to_legacy())
        self.raycast_color_image.update_image(
            (raycast_color).to(o3c.uint8, False, 255.0).to_legacy())

        if self.is_scene_updated:
            if pcd is not None and pcd.point.positions.shape[0] > 0:
                self.widget3d.scene.scene.update_geometry(
                    'points', pcd, rendering.Scene.UPDATE_POINTS_FLAG |
                                   rendering.Scene.UPDATE_COLORS_FLAG)

        self.widget3d.scene.remove_geometry("frustum")
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 5.0
        self.widget3d.scene.add_geometry("frustum", frustum, mat)

    # Major loop
    def update_main(self):
        # Create camera configuration for D435iCamera
        camera_config = {
            "camera": {
                "width": 640,
                "height": 480,
                "fps": 30
            }
        }

        # Initialize D435iCamera with shared camera manager
        try:
            self.camera = D435iCamera(camera_config)
            print("D435iCamera initialized successfully for SLAM")
        except Exception as e:
            print(f"Failed to initialize D435iCamera: {e}")
            return

        # Get camera intrinsics from the shared camera manager
        intrinsic_matrix = self.camera.get_intrinsics()
        intrinsic = o3c.Tensor(intrinsic_matrix)

        device = o3d.core.Device(self.config.device)

        T_frame_to_model = o3c.Tensor(np.identity(4))

        # Get initial frame for setup - retry with timeout
        print("Waiting for initial frames from camera...")
        color_image, depth_image = None, None
        for attempt in range(50):  # 5 second timeout
            color_image, depth_image = self.camera.get_frames()
            if color_image is not None and depth_image is not None:
                print(f"Got initial frames after {attempt + 1} attempts")
                break
            time.sleep(0.1)

        if color_image is None or depth_image is None:
            print("Failed to get initial frames from camera after timeout")
            return

        # Convert to Open3D tensors
        depth_ref = o3d.t.geometry.Image(depth_image.astype(np.uint16)).to(device)
        color_ref = o3d.t.geometry.Image(color_image).to(device)

        input_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,
                                                 depth_ref.columns, intrinsic,
                                                 device)
        raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,
                                                   depth_ref.columns, intrinsic,
                                                   device)

        input_frame.set_data_from_image('depth', depth_ref)
        input_frame.set_data_from_image('color', color_ref)

        raycast_frame.set_data_from_image('depth', depth_ref)
        raycast_frame.set_data_from_image('color', color_ref)

        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.init_render(depth_ref, color_ref))

        fps_interval_len = 30
        self.idx = 0
        pcd = None

        start = time.time()
        try:
            while not self.is_done:
                if not self.is_started or not self.is_running:
                    time.sleep(0.05)
                    continue

                # Get frames from D435iCamera (shared camera manager)
                color_image, depth_image = self.camera.get_frames()

                if color_image is None or depth_image is None:
                    time.sleep(0.01)  # Brief pause before next attempt
                    continue

                # Convert to Open3D tensors
                depth = o3d.t.geometry.Image(depth_image.astype(np.uint16)).to(device)
                color = o3d.t.geometry.Image(color_image).to(device)

                input_frame.set_data_from_image('depth', depth)
                input_frame.set_data_from_image('color', color)

                if self.idx > 0:
                    result = self.model.track_frame_to_model(
                        input_frame,
                        raycast_frame,
                        float(self.scale_slider.int_value),
                        self.max_slider.double_value,
                    )
                    T_frame_to_model = T_frame_to_model @ result.transformation

                self.poses.append(T_frame_to_model.cpu().numpy())
                self.model.update_frame_pose(self.idx, T_frame_to_model)
                self.model.integrate(input_frame,
                                     float(self.scale_slider.int_value),
                                     self.max_slider.double_value,
                                     self.trunc_multiplier_slider.double_value)
                self.model.synthesize_model_frame(
                    raycast_frame, float(self.scale_slider.int_value),
                    self.config.depth_min, self.max_slider.double_value,
                    self.trunc_multiplier_slider.double_value,
                    self.raycast_box.checked)

                if (self.idx % self.interval_slider.int_value == 0 and
                    self.update_box.checked) \
                        or (self.idx == 3):
                    pcd = self.model.voxel_grid.extract_point_cloud(
                        3.0, self.est_point_count_slider.int_value).to(
                        o3d.core.Device('CUDA:0'))
                    self.is_scene_updated = True
                else:
                    self.is_scene_updated = False

                # Update occupancy grid periodically
                if (self.idx - self.last_occupancy_update) >= self.occupancy_update_interval and pcd is not None:
                    occupancy_grid = self.create_occupancy_grid_from_pointcloud(pcd)
                    if occupancy_grid is not None:
                        self.current_occupancy_grid = occupancy_grid.copy()  # Make a copy to avoid threading issues
                        self.last_occupancy_update = self.idx

                        # Update GUI with copy
                        gui.Application.instance.post_to_main_thread(
                            self.window, lambda grid=occupancy_grid.copy(): self._update_occupancy_grid_gui(grid)
                        )

                frustum = o3d.geometry.LineSet.create_camera_visualization(
                    color.columns, color.rows, intrinsic.numpy(),
                    np.linalg.inv(T_frame_to_model.cpu().numpy()), 0.2)
                frustum.paint_uniform_color([0.961, 0.475, 0.000])

                # Output FPS
                if (self.idx % fps_interval_len == 0):
                    end = time.time()
                    elapsed = end - start
                    start = time.time()
                    self.output_fps.text = 'FPS: {:.3f}'.format(fps_interval_len /
                                                                elapsed)

                # Output info
                info = 'Frame {}\n\n'.format(self.idx)
                info += 'Transformation:\n{}\n'.format(
                    np.array2string(T_frame_to_model.numpy(),
                                    precision=3,
                                    max_line_width=40,
                                    suppress_small=True))
                info += 'Active voxel blocks: {}/{}\n'.format(
                    self.model.voxel_grid.hashmap().size(),
                    self.model.voxel_grid.hashmap().capacity())
                info += 'Surface points: {}/{}\n'.format(
                    0 if pcd is None else pcd.point.positions.shape[0],
                    self.est_point_count_slider.int_value)

                # Add camera status info
                camera_status = self.camera.get_device_status()
                info += f'Camera status: {"Streaming" if camera_status["is_streaming"] else "Not streaming"}\n'
                info += f'Subscribers: {camera_status["subscribers"]}\n'

                # Add occupancy grid info
                if self.current_occupancy_grid is not None:
                    grid = self.current_occupancy_grid
                    info += f'\nOccupancy Grid:\n'
                    info += f'  Size: {grid.shape[0]}x{grid.shape[1]}\n'
                    info += f'  Resolution: {self.occupancy_grid_resolution}m/cell\n'
                    unknown_cells = np.sum(grid == 50)
                    info += f'  Coverage: {(grid.size - unknown_cells) / grid.size * 100:.1f}%\n'

                self.output_info.text = info

                # Capture variables for GUI update
                input_depth_img = input_frame.get_data_as_image('depth')
                input_color_img = input_frame.get_data_as_image('color')
                raycast_depth_img = raycast_frame.get_data_as_image('depth')
                raycast_color_img = raycast_frame.get_data_as_image('color')

                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self._update_render_gui(
                        input_depth_img, input_color_img, raycast_depth_img, raycast_color_img, pcd, frustum))

                self.idx += 1

        finally:
            if self.camera:
                self.camera.stop()

        time.sleep(0.5)


if __name__ == '__main__':
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print('\nShutdown signal received, closing application...')
        if 'w' in locals():
            w._on_close()
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)

    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
             'reference. It overrides the default config file, but will be '
             'overridden by other command line inputs.')
    parser.add('--default_dataset',
               help='Default dataset is used when config file is not provided. '
                    'Default dataset may be selected from the following options: '
                    '[lounge, bedroom, jack_jack]',
               default='lounge')
    parser.add('--path_npz',
               help='path to the npz file that stores voxel block grid.',
               default='output.npz')
    config = parser.get_config()

    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    w = ReconstructionWindow(config, mono)

    try:
        app.run()
    except KeyboardInterrupt:
        print('\nKeyboard interrupt received')
        w._on_close()
    except Exception as e:
        print(f'Application error: {e}')
        w._on_close()
    finally:
        print('Application terminated')