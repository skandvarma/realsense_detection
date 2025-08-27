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
from common import load_rgbd_file_names, save_poses, load_intrinsic, extract_trianglemesh, get_default_dataset, \
    extract_rgbd_frames
from collections import deque
from scipy.spatial.transform import Rotation as R
from scipy import signal

# Add project root to Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..'))
sys.path.insert(0, project_root)

# Import D435iCamera from shared camera manager
from src.camera.realsense_manager import D435iCamera


def set_enabled(widget, enable):
    widget.enabled = enable
    for child in widget.get_children():
        child.enabled = enable


class IMUProcessor:
    """Handles IMU data processing and fusion with visual odometry"""

    def __init__(self, buffer_size=100):
        self.buffer_size = buffer_size
        self.accel_buffer = deque(maxlen=buffer_size)
        self.gyro_buffer = deque(maxlen=buffer_size)
        self.timestamp_buffer = deque(maxlen=buffer_size)

        # Calibration parameters
        self.gravity = np.array([0, 0, -9.81])
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

        # Current state
        self.orientation = np.eye(3)
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)

        # Filter parameters
        self.alpha = 0.98  # Complementary filter coefficient
        self.last_timestamp = None

        # Low-pass filter for accelerometer
        self.b, self.a = signal.butter(4, 0.1, 'low')

    def add_measurement(self, accel, gyro, timestamp):
        """Add new IMU measurement to buffers"""
        if accel is not None and gyro is not None:
            # Ensure numpy arrays
            if not isinstance(accel, np.ndarray):
                accel = np.array(accel)
            if not isinstance(gyro, np.ndarray):
                gyro = np.array(gyro)

            self.accel_buffer.append(accel)
            self.gyro_buffer.append(gyro)
            self.timestamp_buffer.append(timestamp)

    def calibrate(self, static_duration=2.0):
        """Calibrate IMU biases during static period"""
        if len(self.accel_buffer) > 0:
            recent_accel = np.array(list(self.accel_buffer)[-30:])
            recent_gyro = np.array(list(self.gyro_buffer)[-30:])

            if len(recent_accel) > 0:
                self.accel_bias = np.mean(recent_accel, axis=0) - self.gravity
                self.gyro_bias = np.mean(recent_gyro, axis=0)

    def process_imu(self, accel, gyro, dt):
        """Process IMU data and return motion prediction"""
        if dt is None or dt <= 0:
            return np.eye(4)

        # Remove biases
        accel_corrected = accel - self.accel_bias
        gyro_corrected = gyro - self.gyro_bias

        # Apply low-pass filter to accelerometer only if enough samples
        if len(self.accel_buffer) > 15:  # Minimum required for filter
            accel_array = np.array(list(self.accel_buffer))
            accel_filtered = signal.filtfilt(self.b, self.a, accel_array, axis=0)
            if len(accel_filtered) > 0:
                accel_corrected = accel_filtered[-1]

        # Update orientation using gyroscope
        angle = np.linalg.norm(gyro_corrected) * dt
        if angle > 0:
            axis = gyro_corrected / np.linalg.norm(gyro_corrected)
            rot_matrix = R.from_rotvec(axis * angle).as_matrix()
            self.orientation = self.orientation @ rot_matrix

        # Transform acceleration to world frame
        accel_world = self.orientation @ accel_corrected

        # Remove gravity
        accel_world -= self.gravity

        # Update velocity and position
        self.velocity += accel_world * dt
        self.position += self.velocity * dt + 0.5 * accel_world * dt * dt

        # Create transformation matrix
        T_imu = np.eye(4)
        T_imu[:3, :3] = self.orientation
        T_imu[:3, 3] = self.position

        return T_imu

    def get_prediction(self, timestamp):
        """Get motion prediction for given timestamp"""
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            return np.eye(4)

        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp

        if len(self.accel_buffer) > 0 and len(self.gyro_buffer) > 0:
            accel = self.accel_buffer[-1]
            gyro = self.gyro_buffer[-1]
            return self.process_imu(accel, gyro, dt)

        return np.eye(4)

    def reset_integration(self):
        """Reset integration states"""
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)


class VisualInertialFusion:
    """Handles fusion between visual odometry and IMU predictions"""

    def __init__(self):
        self.visual_confidence = 1.0
        self.imu_confidence = 0.3
        self.last_visual_pose = np.eye(4)
        self.drift_correction = np.eye(4)

    def update_confidence(self, tracking_quality, motion_magnitude):
        """Update confidence weights based on tracking quality"""
        # Increase IMU confidence during fast motion
        if motion_magnitude > 0.5:
            self.imu_confidence = min(0.7, self.imu_confidence + 0.1)
            self.visual_confidence = max(0.3, self.visual_confidence - 0.1)
        else:
            self.imu_confidence = max(0.3, self.imu_confidence - 0.05)
            self.visual_confidence = min(1.0, self.visual_confidence + 0.05)

    def fuse_poses(self, visual_pose, imu_prediction, tracking_success=True):
        """Fuse visual and IMU pose estimates"""
        if not tracking_success:
            # Rely more on IMU when visual tracking fails
            return imu_prediction

        # Weighted average of transformations
        weight_visual = self.visual_confidence / (self.visual_confidence + self.imu_confidence)
        weight_imu = self.imu_confidence / (self.visual_confidence + self.imu_confidence)

        # Separate rotation and translation
        visual_rot = visual_pose[:3, :3]
        visual_trans = visual_pose[:3, 3]
        imu_rot = imu_prediction[:3, :3]
        imu_trans = imu_prediction[:3, 3]

        # Fuse rotations using SLERP
        visual_quat = R.from_matrix(visual_rot).as_quat()
        imu_quat = R.from_matrix(imu_rot).as_quat()

        # Simple quaternion interpolation
        fused_quat = weight_visual * visual_quat + weight_imu * imu_quat
        fused_quat /= np.linalg.norm(fused_quat)
        fused_rot = R.from_quat(fused_quat).as_matrix()

        # Fuse translations
        fused_trans = weight_visual * visual_trans + weight_imu * imu_trans

        # Construct fused transformation
        fused_pose = np.eye(4)
        fused_pose[:3, :3] = fused_rot
        fused_pose[:3, 3] = fused_trans

        return fused_pose

    def detect_drift(self, visual_pose, imu_pose):
        """Detect and correct systematic drift"""
        # Calculate difference between visual and IMU estimates
        diff = np.linalg.inv(visual_pose) @ imu_pose

        # Check if difference is significant
        rotation_diff = np.arccos(np.clip((np.trace(diff[:3, :3]) - 1) / 2, -1, 1))
        translation_diff = np.linalg.norm(diff[:3, 3])

        if rotation_diff > 0.1 or translation_diff > 0.1:
            # Apply gradual drift correction
            alpha = 0.01  # Correction rate
            self.drift_correction = (1 - alpha) * self.drift_correction + alpha * diff

        return self.drift_correction


class ReconstructionWindow:

    def __init__(self, config, font_id):
        self.config = config

        self.window = gui.Application.instance.create_window(
            'Open3D - Visual-Inertial Reconstruction', 1280, 720)

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

        ### IMU fusion enabled?
        imu_label = gui.Label('IMU fusion?')
        self.imu_box = gui.Checkbox('')
        self.imu_box.checked = True
        self.adjustable_prop_grid.add_child(imu_label)
        self.adjustable_prop_grid.add_child(self.imu_box)

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

        # Initialize IMU components
        self.imu_processor = IMUProcessor()
        self.fusion = VisualInertialFusion()
        self.imu_thread = None
        self.imu_running = False

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
        self.imu_running = False

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

        return True

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

    def imu_acquisition_thread(self):
        """Separate thread for continuous IMU data acquisition"""
        while self.imu_running and not self.is_done:
            if self.camera:
                imu_data = self.camera.get_imu_data()
                if imu_data:
                    timestamp = time.time()
                    # Handle tuple format (accel, gyro) from get_imu_data
                    if isinstance(imu_data, tuple) and len(imu_data) == 2:
                        accel, gyro = imu_data
                        self.imu_processor.add_measurement(accel, gyro, timestamp)
                    elif isinstance(imu_data, dict):
                        self.imu_processor.add_measurement(
                            imu_data.get('accel'),
                            imu_data.get('gyro'),
                            timestamp
                        )
            time.sleep(0.005)  # 200Hz IMU sampling

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
            print("D435iCamera initialized successfully for Visual-Inertial SLAM")

            # Start IMU acquisition thread
            self.imu_running = True
            self.imu_thread = threading.Thread(target=self.imu_acquisition_thread)
            self.imu_thread.start()

        except Exception as e:
            print(f"Failed to initialize D435iCamera: {e}")
            return

        # Get camera intrinsics from the shared camera manager
        intrinsic_matrix = self.camera.get_intrinsics()
        intrinsic = o3c.Tensor(intrinsic_matrix)

        device = o3d.core.Device(self.config.device)

        T_frame_to_model = o3c.Tensor(np.identity(4))
        T_frame_to_model_prev = np.identity(4)

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

        # IMU calibration during static period
        print("Calibrating IMU... Please keep the camera still for 2 seconds")
        time.sleep(2.0)
        self.imu_processor.calibrate()
        print("IMU calibration complete")

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

        # Tracking quality metrics
        tracking_success = True
        motion_magnitude = 0.0

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

                current_timestamp = time.time()

                # Convert to Open3D tensors
                depth = o3d.t.geometry.Image(depth_image.astype(np.uint16)).to(device)
                color = o3d.t.geometry.Image(color_image).to(device)

                input_frame.set_data_from_image('depth', depth)
                input_frame.set_data_from_image('color', color)

                if self.idx > 0:
                    # Get IMU prediction if enabled
                    imu_prediction = np.eye(4)
                    if self.imu_box.checked and self.imu_processor:
                        imu_prediction = self.imu_processor.get_prediction(current_timestamp)

                    # Visual tracking
                    result = self.model.track_frame_to_model(
                        input_frame,
                        raycast_frame,
                        float(self.scale_slider.int_value),
                        self.max_slider.double_value,
                    )

                    visual_delta = result.transformation.cpu().numpy()
                    tracking_success = result.fitness > 0.5  # Adjust threshold as needed

                    # Calculate motion magnitude for confidence weighting
                    motion_magnitude = np.linalg.norm(visual_delta[:3, 3])

                    if self.imu_box.checked:
                        # Update fusion confidence
                        self.fusion.update_confidence(result.fitness, motion_magnitude)

                        # Fuse visual and IMU estimates
                        fused_transformation = self.fusion.fuse_poses(
                            visual_delta,
                            np.linalg.inv(T_frame_to_model_prev) @ imu_prediction,
                            tracking_success
                        )

                        T_frame_to_model = T_frame_to_model @ o3c.Tensor(fused_transformation)

                        # Reset IMU integration after fusion
                        self.imu_processor.reset_integration()
                    else:
                        # Pure visual odometry
                        T_frame_to_model = T_frame_to_model @ result.transformation

                T_frame_to_model_prev = T_frame_to_model.cpu().numpy()
                self.poses.append(T_frame_to_model_prev)
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
                    np.array2string(T_frame_to_model_prev,
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

                # Add IMU fusion status
                if self.imu_box.checked:
                    info += f'IMU Fusion: ON\n'
                    info += f'Visual confidence: {self.fusion.visual_confidence:.2f}\n'
                    info += f'IMU confidence: {self.fusion.imu_confidence:.2f}\n'
                    info += f'Motion magnitude: {motion_magnitude:.3f}\n'
                    info += f'Tracking quality: {"Good" if tracking_success else "Poor"}\n'
                else:
                    info += f'IMU Fusion: OFF\n'

                self.output_info.text = info

                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self.update_render(
                        input_frame.get_data_as_image('depth'),
                        input_frame.get_data_as_image('color'),
                        raycast_frame.get_data_as_image('depth'),
                        raycast_frame.get_data_as_image('color'), pcd, frustum))

                self.idx += 1

        finally:
            self.imu_running = False
            if self.imu_thread:
                self.imu_thread.join()
            if self.camera:
                self.camera.stop()

        time.sleep(0.5)


if __name__ == '__main__':
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
    app.run()