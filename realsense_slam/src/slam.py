import open3d as o3d
import numpy as np
import json
import time
import cv2
import os


class SimpleSLAM:
    def __init__(self, intrinsics, config):
        self.intrinsics = intrinsics
        self.config = config

        # Initialize pose and trajectory
        self.current_pose = np.eye(4)
        self.trajectory = [self.current_pose.copy()]

        # Initialize point cloud integration - optimized approach
        self.global_pcd = o3d.geometry.PointCloud()
        self.frame_pcds = []
        self.max_frames = 5  # Reduced from 10
        self.max_map_points = 100000  # Hard limit on map size

        # More aggressive keyframe selection for performance
        self.last_keyframe_pose = np.eye(4)
        self.keyframe_distance = 0.1  # Increased from 0.05
        self.keyframe_angle = 0.3  # Increased from 0.2

        self.prev_rgbd = None
        self.frame_count = 0

        # Localized IMU integration (like the orientation tracker)
        self.initial_orientation = None  # Reference orientation
        self.current_orientation = np.eye(3)  # Current absolute orientation
        self.relative_orientation = np.eye(3)  # Relative to initial

        # IMU bias estimation
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.bias_samples = []
        self.bias_estimated = False
        self.reference_set = False

        # IMU state tracking
        self.prev_accel = None
        self.prev_gyro = None
        self.prev_timestamp = None
        self.velocity = np.zeros(3)

        # Zero-velocity detection
        self.stationary_threshold = 0.5  # m/s²
        self.stationary_count = 0

        print("SLAM: Initializing with localized IMU integration...")
        print("SLAM: Keep device STATIONARY for first 3 seconds for IMU calibration")

    def estimate_imu_bias(self, accel, gyro):
        """Estimate IMU bias during initial stationary period"""
        if len(self.bias_samples) < 100:
            self.bias_samples.append([accel, gyro])
            return False

        if not self.bias_estimated:
            samples = np.array(self.bias_samples)
            gyro_samples = samples[:, 1]
            accel_samples = samples[:, 0]

            # Estimate gyro bias
            self.gyro_bias = np.mean(gyro_samples, axis=0)

            # Estimate initial orientation from gravity
            mean_accel = np.mean(accel_samples, axis=0)
            gravity_magnitude = np.linalg.norm(mean_accel)

            if gravity_magnitude > 5.0:  # Valid gravity reading
                gravity_direction = mean_accel / gravity_magnitude

                # Calculate initial orientation to align device Z with -gravity
                device_z = np.array([0, 0, 1])
                target_z = -gravity_direction

                v = np.cross(device_z, target_z)
                s = np.linalg.norm(v)
                c = np.dot(device_z, target_z)

                if s > 1e-6:
                    vx = np.array([[0, -v[2], v[1]],
                                   [v[2], 0, -v[0]],
                                   [-v[1], v[0], 0]])
                    self.initial_orientation = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s * s))
                else:
                    self.initial_orientation = np.eye(3)

                self.current_orientation = self.initial_orientation.copy()
                self.bias_estimated = True
                self.reference_set = True

                print(f"SLAM: IMU bias estimated and reference set")
                print(f"SLAM: Gyro bias: [{self.gyro_bias[0]:.4f}, {self.gyro_bias[1]:.4f}, {self.gyro_bias[2]:.4f}]")

        return self.bias_estimated

    def is_stationary(self, accel, gyro):
        """Detect if device is stationary for zero-velocity updates"""
        if not self.bias_estimated:
            return False

        accel_corrected = accel - self.accel_bias
        gyro_corrected = gyro - self.gyro_bias

        # Check if acceleration is close to gravity only
        total_accel = np.linalg.norm(accel_corrected)
        accel_deviation = abs(total_accel - 9.81)

        # Check if gyro is small
        gyro_magnitude = np.linalg.norm(gyro_corrected)

        is_still = (accel_deviation < self.stationary_threshold and
                    gyro_magnitude < 0.1)

        if is_still:
            self.stationary_count += 1
        else:
            self.stationary_count = 0

        return self.stationary_count > 5

    def update_imu_orientation(self, gyro, dt):
        """Update IMU orientation using bias-corrected gyroscope"""
        if not self.bias_estimated:
            return

        gyro_corrected = gyro - self.gyro_bias
        angular_speed = np.linalg.norm(gyro_corrected)

        if angular_speed > 0.01:
            axis = gyro_corrected / angular_speed
            angle = angular_speed * dt

            # Rodrigues' rotation formula
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])

            delta_R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            self.current_orientation = np.dot(self.current_orientation, delta_R)

            # Calculate relative orientation from initial reference
            self.relative_orientation = np.dot(self.current_orientation, self.initial_orientation.T)

    def correct_imu_with_accel(self, accel, alpha=0.02):
        """Correct IMU orientation drift using accelerometer"""
        if not self.bias_estimated:
            return

        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1:
            accel_unit = accel / accel_norm

            # Expected gravity direction in current device frame
            gravity_device = np.array([0, 0, -1])
            gravity_world = np.dot(self.current_orientation, gravity_device)

            # Correction to align with measured gravity
            cross_product = np.cross(gravity_world, accel_unit)
            sin_angle = np.linalg.norm(cross_product)
            cos_angle = np.dot(gravity_world, accel_unit)

            if sin_angle > 0.01:
                axis = cross_product / sin_angle
                correction_angle = alpha * np.arctan2(sin_angle, cos_angle)

                K = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])

                correction_R = np.eye(3) + np.sin(correction_angle) * K + \
                               (1 - np.cos(correction_angle)) * np.dot(K, K)

                self.current_orientation = np.dot(correction_R, self.current_orientation)
                self.relative_orientation = np.dot(self.current_orientation, self.initial_orientation.T)

    def predict_motion_with_imu(self, accel, gyro, timestamp):
        """Predict camera motion using localized IMU data"""
        if not self.bias_estimated or self.prev_timestamp is None:
            return np.eye(4)

        dt = timestamp - self.prev_timestamp
        if dt <= 0 or dt > 0.5:
            return np.eye(4)

        # Update IMU orientation
        if self.prev_gyro is not None:
            self.update_imu_orientation(gyro, dt)
            self.correct_imu_with_accel(accel)

        # Predict rotation using averaged gyro
        if self.prev_gyro is not None:
            avg_gyro = (gyro + self.prev_gyro) / 2
            gyro_corrected = avg_gyro - self.gyro_bias

            angle = np.linalg.norm(gyro_corrected) * dt
            if angle > 0.001:
                axis = gyro_corrected / np.linalg.norm(gyro_corrected)

                K = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])

                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            else:
                R = np.eye(3)

            # Conservative translation prediction
            is_stationary = self.is_stationary(accel, gyro)
            if is_stationary:
                t = np.zeros(3)
                self.velocity *= 0.9  # Damping when stationary
            else:
                # Simple velocity-based prediction with heavy damping
                t = self.velocity * dt * 0.1  # Very conservative

            transform = np.eye(4)
            transform[:3, :3] = R
            transform[:3, 3] = t

            return transform

        return np.eye(4)

    def validate_with_imu_improved(self, visual_transform, accel, gyro, timestamp):
        """Improved IMU validation using localized reference"""
        if not self.bias_estimated or self.prev_gyro is None:
            return True

        dt = timestamp - self.prev_timestamp
        if dt <= 0 or dt > 0.5:
            return True

        # Extract rotation from visual transform
        visual_rotation_angle = np.arccos(np.clip((np.trace(visual_transform[:3, :3]) - 1) / 2, -1, 1))

        # Expected rotation from IMU
        gyro_corrected = gyro - self.gyro_bias
        imu_rotation_angle = np.linalg.norm(gyro_corrected) * dt

        # More lenient validation - only reject if very different
        rotation_threshold = 0.8  # Reduced from 1.0 for better accuracy
        if abs(visual_rotation_angle - imu_rotation_angle) > rotation_threshold:
            return False

        # Check if stationary - visual odometry should show minimal motion
        if self.is_stationary(accel, gyro):
            translation_magnitude = np.linalg.norm(visual_transform[:3, 3])
            if translation_magnitude > 0.05:  # 5cm threshold when stationary
                return False

        # Validate translation magnitude
        translation_magnitude = np.linalg.norm(visual_transform[:3, 3])
        max_translation = 0.15  # 15cm max per frame
        if translation_magnitude > max_translation:
            return False

        return True

    def update_imu_state(self, accel, gyro, timestamp):
        """Update IMU state for next prediction"""
        if self.bias_estimated and self.prev_timestamp is not None:
            dt = timestamp - self.prev_timestamp
            if 0 < dt < 0.5 and accel is not None:
                # Update velocity with gravity compensation and damping
                if not self.is_stationary(accel, gyro):
                    accel_corrected = accel - self.accel_bias
                    world_accel = np.dot(self.relative_orientation, accel_corrected)
                    world_accel[2] += 9.81  # Remove gravity

                    self.velocity += world_accel * dt
                    self.velocity *= 0.95  # Strong damping
                else:
                    self.velocity *= 0.8  # Stronger damping when stationary

        self.prev_accel = accel
        self.prev_gyro = gyro
        self.prev_timestamp = timestamp

    def is_keyframe(self, current_pose):
        """Determine if current frame should be a keyframe"""
        if len(self.trajectory) < 2:
            return True

        # Calculate translation and rotation difference
        trans_diff = np.linalg.norm(current_pose[:3, 3] - self.last_keyframe_pose[:3, 3])
        rot_diff = np.arccos(
            np.clip((np.trace(current_pose[:3, :3] @ self.last_keyframe_pose[:3, :3].T) - 1) / 2, -1, 1))

        return trans_diff > self.keyframe_distance or rot_diff > self.keyframe_angle

    def process_frame(self, rgb, depth, accel=None, gyro=None):
        self.frame_count += 1
        current_time = time.time()

        # IMU bias estimation during initialization
        if accel is not None and gyro is not None and not self.bias_estimated:
            bias_ready = self.estimate_imu_bias(accel, gyro)
            if not bias_ready:
                return  # Skip processing until bias is estimated

        # Create Open3D camera intrinsic
        height, width = depth.shape
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height,
            self.intrinsics[0, 0], self.intrinsics[1, 1],
            self.intrinsics[0, 2], self.intrinsics[1, 2]
        )

        # Convert to Open3D format
        color_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0, depth_trunc=self.config["slam"]["max_depth"],
            convert_rgb_to_intensity=False
        )

        if self.prev_rgbd is not None:
            # IMU prediction for better initial guess
            initial_guess = self.predict_motion_with_imu(accel, gyro, current_time)

            # Visual odometry with IMU-predicted initial guess
            option = o3d.pipelines.odometry.OdometryOption()

            success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                self.prev_rgbd, rgbd, intrinsic, initial_guess,
                o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option
            )

            # Improved IMU validation
            if success and self.bias_estimated:
                success = self.validate_with_imu_improved(trans, accel, gyro, current_time)

            if success:
                # Update pose
                self.current_pose = np.dot(self.current_pose, trans)
                self.trajectory.append(self.current_pose.copy())

                # Only process keyframes for mapping
                if self.is_keyframe(self.current_pose):
                    self.add_keyframe(rgbd, intrinsic)
                    self.last_keyframe_pose = self.current_pose.copy()
            else:
                # If visual odometry fails, use IMU prediction as fallback
                if self.bias_estimated and accel is not None and gyro is not None:
                    imu_prediction = self.predict_motion_with_imu(accel, gyro, current_time)
                    # Only use IMU if prediction is reasonable
                    if np.linalg.norm(imu_prediction[:3, 3]) < 0.05:  # Small translation
                        self.current_pose = np.dot(self.current_pose, imu_prediction)
                        self.trajectory.append(self.current_pose.copy())

        # Update IMU state
        if accel is not None and gyro is not None:
            self.update_imu_state(accel, gyro, current_time)

        self.prev_rgbd = rgbd

    def add_keyframe(self, rgbd, intrinsic):
        """Add keyframe to map with aggressive optimization"""
        # Generate point cloud from keyframe
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        # Early aggressive downsampling
        pcd = pcd.voxel_down_sample(self.config["slam"]["voxel_size"] * 1.5)

        # Remove statistical outliers early
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=8, std_ratio=1.5)

        # Transform to global coordinates
        pcd.transform(self.current_pose)

        # Store frame point cloud
        self.frame_pcds.append(pcd)

        # Keep only recent frames
        if len(self.frame_pcds) > self.max_frames:
            self.frame_pcds.pop(0)

        # Update global map efficiently
        self.update_global_map()

    def update_global_map(self):
        """Efficiently update global point cloud map with size limits"""
        if not self.frame_pcds:
            return

        # Combine recent frames
        combined_pcd = o3d.geometry.PointCloud()
        for pcd in self.frame_pcds:
            combined_pcd += pcd

        # Aggressive downsampling to control size
        combined_pcd = combined_pcd.voxel_down_sample(self.config["slam"]["voxel_size"] * 2)

        # Enforce hard limit on map size
        if len(combined_pcd.points) > self.max_map_points:
            # Random downsampling to maintain map size
            indices = np.random.choice(len(combined_pcd.points), self.max_map_points, replace=False)
            combined_pcd = combined_pcd.select_by_index(indices)

        # Remove outliers for cleaner map
        combined_pcd, _ = combined_pcd.remove_statistical_outlier(nb_neighbors=6, std_ratio=2.0)

        self.global_pcd = combined_pcd

    def get_map(self):
        return self.global_pcd

    def get_trajectory(self):
        return self.trajectory

    def save_session(self, filename):
        # Save point cloud
        o3d.io.write_point_cloud(f"{filename}_map.ply", self.global_pcd)

        # Save trajectory
        trajectory_data = {
            "poses": [pose.tolist() for pose in self.trajectory],
            "timestamps": [i for i in range(len(self.trajectory))]
        }
        with open(f"{filename}_trajectory.json", 'w') as f:
            json.dump(trajectory_data, f)


def main():
    # Test SLAM with real RealSense camera data
    with open('../config/config.json', 'r') as f:
        config = json.load(f)

    # Import camera module
    from camera import D435iCamera

    print("Testing improved SLAM with localized IMU integration...")

    # Initialize camera
    camera = D435iCamera(config)
    intrinsics = camera.get_intrinsics()
    slam = SimpleSLAM(intrinsics, config)

    try:
        # Process real camera frames
        for i in range(50):
            rgb, depth = camera.get_frames()
            accel, gyro = camera.get_imu_data()

            if rgb is not None and depth is not None:
                # Convert depth to meters (RealSense outputs in mm)
                depth_meters = depth.astype(np.float32) / 1000.0
                slam.process_frame(rgb, depth_meters, accel, gyro)
                print(f"Processed frame {i}, trajectory length: {len(slam.get_trajectory())}")
            else:
                print(f"Frame {i}: No data received")

        map_cloud = slam.get_map()
        print(f"Map contains {len(map_cloud.points)} points")

        # Save session
        os.makedirs('../data/sessions', exist_ok=True)
        slam.save_session("../data/sessions/test_session")
        print("Session saved")

    finally:
        camera.stop()


if __name__ == "__main__":
    main()