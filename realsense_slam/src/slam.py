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

        # Initialize point cloud integration - lighter approach
        self.global_pcd = o3d.geometry.PointCloud()
        self.frame_pcds = []  # Store recent frame point clouds
        self.max_frames = 10  # Keep only recent frames for speed

        # Keyframe selection parameters
        self.last_keyframe_pose = np.eye(4)
        self.keyframe_distance = 0.05  # meters
        self.keyframe_angle = 0.2  # radians

        self.prev_rgbd = None
        self.frame_count = 0

        # IMU integration
        self.prev_accel = None
        self.prev_gyro = None
        self.prev_timestamp = None

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
            # Simple odometry for compatibility
            option = o3d.pipelines.odometry.OdometryOption()

            # Use basic approach without problematic attributes
            success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                self.prev_rgbd, rgbd, intrinsic, np.eye(4),
                o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option
            )

            # Simple IMU validation if available
            if success and accel is not None and gyro is not None:
                success = self._validate_with_imu(trans, accel, gyro, current_time)

            if success:
                # Update pose
                self.current_pose = np.dot(self.current_pose, trans)
                self.trajectory.append(self.current_pose.copy())

                # Only process keyframes for mapping
                if self.is_keyframe(self.current_pose):
                    self.add_keyframe(rgbd, intrinsic)
                    self.last_keyframe_pose = self.current_pose.copy()

        # Store IMU data for next frame
        self.prev_accel = accel
        self.prev_gyro = gyro
        self.prev_timestamp = current_time
        self.prev_rgbd = rgbd

    def _validate_with_imu(self, visual_transform, accel, gyro, timestamp):
        """Simple IMU validation of visual odometry"""
        if self.prev_accel is None or self.prev_gyro is None or self.prev_timestamp is None:
            return True

        dt = timestamp - self.prev_timestamp
        if dt <= 0 or dt > 0.5:  # Skip if time interval is invalid
            return True

        # Extract rotation from visual transform
        visual_rotation = np.linalg.norm(visual_transform[:3, 3])

        # Simple gyro integration check
        gyro_rotation = np.linalg.norm(gyro) * dt

        # If visual and gyro disagree significantly, reject
        if abs(visual_rotation - gyro_rotation) > 0.5:  # 0.5 rad threshold
            return False

        return True

    def add_keyframe(self, rgbd, intrinsic):
        """Add keyframe to map with ICP refinement"""
        # Generate point cloud from keyframe
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        # Transform to global coordinates
        pcd.transform(self.current_pose)

        # Downsample for performance
        pcd = pcd.voxel_down_sample(self.config["slam"]["voxel_size"])

        # Store frame point cloud
        self.frame_pcds.append(pcd)

        # Keep only recent frames
        if len(self.frame_pcds) > self.max_frames:
            self.frame_pcds.pop(0)

        # Update global map by combining recent frames
        self.update_global_map()

    def update_global_map(self):
        """Efficiently update global point cloud map"""
        if not self.frame_pcds:
            return

        # Combine recent frames
        combined_pcd = o3d.geometry.PointCloud()
        for pcd in self.frame_pcds:
            combined_pcd += pcd

        # Global downsampling and filtering
        combined_pcd = combined_pcd.voxel_down_sample(self.config["slam"]["voxel_size"])

        # Remove outliers for cleaner map
        combined_pcd, _ = combined_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)

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

    print("Testing SLAM with real D435i camera and IMU data...")

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