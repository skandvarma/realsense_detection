import open3d as o3d
import numpy as np
import json
import cv2


class MinimalSLAM:
    def __init__(self, intrinsics, config):
        self.intrinsics = intrinsics
        self.config = config

        # Performance tuning parameters
        self.params = {
            'voxel_size': config.get('slam', {}).get('voxel_size', 0.05),
            'max_points': config.get('slam', {}).get('max_points', 20000),
            'icp_threshold': config.get('slam', {}).get('icp_threshold', 0.02),
            'max_depth': config.get('slam', {}).get('max_depth', 3.0),
            'process_every_n': config.get('slam', {}).get('process_every_n', 1),
            'accumulate_every_n': config.get('slam', {}).get('accumulate_every_n', 3)
        }

        # State
        self.current_pose = np.eye(4)
        self.trajectory = [self.current_pose.copy()]
        self.map_cloud = o3d.geometry.PointCloud()
        self.prev_cloud = None
        self.frame_count = 0

        # Camera intrinsic for Open3D
        height, width = config['camera']['height'], config['camera']['width']
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height,
            intrinsics[0, 0], intrinsics[1, 1],
            intrinsics[0, 2], intrinsics[1, 2]
        )

        print(f"Minimal SLAM initialized with params: {self.params}")

    def create_point_cloud(self, rgb, depth):
        """Create point cloud from RGB-D with minimal processing"""
        # Convert to Open3D format
        color_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=self.params['max_depth'],
            convert_rgb_to_intensity=False
        )

        # Generate point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)

        # Minimal preprocessing - just downsample
        if len(pcd.points) > 0:
            pcd = pcd.voxel_down_sample(self.params['voxel_size'])

        return pcd

    def estimate_pose(self, source_pcd, target_pcd):
        """Simple ICP pose estimation"""
        if len(source_pcd.points) < 100 or len(target_pcd.points) < 100:
            return False, np.eye(4)

        # Fast ICP with minimal iterations
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd,
            self.params['icp_threshold'],
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=10  # Minimal iterations for speed
            )
        )

        return result.fitness > 0.1, result.transformation

    def accumulate_to_map(self, pcd):
        """Accumulate point cloud to global map"""
        # Transform to global coordinates
        pcd_global = pcd.transform(self.current_pose)

        # Simple accumulation
        self.map_cloud += pcd_global

        # Control map size for performance
        if len(self.map_cloud.points) > self.params['max_points']:
            # Aggressive downsampling to maintain size
            self.map_cloud = self.map_cloud.voxel_down_sample(self.params['voxel_size'] * 1.5)

            # If still too large, random downsample
            if len(self.map_cloud.points) > self.params['max_points']:
                indices = np.random.choice(
                    len(self.map_cloud.points),
                    self.params['max_points'],
                    replace=False
                )
                self.map_cloud = self.map_cloud.select_by_index(indices)

    def process_frame(self, rgb, depth):
        """Process single frame with minimal operations"""
        self.frame_count += 1

        # Skip frames for performance if configured
        if self.frame_count % self.params['process_every_n'] != 0:
            return

        # Create point cloud
        current_pcd = self.create_point_cloud(rgb, depth)

        if len(current_pcd.points) == 0:
            return

        # Pose estimation
        if self.prev_cloud is not None:
            success, transform = self.estimate_pose(current_pcd, self.prev_cloud)

            if success:
                # Update pose
                self.current_pose = np.dot(self.current_pose, transform)
                self.trajectory.append(self.current_pose.copy())

        # Accumulate to map less frequently
        if self.frame_count % self.params['accumulate_every_n'] == 0:
            self.accumulate_to_map(current_pcd)

        # Store for next frame
        self.prev_cloud = current_pcd

    def get_map(self):
        """Get current map"""
        return self.map_cloud

    def get_trajectory(self):
        """Get current trajectory"""
        return self.trajectory

    def save_session(self, filename):
        """Save session with minimal data"""
        # Save point cloud
        o3d.io.write_point_cloud(f"{filename}_map.ply", self.map_cloud)

        # Save trajectory
        trajectory_data = {
            "poses": [pose.tolist() for pose in self.trajectory],
            "frame_count": self.frame_count,
            "params": self.params
        }
        with open(f"{filename}_trajectory.json", 'w') as f:
            json.dump(trajectory_data, f)

    def update_params(self, **kwargs):
        """Update tunable parameters during runtime"""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
                print(f"Updated {key} to {value}")


def main():
    # Test minimal SLAM
    with open('../config/config.json', 'r') as f:
        config = json.load(f)

    from camera import D435iCamera

    print("Testing Minimal SLAM...")
    camera = D435iCamera(config)
    intrinsics = camera.get_intrinsics()
    slam = MinimalSLAM(intrinsics, config)

    try:
        for i in range(100):
            rgb, depth = camera.get_frames()
            if rgb is not None and depth is not None:
                depth_meters = depth.astype(np.float32) / 1000.0
                slam.process_frame(rgb, depth_meters)

                if i % 10 == 0:
                    print(f"Frame {i}: trajectory={len(slam.get_trajectory())}, map={len(slam.get_map().points)}")

        slam.save_session("../data/sessions/minimal_test")
        print("Minimal SLAM test complete")

    finally:
        camera.stop()


if __name__ == "__main__":
    main()