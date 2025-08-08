import open3d as o3d
import numpy as np
import cv2


class MinimalVisualizer:
    def __init__(self, config):
        self.config = config

        # Performance tuning parameters
        self.params = {
            'viz_points_limit': config.get('viz', {}).get('viz_points_limit', 15000),
            'update_every_n': config.get('viz', {}).get('update_every_n', 2),
            'show_trajectory': config.get('viz', {}).get('show_trajectory', True),
            'show_current_frame': config.get('viz', {}).get('show_current_frame', False),
            'point_size': config.get('viz', {}).get('point_size', 2),
            'trajectory_every_n': config.get('viz', {}).get('trajectory_every_n', 5)
        }

        # Initialize visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Enhanced SLAM - Dual Trajectory", width=1000, height=700)

        # Setup rendering
        opt = self.vis.get_render_option()
        opt.point_size = self.params['point_size']
        opt.background_color = np.array(config.get('viz', {}).get('background', [0, 0, 0]))

        # Map geometry
        self.map_pcd = o3d.geometry.PointCloud()
        self.map_added = False

        # Visual SLAM trajectory (GREEN)
        self.visual_traj_line = None
        self.visual_traj_added = False

        # IMU trajectory (RED)
        self.imu_traj_line = None
        self.imu_traj_added = False

        # Coordinate frame for reference
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        self.vis.add_geometry(self.coord_frame)

        # Frame counter
        self.frame_count = 0

        print(f"Enhanced Visualizer initialized - Visual trajectory: GREEN, IMU trajectory: RED")

    def update_map(self, point_cloud):
        """Update map visualization with minimal operations"""
        self.frame_count += 1

        # Skip updates for performance
        if self.frame_count % self.params['update_every_n'] != 0:
            return

        if point_cloud is None or len(point_cloud.points) == 0:
            return

        # Limit points for visualization performance
        viz_pcd = point_cloud
        if len(point_cloud.points) > self.params['viz_points_limit']:
            indices = np.random.choice(
                len(point_cloud.points),
                self.params['viz_points_limit'],
                replace=False
            )
            viz_pcd = point_cloud.select_by_index(indices)

        # Update or add geometry
        if not self.map_added:
            self.map_pcd = viz_pcd
            self.vis.add_geometry(self.map_pcd)
            self.map_added = True
        else:
            self.map_pcd.points = viz_pcd.points
            if len(viz_pcd.colors) > 0:
                self.map_pcd.colors = viz_pcd.colors
            self.vis.update_geometry(self.map_pcd)

    def update_visual_trajectory(self, poses):
        """Update visual SLAM trajectory (GREEN)"""
        if not self.params['show_trajectory'] or len(poses) < 2:
            return

        # Only update every N poses for performance
        if len(poses) % self.params['trajectory_every_n'] != 0:
            return

        # Extract positions
        positions = [pose[:3, 3] for pose in poses[::self.params['trajectory_every_n']]]

        if len(positions) < 2:
            return

        # Create or update line set
        if not self.visual_traj_added:
            self.visual_traj_line = o3d.geometry.LineSet()
            self.vis.add_geometry(self.visual_traj_line)
            self.visual_traj_added = True

        # Create lines
        points = o3d.utility.Vector3dVector(positions)
        lines = [[i, i + 1] for i in range(len(positions) - 1)]

        self.visual_traj_line.points = points
        self.visual_traj_line.lines = o3d.utility.Vector2iVector(lines)

        # GREEN color for visual trajectory
        colors = [[0, 1, 0]] * len(lines)  # Green trajectory
        self.visual_traj_line.colors = o3d.utility.Vector3dVector(colors)

        self.vis.update_geometry(self.visual_traj_line)

    def update_imu_trajectory(self, poses):
        """Update IMU trajectory (RED)"""
        if not self.params['show_trajectory'] or len(poses) < 2:
            return

        # Extract positions
        positions = [pose[:3, 3] for pose in poses[::max(1, self.params['trajectory_every_n'] // 2)]]

        if len(positions) < 2:
            return

        # Create or update line set
        if not self.imu_traj_added:
            self.imu_traj_line = o3d.geometry.LineSet()
            self.vis.add_geometry(self.imu_traj_line)
            self.imu_traj_added = True

        # Create lines
        points = o3d.utility.Vector3dVector(positions)
        lines = [[i, i + 1] for i in range(len(positions) - 1)]

        self.imu_traj_line.points = points
        self.imu_traj_line.lines = o3d.utility.Vector2iVector(lines)

        # RED color for IMU trajectory
        colors = [[1, 0, 0]] * len(lines)  # Red trajectory
        self.imu_traj_line.colors = o3d.utility.Vector3dVector(colors)

        self.vis.update_geometry(self.imu_traj_line)

    def update_trajectory(self, poses):
        """Legacy method - now calls visual trajectory update"""
        self.update_visual_trajectory(poses)

    def show_frame(self, rgb_image, motion_stats=None, frame_count=0):
        """Show current RGB frame with alignment info"""
        if rgb_image is not None:
            display_img = cv2.resize(rgb_image, (480, 360))

            # Add alignment information overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # Scale factor display
            if motion_stats and 'scale_factor' in motion_stats:
                scale_text = f"Scale Factor: {motion_stats['scale_factor']:.3f}"
                cv2.putText(display_img, scale_text, (10, 30), font, font_scale, (0, 255, 255), thickness)

            # Trajectory info
            cv2.putText(display_img, "GREEN: Visual SLAM", (10, 60), font, 0.5, (0, 255, 0), 1)
            cv2.putText(display_img, "RED: IMU Trajectory", (10, 80), font, 0.5, (0, 0, 255), 1)

            # Frame counter
            cv2.putText(display_img, f"Frame: {frame_count}", (10, display_img.shape[0] - 10),
                        font, 0.5, (128, 128, 128), 1)

            cv2.imshow("Enhanced SLAM - Trajectory Alignment", display_img)
            cv2.waitKey(1)

    def spin_once(self):
        """Update visualization once"""
        self.vis.poll_events()
        self.vis.update_renderer()
        return True

    def close(self):
        """Clean shutdown"""
        self.vis.destroy_window()
        cv2.destroyAllWindows()

    def update_params(self, **kwargs):
        """Update visualization parameters at runtime"""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
                print(f"Viz param {key} updated to {value}")

                # Apply some changes immediately
                if key == 'point_size':
                    opt = self.vis.get_render_option()
                    opt.point_size = value

    def reset_view(self):
        """Reset camera view to default"""
        self.vis.reset_view_point(True)

    def get_alignment_quality_metrics(self, visual_poses, imu_poses):
        """Calculate alignment quality metrics for display"""
        if len(visual_poses) < 5 or len(imu_poses) < 5:
            return {}

        # Take recent poses for comparison
        recent_visual = visual_poses[-5:]
        recent_imu = imu_poses[-5:]

        # Calculate recent distances
        visual_dist = 0
        imu_dist = 0

        for i in range(1, len(recent_visual)):
            visual_dist += np.linalg.norm(recent_visual[i][:3, 3] - recent_visual[i - 1][:3, 3])

        for i in range(1, min(len(recent_imu), len(recent_visual))):
            imu_dist += np.linalg.norm(recent_imu[i][:3, 3] - recent_imu[i - 1][:3, 3])

        alignment_ratio = imu_dist / visual_dist if visual_dist > 0 else 0

        return {
            'visual_distance': visual_dist,
            'imu_distance': imu_dist,
            'alignment_ratio': alignment_ratio
        }


class PerformanceMonitor:
    """Performance monitoring for tuning"""

    def __init__(self):
        self.frame_times = []
        self.map_sizes = []

    def log_frame(self, frame_time, map_size):
        self.frame_times.append(frame_time)
        self.map_sizes.append(map_size)

        # Print stats every 30 frames
        if len(self.frame_times) % 30 == 0:
            avg_time = np.mean(self.frame_times[-30:])
            avg_size = np.mean(self.map_sizes[-30:])
            fps = 1.0 / avg_time if avg_time > 0 else 0

            print(f"Performance: {fps:.1f} FPS, {avg_time * 1000:.1f}ms/frame, {avg_size:.0f} points")

    def suggest_params(self):
        """Suggest parameter adjustments based on performance"""
        if len(self.frame_times) < 30:
            return {}

        avg_time = np.mean(self.frame_times[-30:])
        suggestions = {}

        if avg_time > 0.05:  # Slower than 20 FPS
            suggestions.update({
                'voxel_size': 0.08,
                'max_points': 15000,
                'update_every_n': 3,
                'viz_points_limit': 10000
            })
            print("Performance suggestions: Increase voxel_size, reduce max_points")

        elif avg_time < 0.02:  # Faster than 50 FPS
            suggestions.update({
                'voxel_size': 0.03,
                'max_points': 30000,
                'update_every_n': 1,
                'viz_points_limit': 20000
            })
            print("Performance suggestions: Decrease voxel_size, increase max_points")

        return suggestions


def main():
    """Test dual trajectory visualizer"""
    import time

    # Test with synthetic data
    config = {
        'viz': {
            'point_size': 2,
            'background': [0, 0, 0],
            'viz_points_limit': 15000,
            'update_every_n': 1,
            'show_trajectory': True
        }
    }

    viz = MinimalVisualizer(config)
    monitor = PerformanceMonitor()

    print("Testing dual trajectory visualizer...")

    # Create test trajectories
    for i in range(100):
        start_time = time.time()

        # Create test point cloud
        points = np.random.random((1000, 3)) * 2 - 1
        colors = np.random.random((1000, 3))

        test_pcd = o3d.geometry.PointCloud()
        test_pcd.points = o3d.utility.Vector3dVector(points)
        test_pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create test visual trajectory (circular)
        visual_poses = []
        for j in range(i + 1):
            pose = np.eye(4)
            angle = j * 0.1
            pose[:3, 3] = [np.cos(angle) * 0.5, np.sin(angle) * 0.5, 0]
            visual_poses.append(pose)

        # Create test IMU trajectory (slightly different)
        imu_poses = []
        for j in range(i + 1):
            pose = np.eye(4)
            angle = j * 0.1
            # Slightly different scale and offset
            pose[:3, 3] = [np.cos(angle) * 0.7 + 0.1, np.sin(angle) * 0.7, j * 0.01]
            imu_poses.append(pose)

        # Update visualization
        viz.update_map(test_pcd)
        viz.update_visual_trajectory(visual_poses)
        viz.update_imu_trajectory(imu_poses)
        viz.spin_once()

        frame_time = time.time() - start_time
        monitor.log_frame(frame_time, len(points))

        time.sleep(0.05)

    print("Test complete - close window to exit")
    while viz.spin_once():
        time.sleep(0.1)

    viz.close()


if __name__ == "__main__":
    main()