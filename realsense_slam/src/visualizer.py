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
        self.vis.create_window("Minimal SLAM", width=800, height=600)

        # Setup rendering
        opt = self.vis.get_render_option()
        opt.point_size = self.params['point_size']
        opt.background_color = np.array(config.get('viz', {}).get('background', [0, 0, 0]))

        # Single geometry for map
        self.map_pcd = o3d.geometry.PointCloud()
        self.map_added = False

        # Optional trajectory line
        self.traj_line = None
        self.traj_added = False

        # Frame counter
        self.frame_count = 0

        print(f"Minimal Visualizer initialized with params: {self.params}")

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
            # Simple random downsampling for visualization
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
            # Direct point and color update - fastest method
            self.map_pcd.points = viz_pcd.points
            if len(viz_pcd.colors) > 0:
                self.map_pcd.colors = viz_pcd.colors
            self.vis.update_geometry(self.map_pcd)

    def update_trajectory(self, poses):
        """Update trajectory with minimal line visualization"""
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
        if not self.traj_added:
            self.traj_line = o3d.geometry.LineSet()
            self.vis.add_geometry(self.traj_line)
            self.traj_added = True

        # Simple line creation
        points = o3d.utility.Vector3dVector(positions)
        lines = [[i, i + 1] for i in range(len(positions) - 1)]

        self.traj_line.points = points
        self.traj_line.lines = o3d.utility.Vector2iVector(lines)

        # Single color for all lines - faster than individual colors
        colors = [[1, 0, 0]] * len(lines)  # Red trajectory
        self.traj_line.colors = o3d.utility.Vector3dVector(colors)

        self.vis.update_geometry(self.traj_line)

    def show_frame(self, rgb_image):
        """Show current RGB frame with minimal processing"""
        if rgb_image is not None:
            # Resize for performance
            display_img = cv2.resize(rgb_image, (320, 240))
            cv2.imshow("Current View", display_img)
            cv2.waitKey(1)

    def spin_once(self):
        """Update visualization once - minimal operations"""
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


class PerformanceMonitor:
    """Simple performance monitoring for tuning"""

    def __init__(self):
        self.frame_times = []
        self.map_sizes = []
        self.last_print = 0

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
                'voxel_size': 0.08,  # Increase for fewer points
                'max_points': 15000,  # Reduce map size
                'update_every_n': 3,  # Update less frequently
                'viz_points_limit': 10000  # Reduce viz points
            })
            print("Performance suggestions: Increase voxel_size, reduce max_points")

        elif avg_time < 0.02:  # Faster than 50 FPS
            suggestions.update({
                'voxel_size': 0.03,  # Decrease for more detail
                'max_points': 30000,  # Increase map size
                'update_every_n': 1,  # Update every frame
                'viz_points_limit': 20000  # More viz points
            })
            print("Performance suggestions: Decrease voxel_size, increase max_points")

        return suggestions


def main():
    """Test minimal visualizer"""
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

    print("Testing minimal visualizer with synthetic data...")

    # Create test trajectory and point clouds
    for i in range(100):
        start_time = time.time()

        # Create test point cloud
        points = np.random.random((1000, 3)) * 2 - 1  # Random points in [-1,1]^3
        colors = np.random.random((1000, 3))

        test_pcd = o3d.geometry.PointCloud()
        test_pcd.points = o3d.utility.Vector3dVector(points)
        test_pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create test trajectory
        poses = []
        for j in range(i + 1):
            pose = np.eye(4)
            pose[:3, 3] = [j * 0.1, 0, 0]  # Simple linear trajectory
            poses.append(pose)

        # Update visualization
        viz.update_map(test_pcd)
        viz.update_trajectory(poses)
        viz.spin_once()

        frame_time = time.time() - start_time
        monitor.log_frame(frame_time, len(points))

        # Auto-tune parameters based on performance
        if i == 60:
            suggestions = monitor.suggest_params()
            if suggestions:
                viz.update_params(**suggestions)

        time.sleep(0.01)  # Small delay

    print("Test complete - close window to exit")
    while viz.spin_once():
        time.sleep(0.1)

    viz.close()


if __name__ == "__main__":
    main()