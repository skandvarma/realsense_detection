import open3d as o3d
import numpy as np
import cv2


class IMUVisualizer:
    def __init__(self):
        # Initialize visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("IMU-only SLAM", width=1000, height=700)

        # Setup rendering options
        opt = self.vis.get_render_option()
        opt.point_size = 4
        opt.background_color = np.array([0.05, 0.05, 0.05])

        # Create coordinate frames
        self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        self.device_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)

        # Create trajectory elements
        self.trajectory_lines = o3d.geometry.LineSet()
        self.trajectory_points = o3d.geometry.PointCloud()

        # Create pose history (keyframe poses)
        self.pose_frames = []
        self.max_pose_frames = 20

        # Create velocity vector
        self.velocity_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01, cone_radius=0.02,
            cylinder_height=0.1, cone_height=0.05
        )
        self.velocity_arrow.paint_uniform_color([1, 1, 0])  # Yellow arrow

        # Add geometries
        self.vis.add_geometry(self.world_frame)
        self.vis.add_geometry(self.device_frame)
        self.vis.add_geometry(self.trajectory_lines)
        self.vis.add_geometry(self.trajectory_points)
        self.vis.add_geometry(self.velocity_arrow)

        # Setup camera view
        self.setup_camera_view()

        print("Enhanced IMU Visualizer initialized")

    def setup_camera_view(self):
        """Setup initial camera view"""
        ctr = self.vis.get_view_control()
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_front([1, 1, 1])
        ctr.set_zoom(0.7)

    def update_trajectory(self, trajectory):
        """Update enhanced trajectory visualization"""
        if len(trajectory) < 2:
            return

        # Extract positions
        positions = [pose[:3, 3] for pose in trajectory]

        # Update trajectory points (map-like points)
        self.trajectory_points.points = o3d.utility.Vector3dVector(positions)

        # Create point colors based on time (blue to red gradient)
        colors = []
        for i in range(len(positions)):
            ratio = i / max(1, len(positions) - 1)
            color = [ratio, 0.3, 1 - ratio]  # Blue to red progression
            colors.append(color)

        self.trajectory_points.colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(self.trajectory_points)

        # Create trajectory lines
        lines = [[i, i + 1] for i in range(len(positions) - 1)]
        self.trajectory_lines.points = o3d.utility.Vector3dVector(positions)
        self.trajectory_lines.lines = o3d.utility.Vector2iVector(lines)

        # Line colors (white/gray for path)
        line_colors = [[0.8, 0.8, 0.8] for _ in range(len(lines))]
        self.trajectory_lines.colors = o3d.utility.Vector3dVector(line_colors)
        self.vis.update_geometry(self.trajectory_lines)

        # Add pose keyframes at intervals
        self.update_pose_history(trajectory)

    def update_pose_history(self, trajectory):
        """Add coordinate frames at key poses"""
        if len(trajectory) < 5:
            return

        # Add pose frame every 10 poses
        current_keyframe_count = len(trajectory) // 10

        while len(self.pose_frames) < current_keyframe_count and len(self.pose_frames) < self.max_pose_frames:
            pose_idx = len(self.pose_frames) * 10
            if pose_idx < len(trajectory):
                pose = trajectory[pose_idx]

                # Create small coordinate frame for this pose
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
                frame.transform(pose)

                self.pose_frames.append(frame)
                self.vis.add_geometry(frame, reset_bounding_box=False)

        # Remove old frames if too many
        while len(self.pose_frames) > self.max_pose_frames:
            old_frame = self.pose_frames.pop(0)
            self.vis.remove_geometry(old_frame, reset_bounding_box=False)

    def update_device_pose(self, pose, velocity=None):
        """Update device pose and velocity visualization"""
        # Remove old device frame
        self.vis.remove_geometry(self.device_frame, reset_bounding_box=False)

        # Create new device frame
        self.device_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        self.device_frame.transform(pose)

        # Add updated device frame
        self.vis.add_geometry(self.device_frame, reset_bounding_box=False)

        # Update velocity arrow if velocity provided
        if velocity is not None:
            self.update_velocity_arrow(pose[:3, 3], velocity)

    def update_velocity_arrow(self, position, velocity):
        """Update velocity vector visualization"""
        velocity_magnitude = np.linalg.norm(velocity)

        if velocity_magnitude > 0.01:  # Only show if moving
            # Remove old arrow
            self.vis.remove_geometry(self.velocity_arrow, reset_bounding_box=False)

            # Create new arrow
            arrow_length = min(velocity_magnitude * 10, 0.3)  # Scale and limit arrow length
            self.velocity_arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.005, cone_radius=0.01,
                cylinder_height=arrow_length * 0.8, cone_height=arrow_length * 0.2
            )
            self.velocity_arrow.paint_uniform_color([1, 1, 0])  # Yellow

            # Orient arrow in velocity direction
            if velocity_magnitude > 0:
                velocity_unit = velocity / velocity_magnitude

                # Create rotation matrix to align with velocity
                z_axis = np.array([0, 0, 1])
                if abs(np.dot(velocity_unit, z_axis)) < 0.99:
                    rotation_axis = np.cross(z_axis, velocity_unit)
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    rotation_angle = np.arccos(np.dot(z_axis, velocity_unit))

                    # Rodrigues rotation formula
                    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                                  [rotation_axis[2], 0, -rotation_axis[0]],
                                  [-rotation_axis[1], rotation_axis[0], 0]])

                    R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)

                    # Apply transformation
                    transform = np.eye(4)
                    transform[:3, :3] = R
                    transform[:3, 3] = position

                    self.velocity_arrow.transform(transform)
                else:
                    # Simple translation if aligned with Z
                    transform = np.eye(4)
                    transform[:3, 3] = position
                    self.velocity_arrow.transform(transform)

            # Add updated arrow
            self.vis.add_geometry(self.velocity_arrow, reset_bounding_box=False)

    def show_status(self, frame_count, trajectory_length, bias_estimated, position=None):
        """Display detailed status information"""
        status_parts = [
            f"Frame: {frame_count}",
            f"Trajectory: {trajectory_length} poses",
            f"Keyframes: {len(self.pose_frames)}"
        ]

        if position is not None:
            pos_str = f"Pos: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]"
            status_parts.append(pos_str)

        if not bias_estimated:
            status_parts.append("CALIBRATING")
        else:
            status_parts.append("TRACKING")

        status_text = " | ".join(status_parts)

        # Print status every second
        if frame_count % 30 == 0:
            print(status_text)

    def spin_once(self):
        """Update visualization once"""
        self.vis.poll_events()
        self.vis.update_renderer()
        return True

    def close(self):
        """Close visualizer"""
        self.vis.destroy_window()
        cv2.destroyAllWindows()


def main():
    """Test IMU visualizer"""
    import time

    viz = IMUVisualizer()

    # Create test trajectory (circular motion)
    trajectory = []
    for i in range(100):
        angle = i * 0.1
        pose = np.eye(4)
        pose[0, 3] = np.cos(angle) * 0.5
        pose[1, 3] = np.sin(angle) * 0.5
        pose[2, 3] = i * 0.01
        trajectory.append(pose)

        viz.update_trajectory(trajectory)
        viz.update_device_pose(pose)
        viz.spin_once()
        time.sleep(0.05)

    print("Test complete - close window to exit")
    while viz.spin_once():
        time.sleep(0.1)

    viz.close()


if __name__ == "__main__":
    main()