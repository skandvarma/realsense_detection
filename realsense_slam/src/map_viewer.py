import open3d as o3d
import numpy as np
import json
import sys
import os


class MapViewer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("SLAM Map Viewer", width=1024, height=768)

        # Setup rendering options
        opt = self.vis.get_render_option()
        opt.point_size = 2
        opt.background_color = np.array([0, 0, 0])

    def setup_camera_view(self):
        """Set camera view to focus on origin (0,0,0)"""
        ctr = self.vis.get_view_control()

        # Set the camera to look at the origin
        ctr.set_lookat([0, 0, 0])  # Look at origin
        ctr.set_up([0, 0, 1])  # Z-axis is up
        ctr.set_front([1, 1, 1])  # Camera position direction
        ctr.set_zoom(0.3)  # Zoom level (smaller = more zoomed out)

        print("Camera view centered at origin (0,0,0)")

    def load_and_view_session(self, session_path):
        print(f"Loading session: {session_path}")

        # Load point cloud map
        map_file = f"{session_path}_slam.ply"
        trajectory_file = f"{session_path}_trajectory.json"

        if not os.path.exists(map_file):
            print(f"Map file not found: {map_file}")
            return

        # Load point cloud
        pcd = o3d.io.read_point_cloud(map_file)
        print(f"Loaded point cloud with {len(pcd.points)} points")

        # Add prominent coordinate frame at origin
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        coord_frame.paint_uniform_color([1, 1, 1])  # White frame for visibility
        self.vis.add_geometry(coord_frame)

        # Add point cloud
        if len(pcd.points) > 0:
            self.vis.add_geometry(pcd)
        else:
            print("Warning: Point cloud is empty!")

        # Load and display trajectory if available
        visual_poses = None
        imu_poses = None

        if os.path.exists(trajectory_file):
            with open(trajectory_file, 'r') as f:
                traj_data = json.load(f)

            # Handle both old and new trajectory formats
            # New format (enhanced SLAM)
            if "visual_poses" in traj_data:
                visual_poses = [np.array(pose) for pose in traj_data["visual_poses"]]
                print(f"Loaded visual trajectory with {len(visual_poses)} poses")

            if "imu_poses" in traj_data:
                imu_poses = [np.array(pose) for pose in traj_data["imu_poses"]]
                print(f"Loaded IMU trajectory with {len(imu_poses)} poses")

            # Old format (legacy SLAM)
            elif "poses" in traj_data:
                visual_poses = [np.array(pose) for pose in traj_data["poses"]]
                print(f"Loaded legacy trajectory with {len(visual_poses)} poses")

            # Display visual trajectory (GREEN)
            if visual_poses and len(visual_poses) > 1:
                self.add_trajectory(visual_poses, color=[0, 1, 0], name="Visual SLAM", line_width=3)

            # Display IMU trajectory (RED)
            if imu_poses and len(imu_poses) > 1:
                self.add_trajectory(imu_poses, color=[1, 0, 0], name="IMU", line_width=3)

            # Print trajectory information
            if "scale_factor" in traj_data:
                print(f"Scale factor: {traj_data['scale_factor']:.3f}")

            if "motion_stats" in traj_data:
                stats = traj_data["motion_stats"]
                if "motion_rate" in stats:
                    print(f"Motion detection rate: {stats['motion_rate']:.1%}")

        # Add origin marker (large sphere at 0,0,0)
        origin_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.0005)
        origin_marker.paint_uniform_color([1, 1, 0])  # Yellow sphere at origin
        origin_marker.translate([0, 0, 0])
        self.vis.add_geometry(origin_marker)

        # Set camera view to origin AFTER adding all geometry
        self.setup_camera_view()

        # Run visualization
        print("\n" + "=" * 50)
        print("SLAM Map Viewer - Centered at Origin")
        print("=" * 50)
        print("View Controls:")
        print("- Mouse: Rotate view around origin")
        print("- Mouse wheel: Zoom in/out")
        print("- Ctrl+Mouse: Pan view")
        print("- 'R' key: Reset view to origin")
        print("- Close window to exit")
        print()
        print("Legend:")
        if visual_poses:
            print("- GREEN trajectory: Visual SLAM")
        if imu_poses:
            print("- RED trajectory: IMU")
        print("- WHITE axes: World coordinate frame")
        print("- YELLOW sphere: Origin (0,0,0)")
        print("=" * 50)

        # Custom event loop to handle 'R' key for reset
        self.run_with_controls()

    def run_with_controls(self):
        """Run visualization with custom controls"""

        def key_callback(vis, key, action, mods):
            if key == ord('R') or key == ord('r'):
                print("Resetting view to origin...")
                self.setup_camera_view()
                return True
            return False

        # Run the visualizer
        self.vis.run()
        self.vis.destroy_window()

    def add_trajectory(self, poses, color=[1, 0, 0], name="Trajectory", line_width=2):
        """Add trajectory visualization to the scene"""
        # Create trajectory line
        positions = [pose[:3, 3] for pose in poses]
        points = o3d.utility.Vector3dVector(positions)
        lines = [[i, i + 1] for i in range(len(positions) - 1)]

        trajectory_lines = o3d.geometry.LineSet()
        trajectory_lines.points = points
        trajectory_lines.lines = o3d.utility.Vector2iVector(lines)

        # Set trajectory color
        colors = [color for _ in range(len(lines))]
        trajectory_lines.colors = o3d.utility.Vector3dVector(colors)

        self.vis.add_geometry(trajectory_lines)

        # Add camera poses as small coordinate frames (every 10th pose to avoid clutter)
        pose_interval = max(1, len(poses) // 20)  # Show max 20 pose frames
        for i, pose in enumerate(poses[::pose_interval]):
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            frame.transform(pose)
            self.vis.add_geometry(frame)

        # Add start and end markers
        if len(positions) > 0:
            # Start marker (larger, green sphere)
            start_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            start_marker.paint_uniform_color([0, 1, 0])  # Green for start
            start_marker.translate(positions[0])
            self.vis.add_geometry(start_marker)

            # End marker (larger, red sphere)
            if len(positions) > 1:
                end_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                end_marker.paint_uniform_color([1, 0, 0])  # Red for end
                end_marker.translate(positions[-1])
                self.vis.add_geometry(end_marker)

    def print_trajectory_analysis(self, visual_poses, imu_poses):
        """Print detailed trajectory analysis"""
        print("\n" + "=" * 40)
        print("TRAJECTORY ANALYSIS")
        print("=" * 40)

        if visual_poses:
            visual_start = visual_poses[0][:3, 3]
            visual_end = visual_poses[-1][:3, 3]
            visual_distance = np.linalg.norm(visual_end - visual_start)
            print(f"Visual SLAM (GREEN):")
            print(f"  Start: [{visual_start[0]:.3f}, {visual_start[1]:.3f}, {visual_start[2]:.3f}]")
            print(f"  End:   [{visual_end[0]:.3f}, {visual_end[1]:.3f}, {visual_end[2]:.3f}]")
            print(f"  Total distance: {visual_distance:.3f}m")

        if imu_poses:
            imu_start = imu_poses[0][:3, 3]
            imu_end = imu_poses[-1][:3, 3]
            imu_distance = np.linalg.norm(imu_end - imu_start)
            print(f"IMU Trajectory (RED):")
            print(f"  Start: [{imu_start[0]:.3f}, {imu_start[1]:.3f}, {imu_start[2]:.3f}]")
            print(f"  End:   [{imu_end[0]:.3f}, {imu_end[1]:.3f}, {imu_end[2]:.3f}]")
            print(f"  Total distance: {imu_distance:.3f}m")

        if visual_poses and imu_poses:
            # Direction analysis
            visual_direction = visual_end - visual_start
            imu_direction = imu_end - imu_start

            # Normalize directions
            visual_dir_norm = visual_direction / np.linalg.norm(visual_direction) if np.linalg.norm(
                visual_direction) > 0 else np.zeros(3)
            imu_dir_norm = imu_direction / np.linalg.norm(imu_direction) if np.linalg.norm(
                imu_direction) > 0 else np.zeros(3)

            # Calculate alignment
            dot_product = np.dot(visual_dir_norm, imu_dir_norm)
            angle_degrees = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))

            print(f"Trajectory Alignment:")
            print(f"  Direction similarity: {dot_product:.3f}")
            print(f"  Angle between trajectories: {angle_degrees:.1f}°")

            if dot_product > 0.8:
                print(f"  Status:  GOOD alignment")
            elif dot_product > 0.3:
                print(f"  Status:  FAIR alignment")
            elif dot_product > -0.3:
                print(f"  Status:  PERPENDICULAR trajectories")
            else:
                print(f"  Status:  OPPOSITE directions - needs coordinate fix")


def main():
    if len(sys.argv) != 2:
        print("Usage: python map_viewer.py <session_path_without_extension>")
        print("Example: python map_viewer.py /path/to/my_session")
        return

    session_path = sys.argv[1]

    viewer = MapViewer()
    viewer.load_and_view_session(session_path)



if __name__ == "__main__":
    main()
