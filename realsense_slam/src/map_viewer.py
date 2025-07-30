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

    def load_and_view_session(self, session_path):
        print(f"Loading session: {session_path}")

        # Load point cloud map
        map_file = f"{session_path}_map.ply"
        trajectory_file = f"{session_path}_trajectory.json"

        if not os.path.exists(map_file):
            print(f"Map file not found: {map_file}")
            return

        # Load point cloud
        pcd = o3d.io.read_point_cloud(map_file)
        print(f"Loaded point cloud with {len(pcd.points)} points")

        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(coord_frame)

        # Add point cloud
        if len(pcd.points) > 0:
            self.vis.add_geometry(pcd)
        else:
            print("Warning: Point cloud is empty!")

        # Load and display trajectory if available
        if os.path.exists(trajectory_file):
            with open(trajectory_file, 'r') as f:
                traj_data = json.load(f)

            poses = [np.array(pose) for pose in traj_data["poses"]]
            print(f"Loaded trajectory with {len(poses)} poses")

            if len(poses) > 1:
                # Create trajectory line
                positions = [pose[:3, 3] for pose in poses]
                points = o3d.utility.Vector3dVector(positions)
                lines = [[i, i + 1] for i in range(len(positions) - 1)]

                trajectory_lines = o3d.geometry.LineSet()
                trajectory_lines.points = points
                trajectory_lines.lines = o3d.utility.Vector2iVector(lines)

                # Set trajectory color (red)
                colors = [[1, 0, 0] for _ in range(len(lines))]
                trajectory_lines.colors = o3d.utility.Vector3dVector(colors)

                self.vis.add_geometry(trajectory_lines)

                # Add camera poses as small coordinate frames
                for i, pose in enumerate(poses[::5]):  # Show every 5th pose
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                    frame.transform(pose)
                    self.vis.add_geometry(frame)

        # Run visualization
        print("Map viewer opened. Close window to exit.")
        print("Controls:")
        print("- Mouse: Rotate view")
        print("- Mouse wheel: Zoom")
        print("- Ctrl+Mouse: Pan")

        self.vis.run()
        self.vis.destroy_window()


def main():
    if len(sys.argv) != 2:
        print("Usage: python map_viewer.py <session_name>")
        print("Example: python map_viewer.py live_mapping")
        print("\nAvailable sessions:")
        session_dir = "../data/sessions"
        if os.path.exists(session_dir):
            sessions = set()
            for file in os.listdir(session_dir):
                if file.endswith("_map.ply"):
                    session_name = file.replace("_map.ply", "")
                    sessions.add(session_name)
            for session in sorted(sessions):
                print(f"  - {session}")
        return

    session_name = sys.argv[1]
    session_path = f"../data/sessions/{session_name}"

    viewer = MapViewer()
    viewer.load_and_view_session(session_path)


if __name__ == "__main__":
    main()


# Alternative: Quick view function for Jupyter notebooks or direct use
def quick_view_map(session_name):
    """Quick function to view a saved map"""
    session_path = f"../data/sessions/{session_name}"
    map_file = f"{session_path}_map.ply"

    if not os.path.exists(map_file):
        print(f"Map file not found: {map_file}")
        return

    # Load and visualize
    pcd = o3d.io.read_point_cloud(map_file)
    print(f"Viewing map with {len(pcd.points)} points")

    if len(pcd.points) > 0:
        # Add coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, coord_frame],
                                          window_name=f"SLAM Map: {session_name}",
                                          width=1024, height=768)
    else:
        print("Point cloud is empty!")

# You can also use this one-liner to quickly view a map:
# python -c "import open3d as o3d; pcd = o3d.io.read_point_cloud('../data/sessions/live_mapping_map.ply'); print(f'Points: {len(pcd.points)}'); o3d.visualization.draw_geometries([pcd]) if len(pcd.points) > 0 else print('Empty map')"