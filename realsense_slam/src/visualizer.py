import open3d as o3d
import numpy as np
import cv2
import time
import json


class SLAMVisualizer:
    def __init__(self, config):
        self.config = config

        # Initialize visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("SLAM Visualization", width=1024, height=768)

        # Setup rendering options
        opt = self.vis.get_render_option()
        opt.point_size = config["viz"]["point_size"]
        opt.background_color = np.array(config["viz"]["background"])

        # Initialize geometries
        self.pcd = o3d.geometry.PointCloud()
        self.trajectory_lines = o3d.geometry.LineSet()

        # Add coordinate frame
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(self.coord_frame)

        self.geometries_added = False
        self.max_viz_points = 50000  # Limit visualization points for performance

        # Calibration status display
        self.calibration_text_shown = False

    def update_map(self, point_cloud):
        if point_cloud is None or len(point_cloud.points) == 0:
            return

        # Downsample for visualization if too large
        viz_pcd = point_cloud
        if len(point_cloud.points) > self.max_viz_points:
            # Random downsampling for visualization
            indices = np.random.choice(len(point_cloud.points), self.max_viz_points, replace=False)
            viz_pcd = point_cloud.select_by_index(indices)

        if not self.geometries_added:
            self.pcd = viz_pcd
            self.vis.add_geometry(self.pcd)
            self.geometries_added = True
        else:
            self.pcd.points = viz_pcd.points
            self.pcd.colors = viz_pcd.colors
            self.vis.update_geometry(self.pcd)

    def update_trajectory(self, poses):
        if len(poses) < 2:
            return

        # Extract positions
        positions = [pose[:3, 3] for pose in poses]

        # Create line set
        points = o3d.utility.Vector3dVector(positions)
        lines = [[i, i + 1] for i in range(len(positions) - 1)]

        if not hasattr(self, 'trajectory_added'):
            self.trajectory_lines.points = points
            self.trajectory_lines.lines = o3d.utility.Vector2iVector(lines)
            # Set trajectory color (red)
            colors = [[1, 0, 0] for _ in range(len(lines))]
            self.trajectory_lines.colors = o3d.utility.Vector3dVector(colors)
            self.vis.add_geometry(self.trajectory_lines)
            self.trajectory_added = True
        else:
            self.trajectory_lines.points = points
            self.trajectory_lines.lines = o3d.utility.Vector2iVector(lines)
            colors = [[1, 0, 0] for _ in range(len(lines))]
            self.trajectory_lines.colors = o3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.trajectory_lines)

    def show_frame(self, rgb_image, calibration_status=None):
        if rgb_image is not None:
            # Resize for faster display
            display_image = cv2.resize(rgb_image, (320, 240))

            # Add calibration status overlay if provided
            if calibration_status is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                color = (0, 255, 0) if calibration_status.get('complete', False) else (0, 165,
                                                                                       255)  # Green if complete, orange if not
                thickness = 2

                if not calibration_status.get('complete', False):
                    # Show calibration progress
                    progress = calibration_status.get('progress', 0)
                    text = f"IMU Calibration: {progress}/100"
                    text2 = "Keep device STATIONARY"

                    cv2.putText(display_image, text, (10, 30), font, font_scale, color, thickness)
                    cv2.putText(display_image, text2, (10, 55), font, font_scale, color, thickness)

                    # Draw progress bar
                    bar_width = 200
                    bar_height = 10
                    bar_x = 10
                    bar_y = 70

                    # Background
                    cv2.rectangle(display_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50),
                                  -1)
                    # Progress
                    progress_width = int((progress / 100.0) * bar_width)
                    cv2.rectangle(display_image, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), color,
                                  -1)
                else:
                    # Show calibration complete
                    text = "IMU Calibrated - Ready!"
                    cv2.putText(display_image, text, (10, 30), font, font_scale, color, thickness)

            cv2.imshow("Current Frame", display_image)
            cv2.waitKey(1)

    def spin_once(self):
        # Fixed: Don't use return value to determine continuation
        self.vis.poll_events()
        self.vis.update_renderer()
        return True  # Always continue unless manually stopped

    def close(self):
        self.vis.destroy_window()
        cv2.destroyAllWindows()


def main():
    # Test 3D visualization with real D435i camera data
    with open('../config/config.json', 'r') as f:
        config = json.load(f)

    # Import and initialize camera
    from camera import D435iCamera

    camera = D435iCamera(config)
    viz = SLAMVisualizer(config)

    print("Testing visualization with real D435i camera...")
    print("Close the 3D window or press Ctrl+C to exit")

    try:
        # Get camera intrinsics for point cloud generation
        intrinsics = camera.get_intrinsics()
        height, width = config["camera"]["height"], config["camera"]["width"]

        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width, height,
            intrinsics[0, 0], intrinsics[1, 1],
            intrinsics[0, 2], intrinsics[1, 2]
        )

        frame_count = 0
        while True:
            # Get real camera frames
            rgb, depth = camera.get_frames()
            if rgb is None or depth is None:
                continue

            # Convert depth to meters and create point cloud
            depth_meters = depth.astype(np.float32) / 1000.0
            color_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            depth_o3d = o3d.geometry.Image(depth_meters)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_trunc=config["slam"]["max_depth"]
            )

            # Generate point cloud from current frame
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d)

            # Update visualization
            viz.update_map(pcd)
            viz.show_frame(rgb)

            # Check if visualization should continue
            viz.spin_once()

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")

            time.sleep(0.033)  # ~30 FPS

    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.stop()
        viz.close()


if __name__ == "__main__":
    main()