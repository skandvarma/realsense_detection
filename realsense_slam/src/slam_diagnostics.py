import open3d as o3d
import numpy as np
import cv2
import json
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime


class SLAMDiagnostics:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Create logs directory
        self.log_dir = "../logs"
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize logging
        self.log_file = f"{self.log_dir}/slam_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.performance_data = {
            'frame_times': [],
            'tracking_success': [],
            'translation_estimates': [],
            'rotation_estimates': [],
            'point_cloud_sizes': [],
            'odometry_errors': []
        }

    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        with open(self.log_file, 'a') as f:
            f.write(log_entry + "\n")

    def test_camera_calibration(self):
        """Test camera calibration quality"""
        self.log("=== CAMERA CALIBRATION TEST ===")

        from camera import D435iCamera
        camera = D435iCamera(self.config)

        try:
            # Get intrinsics
            intrinsics = camera.get_intrinsics()
            self.log(f"Camera intrinsics matrix:\n{intrinsics}")

            # Check if intrinsics are reasonable
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]

            self.log(f"Focal lengths: fx={fx:.2f}, fy={fy:.2f}")
            self.log(f"Principal point: cx={cx:.2f}, cy={cy:.2f}")

            # Test frame capture
            self.log("Testing frame capture...")
            frame_count = 0
            depth_stats = []

            for i in range(30):
                rgb, depth = camera.get_frames()
                if rgb is not None and depth is not None:
                    frame_count += 1
                    valid_depth = depth[depth > 0]
                    if len(valid_depth) > 0:
                        depth_stats.append({
                            'min': valid_depth.min(),
                            'max': valid_depth.max(),
                            'mean': valid_depth.mean(),
                            'valid_pixels': len(valid_depth)
                        })
                time.sleep(0.1)

            self.log(f"Successfully captured {frame_count}/30 frames")

            if depth_stats:
                avg_valid_pixels = np.mean([s['valid_pixels'] for s in depth_stats])
                avg_depth = np.mean([s['mean'] for s in depth_stats])
                self.log(f"Average valid depth pixels: {avg_valid_pixels:.0f}")
                self.log(f"Average depth: {avg_depth:.1f}mm")

                # Check depth quality
                total_pixels = self.config['camera']['width'] * self.config['camera']['height']
                depth_coverage = avg_valid_pixels / total_pixels * 100
                self.log(f"Depth coverage: {depth_coverage:.1f}%")

                if depth_coverage < 50:
                    self.log("WARNING: Low depth coverage - check lighting and surface textures", "WARN")
                if avg_depth < 300 or avg_depth > 3000:
                    self.log("WARNING: Unusual depth range - check camera positioning", "WARN")

        finally:
            camera.stop()

    def test_visual_odometry(self):
        """Test visual odometry performance"""
        self.log("=== VISUAL ODOMETRY TEST ===")

        from camera import D435iCamera
        camera = D435iCamera(self.config)

        try:
            intrinsics = camera.get_intrinsics()
            height, width = self.config['camera']['height'], self.config['camera']['width']

            # Create Open3D camera intrinsic
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width, height,
                intrinsics[0, 0], intrinsics[1, 1],
                intrinsics[0, 2], intrinsics[1, 2]
            )

            prev_rgbd = None
            successful_tracks = 0
            total_attempts = 0

            self.log("Testing odometry tracking (move camera slowly)...")

            for i in range(50):
                rgb, depth = camera.get_frames()
                if rgb is None or depth is None:
                    continue

                # Convert to Open3D format
                depth_meters = depth.astype(np.float32) / 1000.0
                color_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                depth_o3d = o3d.geometry.Image(depth_meters)

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d,
                    depth_scale=1.0, depth_trunc=self.config["slam"]["max_depth"],
                    convert_rgb_to_intensity=False
                )

                if prev_rgbd is not None:
                    start_time = time.time()

                    option = o3d.pipelines.odometry.OdometryOption()
                    success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                        prev_rgbd, rgbd, intrinsic, np.eye(4),
                        o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option
                    )

                    processing_time = time.time() - start_time
                    self.performance_data['frame_times'].append(processing_time)

                    total_attempts += 1
                    if success:
                        successful_tracks += 1

                        # Analyze transformation
                        translation = np.linalg.norm(trans[:3, 3])
                        rotation_angle = np.arccos(np.clip((np.trace(trans[:3, :3]) - 1) / 2, -1, 1))

                        self.performance_data['translation_estimates'].append(translation)
                        self.performance_data['rotation_estimates'].append(rotation_angle)

                        self.log(
                            f"Frame {i}: SUCCESS - trans: {translation:.4f}m, rot: {np.degrees(rotation_angle):.2f}deg, time: {processing_time:.3f}s")
                    else:
                        self.log(f"Frame {i}: FAILED - odometry tracking failed", "WARN")

                    self.performance_data['tracking_success'].append(success)

                prev_rgbd = rgbd
                time.sleep(0.1)

            # Compute statistics
            if total_attempts > 0:
                success_rate = successful_tracks / total_attempts * 100
                avg_processing_time = np.mean(self.performance_data['frame_times'])

                self.log(f"Tracking success rate: {success_rate:.1f}% ({successful_tracks}/{total_attempts})")
                self.log(f"Average processing time: {avg_processing_time:.3f}s")

                if success_rate < 70:
                    self.log("WARNING: Low tracking success rate - consider better lighting or slower movement", "WARN")
                if avg_processing_time > 0.1:
                    self.log("WARNING: Slow processing - may cause real-time issues", "WARN")

        finally:
            camera.stop()

    def test_point_cloud_generation(self):
        """Test point cloud generation quality"""
        self.log("=== POINT CLOUD GENERATION TEST ===")

        from camera import D435iCamera
        camera = D435iCamera(self.config)

        try:
            intrinsics = camera.get_intrinsics()
            height, width = self.config['camera']['height'], self.config['camera']['width']

            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width, height,
                intrinsics[0, 0], intrinsics[1, 1],
                intrinsics[0, 2], intrinsics[1, 2]
            )

            self.log("Generating point clouds from camera frames...")

            for i in range(10):
                rgb, depth = camera.get_frames()
                if rgb is None or depth is None:
                    continue

                depth_meters = depth.astype(np.float32) / 1000.0
                color_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                depth_o3d = o3d.geometry.Image(depth_meters)

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d, depth_trunc=self.config["slam"]["max_depth"]
                )

                # Generate point cloud
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

                num_points = len(pcd.points)
                self.performance_data['point_cloud_sizes'].append(num_points)

                self.log(f"Frame {i}: Generated {num_points} points")

                if num_points < 1000:
                    self.log(f"WARNING: Very few points generated in frame {i}", "WARN")

                time.sleep(0.2)

            if self.performance_data['point_cloud_sizes']:
                avg_points = np.mean(self.performance_data['point_cloud_sizes'])
                self.log(f"Average points per frame: {avg_points:.0f}")

                if avg_points < 5000:
                    self.log("WARNING: Low point density - check depth camera quality and scene texture", "WARN")

        finally:
            camera.stop()

    def run_full_slam_test(self, duration_seconds=30):
        """Run complete SLAM test with logging"""
        self.log("=== FULL SLAM PERFORMANCE TEST ===")

        from camera import D435iCamera
        from slam import SimpleSLAM

        camera = D435iCamera(self.config)
        intrinsics = camera.get_intrinsics()
        slam = SimpleSLAM(intrinsics, self.config)

        start_time = time.time()
        frame_count = 0
        keyframe_count = 0

        try:
            self.log(f"Running SLAM for {duration_seconds} seconds...")
            self.log("Move the camera around to test mapping performance")

            while time.time() - start_time < duration_seconds:
                rgb, depth = camera.get_frames()
                if rgb is None or depth is None:
                    continue

                frame_start = time.time()

                # Process frame
                depth_meters = depth.astype(np.float32) / 1000.0
                slam.process_frame(rgb, depth_meters)

                frame_time = time.time() - frame_start
                frame_count += 1

                # Log progress
                if frame_count % 30 == 0:
                    trajectory_length = len(slam.get_trajectory())
                    map_points = len(slam.get_map().points)

                    self.log(
                        f"Frame {frame_count}: trajectory={trajectory_length} poses, map={map_points} points, frame_time={frame_time:.3f}s")

                    if map_points == 0:
                        self.log("WARNING: No map points generated yet", "WARN")

                time.sleep(0.033)  # ~30 FPS

            # Final statistics
            final_trajectory = slam.get_trajectory()
            final_map = slam.get_map()

            self.log(f"SLAM TEST COMPLETED:")
            self.log(f"- Total frames processed: {frame_count}")
            self.log(f"- Final trajectory length: {len(final_trajectory)} poses")
            self.log(f"- Final map size: {len(final_map.points)} points")
            self.log(f"- Average FPS: {frame_count / duration_seconds:.1f}")

            # Save test results
            test_session = f"diagnostic_test_{datetime.now().strftime('%H%M%S')}"
            os.makedirs("../data/sessions", exist_ok=True)
            slam.save_session(f"../data/sessions/{test_session}")
            self.log(f"Test results saved as: {test_session}")

            return len(final_map.points) > 0, len(final_trajectory) > 1

        finally:
            camera.stop()

    def generate_report(self):
        """Generate diagnostic report"""
        self.log("=== DIAGNOSTIC REPORT ===")

        if self.performance_data['tracking_success']:
            success_rate = np.mean(self.performance_data['tracking_success']) * 100
            self.log(f"Overall tracking success rate: {success_rate:.1f}%")

        if self.performance_data['frame_times']:
            avg_time = np.mean(self.performance_data['frame_times'])
            max_time = np.max(self.performance_data['frame_times'])
            self.log(f"Processing time - avg: {avg_time:.3f}s, max: {max_time:.3f}s")

        self.log("\nRECOMMENDations:")

        if self.performance_data['tracking_success']:
            if np.mean(self.performance_data['tracking_success']) < 0.7:
                self.log("- Improve lighting conditions")
                self.log("- Move camera more slowly")
                self.log("- Ensure scene has sufficient texture")

        if self.performance_data['point_cloud_sizes']:
            if np.mean(self.performance_data['point_cloud_sizes']) < 5000:
                self.log("- Check depth camera alignment")
                self.log("- Ensure proper distance from objects (0.3-3m)")
                self.log("- Verify depth camera is not blocked")

        self.log(f"\nFull diagnostic log saved to: {self.log_file}")


def main():
    print("SLAM Diagnostic Tool")
    print("===================")

    config_path = "../config/config.json"
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    diagnostics = SLAMDiagnostics(config_path)

    print("Select test to run:")
    print("1. Camera calibration test")
    print("2. Visual odometry test")
    print("3. Point cloud generation test")
    print("4. Full SLAM performance test")
    print("5. Run all tests")

    choice = input("Enter choice (1-5): ").strip()

    if choice == "1":
        diagnostics.test_camera_calibration()
    elif choice == "2":
        diagnostics.test_visual_odometry()
    elif choice == "3":
        diagnostics.test_point_cloud_generation()
    elif choice == "4":
        success_map, success_traj = diagnostics.run_full_slam_test(30)
        if not success_map:
            print("WARNING: SLAM failed to generate map points")
        if not success_traj:
            print("WARNING: SLAM failed to track trajectory")
    elif choice == "5":
        diagnostics.test_camera_calibration()
        diagnostics.test_visual_odometry()
        diagnostics.test_point_cloud_generation()
        diagnostics.run_full_slam_test(30)
    else:
        print("Invalid choice")
        return

    diagnostics.generate_report()


if __name__ == "__main__":
    main()