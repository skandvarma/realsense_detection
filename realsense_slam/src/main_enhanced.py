import json
import os
import time
import cv2
import numpy as np
from camera import D435iCamera
from enhanced_slam import EnhancedMinimalSLAM
from visualizer import MinimalVisualizer, PerformanceMonitor


class EnhancedSLAMSystem:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.camera = None
        self.slam = None
        self.visualizer = None
        self.monitor = PerformanceMonitor()
        self.running = False

    def run_enhanced_slam(self, session_name):
        print("=== Enhanced SLAM with Trajectory Alignment ===")
        print(f"Session: {session_name}")
        print("Features:")
        print("- Visual motion detection + IMU drift filtering")
        print("- Automatic scale alignment between trajectories")
        print("- GREEN line: Visual SLAM trajectory")
        print("- RED line: IMU trajectory")
        print("- Real-time scale factor estimation")
        print("Controls: Press 'q' to quit")
        print("=" * 50)

        # Initialize components
        self.camera = D435iCamera(self.config)
        intrinsics = self.camera.get_intrinsics()
        self.slam = EnhancedMinimalSLAM(intrinsics, self.config)
        self.visualizer = MinimalVisualizer(self.config)

        self.running = True
        frame_count = 0
        start_time = time.time()
        last_stats_time = time.time()

        try:
            while self.running:
                frame_start = time.time()

                # Get camera frames and IMU
                rgb, depth = self.camera.get_frames()
                accel, gyro = self.camera.get_imu_data()

                if rgb is None or depth is None:
                    continue

                # Process enhanced SLAM with motion detection and alignment
                depth_meters = depth.astype(np.float32) / 1000.0
                self.slam.process_frame(rgb, depth_meters, accel, gyro)

                # Get both trajectories
                map_cloud = self.slam.get_map()
                visual_trajectory = self.slam.get_trajectory()  # Returns aligned visual trajectory
                imu_trajectory = self.slam.get_imu_trajectory()  # Returns IMU trajectory

                # Update visualization with both trajectories
                self.visualizer.update_map(map_cloud)
                self.visualizer.update_visual_trajectory(visual_trajectory)  # GREEN
                self.visualizer.update_imu_trajectory(imu_trajectory)  # RED

                # Enhanced frame display with alignment info
                motion_stats = self.slam.get_motion_stats()
                self.visualizer.show_frame(rgb, motion_stats, frame_count)

                if not self.visualizer.spin_once():
                    break

                # Performance monitoring
                frame_time = time.time() - frame_start
                map_size = len(map_cloud.points) if map_cloud else 0
                self.monitor.log_frame(frame_time, map_size)

                frame_count += 1

                # Detailed status every 3 seconds
                current_time = time.time()
                if current_time - last_stats_time >= 3.0:
                    self.print_alignment_status(frame_count, visual_trajectory, imu_trajectory,
                                                map_size, motion_stats, start_time)
                    last_stats_time = current_time

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.save_enhanced_session(session_name)
            self.cleanup()

    def print_alignment_status(self, frame_count, visual_trajectory, imu_trajectory,
                               map_size, motion_stats, start_time):
        """Print detailed status including trajectory alignment"""
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed

        print(f"\n=== Alignment Status (Frame {frame_count}) ===")
        print(f"Runtime: {elapsed:.1f}s | FPS: {avg_fps:.1f}")
        print(f"Map: {map_size} points")

        # Trajectory information
        print(f"Trajectories:")
        print(f"  Visual SLAM (GREEN): {len(visual_trajectory)} poses")
        print(f"  IMU (RED): {len(imu_trajectory)} poses")

        # Scale alignment info
        if motion_stats and 'scale_factor' in motion_stats:
            scale_factor = motion_stats['scale_factor']
            print(f"  Scale Factor: {scale_factor:.3f}")

            if 0.8 <= scale_factor <= 1.2:
                print(f"  Scale Status: ✓ GOOD (near 1.0)")
            elif 0.5 <= scale_factor <= 2.0:
                print(f"  Scale Status: ⚠ FAIR (moderate difference)")
            else:
                print(f"  Scale Status: ✗ POOR (large difference)")

        # Motion analysis
        if motion_stats:
            print(f"Motion Analysis:")
            print(f"  Motion Rate: {motion_stats.get('motion_rate', 0):.1%}")
            print(f"  Drift Detection: {motion_stats.get('drift_detection_rate', 0):.1%}")

        # Trajectory quality assessment
        if len(visual_trajectory) > 5 and len(imu_trajectory) > 5:
            quality_metrics = self.assess_trajectory_quality(visual_trajectory, imu_trajectory)
            print(f"Alignment Quality:")
            print(f"  Recent Distance Ratio: {quality_metrics['distance_ratio']:.2f}")
            if 0.8 <= quality_metrics['distance_ratio'] <= 1.2:
                print(f"  Alignment Status: ✓ GOOD")
            else:
                print(f"  Alignment Status: ⚠ Needs attention")

        print("=" * 45)

    def assess_trajectory_quality(self, visual_trajectory, imu_trajectory):
        """Assess the quality of trajectory alignment"""
        # Get recent segments for comparison
        recent_visual = visual_trajectory[-10:] if len(visual_trajectory) >= 10 else visual_trajectory
        recent_imu = imu_trajectory[-10:] if len(imu_trajectory) >= 10 else imu_trajectory

        # Calculate distances
        visual_dist = 0
        for i in range(1, len(recent_visual)):
            visual_dist += np.linalg.norm(recent_visual[i][:3, 3] - recent_visual[i - 1][:3, 3])

        imu_dist = 0
        for i in range(1, min(len(recent_imu), len(recent_visual))):
            imu_dist += np.linalg.norm(recent_imu[i][:3, 3] - recent_imu[i - 1][:3, 3])

        distance_ratio = imu_dist / visual_dist if visual_dist > 0.001 else 1.0

        return {
            'visual_distance': visual_dist,
            'imu_distance': imu_dist,
            'distance_ratio': distance_ratio
        }

    def save_enhanced_session(self, session_name):
        """Save enhanced session with both trajectories"""
        if self.slam is not None:
            os.makedirs("../data/sessions", exist_ok=True)
            session_path = f"../data/sessions/{session_name}"
            self.slam.save_session(session_path)

            # Print final alignment analysis
            motion_stats = self.slam.get_motion_stats()
            visual_trajectory = self.slam.get_trajectory()
            imu_trajectory = self.slam.get_imu_trajectory()

            print(f"\n=== Final Trajectory Alignment Analysis ===")
            print(f"Session: {session_name}")

            if motion_stats:
                print(f"Processing Statistics:")
                print(f"  Total frames: {motion_stats.get('total_frames', 0)}")
                print(f"  Motion detected: {motion_stats.get('motion_rate', 0):.1%} of time")
                print(f"  Scale factor: {motion_stats.get('scale_factor', 1.0):.3f}")

                scale_factor = motion_stats.get('scale_factor', 1.0)
                if abs(scale_factor - 1.0) < 0.1:
                    print(f"  Scale alignment: ✓ EXCELLENT (very close to 1.0)")
                elif abs(scale_factor - 1.0) < 0.3:
                    print(f"  Scale alignment: ✓ GOOD")
                elif abs(scale_factor - 1.0) < 0.5:
                    print(f"  Scale alignment: ⚠ FAIR")
                else:
                    print(f"  Scale alignment: ✗ POOR")

            print(f"Trajectory Lengths:")
            print(f"  Visual SLAM: {len(visual_trajectory)} poses")
            print(f"  IMU: {len(imu_trajectory)} poses")

            if len(visual_trajectory) > 10 and len(imu_trajectory) > 10:
                # Calculate total distances
                visual_total_dist = 0
                for i in range(1, len(visual_trajectory)):
                    visual_total_dist += np.linalg.norm(
                        visual_trajectory[i][:3, 3] - visual_trajectory[i - 1][:3, 3]
                    )

                imu_total_dist = 0
                for i in range(1, min(len(imu_trajectory), len(visual_trajectory))):
                    imu_total_dist += np.linalg.norm(
                        imu_trajectory[i][:3, 3] - imu_trajectory[i - 1][:3, 3]
                    )

                print(f"Total Distances:")
                print(f"  Visual SLAM: {visual_total_dist:.3f}m")
                print(f"  IMU: {imu_total_dist:.3f}m")
                print(f"  Ratio: {imu_total_dist / visual_total_dist:.3f}" if visual_total_dist > 0 else "  Ratio: N/A")

            print(f"Session saved to: {session_path}")

    def cleanup(self):
        """Clean shutdown"""
        if self.camera:
            self.camera.stop()
        if self.visualizer:
            self.visualizer.close()
        print("Enhanced SLAM with trajectory alignment shutdown complete")


def create_enhanced_config():
    """Create enhanced config with motion detection and alignment parameters"""
    config = {
        "camera": {"width": 640, "height": 480, "fps": 30},
        "slam": {
            "voxel_size": 0.05,
            "max_points": 20000,
            "icp_threshold": 0.02,
            "max_depth": 3.0,
            "process_every_n": 1,
            "accumulate_every_n": 3
        },
        "viz": {
            "point_size": 2,
            "background": [0, 0, 0],
            "viz_points_limit": 15000,
            "update_every_n": 2,
            "show_trajectory": True,
            "trajectory_every_n": 3
        },
        "motion_detection": {
            "imu_drift_threshold": 0.17078,
            "visual_feature_threshold": 5.0,
            "visual_flow_threshold": 2.0,
            "confidence_thresholds": {
                "high": 0.9,
                "medium": 0.7,
                "low": 0.4
            }
        },
        "trajectory_alignment": {
            "scale_estimation_window": 20,
            "scale_update_interval": 30,
            "scale_smoothing_factor": 0.1
        }
    }
    return config


def main():
    print("Enhanced SLAM with Automatic Trajectory Alignment")
    print("=" * 50)
    print("This system automatically aligns visual SLAM and IMU trajectories")
    print("by estimating scale factors in real-time.")
    print("\nVisualization:")
    print("- GREEN line: Visual SLAM trajectory (scale-corrected)")
    print("- RED line: IMU trajectory")
    print("- Point cloud: 3D map from visual SLAM")
    print("\nThe system will show real-time scale factor and alignment quality.")

    # Setup directories
    os.makedirs("../config", exist_ok=True)
    os.makedirs("../data/sessions", exist_ok=True)

    # Load or create config
    config_path = "../config/config.json"
    if not os.path.exists(config_path):
        config = create_enhanced_config()
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created enhanced config at {config_path}")

    # Initialize system
    system = EnhancedSLAMSystem(config_path)

    print(f"\nStarting enhanced SLAM with trajectory alignment...")
    print("Move the camera around to see both trajectories develop.")
    print("The system will automatically estimate and apply scale alignment.\n")

    try:
        system.run_enhanced_slam("aligned_trajectory_test")
    except Exception as e:
        print(f"Enhanced SLAM failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()