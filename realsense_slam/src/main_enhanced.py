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
        print(f"Session: {session_name}")
        print("Controls: Press 'q' to quit")
        print("=" * 55)

        # Initialize components
        self.camera = D435iCamera(self.config)
        intrinsics = self.camera.get_intrinsics()
        self.slam = EnhancedMinimalSLAM(intrinsics, self.config)
        self.visualizer = MinimalVisualizer(self.config)

        self.running = True
        frame_count = 0
        start_time = time.time()

        # Motion tracking for display
        last_motion_status = "UNKNOWN"
        last_stats_time = time.time()

        try:
            while self.running:
                frame_start = time.time()

                # Get camera frames and IMU
                rgb, depth = self.camera.get_frames()
                accel, gyro = self.camera.get_imu_data()

                if rgb is None or depth is None:
                    continue

                # Process enhanced SLAM with motion detection
                depth_meters = depth.astype(np.float32) / 1000.0
                self.slam.process_frame(rgb, depth_meters, accel, gyro)

                # Update visualization
                map_cloud = self.slam.get_map()
                trajectory = self.slam.get_trajectory()

                self.visualizer.update_map(map_cloud)
                self.visualizer.update_trajectory(trajectory)

                # Enhanced frame display with motion status
                motion_stats = self.slam.get_motion_stats()
                self.show_enhanced_frame(rgb, motion_stats, frame_count)

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
                    self.print_detailed_status(frame_count, trajectory, map_size, motion_stats, start_time)
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

    def show_enhanced_frame(self, rgb, motion_stats, frame_count):
        """Show RGB frame with motion detection overlay"""
        display_img = rgb.copy()

        # Add motion status overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Get recent motion log entry
        if hasattr(self.slam, 'motion_log') and self.slam.motion_log:
            recent_motion = self.slam.motion_log[-1]

            # Motion status
            motion_text = "MOTION" if recent_motion['motion_detected'] else "STATIONARY"
            motion_color = (0, 255, 0) if recent_motion['motion_detected'] else (0, 0, 255)
            cv2.putText(display_img, f"Status: {motion_text}", (10, 30), font, font_scale, motion_color, thickness)

            # Agreement status
            agreement = recent_motion['agreement']
            agreement_color = (0, 255, 0) if 'agree' in agreement else (0, 165, 255)
            cv2.putText(display_img, f"Agreement: {agreement}", (10, 55), font, 0.5, agreement_color, 1)

            # Visual and IMU indicators
            visual_text = "V+" if recent_motion['visual_motion'] else "V-"
            imu_text = "I+" if recent_motion['imu_motion'] else "I-"
            cv2.putText(display_img, f"{visual_text} {imu_text}", (10, 75), font, 0.5, (255, 255, 255), 1)

        # Motion statistics
        if motion_stats:
            drift_rate = motion_stats.get('drift_detection_rate', 0)
            motion_rate = motion_stats.get('motion_rate', 0)

            cv2.putText(display_img, f"Motion Rate: {motion_rate:.1%}", (10, 100), font, 0.5, (255, 255, 255), 1)
            cv2.putText(display_img, f"Drift Rate: {drift_rate:.1%}", (10, 120), font, 0.5, (255, 255, 0), 1)

        # Frame counter
        cv2.putText(display_img, f"Frame: {frame_count}", (10, display_img.shape[0] - 10), font, 0.5, (128, 128, 128),
                    1)

        # Resize and show
        display_img = cv2.resize(display_img, (480, 360))
        cv2.imshow("Enhanced SLAM - Motion Detection", display_img)

    def print_detailed_status(self, frame_count, trajectory, map_size, motion_stats, start_time):
        """Print detailed status including motion analysis"""
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed

        print(f"\n=== Status Update (Frame {frame_count}) ===")
        print(f"Runtime: {elapsed:.1f}s | FPS: {avg_fps:.1f}")
        print(f"Trajectory: {len(trajectory)} poses | Map: {map_size} points")

        if motion_stats:
            print(f"Motion Analysis:")
            print(f"  Motion Rate: {motion_stats.get('motion_rate', 0):.1%} "
                  f"({motion_stats.get('motion_frames', 0)}/{motion_stats.get('total_frames', 0)} frames)")
            print(f"  Drift Detection: {motion_stats.get('drift_detection_rate', 0):.1%}")

            # Agreement breakdown
            agreements = motion_stats.get('agreement_counts', {})
            if agreements:
                print(f"  Agreement Types:")
                for agreement, count in agreements.items():
                    percentage = count / motion_stats.get('total_frames', 1) * 100
                    print(f"    {agreement}: {percentage:.1f}%")

        print("=" * 40)

    def save_enhanced_session(self, session_name):
        """Save enhanced session with motion analysis"""
        if self.slam is not None:
            os.makedirs("../data/sessions", exist_ok=True)
            session_path = f"../data/sessions/{session_name}"
            self.slam.save_session(session_path)

            # Print final motion analysis
            motion_stats = self.slam.get_motion_stats()
            print(f"\n=== Final Motion Analysis ===")
            if motion_stats:
                print(f"Total frames processed: {motion_stats.get('total_frames', 0)}")
                print(f"Motion detected: {motion_stats.get('motion_rate', 0):.1%} of time")
                print(f"IMU drift filtered: {motion_stats.get('drift_detection_rate', 0):.1%} of time")

                agreements = motion_stats.get('agreement_counts', {})
                if agreements:
                    print(f"Motion agreement breakdown:")
                    for agreement, count in agreements.items():
                        percentage = count / motion_stats.get('total_frames', 1) * 100
                        print(f"  {agreement}: {percentage:.1f}%")

            print(f"Session saved to {session_path}")

    def cleanup(self):
        """Clean shutdown"""
        if self.camera:
            self.camera.stop()
        if self.visualizer:
            self.visualizer.close()
        print("Enhanced SLAM shutdown complete")


def create_enhanced_config():
    """Create enhanced config with motion detection parameters"""
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
            "trajectory_every_n": 5
        },
        "motion_detection": {
            "imu_drift_threshold": 0.17078,  # From your test
            "visual_feature_threshold": 5.0,
            "visual_flow_threshold": 2.0,
            "confidence_thresholds": {
                "high": 0.9,
                "medium": 0.7,
                "low": 0.4
            }
        }
    }
    return config


def main():
    print("Enhanced SLAM with Visual-IMU Motion Detection")
    print("=" * 45)

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

    print(f"\nIMU Drift Filtering: Using {0.17078:.5f} m/s threshold")
    print("Visual motion detection: Feature matching + optical flow")
    print("Starting enhanced SLAM...\n")

    try:
        system.run_enhanced_slam("enhanced_live")
    except Exception as e:
        print(f"Enhanced SLAM failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()