import json
import os
import signal
import sys
import numpy as np
from camera import D435iCamera
from slam import SimpleSLAM
from visualizer import SLAMVisualizer


class SLAMSystem:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.camera = None
        self.slam = None
        self.visualizer = None
        self.running = False

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        print("\nShutting down SLAM system...")
        self.running = False

    def run_session(self, session_name):
        import time

        print(f"Starting SLAM session: {session_name}")
        print("=" * 50)
        print("IMPORTANT: Keep device STATIONARY for first 3 seconds!")
        print("This allows IMU bias estimation for better tracking.")
        print("=" * 50)

        # Initialize components
        self.camera = D435iCamera(self.config)
        intrinsics = self.camera.get_intrinsics()
        self.slam = SimpleSLAM(intrinsics, self.config)
        self.visualizer = SLAMVisualizer(self.config)

        self.running = True
        frame_count = 0

        # Optimized performance parameters
        slam_skip = 8  # Process every 8th frame initially
        viz_skip = 15  # Update visualization every 15th frame

        # Adaptive processing based on performance
        min_slam_skip = 6
        max_slam_skip = 15
        target_slam_time = 0.1  # Target 100ms for SLAM processing

        # Performance monitoring
        start_time = time.time()
        slam_times = []
        viz_times = []
        last_performance_check = time.time()

        # IMU calibration tracking
        calibration_complete = False
        calibration_start_time = time.time()

        try:
            while self.running:
                # Get camera frames and IMU data
                rgb, depth = self.camera.get_frames()
                accel, gyro = self.camera.get_imu_data()

                if rgb is None or depth is None:
                    continue

                # Always show video feed with calibration status
                calibration_status = None
                if not calibration_complete:
                    progress = len(self.slam.bias_samples) if hasattr(self.slam, 'bias_samples') else 0
                    calibration_status = {'complete': False, 'progress': progress}
                else:
                    calibration_status = {'complete': True}

                self.visualizer.show_frame(rgb, calibration_status)

                # Check IMU calibration status
                if not calibration_complete:
                    if hasattr(self.slam, 'bias_estimated') and self.slam.bias_estimated:
                        calibration_complete = True
                        calibration_time = time.time() - calibration_start_time
                        print(f"\n✓ IMU calibration complete in {calibration_time:.1f}s")
                        print("✓ You can now move the device for SLAM mapping\n")
                    else:
                        # Show calibration progress
                        elapsed = time.time() - calibration_start_time
                        if hasattr(self.slam, 'bias_samples'):
                            progress = len(self.slam.bias_samples)
                            if frame_count % 30 == 0:  # Update every second
                                print(f"Calibrating IMU... {progress}/100 samples ({elapsed:.1f}s)")

                # Adaptive SLAM processing
                process_slam = frame_count % slam_skip == 0

                if process_slam:
                    # Convert depth to meters
                    depth_meters = depth.astype(np.float32) / 1000.0

                    # Time SLAM processing
                    slam_start = time.time()
                    self.slam.process_frame(rgb, depth_meters, accel, gyro)
                    slam_processing_time = time.time() - slam_start
                    slam_times.append(slam_processing_time)

                    # Adaptive skip adjustment based on performance
                    if len(slam_times) >= 5:
                        avg_slam_time = np.mean(slam_times[-5:])
                        if avg_slam_time > target_slam_time and slam_skip < max_slam_skip:
                            slam_skip += 1
                        elif avg_slam_time < target_slam_time * 0.7 and slam_skip > min_slam_skip:
                            slam_skip -= 1

                # Update 3D visualization less frequently
                if frame_count % viz_skip == 0 and calibration_complete:
                    viz_start = time.time()

                    map_cloud = self.slam.get_map()
                    trajectory = self.slam.get_trajectory()

                    if len(map_cloud.points) > 0:
                        self.visualizer.update_map(map_cloud)
                    if len(trajectory) > 1:
                        self.visualizer.update_trajectory(trajectory)

                    viz_times.append(time.time() - viz_start)

                # Update visualizer (lightweight operation)
                self.visualizer.spin_once()

                frame_count += 1

                # Performance monitoring - print stats every 1.5 seconds (only after calibration)
                current_time = time.time()
                if (current_time - last_performance_check >= 1.5 and
                        calibration_complete and slam_times):

                    avg_slam_time = np.mean(slam_times[-10:]) if slam_times else 0
                    avg_viz_time = np.mean(viz_times[-5:]) if viz_times else 0
                    total_time = current_time - start_time
                    fps = frame_count / total_time

                    # IMU status
                    imu_status = "IMU OK" if accel is not None and gyro is not None else "IMU None"

                    # Enhanced status with IMU integration info
                    map_points = len(self.slam.get_map().points)
                    trajectory_length = len(self.slam.get_trajectory())

                    # Check if IMU is providing good data
                    imu_integration_status = "INACTIVE"
                    if hasattr(self.slam, 'bias_estimated') and self.slam.bias_estimated:
                        if hasattr(self.slam, 'is_stationary'):
                            is_stationary = self.slam.is_stationary(accel,
                                                                    gyro) if accel is not None and gyro is not None else False
                            imu_integration_status = "STATIONARY" if is_stationary else "TRACKING"
                        else:
                            imu_integration_status = "ACTIVE"

                    print(f"Frame {frame_count}: "
                          f"FPS: {fps:.1f}, "
                          f"SLAM: {avg_slam_time * 1000:.1f}ms (skip:{slam_skip}), "
                          f"Viz: {avg_viz_time * 1000:.1f}ms, "
                          f"Trajectory: {trajectory_length} poses, "
                          f"Map: {map_points} pts, "
                          f"Keyframes: {len(self.slam.frame_pcds)}, "
                          f"IMU: {imu_integration_status}")

                    last_performance_check = current_time

        except Exception as e:
            print(f"Error during SLAM session: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.save_current_session(session_name)
            self.cleanup()

    def save_current_session(self, session_name):
        if self.slam is not None:
            os.makedirs("../data/sessions", exist_ok=True)
            session_path = f"../data/sessions/{session_name}"
            self.slam.save_session(session_path)
            print(f"Session saved to {session_path}")

    def cleanup(self):
        if self.camera:
            self.camera.stop()
        if self.visualizer:
            self.visualizer.close()
        print("SLAM system shutdown complete")


def main():
    # Create directories
    os.makedirs("../config", exist_ok=True)
    os.makedirs("../data/maps", exist_ok=True)
    os.makedirs("../data/sessions", exist_ok=True)

    # Create optimized config file if it doesn't exist
    config_path = "../config/config.json"
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump({
                "camera": {"width": 640, "height": 480, "fps": 30},
                "slam": {"voxel_size": 0.04, "max_depth": 3.0},  # Optimized voxel size
                "viz": {"point_size": 2, "background": [0, 0, 0]}
            }, f, indent=2)

    print("Real-Time SLAM with Localized IMU Integration")
    print("=" * 45)
    print("Features:")
    print("- Localized IMU reference frame")
    print("- Automatic bias estimation")
    print("- Zero-velocity detection")
    print("- Improved visual-IMU fusion")
    print("- Adaptive performance tuning")
    print("=" * 45)

    # Initialize and run SLAM system
    system = SLAMSystem(config_path)

    try:
        system.run_session("live_mapping")
    except KeyboardInterrupt:
        print("SLAM session interrupted by user")
    except Exception as e:
        print(f"SLAM session failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()