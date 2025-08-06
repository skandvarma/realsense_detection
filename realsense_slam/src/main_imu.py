import json
import os
import signal
import sys
import time
from camera import D435iCamera
from slam_imu import IMUOnlySLAM
from visualizer_imu import IMUVisualizer


class IMUSLAMSystem:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.camera = None
        self.slam = None
        self.visualizer = None
        self.running = False

        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        print("Shutting down IMU SLAM system...")
        self.running = False

    def run_imu_slam(self, session_name):
        print("Starting IMU-only SLAM session")
        print("Keep device STATIONARY for first 3 seconds")

        # Initialize components
        self.camera = D435iCamera(self.config)
        self.slam = IMUOnlySLAM()
        self.visualizer = IMUVisualizer()

        self.running = True
        frame_count = 0
        start_time = time.time()
        last_print_time = time.time()

        try:
            while self.running:
                # Get frames to trigger IMU data update
                self.camera.get_frames()  # This updates IMU data internally

                # Get IMU data from camera
                accel, gyro = self.camera.get_imu_data()

                if accel is not None and gyro is not None:
                    # Process IMU data
                    current_time = time.time()
                    self.slam.process_imu_data(accel, gyro, current_time)

                    # Update visualization
                    trajectory = self.slam.get_trajectory()
                    if len(trajectory) > 0:
                        self.visualizer.update_trajectory(trajectory)
                        self.visualizer.update_device_pose(
                            self.slam.get_current_pose(),
                            self.slam.velocity
                        )

                    # Show detailed status
                    if current_time - last_print_time >= 1.0:
                        trajectory_length = len(trajectory)
                        current_position = self.slam.position - self.slam.reset_position
                        self.visualizer.show_status(
                            frame_count,
                            trajectory_length,
                            self.slam.bias_estimated,
                            current_position
                        )
                        last_print_time = current_time

                    frame_count += 1

                # Update visualization
                self.visualizer.spin_once()

                # Control frame rate
                time.sleep(0.01)  # 100Hz

        except Exception as e:
            print(f"Error during IMU SLAM: {e}")
        finally:
            self.save_session(session_name)
            self.cleanup()

    def save_session(self, session_name):
        """Save IMU trajectory data"""
        if self.slam is not None:
            os.makedirs("../data/imu_sessions", exist_ok=True)
            session_path = f"../data/imu_sessions/{session_name}"
            self.slam.save_trajectory(session_path)
            print(f"IMU session saved to {session_path}")

    def cleanup(self):
        """Clean up resources"""
        if self.camera:
            self.camera.stop()
        if self.visualizer:
            self.visualizer.close()
        print("IMU SLAM system shutdown complete")


def main():
    # Create directories
    os.makedirs("../config", exist_ok=True)
    os.makedirs("../data/imu_sessions", exist_ok=True)

    # Use existing config or create minimal one
    config_path = "../config/config.json"
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump({
                "camera": {"width": 640, "height": 480, "fps": 30},
                "slam": {"voxel_size": 0.02, "max_depth": 3.0},
                "viz": {"point_size": 2, "background": [0, 0, 0]}
            }, f, indent=2)

    print("IMU-only SLAM System")
    print("Uses only accelerometer and gyroscope data")
    print("Automatic drift reset every 5 seconds")

    # Initialize and run system
    system = IMUSLAMSystem(config_path)

    try:
        system.run_imu_slam("imu_mapping")
    except KeyboardInterrupt:
        print("IMU SLAM interrupted by user")
    except Exception as e:
        print(f"IMU SLAM failed: {e}")


if __name__ == "__main__":
    main()