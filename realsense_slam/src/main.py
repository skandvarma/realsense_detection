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

        # Initialize components
        self.camera = D435iCamera(self.config)
        intrinsics = self.camera.get_intrinsics()
        self.slam = SimpleSLAM(intrinsics, self.config)
        self.visualizer = SLAMVisualizer(self.config)

        self.running = True
        frame_count = 0

        # Performance parameters
        slam_skip = 5  # Process every 3rd frame for SLAM (reduces lag)
        viz_skip = 10  # Update visualization every 11th frame

        # Performance monitoring
        start_time = time.time()
        slam_times = []
        viz_times = []

        try:
            while self.running:
                # Get camera frames and IMU data
                rgb, depth = self.camera.get_frames()
                accel, gyro = self.camera.get_imu_data()

                if rgb is None or depth is None:
                    continue

                # Always show video feed for smoothness
                self.visualizer.show_frame(rgb)

                # Process SLAM only on selected frames
                if frame_count % slam_skip == 0:
                    # Convert depth to meters
                    depth_meters = depth.astype(np.float32) / 1000.0

                    # Time SLAM processing
                    slam_start = time.time()
                    self.slam.process_frame(rgb, depth_meters, accel, gyro)
                    slam_times.append(time.time() - slam_start)

                # Update 3D visualization less frequently
                if frame_count % viz_skip == 0:
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

                # Performance monitoring - print stats every 1 second at 30fps
                if frame_count % 30 == 0:
                    avg_slam_time = np.mean(slam_times[-10:]) if slam_times else 0
                    avg_viz_time = np.mean(viz_times[-6:]) if viz_times else 0
                    total_time = time.time() - start_time
                    fps = frame_count / total_time

                    imu_status = "IMU OK" if accel is not None and gyro is not None else "IMU None"
                    print(f"Frame {frame_count}: "
                          f"FPS: {fps:.1f}, "
                          f"SLAM: {avg_slam_time * 1000:.1f}ms, "
                          f"Viz: {avg_viz_time * 1000:.1f}ms, "
                          f"Trajectory: {len(self.slam.get_trajectory())} poses, "
                          f"Map points: {len(self.slam.get_map().points)}, "
                          f"Keyframes: {len(self.slam.frame_pcds)}, "
                          f"{imu_status}")

        except Exception as e:
            print(f"Error during SLAM session: {e}")
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

    # Create config file if it doesn't exist
    config_path = "../config/config.json"
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump({
                "camera": {"width": 640, "height": 480, "fps": 30},
                "slam": {"voxel_size": 0.035, "max_depth": 3.0},
                "viz": {"point_size": 3, "background": [0, 0, 0]}
            }, f, indent=2)

    # Initialize and run SLAM system
    system = SLAMSystem(config_path)

    try:
        system.run_session("live_mapping")
    except KeyboardInterrupt:
        print("SLAM session interrupted by user")
    except Exception as e:
        print(f"SLAM session failed: {e}")


if __name__ == "__main__":
    main()