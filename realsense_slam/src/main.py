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
        print(f"Starting SLAM session: {session_name}")

        # Initialize components
        self.camera = D435iCamera(self.config)
        intrinsics = self.camera.get_intrinsics()
        self.slam = SimpleSLAM(intrinsics, self.config)
        self.visualizer = SLAMVisualizer(self.config)

        self.running = True
        frame_count = 0

        # Performance parameters
        slam_skip = 2  # Process every 3rd frame for SLAM (reduces lag)
        viz_skip = 5  # Update visualization every 6th frame

        try:
            while self.running:
                # Get camera frames (always for smooth video)
                rgb, depth = self.camera.get_frames()
                if rgb is None or depth is None:
                    continue

                # Always show video feed for smoothness
                self.visualizer.show_frame(rgb)

                # Process SLAM only on selected frames
                if frame_count % slam_skip == 0:
                    # Convert depth to meters
                    depth_meters = depth.astype(np.float32) / 1000.0
                    self.slam.process_frame(rgb, depth_meters)

                # Update 3D visualization less frequently
                if frame_count % viz_skip == 0:
                    map_cloud = self.slam.get_map()
                    trajectory = self.slam.get_trajectory()

                    if len(map_cloud.points) > 0:
                        self.visualizer.update_map(map_cloud)
                    if len(trajectory) > 1:
                        self.visualizer.update_trajectory(trajectory)

                # Update visualizer (lightweight operation)
                self.visualizer.spin_once()

                frame_count += 1

                # Performance monitoring
                if frame_count % 90 == 0:  # Every 3 seconds at 30fps
                    print(
                        f"Processed {frame_count} frames, SLAM trajectory: {len(self.slam.get_trajectory())} poses, Map points: {len(self.slam.get_map().points)}")

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
                "slam": {"voxel_size": 0.01, "max_depth": 3.0},
                "viz": {"point_size": 2, "background": [0, 0, 0]}
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