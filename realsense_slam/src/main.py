import json
import os
import time
import cv2
import numpy as np
from camera import D435iCamera
from slam import MinimalSLAM
from visualizer import MinimalVisualizer, PerformanceMonitor


class MinimalSLAMSystem:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.camera = None
        self.slam = None
        self.visualizer = None
        self.monitor = PerformanceMonitor()
        self.running = False

        # Auto-tuning settings
        self.auto_tune = True
        self.tune_check_interval = 150  # Check every 150 frames
        self.last_tune_check = 0

    def apply_preset(self, preset_name):
        """Apply performance preset from config"""
        if preset_name not in self.config.get('performance_presets', {}):
            print(f"Preset '{preset_name}' not found")
            return

        preset = self.config['performance_presets'][preset_name]

        # Update config
        if 'slam' in preset:
            self.config['slam'].update(preset['slam'])
        if 'viz' in preset:
            self.config['viz'].update(preset['viz'])

        print(f"Applied '{preset_name}' preset")

        # Update running components if they exist
        if self.slam:
            self.slam.update_params(**preset.get('slam', {}))
        if self.visualizer:
            self.visualizer.update_params(**preset.get('viz', {}))

    def run_minimal_slam(self, session_name):
        print("=== Minimal Real-time SLAM ===")
        print(f"Session: {session_name}")
        print("Controls during runtime:")
        print("- Press '1' for high performance preset")
        print("- Press '2' for balanced preset")
        print("- Press '3' for high quality preset")
        print("- Press 'q' to quit")
        print("=====================================")

        # Initialize components
        self.camera = D435iCamera(self.config)
        intrinsics = self.camera.get_intrinsics()
        self.slam = MinimalSLAM(intrinsics, self.config)
        self.visualizer = MinimalVisualizer(self.config)

        self.running = True
        frame_count = 0
        start_time = time.time()

        try:
            while self.running:
                frame_start = time.time()

                # Get camera frames
                rgb, depth = self.camera.get_frames()
                if rgb is None or depth is None:
                    continue

                # Process SLAM
                depth_meters = depth.astype(np.float32) / 1000.0
                self.slam.process_frame(rgb, depth_meters)

                # Update visualization
                map_cloud = self.slam.get_map()
                trajectory = self.slam.get_trajectory()

                self.visualizer.update_map(map_cloud)
                self.visualizer.update_trajectory(trajectory)
                self.visualizer.show_frame(rgb)

                if not self.visualizer.spin_once():
                    break

                # Performance monitoring
                frame_time = time.time() - frame_start
                map_size = len(map_cloud.points) if map_cloud else 0
                self.monitor.log_frame(frame_time, map_size)

                frame_count += 1

                # Auto-tuning based on performance
                if (self.auto_tune and
                        frame_count - self.last_tune_check >= self.tune_check_interval):

                    suggestions = self.monitor.suggest_params()
                    if suggestions:
                        slam_params = {k: v for k, v in suggestions.items()
                                       if k in ['voxel_size', 'max_points', 'process_every_n', 'accumulate_every_n']}
                        viz_params = {k: v for k, v in suggestions.items()
                                      if k in ['viz_points_limit', 'update_every_n']}

                        if slam_params:
                            self.slam.update_params(**slam_params)
                        if viz_params:
                            self.visualizer.update_params(**viz_params)

                    self.last_tune_check = frame_count

                # Status update every 5 seconds
                if frame_count % 150 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / elapsed

                    print(f"Status: Frame {frame_count}, "
                          f"Trajectory: {len(trajectory)} poses, "
                          f"Map: {map_size} points, "
                          f"Avg FPS: {avg_fps:.1f}")

                # Check for preset changes (simple keyboard input simulation)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('1'):
                    self.apply_preset('high_performance')
                elif key == ord('2'):
                    self.apply_preset('balanced')
                elif key == ord('3'):
                    self.apply_preset('high_quality')
                elif key == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.save_session(session_name)
            self.cleanup()

    def save_session(self, session_name):
        """Save session data"""
        if self.slam is not None:
            os.makedirs("../data/sessions", exist_ok=True)
            session_path = f"../data/sessions/{session_name}"
            self.slam.save_session(session_path)
            print(f"Session saved to {session_path}")

    def cleanup(self):
        """Clean shutdown"""
        if self.camera:
            self.camera.stop()
        if self.visualizer:
            self.visualizer.close()
        print("Cleanup complete")


def create_default_config():
    """Create default minimal config if not exists"""
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
        }
    }
    return config


def main():
    print("Minimal Real-time SLAM System")
    print("=============================")
    print("Optimized for low latency and tuneable quality")

    # Setup directories
    os.makedirs("../config", exist_ok=True)
    os.makedirs("../data/sessions", exist_ok=True)

    # Load or create config
    config_path = "../config/config.json"
    if not os.path.exists(config_path):
        config = create_default_config()
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created default config at {config_path}")

    # Initialize system
    system = MinimalSLAMSystem(config_path)

    # Show initial parameters
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("\nInitial Parameters:")
    print(f"SLAM voxel_size: {config['slam']['voxel_size']}")
    print(f"SLAM max_points: {config['slam']['max_points']}")
    print(f"VIZ points_limit: {config['viz']['viz_points_limit']}")
    print(f"VIZ update_every_n: {config['viz']['update_every_n']}")

    try:
        system.run_minimal_slam("minimal_live")
    except Exception as e:
        print(f"System failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()