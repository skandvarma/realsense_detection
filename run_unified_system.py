#!/usr/bin/env python3
"""
Unified launcher with SLAM data saving and selective headless mode.
SLAM runs headless and saves data, Detection shows GUI.
"""

import sys
import os
import time
import threading
import signal
import cv2
import json
import numpy as np
from pathlib import Path

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'realsense_slam/src'))
sys.path.insert(0, os.path.dirname(__file__))

# Environment setup for stability
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from src.camera.realsense_manager import RealSenseDetectionCamera
from src.detection import DetectorFactory, Postprocessor
from src.camera.depth_processor import DepthProcessor
from src.camera.realsense_manager import CameraShareManager


class UnifiedSystem:
    """Unified system for running SLAM and Detection together."""

    def __init__(self, slam_headless=True, detection_headless=False):
        """Initialize unified system.

        Args:
            slam_headless: If True, SLAM runs without GUI (default: True)
            detection_headless: If False, Detection shows GUI (default: False)
        """
        self.slam_headless = slam_headless
        self.detection_headless = detection_headless
        self.running = False
        self.slam_thread = None
        self.detection_thread = None
        self.slam_system = None
        self.detection_system = None
        self.shared_camera_manager = None
        self.detection_prompt = None  # Store text prompt for detection

        # MINIMAL ADDITION: Session management
        self.session_name = "unified"  # MINIMAL CHANGE: Fixed name without timestamp
        self.save_interval = 30.0  # Save every 30 seconds
        self.last_save_time = time.time()

        # Create data directories
        self.data_dir = Path("realsense_slam/data/sessions")  # MINIMAL CHANGE: New path
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Session data will be saved to: {self.data_dir}/{self.session_name}_*")

        # Window positions for detection
        self.window_positions = {
            'detection': (100, 100),
        }

        # Shared data
        self.shared_data = {
            'slam_frame': None,
            'detection_frame': None,
            'slam_trajectory': [],
            'detections': [],
            'lock': threading.Lock()
        }

        # Load configuration
        self.config = self.load_config()

        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)

        print("=" * 60)
        print("UNIFIED SLAM + DETECTION SYSTEM")
        print(f"SLAM: {'HEADLESS' if self.slam_headless else 'GUI'} mode")
        print(f"Detection: {'HEADLESS' if self.detection_headless else 'GUI'} mode")
        print(f"Session: {self.session_name}")
        print("=" * 60)

        # Initialize shared camera
        self.initialize_shared_camera()

    def initialize_shared_camera(self):
        """Create and initialize shared camera manager."""
        try:
            print("\nInitializing shared camera manager...")
            self.shared_camera_manager = CameraShareManager()

            if not self.shared_camera_manager.initialize_camera(self.config):
                raise RuntimeError("Failed to initialize shared camera")

            time.sleep(1.0)
            print("Shared camera manager initialized successfully")

        except Exception as e:
            print(f"Camera initialization error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def load_config(self):
        """Load configuration for both systems."""
        slam_config_path = "realsense_slam/config/config.json"
        if os.path.exists(slam_config_path):
            with open(slam_config_path, 'r') as f:
                config = json.load(f)
        else:
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

        # Try to load detection config from YAML
        try:
            from src.utils.config import ConfigManager
            yaml_config = ConfigManager.load_config('config.yaml')
            if 'detection' in yaml_config:
                print("Using detection config from config.yaml")
                config['detection'] = yaml_config['detection']
                return config
        except ImportError:
            print("PyYAML not installed, using default detection config")
        except Exception as e:
            print(f"Error loading YAML config: {e}, using default detection config")

        # Fallback to default detection config
        config['detection'] = {
            'active_model': 'yolo',
            'yolo': {
                'model_path': 'data/models/',
                'weights': 'yolo11m.pt',
                'confidence_threshold': 0.2,
                'input_size': [640, 640]
            },
            'detr': {
                'variant': 'grounding-dino',
                'model_path': 'data/models/',
                'model_name': 'IDEA-Research/grounding-dino-base',
                'confidence_threshold': 0.2,
                'max_detections': 10,
                'input_size': [640, 640],
                'normalize': True,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'target_classes': []
            }
        }

        return config

    def run_slam_system(self):
        """Run SLAM system using enhanced_slam_v2 with Open3D GUI."""
        try:
            print("\n[SLAM] Initializing Open3D GUI SLAM...")

            # Import Open3D GUI components
            import open3d as o3d
            import open3d.visualization.gui as gui

            # Import the new SLAM system
            sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                            'realsense_slam/src/Open3D/examples/python/t_reconstruction_system'))
            from enhanced_slam_v2 import ReconstructionWindow

            # Create a minimal config object for Open3D SLAM
            class SLAMConfig:
                def __init__(self, config_dict, session_name):
                    # Map our config to Open3D expected format
                    self.depth_scale = 1000.0
                    self.voxel_size = config_dict.get('slam', {}).get('voxel_size', 0.05)
                    self.trunc_voxel_multiplier = 8.0
                    self.block_count = 60000
                    self.est_point_count = 2000000
                    self.depth_max = config_dict.get('slam', {}).get('max_depth', 3.0)
                    self.depth_min = 0.1
                    self.odometry_distance_thr = 0.15
                    self.device = "CUDA:0"  # or "CUDA:0" if available
                    self.path_npz = f"realsense_slam/data/sessions/{session_name}_slam.npz"
                    # Add missing engine attribute
                    self.engine = "tensor"  # or "legacy" - determines mesh extraction method
                    # Add missing surface weight threshold
                    self.surface_weight_thr = 3.0  # Minimum weight for surface extraction

            slam_config = SLAMConfig(self.config, self.session_name)

            if self.slam_headless:
                print("[SLAM] Running in headless mode - GUI disabled")
                # For headless mode, we'll run a simplified version
                self.run_headless_slam()
                return
            else:
                print("[SLAM] Running with Open3D GUI")
                # Initialize Open3D application
                app = gui.Application.instance
                app.initialize()

                # Create font for the GUI
                mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))

                # Create SLAM window
                slam_window = ReconstructionWindow(slam_config, mono)

                print("[SLAM] Open3D GUI SLAM started")

                # Run the application (this blocks)
                app.run()

        except Exception as e:
            print(f"[SLAM] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[SLAM] Open3D SLAM system stopped")

    def run_headless_slam(self):
        """Fallback headless SLAM for when GUI is disabled."""
        try:
            from src.camera.realsense_manager import D435iCamera

            # Use the original headless SLAM as fallback
            slam_camera = D435iCamera(self.config)

            print("[SLAM] Headless mode: Using simplified frame capture")

            frame_count = 0
            last_save_time = time.time()

            while self.running:
                try:
                    # Simple frame capture and logging
                    rgb, depth = slam_camera.get_frames()

                    if rgb is not None and depth is not None:
                        # Update shared data
                        with self.shared_data['lock']:
                            self.shared_data['slam_frame'] = rgb.copy()
                            # For headless, we'll just track frame count as "trajectory"
                            self.shared_data['slam_trajectory'] = list(range(frame_count))

                        frame_count += 1

                        # Log progress every 100 frames
                        if frame_count % 100 == 0:
                            print(f"[SLAM] Processed {frame_count} frames (headless mode)")

                        # Save periodically (minimal data)
                        current_time = time.time()
                        if current_time - last_save_time >= self.save_interval:
                            self.save_headless_slam_data(frame_count)
                            last_save_time = current_time

                    time.sleep(0.033)  # ~30 FPS

                except Exception as e:
                    if frame_count % 100 == 0:
                        print(f"[SLAM] Frame error: {e}")
                    time.sleep(0.1)

        except Exception as e:
            print(f"[SLAM] Headless error: {e}")

    def save_headless_slam_data(self, frame_count):
        """Save minimal data for headless SLAM."""
        try:
            session_path = self.data_dir / f"{self.session_name}_headless.json"
            data = {
                "session_name": self.session_name,
                "frames_processed": frame_count,
                "timestamp": time.time(),
                "mode": "headless"
            }

            import json
            with open(session_path, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"[SLAM] Headless data saved: {frame_count} frames")

        except Exception as e:
            print(f"[SLAM] Error saving headless data: {e}")

    def save_slam_data(self):
        """Save SLAM data (updated for enhanced_slam_v2)."""
        try:
            if self.slam_headless:
                # Already handled in run_headless_slam
                return

            # For GUI mode, the enhanced_slam_v2 saves data automatically
            # when the window is closed, so we just log here
            session_path = str(self.data_dir / self.session_name)
            print(f"[SLAM] GUI SLAM auto-saves to: {session_path}_slam.npz")
            print(f"[SLAM] Trajectory and mesh will be saved on window close")

        except Exception as e:
            print(f"[SLAM] Error in save_slam_data: {e}")

    def run_detection_system(self):
        """Run detection system with GUI display."""
        try:
            print("\n[Detection] Initializing...")

            # Wait for SLAM to initialize first
            time.sleep(3.0)

            camera = RealSenseDetectionCamera(self.config)

            # Initialize detector
            try:
                factory = DetectorFactory()
                detector = factory.create_detector(self.config)
            except Exception as e:
                print(f"[Detection] Warning: Failed to initialize detector: {e}")
                print("[Detection] Running without detection")
                return

            if not detector:
                print("[Detection] Failed to create detector")
                return

            # Set text prompt if using Grounding DINO
            if hasattr(detector, 'is_grounding_dino') and detector.is_grounding_dino:
                if self.detection_prompt:
                    detector.update_text_prompt(self.detection_prompt)
                    print(f"[Detection] Using Grounding DINO with prompt: '{self.detection_prompt}'")
                else:
                    print("[Detection] Warning: No prompt provided for Grounding DINO")

            # Initialize processors
            depth_processor = DepthProcessor(self.shared_camera_manager, self.config)
            depth_processor.update_camera_parameters()
            postprocessor = Postprocessor(self.config, depth_processor)

            self.detection_system = {
                'camera': camera,
                'detector': detector,
                'postprocessor': postprocessor
            }

            print("[Detection] System started (GUI mode)")
            frame_count = 0
            last_log_time = time.time()

            while self.running:
                try:
                    rgb, depth = camera.get_frames()

                    if rgb is not None:
                        # Run detection
                        kwargs = {'frame_id': frame_count}

                        # For Grounding DINO, always pass the text prompt
                        if hasattr(detector, 'is_grounding_dino') and detector.is_grounding_dino:
                            kwargs['text_prompt'] = self.detection_prompt or detector.get_current_prompt()

                        result = detector.detect(rgb, **kwargs)

                        if result.success:
                            # Postprocess
                            result = postprocessor.process_detection_result(
                                result, depth, frame_count
                            )

                            # Update shared data
                            with self.shared_data['lock']:
                                self.shared_data['detection_frame'] = rgb.copy()
                                self.shared_data['detections'] = result.detections

                            # Log periodically
                            current_time = time.time()
                            if current_time - last_log_time >= 3.0:
                                print(
                                    f"[Detection] Frame {frame_count}: {len(result.detections)} objects, FPS: {result.fps:.1f}")
                                last_log_time = current_time

                            # SHOW DETECTION GUI
                            if not self.detection_headless:
                                self.display_detection_frame(rgb, result, frame_count)

                        frame_count += 1

                except Exception as e:
                    if frame_count % 100 == 0:
                        print(f"[Detection] Frame error: {e}")

                # Check for quit key
                if not self.detection_headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[Detection] Quit requested")
                        self.stop()
                        break
                else:
                    time.sleep(0.01)

        except Exception as e:
            print(f"[Detection] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.detection_system:
                if 'detector' in self.detection_system:
                    try:
                        self.detection_system['detector'].cleanup()
                    except:
                        pass
                if 'camera' in self.detection_system:
                    try:
                        self.detection_system['camera'].cleanup()
                    except:
                        pass
            print("[Detection] System stopped")

    def display_detection_frame(self, frame, result, frame_count):
        """Display detection frame with overlays."""
        display_img = cv2.resize(frame, (800, 600))

        # Draw detections
        for detection in result.detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            # Scale coordinates
            x1 = int(x1 * 800 / frame.shape[1])
            y1 = int(y1 * 600 / frame.shape[0])
            x2 = int(x2 * 800 / frame.shape[1])
            y2 = int(y2 * 600 / frame.shape[0])

            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label with class and confidence
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            if hasattr(detection, 'distance') and detection.distance > 0:
                label += f" ({detection.distance:.1f}m)"

            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display_img, (x1, y1 - label_size[1] - 4),
                          (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(display_img, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Add info overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        info_bg = display_img.copy()
        cv2.rectangle(info_bg, (0, 0), (250, 100), (0, 0, 0), -1)
        cv2.addWeighted(info_bg, 0.3, display_img, 0.7, 0, display_img)

        cv2.putText(display_img, "DETECTION SYSTEM", (10, 25), font, 0.7, (0, 255, 255), 2)
        cv2.putText(display_img, f"Objects: {len(result.detections)}", (10, 50), font, 0.5, (255, 255, 255), 1)
        cv2.putText(display_img, f"FPS: {result.fps:.1f}", (10, 70), font, 0.5, (255, 255, 255), 1)
        cv2.putText(display_img, f"Frame: {frame_count}", (10, 90), font, 0.5, (255, 255, 255), 1)

        # Add SLAM info if available
        with self.shared_data['lock']:
            trajectory_len = len(self.shared_data['slam_trajectory'])
        cv2.putText(display_img, f"SLAM Poses: {trajectory_len}", (10, 590), font, 0.5, (0, 255, 0), 1)

        # Show session name
        cv2.putText(display_img, f"Session: {self.session_name}", (300, 590), font, 0.4, (200, 200, 200), 1)

        # Instructions
        cv2.putText(display_img, "Press 'q' to quit", (650, 590), font, 0.5, (255, 255, 0), 1)

        # Display window
        cv2.namedWindow("Detection View", cv2.WINDOW_NORMAL)
        cv2.imshow("Detection View", display_img)

    def start(self):
        """Start both systems with proper GUI handling."""
        # Get detection prompt if needed
        if self.config['detection']['active_model'] == 'detr' and \
                self.config['detection']['detr']['variant'] == 'grounding-dino':
            self.detection_prompt = input(
                "\nEnter detection prompt for Grounding DINO (e.g., 'person . car . dog'): "
            ).strip()
            print(f"Using detection prompt: '{self.detection_prompt}'")

        self.running = True

        if self.slam_headless:
            # Start SLAM in thread for headless mode
            print("\nStarting SLAM system (headless)...")
            self.slam_thread = threading.Thread(target=self.run_slam_system, daemon=True)
            self.slam_thread.start()
            time.sleep(2.0)

            # Start Detection
            print("Starting Detection system...")
            self.detection_thread = threading.Thread(target=self.run_detection_system, daemon=True)
            self.detection_thread.start()
            time.sleep(1.0)

            print("\n" + "=" * 60)
            print("Systems running successfully!")
            print("SLAM: Headless mode")
            print("Detection GUI: Press 'q' in detection window to quit")
            print("=" * 60 + "\n")

            # Keep main thread alive
            try:
                while self.running:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                self.stop()
        else:
            # For GUI SLAM, start detection first, then SLAM GUI (which blocks)
            print("\nStarting Detection system...")
            self.detection_thread = threading.Thread(target=self.run_detection_system, daemon=True)
            self.detection_thread.start()
            time.sleep(2.0)

            print("Starting SLAM system (GUI mode)...")
            print("Note: SLAM GUI will take control of main thread")
            # This will block until SLAM GUI is closed
            self.run_slam_system()

    def stop(self):
        """Stop both systems gracefully."""
        print("\n" + "=" * 60)
        print("Shutting down unified system...")

        self.running = False

        # Save final SLAM data
        print("Saving final SLAM data...")
        self.save_slam_data()

        # Wait for threads
        if self.slam_thread:
            self.slam_thread.join(timeout=5)
        if self.detection_thread:
            self.detection_thread.join(timeout=5)

        # Close windows
        cv2.destroyAllWindows()

        # Cleanup camera
        if self.shared_camera_manager:
            self.shared_camera_manager.cleanup()

        print(f"Session data saved to: realsense_slam/src/data/sessions/{self.session_name}_*")
        print("Unified system stopped successfully")
        print("=" * 60)

    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\nReceived interrupt signal...")

        # If SLAM GUI is running, try to close it properly
        if not self.slam_headless and hasattr(self, 'slam_window'):
            try:
                # Close the Open3D GUI window to trigger save
                import open3d.visualization.gui as gui
                gui.Application.instance.quit()
            except:
                pass

        self.stop()
        sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified SLAM + Detection System")
    parser.add_argument("--slam-gui", action="store_true",
                        help="Show SLAM GUI (default: headless)")
    parser.add_argument("--detection-headless", action="store_true",
                        help="Hide detection GUI (default: show GUI)")
    parser.add_argument("--session-name", type=str,
                        help="Custom session name for saving data")
    args = parser.parse_args()

    # Create system with specified settings
    system = UnifiedSystem(
        slam_headless=not args.slam_gui,  # SLAM is headless by default
        detection_headless=args.detection_headless  # Detection shows GUI by default
    )

    # Override session name if provided
    if args.session_name:
        system.session_name = args.session_name
        print(f"Using custom session name: {args.session_name}")

    try:
        system.start()
    except Exception as e:
        print(f"System error: {e}")
        system.stop()
    finally:
        cv2.destroyAllWindows()