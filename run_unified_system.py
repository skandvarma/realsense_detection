#!/usr/bin/env python3
"""
Robust unified launcher for running SLAM and Detection simultaneously.
This ensures proper initialization, frame distribution, and cleanup.
"""

import sys
import os
import time
import threading
import signal
import queue
import cv2
import json
import numpy as np
from pathlib import Path

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'realsense_slam/src'))
sys.path.insert(0, os.path.dirname(__file__))

#  Import SLAM components
# from realsense_slam.src.camera import D435iCamera
# from realsense_slam.src.enhanced_slam import EnhancedMinimalSLAM
# from realsense_slam.src.visualizer import MinimalVisualizer
from realsense_slam.src.main_enhanced import EnhancedSLAMSystem

# Import Detection components
from src.camera.realsense_manager import RealSenseDetectionCamera
from src.detection import DetectorFactory, Postprocessor
from src.camera.depth_processor import DepthProcessor

# Import CameraShareManager for shared access
from src.camera.realsense_manager import CameraShareManager


class UnifiedSystem:
    """Unified system for running SLAM and Detection together."""

    def __init__(self):
        """Initialize unified system."""
        self.running = False
        self.slam_thread = None
        self.detection_thread = None
        self.slam_system = None
        self.detection_system = None
        self.shared_camera_manager = None  # Add shared camera manager

        # Window management
        self.window_positions = {
            'slam': (0, 0),
            'detection': (650, 0),
            'combined': (0, 400)
        }

        # Shared data for visualization
        self.shared_data = {
            'slam_frame': None,
            'detection_frame': None,
            'slam_trajectory': [],
            'detections': [],
            'lock': threading.Lock()
        }

        # Load configuration
        self.config = self.load_config()

        # Setup signal handler for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)

        print("=" * 60)
        print("UNIFIED SLAM + DETECTION SYSTEM")
        print("=" * 60)

        # Initialize shared camera manager
        self.initialize_shared_camera()

    def initialize_shared_camera(self):
        """Create and initialize shared camera manager."""
        try:
            print("\nInitializing shared camera manager...")
            self.shared_camera_manager = CameraShareManager()

            # FIX: Initialize camera with config and ensure it's ready
            if not self.shared_camera_manager.initialize_camera(self.config):
                raise RuntimeError("Failed to initialize shared camera")

            # FIX: Wait a moment to ensure camera is fully ready
            time.sleep(1.0)

            print("Shared camera manager initialized successfully")

        except Exception as e:
            print(f"Camera initialization error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def load_config(self):
        """Load configuration for both systems."""
        # Try to load SLAM config
        slam_config_path = "realsense_slam/config/config.json"
        if os.path.exists(slam_config_path):
            with open(slam_config_path, 'r') as f:
                config = json.load(f)
        else:
            # Default config
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

        # Add detection config
        config['detection'] = {
            'active_model': 'yolo',
            'yolo': {
                'model_path': 'data/models/',
                'weights': 'yolo11m.pt',
                'confidence_threshold': 0.2,
                'input_size': [640, 640]
            },
            'detr': {
                'model_path': 'data/models/',
                'model_name': 'IDEA-Research/grounding-dino-base',
                'confidence_threshold': 0.15,
                'input_size': [640, 480]
            }
        }

        return config

    def run_detection_system(self):
        """Run detection system in thread."""
        try:
            print("\n[Detection] Initializing...")

            # FIX: Use the already initialized shared camera instead of creating new one
            camera = RealSenseDetectionCamera(self.config)

            # Initialize detector
            factory = DetectorFactory()
            detector = factory.create_detector(self.config)

            if not detector:
                print("[Detection] Failed to create detector")
                return

            # Initialize depth processor for 3D using the shared camera manager
            depth_processor = DepthProcessor(self.shared_camera_manager, self.config)
            depth_processor.update_camera_parameters()
            postprocessor = Postprocessor(self.config, depth_processor)

            self.detection_system = {
                'camera': camera,
                'detector': detector,
                'postprocessor': postprocessor
            }

            print("[Detection] System started")
            frame_count = 0

            while self.running:
                # Get frames
                rgb, depth = camera.get_frames()

                if rgb is not None:
                    # Run detection
                    result = detector.detect(rgb, frame_id=frame_count)

                    if result.success:
                        # Postprocess
                        result = postprocessor.process_detection_result(
                            result, depth, frame_count
                        )

                        # Update shared data
                        with self.shared_data['lock']:
                            self.shared_data['detection_frame'] = rgb.copy()
                            self.shared_data['detections'] = result.detections

                        # Display detection-specific window
                        self.display_detection_frame(rgb, result)

                    frame_count += 1

                time.sleep(0.01)  # Small delay to prevent CPU overload

        except Exception as e:
            print(f"[Detection] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.detection_system:
                if 'detector' in self.detection_system:
                    self.detection_system['detector'].cleanup()
                if 'camera' in self.detection_system:
                    self.detection_system['camera'].cleanup()
            print("[Detection] System stopped")

    def run_slam_system(self):
        """Run SLAM system in thread using main_enhanced.py."""
        try:
            print("\n[SLAM] Initializing...")

            # Import the EnhancedSLAMSystem from main_enhanced
            from realsense_slam.src.main_enhanced import EnhancedSLAMSystem

            # Create the SLAM system
            self.slam_system = EnhancedSLAMSystem(self.config)

            # FIX: Instead of calling run_enhanced_slam (which creates its own camera),
            # we'll manually set up the camera and run the SLAM loop

            # Create shared camera for SLAM
            from src.camera.realsense_manager import D435iCamera
            slam_camera = D435iCamera(self.config)

            # Set the camera in the SLAM system (bypass internal camera creation)
            self.slam_system.camera = slam_camera

            # Get intrinsics and initialize SLAM manually
            intrinsics = slam_camera.get_intrinsics()

            # Import and create the enhanced SLAM processor
            from realsense_slam.src.enhanced_slam import EnhancedMinimalSLAM
            self.slam_system.slam = EnhancedMinimalSLAM(intrinsics, self.config)

            # Set up visualizer if needed (optional, since we're doing headless)
            try:
                from realsense_slam.src.visualizer import MinimalVisualizer
                self.slam_system.visualizer = MinimalVisualizer(self.config)
            except:
                self.slam_system.visualizer = None

            print("[SLAM] System started")

            # Now run the SLAM processing loop (similar to what run_enhanced_slam does)
            frame_count = 0
            start_time = time.time()
            last_stats_time = time.time()

            while self.running:
                try:
                    frame_start = time.time()

                    # Get camera frames and IMU
                    rgb, depth = slam_camera.get_frames()
                    accel, gyro = slam_camera.get_imu_data()

                    if rgb is None or depth is None:
                        continue

                    # Process enhanced SLAM with motion detection
                    depth_meters = depth.astype(np.float32) / 1000.0
                    self.slam_system.slam.process_frame(rgb, depth_meters, accel, gyro)

                    # Update shared data for visualization
                    with self.shared_data['lock']:
                        self.shared_data['slam_frame'] = rgb.copy()
                        self.shared_data['slam_trajectory'] = self.slam_system.slam.get_trajectory()

                    # Get motion statistics
                    motion_stats = self.slam_system.slam.get_motion_stats()

                    frame_count += 1

                    # Detailed status every 3 seconds (like in original main_enhanced)
                    current_time = time.time()
                    if current_time - last_stats_time >= 3.0:
                        trajectory = self.slam_system.slam.get_trajectory()
                        map_size = len(self.slam_system.slam.get_map().points) if self.slam_system.slam.get_map() else 0

                        print(f"[SLAM] Frame {frame_count}: {len(trajectory)} poses, {map_size} points")

                        if motion_stats:
                            motion_rate = motion_stats.get('motion_rate', 0)
                            drift_rate = motion_stats.get('drift_detection_rate', 0)
                            scale_factor = motion_stats.get('scale_factor', 1.0)
                            print(
                                f"[SLAM] Motion: {motion_rate:.1%}, Drift: {drift_rate:.1%}, Scale: {scale_factor:.3f}")

                        last_stats_time = current_time

                    time.sleep(0.01)

                except Exception as e:
                    print(f"[SLAM] Frame processing error: {e}")
                    time.sleep(0.1)

        except Exception as e:
            print(f"[SLAM] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup using EnhancedSLAMSystem cleanup method
            if self.slam_system:
                try:
                    self.slam_system.cleanup()
                except:
                    pass
            print("[SLAM] System stopped")

    def display_slam_frame(self, frame, motion_stats, frame_count):
        """Display SLAM-specific frame."""
        display_img = cv2.resize(frame, (640, 480))

        # Add SLAM overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display_img, "SLAM System", (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img, f"Frame: {frame_count}", (10, 60), font, 0.5, (255, 255, 255), 1)

        if motion_stats:
            motion_rate = motion_stats.get('motion_rate', 0)
            cv2.putText(display_img, f"Motion: {motion_rate:.1%}", (10, 90), font, 0.5, (255, 255, 255), 1)

        # Position window
        cv2.namedWindow("SLAM View", cv2.WINDOW_NORMAL)
        cv2.moveWindow("SLAM View", *self.window_positions['slam'])
        cv2.imshow("SLAM View", display_img)
        cv2.waitKey(1)

    def display_detection_frame(self, frame, result):
        """Display detection-specific frame."""
        display_img = cv2.resize(frame, (640, 480))

        # Draw detections
        for detection in result.detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            # Scale coordinates for display
            x1 = int(x1 * 640 / frame.shape[1])
            y1 = int(y1 * 480 / frame.shape[0])
            x2 = int(x2 * 640 / frame.shape[1])
            y2 = int(y2 * 480 / frame.shape[0])

            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(display_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add detection info
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display_img, "Detection System", (10, 30), font, 0.7, (0, 0, 255), 2)
        cv2.putText(display_img, f"Objects: {len(result.detections)}", (10, 60), font, 0.5, (255, 255, 255), 1)
        cv2.putText(display_img, f"FPS: {result.fps:.1f}", (10, 90), font, 0.5, (255, 255, 255), 1)

        # Position window
        cv2.namedWindow("Detection View", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Detection View", *self.window_positions['detection'])
        cv2.imshow("Detection View", display_img)
        cv2.waitKey(1)

    def display_combined_view(self):
        """Display combined view in separate thread."""
        while self.running:
            with self.shared_data['lock']:
                slam_frame = self.shared_data['slam_frame']
                detection_frame = self.shared_data['detection_frame']

                if slam_frame is not None and detection_frame is not None:
                    # Create side-by-side view
                    slam_small = cv2.resize(slam_frame, (320, 240))
                    det_small = cv2.resize(detection_frame, (320, 240))
                    combined = np.hstack([slam_small, det_small])

                    # Add labels
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(combined, "SLAM", (10, 30), font, 0.7, (0, 255, 0), 2)
                    cv2.putText(combined, "Detection", (330, 30), font, 0.7, (0, 0, 255), 2)

                    # Position window
                    cv2.namedWindow("Combined View", cv2.WINDOW_NORMAL)
                    cv2.moveWindow("Combined View", *self.window_positions['combined'])
                    cv2.imshow("Combined View", combined)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.stop()
                        break

            time.sleep(0.03)  # ~30 FPS for combined view

    def start(self):
        """Start both systems."""
        self.running = True

        # FIX: Start SLAM first to establish camera connection
        print("\nStarting SLAM system first...")
        self.slam_thread = threading.Thread(target=self.run_slam_system, daemon=True)
        self.slam_thread.start()

        # FIX: Add delay to ensure SLAM initializes camera properly
        time.sleep(2.0)

        # Then start Detection system (will reuse camera)
        print("Starting Detection system...")
        self.detection_thread = threading.Thread(target=self.run_detection_system, daemon=True)
        self.detection_thread.start()

        # FIX: Add delay before starting combined view
        time.sleep(1.0)

        # Run combined view in main thread
        print("\n" + "=" * 60)
        print("Systems running. Press 'q' in any window to quit")
        print("=" * 60 + "\n")

        self.display_combined_view()

    def stop(self):
        """Stop both systems gracefully."""
        print("\n" + "=" * 60)
        print("Shutting down unified system...")
        print("=" * 60)

        self.running = False

        # Wait for threads to finish
        if self.slam_thread:
            self.slam_thread.join(timeout=5)
        if self.detection_thread:
            self.detection_thread.join(timeout=5)

        # Close all windows
        cv2.destroyAllWindows()

        # Clean up shared camera manager
        if self.shared_camera_manager:
            self.shared_camera_manager.cleanup()
            print("Shared camera manager cleaned up")

        print("Unified system stopped successfully")

    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\nReceived interrupt signal...")
        self.stop()
        sys.exit(0)


if __name__ == "__main__":
    system = UnifiedSystem()
    try:
        system.start()
    except Exception as e:
        print(f"System error: {e}")
        system.stop()
    finally:
        cv2.destroyAllWindows()