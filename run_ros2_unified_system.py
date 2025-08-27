#!/usr/bin/env python3
"""
Unified ROS2-based Detection System that runs alongside RTAB-Map.
RTAB-Map runs as a separate ROS2 process, Detection uses ROS2 topics.
"""

import sys
import os
import time
import threading
import signal
import cv2
import numpy as np
from pathlib import Path

# Add paths for imports
sys.path.insert(0, os.path.dirname(__file__))

# Environment setup for stability
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from src.camera.realsense_manager import RealSenseDetectionCamera, CameraShareManager
from src.detection import DetectorFactory, Postprocessor
from src.camera.depth_processor import DepthProcessor
from src.utils.config import ConfigManager
from src.utils.logger import get_logger

# Optional: Speech imports for Grounding DINO
try:
    import pyaudio
    import wave
    import tempfile
    import whisper

    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False


class UnifiedROS2System:
    """Unified system for running Detection with ROS2 camera topics alongside RTAB-Map."""

    def __init__(self, detection_headless=False, config_path="config.yaml"):
        self.detection_headless = detection_headless
        self.config_path = config_path
        self.running = False
        self.detection_thread = None
        self.detection_system = None
        self.shared_camera_manager = None
        self.detection_prompt = None
        self.logger = get_logger("UnifiedROS2System")

        # Load configuration
        self.config = self.load_config()

        # Optional: load Whisper model for speech input
        self.whisper_model = None
        if SPEECH_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("base")
                self.logger.info("Whisper model loaded for speech input")
            except Exception as e:
                self.logger.warning(f"Failed to load Whisper model: {e}")

        # Signal handling
        signal.signal(signal.SIGINT, self.signal_handler)

        print("=" * 60)
        print("UNIFIED ROS2 DETECTION SYSTEM")
        print(f"Detection: {'HEADLESS' if self.detection_headless else 'GUI'} mode")
        print("RTAB-Map: Run separately via ROS2")
        print("=" * 60)

        self.check_ros2_topics()
        self.initialize_shared_camera()

    def check_ros2_topics(self):
        """Check if ROS2 camera topics are available."""
        import subprocess
        try:
            result = subprocess.run(['ros2', 'topic', 'list'],
                                    capture_output=True, text=True, timeout=5)
            topics = result.stdout

            if '/camera/camera/color/image_raw' not in topics:
                self.logger.warning("Camera topics not found! Make sure the RealSense ROS2 node is running.")
                self.logger.info("Run: ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true")
                print("\nWARNING: Camera topics not detected. The system will wait for them...")
            else:
                self.logger.info("ROS2 camera topics detected successfully")

        except Exception as e:
            self.logger.warning(f"Could not check ROS2 topics: {e}")

    def load_config(self):
        """Load configuration from YAML."""
        try:
            config = ConfigManager.load_config(self.config_path)

            # Ensure ROS2 configuration is present
            if 'camera' not in config:
                config['camera'] = {}

            if 'ros2' not in config['camera']:
                config['camera']['ros2'] = {
                    'enabled': True,
                    'color_topic': '/camera/camera/color/image_raw',
                    'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
                    'camera_info_topic': '/camera/camera/color/camera_info'
                }

            self.logger.info("Configuration loaded successfully")
            return config

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            # Return minimal config
            return {
                'camera': {
                    'ros2': {
                        'enabled': True,
                        'color_topic': '/camera/camera/color/image_raw',
                        'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
                        'camera_info_topic': '/camera/camera/color/camera_info'
                    },
                    'width': 640,
                    'height': 480,
                    'fps': 30
                },
                'detection': {
                    'confidence_threshold': 0.5,
                    'nms_threshold': 0.4
                }
            }

    def get_speech_prompt(self):
        """Record speech and transcribe into a text prompt using local Whisper."""
        if not SPEECH_AVAILABLE or not self.whisper_model:
            return None

        try:
            chunk = 1024
            fmt = pyaudio.paInt16
            channels = 1
            rate = 16000
            record_seconds = 5

            audio = pyaudio.PyAudio()
            print(f"\n🎤 Recording for {record_seconds} seconds... Speak your detection prompt:")

            stream = audio.open(format=fmt, channels=channels, rate=rate,
                                input=True, frames_per_buffer=chunk)
            frames = []
            for _ in range(int(rate / chunk * record_seconds)):
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            audio.terminate()

            print("Recording finished. Transcribing...")
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(audio.get_sample_size(fmt))
                wf.setframerate(rate)
                wf.writeframes(b''.join(frames))

            result = self.whisper_model.transcribe(temp_file.name)
            os.unlink(temp_file.name)

            # Format transcription for Grounding DINO
            text = result["text"].strip().lower()
            # Replace commas with dots for Grounding DINO format
            text = text.replace(",", " .").replace(".", " .")
            # Clean up multiple spaces and dots
            text = " ".join(text.split())

            return text

        except Exception as e:
            self.logger.error(f"Speech input failed: {e}")
            return None

    def initialize_shared_camera(self):
        """Initialize ROS2-based shared camera manager."""
        try:
            self.logger.info("Initializing ROS2-based shared camera manager...")
            self.shared_camera_manager = CameraShareManager()

            if not self.shared_camera_manager.initialize_camera(self.config['camera']):
                raise RuntimeError("Failed to initialize ROS2 camera subscriptions")

            # Wait for camera to start publishing
            self.logger.info("Waiting for camera data...")
            timeout = 10
            start_time = time.time()

            while time.time() - start_time < timeout:
                frames = self.shared_camera_manager.capture_frames()
                if frames and frames.get('frame_valid'):
                    self.logger.info("Camera data received successfully")
                    break
                time.sleep(0.1)
            else:
                self.logger.warning("Timeout waiting for camera data, continuing anyway...")

        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def run_detection_system(self):
        """Run detection system with ROS2 camera feed."""
        try:
            self.logger.info("Initializing detection system...")

            # Create camera interface
            camera = RealSenseDetectionCamera(self.config)

            # Initialize detector
            try:
                factory = DetectorFactory()
                detector = factory.create_detector(self.config)
            except Exception as e:
                self.logger.error(f"Failed to initialize detector: {e}")
                self.logger.info("Running without detection (camera view only)")
                detector = None

            if detector:
                # Set text prompt if using Grounding DINO
                if hasattr(detector, 'is_grounding_dino') and detector.is_grounding_dino:
                    if self.detection_prompt:
                        detector.update_text_prompt(self.detection_prompt)
                        self.logger.info(f"Using Grounding DINO with prompt: '{self.detection_prompt}'")
                    else:
                        self.logger.warning("No prompt provided for Grounding DINO")

            # Initialize processors
            depth_processor = DepthProcessor(self.shared_camera_manager, self.config)
            depth_processor.update_camera_parameters()
            postprocessor = Postprocessor(self.config, depth_processor)

            self.detection_system = {
                'camera': camera,
                'detector': detector,
                'postprocessor': postprocessor,
                'depth_processor': depth_processor
            }

            self.logger.info("Detection system started")
            frame_count = 0
            last_log_time = time.time()
            detection_times = []

            while self.running:
                try:
                    # Get frames from ROS2 topics
                    rgb, depth = camera.get_frames()

                    if rgb is not None:
                        start_time = time.time()

                        if detector:
                            # Run detection
                            kwargs = {'frame_id': frame_count}

                            # For Grounding DINO, pass text prompt
                            if hasattr(detector, 'is_grounding_dino') and detector.is_grounding_dino:
                                kwargs['text_prompt'] = self.detection_prompt or detector.get_current_prompt()

                            result = detector.detect(rgb, **kwargs)

                            if result.success:
                                # Postprocess with depth
                                result = postprocessor.process_detection_result(
                                    result, depth, frame_count
                                )

                                # Track timing
                                detection_time = time.time() - start_time
                                detection_times.append(detection_time)

                                # Log periodically
                                current_time = time.time()
                                if current_time - last_log_time >= 3.0:
                                    avg_detection_time = np.mean(detection_times[-30:]) if detection_times else 0
                                    detection_fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0

                                    self.logger.info(
                                        f"Frame {frame_count}: {len(result.detections)} objects detected | "
                                        f"Detection FPS: {detection_fps:.1f} | Model FPS: {result.fps:.1f}"
                                    )
                                    last_log_time = current_time

                                # Display GUI
                                if not self.detection_headless:
                                    self.display_detection_frame(rgb, result, frame_count, depth)
                        else:
                            # No detector, just show camera feed
                            result = None
                            if not self.detection_headless:
                                self.display_camera_only(rgb, depth, frame_count)

                        frame_count += 1
                    else:
                        # No frames received
                        time.sleep(0.01)

                except Exception as e:
                    if frame_count % 100 == 0:
                        self.logger.error(f"Frame processing error: {e}")
                    time.sleep(0.01)

                # Check for quit key
                if not self.detection_headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("Quit requested")
                        self.stop()
                        break
                    elif key == ord('s'):
                        # Save current frame
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(f"detection_frame_{timestamp}.jpg", rgb)
                        self.logger.info(f"Frame saved: detection_frame_{timestamp}.jpg")

        except Exception as e:
            self.logger.error(f"Detection system error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            if self.detection_system:
                if 'detector' in self.detection_system and self.detection_system['detector']:
                    try:
                        self.detection_system['detector'].cleanup()
                    except:
                        pass
                if 'camera' in self.detection_system:
                    try:
                        self.detection_system['camera'].cleanup()
                    except:
                        pass

            cv2.destroyAllWindows()
            self.logger.info("Detection system stopped")

    def display_detection_frame(self, frame, result, frame_count, depth_frame=None):
        """Display detection results with overlays."""
        display_img = frame.copy()

        # Draw detections
        for detection in result.detections:
            x1, y1, x2, y2 = map(int, detection.bbox)

            # Choose color based on confidence
            conf = detection.confidence
            color = (0, int(255 * conf), int(255 * (1 - conf)))

            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)

            # Create label
            label = f"{detection.class_name}: {conf:.2f}"
            if hasattr(detection, 'distance') and detection.distance > 0:
                label += f" | {detection.distance:.1f}m"
            if detection.detection_id is not None:
                label += f" | ID:{detection.detection_id}"

            # Draw label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display_img, (x1, y1 - label_size[1] - 4),
                          (x1 + label_size[0], y1), color, -1)
            cv2.putText(display_img, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(display_img, (center_x, center_y), 3, color, -1)

        # Add info overlay
        self._add_info_overlay(display_img, result, frame_count)

        # Create depth visualization if available
        if depth_frame is not None:
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03),
                cv2.COLORMAP_JET
            )
            # Resize for side-by-side display
            h, w = display_img.shape[:2]
            depth_resized = cv2.resize(depth_colormap, (w // 3, h // 3))
            # Add depth preview to corner
            display_img[10:10 + h // 3, w - w // 3 - 10:w - 10] = depth_resized
            cv2.rectangle(display_img, (w - w // 3 - 10, 10), (w - 10, 10 + h // 3), (255, 255, 255), 2)
            cv2.putText(display_img, "Depth", (w - w // 3 - 5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("ROS2 Detection System", display_img)

    def display_camera_only(self, frame, depth_frame, frame_count):
        """Display camera feed when no detector is available."""
        display_img = frame.copy()

        # Add text overlay
        cv2.putText(display_img, "CAMERA VIEW (No Detector)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(display_img, f"Frame: {frame_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add depth preview if available
        if depth_frame is not None:
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03),
                cv2.COLORMAP_JET
            )
            h, w = display_img.shape[:2]
            depth_resized = cv2.resize(depth_colormap, (w // 3, h // 3))
            display_img[10:10 + h // 3, w - w // 3 - 10:w - 10] = depth_resized

        cv2.imshow("ROS2 Detection System", display_img)

    def _add_info_overlay(self, display_img, result, frame_count):
        """Add information overlay to the display."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w = display_img.shape[:2]

        # Create semi-transparent overlay for info
        overlay = display_img.copy()
        cv2.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0, display_img)

        # Add info text
        cv2.putText(display_img, "ROS2 DETECTION SYSTEM", (10, 25), font, 0.7, (0, 255, 255), 2)
        cv2.putText(display_img, f"Objects: {len(result.detections)}", (10, 50), font, 0.5, (255, 255, 255), 1)
        cv2.putText(display_img, f"Model FPS: {result.fps:.1f}", (10, 70), font, 0.5, (255, 255, 255), 1)
        cv2.putText(display_img, f"Frame: {frame_count}", (10, 90), font, 0.5, (255, 255, 255), 1)
        cv2.putText(display_img, f"Model: {result.model_name}", (10, 110), font, 0.5, (255, 255, 255), 1)

        # Add camera status
        status = self.shared_camera_manager.get_device_status()
        status_text = f"Camera: {'Active' if status['is_streaming'] else 'Inactive'}"
        cv2.putText(display_img, status_text, (10, h - 40), font, 0.5, (0, 255, 0), 1)

        # Add RTAB-Map reminder
        cv2.putText(display_img, "RTAB-Map: Run separately via ROS2", (10, h - 20), font, 0.4, (200, 200, 200), 1)

        # Instructions
        cv2.putText(display_img, "Press 'q' to quit, 's' to save frame", (w - 300, h - 20),
                    font, 0.5, (255, 255, 0), 1)

    def start(self):
        """Start the unified system."""
        # Check for Grounding DINO prompt if needed
        detector_config = self.config.get('detection', {})
        active_model = detector_config.get('active_model', 'yolo')

        if active_model == 'detr' and detector_config.get('detr', {}).get('variant') == 'grounding-dino':
            # Try speech input first
            if SPEECH_AVAILABLE and self.whisper_model:
                print("\n🎤 Using speech input for Grounding DINO prompt...")
                self.detection_prompt = self.get_speech_prompt()
                if self.detection_prompt:
                    print(f"Speech prompt: '{self.detection_prompt}'")

            # Fallback to manual input
            if not self.detection_prompt:
                self.detection_prompt = input(
                    "\nEnter detection prompt for Grounding DINO (e.g., 'person . car . dog'): "
                ).strip()

            if self.detection_prompt:
                print(f"Using detection prompt: '{self.detection_prompt}'")

        self.running = True

        # Start detection system
        self.logger.info("Starting detection system...")
        self.detection_thread = threading.Thread(target=self.run_detection_system, daemon=True)
        self.detection_thread.start()

        print("\n" + "=" * 60)
        print("SYSTEM RUNNING")
        print("Detection: Active" + (" (headless)" if self.detection_headless else " (GUI)"))
        print("RTAB-Map: Launch separately with:")
        print("  ros2 launch rtabmap_launch rtabmap.launch.py \\")
        print("    rgb_topic:=/camera/camera/color/image_raw \\")
        print("    depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \\")
        print("    camera_info_topic:=/camera/camera/color/camera_info")
        print("=" * 60 + "\n")

        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop the system gracefully."""
        print("\n" + "=" * 60)
        print("Shutting down unified system...")

        self.running = False

        # Wait for thread
        if self.detection_thread:
            self.detection_thread.join(timeout=5)

        # Close windows
        cv2.destroyAllWindows()

        # Cleanup camera
        if self.shared_camera_manager:
            self.shared_camera_manager.cleanup()

        print("Unified system stopped successfully")
        print("=" * 60)

    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\nReceived interrupt signal...")
        self.stop()
        sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified ROS2 Detection System")
    parser.add_argument("--detection-headless", action="store_true",
                        help="Run detection without GUI display")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()

    # Create and start the system
    system = UnifiedROS2System(
        detection_headless=args.detection_headless,
        config_path=args.config
    )

    try:
        system.start()
    except Exception as e:
        print(f"System error: {e}")
        system.stop()
    finally:
        cv2.destroyAllWindows()