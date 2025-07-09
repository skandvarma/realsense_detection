#!/usr/bin/env python3
"""
Comprehensive camera testing and validation utility for RealSense D435i.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ConfigManager
from src.utils.logger import get_logger, PerformanceMonitor
from src.camera.realsense_manager import RealSenseManager
from src.camera.depth_processor import DepthProcessor


class CameraTestSuite:
    """Comprehensive test suite for RealSense camera functionality."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize test suite.

        Args:
            config_path: Path to configuration file
        """
        self.logger = get_logger("CameraTestSuite")
        self.performance = PerformanceMonitor()

        # Load configuration
        try:
            self.config = ConfigManager.load_config(config_path)
            self.logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            # Use minimal default config for testing
            self.config = self._get_default_config()

        # Initialize components
        self.camera_manager = None
        self.depth_processor = None

        # Test results
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'overall_status': 'PENDING'
        }

        # Interactive mode settings
        self.interactive_mode = False
        self.save_frames = False
        self.output_dir = Path("test_output")
        self.output_dir.mkdir(exist_ok=True)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get minimal default configuration for testing."""
        return {
            'camera': {
                'device_id': 0,
                'resolution': {'width': 640, 'height': 480},
                'streams': {
                    'color': {'enabled': True, 'width': 640, 'height': 480, 'framerate': 30, 'format': 'RGB8'},
                    'depth': {'enabled': True, 'width': 640, 'height': 480, 'framerate': 30, 'format': 'Z16'}
                },
                'alignment': {'align_depth_to_color': True},
                'filters': {
                    'spatial': {'enabled': False},
                    'temporal': {'enabled': False},
                    'hole_filling': {'enabled': False}
                },
                'depth': {'min_distance': 0.1, 'max_distance': 10.0}
            }
        }

    def run_all_tests(self) -> bool:
        """
        Run all automated tests.

        Returns:
            True if all tests passed, False otherwise
        """
        self.logger.info("Starting comprehensive camera test suite")

        tests = [
            ('device_detection', self.test_device_detection),
            ('camera_initialization', self.test_camera_initialization),
            ('stream_configuration', self.test_stream_configuration),
            ('frame_capture', self.test_frame_capture),
            ('depth_processing', self.test_depth_processing),
            ('performance_benchmark', self.test_performance),
            ('stability_test', self.test_stability)
        ]

        all_passed = True

        for test_name, test_func in tests:
            self.logger.info(f"Running test: {test_name}")

            try:
                result = test_func()
                self.test_results['tests'][test_name] = {
                    'status': 'PASS' if result else 'FAIL',
                    'timestamp': datetime.now().isoformat()
                }

                if result:
                    self.logger.info(f"✓ {test_name} PASSED")
                else:
                    self.logger.error(f"✗ {test_name} FAILED")
                    all_passed = False

            except Exception as e:
                self.logger.error(f"✗ {test_name} ERROR: {e}")
                self.test_results['tests'][test_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                all_passed = False

        self.test_results['overall_status'] = 'PASS' if all_passed else 'FAIL'

        # Save test results
        self._save_test_results()

        # Cleanup
        self._cleanup()

        self.logger.info(f"Test suite completed. Overall status: {self.test_results['overall_status']}")
        return all_passed

    def test_device_detection(self) -> bool:
        """Test RealSense device detection and enumeration."""
        try:
            # Create temporary camera manager for device detection
            temp_manager = RealSenseManager(self.config)
            devices = temp_manager.detect_devices()

            if not devices:
                self.logger.error("No RealSense devices detected")
                return False

            self.logger.info(f"Detected {len(devices)} device(s):")
            for i, device in enumerate(devices):
                self.logger.info(f"  Device {i}: {device['name']} (S/N: {device['serial_number']})")

            # Store device info in test results
            self.test_results['tests']['device_detection'] = {
                'device_count': len(devices),
                'devices': devices
            }

            return True

        except Exception as e:
            self.logger.error(f"Device detection failed: {e}")
            return False

    def test_camera_initialization(self) -> bool:
        """Test camera initialization with current configuration."""
        try:
            self.camera_manager = RealSenseManager(self.config)

            if not self.camera_manager.initialize_camera():
                self.logger.error("Camera initialization failed")
                return False

            # Verify camera is streaming
            if not self.camera_manager.is_streaming:
                self.logger.error("Camera not streaming after initialization")
                return False

            # Get device status
            status = self.camera_manager.get_device_status()
            self.logger.info(f"Camera status: {status}")

            # Initialize depth processor
            self.depth_processor = DepthProcessor(self.camera_manager, self.config)
            self.depth_processor.update_camera_parameters()

            return True

        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            return False

    def test_stream_configuration(self) -> bool:
        """Test individual stream configuration and basic functionality."""
        if not self.camera_manager or not self.camera_manager.is_streaming:
            self.logger.error("Camera not initialized for stream testing")
            return False

        try:
            stream_tests = {}

            # Test frame capture
            frames = self.camera_manager.capture_frames()
            if not frames or not frames.get('frame_valid', False):
                self.logger.error("Failed to capture valid frames")
                return False

            # Test individual streams
            for stream_name in ['color', 'depth']:
                if stream_name in frames:
                    frame_data = frames[stream_name]

                    if frame_data is not None and frame_data.size > 0:
                        h, w = frame_data.shape[:2]
                        stream_tests[stream_name] = {
                            'available': True,
                            'resolution': f"{w}x{h}",
                            'data_type': str(frame_data.dtype),
                            'size_bytes': frame_data.nbytes
                        }
                        self.logger.info(f"{stream_name} stream: {w}x{h}, {frame_data.dtype}")
                    else:
                        stream_tests[stream_name] = {'available': False}
                        self.logger.warning(f"{stream_name} stream not available")

            # Check for required streams
            required_streams = ['color', 'depth']
            for stream in required_streams:
                if not stream_tests.get(stream, {}).get('available', False):
                    self.logger.error(f"Required stream {stream} not available")
                    return False

            self.test_results['tests']['stream_configuration'] = {'streams': stream_tests}
            return True

        except Exception as e:
            self.logger.error(f"Stream configuration test failed: {e}")
            return False

    def test_frame_capture(self) -> bool:
        """Test frame capture reliability and timing."""
        if not self.camera_manager or not self.camera_manager.is_streaming:
            self.logger.error("Camera not initialized for frame capture testing")
            return False

        try:
            capture_count = 30
            successful_captures = 0
            capture_times = []

            self.logger.info(f"Testing frame capture reliability ({capture_count} frames)")

            for i in range(capture_count):
                start_time = time.time()

                frames = self.camera_manager.capture_frames()

                capture_time = time.time() - start_time
                capture_times.append(capture_time)

                if frames and frames.get('frame_valid', False):
                    successful_captures += 1

                    # Add performance data
                    self.performance.add_metric('frame_capture_time', capture_time)

                # Small delay to avoid overwhelming the camera
                time.sleep(0.01)

            success_rate = successful_captures / capture_count
            avg_capture_time = np.mean(capture_times) if capture_times else 0

            self.logger.info(f"Frame capture test: {successful_captures}/{capture_count} successful")
            self.logger.info(f"Success rate: {success_rate:.2%}")
            self.logger.info(f"Average capture time: {avg_capture_time * 1000:.2f}ms")

            # Store results
            self.test_results['tests']['frame_capture'] = {
                'total_attempts': capture_count,
                'successful_captures': successful_captures,
                'success_rate': success_rate,
                'avg_capture_time_ms': avg_capture_time * 1000
            }

            # Consider test passed if success rate > 90%
            return success_rate > 0.9

        except Exception as e:
            self.logger.error(f"Frame capture test failed: {e}")
            return False

    def test_depth_processing(self) -> bool:
        """Test depth processing and 3D coordinate conversion."""
        if not self.depth_processor:
            self.logger.error("Depth processor not initialized")
            return False

        try:
            # Capture frames for depth testing
            frames = self.camera_manager.capture_frames()
            if not frames or 'depth' not in frames:
                self.logger.error("No depth frame available for testing")
                return False

            depth_frame = frames['depth']

            # Test depth filtering
            filtered_depth = self.depth_processor.filter_depth_frame(depth_frame)
            if filtered_depth is None:
                self.logger.error("Depth filtering failed")
                return False

            # Test 3D coordinate conversion
            h, w = depth_frame.shape
            center_x, center_y = w // 2, h // 2

            point_3d = self.depth_processor.pixel_to_3d(center_x, center_y, depth_frame)
            if point_3d is None:
                self.logger.warning("3D conversion failed for center point")
                # Try a different point
                for offset in [50, 100, 150]:
                    test_x = min(center_x + offset, w - 1)
                    test_y = min(center_y + offset, h - 1)
                    point_3d = self.depth_processor.pixel_to_3d(test_x, test_y, depth_frame)
                    if point_3d:
                        break

            if point_3d:
                self.logger.info(
                    f"3D point conversion successful: ({point_3d.x:.3f}, {point_3d.y:.3f}, {point_3d.z:.3f})")
                depth_3d_working = True
            else:
                self.logger.warning("3D point conversion failed for all test points")
                depth_3d_working = False

            # Test depth statistics
            depth_stats = self.depth_processor.get_depth_statistics(depth_frame)
            if depth_stats:
                self.logger.info(f"Depth statistics: mean={depth_stats.get('mean_depth', 0):.3f}m, "
                                 f"valid_pixels={depth_stats.get('valid_pixel_count', 0)}")

            # Test bounding box conversion
            test_bbox = (100, 100, 200, 200)  # Arbitrary test bbox
            bbox_3d = self.depth_processor.detection_to_3d(test_bbox, depth_frame)
            bbox_3d_working = bbox_3d is not None

            if bbox_3d_working:
                self.logger.info(f"3D bbox conversion successful: center=({bbox_3d.center.x:.3f}, "
                                 f"{bbox_3d.center.y:.3f}, {bbox_3d.center.z:.3f})")

            # Store results
            self.test_results['tests']['depth_processing'] = {
                'depth_filtering': True,
                '3d_conversion': depth_3d_working,
                'bbox_3d_conversion': bbox_3d_working,
                'depth_statistics': depth_stats
            }

            return depth_3d_working  # Primary requirement

        except Exception as e:
            self.logger.error(f"Depth processing test failed: {e}")
            return False

    def test_performance(self) -> bool:
        """Test camera performance and measure FPS."""
        if not self.camera_manager or not self.camera_manager.is_streaming:
            self.logger.error("Camera not initialized for performance testing")
            return False

        try:
            test_duration = 10.0  # seconds
            frame_count = 0
            start_time = time.time()

            self.logger.info(f"Performance test starting (duration: {test_duration}s)")

            while time.time() - start_time < test_duration:
                frame_start = time.time()

                frames = self.camera_manager.capture_frames()

                frame_time = time.time() - frame_start

                if frames and frames.get('frame_valid', False):
                    frame_count += 1
                    self.performance.add_metric('performance_test_frame_time', frame_time)

            actual_duration = time.time() - start_time
            fps = frame_count / actual_duration

            # Get performance statistics
            frame_stats = self.performance.get_statistics('performance_test_frame_time')
            system_stats = self.performance.get_system_stats()

            self.logger.info(f"Performance test results:")
            self.logger.info(f"  Frames captured: {frame_count}")
            self.logger.info(f"  Duration: {actual_duration:.2f}s")
            self.logger.info(f"  Average FPS: {fps:.2f}")

            if frame_stats:
                self.logger.info(f"  Frame time: avg={frame_stats['mean'] * 1000:.2f}ms, "
                                 f"min={frame_stats['min'] * 1000:.2f}ms, max={frame_stats['max'] * 1000:.2f}ms")

            if system_stats:
                self.logger.info(f"  System load: CPU={system_stats.get('cpu_percent', 0):.1f}%, "
                                 f"Memory={system_stats.get('memory_percent', 0):.1f}%")

            # Store results
            self.test_results['tests']['performance_benchmark'] = {
                'fps': fps,
                'frame_count': frame_count,
                'duration': actual_duration,
                'frame_stats': frame_stats,
                'system_stats': system_stats
            }

            # Consider test passed if FPS >= 15 (reasonable for most applications)
            return fps >= 15.0

        except Exception as e:
            self.logger.error(f"Performance test failed: {e}")
            return False

    def test_stability(self) -> bool:
        """Test camera stability over extended period."""
        if not self.camera_manager or not self.camera_manager.is_streaming:
            self.logger.error("Camera not initialized for stability testing")
            return False

        try:
            test_duration = 30.0  # seconds
            frame_count = 0
            error_count = 0
            start_time = time.time()

            self.logger.info(f"Stability test starting (duration: {test_duration}s)")

            while time.time() - start_time < test_duration:
                try:
                    frames = self.camera_manager.capture_frames()

                    if frames and frames.get('frame_valid', False):
                        frame_count += 1
                    else:
                        error_count += 1

                except Exception as e:
                    error_count += 1
                    self.logger.warning(f"Frame capture error during stability test: {e}")

                time.sleep(0.033)  # ~30 FPS target

            actual_duration = time.time() - start_time
            error_rate = error_count / (frame_count + error_count) if (frame_count + error_count) > 0 else 1.0

            self.logger.info(f"Stability test results:")
            self.logger.info(f"  Successful frames: {frame_count}")
            self.logger.info(f"  Errors: {error_count}")
            self.logger.info(f"  Error rate: {error_rate:.2%}")

            # Store results
            self.test_results['tests']['stability_test'] = {
                'duration': actual_duration,
                'successful_frames': frame_count,
                'error_count': error_count,
                'error_rate': error_rate
            }

            # Consider test passed if error rate < 5%
            return error_rate < 0.05

        except Exception as e:
            self.logger.error(f"Stability test failed: {e}")
            return False

    def run_interactive_test(self):
        """Run interactive test with live camera feed."""
        self.interactive_mode = True

        try:
            # Initialize camera if not already done
            if not self.camera_manager:
                self.camera_manager = RealSenseManager(self.config)
                if not self.camera_manager.initialize_camera():
                    self.logger.error("Failed to initialize camera for interactive test")
                    return

                self.depth_processor = DepthProcessor(self.camera_manager, self.config)
                self.depth_processor.update_camera_parameters()

            self.logger.info("Starting interactive camera test")
            self.logger.info("Controls:")
            self.logger.info("  'q': Quit")
            self.logger.info("  's': Save current frame")
            self.logger.info("  'd': Toggle display mode (single/dual/depth-only)")
            self.logger.info("  'f': Show frame info")
            self.logger.info("  'p': Show performance stats")
            self.logger.info("  'o': Toggle overlay mode (side-by-side/picture-in-picture/separate)")

            display_mode = 'dual'  # 'color', 'depth', 'dual'
            overlay_mode = 'side_by_side'  # 'side_by_side', 'pip', 'separate'
            frame_counter = 0
            fps_counter = 0
            fps_start_time = time.time()

            while True:
                loop_start = time.time()

                # Capture frames
                frames = self.camera_manager.capture_frames()

                if not frames or not frames.get('frame_valid', False):
                    self.logger.warning("No valid frames received")
                    continue

                frame_counter += 1
                fps_counter += 1

                # Prepare display based on mode
                color_frame = None
                depth_frame_colored = None

                # Prepare color frame
                if 'color' in frames:
                    color_frame = frames['color']
                    if color_frame.shape[2] == 3:  # RGB to BGR for OpenCV
                        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)

                # Prepare depth frame
                if 'depth' in frames:
                    depth_frame_colored = self.depth_processor.create_depth_colormap(frames['depth'])

                # Create display frame based on mode
                if display_mode == 'color' and color_frame is not None:
                    display_frame = color_frame
                    window_title = "RealSense Test - Color View"
                elif display_mode == 'depth' and depth_frame_colored is not None:
                    display_frame = depth_frame_colored
                    window_title = "RealSense Test - Depth View"
                elif display_mode == 'dual' and color_frame is not None and depth_frame_colored is not None:
                    if overlay_mode == 'separate':
                        # Display in separate windows
                        # Add frame info to each window
                        color_display = color_frame.copy()
                        depth_display = depth_frame_colored.copy()

                        # Add overlays to both frames
                        for frame, label in [(color_display, "COLOR"), (depth_display, "DEPTH")]:
                            cv2.putText(frame, f"Frame: {frame_counter}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            if fps > 0:
                                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(frame, f"Mode: {label}", (10, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Display separate windows
                        cv2.imshow("RealSense Test - Color", color_display)
                        cv2.imshow("RealSense Test - Depth", depth_display)
                        display_frame = None  # Skip main display
                    else:
                        display_frame = self._create_dual_display(color_frame, depth_frame_colored, overlay_mode)
                        window_title = f"RealSense Test - Dual View ({overlay_mode.replace('_', ' ').title()})"
                else:
                    # Fallback to available frame
                    if color_frame is not None:
                        display_frame = color_frame
                        window_title = "RealSense Test - Color View (Fallback)"
                    elif depth_frame_colored is not None:
                        display_frame = depth_frame_colored
                        window_title = "RealSense Test - Depth View (Fallback)"
                    else:
                        continue

                # FPS calculation
                current_time = time.time()
                if current_time - fps_start_time >= 1.0:
                    fps = fps_counter / (current_time - fps_start_time)
                    fps_counter = 0
                    fps_start_time = current_time
                else:
                    fps = 0

                # Add overlay information and display
                if display_frame is not None:
                    # Add text overlay for combined display
                    cv2.putText(display_frame, f"Frame: {frame_counter}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if fps > 0:
                        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.putText(display_frame, f"Mode: {display_mode.upper()}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if display_mode == 'dual' and overlay_mode != 'separate':
                        cv2.putText(display_frame, f"Layout: {overlay_mode.replace('_', ' ').title()}", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Display main frame
                    cv2.imshow(window_title, display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_current_frame(frames, frame_counter)
                elif key == ord('d'):
                    # Cycle through display modes
                    modes = ['dual', 'color', 'depth']
                    current_index = modes.index(display_mode)
                    display_mode = modes[(current_index + 1) % len(modes)]
                    self.logger.info(f"Switched to {display_mode} view")
                elif key == ord('o'):
                    # Toggle overlay mode for dual view
                    if display_mode == 'dual':
                        modes = ['side_by_side', 'pip', 'separate']
                        current_index = modes.index(overlay_mode)
                        overlay_mode = modes[(current_index + 1) % len(modes)]
                        self.logger.info(f"Switched to {overlay_mode.replace('_', ' ')} layout")

                        # Close separate windows if switching away from separate mode
                        if overlay_mode != 'separate':
                            cv2.destroyWindow("RealSense Test - Color")
                            cv2.destroyWindow("RealSense Test - Depth")
                elif key == ord('f'):
                    self._show_frame_info(frames)
                elif key == ord('p'):
                    self._show_performance_info()

                # Performance tracking
                loop_time = time.time() - loop_start
                self.performance.add_metric('interactive_loop_time', loop_time)

            cv2.destroyAllWindows()
            self.logger.info("Interactive test completed")

        except KeyboardInterrupt:
            self.logger.info("Interactive test interrupted by user")
        except Exception as e:
            self.logger.error(f"Interactive test failed: {e}")
        finally:
            cv2.destroyAllWindows()
            # Ensure all specific windows are closed
            for window_name in ["RealSense Test - Color", "RealSense Test - Depth"]:
                try:
                    cv2.destroyWindow(window_name)
                except:
                    pass

    def _create_dual_display(self, color_frame: np.ndarray, depth_frame: np.ndarray,
                             layout: str = 'side_by_side') -> np.ndarray:
        """
        Create dual display combining color and depth frames.

        Args:
            color_frame: Color frame (BGR format)
            depth_frame: Colorized depth frame
            layout: Layout type ('side_by_side', 'pip', or 'separate')

        Returns:
            Combined display frame (None if separate windows)
        """
        try:
            if layout == 'side_by_side':
                # Ensure both frames have the same height
                h1, w1 = color_frame.shape[:2]
                h2, w2 = depth_frame.shape[:2]

                target_height = min(h1, h2)

                # Resize frames to same height if needed
                if h1 != target_height:
                    color_frame = cv2.resize(color_frame, (int(w1 * target_height / h1), target_height))
                if h2 != target_height:
                    depth_frame = cv2.resize(depth_frame, (int(w2 * target_height / h2), target_height))

                # Add labels
                cv2.putText(color_frame, "COLOR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(depth_frame, "DEPTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Combine horizontally
                combined = np.hstack((color_frame, depth_frame))
                return combined

            elif layout == 'pip':
                # Picture-in-picture: depth in corner of color frame
                h_color, w_color = color_frame.shape[:2]

                # Calculate PiP size (1/4 of the main frame)
                pip_height = h_color // 4
                pip_width = w_color // 4

                # Resize depth frame for PiP
                depth_pip = cv2.resize(depth_frame, (pip_width, pip_height))

                # Create a copy of color frame
                combined = color_frame.copy()

                # Position PiP in top-right corner with some margin
                margin = 10
                y_start = margin
                y_end = y_start + pip_height
                x_start = w_color - pip_width - margin
                x_end = x_start + pip_width

                # Add border around PiP
                cv2.rectangle(combined, (x_start - 2, y_start - 2), (x_end + 2, y_end + 2), (255, 255, 255), 2)

                # Overlay depth PiP
                combined[y_start:y_end, x_start:x_end] = depth_pip

                # Add label to PiP
                cv2.putText(combined, "DEPTH", (x_start + 5, y_start + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                return combined

        except Exception as e:
            self.logger.warning(f"Error creating dual display: {e}")
            # Fallback to color frame
            return color_frame

    def _save_current_frame(self, frames: Dict[str, Any], frame_number: int):
        """Save current frame data."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if 'color' in frames:
                color_frame = frames['color']
                if color_frame.shape[2] == 3:  # RGB to BGR for saving
                    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)

                color_path = self.output_dir / f"color_frame_{timestamp}_{frame_number:06d}.png"
                cv2.imwrite(str(color_path), color_frame)
                self.logger.info(f"Color frame saved: {color_path}")

            if 'depth' in frames:
                depth_frame = frames['depth']

                # Save raw depth
                depth_path = self.output_dir / f"depth_frame_{timestamp}_{frame_number:06d}.png"
                cv2.imwrite(str(depth_path), depth_frame)

                # Save colorized depth
                colorized_depth = self.depth_processor.create_depth_colormap(depth_frame)
                colorized_path = self.output_dir / f"depth_colorized_{timestamp}_{frame_number:06d}.png"
                cv2.imwrite(str(colorized_path), colorized_depth)

                self.logger.info(f"Depth frames saved: {depth_path}")

        except Exception as e:
            self.logger.error(f"Error saving frame: {e}")

    def _show_frame_info(self, frames: Dict[str, Any]):
        """Display current frame information."""
        self.logger.info("=== Frame Information ===")

        for stream_name, frame_data in frames.items():
            if isinstance(frame_data, np.ndarray):
                h, w = frame_data.shape[:2]
                self.logger.info(f"{stream_name}: {w}x{h}, {frame_data.dtype}, {frame_data.nbytes} bytes")
            elif stream_name.endswith('_timestamp'):
                self.logger.info(f"{stream_name}: {frame_data}")

        if 'depth' in frames and self.depth_processor:
            depth_stats = self.depth_processor.get_depth_statistics(frames['depth'])
            if depth_stats:
                self.logger.info(f"Depth stats: mean={depth_stats.get('mean_depth', 0):.3f}m, "
                                 f"valid_pixels={depth_stats.get('valid_pixel_count', 0)}")

    def _show_performance_info(self):
        """Display performance information."""
        self.performance.log_performance_summary()

    def _save_test_results(self):
        """Save test results to file."""
        try:
            results_path = self.output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            with open(results_path, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)

            self.logger.info(f"Test results saved to: {results_path}")

        except Exception as e:
            self.logger.error(f"Error saving test results: {e}")

    def _cleanup(self):
        """Clean up resources."""
        try:
            if self.camera_manager:
                self.camera_manager.cleanup()

            cv2.destroyAllWindows()

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


def main():
    """Main function for camera testing script."""
    parser = argparse.ArgumentParser(description="RealSense Camera Test Suite")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run interactive test with live feed")
    parser.add_argument("--test", "-t", choices=['all', 'basic', 'performance', 'stability'],
                        default='basic', help="Test suite to run")
    parser.add_argument("--output", "-o", default="test_output",
                        help="Output directory for test results")

    args = parser.parse_args()

    # Set up output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Create test suite
    test_suite = CameraTestSuite(args.config)
    test_suite.output_dir = output_dir

    try:
        if args.interactive:
            # Run interactive test
            test_suite.run_interactive_test()
        else:
            # Run automated tests
            if args.test == 'all':
                success = test_suite.run_all_tests()
            elif args.test == 'basic':
                # Run basic tests only
                basic_tests = [
                    test_suite.test_device_detection,
                    test_suite.test_camera_initialization,
                    test_suite.test_stream_configuration,
                    test_suite.test_frame_capture
                ]
                success = all(test() for test in basic_tests)
            elif args.test == 'performance':
                # Initialize camera first
                if test_suite.test_camera_initialization():
                    success = test_suite.test_performance()
                else:
                    success = False
            elif args.test == 'stability':
                # Initialize camera first
                if test_suite.test_camera_initialization():
                    success = test_suite.test_stability()
                else:
                    success = False

            # Exit with appropriate code
            sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Test suite failed with error: {e}")
        sys.exit(1)
    finally:
        test_suite._cleanup()


if __name__ == "__main__":
    main()