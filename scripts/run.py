#!/usr/bin/env python3
"""
Complete CUDA-optimized RealSense detection application with real-time pipeline.
Main application coordinator with GPU resource management and interactive controls.
"""

import os
import sys
import time
import argparse
import signal
import threading
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ConfigManager
from src.utils.logger import get_logger, PerformanceMonitor, DataRecorder


class DetectionApplication:
    """Main application coordinator with GPU resource management."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the detection application.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.logger = get_logger("DetectionApplication")

        # Load configuration
        try:
            self.config = ConfigManager.load_config(config_path)
            self.logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

        # Core components
        self.pipeline: Optional[RealSenseDetectionPipeline] = None
        self.visualizer: Optional[CUDAVisualizer] = None
        self.memory_manager: Optional[CUDAMemoryManager] = None

        # Application state
        self.running = False
        self.paused = False
        self.performance_monitor = PerformanceMonitor(window_size=100)

        # Interactive control
        self.interactive_mode = True
        self.auto_model_switching = False
        self.recording_session = False

        # Statistics tracking
        self.session_stats = {
            'start_time': 0,
            'frames_processed': 0,
            'detections_count': 0,
            'tracks_created': 0,
            'model_switches': 0,
            'errors_count': 0
        }

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info("DetectionApplication initialized")

    def initialize(self) -> bool:
        """
        Initialize all application components.

        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing CUDA-accelerated detection application...")

            # Check CUDA availability
            self._check_system_requirements()

            # Initialize GPU memory manager
            self.memory_manager = CUDAMemoryManager(self.config)
            self.logger.info("GPU memory manager initialized")

            # Initialize main pipeline
            self.pipeline = RealSenseDetectionPipeline(self.config)
            if not self.pipeline.initialize():
                self.logger.error("Failed to initialize detection pipeline")
                return False

            # Initialize visualizer
            self.visualizer = CUDAVisualizer(self.config, self.memory_manager)
            self.logger.info("Visualization system initialized")

            # Setup error recovery callbacks
            self.pipeline.add_recovery_callback(self._handle_pipeline_error)

            self.logger.info("Application initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Application initialization failed: {e}")
            return False

    def _check_system_requirements(self):
        """Check system requirements and log capabilities."""
        import torch

        # CUDA check
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            self.logger.info(f"CUDA available with {device_count} GPU(s)")

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024 ** 3
                self.logger.info(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        else:
            self.logger.warning("CUDA not available - using CPU fallback")

        # Detection models check
        available_models = []
        if YOLO_AVAILABLE:
            available_models.append("YOLO")
        if DETR_AVAILABLE:
            available_models.append("DETR")

        self.logger.info(f"Available detection models: {', '.join(available_models)}")

        if not available_models:
            raise RuntimeError("No detection models available")

    def run(self, duration: Optional[float] = None,
            enable_visualization: bool = True,
            enable_recording: bool = False) -> bool:
        """
        Run the main application loop.

        Args:
            duration: Optional maximum duration in seconds
            enable_visualization: Enable real-time visualization
            enable_recording: Enable session recording

        Returns:
            True if completed successfully
        """
        if not self.pipeline:
            self.logger.error("Application not initialized")
            return False

        try:
            self.logger.info("Starting detection application with direct components")
            self._print_startup_info()

            # No separate pipeline start needed - components are ready
            self.running = True

            # Initialize session statistics
            self.session_stats['start_time'] = time.time()

            # Main execution loop using direct components
            self._main_loop_direct_components(duration, enable_visualization)

            # Session summary
            self._print_session_summary()

            return True

        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
            return True
        except Exception as e:
            self.logger.error(f"Application runtime error: {e}")
            return False
        # Stop recording
        if self.recording_session:
            self._stop_recording_session()

        # Cleanup components
        self._cleanup()

    def _print_startup_info(self):
        """Print startup information and controls."""
        print("\n" + "=" * 80)
        print("🚀 CUDA-ACCELERATED REALSENSE DETECTION PIPELINE")
        print("=" * 80)

        # System info
        print(f"📊 System Status:")
        print(f"   • Detection Model: {self.config['detection']['active_model']}")
        print(f"   • Target FPS: {self.config.get('integration', {}).get('performance', {}).get('target_fps', 30)}")
        print(f"   • GPU Acceleration: {torch.cuda.is_available()}")

        # Detection info
        if hasattr(self, 'detector') and self.detector:
            active_detector = self.detector.get_active_detector() if hasattr(self.detector,
                                                                             'get_active_detector') else self.detector
            if active_detector and hasattr(active_detector, 'model_info'):
                print(f"   • Model: {active_detector.model_info.model_name}")
                print(f"   • Device: {active_detector.model_info.inference_device}")

        # Camera info
        if hasattr(self, 'camera_manager') and self.camera_manager:
            camera_status = self.camera_manager.get_device_status()
            print(f"   • Camera: {'Connected' if camera_status.get('device_connected', False) else 'Disconnected'}")
            if camera_status.get('device_name'):
                print(f"   • Camera Model: {camera_status['device_name']}")

        # 3D processing
        print(
            f"   • 3D Processing: {'Enabled' if hasattr(self, 'depth_processor') and self.depth_processor else 'Disabled'}")

        # Controls
        if self.interactive_mode:
            print(f"\n🎮 Interactive Controls:")
            print(f"   • 'q' / Ctrl+C  - Quit application")
            print(f"   • 'p'           - Pause/Resume pipeline")
            print(f"   • 'm'           - Switch detection model")
            print(f"   • 'r'           - Start/Stop recording")
            print(f"   • 's'           - Save performance snapshot")
            print(f"   • 'c'           - Clear performance statistics")
            print(f"   • 'h'           - Show help")
            print(f"   • Space         - Process single frame (when paused)")

        print("=" * 80 + "\n")

    def _main_loop_direct_components(self, duration: Optional[float], enable_visualization: bool):
        """Main loop using direct components (working version)."""
        start_time = time.time()
        last_stats_time = start_time
        stats_interval = 5.0
        frame_count = 0

        # Initialize OpenCV windows if visualization enabled
        if enable_visualization:
            cv2.namedWindow('Detection Results', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Detection Results', 1280, 720)

        try:
            while self.running:
                loop_start = time.time()

                # Check duration limit
                if duration and (loop_start - start_time) >= duration:
                    self.logger.info(f"Duration limit reached ({duration}s)")
                    break

                # Process frame if not paused
                if not self.paused:
                    success = self._process_frame_direct(enable_visualization, frame_count)
                    if success:
                        frame_count += 1
                        self.session_stats['frames_processed'] = frame_count

                # Update statistics periodically
                if loop_start - last_stats_time >= stats_interval:
                    self._update_stats_direct(frame_count, loop_start - start_time)
                    last_stats_time = loop_start

                # Handle interactive input
                if self.interactive_mode and enable_visualization:
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:
                        self._handle_key_input_direct(key)

                # Frame rate control
                self._control_frame_rate(loop_start)

        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
            self.session_stats['errors_count'] += 1
        finally:
            if enable_visualization:
                cv2.destroyAllWindows()

    def _process_frame_direct(self, enable_visualization: bool, frame_id: int) -> bool:
        """Process a single frame using direct components."""
        try:
            # Capture frame from camera
            frames = self.camera_manager.capture_frames()
            if not frames or not frames.get('frame_valid', False):
                return False

            color_frame = frames.get('color')
            depth_frame = frames.get('depth')

            if color_frame is None:
                return False

            # Convert RGB to BGR for detection (if needed)
            if color_frame.shape[2] == 3:
                detection_image = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
            else:
                detection_image = color_frame

            # Run detection
            result = self.detector.detect(detection_image, frame_id=frame_id)

            if result.success:
                # Apply postprocessing
                enhanced_result = self.postprocessor.process_detection_result(
                    result, depth_frame, frame_id
                )

                # Update statistics
                self.session_stats['detections_count'] += len(enhanced_result.detections)

                # Visualization
                if enable_visualization:
                    self._visualize_frame_direct(detection_image, enhanced_result, depth_frame)

                return True

            return False

        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            self.session_stats['errors_count'] += 1
            return False

    def _visualize_frame_direct(self, image: np.ndarray, result, depth_frame: Optional[np.ndarray]):
        """Simple OpenCV-based visualization."""
        import cv2

        vis_image = image.copy()

        # Draw detection results
        if result and result.detections:
            for detection in result.detections:
                x1, y1, x2, y2 = map(int, detection.bbox)

                # Draw bounding box
                color = (0, 255, 0)  # Green
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = f"{detection.class_name}: {detection.confidence:.2f}"
                if detection.detection_id is not None:
                    label += f" ID:{detection.detection_id}"

                # Add 3D position if available
                if hasattr(detection, 'center_3d') and detection.center_3d != (0, 0, 0):
                    label += f" ({detection.center_3d[0]:.1f},{detection.center_3d[1]:.1f},{detection.center_3d[2]:.1f}m)"

                cv2.putText(vis_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Add performance info
        current_time = time.time()
        elapsed = current_time - self.session_stats['start_time']
        fps = self.session_stats['frames_processed'] / elapsed if elapsed > 0 else 0

        info_text = f"Frames: {self.session_stats['frames_processed']}, FPS: {fps:.1f}, Detections: {self.session_stats['detections_count']}"
        cv2.putText(vis_image, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display
        cv2.imshow('Detection Results', vis_image)

    def _update_stats_direct(self, frame_count: int, elapsed_time: float):
        """Update and display statistics for direct components."""
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        print(f"\n📊 Performance Update ({elapsed_time:.1f}s)")
        print(f"   • Frames Processed: {frame_count}")
        print(f"   • Average FPS: {fps:.1f}")
        print(f"   • Total Detections: {self.session_stats['detections_count']}")

        # Get detector performance stats
        if hasattr(self.detector, 'get_performance_stats'):
            perf_stats = self.detector.get_performance_stats()
            if 'avg_inference_time' in perf_stats:
                print(f"   • Avg Detection Time: {perf_stats['avg_inference_time'] * 1000:.1f}ms")
            if 'avg_fps' in perf_stats:
                print(f"   • Detection FPS: {perf_stats['avg_fps']:.1f}")

        # Memory usage if available
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024 ** 2
            print(f"   • GPU Memory: {memory_mb:.1f}MB")

    def _handle_key_input_direct(self, key: int):
        """Handle keyboard input for direct components."""
        if key == ord('q') or key == 27:  # 'q' or ESC
            self.running = False
        elif key == ord('p'):
            self._toggle_pause()
        elif key == ord('m'):
            self._switch_detection_model_direct()
        elif key == ord('s'):
            self._save_performance_snapshot()
        elif key == ord('h'):
            self._show_help()

    def _switch_detection_model_direct(self):
        """Switch detection model for direct components."""
        current_model = self.config['detection']['active_model']

        if current_model == 'yolo':
            new_model = 'detr'
        else:
            new_model = 'yolo'

        self.logger.info(f"Switching from {current_model} to {new_model}...")

        try:
            # Create new detector
            factory = DetectorFactory()
            new_detector = factory.create_detector_with_fallback(self.config, new_model)

            if new_detector:
                # Cleanup old detector
                if self.detector:
                    self.detector.cleanup()

                # Replace detector
                self.detector = new_detector
                self.config['detection']['active_model'] = new_model
                self.session_stats['model_switches'] += 1

                self.logger.info(f"Successfully switched to {new_model}")
            else:
                self.logger.error(f"Failed to create {new_model} detector")

        except Exception as e:
            self.logger.error(f"Model switch failed: {e}")

    def _cleanup(self):
        """Clean up direct components."""
        self.logger.info("Cleaning up application resources...")

        self.running = False

        # Cleanup detector
        if hasattr(self, 'detector') and self.detector:
            self.detector.cleanup()

        # Cleanup camera
        if hasattr(self, 'camera_manager') and self.camera_manager:
            self.camera_manager.cleanup()

        # Cleanup memory manager
        if self.memory_manager:
            self.memory_manager.cleanup()

        # Close OpenCV windows
        cv2.destroyAllWindows()

        self.logger.info("Application cleanup completed")

    def _process_single_iteration(self, enable_visualization: bool):
        """Process a single iteration of the pipeline."""
        try:
            # Process frame through pipeline
            if self.pipeline.use_threading:
                # In threaded mode, just check if pipeline is healthy
                pipeline_stats = self.pipeline.get_stats()
                if pipeline_stats.frames_processed > self.session_stats['frames_processed']:
                    self.session_stats['frames_processed'] = pipeline_stats.frames_processed

                    # Get latest frame data for visualization
                    if enable_visualization and self.visualizer:
                        # In a real implementation, you'd get the latest processed frame
                        # For now, we'll simulate with pipeline stats
                        self._update_visualization_with_stats(pipeline_stats)
            else:
                # In single-threaded mode, process frame manually
                frame = self.pipeline.process_frame()
                if frame and frame.valid:
                    self.session_stats['frames_processed'] += 1

                    # Update visualization
                    if enable_visualization and self.visualizer:
                        self._update_visualization_with_frame(frame)

                    # Update detection statistics
                    if frame.enhanced_result:
                        self.session_stats['detections_count'] += len(frame.enhanced_result.detections)

        except Exception as e:
            self.logger.error(f"Iteration processing error: {e}")
            self.session_stats['errors_count'] += 1

    def _update_visualization_with_stats(self, stats):
        """Update visualization with pipeline statistics."""
        # Create a dummy frame with statistics overlay
        import numpy as np

        # Create info frame
        info_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # This would be replaced with actual frame data in real implementation
        metadata = {
            'fps': stats.avg_fps,
            'detections': stats.frames_processed,
            'gpu_memory': stats.memory_usage_mb,
            'processing_time': stats.avg_detection_time * 1000
        }

        self.visualizer.render_frame(
            rgb_frame=info_frame,
            metadata=metadata
        )

    def _update_visualization_with_frame(self, frame):
        """Update visualization with processed frame."""
        # Extract frame data
        rgb_frame = frame.color_frame
        depth_frame = frame.depth_frame
        detection_result = frame.enhanced_result

        # Get active tracks from pipeline
        tracks = []
        if hasattr(self.pipeline, 'postprocessor') and self.pipeline.postprocessor:
            tracking_stats = self.pipeline.postprocessor.get_tracking_statistics()
            # tracks would be extracted from the tracking system

        # Prepare metadata
        metadata = {
            'frame_id': frame.frame_id,
            'detection_time': frame.detection_time,
            'total_time': frame.total_time,
            'fps': 1.0 / frame.total_time if frame.total_time > 0 else 0
        }

        self.visualizer.render_frame(
            rgb_frame=rgb_frame,
            depth_frame=depth_frame,
            detection_result=detection_result,
            tracks=tracks,
            metadata=metadata
        )

    def _update_and_display_stats(self):
        """Update and display performance statistics."""
        pipeline_stats = self.pipeline.get_stats()

        # Calculate session metrics
        current_time = time.time()
        session_duration = current_time - self.session_stats['start_time']
        avg_fps = self.session_stats['frames_processed'] / session_duration if session_duration > 0 else 0

        # Display statistics
        print(f"\n📊 Performance Update ({session_duration:.1f}s)")
        print(f"   • Frames Processed: {pipeline_stats.frames_processed}")
        print(f"   • Frames Dropped: {pipeline_stats.frames_dropped}")
        print(f"   • Average FPS: {avg_fps:.1f}")
        print(f"   • Detection FPS: {pipeline_stats.detection_fps:.1f}")
        print(f"   • GPU Memory: {pipeline_stats.memory_usage_mb:.1f}MB")
        print(f"   • GPU Utilization: {pipeline_stats.gpu_utilization:.1f}%")

        if hasattr(pipeline_stats, 'confirmed_tracks'):
            print(f"   • Active Tracks: {pipeline_stats.confirmed_tracks}")

    def _handle_user_input(self):
        """Handle interactive user input (non-blocking)."""
        # This is a simplified version - in a real implementation,
        # you'd use proper non-blocking input handling
        import select
        import sys

        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1).lower()

            if key == 'q':
                self.running = False
            elif key == 'p':
                self._toggle_pause()
            elif key == 'm':
                self._switch_detection_model()
            elif key == 'r':
                self._toggle_recording()
            elif key == 's':
                self._save_performance_snapshot()
            elif key == 'c':
                self._clear_statistics()
            elif key == 'h':
                self._show_help()
            elif key == ' ' and self.paused:
                self._process_single_frame()

    def _toggle_pause(self):
        """Toggle pipeline pause state."""
        if self.paused:
            self.pipeline.resume()
            self.paused = False
            self.logger.info("Pipeline resumed")
        else:
            self.pipeline.pause()
            self.paused = True
            self.logger.info("Pipeline paused")

    def _switch_detection_model(self):
        """Switch between available detection models."""
        current_config = self.config['detection']['active_model']

        if current_config == 'yolo' and DETR_AVAILABLE:
            new_model = 'detr'
        elif current_config == 'detr' and YOLO_AVAILABLE:
            new_model = 'yolo'
        else:
            self.logger.warning("No alternative detection model available")
            return

        self.logger.info(f"Switching from {current_config} to {new_model}...")

        if self.pipeline.switch_detection_model(new_model):
            self.config['detection']['active_model'] = new_model
            self.session_stats['model_switches'] += 1
            self.logger.info(f"Successfully switched to {new_model}")
        else:
            self.logger.error(f"Failed to switch to {new_model}")

    def _toggle_recording(self):
        """Toggle session recording."""
        if self.recording_session:
            self._stop_recording_session()
        else:
            self._start_recording_session()

    def _start_recording_session(self):
        """Start recording session."""
        if self.recording_session:
            return

        # Enable pipeline recording
        if hasattr(self.pipeline, 'data_recorder') and self.pipeline.data_recorder:
            self.pipeline.data_recorder.start_recording()

        # Enable visualization recording
        if self.visualizer:
            self.visualizer.toggle_recording()

        self.recording_session = True
        self.logger.info("Recording session started")

    def _stop_recording_session(self):
        """Stop recording session."""
        if not self.recording_session:
            return

        # Stop pipeline recording
        if hasattr(self.pipeline, 'data_recorder') and self.pipeline.data_recorder:
            self.pipeline.data_recorder.stop_recording()

        # Stop visualization recording
        if self.visualizer:
            self.visualizer.toggle_recording()

        self.recording_session = False
        self.logger.info("Recording session stopped")

    def _save_performance_snapshot(self):
        """Save current performance snapshot."""
        timestamp = int(time.time())
        filename = f"performance_snapshot_{timestamp}.json"

        # Collect comprehensive performance data
        snapshot = {
            'timestamp': timestamp,
            'session_stats': self.session_stats.copy(),
            'pipeline_stats': self.pipeline.get_stats().__dict__ if self.pipeline else {},
            'memory_stats': self.memory_manager.get_memory_stats().__dict__ if self.memory_manager else {},
            'system_info': self._get_system_info()
        }

        if self.visualizer:
            snapshot['visualization_stats'] = self.visualizer.get_visualization_stats()

        try:
            with open(filename, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            self.logger.info(f"Performance snapshot saved: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save performance snapshot: {e}")

    def _clear_statistics(self):
        """Clear performance statistics."""
        self.session_stats = {
            'start_time': time.time(),
            'frames_processed': 0,
            'detections_count': 0,
            'tracks_created': 0,
            'model_switches': 0,
            'errors_count': 0
        }

        if self.pipeline:
            # Reset pipeline statistics (if available)
            pass

        self.logger.info("Performance statistics cleared")

    def _show_help(self):
        """Display help information."""
        help_text = """
        📚 CUDA-Accelerated Detection Pipeline Help
        ==========================================

        Interactive Controls:
        • 'q' - Quit application
        • 'p' - Pause/Resume pipeline
        • 'm' - Switch detection model (YOLO ↔ DETR)
        • 'r' - Start/Stop recording
        • 's' - Save performance snapshot
        • 'c' - Clear performance statistics  
        • 'h' - Show this help
        • Space - Process single frame (when paused)

        Performance Monitoring:
        • Real-time FPS and GPU utilization
        • Detection and tracking statistics
        • Memory usage monitoring
        • Automatic performance logging

        Recording Features:
        • Video recording with GPU acceleration
        • Detection data logging (JSON/CSV)
        • Performance snapshot export
        • Session replay capabilities
        """
        print(help_text)

    def _process_single_frame(self):
        """Process a single frame when paused."""
        if not self.paused:
            return

        self.logger.info("Processing single frame...")
        frame = self.pipeline.process_frame()

        if frame and frame.valid:
            self.session_stats['frames_processed'] += 1
            if self.visualizer:
                self._update_visualization_with_frame(frame)
            print(
                f"Frame {frame.frame_id} processed: {len(frame.enhanced_result.detections) if frame.enhanced_result else 0} detections")
        else:
            print("Failed to process frame")

    def _control_frame_rate(self, loop_start: float):
        """Control frame rate to prevent excessive CPU usage."""
        if self.pipeline:
            target_frame_time = 1.0 / self.pipeline.target_fps
            elapsed = time.time() - loop_start

            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for diagnostics."""
        import platform
        import psutil
        import torch

        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 ** 3,
            'cuda_available': torch.cuda.is_available()
        }

        if torch.cuda.is_available():
            info['cuda_devices'] = torch.cuda.device_count()
            info['cuda_version'] = torch.version.cuda

        return info

    def _handle_pipeline_error(self, error: Exception):
        """Handle pipeline errors with recovery attempts."""
        self.logger.warning(f"Pipeline error occurred: {error}")
        self.session_stats['errors_count'] += 1

        # Implement recovery strategies
        if "memory" in str(error).lower():
            self.logger.info("Attempting memory cleanup...")
            if self.memory_manager:
                self.memory_manager.optimize_memory_layout()

        elif "camera" in str(error).lower():
            self.logger.info("Attempting camera reconnection...")
            # Could implement camera reconnection logic here

    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

    def _print_session_summary(self):
        """Print session summary statistics."""
        duration = time.time() - self.session_stats['start_time']
        avg_fps = self.session_stats['frames_processed'] / duration if duration > 0 else 0

        print("\n" + "=" * 80)
        print("📊 SESSION SUMMARY")
        print("=" * 80)
        print(f"Duration: {duration:.1f}s")
        print(f"Frames Processed: {self.session_stats['frames_processed']}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Total Detections: {self.session_stats['detections_count']}")
        print(f"Model Switches: {self.session_stats['model_switches']}")
        print(f"Errors: {self.session_stats['errors_count']}")

        if self.pipeline:
            pipeline_stats = self.pipeline.get_stats()
            print(
                f"Pipeline Efficiency: {(1 - pipeline_stats.frames_dropped / max(pipeline_stats.frames_processed, 1)) * 100:.1f}%")

        print("=" * 80)

    def _cleanup(self):
        """Clean up application resources."""
        self.logger.info("Cleaning up application resources...")

        self.running = False

        # Stop recording
        if self.recording_session:
            self._stop_recording_session()

        # Cleanup pipeline
        if self.pipeline:
            self.pipeline.cleanup()

        # Cleanup visualizer
        if self.visualizer:
            self.visualizer.cleanup()

        # Cleanup memory manager
        if self.memory_manager:
            self.memory_manager.cleanup()

        self.logger.info("Application cleanup completed")


def main():
    """Main entry point for the detection application."""
    parser = argparse.ArgumentParser(
        description="CUDA-Accelerated RealSense Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run with default settings
  %(prog)s --config custom_config.yaml       # Use custom configuration
  %(prog)s --duration 60 --no-visualization  # Run for 60s without display
  %(prog)s --benchmark --duration 30         # Benchmark mode for 30s
  %(prog)s --record --output-dir recordings   # Record session data
        """
    )

    # Configuration options
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="Path to configuration file (default: config.yaml)")

    # Runtime options
    parser.add_argument("--duration", "-d", type=float,
                        help="Maximum duration in seconds (default: unlimited)")
    parser.add_argument("--target-fps", type=int, default=30,
                        help="Target FPS for processing (default: 30)")

    # Feature toggles
    parser.add_argument("--no-visualization", action="store_true",
                        help="Disable real-time visualization")
    parser.add_argument("--no-interaction", action="store_true",
                        help="Disable interactive controls")
    parser.add_argument("--record", action="store_true",
                        help="Enable session recording")

    # GPU options
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU-only mode (disable GPU acceleration)")
    parser.add_argument("--gpu-device", type=int, default=0,
                        help="GPU device ID to use (default: 0)")

    # Output options
    parser.add_argument("--output-dir", default="output",
                        help="Output directory for recordings and logs")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help="Logging level")

    # Special modes
    parser.add_argument("--benchmark", action="store_true",
                        help="Run in benchmark mode")
    parser.add_argument("--profile", action="store_true",
                        help="Enable detailed profiling")

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize application
        app = DetectionApplication(args.config)
        app.interactive_mode = not args.no_interaction

        # Override configuration with command line arguments
        if args.force_cpu:
            app.config['gpu'] = {'use_gpu': False}

        if args.target_fps:
            app.config['integration']['performance']['target_fps'] = args.target_fps

        # Initialize components
        if not app.initialize():
            print("❌ Failed to initialize application")
            return 1

        # Special modes
        if args.benchmark:
            print("🚀 Running in benchmark mode...")
            # Would run benchmark-specific logic here

        if args.profile:
            print("📊 Profiling enabled...")
            # Would enable detailed profiling here

        # Run application
        success = app.run(
            duration=args.duration,
            enable_visualization=not args.no_visualization,
            enable_recording=args.record
        )

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Application failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())