"""
Complete RealSense D435i camera management system with SLAM compatibility.
"""

import time
import numpy as np
import pyrealsense2 as rs
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import threading
from dataclasses import dataclass
import logging
import os
import sys

# Simple logger for standalone use
def get_logger(name):
    """Simple logger for standalone use."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# Simple config manager placeholder
class ConfigManager:
    """Simple config manager placeholder."""
    pass


@dataclass
class StreamConfig:
    """Configuration for a single stream."""
    width: int
    height: int
    framerate: int
    format: str
    enabled: bool = True


@dataclass
class IntrinsicParameters:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    ppx: float
    ppy: float
    width: int
    height: int
    distortion_model: str
    coeffs: List[float]


# Singleton Camera Share Manager
class CameraShareManager:
    _instance = None
    _lock = threading.Lock()
    _init_lock = threading.Lock()  # NEW: Dedicated lock for initialization

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance


    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.pipeline = None
            self.config_rs = None
            self.profile = None
            self.align = None
            self.intrinsics = None
            self.depth_intrinsics = None
            self.depth_scale = None
            self.context = rs.context()
            self.device = None
            self._init_lock = threading.Lock()

            # Enhanced features
            self.filters = {}
            self.filter_enabled = {}

            # Frame sharing with subscriber-specific buffers
            self.latest_frames = {}
            self.frame_lock = threading.Lock()
            self.subscribers = {}
            self.subscriber_frames = {}  # NEW: Individual frame buffers per subscriber
            self.frame_thread = None
            self.running = False

            # IMU data
            self.latest_accel = None
            self.latest_gyro = None

            # Error handling
            self.retry_count = 0
            self.max_retries = 3
            self.last_frame_time = time.time()
            self.is_streaming = False

            # NEW: Frame distribution tracking
            self.frame_counter = 0
            self.subscriber_last_frame = {}  # Track last frame each subscriber received

            self.logger = get_logger("CameraShareManager")
            self._initialized = True
            self.logger.info("Enhanced CameraShareManager initialized")

    def initialize_camera(self, config):
        """Initialize camera with config from first subscriber."""
        # FIX: More robust check for already initialized state
        if self.pipeline is not None and self.is_streaming and self.running:
            self.logger.info("Camera already initialized and streaming, reusing existing instance")
            return True

        # FIX: Only cleanup if we're in a partial/failed state
        if self.pipeline is not None and not self.is_streaming:
            self.logger.info("Cleaning up partial initialization...")
            self.cleanup()
            time.sleep(0.5)

        try:
            self.logger.info("Initializing shared camera...")

            # Detect devices
            devices = self._detect_devices()
            if not devices:
                self.logger.error("No RealSense devices detected")
                return False

            self.pipeline = rs.pipeline()
            self.config_rs = rs.config()

            # Use device serial if multiple devices
            if len(devices) > 0:
                device_serial = devices[0]['serial_number']
                self.config_rs.enable_device(device_serial)
                self.logger.info(f"Using device: {device_serial}")

            # Configure streams from SLAM config format
            if 'camera' in config:
                cam_config = config["camera"]
                self.config_rs.enable_stream(
                    rs.stream.depth,
                    cam_config["width"],
                    cam_config["height"],
                    rs.format.z16,
                    cam_config["fps"]
                )
                self.config_rs.enable_stream(
                    rs.stream.color,
                    cam_config["width"],
                    cam_config["height"],
                    rs.format.bgr8,
                    cam_config["fps"]
                )
                self.config_rs.enable_stream(rs.stream.accel)
                self.config_rs.enable_stream(rs.stream.gyro)
            else:
                # Default config
                self.config_rs.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                self.config_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                self.config_rs.enable_stream(rs.stream.accel)
                self.config_rs.enable_stream(rs.stream.gyro)

            # Enhanced features - setup filters
            self._setup_filters()

            # Start streaming
            self.profile = self.pipeline.start(self.config_rs)
            self.device = self.profile.get_device()

            # Enable alignment
            self.align = rs.align(rs.stream.color)

            # Get intrinsics AFTER pipeline is started
            try:
                color_stream = self.profile.get_stream(rs.stream.color)
                self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
                self.logger.info(f"Color intrinsics: fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}")
            except Exception as e:
                self.logger.error(f"Error getting intrinsics: {e}")
                self.intrinsics = None

            # Enhanced: Get depth intrinsics and scale
            try:
                depth_stream = self.profile.get_stream(rs.stream.depth)
                self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
                depth_sensor = self.profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
            except Exception as e:
                self.logger.error(f"Error getting depth parameters: {e}")
                self.depth_intrinsics = None
                self.depth_scale = None

            # Start frame capture thread
            self.running = True
            self.is_streaming = True
            self.frame_thread = threading.Thread(target=self._frame_capture_loop, daemon=True)
            self.frame_thread.start()

            self.logger.info("Shared camera initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            self.cleanup()
            return False


    def _detect_devices(self):
        """Detect available devices."""
        devices = []
        try:
            device_list = self.context.query_devices()
            for i, device in enumerate(device_list):
                device_info = {
                    'index': i,
                    'name': device.get_info(rs.camera_info.name),
                    'serial_number': device.get_info(rs.camera_info.serial_number),
                    'firmware_version': device.get_info(rs.camera_info.firmware_version),
                }
                devices.append(device_info)
                self.logger.info(f"Detected device {i}: {device_info['name']} (S/N: {device_info['serial_number']})")
        except Exception as e:
            self.logger.error(f"Error detecting devices: {e}")
        return devices

    def _setup_filters(self):
        """Setup enhanced depth filters."""
        try:
            # Spatial filter
            self.filters['spatial'] = rs.spatial_filter()
            self.filters['spatial'].set_option(rs.option.filter_smooth_alpha, 0.5)
            self.filters['spatial'].set_option(rs.option.filter_smooth_delta, 20)
            self.filters['spatial'].set_option(rs.option.filter_magnitude, 2)

            # Temporal filter
            self.filters['temporal'] = rs.temporal_filter()
            self.filters['temporal'].set_option(rs.option.filter_smooth_alpha, 0.4)
            self.filters['temporal'].set_option(rs.option.filter_smooth_delta, 20)

            # Hole filling filter
            self.filters['hole_filling'] = rs.hole_filling_filter()
            self.filters['hole_filling'].set_option(rs.option.holes_fill, 1)

            self.filter_enabled = {'spatial': True, 'temporal': True, 'hole_filling': True}
            self.logger.info("Depth filters configured")
        except Exception as e:
            self.logger.warning(f"Could not setup filters: {e}")
            self.filter_enabled = {}

    def _frame_capture_loop(self):
        """Enhanced frame capture loop with better distribution."""
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                # Enhanced: Apply filters to depth frame
                if depth_frame:
                    filtered_depth = depth_frame
                    for filter_name in ['spatial', 'temporal', 'hole_filling']:
                        if self.filter_enabled.get(filter_name, False):
                            try:
                                filtered_depth = self.filters[filter_name].process(filtered_depth)
                            except:
                                pass
                    depth_frame = filtered_depth

                # Get IMU data
                accel_frame = frames.first_or_default(rs.stream.accel)
                gyro_frame = frames.first_or_default(rs.stream.gyro)

                if accel_frame:
                    accel_data = accel_frame.as_motion_frame().get_motion_data()
                    self.latest_accel = np.array([accel_data.x, accel_data.y, accel_data.z])

                if gyro_frame:
                    gyro_data = gyro_frame.as_motion_frame().get_motion_data()
                    self.latest_gyro = np.array([gyro_data.x, gyro_data.y, gyro_data.z])

                if depth_frame and color_frame:
                    # Convert to numpy arrays
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())

                    with self.frame_lock:
                        self.frame_counter += 1
                        frame_data = {
                            'color': color_image,
                            'depth': depth_image,
                            'timestamp': time.time(),
                            'color_timestamp': color_frame.get_timestamp(),
                            'depth_timestamp': depth_frame.get_timestamp(),
                            'frame_valid': True,
                            'frame_id': self.frame_counter  # NEW: Unique frame ID
                        }

                        self.latest_frames = frame_data

                        # NEW: Update subscriber-specific buffers
                        for subscriber_id in self.subscribers:
                            self.subscriber_frames[subscriber_id] = frame_data.copy()

                        self.last_frame_time = time.time()
                        self.retry_count = 0

            except RuntimeError as e:
                self.logger.warning(f"Frame capture error: {e}")
                self.retry_count += 1

                if self.retry_count >= self.max_retries:
                    self.logger.error("Max retries reached, attempting restart...")
                    if self._attempt_restart():
                        self.retry_count = 0
                    else:
                        self.logger.error("Camera restart failed, stopping capture")
                        break

                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Unexpected frame capture error: {e}")
                time.sleep(0.1)

    def get_frames_for_subscriber(self, subscriber_id):
        """Get frames for specific subscriber with improved initial frame handling."""
        with self.frame_lock:
            if subscriber_id in self.subscriber_frames:
                frames = self.subscriber_frames[subscriber_id]
                if frames and 'color' in frames:
                    # For initial frames or if frame tracking fails, just return the frames
                    last_frame_id = self.subscriber_last_frame.get(subscriber_id, -1)
                    current_frame_id = frames.get('frame_id', 0)

                    # More permissive: return frames if new OR if we haven't got any frames yet
                    if current_frame_id > last_frame_id or last_frame_id == -1:
                        self.subscriber_last_frame[subscriber_id] = current_frame_id
                        return frames['color'].copy(), frames['depth'].copy()

                    # Fallback: if frame ID logic blocks, still return frames after some time
                    if time.time() - frames.get('timestamp', 0) < 1.0:  # Fresh frame
                        return frames['color'].copy(), frames['depth'].copy()

            # Final fallback: use latest_frames if subscriber buffer is empty
            if self.latest_frames and 'color' in self.latest_frames:
                return self.latest_frames['color'].copy(), self.latest_frames['depth'].copy()

            return None, None

    def _attempt_restart(self):
        """Attempt to restart the camera pipeline."""
        try:
            self.logger.info("Restarting shared camera...")
            if self.pipeline:
                self.pipeline.stop()
            time.sleep(1)

            # Restart with same config
            self.profile = self.pipeline.start(self.config_rs)
            self.logger.info("Shared camera restarted successfully")
            return True
        except Exception as e:
            self.logger.error(f"Camera restart failed: {e}")
            return False

    def capture_frames(self):
        """Get latest frames (thread-safe)."""
        with self.frame_lock:
            if self.latest_frames and 'color' in self.latest_frames:
                return self.latest_frames.copy()
            return None

    def get_frames(self):
        """Get latest frames as tuple (thread-safe)."""
        with self.frame_lock:
            if self.latest_frames and 'color' in self.latest_frames:
                return self.latest_frames['color'].copy(), self.latest_frames['depth'].copy()
            return None, None

    def get_imu_data(self):
        """Get latest IMU data."""
        return self.latest_accel, self.latest_gyro

    def register_subscriber(self, name):
        """Enhanced subscriber registration."""
        subscriber_id = f"{name}_{id(object())}"
        self.subscribers[subscriber_id] = {
            'name': name,
            'created_at': time.time()
        }
        self.subscriber_frames[subscriber_id] = {}  # Initialize frame buffer
        self.subscriber_last_frame[subscriber_id] = -1  # Initialize frame tracker
        self.logger.info(f"Registered subscriber: {subscriber_id} ({name})")
        return subscriber_id

    def unregister_subscriber(self, subscriber_id):
        """Enhanced subscriber cleanup."""
        if subscriber_id in self.subscribers:
            del self.subscribers[subscriber_id]
            if subscriber_id in self.subscriber_frames:
                del self.subscriber_frames[subscriber_id]
            if subscriber_id in self.subscriber_last_frame:
                del self.subscriber_last_frame[subscriber_id]
            self.logger.info(f"Unregistered subscriber: {subscriber_id}")

        # If no more subscribers, cleanup
        if not self.subscribers:
            self.logger.info("No more subscribers, cleaning up shared camera")
            self.cleanup()

    def get_device_status(self):
        """Get device status."""
        return {
            'is_streaming': self.is_streaming,
            'subscribers': len(self.subscribers),
            'last_frame_age': time.time() - self.last_frame_time,
            'retry_count': self.retry_count,
            'filters_enabled': list(self.filter_enabled.keys())
        }

    def cleanup(self):
        """Cleanup camera resources."""
        self.running = False
        self.is_streaming = False

        if self.frame_thread and self.frame_thread.is_alive():
            self.frame_thread.join(timeout=2)

        if self.pipeline:
            try:
                self.pipeline.stop()
                self.logger.info("Shared camera pipeline stopped")
            except:
                pass

        self.pipeline = None
        self.subscribers.clear()
        self.logger.info("Shared camera cleanup completed")


class RealSenseManager:
    """Complete RealSense D435i camera management system."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RealSense camera manager.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.camera_config = config['camera']
        self.logger = get_logger("RealSenseManager")

        # RealSense components
        self.pipeline = None
        self.pipeline_config = None
        self.device = None
        self.context = rs.context()

        # Stream configurations
        self.stream_configs = self._parse_stream_configs()

        # Filters
        self.filters = {}
        self.filter_enabled = {}

        # Frame data
        self.frames = None
        self.aligned_frames = None
        self.align = None

        # Intrinsic parameters
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.depth_scale = None

        # State management
        self.is_streaming = False
        self.device_id = self.camera_config.get('device_id', 0)
        self.retry_count = 0
        self.max_retries = 3

        # Threading
        self.frame_lock = threading.Lock()
        self.last_frame_time = 0

        self.logger.info("RealSenseManager initialized")

    def _parse_stream_configs(self) -> Dict[str, StreamConfig]:
        """Parse stream configurations from config."""
        streams = {}
        stream_configs = self.camera_config.get('streams', {})

        for stream_name, stream_data in stream_configs.items():
            if stream_data.get('enabled', True):
                streams[stream_name] = StreamConfig(
                    width=stream_data.get('width', 640),
                    height=stream_data.get('height', 480),
                    framerate=stream_data.get('framerate', 30),
                    format=stream_data.get('format', 'RGB8'),
                    enabled=True
                )

        return streams

    def detect_devices(self) -> List[Dict[str, Any]]:
        """
        Detect available RealSense devices.

        Returns:
            List of device information dictionaries
        """
        devices = []

        try:
            device_list = self.context.query_devices()

            for i, device in enumerate(device_list):
                device_info = {
                    'index': i,
                    'name': device.get_info(rs.camera_info.name),
                    'serial_number': device.get_info(rs.camera_info.serial_number),
                    'firmware_version': device.get_info(rs.camera_info.firmware_version),
                    'product_id': device.get_info(rs.camera_info.product_id),
                    'usb_type': device.get_info(rs.camera_info.usb_type_descriptor)
                }
                devices.append(device_info)

                self.logger.info(f"Detected device {i}: {device_info['name']} "
                                 f"(S/N: {device_info['serial_number']})")

        except Exception as e:
            self.logger.error(f"Error detecting devices: {e}")

        return devices

    def initialize_camera(self) -> bool:
        """
        Initialize camera with configured settings.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Check for available devices
            devices = self.detect_devices()
            if not devices:
                self.logger.error("No RealSense devices detected")
                return False

            # Create pipeline and config
            self.pipeline = rs.pipeline()
            self.pipeline_config = rs.config()

            # Configure device if specific device ID is requested
            if self.device_id < len(devices):
                device_serial = devices[self.device_id]['serial_number']
                self.pipeline_config.enable_device(device_serial)
                self.logger.info(f"Using device {self.device_id}: {device_serial}")

            # Configure streams
            self._configure_streams()

            # Start pipeline
            profile = self.pipeline.start(self.pipeline_config)

            # Get device and setup filters
            self.device = profile.get_device()
            self._setup_filters()

            # Extract intrinsic parameters
            self._extract_intrinsics(profile)

            # Setup alignment
            if self.camera_config.get('alignment', {}).get('align_depth_to_color', True):
                self.align = rs.align(rs.stream.color)

            self.is_streaming = True
            self.logger.info("Camera initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            self.cleanup()
            return False

    def _configure_streams(self):
        """Configure enabled streams in the pipeline."""
        for stream_name, stream_config in self.stream_configs.items():
            try:
                if stream_name == 'color':
                    self.pipeline_config.enable_stream(
                        rs.stream.color,
                        stream_config.width,
                        stream_config.height,
                        self._get_format(stream_config.format),
                        stream_config.framerate
                    )
                    self.logger.debug(
                        f"Configured color stream: {stream_config.width}x{stream_config.height}@{stream_config.framerate}")

                elif stream_name == 'depth':
                    self.pipeline_config.enable_stream(
                        rs.stream.depth,
                        stream_config.width,
                        stream_config.height,
                        self._get_format(stream_config.format),
                        stream_config.framerate
                    )
                    self.logger.debug(
                        f"Configured depth stream: {stream_config.width}x{stream_config.height}@{stream_config.framerate}")

                elif stream_name == 'infrared':
                    self.pipeline_config.enable_stream(
                        rs.stream.infrared,
                        1,  # IR channel 1
                        stream_config.width,
                        stream_config.height,
                        self._get_format(stream_config.format),
                        stream_config.framerate
                    )
                    self.logger.debug(
                        f"Configured infrared stream: {stream_config.width}x{stream_config.height}@{stream_config.framerate}")

            except Exception as e:
                self.logger.warning(f"Failed to configure {stream_name} stream: {e}")

    def _get_format(self, format_str: str) -> rs.format:
        """Convert format string to RealSense format enum."""
        format_map = {
            'RGB8': rs.format.rgb8,
            'BGR8': rs.format.bgr8,
            'Z16': rs.format.z16,
            'Y8': rs.format.y8,
            'Y16': rs.format.y16
        }
        return format_map.get(format_str, rs.format.rgb8)

    def _setup_filters(self):
        """Setup and configure depth filters."""
        filter_config = self.camera_config.get('filters', {})

        # Spatial filter
        spatial_config = filter_config.get('spatial', {})
        if spatial_config.get('enabled', False):
            self.filters['spatial'] = rs.spatial_filter()
            self.filters['spatial'].set_option(rs.option.filter_smooth_alpha,
                                               spatial_config.get('smooth_alpha', 0.5))
            self.filters['spatial'].set_option(rs.option.filter_smooth_delta,
                                               spatial_config.get('smooth_delta', 20))
            self.filters['spatial'].set_option(rs.option.filter_magnitude,
                                               spatial_config.get('magnitude', 2))
            self.filters['spatial'].set_option(rs.option.holes_fill,
                                               spatial_config.get('hole_fill', 0))
            self.filter_enabled['spatial'] = True
            self.logger.debug("Spatial filter configured")

        # Temporal filter
        temporal_config = filter_config.get('temporal', {})
        if temporal_config.get('enabled', False):
            self.filters['temporal'] = rs.temporal_filter()
            self.filters['temporal'].set_option(rs.option.filter_smooth_alpha,
                                                temporal_config.get('smooth_alpha', 0.4))
            self.filters['temporal'].set_option(rs.option.filter_smooth_delta,
                                                temporal_config.get('smooth_delta', 20))
            self.filters['temporal'].set_option(rs.option.holes_fill,
                                                temporal_config.get('persistence_control', 3))
            self.filter_enabled['temporal'] = True
            self.logger.debug("Temporal filter configured")

        # Hole filling filter
        hole_fill_config = filter_config.get('hole_filling', {})
        if hole_fill_config.get('enabled', False):
            self.filters['hole_filling'] = rs.hole_filling_filter()
            self.filters['hole_filling'].set_option(rs.option.holes_fill,
                                                    hole_fill_config.get('hole_fill', 1))
            self.filter_enabled['hole_filling'] = True
            self.logger.debug("Hole filling filter configured")

    def _extract_intrinsics(self, profile):
        """Extract camera intrinsic parameters."""
        try:
            # Color intrinsics
            if 'color' in self.stream_configs:
                color_stream = profile.get_stream(rs.stream.color)
                color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

                self.color_intrinsics = IntrinsicParameters(
                    fx=color_intrinsics.fx,
                    fy=color_intrinsics.fy,
                    ppx=color_intrinsics.ppx,
                    ppy=color_intrinsics.ppy,
                    width=color_intrinsics.width,
                    height=color_intrinsics.height,
                    distortion_model=str(color_intrinsics.model),
                    coeffs=list(color_intrinsics.coeffs)
                )
                self.logger.debug(f"Color intrinsics: fx={color_intrinsics.fx:.2f}, fy={color_intrinsics.fy:.2f}")

            # Depth intrinsics and scale
            if 'depth' in self.stream_configs:
                depth_stream = profile.get_stream(rs.stream.depth)
                depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

                self.depth_intrinsics = IntrinsicParameters(
                    fx=depth_intrinsics.fx,
                    fy=depth_intrinsics.fy,
                    ppx=depth_intrinsics.ppx,
                    ppy=depth_intrinsics.ppy,
                    width=depth_intrinsics.width,
                    height=depth_intrinsics.height,
                    distortion_model=str(depth_intrinsics.model),
                    coeffs=list(depth_intrinsics.coeffs)
                )

                # Get depth scale
                depth_sensor = profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                self.logger.debug(f"Depth scale: {self.depth_scale}")

        except Exception as e:
            self.logger.error(f"Error extracting intrinsics: {e}")

    def capture_frames(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Capture synchronized frames from all enabled streams.

        Returns:
            Dictionary containing frame data or None if capture failed
        """
        if not self.is_streaming:
            return None

        try:
            with self.frame_lock:
                # Wait for frames with timeout
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)

                if not frames:
                    return None

                # Apply alignment if configured
                if self.align:
                    frames = self.align.process(frames)

                frame_data = {}
                current_time = time.time()

                # Extract color frame
                if 'color' in self.stream_configs:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        frame_data['color'] = np.asanyarray(color_frame.get_data())
                        frame_data['color_timestamp'] = color_frame.get_timestamp()
                        frame_data['color_frame_number'] = color_frame.get_frame_number()

                # Extract depth frame
                if 'depth' in self.stream_configs:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        # Apply filters
                        filtered_depth = depth_frame
                        for filter_name in ['spatial', 'temporal', 'hole_filling']:
                            if self.filter_enabled.get(filter_name, False):
                                filtered_depth = self.filters[filter_name].process(filtered_depth)

                        frame_data['depth'] = np.asanyarray(filtered_depth.get_data())
                        frame_data['depth_timestamp'] = filtered_depth.get_timestamp()
                        frame_data['depth_frame_number'] = filtered_depth.get_frame_number()
                        frame_data['depth_frame_ref'] = filtered_depth  # Keep reference for 3D calculations

                # Extract infrared frame
                if 'infrared' in self.stream_configs:
                    ir_frame = frames.get_infrared_frame(1)
                    if ir_frame:
                        frame_data['infrared'] = np.asanyarray(ir_frame.get_data())
                        frame_data['infrared_timestamp'] = ir_frame.get_timestamp()
                        frame_data['infrared_frame_number'] = ir_frame.get_frame_number()

                # Add metadata
                frame_data['system_timestamp'] = current_time
                frame_data['frame_valid'] = len(frame_data) > 1  # At least one actual frame

                self.last_frame_time = current_time
                self.retry_count = 0  # Reset retry count on successful capture

                return frame_data

        except RuntimeError as e:
            self.logger.warning(f"Frame capture timeout or device error: {e}")
            self.retry_count += 1

            if self.retry_count >= self.max_retries:
                self.logger.error("Max retries reached, attempting to restart camera")
                if self._attempt_restart():
                    self.retry_count = 0
                else:
                    self.is_streaming = False

            return None

        except Exception as e:
            self.logger.error(f"Unexpected error during frame capture: {e}")
            return None

    def _attempt_restart(self) -> bool:
        """Attempt to restart the camera pipeline."""
        try:
            self.logger.info("Attempting camera restart...")

            # Stop current pipeline
            if self.pipeline:
                self.pipeline.stop()

            time.sleep(1)  # Brief pause

            # Reinitialize
            return self.initialize_camera()

        except Exception as e:
            self.logger.error(f"Camera restart failed: {e}")
            return False

    def get_device_status(self) -> Dict[str, Any]:
        """
        Get current device status and health information.

        Returns:
            Dictionary containing device status information
        """
        status = {
            'is_streaming': self.is_streaming,
            'device_connected': False,
            'last_frame_age': 0,
            'retry_count': self.retry_count,
            'streams_configured': list(self.stream_configs.keys()),
            'filters_enabled': [name for name, enabled in self.filter_enabled.items() if enabled]
        }

        try:
            if self.device:
                status['device_connected'] = True
                status['device_name'] = self.device.get_info(rs.camera_info.name)
                status['serial_number'] = self.device.get_info(rs.camera_info.serial_number)
                status['firmware_version'] = self.device.get_info(rs.camera_info.firmware_version)

            if self.last_frame_time > 0:
                status['last_frame_age'] = time.time() - self.last_frame_time

        except Exception as e:
            self.logger.warning(f"Error getting device status: {e}")

        return status

    def set_filter_parameter(self, filter_name: str, parameter: str, value: float) -> bool:
        """
        Dynamically adjust filter parameters.

        Args:
            filter_name: Name of the filter ('spatial', 'temporal', 'hole_filling')
            parameter: Parameter name
            value: New parameter value

        Returns:
            True if parameter was set successfully
        """
        try:
            if filter_name in self.filters:
                # Map parameter names to RealSense options
                param_map = {
                    'smooth_alpha': rs.option.filter_smooth_alpha,
                    'smooth_delta': rs.option.filter_smooth_delta,
                    'magnitude': rs.option.filter_magnitude,
                    'hole_fill': rs.option.holes_fill
                }

                if parameter in param_map:
                    self.filters[filter_name].set_option(param_map[parameter], value)
                    self.logger.debug(f"Set {filter_name}.{parameter} = {value}")
                    return True
                else:
                    self.logger.warning(f"Unknown parameter: {parameter}")

            else:
                self.logger.warning(f"Filter not found: {filter_name}")

        except Exception as e:
            self.logger.error(f"Error setting filter parameter: {e}")

        return False

    def save_configuration(self, output_path: str):
        """Save current camera configuration and intrinsics."""
        try:
            config_data = {
                'camera_config': self.camera_config,
                'color_intrinsics': self.color_intrinsics.__dict__ if self.color_intrinsics else None,
                'depth_intrinsics': self.depth_intrinsics.__dict__ if self.depth_intrinsics else None,
                'depth_scale': self.depth_scale,
                'device_status': self.get_device_status()
            }

            import json
            with open(output_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)

            self.logger.info(f"Camera configuration saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")

    def cleanup(self):
        """Clean up resources and stop camera."""
        try:
            self.is_streaming = False

            if hasattr(self, 'pipeline') and self.pipeline:
                try:
                    self.pipeline.stop()
                    self.logger.info("Pipeline stopped")
                except Exception as e:
                    self.logger.debug(f"Pipeline stop error (normal if not started): {e}")

            # Clear references
            self.pipeline = None
            self.pipeline_config = None
            self.device = None
            self.frames = None
            self.aligned_frames = None

            self.logger.info("Camera cleanup completed")

        except Exception as e:
            self.logger.debug(f"Cleanup error: {e}")

    def __enter__(self):
        """Context manager entry."""
        if self.initialize_camera():
            return self
        else:
            raise RuntimeError("Failed to initialize camera")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()



# SLAM Compatibility Class - Using Shared Camera Manager
class D435iCamera:
    """SLAM-compatible interface using shared CameraShareManager internally."""

    def __init__(self, config):
        self.config = config
        self.camera_manager = CameraShareManager()

        # FIX: Check if already initialized BEFORE attempting to initialize
        if not self.camera_manager.is_streaming:
            if not self.camera_manager.initialize_camera(config):
                raise RuntimeError("Failed to initialize camera")
        else:
            self.camera_manager.logger.info("Camera already initialized, skipping initialization")

        self.subscriber_id = self.camera_manager.register_subscriber("SLAM")

        # FIX: Always set intrinsics attribute
        self.intrinsics = self.camera_manager.intrinsics
        print(f"D435iCamera initialized for SLAM (subscriber: {self.subscriber_id})")

    def get_frames(self):
        """Get RGB and depth frames (SLAM interface)."""
        # NEW: Use subscriber-specific frame getter
        return self.camera_manager.get_frames_for_subscriber(self.subscriber_id)

    def get_imu_data(self):
        """Get latest IMU readings (SLAM interface)."""
        return self.camera_manager.get_imu_data()

    def get_intrinsics(self):
        """Return intrinsic matrix (SLAM interface)."""
        if self.intrinsics is None:
            # Fallback intrinsics
            width = self.config["camera"]["width"]
            height = self.config["camera"]["height"]
            fx, fy = 500.0, 500.0  # Default focal lengths
            cx, cy = width / 2, height / 2

            print(f"Using fallback intrinsics: {fx}, {fy}, {cx}, {cy}")
        else:
            fx, fy = self.intrinsics.fx, self.intrinsics.fy
            cx, cy = self.intrinsics.ppx, self.intrinsics.ppy

        intrinsic_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        return intrinsic_matrix

    def get_device_status(self):
        """Get device status (enhanced functionality)."""
        return self.camera_manager.get_device_status()

    def stop(self):
        """Stop camera (SLAM interface)."""
        if hasattr(self, 'subscriber_id'):
            self.camera_manager.unregister_subscriber(self.subscriber_id)
            print(f"D435iCamera stopped (subscriber {self.subscriber_id})")


# Detection system compatibility
class RealSenseDetectionCamera:
    """Detection system compatible interface using shared camera manager."""

    def __init__(self, config):
        """Initialize camera for detection system."""
        self.config = config

        # Use the shared camera manager
        self.camera_manager = CameraShareManager()

        # FIX: Check if already initialized BEFORE attempting to initialize
        if not self.camera_manager.is_streaming:
            if not self.camera_manager.initialize_camera(config):
                raise RuntimeError("Failed to initialize camera")
        else:
            self.camera_manager.logger.info("Camera already initialized, skipping initialization")

        # Register as subscriber with unique name
        self.subscriber_id = self.camera_manager.register_subscriber("Detection")

    def capture_frames(self):
        """Get frames for detection system."""
        # NEW: Use subscriber-specific frame getter
        rgb, depth = self.camera_manager.get_frames_for_subscriber(self.subscriber_id)

        if rgb is not None and depth is not None:
            return {
                'color': rgb,
                'depth': depth,
                'timestamp': time.time(),
                'frame_valid': True
            }
        return None

    def get_frames(self):
        """Alternative method for getting frames."""
        return self.camera_manager.get_frames_for_subscriber(self.subscriber_id)

    def get_device_status(self):
        """Get device status."""
        return self.camera_manager.get_device_status()

    def cleanup(self):
        """Cleanup detection camera."""
        if hasattr(self, 'subscriber_id'):
            self.camera_manager.unregister_subscriber(self.subscriber_id)
            print(f"Detection camera stopped (subscriber {self.subscriber_id})")