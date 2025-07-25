"""
CUDA-accelerated RealSense detection pipeline with GPU optimization.
Main coordinator for camera, detection, and 3D processing with multi-threading.
"""

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable

import cv2
import numpy as np
import torch

from .gpu_memory_manager import CUDAMemoryManager, MemoryPoolType
from ..camera.depth_processor import DepthProcessor
from ..camera.realsense_manager import RealSenseManager
from ..detection import DetectorFactory, DetectorWrapper, Postprocessor
from ..detection.base_detector import DetectionResult
from ..utils.config import ConfigManager
from ..utils.logger import get_logger, PerformanceMonitor, DataRecorder


class PipelineState(Enum):
    """Pipeline execution states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class StreamType(Enum):
    """CUDA stream types for different pipeline stages."""
    CAMERA_PROCESSING = "camera_processing"
    DETECTION_INFERENCE = "detection_inference"
    POSTPROCESSING = "postprocessing"
    VISUALIZATION = "visualization"
    MEMORY_TRANSFER = "memory_transfer"


@dataclass
class PipelineFrame:
    """Complete frame data with metadata."""
    frame_id: int
    timestamp: float

    # Camera data
    color_frame: Optional[np.ndarray] = None
    depth_frame: Optional[np.ndarray] = None
    camera_metadata: Dict[str, Any] = field(default_factory=dict)

    # GPU tensors
    color_tensor: Optional[torch.Tensor] = None
    depth_tensor: Optional[torch.Tensor] = None
    tensor_block_ids: List[int] = field(default_factory=list)

    # Detection results
    detection_result: Optional[DetectionResult] = None
    enhanced_result: Optional[DetectionResult] = None

    # Processing times
    capture_time: float = 0.0
    upload_time: float = 0.0
    detection_time: float = 0.0
    postprocess_time: float = 0.0
    total_time: float = 0.0

    # Status
    valid: bool = True
    error_message: str = ""


@dataclass
class PipelineStats:
    """Pipeline performance statistics."""
    frames_processed: int = 0
    frames_dropped: int = 0
    avg_fps: float = 0.0
    detection_fps: float = 0.0

    # GPU metrics
    gpu_utilization: float = 0.0
    memory_usage_mb: float = 0.0
    memory_efficiency: float = 0.0

    # Timing metrics
    avg_capture_time: float = 0.0
    avg_detection_time: float = 0.0
    avg_postprocess_time: float = 0.0

    # Error metrics
    camera_errors: int = 0
    detection_errors: int = 0
    memory_errors: int = 0

    # Queue sizes
    camera_queue_size: int = 0
    detection_queue_size: int = 0
    output_queue_size: int = 0


class RealSenseDetectionPipeline:
    """CUDA-accelerated pipeline for RealSense camera + object detection + 3D processing."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the complete detection pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger("RealSenseDetectionPipeline")

        # State management
        self.state = PipelineState.STOPPED
        self.state_lock = threading.RLock()

        # Components initialization flags
        self._components_initialized = False
        self._cuda_initialized = False

        # Configuration
        pipeline_config = config.get('integration', {})
        performance_config = pipeline_config.get('performance', {})

        self.target_fps = performance_config.get('target_fps', 30)
        self.use_threading = performance_config.get('use_threading', True)
        self.max_queue_size = performance_config.get('max_queue_size', 10)
        self.enable_3d = pipeline_config.get('enable_3d', True)

        # CUDA streams for pipeline stages
        self.cuda_streams: Dict[StreamType, torch.cuda.Stream] = {}
        self.stream_lock = threading.Lock()

        # Core components
        self.memory_manager: Optional[CUDAMemoryManager] = None
        self.camera_manager: Optional[RealSenseManager] = None
        self.depth_processor: Optional[DepthProcessor] = None
        self.detector: Optional[DetectorWrapper] = None
        self.postprocessor: Optional[Postprocessor] = None

        # Threading components
        self.frame_queues: Dict[str, queue.Queue] = {}
        self.worker_threads: Dict[str, threading.Thread] = {}
        self.thread_events: Dict[str, threading.Event] = {}

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor(window_size=100)
        self.data_recorder: Optional[DataRecorder] = None

        # Frame management
        self.frame_counter = 0
        self.processed_frames = 0
        self.dropped_frames = 0

        # Timing for FPS calculation
        self.last_stats_time = time.time()
        self.frame_times = deque(maxlen=60)  # Keep last 60 frame times

        # Error recovery
        self.error_count = 0
        self.max_errors = 10
        self.recovery_callbacks: List[Callable] = []

        # Hot-reloading support
        self.config_reload_interval = 30.0  # seconds
        self.last_config_check = time.time()

        self.logger.info("RealSenseDetectionPipeline initialized")

    def initialize(self) -> bool:
        """
        Initialize all pipeline components.

        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing pipeline components...")

            # Initialize CUDA and memory management
            if not self._initialize_cuda():
                return False

            # Initialize core components
            if not self._initialize_components():
                return False

            # Initialize threading infrastructure
            if self.use_threading and not self._initialize_threading():
                return False

            # Initialize performance monitoring
            self._initialize_monitoring()

            self._components_initialized = True
            self.logger.info("Pipeline initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {e}")
            return False

    def _initialize_cuda(self) -> bool:
        """Initialize CUDA streams and memory management."""
        try:
            if not torch.cuda.is_available():
                self.logger.warning("CUDA not available, falling back to CPU")
                return True

            # Initialize memory manager
            self.memory_manager = CUDAMemoryManager(self.config)

            # Create CUDA streams for different pipeline stages
            with self.stream_lock:
                for stream_type in StreamType:
                    self.cuda_streams[stream_type] = torch.cuda.Stream()

            self._cuda_initialized = True
            self.logger.info("CUDA initialization completed")
            return True

        except Exception as e:
            self.logger.error(f"CUDA initialization failed: {e}")
            return False

    def _initialize_components(self) -> bool:
        """Initialize camera, detection, and processing components."""
        try:
            # Initialize camera system
            self.camera_manager = RealSenseManager(self.config)
            if not self.camera_manager.initialize_camera():
                self.logger.error("Failed to initialize camera")
                return False

            # Initialize depth processor
            if self.enable_3d:
                self.depth_processor = DepthProcessor(self.camera_manager, self.config)
                self.depth_processor.update_camera_parameters()
                self.logger.info("3D processing enabled")

            # Initialize detection system
            factory = DetectorFactory()
            detector = factory.create_detector_with_fallback(self.config)
            if not detector:
                self.logger.error("Failed to initialize detection system")
                return False
            self.detector = detector

            # Initialize postprocessor
            self.postprocessor = Postprocessor(self.config, self.depth_processor)

            self.logger.info("Core components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            return False

    def _initialize_threading(self) -> bool:
        """Initialize multi-threaded pipeline architecture."""
        try:
            # Create queues for inter-thread communication
            self.frame_queues = {
                'camera_output': queue.Queue(maxsize=self.max_queue_size),
                'detection_input': queue.Queue(maxsize=self.max_queue_size),
                'detection_output': queue.Queue(maxsize=self.max_queue_size),
                'visualization_input': queue.Queue(maxsize=self.max_queue_size)
            }

            # Create thread control events
            self.thread_events = {
                'camera_stop': threading.Event(),
                'detection_stop': threading.Event(),
                'postprocess_stop': threading.Event(),
                'visualization_stop': threading.Event()
            }

            # Create worker threads
            self.worker_threads = {
                'camera': threading.Thread(target=self._camera_worker, daemon=True),
                'detection': threading.Thread(target=self._detection_worker, daemon=True),
                'postprocess': threading.Thread(target=self._postprocess_worker, daemon=True)
            }

            self.logger.info("Threading infrastructure initialized")
            return True

        except Exception as e:
            self.logger.error(f"Threading initialization failed: {e}")
            return False

    def _initialize_monitoring(self):
        """Initialize performance monitoring and data recording."""
        # Initialize data recorder if enabled
        output_config = self.config.get('output', {})
        if output_config.get('logging', {}).get('save_detections', False):
            base_dir = output_config.get('directories', {}).get('base_output', 'output')
            self.data_recorder = DataRecorder(base_dir)

        self.logger.info("Monitoring systems initialized")

    def start(self) -> bool:
        """
        Start the pipeline execution.

        Returns:
            True if started successfully
        """
        with self.state_lock:
            if self.state != PipelineState.STOPPED:
                self.logger.warning(f"Cannot start pipeline in state: {self.state}")
                return False

            self.state = PipelineState.STARTING

        try:
            if not self._components_initialized:
                if not self.initialize():
                    self.state = PipelineState.ERROR
                    return False

            # Start data recording if enabled
            if self.data_recorder:
                self.data_recorder.start_recording()

            # Start worker threads
            if self.use_threading:
                self._start_worker_threads()

            # Reset counters
            self.frame_counter = 0
            self.processed_frames = 0
            self.dropped_frames = 0
            self.error_count = 0
            self.last_stats_time = time.time()

            with self.state_lock:
                self.state = PipelineState.RUNNING

            self.logger.info("Pipeline started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {e}")
            with self.state_lock:
                self.state = PipelineState.ERROR
            return False

    def _start_worker_threads(self):
        """Start all worker threads."""
        for name, thread in self.worker_threads.items():
            if not thread.is_alive():
                thread.start()
                self.logger.debug(f"Started {name} worker thread")

    def stop(self) -> bool:
        """
        Stop the pipeline execution.

        Returns:
            True if stopped successfully
        """
        with self.state_lock:
            if self.state in [PipelineState.STOPPED, PipelineState.STOPPING]:
                return True

            self.state = PipelineState.STOPPING

        try:
            # Signal threads to stop
            for event in self.thread_events.values():
                event.set()

            # Wait for threads to finish
            for name, thread in self.worker_threads.items():
                if thread.is_alive():
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        self.logger.warning(f"{name} thread did not stop gracefully")

            # Stop data recording
            if self.data_recorder:
                self.data_recorder.stop_recording()

            # Clear queues
            for q in self.frame_queues.values():
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

            with self.state_lock:
                self.state = PipelineState.STOPPED

            self.logger.info("Pipeline stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping pipeline: {e}")
            return False

    def pause(self) -> bool:
        """Pause the pipeline execution."""
        with self.state_lock:
            if self.state != PipelineState.RUNNING:
                return False
            self.state = PipelineState.PAUSED

        self.logger.info("Pipeline paused")
        return True

    def resume(self) -> bool:
        """Resume the pipeline execution."""
        with self.state_lock:
            if self.state != PipelineState.PAUSED:
                return False
            self.state = PipelineState.RUNNING

        self.logger.info("Pipeline resumed")
        return True

    def process_frame(self) -> Optional[PipelineFrame]:
        """
        Process a single frame through the complete pipeline.

        Returns:
            Processed PipelineFrame or None if failed
        """
        if self.state != PipelineState.RUNNING:
            return None

        frame_start_time = time.time()
        frame_id = self.frame_counter
        self.frame_counter += 1

        # Create pipeline frame
        pipeline_frame = PipelineFrame(
            frame_id=frame_id,
            timestamp=frame_start_time
        )

        try:
            # Stage 1: Camera capture
            if not self._capture_frame(pipeline_frame):
                self.dropped_frames += 1
                return None

            # Stage 2: GPU upload and preprocessing
            if self._cuda_initialized:
                self._upload_to_gpu(pipeline_frame)

            # Stage 3: Object detection
            self._run_detection(pipeline_frame)

            # Stage 4: Postprocessing and 3D integration
            self._run_postprocessing(pipeline_frame)

            # Stage 5: Update statistics
            pipeline_frame.total_time = time.time() - frame_start_time
            self._update_performance_stats(pipeline_frame)

            # Stage 6: Data recording
            self._record_frame_data(pipeline_frame)

            self.processed_frames += 1
            return pipeline_frame

        except Exception as e:
            self.logger.error(f"Frame processing failed: {e}")
            pipeline_frame.valid = False
            pipeline_frame.error_message = str(e)
            self._handle_error(e)
            return pipeline_frame

    def _capture_frame(self, pipeline_frame: PipelineFrame) -> bool:
        """Capture frame from camera."""
        capture_start = time.time()

        try:
            frames = self.camera_manager.capture_frames()
            if not frames or not frames.get('frame_valid', False):
                return False

            pipeline_frame.color_frame = frames.get('color')
            pipeline_frame.depth_frame = frames.get('depth')
            pipeline_frame.camera_metadata = {
                'color_timestamp': frames.get('color_timestamp', 0),
                'depth_timestamp': frames.get('depth_timestamp', 0),
                'system_timestamp': frames.get('system_timestamp', 0)
            }

            pipeline_frame.capture_time = time.time() - capture_start
            return True

        except Exception as e:
            self.logger.error(f"Frame capture failed: {e}")
            return False

    def _upload_to_gpu(self, pipeline_frame: PipelineFrame):
        """Upload frame data to GPU with optimized memory management."""
        if not self.memory_manager or pipeline_frame.color_frame is None:
            return

        upload_start = time.time()

        try:
            with torch.cuda.stream(self.cuda_streams[StreamType.MEMORY_TRANSFER]):
                # Upload color frame
                color_frame = pipeline_frame.color_frame
                if color_frame.shape[2] == 3:  # RGB to BGR if needed
                    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)

                color_tensor, color_block_id = self.memory_manager.allocate_tensor(
                    color_frame.shape, torch.uint8, MemoryPoolType.FRAME_BUFFERS
                )
                color_tensor.copy_(torch.from_numpy(color_frame), non_blocking=True)

                pipeline_frame.color_tensor = color_tensor
                pipeline_frame.tensor_block_ids.append(color_block_id)

                # Upload depth frame if available
                if pipeline_frame.depth_frame is not None:
                    depth_tensor, depth_block_id = self.memory_manager.allocate_tensor(
                        pipeline_frame.depth_frame.shape, torch.uint16, MemoryPoolType.FRAME_BUFFERS
                    )
                    depth_tensor.copy_(torch.from_numpy(pipeline_frame.depth_frame), non_blocking=True)

                    pipeline_frame.depth_tensor = depth_tensor
                    pipeline_frame.tensor_block_ids.append(depth_block_id)

            pipeline_frame.upload_time = time.time() - upload_start

        except Exception as e:
            self.logger.error(f"GPU upload failed: {e}")

    def _run_detection(self, pipeline_frame: PipelineFrame):
        """Run object detection on the frame."""
        if pipeline_frame.color_frame is None:
            return

        detection_start = time.time()

        try:
            with torch.cuda.stream(self.cuda_streams[StreamType.DETECTION_INFERENCE]):
                # Convert to BGR if needed for detection
                detection_image = pipeline_frame.color_frame
                if detection_image.shape[2] == 3:
                    detection_image = cv2.cvtColor(detection_image, cv2.COLOR_RGB2BGR)

                # Run detection
                result = self.detector.detect(
                    detection_image,
                    frame_id=pipeline_frame.frame_id
                )

                pipeline_frame.detection_result = result
                pipeline_frame.detection_time = time.time() - detection_start

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            pipeline_frame.detection_result = None

    def _run_postprocessing(self, pipeline_frame: PipelineFrame):
        """Run postprocessing and 3D integration."""
        if not pipeline_frame.detection_result:
            return

        postprocess_start = time.time()

        try:
            with torch.cuda.stream(self.cuda_streams[StreamType.POSTPROCESSING]):
                # Apply postprocessing with depth information
                enhanced_result = self.postprocessor.process_detection_result(
                    pipeline_frame.detection_result,
                    pipeline_frame.depth_frame,
                    pipeline_frame.frame_id
                )

                pipeline_frame.enhanced_result = enhanced_result
                pipeline_frame.postprocess_time = time.time() - postprocess_start

        except Exception as e:
            self.logger.error(f"Postprocessing failed: {e}")

    def _update_performance_stats(self, pipeline_frame: PipelineFrame):
        """Update performance monitoring statistics."""
        # Add timing metrics
        self.performance_monitor.add_metric('capture_time', pipeline_frame.capture_time)
        self.performance_monitor.add_metric('detection_time', pipeline_frame.detection_time)
        self.performance_monitor.add_metric('postprocess_time', pipeline_frame.postprocess_time)
        self.performance_monitor.add_metric('total_frame_time', pipeline_frame.total_time)

        # Track frame timing for FPS calculation
        current_time = time.time()
        self.frame_times.append(current_time)

    def _record_frame_data(self, pipeline_frame: PipelineFrame):
        """Record frame data if logging is enabled."""
        if not self.data_recorder or not pipeline_frame.enhanced_result:
            return

        try:
            # Convert detections to recordable format
            detection_data = {
                'frame_id': pipeline_frame.frame_id,
                'timestamp': pipeline_frame.timestamp,
                'objects': []
            }

            for detection in pipeline_frame.enhanced_result.detections:
                obj_data = {
                    'id': detection.detection_id,
                    'class': detection.class_name,
                    'confidence': detection.confidence,
                    'bbox': list(detection.bbox)
                }

                # Add 3D information if available
                if hasattr(detection, 'center_3d') and detection.center_3d != (0, 0, 0):
                    obj_data['position'] = {
                        'x': detection.center_3d[0],
                        'y': detection.center_3d[1],
                        'z': detection.center_3d[2]
                    }
                    obj_data['distance'] = detection.distance

                detection_data['objects'].append(obj_data)

            self.data_recorder.record_detection(
                detection_data, pipeline_frame.frame_id, pipeline_frame.timestamp
            )

            # Record frame metadata
            metadata = {
                'capture_time': pipeline_frame.capture_time,
                'detection_time': pipeline_frame.detection_time,
                'postprocess_time': pipeline_frame.postprocess_time,
                'total_time': pipeline_frame.total_time,
                'detection_count': len(pipeline_frame.enhanced_result.detections)
            }

            self.data_recorder.record_frame_metadata(
                metadata, pipeline_frame.frame_id, pipeline_frame.timestamp
            )

        except Exception as e:
            self.logger.error(f"Data recording failed: {e}")

    def _handle_error(self, error: Exception):
        """Handle pipeline errors with recovery mechanisms."""
        self.error_count += 1

        if self.error_count >= self.max_errors:
            self.logger.critical(f"Too many errors ({self.error_count}), stopping pipeline")
            self.stop()
            return

        # Run recovery callbacks
        for callback in self.recovery_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"Recovery callback failed: {e}")

    def _camera_worker(self):
        """Worker thread for camera frame capture."""
        self.logger.info("Camera worker thread started")

        while not self.thread_events['camera_stop'].is_set():
            try:
                if self.state != PipelineState.RUNNING:
                    time.sleep(0.01)
                    continue

                # Capture frame
                frames = self.camera_manager.capture_frames()
                if frames and frames.get('frame_valid', False):
                    # Create pipeline frame
                    pipeline_frame = PipelineFrame(
                        frame_id=self.frame_counter,
                        timestamp=time.time(),
                        color_frame=frames.get('color'),
                        depth_frame=frames.get('depth'),
                        camera_metadata=frames
                    )

                    self.frame_counter += 1

                    # Put in queue for detection
                    try:
                        self.frame_queues['camera_output'].put(pipeline_frame, timeout=0.01)
                    except queue.Full:
                        self.dropped_frames += 1

            except Exception as e:
                self.logger.error(f"Camera worker error: {e}")
                time.sleep(0.1)

        self.logger.info("Camera worker thread stopped")

    def _detection_worker(self):
        """Worker thread for object detection."""
        self.logger.info("Detection worker thread started")

        while not self.thread_events['detection_stop'].is_set():
            try:
                # Get frame from camera queue
                try:
                    pipeline_frame = self.frame_queues['camera_output'].get(timeout=0.1)
                except queue.Empty:
                    continue

                if self.state != PipelineState.RUNNING:
                    continue

                # Run detection
                self._run_detection(pipeline_frame)

                # Put in postprocessing queue
                try:
                    self.frame_queues['detection_output'].put(pipeline_frame, timeout=0.01)
                except queue.Full:
                    self.dropped_frames += 1

            except Exception as e:
                self.logger.error(f"Detection worker error: {e}")

        self.logger.info("Detection worker thread stopped")

    def _postprocess_worker(self):
        """Worker thread for postprocessing and 3D integration."""
        self.logger.info("Postprocess worker thread started")

        while not self.thread_events['postprocess_stop'].is_set():
            try:
                # Get frame from detection queue
                try:
                    pipeline_frame = self.frame_queues['detection_output'].get(timeout=0.1)
                except queue.Empty:
                    continue

                if self.state != PipelineState.RUNNING:
                    continue

                # Run postprocessing
                self._run_postprocessing(pipeline_frame)

                # Update stats and record
                pipeline_frame.total_time = time.time() - pipeline_frame.timestamp
                self._update_performance_stats(pipeline_frame)
                self._record_frame_data(pipeline_frame)

                self.processed_frames += 1

                # Clean up GPU memory
                if self.memory_manager:
                    for block_id in pipeline_frame.tensor_block_ids:
                        self.memory_manager.release_tensor(block_id, MemoryPoolType.FRAME_BUFFERS)

            except Exception as e:
                self.logger.error(f"Postprocess worker error: {e}")

        self.logger.info("Postprocess worker thread stopped")

    def switch_detection_model(self, model_type: str) -> bool:
        """
        Switch detection model at runtime.

        Args:
            model_type: New model type ('yolo' or 'detr')

        Returns:
            True if switch successful
        """
        try:
            self.logger.info(f"Switching detection model to: {model_type}")

            # Update config
            self.config['detection']['active_model'] = model_type

            # Create new detector
            factory = DetectorFactory()
            new_detector = factory.create_detector_with_fallback(self.config, model_type)

            if new_detector:
                # Replace detector
                old_detector = self.detector
                self.detector = new_detector

                # Cleanup old detector
                if old_detector:
                    old_detector.cleanup()

                self.logger.info(f"Successfully switched to {model_type}")
                return True
            else:
                self.logger.error(f"Failed to create {model_type} detector")
                return False

        except Exception as e:
            self.logger.error(f"Model switch failed: {e}")
            return False

    def get_stats(self) -> PipelineStats:
        """Get comprehensive pipeline statistics."""
        current_time = time.time()

        # Calculate FPS
        if len(self.frame_times) > 1:
            time_span = self.frame_times[-1] - self.frame_times[0]
            fps = (len(self.frame_times) - 1) / time_span if time_span > 0 else 0
        else:
            fps = 0

        # Get detection FPS
        detection_stats = self.performance_monitor.get_statistics('detection_time')
        detection_fps = 1.0 / detection_stats['mean'] if detection_stats and detection_stats['mean'] > 0 else 0

        # Get GPU stats
        gpu_utilization = 0.0
        memory_usage_mb = 0.0
        memory_efficiency = 0.0

        if self.memory_manager:
            memory_stats = self.memory_manager.get_memory_stats()
            memory_usage_mb = memory_stats.total_allocated / 1024 ** 2
            memory_efficiency = memory_stats.efficiency_ratio

            # Estimate GPU utilization (simplified)
            if torch.cuda.is_available():
                gpu_utilization = min(100.0,
                                      memory_stats.total_allocated / memory_stats.total_reserved * 100) if memory_stats.total_reserved > 0 else 0

        # Get timing statistics
        capture_stats = self.performance_monitor.get_statistics('capture_time')
        detection_timing_stats = self.performance_monitor.get_statistics('detection_time')
        postprocess_stats = self.performance_monitor.get_statistics('postprocess_time')

        return PipelineStats(
            frames_processed=self.processed_frames,
            frames_dropped=self.dropped_frames,
            avg_fps=fps,
            detection_fps=detection_fps,

            gpu_utilization=gpu_utilization,
            memory_usage_mb=memory_usage_mb,
            memory_efficiency=memory_efficiency,

            avg_capture_time=capture_stats['mean'] if capture_stats else 0,
            avg_detection_time=detection_timing_stats['mean'] if detection_timing_stats else 0,
            avg_postprocess_time=postprocess_stats['mean'] if postprocess_stats else 0,

            camera_errors=0,  # TODO: Track from camera manager
            detection_errors=0,  # TODO: Track from detector
            memory_errors=0,  # TODO: Track from memory manager

            camera_queue_size=self.frame_queues['camera_output'].qsize() if 'camera_output' in self.frame_queues else 0,
            detection_queue_size=self.frame_queues[
                'detection_output'].qsize() if 'detection_output' in self.frame_queues else 0,
            output_queue_size=0
        )

    def add_recovery_callback(self, callback: Callable[[Exception], None]):
        """Add error recovery callback."""
        self.recovery_callbacks.append(callback)

    def reload_config(self, config_path: str) -> bool:
        """
        Hot-reload configuration at runtime.

        Args:
            config_path: Path to updated configuration file

        Returns:
            True if reload successful
        """
        try:
            self.logger.info("Reloading configuration...")

            new_config = ConfigManager.load_config(config_path)

            # Apply safe configuration updates
            # Only update performance and detection parameters
            safe_updates = ['performance', 'detection', 'visualization']

            for section in safe_updates:
                if section in new_config:
                    self.config[section] = new_config[section]

            # Update target FPS
            performance_config = self.config.get('integration', {}).get('performance', {})
            self.target_fps = performance_config.get('target_fps', 30)

            self.logger.info("Configuration reloaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Configuration reload failed: {e}")
            return False

    def cleanup(self):
        """Clean up all pipeline resources."""
        self.logger.info("Cleaning up pipeline...")

        # Stop pipeline
        self.stop()

        # Cleanup components
        if self.detector:
            self.detector.cleanup()

        if self.camera_manager:
            self.camera_manager.cleanup()

        if self.memory_manager:
            self.memory_manager.cleanup()

        # Clear CUDA streams
        with self.stream_lock:
            for stream in self.cuda_streams.values():
                stream.synchronize()
            self.cuda_streams.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("Pipeline cleanup completed")

    def __enter__(self):
        """Context manager entry."""
        if self.initialize():
            return self
        else:
            raise RuntimeError("Failed to initialize pipeline")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Utility functions for pipeline management
def create_pipeline(config_path: str = "config.yaml") -> RealSenseDetectionPipeline:
    """
    Create and initialize a detection pipeline.

    Args:
        config_path: Path to configuration file

    Returns:
        Initialized RealSenseDetectionPipeline
    """
    config = ConfigManager.load_config(config_path)
    pipeline = RealSenseDetectionPipeline(config)

    if not pipeline.initialize():
        raise RuntimeError("Failed to initialize pipeline")

    return pipeline


def run_pipeline_session(config_path: str = "config.yaml", duration: float = 60.0) -> Dict[str, Any]:
    """
    Run a complete pipeline session with automatic cleanup.

    Args:
        config_path: Path to configuration file
        duration: Session duration in seconds

    Returns:
        Session statistics
    """
    with create_pipeline(config_path) as pipeline:
        if not pipeline.start():
            raise RuntimeError("Failed to start pipeline")

        logger = get_logger("PipelineSession")
        logger.info(f"Starting {duration}s pipeline session")

        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                # Process frame in single-threaded mode
                if not pipeline.use_threading:
                    frame = pipeline.process_frame()
                    if frame and frame.valid:
                        logger.debug(f"Processed frame {frame.frame_id}")

                time.sleep(1.0 / pipeline.target_fps)

        except KeyboardInterrupt:
            logger.info("Session interrupted by user")

        # Get final statistics
        stats = pipeline.get_stats()

        logger.info("Session completed")
        logger.info(f"Processed {stats.frames_processed} frames")
        logger.info(f"Average FPS: {stats.avg_fps:.2f}")
        logger.info(f"Detection FPS: {stats.detection_fps:.2f}")

        return stats.__dict__