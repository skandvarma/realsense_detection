"""
Logging system and data recording utilities for RealSense object detection project.
"""

import os
import json
import time
import logging
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from logging.handlers import RotatingFileHandler
import threading
from collections import defaultdict, deque


class Logger:
    """Enhanced logging system with multiple output handlers and custom formatting."""

    def __init__(self, name: str = "RealSenseDetection", level: str = "ERROR",
                 log_dir: str = "logs", max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5):
        """
        Initialize the logger with configurable settings.

        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            max_bytes: Maximum size for rotating log files
            backup_count: Number of backup files to keep
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create log directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up formatters
        self._setup_formatters()

        # Set up handlers
        self._setup_console_handler()
        self._setup_file_handler(max_bytes, backup_count)
        self._setup_error_handler()

        self.logger.info(f"Logger initialized: {name}")

    def _setup_formatters(self):
        """Set up custom formatters for different output types."""
        # Console formatter (more concise)
        self.console_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )

        # File formatter (more detailed)
        self.file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Error formatter (most detailed)
        self.error_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)-8s | %(pathname)s:%(lineno)d | %(funcName)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _setup_console_handler(self):
        """Set up console output handler."""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)

    def _setup_file_handler(self, max_bytes: int, backup_count: int):
        """Set up rotating file handler for general logs."""
        log_file = self.log_dir / "detection.log"
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(file_handler)

    def _setup_error_handler(self):
        """Set up separate handler for error logs."""
        error_log_file = self.log_dir / "errors.log"
        error_handler = RotatingFileHandler(
            error_log_file, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.error_formatter)
        self.logger.addHandler(error_handler)

    def debug(self, message: str, **kwargs):
        """Log debug message with optional context."""
        self._log_with_context(logging.DEBUG, message, kwargs)

    def info(self, message: str, **kwargs):
        """Log info message with optional context."""
        self._log_with_context(logging.INFO, message, kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with optional context."""
        self._log_with_context(logging.WARNING, message, kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with optional context."""
        self._log_with_context(logging.ERROR, message, kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message with optional context."""
        self._log_with_context(logging.CRITICAL, message, kwargs)

    def _log_with_context(self, level: int, message: str, context: Dict[str, Any]):
        """Log message with additional context information."""
        if context:
            context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
            full_message = f"{message} | {context_str}"
        else:
            full_message = message

        self.logger.log(level, full_message)


class DataRecorder:
    """Session-based recording system for detection data and frame metadata."""

    def __init__(self, base_dir: str = "output", session_name: Optional[str] = None):
        """
        Initialize the data recorder.

        Args:
            base_dir: Base directory for recording sessions
            session_name: Custom session name (auto-generated if None)
        """
        self.base_dir = Path(base_dir)
        self.session_name = session_name or self._generate_session_name()
        self.session_dir = self.base_dir / "sessions" / self.session_name

        # Create session directories
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.detections_dir = self.session_dir / "detections"
        self.frames_dir = self.session_dir / "frames"
        self.metadata_dir = self.session_dir / "metadata"

        for directory in [self.detections_dir, self.frames_dir, self.metadata_dir]:
            directory.mkdir(exist_ok=True)

        # Recording state
        self.is_recording = False
        self.start_time = None
        self.frame_count = 0
        self.detection_count = 0

        # Data buffers
        self.detection_buffer = []
        self.metadata_buffer = []
        self.buffer_lock = threading.Lock()

        # Session metadata
        self.session_metadata = {
            'session_name': self.session_name,
            'created_at': datetime.now().isoformat(),
            'frame_count': 0,
            'detection_count': 0,
            'duration': 0.0
        }

        self._logger = Logger("DataRecorder")
        self._logger.info(f"DataRecorder initialized for session: {self.session_name}")

    def _generate_session_name(self) -> str:
        """Generate a unique session name based on timestamp."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def start_recording(self):
        """Start the recording session."""
        with self.buffer_lock:
            self.is_recording = True
            self.start_time = time.time()
            self.frame_count = 0
            self.detection_count = 0

        self._logger.info("Recording session started")

    def stop_recording(self):
        """Stop the recording session and save all buffered data."""
        with self.buffer_lock:
            self.is_recording = False
            end_time = time.time()

            if self.start_time:
                duration = end_time - self.start_time
                self.session_metadata['duration'] = duration
                self.session_metadata['frame_count'] = self.frame_count
                self.session_metadata['detection_count'] = self.detection_count
                self.session_metadata['ended_at'] = datetime.now().isoformat()

        # Save all buffered data
        self._flush_buffers()
        self._save_session_metadata()

        self._logger.info(f"Recording session stopped. Duration: {duration:.2f}s, "
                          f"Frames: {self.frame_count}, Detections: {self.detection_count}")

    def record_detection(self, detection_data: Dict[str, Any], frame_number: int, timestamp: float):
        """
        Record detection data for a frame.

        Args:
            detection_data: Dictionary containing detection information
            frame_number: Frame sequence number
            timestamp: Frame timestamp
        """
        if not self.is_recording:
            return

        detection_record = {
            'timestamp': timestamp,
            'frame_number': frame_number,
            'session_name': self.session_name,
            'detections': detection_data
        }

        with self.buffer_lock:
            self.detection_buffer.append(detection_record)
            self.detection_count += len(detection_data.get('objects', []))

            # Auto-flush buffer if it gets too large
            if len(self.detection_buffer) >= 100:
                self._flush_detection_buffer()

    def record_frame_metadata(self, metadata: Dict[str, Any], frame_number: int, timestamp: float):
        """
        Record frame metadata.

        Args:
            metadata: Dictionary containing frame metadata
            frame_number: Frame sequence number
            timestamp: Frame timestamp
        """
        if not self.is_recording:
            return

        metadata_record = {
            'timestamp': timestamp,
            'frame_number': frame_number,
            'session_name': self.session_name,
            'metadata': metadata
        }

        with self.buffer_lock:
            self.metadata_buffer.append(metadata_record)
            self.frame_count += 1

            # Auto-flush buffer if it gets too large
            if len(self.metadata_buffer) >= 100:
                self._flush_metadata_buffer()

    def _flush_buffers(self):
        """Flush all data buffers to disk."""
        self._flush_detection_buffer()
        self._flush_metadata_buffer()

    def _flush_detection_buffer(self):
        """Flush detection buffer to disk."""
        if not self.detection_buffer:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        detection_file = self.detections_dir / f"detections_{timestamp}.json"

        try:
            with open(detection_file, 'w') as f:
                json.dump(self.detection_buffer, f, indent=2, default=str)

            self._logger.debug(f"Flushed {len(self.detection_buffer)} detection records")
            self.detection_buffer.clear()

        except Exception as e:
            self._logger.error(f"Error flushing detection buffer: {e}")

    def _flush_metadata_buffer(self):
        """Flush metadata buffer to disk."""
        if not self.metadata_buffer:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metadata_file = self.metadata_dir / f"metadata_{timestamp}.json"

        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata_buffer, f, indent=2, default=str)

            self._logger.debug(f"Flushed {len(self.metadata_buffer)} metadata records")
            self.metadata_buffer.clear()

        except Exception as e:
            self._logger.error(f"Error flushing metadata buffer: {e}")

    def _save_session_metadata(self):
        """Save session metadata to disk."""
        metadata_file = self.session_dir / "session_info.json"

        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.session_metadata, f, indent=2, default=str)

        except Exception as e:
            self._logger.error(f"Error saving session metadata: {e}")

    def export_data(self, format_type: str = "json", output_file: Optional[str] = None) -> str:
        """
        Export recorded data in specified format.

        Args:
            format_type: Export format ('json', 'csv')
            output_file: Custom output filename

        Returns:
            Path to the exported file
        """
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"export_{self.session_name}_{timestamp}.{format_type}"

        export_path = self.session_dir / output_file

        # Collect all detection data
        all_detections = []
        for detection_file in self.detections_dir.glob("*.json"):
            with open(detection_file, 'r') as f:
                detections = json.load(f)
                all_detections.extend(detections)

        if format_type == "json":
            with open(export_path, 'w') as f:
                json.dump(all_detections, f, indent=2, default=str)

        elif format_type == "csv":
            import csv
            with open(export_path, 'w', newline='') as f:
                if all_detections:
                    # Flatten the detection data for CSV
                    fieldnames = ['timestamp', 'frame_number', 'session_name', 'object_id',
                                  'class_name', 'confidence', 'x', 'y', 'z', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for record in all_detections:
                        base_row = {
                            'timestamp': record['timestamp'],
                            'frame_number': record['frame_number'],
                            'session_name': record['session_name']
                        }

                        for obj in record.get('detections', {}).get('objects', []):
                            row = base_row.copy()
                            row.update({
                                'object_id': obj.get('id'),
                                'class_name': obj.get('class'),
                                'confidence': obj.get('confidence'),
                                'x': obj.get('position', {}).get('x'),
                                'y': obj.get('position', {}).get('y'),
                                'z': obj.get('position', {}).get('z'),
                                'bbox_x1': obj.get('bbox', [None, None, None, None])[0],
                                'bbox_y1': obj.get('bbox', [None, None, None, None])[1],
                                'bbox_x2': obj.get('bbox', [None, None, None, None])[2],
                                'bbox_y2': obj.get('bbox', [None, None, None, None])[3],
                            })
                            writer.writerow(row)

        self._logger.info(f"Data exported to: {export_path}")
        return str(export_path)


class PerformanceMonitor:
    """Performance monitoring and statistical analysis utilities."""

    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor.

        Args:
            window_size: Size of the sliding window for statistics
        """
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.timers = {}
        self.start_times = {}
        self._logger = Logger("PerformanceMonitor")

    def start_timer(self, name: str):
        """Start a named timer."""
        self.start_times[name] = time.perf_counter()

    def end_timer(self, name: str) -> float:
        """
        End a named timer and record the elapsed time.

        Args:
            name: Timer name

        Returns:
            Elapsed time in seconds
        """
        if name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[name]
            self.add_metric(f"{name}_time", elapsed)
            del self.start_times[name]
            return elapsed
        return 0.0

    def add_metric(self, name: str, value: float):
        """Add a metric value to the monitoring system."""
        self.metrics[name].append(value)

    def get_fps(self, timer_name: str = "frame_time") -> float:
        """Calculate FPS based on frame timing."""
        frame_times = self.metrics.get(timer_name)
        if frame_times and len(frame_times) > 1:
            avg_frame_time = sum(frame_times) / len(frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        return 0.0

    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistical summary for a metric."""
        values = list(self.metrics.get(metric_name, []))
        if not values:
            return {}

        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'std': self._calculate_std(values) if len(values) > 1 else 0.0
        }

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system performance statistics."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / (1024 ** 3),
                'disk_usage_percent': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage(
                    'C:\\').percent
            }
        except Exception as e:
            self._logger.warning(f"Error getting system stats: {e}")
            return {}

    def log_performance_summary(self):
        """Log a summary of all performance metrics."""
        self._logger.info("=== Performance Summary ===")

        # System stats
        sys_stats = self.get_system_stats()
        if sys_stats:
            self._logger.info(f"System: CPU {sys_stats.get('cpu_percent', 0):.1f}%, "
                              f"Memory {sys_stats.get('memory_percent', 0):.1f}%")

        # Timing metrics
        for metric_name in self.metrics:
            if metric_name.endswith('_time'):
                stats = self.get_statistics(metric_name)
                if stats:
                    self._logger.info(f"{metric_name}: avg={stats['mean'] * 1000:.2f}ms, "
                                      f"min={stats['min'] * 1000:.2f}ms, max={stats['max'] * 1000:.2f}ms")

        # FPS
        fps = self.get_fps()
        if fps > 0:
            self._logger.info(f"Average FPS: {fps:.2f}")

        self._logger.info("=== End Performance Summary ===")


# Convenience function for easy logger creation
def get_logger(name: str = "RealSenseDetection", level: str = "INFO", log_dir: str = "logs") -> Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files

    Returns:
        Configured Logger instance
    """
    return Logger(name, level, log_dir)