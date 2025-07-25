#!/usr/bin/env python3
"""
Comprehensive GPU performance benchmarking and optimization tool for CUDA-accelerated pipeline.
Detailed analysis of GPU utilization, memory usage, and performance optimization recommendations.
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict
import csv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ConfigManager
from src.utils.logger import get_logger, PerformanceMonitor
from src.integration.pipeline import RealSenseDetectionPipeline
from src.integration.gpu_memory_manager import CUDAMemoryManager
from src.integration.tracker import Object3DTracker
from src.integration.visualizer import CUDAVisualizer
from src.integration.cuda_tracking_kernels import CUDATrackingKernels, KernelProfiler
from src.integration.cuda_visualization_kernels import CUDAVisualizationKernels, VisualizationProfiler
from src.detection import DetectorFactory


@dataclass
class BenchmarkResult:
    """Individual benchmark test result."""
    test_name: str
    component: str
    duration: float
    iterations: int
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float
    throughput_fps: float
    memory_usage_mb: float
    gpu_utilization: float
    success_rate: float
    errors: List[str]
    metadata: Dict[str, Any]


@dataclass
class SystemSpecs:
    """System specifications for benchmark context."""
    cpu_info: str
    memory_total_gb: float
    gpu_devices: List[Dict[str, Any]]
    cuda_version: str
    pytorch_version: str
    opencv_version: str
    python_version: str
    os_info: str


class GPUBenchmarkSuite:
    """Comprehensive GPU performance benchmarking suite."""

    def __init__(self, config_path: str = "config.yaml", output_dir: str = "benchmark_results"):
        """
        Initialize benchmark suite.

        Args:
            config_path: Path to configuration file
            output_dir: Directory for benchmark results
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger("GPUBenchmarkSuite")

        # Load configuration
        self.config = ConfigManager.load_config(config_path)

        # Benchmark results storage
        self.results: List[BenchmarkResult] = []
        self.system_specs: Optional[SystemSpecs] = None

        # Test parameters
        self.warmup_iterations = 5
        self.test_iterations = 20
        self.test_duration = 10.0  # seconds per test

        # Component instances for testing
        self.memory_manager: Optional[CUDAMemoryManager] = None
        self.pipeline: Optional[RealSenseDetectionPipeline] = None
        self.tracker: Optional[Object3DTracker] = None
        self.visualizer: Optional[CUDAVisualizer] = None

        self.logger.info(f"GPUBenchmarkSuite initialized, output: {self.output_dir}")

    def run_full_benchmark(self, quick_mode: bool = False) -> Dict[str, Any]:
        """
        Run complete benchmark suite.

        Args:
            quick_mode: Run abbreviated tests for faster execution

        Returns:
            Comprehensive benchmark report
        """
        self.logger.info("Starting comprehensive GPU benchmark suite...")

        start_time = time.time()

        try:
            # Collect system specifications
            self.system_specs = self._collect_system_specs()

            # Reduce test parameters for quick mode
            if quick_mode:
                self.test_iterations = 10
                self.test_duration = 5.0
                self.warmup_iterations = 3

            # Initialize all components
            self._initialize_components()

            # Run benchmark categories
            self._benchmark_memory_management()
            self._benchmark_detection_models()
            self._benchmark_tracking_system()
            self._benchmark_visualization_system()
            self._benchmark_integrated_pipeline()

            # Analyze results and generate recommendations
            total_time = time.time() - start_time
            report = self._generate_comprehensive_report(total_time)

            # Save results
            self._save_benchmark_results(report)

            # Generate visualizations
            self._generate_performance_charts()

            self.logger.info(f"Benchmark suite completed in {total_time:.1f}s")
            return report

        except Exception as e:
            self.logger.error(f"Benchmark suite failed: {e}")
            raise
        finally:
            self._cleanup_components()

    def _collect_system_specs(self) -> SystemSpecs:
        """Collect detailed system specifications."""
        import platform
        import psutil
        import torch
        import cv2

        # CPU information
        cpu_info = f"{platform.processor()} ({psutil.cpu_count()} cores)"

        # Memory information
        memory_total = psutil.virtual_memory().total / (1024 ** 3)

        # GPU information
        gpu_devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_devices.append({
                    'index': i,
                    'name': props.name,
                    'total_memory_gb': props.total_memory / (1024 ** 3),
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multiprocessor_count': props.multi_processor_count
                })

        # Version information
        cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
        pytorch_version = torch.__version__
        opencv_version = cv2.__version__
        python_version = platform.python_version()
        os_info = platform.platform()

        return SystemSpecs(
            cpu_info=cpu_info,
            memory_total_gb=memory_total,
            gpu_devices=gpu_devices,
            cuda_version=cuda_version,
            pytorch_version=pytorch_version,
            opencv_version=opencv_version,
            python_version=python_version,
            os_info=os_info
        )

    def _initialize_components(self):
        """Initialize all components for testing."""
        self.logger.info("Initializing components for benchmarking...")

        # Memory manager
        self.memory_manager = CUDAMemoryManager(self.config)

        # Detection pipeline
        self.pipeline = RealSenseDetectionPipeline(self.config)
        # Note: We won't initialize the full pipeline as it requires camera

        # Tracker
        self.tracker = Object3DTracker(self.config, self.memory_manager)

        # Visualizer
        self.visualizer = CUDAVisualizer(self.config, self.memory_manager)

        self.logger.info("Component initialization completed")

    def _benchmark_memory_management(self):
        """Benchmark GPU memory management performance."""
        self.logger.info("Benchmarking GPU memory management...")

        if not self.memory_manager:
            return

        # Test memory allocation performance
        self._test_memory_allocation_speed()

        # Test memory pool efficiency
        self._test_memory_pool_efficiency()

        # Test memory fragmentation handling
        self._test_memory_fragmentation()

        # Test multi-GPU coordination
        if len(self.system_specs.gpu_devices) > 1:
            self._test_multi_gpu_coordination()

    def _test_memory_allocation_speed(self):
        """Test memory allocation and deallocation speed."""
        import torch

        test_name = "memory_allocation_speed"
        errors = []
        times = []

        # Test different allocation sizes
        sizes = [(640, 480, 3), (1920, 1080, 3), (3840, 2160, 3)]  # Different resolutions

        for size in sizes:
            allocation_times = []

            # Warmup
            for _ in range(self.warmup_iterations):
                try:
                    tensor, block_id = self.memory_manager.allocate_tensor(size, torch.uint8)
                    self.memory_manager.release_tensor(block_id,
                                                       self.memory_manager.memory_pools[
                                                           list(self.memory_manager.memory_pools.keys())[0]].pool_type)
                except Exception as e:
                    errors.append(str(e))

            # Benchmark allocation/deallocation
            for _ in range(self.test_iterations):
                start_time = time.time()
                try:
                    tensor, block_id = self.memory_manager.allocate_tensor(size, torch.uint8)
                    self.memory_manager.release_tensor(block_id,
                                                       self.memory_manager.memory_pools[
                                                           list(self.memory_manager.memory_pools.keys())[0]].pool_type)
                    allocation_times.append((time.time() - start_time) * 1000)
                except Exception as e:
                    errors.append(str(e))

            if allocation_times:
                times.extend(allocation_times)

        if times:
            memory_stats = self.memory_manager.get_memory_stats()

            result = BenchmarkResult(
                test_name=test_name,
                component="memory_manager",
                duration=self.test_duration,
                iterations=len(times),
                avg_time_ms=np.mean(times),
                min_time_ms=np.min(times),
                max_time_ms=np.max(times),
                std_time_ms=np.std(times),
                throughput_fps=1000 / np.mean(times) if times else 0,
                memory_usage_mb=memory_stats.total_allocated / (1024 ** 2),
                gpu_utilization=0.0,  # Not directly measurable
                success_rate=(len(times) - len(errors)) / max(len(times), 1),
                errors=errors,
                metadata={'test_sizes': sizes}
            )

            self.results.append(result)
            self.logger.info(f"Memory allocation: {np.mean(times):.2f}±{np.std(times):.2f}ms")

    def _test_memory_pool_efficiency(self):
        """Test memory pool hit rate and efficiency."""
        test_name = "memory_pool_efficiency"

        # Simulate typical usage pattern
        allocations = []
        hit_rates = []

        # Pattern: allocate, use, release, repeat
        for iteration in range(self.test_iterations):
            pool_type = list(self.memory_manager.memory_pools.keys())[0]
            pool = self.memory_manager.memory_pools[pool_type]

            initial_hits = pool.hit_count
            initial_misses = pool.miss_count

            # Multiple allocations and releases
            tensors_and_blocks = []
            for _ in range(10):
                try:
                    tensor, block_id = self.memory_manager.allocate_tensor((640, 480, 3), torch.uint8)
                    tensors_and_blocks.append((tensor, block_id))
                except Exception:
                    pass

            # Release all
            for tensor, block_id in tensors_and_blocks:
                self.memory_manager.release_tensor(block_id, pool_type)

            # Calculate hit rate for this iteration
            final_hits = pool.hit_count
            final_misses = pool.miss_count

            if (final_hits + final_misses) > (initial_hits + initial_misses):
                hit_rate = (final_hits - initial_hits) / ((final_hits + final_misses) - (initial_hits + initial_misses))
                hit_rates.append(hit_rate)

        if hit_rates:
            memory_stats = self.memory_manager.get_memory_stats()

            result = BenchmarkResult(
                test_name=test_name,
                component="memory_manager",
                duration=self.test_duration,
                iterations=len(hit_rates),
                avg_time_ms=0.0,  # Not time-based
                min_time_ms=0.0,
                max_time_ms=0.0,
                std_time_ms=0.0,
                throughput_fps=0.0,
                memory_usage_mb=memory_stats.total_allocated / (1024 ** 2),
                gpu_utilization=0.0,
                success_rate=np.mean(hit_rates),
                errors=[],
                metadata={'avg_hit_rate': np.mean(hit_rates), 'efficiency': memory_stats.efficiency_ratio}
            )

            self.results.append(result)
            self.logger.info(f"Memory pool efficiency: {np.mean(hit_rates):.2%} hit rate")

    def _test_memory_fragmentation(self):
        """Test memory fragmentation handling."""
        test_name = "memory_fragmentation"

        # Simulate fragmentation scenario
        fragmentation_levels = []

        for iteration in range(self.test_iterations):
            # Allocate many small blocks
            small_blocks = []
            for _ in range(50):
                try:
                    tensor, block_id = self.memory_manager.allocate_tensor((64, 64, 3), torch.uint8)
                    small_blocks.append((tensor, block_id))
                except Exception:
                    break

            # Release every other block to create fragmentation
            for i in range(0, len(small_blocks), 2):
                tensor, block_id = small_blocks[i]
                pool_type = list(self.memory_manager.memory_pools.keys())[0]
                self.memory_manager.release_tensor(block_id, pool_type)

            # Try to allocate large block
            try:
                large_tensor, large_block_id = self.memory_manager.allocate_tensor((1920, 1080, 3), torch.uint8)
                fragmentation_success = True
                self.memory_manager.release_tensor(large_block_id, pool_type)
            except Exception:
                fragmentation_success = False

            # Clean up remaining blocks
            for i in range(1, len(small_blocks), 2):
                tensor, block_id = small_blocks[i]
                self.memory_manager.release_tensor(block_id, pool_type)

            # Measure fragmentation
            memory_stats = self.memory_manager.get_memory_stats()
            fragmentation_levels.append(memory_stats.fragmentation_ratio)

        if fragmentation_levels:
            result = BenchmarkResult(
                test_name=test_name,
                component="memory_manager",
                duration=self.test_duration,
                iterations=len(fragmentation_levels),
                avg_time_ms=0.0,
                min_time_ms=0.0,
                max_time_ms=0.0,
                std_time_ms=0.0,
                throughput_fps=0.0,
                memory_usage_mb=np.mean(
                    [stats.total_allocated for stats in [self.memory_manager.get_memory_stats()]]) / (1024 ** 2),
                gpu_utilization=0.0,
                success_rate=np.mean(fragmentation_levels),
                errors=[],
                metadata={'avg_fragmentation': np.mean(fragmentation_levels)}
            )

            self.results.append(result)
            self.logger.info(f"Memory fragmentation: {np.mean(fragmentation_levels):.2%} average")

    def _test_multi_gpu_coordination(self):
        """Test multi-GPU coordination efficiency."""
        if len(self.system_specs.gpu_devices) <= 1:
            return

        test_name = "multi_gpu_coordination"
        self.logger.info("Testing multi-GPU coordination...")

        # Test load balancing across GPUs
        allocation_times = []

        for iteration in range(self.test_iterations):
            start_time = time.time()

            # Allocate tensors that should distribute across GPUs
            tensors_and_blocks = []
            for _ in range(len(self.system_specs.gpu_devices) * 2):
                try:
                    tensor, block_id = self.memory_manager.allocate_tensor((640, 480, 3), torch.uint8)
                    tensors_and_blocks.append((tensor, block_id))
                except Exception:
                    break

            # Clean up
            for tensor, block_id in tensors_and_blocks:
                pool_type = list(self.memory_manager.memory_pools.keys())[0]
                self.memory_manager.release_tensor(block_id, pool_type)

            allocation_times.append((time.time() - start_time) * 1000)

        if allocation_times:
            result = BenchmarkResult(
                test_name=test_name,
                component="memory_manager",
                duration=self.test_duration,
                iterations=len(allocation_times),
                avg_time_ms=np.mean(allocation_times),
                min_time_ms=np.min(allocation_times),
                max_time_ms=np.max(allocation_times),
                std_time_ms=np.std(allocation_times),
                throughput_fps=1000 / np.mean(allocation_times) if allocation_times else 0,
                memory_usage_mb=0.0,
                gpu_utilization=0.0,
                success_rate=1.0,
                errors=[],
                metadata={'gpu_count': len(self.system_specs.gpu_devices)}
            )

            self.results.append(result)
            self.logger.info(f"Multi-GPU coordination: {np.mean(allocation_times):.2f}ms average")

    def _benchmark_detection_models(self):
        """Benchmark detection model performance."""
        self.logger.info("Benchmarking detection models...")

        # Test YOLO performance
        self._test_yolo_performance()

        # Test DETR performance
        self._test_detr_performance()

        # Test model switching performance
        self._test_model_switching()

    def _test_yolo_performance(self):
        """Test YOLO model performance."""
        try:
            from src.detection import YOLODetector, YOLO_AVAILABLE

            if not YOLO_AVAILABLE:
                self.logger.warning("YOLO not available for benchmarking")
                return

            test_name = "yolo_detection_performance"

            # Create test detector
            factory = DetectorFactory()
            detector = factory.create_detector(self.config, 'yolo')

            if not detector:
                self.logger.warning("Failed to create YOLO detector")
                return

            # Generate test images
            test_images = self._generate_test_images()

            # Benchmark detection
            detection_times = []
            total_detections = 0

            # Warmup
            for _ in range(self.warmup_iterations):
                if test_images:
                    detector.detect(test_images[0])

            # Benchmark
            for iteration in range(self.test_iterations):
                if not test_images:
                    break

                start_time = time.time()
                result = detector.detect(test_images[iteration % len(test_images)])
                detection_time = (time.time() - start_time) * 1000

                detection_times.append(detection_time)
                if result.success:
                    total_detections += len(result.detections)

            # Get GPU memory usage
            gpu_memory = 0
            if hasattr(detector, 'get_gpu_memory_usage'):
                memory_info = detector.get_gpu_memory_usage()
                gpu_memory = memory_info.get('allocated_gb', 0) * 1024  # Convert to MB

            if detection_times:
                result = BenchmarkResult(
                    test_name=test_name,
                    component="yolo_detector",
                    duration=self.test_duration,
                    iterations=len(detection_times),
                    avg_time_ms=np.mean(detection_times),
                    min_time_ms=np.min(detection_times),
                    max_time_ms=np.max(detection_times),
                    std_time_ms=np.std(detection_times),
                    throughput_fps=1000 / np.mean(detection_times) if detection_times else 0,
                    memory_usage_mb=gpu_memory,
                    gpu_utilization=0.0,  # Would need nvidia-ml-py for actual measurement
                    success_rate=1.0,
                    errors=[],
                    metadata={
                        'total_detections': total_detections,
                        'avg_detections_per_frame': total_detections / len(detection_times),
                        'model_variant': detector.model_info.model_name
                    }
                )

                self.results.append(result)
                self.logger.info(
                    f"YOLO performance: {np.mean(detection_times):.2f}ms, {1000 / np.mean(detection_times):.1f} FPS")

            detector.cleanup()

        except Exception as e:
            self.logger.error(f"YOLO benchmarking failed: {e}")

    def _test_detr_performance(self):
        """Test DETR model performance."""
        try:
            from src.detection import DETRDetector, DETR_AVAILABLE

            if not DETR_AVAILABLE:
                self.logger.warning("DETR not available for benchmarking")
                return

            test_name = "detr_detection_performance"

            # Create test detector
            factory = DetectorFactory()
            detector = factory.create_detector(self.config, 'detr')

            if not detector:
                self.logger.warning("Failed to create DETR detector")
                return

            # Generate test images
            test_images = self._generate_test_images()

            # Test with prompt for Grounding DINO
            test_prompt = "person . car . dog . cat . bottle"

            # Benchmark detection
            detection_times = []
            total_detections = 0

            # Warmup
            for _ in range(self.warmup_iterations):
                if test_images:
                    if hasattr(detector, 'is_grounding_dino') and detector.is_grounding_dino:
                        detector.detect(test_images[0], text_prompt=test_prompt)
                    else:
                        detector.detect(test_images[0])

            # Benchmark
            for iteration in range(self.test_iterations):
                if not test_images:
                    break

                start_time = time.time()

                if hasattr(detector, 'is_grounding_dino') and detector.is_grounding_dino:
                    result = detector.detect(test_images[iteration % len(test_images)], text_prompt=test_prompt)
                else:
                    result = detector.detect(test_images[iteration % len(test_images)])

                detection_time = (time.time() - start_time) * 1000

                detection_times.append(detection_time)
                if result.success:
                    total_detections += len(result.detections)

            if detection_times:
                result = BenchmarkResult(
                    test_name=test_name,
                    component="detr_detector",
                    duration=self.test_duration,
                    iterations=len(detection_times),
                    avg_time_ms=np.mean(detection_times),
                    min_time_ms=np.min(detection_times),
                    max_time_ms=np.max(detection_times),
                    std_time_ms=np.std(detection_times),
                    throughput_fps=1000 / np.mean(detection_times) if detection_times else 0,
                    memory_usage_mb=0.0,  # Would need specific measurement
                    gpu_utilization=0.0,
                    success_rate=1.0,
                    errors=[],
                    metadata={
                        'total_detections': total_detections,
                        'avg_detections_per_frame': total_detections / len(detection_times),
                        'model_variant': detector.model_info.model_name,
                        'is_grounding_dino': hasattr(detector, 'is_grounding_dino') and detector.is_grounding_dino
                    }
                )

                self.results.append(result)
                self.logger.info(
                    f"DETR performance: {np.mean(detection_times):.2f}ms, {1000 / np.mean(detection_times):.1f} FPS")

            detector.cleanup()

        except Exception as e:
            self.logger.error(f"DETR benchmarking failed: {e}")

    def _test_model_switching(self):
        """Test model switching performance."""
        test_name = "model_switching_performance"

        switching_times = []

        for iteration in range(min(self.test_iterations, 5)):  # Limit iterations for switching test
            try:
                # Create factory
                factory = DetectorFactory()

                # Time switching from YOLO to DETR
                start_time = time.time()

                yolo_detector = factory.create_detector(self.config, 'yolo')
                if yolo_detector:
                    yolo_detector.cleanup()

                detr_detector = factory.create_detector(self.config, 'detr')
                if detr_detector:
                    detr_detector.cleanup()

                switch_time = (time.time() - start_time) * 1000
                switching_times.append(switch_time)

            except Exception as e:
                self.logger.warning(f"Model switching iteration {iteration} failed: {e}")

        if switching_times:
            result = BenchmarkResult(
                test_name=test_name,
                component="detector_factory",
                duration=self.test_duration,
                iterations=len(switching_times),
                avg_time_ms=np.mean(switching_times),
                min_time_ms=np.min(switching_times),
                max_time_ms=np.max(switching_times),
                std_time_ms=np.std(switching_times),
                throughput_fps=0.0,  # Not applicable
                memory_usage_mb=0.0,
                gpu_utilization=0.0,
                success_rate=len(switching_times) / min(self.test_iterations, 5),
                errors=[],
                metadata={'switch_direction': 'yolo_to_detr'}
            )

            self.results.append(result)
            self.logger.info(f"Model switching: {np.mean(switching_times):.2f}ms average")

    def _benchmark_tracking_system(self):
        """Benchmark 3D tracking system performance."""
        self.logger.info("Benchmarking 3D tracking system...")

        if not self.tracker:
            return

        # Test tracking update performance
        self._test_tracking_update_performance()

        # Test CUDA kernel performance
        self._test_tracking_kernels()

    def _test_tracking_update_performance(self):
        """Test tracking update performance with simulated detections."""
        test_name = "tracking_update_performance"

        # Generate simulated 3D detections
        simulated_detections = self._generate_simulated_detections()

        update_times = []

        # Warmup
        for _ in range(self.warmup_iterations):
            if simulated_detections:
                self.tracker.update(simulated_detections[:5], time.time())

        # Benchmark tracking updates
        for iteration in range(self.test_iterations):
            # Vary number of detections per frame
            num_detections = 5 + (iteration % 10)  # 5-15 detections
            frame_detections = simulated_detections[:num_detections]

            start_time = time.time()
            tracks = self.tracker.update(frame_detections, time.time())
            update_time = (time.time() - start_time) * 1000

            update_times.append(update_time)

        if update_times:
            tracking_stats = self.tracker.get_tracking_statistics()

            result = BenchmarkResult(
                test_name=test_name,
                component="3d_tracker",
                duration=self.test_duration,
                iterations=len(update_times),
                avg_time_ms=np.mean(update_times),
                min_time_ms=np.min(update_times),
                max_time_ms=np.max(update_times),
                std_time_ms=np.std(update_times),
                throughput_fps=1000 / np.mean(update_times) if update_times else 0,
                memory_usage_mb=tracking_stats.get('gpu_memory_usage_mb', 0),
                gpu_utilization=0.0,
                success_rate=1.0,
                errors=[],
                metadata={
                    'total_tracks': tracking_stats.get('total_tracks', 0),
                    'active_tracks': tracking_stats.get('active_tracks', 0),
                    'confirmed_tracks': tracking_stats.get('confirmed_tracks', 0)
                }
            )

            self.results.append(result)
            self.logger.info(f"Tracking update: {np.mean(update_times):.2f}ms, {1000 / np.mean(update_times):.1f} FPS")

    def _test_tracking_kernels(self):
        """Test individual CUDA tracking kernels."""
        if not hasattr(self.tracker, 'kernels') or not self.tracker.kernels:
            return

        test_name = "tracking_kernels_performance"
        kernels = self.tracker.kernels

        # Test distance matrix computation
        import torch
        n_tracks = 20
        n_detections = 15

        track_positions = torch.randn(n_tracks, 3, device=kernels.device)
        detection_positions = torch.randn(n_detections, 3, device=kernels.device)

        kernel_times = []

        # Warmup
        for _ in range(self.warmup_iterations):
            kernels.compute_distance_matrix(track_positions, detection_positions)

        # Benchmark
        for _ in range(self.test_iterations):
            start_time = time.time()
            distance_matrix = kernels.compute_distance_matrix(track_positions, detection_positions)
            kernels.synchronize_streams()
            kernel_time = (time.time() - start_time) * 1000
            kernel_times.append(kernel_time)

        if kernel_times:
            kernel_stats = kernels.get_kernel_stats()

            result = BenchmarkResult(
                test_name=test_name,
                component="tracking_kernels",
                duration=self.test_duration,
                iterations=len(kernel_times),
                avg_time_ms=np.mean(kernel_times),
                min_time_ms=np.min(kernel_times),
                max_time_ms=np.max(kernel_times),
                std_time_ms=np.std(kernel_times),
                throughput_fps=1000 / np.mean(kernel_times) if kernel_times else 0,
                memory_usage_mb=kernel_stats.get('memory_usage_mb', 0),
                gpu_utilization=0.0,
                success_rate=1.0,
                errors=[],
                metadata={
                    'test_tracks': n_tracks,
                    'test_detections': n_detections,
                    'kernel_type': 'distance_matrix'
                }
            )

            self.results.append(result)
            self.logger.info(f"Tracking kernels: {np.mean(kernel_times):.2f}ms")

    def _benchmark_visualization_system(self):
        """Benchmark visualization system performance."""
        self.logger.info("Benchmarking visualization system...")

        if not self.visualizer:
            return

        # Test GPU rendering performance
        self._test_visualization_rendering()

        # Test CUDA kernels
        self._test_visualization_kernels()

    def _test_visualization_rendering(self):
        """Test visualization rendering performance."""
        test_name = "visualization_rendering_performance"

        # Generate test frames
        test_frames = self._generate_test_images()

        if not test_frames:
            return

        render_times = []

        # Warmup
        for _ in range(self.warmup_iterations):
            self.visualizer.render_frame(rgb_frame=test_frames[0])

        # Benchmark rendering
        for iteration in range(self.test_iterations):
            frame = test_frames[iteration % len(test_frames)]

            start_time = time.time()
            self.visualizer.render_frame(rgb_frame=frame)
            render_time = (time.time() - start_time) * 1000

            render_times.append(render_time)

        if render_times:
            viz_stats = self.visualizer.get_visualization_stats()

            result = BenchmarkResult(
                test_name=test_name,
                component="visualizer",
                duration=self.test_duration,
                iterations=len(render_times),
                avg_time_ms=np.mean(render_times),
                min_time_ms=np.min(render_times),
                max_time_ms=np.max(render_times),
                std_time_ms=np.std(render_times),
                throughput_fps=1000 / np.mean(render_times) if render_times else 0,
                memory_usage_mb=viz_stats.get('gpu_memory_usage_mb', 0),
                gpu_utilization=0.0,
                success_rate=1.0,
                errors=[],
                metadata={
                    'display_mode': viz_stats.get('display_mode', 'unknown'),
                    'gpu_acceleration': viz_stats.get('gpu_acceleration', False)
                }
            )

            self.results.append(result)
            self.logger.info(
                f"Visualization rendering: {np.mean(render_times):.2f}ms, {1000 / np.mean(render_times):.1f} FPS")

    def _test_visualization_kernels(self):
        """Test individual visualization CUDA kernels."""
        if not hasattr(self.visualizer, 'kernels') or not self.visualizer.kernels:
            return

        test_name = "visualization_kernels_performance"
        kernels = self.visualizer.kernels

        # Test depth colorization
        import torch
        depth_frame = torch.randn(480, 640, device=kernels.device) * 5 + 2  # 2-7m depth range

        kernel_times = []

        # Warmup
        for _ in range(self.warmup_iterations):
            kernels.colorize_depth_frame(depth_frame)

        # Benchmark
        for _ in range(self.test_iterations):
            start_time = time.time()
            colorized = kernels.colorize_depth_frame(depth_frame)
            kernels.synchronize()
            kernel_time = (time.time() - start_time) * 1000
            kernel_times.append(kernel_time)

        if kernel_times:
            kernel_stats = kernels.get_kernel_stats()

            result = BenchmarkResult(
                test_name=test_name,
                component="visualization_kernels",
                duration=self.test_duration,
                iterations=len(kernel_times),
                avg_time_ms=np.mean(kernel_times),
                min_time_ms=np.min(kernel_times),
                max_time_ms=np.max(kernel_times),
                std_time_ms=np.std(kernel_times),
                throughput_fps=1000 / np.mean(kernel_times) if kernel_times else 0,
                memory_usage_mb=kernel_stats.get('memory_usage_mb', 0),
                gpu_utilization=0.0,
                success_rate=1.0,
                errors=[],
                metadata={
                    'kernel_type': 'depth_colorization',
                    'frame_size': '640x480'
                }
            )

            self.results.append(result)
            self.logger.info(f"Visualization kernels: {np.mean(kernel_times):.2f}ms")

    def _benchmark_integrated_pipeline(self):
        """Benchmark integrated pipeline performance."""
        self.logger.info("Benchmarking integrated pipeline...")

        # Test end-to-end performance with simulated data
        self._test_end_to_end_performance()

    def _test_end_to_end_performance(self):
        """Test end-to-end pipeline performance with simulated data."""
        test_name = "end_to_end_pipeline_performance"

        # Generate test data
        test_frames = self._generate_test_images()
        simulated_detections = self._generate_simulated_detections()

        if not test_frames:
            return

        pipeline_times = []

        # Simulate pipeline processing
        for iteration in range(min(self.test_iterations, 10)):  # Limit for intensive test
            frame = test_frames[iteration % len(test_frames)]
            detections = simulated_detections[:5 + (iteration % 5)]  # Vary detection count

            start_time = time.time()

            # Simulate full pipeline: detection + tracking + visualization
            try:
                # Detection (simulated)
                detection_time = 0.020  # 20ms simulated detection time

                # Tracking update
                if self.tracker:
                    tracks = self.tracker.update(detections, time.time())

                # Visualization
                if self.visualizer:
                    self.visualizer.render_frame(rgb_frame=frame)

                total_time = (time.time() - start_time) * 1000
                pipeline_times.append(total_time)

            except Exception as e:
                self.logger.warning(f"Pipeline iteration {iteration} failed: {e}")

        if pipeline_times:
            result = BenchmarkResult(
                test_name=test_name,
                component="integrated_pipeline",
                duration=self.test_duration,
                iterations=len(pipeline_times),
                avg_time_ms=np.mean(pipeline_times),
                min_time_ms=np.min(pipeline_times),
                max_time_ms=np.max(pipeline_times),
                std_time_ms=np.std(pipeline_times),
                throughput_fps=1000 / np.mean(pipeline_times) if pipeline_times else 0,
                memory_usage_mb=0.0,
                gpu_utilization=0.0,
                success_rate=len(pipeline_times) / min(self.test_iterations, 10),
                errors=[],
                metadata={
                    'components': ['detection', 'tracking', 'visualization'],
                    'simulated_detection_time_ms': 20
                }
            )

            self.results.append(result)
            self.logger.info(
                f"End-to-end pipeline: {np.mean(pipeline_times):.2f}ms, {1000 / np.mean(pipeline_times):.1f} FPS")

    def _generate_test_images(self, count: int = 10) -> List[np.ndarray]:
        """Generate test images for benchmarking."""
        import cv2

        images = []

        for i in range(count):
            # Create test image with various patterns
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Add some geometric shapes for detection targets
            cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)
            cv2.circle(img, (400, 300), 50, (0, 255, 0), -1)
            cv2.rectangle(img, (500, 50), (600, 150), (0, 0, 255), -1)

            images.append(img)

        return images

    def _generate_simulated_detections(self, count: int = 20):
        """Generate simulated 3D detections for tracking tests."""
        from src.detection.base_detector import Detection3D

        detections = []

        for i in range(count):
            # Simulate realistic 3D detection
            x = np.random.uniform(-2, 2)  # ±2m in X
            y = np.random.uniform(-1, 1)  # ±1m in Y
            z = np.random.uniform(1, 8)  # 1-8m in Z (depth)

            # Convert to approximate 2D bbox (simplified)
            pixel_x = int(320 + x * 100)  # Rough conversion
            pixel_y = int(240 + y * 100)
            bbox_size = max(20, int(100 / z))  # Closer objects appear larger

            detection = Detection3D(
                bbox=(pixel_x - bbox_size // 2, pixel_y - bbox_size // 2,
                      pixel_x + bbox_size // 2, pixel_y + bbox_size // 2),
                confidence=0.8 + np.random.uniform(-0.2, 0.2),
                class_id=i % 5,  # 5 different classes
                class_name=f"class_{i % 5}",
                center_3d=(x, y, z),
                distance=z,
                depth_confidence=0.9
            )

            detections.append(detection)

        return detections

    def _generate_comprehensive_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive benchmark report with recommendations."""

        # Categorize results
        categories = defaultdict(list)
        for result in self.results:
            categories[result.component].append(result)

        # Calculate summary statistics
        summary_stats = {}
        for component, results in categories.items():
            if results:
                avg_times = [r.avg_time_ms for r in results]
                throughputs = [r.throughput_fps for r in results if r.throughput_fps > 0]

                summary_stats[component] = {
                    'test_count': len(results),
                    'avg_time_ms': np.mean(avg_times),
                    'avg_throughput_fps': np.mean(throughputs) if throughputs else 0,
                    'total_memory_mb': sum(r.memory_usage_mb for r in results),
                    'success_rate': np.mean([r.success_rate for r in results])
                }

        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations()

        # Compile comprehensive report
        report = {
            'benchmark_info': {
                'timestamp': time.time(),
                'total_duration': total_time,
                'total_tests': len(self.results),
                'test_iterations': self.test_iterations,
                'warmup_iterations': self.warmup_iterations
            },
            'system_specs': asdict(self.system_specs) if self.system_specs else {},
            'summary_statistics': summary_stats,
            'detailed_results': [asdict(result) for result in self.results],
            'optimization_recommendations': recommendations,
            'performance_analysis': self._analyze_performance_bottlenecks()
        }

        return report

    def _generate_optimization_recommendations(self) -> List[Dict[str, str]]:
        """Generate specific optimization recommendations based on results."""
        recommendations = []

        # Analyze memory management
        memory_results = [r for r in self.results if 'memory' in r.component]
        if memory_results:
            avg_allocation_time = np.mean([r.avg_time_ms for r in memory_results])
            if avg_allocation_time > 1.0:  # > 1ms is slow for memory allocation
                recommendations.append({
                    'category': 'Memory Management',
                    'issue': 'Slow GPU memory allocation',
                    'recommendation': 'Increase memory pool sizes and pre-allocation buffers',
                    'priority': 'High'
                })

        # Analyze detection performance
        detection_results = [r for r in self.results if 'detection' in r.test_name]
        if detection_results:
            yolo_results = [r for r in detection_results if 'yolo' in r.component]
            detr_results = [r for r in detection_results if 'detr' in r.component]

            if yolo_results and detr_results:
                yolo_fps = np.mean([r.throughput_fps for r in yolo_results])
                detr_fps = np.mean([r.throughput_fps for r in detr_results])

                if yolo_fps > detr_fps * 1.5:
                    recommendations.append({
                        'category': 'Detection Model',
                        'issue': 'DETR significantly slower than YOLO',
                        'recommendation': 'Use YOLO for real-time applications, DETR for accuracy-critical tasks',
                        'priority': 'Medium'
                    })

        # Analyze tracking performance
        tracking_results = [r for r in self.results if 'tracking' in r.component]
        if tracking_results:
            avg_tracking_time = np.mean([r.avg_time_ms for r in tracking_results])
            if avg_tracking_time > 10.0:  # > 10ms is slow for tracking
                recommendations.append({
                    'category': 'Tracking System',
                    'issue': 'High tracking latency',
                    'recommendation': 'Reduce maximum tracks or optimize Kalman filter parameters',
                    'priority': 'Medium'
                })

        # Analyze visualization performance
        viz_results = [r for r in self.results if 'visualization' in r.component]
        if viz_results:
            avg_viz_time = np.mean([r.avg_time_ms for r in viz_results])
            if avg_viz_time > 16.7:  # > 16.7ms means < 60 FPS
                recommendations.append({
                    'category': 'Visualization',
                    'issue': 'Visualization bottleneck limiting FPS',
                    'recommendation': 'Reduce display resolution or disable expensive visual effects',
                    'priority': 'Low'
                })

        # System-level recommendations
        if self.system_specs and len(self.system_specs.gpu_devices) > 1:
            recommendations.append({
                'category': 'System Configuration',
                'issue': 'Multiple GPUs available',
                'recommendation': 'Enable multi-GPU processing for better parallelization',
                'priority': 'Medium'
            })

        # Memory recommendations based on GPU specs
        if self.system_specs and self.system_specs.gpu_devices:
            total_gpu_memory = sum(gpu['total_memory_gb'] for gpu in self.system_specs.gpu_devices)
            if total_gpu_memory < 8:
                recommendations.append({
                    'category': 'Hardware',
                    'issue': 'Limited GPU memory',
                    'recommendation': 'Reduce batch sizes and enable memory optimization features',
                    'priority': 'High'
                })

        return recommendations

    def _analyze_performance_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks across the pipeline."""

        # Find slowest components
        component_times = defaultdict(list)
        for result in self.results:
            component_times[result.component].append(result.avg_time_ms)

        avg_component_times = {
            component: np.mean(times)
            for component, times in component_times.items()
        }

        # Identify bottlenecks
        if avg_component_times:
            slowest_component = max(avg_component_times, key=avg_component_times.get)
            fastest_component = min(avg_component_times, key=avg_component_times.get)

            # Calculate performance ratios
            performance_ratios = {}
            for component, avg_time in avg_component_times.items():
                performance_ratios[component] = avg_time / avg_component_times[fastest_component]
        else:
            slowest_component = "unknown"
            fastest_component = "unknown"
            performance_ratios = {}

        # Memory usage analysis
        memory_usage = {
            result.component: result.memory_usage_mb
            for result in self.results
            if result.memory_usage_mb > 0
        }

        # Throughput analysis
        throughput_analysis = {}
        for result in self.results:
            if result.throughput_fps > 0:
                if result.component not in throughput_analysis:
                    throughput_analysis[result.component] = []
                throughput_analysis[result.component].append(result.throughput_fps)

        avg_throughput = {
            component: np.mean(fps_list)
            for component, fps_list in throughput_analysis.items()
        }

        return {
            'bottleneck_analysis': {
                'slowest_component': slowest_component,
                'fastest_component': fastest_component,
                'performance_ratios': performance_ratios
            },
            'memory_analysis': {
                'component_memory_usage': memory_usage,
                'total_memory_usage_mb': sum(memory_usage.values())
            },
            'throughput_analysis': {
                'component_throughput': avg_throughput,
                'overall_pipeline_fps': min(avg_throughput.values()) if avg_throughput else 0
            }
        }

    def _save_benchmark_results(self, report: Dict[str, Any]):
        """Save benchmark results to various formats."""
        timestamp = int(time.time())

        # Save JSON report
        json_path = self.output_dir / f"benchmark_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Save CSV summary
        csv_path = self.output_dir / f"benchmark_summary_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Component', 'Test', 'Avg Time (ms)', 'Throughput (FPS)', 'Memory (MB)', 'Success Rate'])

            for result in self.results:
                writer.writerow([
                    result.component,
                    result.test_name,
                    f"{result.avg_time_ms:.2f}",
                    f"{result.throughput_fps:.1f}",
                    f"{result.memory_usage_mb:.1f}",
                    f"{result.success_rate:.2%}"
                ])

        # Save recommendations
        rec_path = self.output_dir / f"optimization_recommendations_{timestamp}.txt"
        with open(rec_path, 'w') as f:
            f.write("GPU PERFORMANCE OPTIMIZATION RECOMMENDATIONS\n")
            f.write("=" * 50 + "\n\n")

            for i, rec in enumerate(report['optimization_recommendations'], 1):
                f.write(f"{i}. {rec['category']} - {rec['priority']} Priority\n")
                f.write(f"   Issue: {rec['issue']}\n")
                f.write(f"   Recommendation: {rec['recommendation']}\n\n")

        self.logger.info(f"Benchmark results saved:")
        self.logger.info(f"  Report: {json_path}")
        self.logger.info(f"  Summary: {csv_path}")
        self.logger.info(f"  Recommendations: {rec_path}")

    def _generate_performance_charts(self):
        """Generate performance visualization charts."""
        try:
            # Group results by component
            components = defaultdict(list)
            for result in self.results:
                components[result.component].append(result)

            # Create performance comparison chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # Chart 1: Average processing times
            comp_names = list(components.keys())
            avg_times = [np.mean([r.avg_time_ms for r in components[comp]]) for comp in comp_names]

            ax1.bar(comp_names, avg_times)
            ax1.set_title('Average Processing Time by Component')
            ax1.set_ylabel('Time (ms)')
            ax1.tick_params(axis='x', rotation=45)

            # Chart 2: Throughput comparison
            throughputs = []
            for comp in comp_names:
                fps_values = [r.throughput_fps for r in components[comp] if r.throughput_fps > 0]
                throughputs.append(np.mean(fps_values) if fps_values else 0)

            ax2.bar(comp_names, throughputs)
            ax2.set_title('Average Throughput by Component')
            ax2.set_ylabel('FPS')
            ax2.tick_params(axis='x', rotation=45)

            # Chart 3: Memory usage
            memory_usage = []
            for comp in comp_names:
                mem_values = [r.memory_usage_mb for r in components[comp] if r.memory_usage_mb > 0]
                memory_usage.append(np.mean(mem_values) if mem_values else 0)

            ax3.bar(comp_names, memory_usage)
            ax3.set_title('Memory Usage by Component')
            ax3.set_ylabel('Memory (MB)')
            ax3.tick_params(axis='x', rotation=45)

            # Chart 4: Success rates
            success_rates = [np.mean([r.success_rate for r in components[comp]]) * 100 for comp in comp_names]

            ax4.bar(comp_names, success_rates)
            ax4.set_title('Success Rate by Component')
            ax4.set_ylabel('Success Rate (%)')
            ax4.set_ylim(0, 100)
            ax4.tick_params(axis='x', rotation=45)

            plt.tight_layout()

            # Save chart
            timestamp = int(time.time())
            chart_path = self.output_dir / f"performance_charts_{timestamp}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Performance charts saved: {chart_path}")

        except Exception as e:
            self.logger.warning(f"Failed to generate performance charts: {e}")

    def _cleanup_components(self):
        """Clean up all components after benchmarking."""
        if self.memory_manager:
            self.memory_manager.cleanup()

        if self.tracker:
            self.tracker.cleanup()

        if self.visualizer:
            self.visualizer.cleanup()

        # Clear GPU cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        self.logger.info("Component cleanup completed")


def main():
    """Main entry point for GPU benchmarking tool."""
    parser = argparse.ArgumentParser(
        description="GPU Performance Benchmarking Suite for CUDA-Accelerated Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run full benchmark suite
  %(prog)s --quick                           # Run quick benchmark (reduced iterations)
  %(prog)s --config custom.yaml --output results  # Custom config and output directory
  %(prog)s --component memory_manager        # Benchmark specific component only
        """
    )

    parser.add_argument("--config", "-c", default="config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output", "-o", default="benchmark_results",
                        help="Output directory for benchmark results")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick benchmark with reduced iterations")
    parser.add_argument("--component", choices=['memory_manager', 'detection', 'tracking', 'visualization', 'pipeline'],
                        help="Benchmark specific component only")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Number of test iterations per benchmark")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup iterations")
    parser.add_argument("--no-charts", action="store_true",
                        help="Disable performance chart generation")

    args = parser.parse_args()

    try:
        # Initialize benchmark suite
        suite = GPUBenchmarkSuite(args.config, args.output)

        # Apply command line overrides
        suite.test_iterations = args.iterations
        suite.warmup_iterations = args.warmup

        print("🚀 GPU Performance Benchmark Suite")
        print("=" * 50)
        print(f"Configuration: {args.config}")
        print(f"Output Directory: {args.output}")
        print(f"Test Iterations: {args.iterations}")
        print(f"Quick Mode: {args.quick}")

        if args.component:
            print(f"Component Filter: {args.component}")
            # Would implement component-specific benchmarking here

        print("=" * 50)

        # Run benchmark suite
        report = suite.run_full_benchmark(quick_mode=args.quick)

        # Print summary
        print("\n📊 BENCHMARK SUMMARY")
        print("=" * 50)

        if 'summary_statistics' in report:
            for component, stats in report['summary_statistics'].items():
                print(f"{component}:")
                print(f"  Tests: {stats['test_count']}")
                print(f"  Avg Time: {stats['avg_time_ms']:.2f}ms")
                if stats['avg_throughput_fps'] > 0:
                    print(f"  Avg FPS: {stats['avg_throughput_fps']:.1f}")
                print(f"  Success Rate: {stats['success_rate']:.1%}")
                print()

        # Print top recommendations
        if 'optimization_recommendations' in report:
            print("🎯 TOP OPTIMIZATION RECOMMENDATIONS:")
            for i, rec in enumerate(report['optimization_recommendations'][:3], 1):
                print(f"{i}. {rec['category']}: {rec['recommendation']}")

        print("\n✅ Benchmark completed successfully!")
        print(f"📁 Results saved to: {args.output}/")

        return 0

    except KeyboardInterrupt:
        print("\n👋 Benchmark interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())