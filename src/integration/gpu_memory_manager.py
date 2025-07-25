"""
CUDA-optimized memory management system for RealSense detection pipeline.
Provides efficient GPU memory allocation, pooling, and sharing between components.
"""

import torch
import threading
import time
import gc
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..utils.logger import get_logger


class MemoryPoolType(Enum):
    """Types of memory pools for different components."""
    FRAME_BUFFERS = "frame_buffers"
    DETECTION_TENSORS = "detection_tensors"
    DEPTH_PROCESSING = "depth_processing"
    VISUALIZATION = "visualization"
    TEMPORARY = "temporary"


@dataclass
class MemoryBlock:
    """Represents a memory block in the pool."""
    tensor: torch.Tensor
    size_bytes: int
    allocated_time: float
    last_used: float
    in_use: bool = False
    pool_type: MemoryPoolType = MemoryPoolType.TEMPORARY
    ref_count: int = 0


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_allocated: int = 0
    total_free: int = 0
    total_reserved: int = 0
    pool_usage: Dict[str, int] = None
    fragmentation_ratio: float = 0.0
    efficiency_ratio: float = 0.0

    def __post_init__(self):
        if self.pool_usage is None:
            self.pool_usage = {}


class CUDAMemoryPool:
    """CUDA memory pool for efficient tensor allocation and reuse."""

    def __init__(self, pool_type: MemoryPoolType, max_size_gb: float = 2.0,
                 cleanup_threshold: float = 0.8):
        """
        Initialize CUDA memory pool.

        Args:
            pool_type: Type of memory pool
            max_size_gb: Maximum pool size in GB
            cleanup_threshold: Cleanup trigger threshold (0.0-1.0)
        """
        self.pool_type = pool_type
        self.max_size_bytes = int(max_size_gb * 1024 ** 3)
        self.cleanup_threshold = cleanup_threshold

        # Memory blocks storage
        self.available_blocks: Dict[Tuple[int, ...], deque] = defaultdict(deque)
        self.allocated_blocks: Dict[int, MemoryBlock] = {}
        self.size_to_blocks: Dict[int, List[int]] = defaultdict(list)

        # Statistics
        self.total_allocated = 0
        self.peak_allocated = 0
        self.allocation_count = 0
        self.hit_count = 0
        self.miss_count = 0

        # Threading
        self.lock = threading.RLock()

        self.logger = get_logger(f"CUDAMemoryPool_{pool_type.value}")
        self.logger.info(f"Initialized {pool_type.value} pool with {max_size_gb:.1f}GB limit")

    def allocate(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                 device: Optional[torch.device] = None) -> Tuple[torch.Tensor, int]:
        """
        Allocate tensor from pool or create new one.

        Args:
            shape: Tensor shape
            dtype: Data type
            device: CUDA device

        Returns:
            Tuple of (tensor, block_id)
        """
        with self.lock:
            # Calculate required size
            element_size = torch.tensor([], dtype=dtype).element_size()
            total_elements = np.prod(shape)
            required_bytes = total_elements * element_size

            # Try to find existing block
            shape_key = tuple(shape)
            if shape_key in self.available_blocks and self.available_blocks[shape_key]:
                # Reuse existing block
                block_queue = self.available_blocks[shape_key]

                for _ in range(len(block_queue)):
                    block_id = block_queue.popleft()
                    if block_id in self.allocated_blocks:
                        block = self.allocated_blocks[block_id]
                        if not block.in_use and block.tensor.dtype == dtype:
                            # Found suitable block
                            block.in_use = True
                            block.last_used = time.time()
                            block.ref_count += 1
                            self.hit_count += 1

                            # Zero out the tensor for clean state
                            block.tensor.zero_()

                            return block.tensor, block_id

            # No suitable block found, create new one
            self.miss_count += 1
            return self._create_new_block(shape, dtype, device, required_bytes)

    def _create_new_block(self, shape: Tuple[int, ...], dtype: torch.dtype,
                          device: Optional[torch.device], required_bytes: int) -> Tuple[torch.Tensor, int]:
        """Create new memory block."""
        # Check if we need cleanup
        if self.total_allocated + required_bytes > self.max_size_bytes * self.cleanup_threshold:
            self._cleanup_unused_blocks()

        # Check hard limit
        if self.total_allocated + required_bytes > self.max_size_bytes:
            self.logger.warning(f"Memory pool {self.pool_type.value} approaching limit")
            # Force cleanup
            self._force_cleanup()

        # Create new tensor
        if device is None:
            device = torch.cuda.current_device()

        try:
            tensor = torch.empty(shape, dtype=dtype, device=device)
        except torch.cuda.OutOfMemoryError:
            # Emergency cleanup and retry
            self.logger.warning("CUDA OOM in pool allocation, emergency cleanup")
            self._emergency_cleanup()
            tensor = torch.empty(shape, dtype=dtype, device=device)

        # Create block
        block_id = id(tensor)
        block = MemoryBlock(
            tensor=tensor,
            size_bytes=required_bytes,
            allocated_time=time.time(),
            last_used=time.time(),
            in_use=True,
            pool_type=self.pool_type,
            ref_count=1
        )

        # Store block
        self.allocated_blocks[block_id] = block
        self.size_to_blocks[required_bytes].append(block_id)

        # Update statistics
        self.total_allocated += required_bytes
        self.peak_allocated = max(self.peak_allocated, self.total_allocated)
        self.allocation_count += 1

        self.logger.debug(f"Created new block: {shape} {dtype} ({required_bytes} bytes)")

        return tensor, block_id

    def release(self, block_id: int) -> bool:
        """
        Release memory block back to pool.

        Args:
            block_id: Block identifier

        Returns:
            True if released successfully
        """
        with self.lock:
            if block_id not in self.allocated_blocks:
                return False

            block = self.allocated_blocks[block_id]
            block.ref_count = max(0, block.ref_count - 1)

            if block.ref_count == 0:
                block.in_use = False
                block.last_used = time.time()

                # Add back to available blocks
                shape_key = tuple(block.tensor.shape)
                self.available_blocks[shape_key].append(block_id)

                self.logger.debug(f"Released block {block_id} to pool")

            return True

    def _cleanup_unused_blocks(self):
        """Clean up unused blocks based on age."""
        current_time = time.time()
        cleanup_age = 30.0  # 30 seconds

        blocks_to_remove = []

        for block_id, block in self.allocated_blocks.items():
            if (not block.in_use and
                    current_time - block.last_used > cleanup_age):
                blocks_to_remove.append(block_id)

        for block_id in blocks_to_remove:
            self._remove_block(block_id)

        if blocks_to_remove:
            self.logger.info(f"Cleaned up {len(blocks_to_remove)} unused blocks")

    def _force_cleanup(self):
        """Force cleanup of all unused blocks."""
        blocks_to_remove = []

        for block_id, block in self.allocated_blocks.items():
            if not block.in_use:
                blocks_to_remove.append(block_id)

        for block_id in blocks_to_remove:
            self._remove_block(block_id)

        if blocks_to_remove:
            self.logger.warning(f"Force cleaned {len(blocks_to_remove)} blocks")

    def _emergency_cleanup(self):
        """Emergency cleanup including garbage collection."""
        self._force_cleanup()

        # Force Python garbage collection
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.warning("Emergency cleanup completed")

    def _remove_block(self, block_id: int):
        """Remove block from pool completely."""
        if block_id not in self.allocated_blocks:
            return

        block = self.allocated_blocks[block_id]

        # Remove from size mapping
        if block.size_bytes in self.size_to_blocks:
            if block_id in self.size_to_blocks[block.size_bytes]:
                self.size_to_blocks[block.size_bytes].remove(block_id)

        # Remove from available blocks
        shape_key = tuple(block.tensor.shape)
        if shape_key in self.available_blocks:
            available_queue = self.available_blocks[shape_key]
            if block_id in available_queue:
                available_queue.remove(block_id)

        # Update statistics
        self.total_allocated -= block.size_bytes

        # Remove tensor reference
        del self.allocated_blocks[block_id]
        del block.tensor
        del block

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            hit_rate = self.hit_count / (self.hit_count + self.miss_count) if (
                                                                                          self.hit_count + self.miss_count) > 0 else 0

            return {
                'pool_type': self.pool_type.value,
                'total_allocated_bytes': self.total_allocated,
                'total_allocated_mb': self.total_allocated / 1024 ** 2,
                'peak_allocated_bytes': self.peak_allocated,
                'peak_allocated_mb': self.peak_allocated / 1024 ** 2,
                'max_size_bytes': self.max_size_bytes,
                'max_size_mb': self.max_size_bytes / 1024 ** 2,
                'utilization_ratio': self.total_allocated / self.max_size_bytes,
                'active_blocks': len([b for b in self.allocated_blocks.values() if b.in_use]),
                'available_blocks': len([b for b in self.allocated_blocks.values() if not b.in_use]),
                'total_blocks': len(self.allocated_blocks),
                'allocation_count': self.allocation_count,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate
            }

    def clear(self):
        """Clear all blocks from pool."""
        with self.lock:
            self.allocated_blocks.clear()
            self.available_blocks.clear()
            self.size_to_blocks.clear()
            self.total_allocated = 0
            self.logger.info(f"Cleared {self.pool_type.value} pool")


class CUDAMemoryManager:
    """Centralized CUDA memory management for all pipeline components."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CUDA memory manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger("CUDAMemoryManager")

        # GPU configuration
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.current_device = 0
        self.devices = []

        # Memory pools for different components
        self.memory_pools: Dict[MemoryPoolType, CUDAMemoryPool] = {}

        # Global settings
        gpu_config = config.get('gpu', {})
        self.total_memory_limit_gb = gpu_config.get('memory_limit_gb', 1.5)
        self.cleanup_interval = gpu_config.get('cleanup_interval', 30.0)
        self.emergency_threshold = gpu_config.get('emergency_threshold', 0.8)

        # Multi-GPU support
        self.use_multi_gpu = gpu_config.get('use_multi_gpu', True) and self.device_count > 1
        self.load_balancing = gpu_config.get('load_balancing', 'memory')  # 'memory' or 'round_robin'

        # Initialize system
        self._initialize_devices()
        self._initialize_memory_pools()
        self._start_cleanup_thread()

        self.logger.info(f"Initialized CUDA memory manager with {self.device_count} devices")
        if self.use_multi_gpu:
            self.logger.info(f"Multi-GPU mode enabled with {self.device_count} devices")

    def _initialize_devices(self):
        """Initialize CUDA devices."""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available")
            return

        for i in range(self.device_count):
            device = torch.device(f'cuda:{i}')
            props = torch.cuda.get_device_properties(device)

            device_info = {
                'device': device,
                'name': props.name,
                'total_memory': props.total_memory,
                'major': props.major,
                'minor': props.minor,
                'multi_processor_count': props.multi_processor_count
            }

            self.devices.append(device_info)
            self.logger.info(f"Device {i}: {props.name} ({props.total_memory / 1024 ** 3:.1f} GB)")

    def _initialize_memory_pools(self):
        """Initialize memory pools for different components."""
        if not self.devices:
            return

        # Calculate pool sizes based on total memory limit
        pool_allocations = {
            MemoryPoolType.FRAME_BUFFERS: 0.3,  # 30% for camera frames
            MemoryPoolType.DETECTION_TENSORS: 0.4,  # 40% for detection models
            MemoryPoolType.DEPTH_PROCESSING: 0.15,  # 15% for depth processing
            MemoryPoolType.VISUALIZATION: 0.1,  # 10% for visualization
            MemoryPoolType.TEMPORARY: 0.05  # 5% for temporary operations
        }

        for pool_type, ratio in pool_allocations.items():
            pool_size = self.total_memory_limit_gb * ratio
            self.memory_pools[pool_type] = CUDAMemoryPool(
                pool_type=pool_type,
                max_size_gb=pool_size,
                cleanup_threshold=0.8
            )

    def _start_cleanup_thread(self):
        """Start background cleanup thread."""

        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self._periodic_cleanup()
                except Exception as e:
                    self.logger.error(f"Cleanup thread error: {e}")

        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        self.logger.info("Started background cleanup thread")

    def _periodic_cleanup(self):
        """Periodic cleanup of memory pools."""
        total_usage = self.get_total_memory_usage()

        if total_usage > self.emergency_threshold:
            self.logger.warning(f"High memory usage detected: {total_usage:.2%}")
            self._emergency_cleanup()
        else:
            # Normal cleanup
            for pool in self.memory_pools.values():
                pool._cleanup_unused_blocks()

    def _emergency_cleanup(self):
        """Emergency cleanup when memory usage is too high."""
        self.logger.warning("Performing emergency memory cleanup")

        # Force cleanup all pools
        for pool in self.memory_pools.values():
            pool._emergency_cleanup()

        # Global CUDA cache clear
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                        pool_type: MemoryPoolType = MemoryPoolType.TEMPORARY,
                        device: Optional[torch.device] = None) -> Tuple[torch.Tensor, int]:
        """
        Allocate tensor from appropriate memory pool.

        Args:
            shape: Tensor shape
            dtype: Data type
            pool_type: Memory pool type
            device: Target device (auto-selected if None)

        Returns:
            Tuple of (tensor, block_id)
        """
        if device is None:
            device = self._select_device()

        if pool_type not in self.memory_pools:
            raise ValueError(f"Unknown pool type: {pool_type}")

        pool = self.memory_pools[pool_type]
        return pool.allocate(shape, dtype, device)

    def release_tensor(self, block_id: int, pool_type: MemoryPoolType) -> bool:
        """
        Release tensor back to memory pool.

        Args:
            block_id: Block identifier
            pool_type: Memory pool type

        Returns:
            True if released successfully
        """
        if pool_type not in self.memory_pools:
            return False

        pool = self.memory_pools[pool_type]
        return pool.release(block_id)

    def _select_device(self) -> torch.device:
        """Select optimal device based on load balancing strategy."""
        if not self.devices:
            return torch.device('cpu')

        if not self.use_multi_gpu or len(self.devices) == 1:
            return self.devices[0]['device']

        if self.load_balancing == 'memory':
            # Select device with most free memory
            best_device = None
            max_free_memory = 0

            for device_info in self.devices:
                device = device_info['device']
                torch.cuda.set_device(device)

                allocated = torch.cuda.memory_allocated(device)
                total = device_info['total_memory']
                free = total - allocated

                if free > max_free_memory:
                    max_free_memory = free
                    best_device = device

            return best_device or self.devices[0]['device']

        elif self.load_balancing == 'round_robin':
            # Simple round-robin selection
            device = self.devices[self.current_device]['device']
            self.current_device = (self.current_device + 1) % len(self.devices)
            return device

        return self.devices[0]['device']

    def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory statistics."""
        total_allocated = 0
        total_free = 0
        total_reserved = 0
        pool_usage = {}

        # Pool statistics
        for pool_type, pool in self.memory_pools.items():
            stats = pool.get_stats()
            pool_usage[pool_type.value] = stats['total_allocated_mb']
            total_allocated += stats['total_allocated_bytes']

        # GPU memory statistics
        if torch.cuda.is_available():
            for device_info in self.devices:
                device = device_info['device']
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                total_memory = device_info['total_memory']

                total_reserved += reserved
                total_free += total_memory - allocated

        # Calculate ratios
        total_memory = sum(d['total_memory'] for d in self.devices) if self.devices else 1
        fragmentation_ratio = (total_reserved - total_allocated) / total_memory if total_memory > 0 else 0
        efficiency_ratio = total_allocated / total_reserved if total_reserved > 0 else 0

        return MemoryStats(
            total_allocated=total_allocated,
            total_free=total_free,
            total_reserved=total_reserved,
            pool_usage=pool_usage,
            fragmentation_ratio=fragmentation_ratio,
            efficiency_ratio=efficiency_ratio
        )

    def get_total_memory_usage(self) -> float:
        """Get total memory usage ratio across all devices."""
        if not self.devices:
            return 0.0

        total_used = 0
        total_available = 0

        for device_info in self.devices:
            device = device_info['device']
            allocated = torch.cuda.memory_allocated(device)
            total_memory = device_info['total_memory']

            total_used += allocated
            total_available += total_memory

        return total_used / total_available if total_available > 0 else 0.0

    def optimize_memory_layout(self):
        """Optimize memory layout by defragmenting pools."""
        self.logger.info("Optimizing memory layout")

        # Force cleanup and defragmentation
        for pool in self.memory_pools.values():
            pool._force_cleanup()

        # Clear CUDA cache to defragment
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("Memory layout optimization completed")

    def get_device_info(self) -> List[Dict[str, Any]]:
        """Get information about all available devices."""
        return [
            {
                'index': i,
                'name': info['name'],
                'total_memory_gb': info['total_memory'] / 1024 ** 3,
                'compute_capability': f"{info['major']}.{info['minor']}",
                'multiprocessor_count': info['multi_processor_count'],
                'allocated_memory_gb': torch.cuda.memory_allocated(
                    info['device']) / 1024 ** 3 if torch.cuda.is_available() else 0,
                'reserved_memory_gb': torch.cuda.memory_reserved(
                    info['device']) / 1024 ** 3 if torch.cuda.is_available() else 0
            }
            for i, info in enumerate(self.devices)
        ]

    def cleanup(self):
        """Clean up all memory pools and resources."""
        self.logger.info("Cleaning up CUDA memory manager")

        for pool in self.memory_pools.values():
            pool.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("CUDA memory manager cleanup completed")


# Context manager for automatic memory management
class ManagedCUDATensor:
    """Context manager for automatic CUDA tensor memory management."""

    def __init__(self, memory_manager: CUDAMemoryManager, shape: Tuple[int, ...],
                 dtype: torch.dtype = torch.float32,
                 pool_type: MemoryPoolType = MemoryPoolType.TEMPORARY):
        self.memory_manager = memory_manager
        self.shape = shape
        self.dtype = dtype
        self.pool_type = pool_type
        self.tensor = None
        self.block_id = None

    def __enter__(self) -> torch.Tensor:
        self.tensor, self.block_id = self.memory_manager.allocate_tensor(
            self.shape, self.dtype, self.pool_type
        )
        return self.tensor

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.block_id is not None:
            self.memory_manager.release_tensor(self.block_id, self.pool_type)


# Utility functions for common memory operations
def create_frame_buffer(memory_manager: CUDAMemoryManager, height: int, width: int,
                        channels: int = 3, dtype: torch.dtype = torch.uint8) -> Tuple[torch.Tensor, int]:
    """Create frame buffer tensor."""
    return memory_manager.allocate_tensor(
        (height, width, channels), dtype, MemoryPoolType.FRAME_BUFFERS
    )


def create_detection_tensor(memory_manager: CUDAMemoryManager, batch_size: int,
                            height: int, width: int, channels: int = 3) -> Tuple[torch.Tensor, int]:
    """Create detection input tensor."""
    return memory_manager.allocate_tensor(
        (batch_size, channels, height, width), torch.float32, MemoryPoolType.DETECTION_TENSORS
    )


def create_depth_buffer(memory_manager: CUDAMemoryManager, height: int, width: int) -> Tuple[torch.Tensor, int]:
    """Create depth processing buffer."""
    return memory_manager.allocate_tensor(
        (height, width), torch.float32, MemoryPoolType.DEPTH_PROCESSING
    )