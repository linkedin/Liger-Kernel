"""
Unified Buffer (UB) Manager for Ascend NPU.

This module provides UB capacity detection and tiling strategy computation
for running Triton kernels on Ascend NPU. It automatically calculates
optimal block sizes based on UB capacity constraints to prevent UB overflow.
"""

import os

from typing import Optional
from typing import Tuple

import torch
import triton

from liger_kernel.utils import is_npu_available

# Default UB capacities for different NPU models (in bits)
_DEFAULT_UB_CAPACITIES = {
    "Ascend910B1": 2097152,  # ~256 KB
    "Ascend910B4": 1572864,  # ~192 KB
    "default": 2097152,  # ~256 KB
}


def _default_strategy(
    ub_capacity_bits: int,
    safety_margin: float,
    dtype_size: int,
    memory_multiplier: float,
    tiling_dims: Optional[Tuple],
    unit_params: Optional[Tuple],
) -> Optional[Tuple]:
    """
    Default tiling strategy: calculate maximum safe block size based on UB capacity.

    This is a unified strategy function that works for all kernels by abstracting
    the memory calculation as: memory_multiplier * BLOCK_SIZE * unit_param * dtype_size * 8 bits

    Args:
        ub_capacity_bits: UB capacity in bits
        safety_margin: Safety margin as a float (e.g., 0.80 for 80%)
        dtype_size: Size of data type in bytes (e.g., 2 for float16, 4 for float32)
        memory_multiplier: Memory multiplier for estimating peak memory usage
        tiling_dims: Dimensions that need tiling as tuple. Used to determine return tuple length.
            - For GEGLU: (n_cols,) -> returns (max_safe_block_size,)
            - For ROPE: (pad_n_q_head, pad_n_kv_head) -> returns (max_safe_block_size, max_safe_block_size)
        unit_params: Parameters related to unit length of each tile as tuple.
            All elements in the tuple will be multiplied together to get the final unit_param.
            - For GEGLU: () (empty tuple, unit_param = 1)
            - For ROPE: (pad_hd,) (unit_param = pad_hd)
            - For kernels with multiple factors: (factor1, factor2, ...) (unit_param = factor1 * factor2 * ...)

    Returns:
        Tuple with same length as tiling_dims, each element is max_safe_block_size (power of 2).

    Note:
        The final block size is computed in compute_default_tiling_strategy by taking
        min(desired_block_size, max_safe_block_size) where desired_block_size = triton.next_power_of_2(tiling_dim).
    """
    if dtype_size is None or dtype_size <= 0:
        dtype_size = 4  # Default to float32
    if memory_multiplier is None or memory_multiplier <= 0:
        memory_multiplier = 10.0  # Default to conservative estimate

    # Extract unit_param from unit_params by multiplying all elements
    # If unit_params is empty or None, unit_param = 1 (e.g., for GEGLU)
    # If unit_params has elements, multiply all of them (e.g., for ROPE: pad_hd)
    # This allows unit_params to contain multiple factors that need to be multiplied
    unit_param = 1.0
    if unit_params is not None and len(unit_params) > 0:
        for param in unit_params:
            param_val = float(param)
            if param_val > 0:
                unit_param *= param_val
        # Ensure unit_param is at least 1.0
        if unit_param <= 0:
            unit_param = 1.0

    # Calculate maximum safe block size based on UB capacity
    # Memory: memory_multiplier * BLOCK_SIZE * unit_param * dtype_size * 8 bits
    SAFE_UB_CAPACITY_BITS = int(ub_capacity_bits * safety_margin)

    # Solve: memory_multiplier * BLOCK_SIZE * unit_param * dtype_size * 8 <= SAFE_UB_CAPACITY_BITS
    # BLOCK_SIZE <= SAFE_UB_CAPACITY_BITS / (memory_multiplier * unit_param * dtype_size * 8)
    max_block_size = int(SAFE_UB_CAPACITY_BITS // (memory_multiplier * unit_param * dtype_size * 8))
    max_block_size = max(1, max_block_size)

    # Find largest power of 2 <= max_block_size
    # Use triton.next_power_of_2(max_block_size + 1) // 2 to get the largest power of 2 <= max_block_size
    safe_block_size = triton.next_power_of_2(max_block_size + 1) // 2

    # Return tuple with same length as tiling_dims
    if tiling_dims is None:
        return (safe_block_size,)
    return (safe_block_size,) * len(tiling_dims)


class UBManager:
    """
    Unified Buffer Manager for Ascend NPU.

    Provides UB capacity detection and management for Ascend NPU devices.
    The UB capacity is used by tiling strategy functions to calculate optimal block sizes.
    """

    def __init__(self, ub_capacity_bits: Optional[int] = None):
        """
        Initialize UB Manager.

        Args:
            ub_capacity_bits: UB capacity in bits. If None, will be detected automatically.
        """
        self._npu_model = self._detect_npu_model()
        self._ub_capacity_bits = ub_capacity_bits or self._detect_ub_capacity()

    @property
    def ub_capacity_bits(self) -> int:
        """Get UB capacity in bits."""
        return self._ub_capacity_bits

    @property
    def ub_capacity_bytes(self) -> int:
        """Get UB capacity in bytes."""
        return self._ub_capacity_bits // 8

    @property
    def npu_model(self) -> str:
        """Get detected NPU model name."""
        return self._npu_model

    def _detect_npu_model(self) -> str:
        """Detect NPU model from device properties."""
        if not is_npu_available():
            return "unknown"

        try:
            dev_props = torch.npu.get_device_properties(0)
            # Try to get model name from device properties
            return dev_props.name
        except Exception:
            pass

        return "default"

    def _detect_ub_capacity(self) -> int:
        """
        Detect UB capacity from environment variable or device properties.

        Returns:
            UB capacity in bits.
        """
        # Check environment variable first
        env_capacity = os.getenv("ASCEND_UB_CAPACITY_BITS")
        if env_capacity is not None:
            try:
                return int(env_capacity)
            except ValueError:
                pass

        # Try to get from device properties
        if is_npu_available():
            try:
                dev_props = torch.npu.get_device_properties(0)
                if hasattr(dev_props, "ub_capacity_bits"):
                    return dev_props.ub_capacity_bits
            except Exception:
                pass

        # Fall back to model-based defaults
        model = self._npu_model
        return _DEFAULT_UB_CAPACITIES.get(model, _DEFAULT_UB_CAPACITIES["default"])


# Global singleton instance
_ub_manager: Optional[UBManager] = None


def get_ub_manager() -> UBManager:
    """Get global UB manager instance."""
    global _ub_manager
    if _ub_manager is None:
        _ub_manager = UBManager()
    return _ub_manager


def compute_default_tiling_strategy(
    safety_margin: float = 0.80,
    dtype_size: Optional[int] = None,
    memory_multiplier: Optional[float] = None,
    tiling_dims: Optional[Tuple] = None,
    unit_params: Optional[Tuple] = None,
) -> Optional[Tuple]:
    """
    Compute tiling strategy using the default strategy function.

    This function directly calls the default strategy and computes the final
    tiling result. All kernels use the same unified strategy function, so
    there's no need for kernel_name-based lookup.

    Args:
        safety_margin: Safety margin as a float (e.g., 0.80 for 80%). Default is 0.80.
        dtype_size: Size of data type in bytes (e.g., 2 for float16, 4 for float32).
            Must be provided. If None or <= 0, defaults to 4 (float32).
        memory_multiplier: Memory multiplier for estimating peak memory usage.
            - For GEGLU: typically 10.0 for backward, 7.0 for forward
            - For ROPE: typically 3.0
            If None, defaults to 10.0 (conservative estimate).
        tiling_dims: Dimensions that need tiling as tuple. Used for calculating desired tiling size.
            - For GEGLU: (n_cols,)
            - For ROPE: (pad_n_q_head, pad_n_kv_head)
        unit_params: Parameters related to unit length of each tile as tuple. Used for calculating UB capacity limits.
            All elements in the tuple will be multiplied together to get the final unit_param.
            - For GEGLU: () (empty tuple, unit_param = 1)
            - For ROPE: (pad_hd,) (unit_param = pad_hd)
            - For kernels with multiple factors: (factor1, factor2, ...) (unit_param = factor1 * factor2 * ...)

    Returns:
        Tiling strategy as a tuple:
            - For GEGLU: (block_size,)
            - For ROPE: (BLOCK_Q, BLOCK_K)
        Returns None if tiling_dims is None.

    Examples:
        >>> # GEGLU forward
        >>> strategy = compute_default_tiling_strategy(safety_margin=0.80, dtype_size=2, memory_multiplier=7.0, tiling_dims=(4096,), unit_params=())
        >>> # ROPE forward
        >>> strategy = compute_default_tiling_strategy(safety_margin=0.90, dtype_size=4, memory_multiplier=3.0, tiling_dims=(32, 32), unit_params=(128,))
    """
    ub_manager = get_ub_manager()

    if dtype_size is None or dtype_size <= 0:
        dtype_size = 4  # Default to float32

    if memory_multiplier is None or memory_multiplier <= 0:
        memory_multiplier = 10.0  # Default conservative estimate

    # Call strategy directly
    max_supported = _default_strategy(
        ub_manager.ub_capacity_bits,
        safety_margin,
        dtype_size,
        memory_multiplier,
        tiling_dims,
        unit_params,
    )

    if max_supported is None or tiling_dims is None:
        return None

    # Calculate final result: min(desired, max_supported)
    # For each element in tiling_dims, compute desired = triton.next_power_of_2(tiling_dims[i])
    # and take min(desired, max_supported[i])
    result = []
    for i, dim_val in enumerate(tiling_dims):
        desired = triton.next_power_of_2(dim_val)
        max_safe = max_supported[i] if i < len(max_supported) else max_supported[0]
        final_val = min(desired, max_safe)
        # Ensure at least 1
        final_val = max(1, final_val)
        result.append(final_val)

    return tuple(result)
