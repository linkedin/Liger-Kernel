"""
Unified Buffer (UB) Manager for Ascend NPU.

This module provides UB capacity detection and tiling strategy lookup
based on best practices for running Triton kernels on Ascend NPU.
"""

import os

from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import triton

from liger_kernel.utils import is_npu_available

# Default UB capacities for different NPU models (in bits)
_DEFAULT_UB_CAPACITIES = {
    "Ascend910B1": 2097152,  # ~256 KB
    "Ascend910B4": 1572864,  # ~192 KB
    "default": 2097152,  # ~256 KB
}


def _geglu_default_strategy(key_params: Optional[Union[Tuple, Dict]]) -> Optional[Tuple]:
    """
    GEGLU default tiling strategy based on UB capacity and n_cols.

    Strategy:
    - Calculates maximum safe block size based on UB capacity and n_cols
    - Memory analysis for GEGLU:
      * Forward: ~7x * BLOCK_SIZE * dtype_size (inputs: a, b; intermediates: a_cubed, tanh_arg, tanh_result, geglu_a; output: c)
      * Backward: ~10x * BLOCK_SIZE * dtype_size (more intermediates for gradient computation)
    - Uses conservative estimate: 10x * BLOCK_SIZE * dtype_size for safety
    - Returns min(desired_block_size, safe_block_size) where desired_block_size = triton.next_power_of_2(n_cols)

    Args:
        key_params: (n_cols, dtype_size) or {"n_cols": int, "dtype_size": int}

    This strategy applies to both forward and backward passes.
    """
    if key_params is None:
        return None  # Use default desired_block_size

    # Get UB capacity from ub_manager
    ub_manager = get_ub_manager()
    ub_capacity_bits = ub_manager.ub_capacity_bits

    # Extract parameters
    if isinstance(key_params, dict):
        n_cols = key_params["n_cols"]
        dtype_size = key_params.get("dtype_size", 4)
    else:
        n_cols = key_params[0]
        dtype_size = key_params[1] if len(key_params) > 1 else 4

    # Calculate desired block size (power of 2)
    desired_block_size = triton.next_power_of_2(n_cols)

    # Calculate maximum safe block size based on UB capacity
    # Memory: 10x * BLOCK_SIZE * dtype_size * 8 bits
    SAFE_UB_CAPACITY_BITS = int(ub_capacity_bits * 0.80)  # 80% safety margin

    # Solve: 10 * BLOCK_SIZE * dtype_size * 8 <= SAFE_UB_CAPACITY_BITS
    # BLOCK_SIZE <= SAFE_UB_CAPACITY_BITS / (10 * dtype_size * 8)
    max_block_size = SAFE_UB_CAPACITY_BITS // (10 * dtype_size * 8)
    max_block_size = max(1, max_block_size)

    # Find largest power of 2 <= max_block_size
    # Use triton.next_power_of_2(max_block_size + 1) // 2 to get the largest power of 2 <= max_block_size
    safe_block_size = triton.next_power_of_2(max_block_size + 1) // 2

    # Use min(desired_block_size, safe_block_size)
    # This ensures we use the desired size when it fits in UB, otherwise use safe size
    block_size = min(desired_block_size, safe_block_size)

    return (block_size,)


def _rope_default_strategy(key_params: Optional[Union[Tuple, Dict]]) -> Optional[Tuple]:
    """
    ROPE default tiling strategy based on UB capacity, pad_n_q_head, pad_n_kv_head, and pad_hd.

    Strategy (based on optimized ROPE kernel):
    - cos_vals and sin_vals are loaded once outside loops (shared): pad_hd // 2 elements each
    - In q heads loop (peak memory):
      * q_left: BLOCK_Q * (pad_hd // 2) elements
      * q_right: BLOCK_Q * (pad_hd // 2) elements
      * new_left: BLOCK_Q * (pad_hd // 2) elements (intermediate result)
      * new_right: BLOCK_Q * (pad_hd // 2) elements (intermediate result)
      * Total: 4 * BLOCK_Q * (pad_hd // 2) = 2 * BLOCK_Q * pad_hd elements
    - In k heads loop (peak memory):
      * k_left: BLOCK_K * (pad_hd // 2) elements
      * k_right: BLOCK_K * (pad_hd // 2) elements
      * new_left: BLOCK_K * (pad_hd // 2) elements (intermediate result)
      * new_right: BLOCK_K * (pad_hd // 2) elements (intermediate result)
      * Total: 4 * BLOCK_K * (pad_hd // 2) = 2 * BLOCK_K * pad_hd elements
    - Since q and k are processed separately, peak memory is max(BLOCK_Q, BLOCK_K) case
    - Plus shared cos/sin: 2 * (pad_hd // 2) = pad_hd elements
    - Conservative estimate: (2 * BLOCK_SIZE * pad_hd + pad_hd) * dtype_size * 8 bits
    - Simplified: (2 * BLOCK_SIZE + 1) * pad_hd * dtype_size * 8 bits
    - For safety, use: 3 * BLOCK_SIZE * pad_hd * dtype_size * 8 bits

    Args:
        key_params: (pad_n_q_head, pad_n_kv_head, pad_hd, dtype_size) or dict

    Returns: (BLOCK_Q, BLOCK_K)
    """
    if key_params is None:
        return None

    # Get UB capacity from ub_manager
    ub_manager = get_ub_manager()
    ub_capacity_bits = ub_manager.ub_capacity_bits

    # Extract parameters
    if isinstance(key_params, dict):
        pad_n_q_head = key_params["pad_n_q_head"]
        pad_n_kv_head = key_params["pad_n_kv_head"]
        pad_hd = key_params["pad_hd"]
        dtype_size = key_params.get("dtype_size", 4)
    else:
        pad_n_q_head = key_params[0]
        pad_n_kv_head = key_params[1]
        pad_hd = key_params[2]
        dtype_size = key_params[3] if len(key_params) > 3 else 4

    # Calculate maximum safe block size
    # Memory per tile: 3 * BLOCK_SIZE * pad_hd * dtype_size * 8 bits (conservative estimate)
    SAFE_UB_CAPACITY_BITS = int(ub_capacity_bits * 0.80)  # 80% safety margin

    # Solve: 3 * BLOCK_SIZE * pad_hd * dtype_size * 8 <= SAFE_UB_CAPACITY_BITS
    # BLOCK_SIZE <= SAFE_UB_CAPACITY_BITS / (3 * pad_hd * dtype_size * 8)
    max_block_size = SAFE_UB_CAPACITY_BITS // (3 * pad_hd * dtype_size * 8)
    max_block_size = max(1, max_block_size)

    # Find largest power of 2 <= max_block_size
    # Use triton.next_power_of_2(max_block_size + 1) // 2 to get the largest power of 2 <= max_block_size
    safe_block_size = triton.next_power_of_2(max_block_size + 1) // 2

    # Calculate BLOCK_Q and BLOCK_K
    # Use min(desired, safe), but ensure it's a power of 2
    BLOCK_Q = min(triton.next_power_of_2(pad_n_q_head), safe_block_size)
    BLOCK_K = min(triton.next_power_of_2(pad_n_kv_head), safe_block_size)

    # Ensure at least 1
    BLOCK_Q = max(1, BLOCK_Q)
    BLOCK_K = max(1, BLOCK_K)

    return (BLOCK_Q, BLOCK_K)


_TILING_STRATEGY_BEST_PRACTICES: Dict[Tuple, Union[Tuple, Callable]] = {
    # GEGLU strategies: default strategy based on UB capacity and n_cols
    # Both forward and backward use the same strategy
    ("geglu_forward", 1572864): _geglu_default_strategy,
    ("geglu_backward", 1572864): _geglu_default_strategy,
    # ROPE strategies: block-based (BLOCK_Q, BLOCK_K)
    # Both forward and backward use the same strategy
    ("rope_forward", 1572864): _rope_default_strategy,
    ("rope_backward", 1572864): _rope_default_strategy,
}


class UBManager:
    """
    Unified Buffer Manager for Ascend NPU.

    Provides UB capacity detection and tiling strategy lookup from best practices.
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

    def get_tiling_strategy(
        self,
        kernel_name: str,
        key_params: Optional[Union[Tuple, Dict]] = None,
    ) -> Optional[Tuple]:
        """
        Get tiling strategy for a specific kernel from best practices.

        Args:
            kernel_name: Name of the kernel (e.g., "geglu_forward", "rope_forward")
            key_params: Optional key parameters for parameter-specific strategies. Can be:
                - For GEGLU: (n_cols, dtype_size) or {"n_cols": int, "dtype_size": int}
                - For ROPE: (n_q_head, n_kv_head, head_dim, dtype_size) or dict
                - If None (default), uses general strategy without parameters

        Returns:
            Tiling strategy as a tuple:
                - For GEGLU: (block_size,)
                - For ROPE: (pad_n_q_head, pad_n_kv_head, pad_hd)
            Returns None if not found in best practices.

        Examples:
            >>> ub_manager = get_ub_manager()
            >>> # GEGLU forward (general strategy)
            >>> strategy = ub_manager.get_tiling_strategy("geglu_forward")
            >>> # GEGLU forward (parameter-specific strategy)
            >>> strategy = ub_manager.get_tiling_strategy("geglu_forward", (4096, 2))
            >>> # ROPE forward (general strategy)
            >>> strategy = ub_manager.get_tiling_strategy("rope_forward")
        """
        # Look up strategy in best practices dictionary
        lookup_key = (kernel_name, self._ub_capacity_bits)
        strategy = _TILING_STRATEGY_BEST_PRACTICES.get(lookup_key)

        if strategy is None:
            return None

        # If strategy is a callable function, call it with key_params and ub_capacity_bits
        if callable(strategy):
            # Normalize key_params for function call
            normalized_params = None
            if key_params is not None:
                if isinstance(key_params, dict):
                    # Convert dict to tuple based on kernel type
                    if kernel_name.startswith("geglu"):
                        normalized_params = (key_params["n_cols"], key_params["dtype_size"])
                    elif kernel_name.startswith("rope"):
                        # ROPE strategy expects (pad_n_q_head, pad_n_kv_head, pad_hd, dtype_size)
                        if "pad_n_q_head" in key_params:
                            normalized_params = (
                                key_params["pad_n_q_head"],
                                key_params["pad_n_kv_head"],
                                key_params["pad_hd"],
                                key_params.get("dtype_size", 4),
                            )
                        else:
                            # Fallback: calculate pad values from n_heads and head_dim
                            import triton

                            normalized_params = (
                                triton.next_power_of_2(key_params.get("n_q_head", key_params.get("pad_n_q_head", 1))),
                                triton.next_power_of_2(key_params.get("n_kv_head", key_params.get("pad_n_kv_head", 1))),
                                triton.next_power_of_2(key_params.get("head_dim", key_params.get("pad_hd", 64))),
                                key_params.get("dtype_size", 4),
                            )
                    else:
                        # Generic: convert dict values to tuple (sorted for consistency)
                        normalized_params = tuple(sorted(key_params.values()))
                else:
                    normalized_params = key_params

            # Call strategy function with key_params
            # Strategy functions can access ub_manager via get_ub_manager() if needed
            return strategy(normalized_params)

        # If strategy is a fixed tuple, return it directly
        return strategy


# Global singleton instance
_ub_manager: Optional[UBManager] = None


def get_ub_manager() -> UBManager:
    """Get global UB manager instance."""
    global _ub_manager
    if _ub_manager is None:
        _ub_manager = UBManager()
    return _ub_manager


def get_tiling_strategy(
    kernel_name: str,
    key_params: Optional[Union[Tuple, Dict]] = None,
) -> Optional[Tuple]:
    """
    Get tiling strategy for a specific kernel from best practices.

    This is a convenience function that wraps UBManager.get_tiling_strategy().

    Args:
        kernel_name: Name of the kernel (e.g., "geglu_forward", "rope_forward")
        key_params: Optional key parameters for parameter-specific strategies. Can be:
            - For GEGLU: (n_cols, dtype_size) or {"n_cols": int, "dtype_size": int}
            - For ROPE: (n_q_head, n_kv_head, head_dim, dtype_size) or dict
            - If None (default), uses general strategy without parameters

    Returns:
        Tiling strategy as a tuple:
            - For GEGLU: (block_size,)
            - For ROPE: (pad_n_q_head, pad_n_kv_head, pad_hd)
        Returns None if not found in best practices.

    Examples:
        >>> # GEGLU forward (general strategy)
        >>> strategy = get_tiling_strategy("geglu_forward")
        >>> # GEGLU forward (parameter-specific strategy)
        >>> strategy = get_tiling_strategy("geglu_forward", (4096, 2))
        >>> # ROPE forward (general strategy)
        >>> strategy = get_tiling_strategy("rope_forward")
    """
    ub_manager = get_ub_manager()
    return ub_manager.get_tiling_strategy(kernel_name, key_params)
