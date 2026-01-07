"""
UB-aware GEGLU implementation for Ascend NPU.

This implementation automatically adjusts block sizes to fit within UB constraints,
preventing UB overflow errors when running on Ascend NPU.

It reuses the original kernels when possible, and only uses tiling when necessary.
"""

import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.utils import is_npu_available

if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
    try:
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh


@triton.jit
def _geglu_tanh_forward_kernel_npu(a, b, c, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    UB-aware GEGLU forward kernel for NPU.

    Uses tiling loop to handle cases where BLOCK_SIZE < n_cols (due to UB constraints).
    When BLOCK_SIZE >= n_cols, the loop executes only once, maintaining original behavior.
    """
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a += program_id * stride
    b += program_id * stride
    c += program_id * stride

    # Process in tiles when BLOCK_SIZE < n_cols
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
        b_row = tl.load(b + col_offsets, mask=mask, other=0)

        # tanh approximation form of GELU is computed with:
        # 0.5 * a * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 * a^3)))
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
        a_cubed = a_row * a_row * a_row
        tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
        tanh_result = tanh(tanh_arg)
        geglu_a = 0.5 * a_row * (1 + tanh_result)
        c_row = geglu_a.cast(b_row.dtype) * b_row

        tl.store(c + col_offsets, c_row, mask=mask)


@triton.jit
def _geglu_tanh_backward_kernel_npu(dc, a, b, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    UB-aware GEGLU backward kernel for NPU.

    Uses tiling loop to handle cases where BLOCK_SIZE < n_cols (due to UB constraints).
    When BLOCK_SIZE >= n_cols, the loop executes only once, maintaining original behavior.
    """
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc += program_id * stride
    a += program_id * stride
    b += program_id * stride

    # Process in tiles when BLOCK_SIZE < n_cols
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
        a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
        b_row = tl.load(b + col_offsets, mask=mask, other=0)

        # recomputation to save memory
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
        a_cubed = a_row * a_row * a_row
        tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
        tanh_result = tanh(tanh_arg)
        geglu_a = 0.5 * a_row * (1 + tanh_result)
        geglu_a = geglu_a.to(dc_row.dtype).to(tl.float32)

        db_row = dc_row.cast(tl.float32) * geglu_a

        # Gradient w.r.t. a can be computed with:
        # b * (0.5 * (1 + tanh(z)) + 0.5 * a * (1 - tanh(z)^2) * (sqrt(2/pi) * (1 + 3 * 0.044715 * a^2)))
        # where z = sqrt(2/pi) * (a + 0.044715 * a^3)
        term1 = 0.5 * (1 + tanh_result)
        tanh_sq = tanh_result * tanh_result
        term2 = 0.5 * a_row * (1 - tanh_sq) * (sqrt_2_over_pi * (1 + 3 * 0.044715 * a_row * a_row))
        da_row = dc_row * b_row * (term1 + term2)

        tl.store(a + col_offsets, da_row, mask=mask)
        tl.store(b + col_offsets, db_row.to(dc_row.dtype), mask=mask)


def geglu_forward(a, b):
    """
    UB-aware GEGLU forward pass for NPU.

    Automatically adjusts block size to fit within UB constraints.
    """
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    # Calculate desired block size
    desired_block_size, num_warps = calculate_settings(n_cols)

    # Compute tiling strategy based on UB capacity
    dtype_size = a.element_size()
    # GEGLU forward tiling strategy:
    # - Calculates maximum safe block size based on UB capacity
    # - Memory analysis:
    #   * Inputs: a, b
    #   * Intermediates: a_cubed, tanh_arg, tanh_result, geglu_a
    #   * Output: c
    #   * Total: ~7x * BLOCK_SIZE * dtype_size
    # - Uses memory_multiplier=7.0 * BLOCK_SIZE * dtype_size * 8 bits for safety
    # - shapes: ((n_cols,),)
    # - tiling_dims: (0,) means first dimension can be tiled
    # - Returns: ((block_size,),)
    shapes = ((n_cols,),)
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.80,
        dtype_size=dtype_size,
        memory_multiplier=7.0,
        shapes=shapes,
        tiling_dims=(0,),
    )

    if tile_shapes is not None and len(tile_shapes) > 0 and len(tile_shapes[0]) > 0:
        # Strategy returns ((block_size,),)
        adjusted_block_size = tile_shapes[0][0]
    else:
        # Fallback to desired block size if no best practice found (no tiling needed)
        adjusted_block_size = desired_block_size
    # Always use the unified NPU kernel
    # When adjusted_block_size >= n_cols, the loop executes only once (no tiling)
    # When adjusted_block_size < n_cols, the loop handles tiling automatically
    _geglu_tanh_forward_kernel_npu[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=adjusted_block_size,
        num_warps=num_warps,
    )
    return a, b, c.view(*ori_shape)


def geglu_backward(a, b, dc):
    """
    UB-aware GEGLU backward pass for NPU.

    Automatically adjusts block size to fit within UB constraints.
    """
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    # Calculate desired block size
    desired_block_size, num_warps = calculate_settings(n_cols)

    # Compute tiling strategy based on UB capacity
    dtype_size = dc.element_size()
    # GEGLU backward tiling strategy:
    # - Calculates maximum safe block size based on UB capacity
    # - Memory analysis:
    #   * More intermediates for gradient computation compared to forward
    #   * Total: ~10x * BLOCK_SIZE * dtype_size
    # - Uses memory_multiplier=10.0 * BLOCK_SIZE * dtype_size * 8 bits for safety
    # - shapes: ((n_cols,),)
    # - tiling_dims: (0,) means first dimension can be tiled
    # - Returns: ((block_size,),)
    shapes = ((n_cols,),)
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.80,
        dtype_size=dtype_size,
        memory_multiplier=10.0,
        shapes=shapes,
        tiling_dims=(0,),
    )

    if tile_shapes is not None and len(tile_shapes) > 0 and len(tile_shapes[0]) > 0:
        # Strategy returns ((block_size,),)
        adjusted_block_size = tile_shapes[0][0]
    else:
        # Fallback to desired block size if no best practice found (no tiling needed)
        adjusted_block_size = desired_block_size

    # Always use the unified NPU kernel
    # When adjusted_block_size >= n_cols, the loop executes only once (no tiling)
    # When adjusted_block_size < n_cols, the loop handles tiling automatically
    _geglu_tanh_backward_kernel_npu[(n_rows,)](
        dc,
        a,
        b,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=adjusted_block_size,
        num_warps=num_warps,
    )

    return a.view(*ori_shape), b.view(*ori_shape)


class LigerGELUMulFunction(torch.autograd.Function):
    """UB-aware GEGLU function for Ascend NPU."""

    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        a, b, c = geglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        a, b = geglu_backward(a, b, dc)
        return a, b
