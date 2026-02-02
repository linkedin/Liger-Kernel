import torch
import triton
import triton.language as tl

from triton.language.math import tanh

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count


@triton.jit
def _geglu_forward_kernel_flat(a_ptr, b_ptr, c_ptr, total_elements, BLOCK_SIZE: tl.constexpr, NUM_STAGES: tl.constexpr):
    """
    High-performance GEGLU forward kernel using flatten 1D approach.

    Uses grid-stride loop pattern for optimal performance on NPU.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Grid-Stride Loop
    start_idx = pid * BLOCK_SIZE
    stride = num_progs * BLOCK_SIZE

    # Constants for GELU tanh approximation
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    gelu_coeff = 0.044715

    for idx in tl.range(start_idx, total_elements, stride, num_stages=NUM_STAGES):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        a_val = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b_val = tl.load(b_ptr + offsets, mask=mask, other=0.0)

        # tanh approximation form of GELU is computed with:
        # 0.5 * a * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 * a^3)))
        a_cubed = a_val * a_val * a_val
        tanh_arg = sqrt_2_over_pi * (a_val + gelu_coeff * a_cubed)
        tanh_result = tanh(tanh_arg)
        geglu_a = 0.5 * a_val * (1.0 + tanh_result)
        c_row = geglu_a.cast(b_val.dtype) * b_val
        tl.store(c_ptr + offsets, c_row, mask=mask)


@triton.jit
def _geglu_backward_kernel_flat(
    dc_ptr, a_ptr, b_ptr, da_ptr, db_ptr, total_elements, BLOCK_SIZE: tl.constexpr, NUM_STAGES: tl.constexpr
):
    """
    High-performance GEGLU backward kernel using flatten 1D approach.

    Uses grid-stride loop pattern for optimal performance on NPU.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    start_idx = pid * BLOCK_SIZE
    stride = num_progs * BLOCK_SIZE

    # Constants for GELU tanh approximation
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    gelu_coeff = 0.044715

    for idx in tl.range(start_idx, total_elements, stride, num_stages=NUM_STAGES):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        dc = tl.load(dc_ptr + offsets, mask=mask, other=0.0)
        a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

        # recomputation to save memory
        a_cubed = a * a * a
        tanh_arg = sqrt_2_over_pi * (a + gelu_coeff * a_cubed)
        tanh_result = tanh(tanh_arg)
        geglu_a = 0.5 * a * (1 + tanh_result)
        geglu_a = geglu_a.to(dc.dtype).to(tl.float32)

        db = dc.cast(tl.float32) * geglu_a

        # Gradient w.r.t. a can be computed with:
        # b * (0.5 * (1 + tanh(z)) + 0.5 * a * (1 - tanh(z)^2) * (sqrt(2/pi) * (1 + 3 * 0.044715 * a^2)))
        # where z = sqrt(2/pi) * (a + 0.044715 * a^3)
        term1 = 0.5 * (1.0 + tanh_result)
        tanh_sq = tanh_result * tanh_result
        a_sq = a * a
        term2 = 0.5 * a * (1.0 - tanh_sq) * (sqrt_2_over_pi * (1.0 + 3.0 * gelu_coeff * a_sq))
        da = dc * b * (term1 + term2)

        tl.store(da_ptr + offsets, da, mask=mask)
        tl.store(db_ptr + offsets, db.to(dc.dtype), mask=mask)


def get_optimal_block_size(total_elements, is_backward=False):
    """
    Calculate optimal Block Size using compute_default_tiling_strategy.

    Args:
        total_elements: Total number of elements to process
        is_backward: Whether this is for backward pass (requires more memory)

    Returns:
        Optimal block size for the kernel
    """
    # Memory multiplier based on peak memory usage analysis
    if is_backward:
        memory_multiplier = 6.0
    else:
        memory_multiplier = 3.0
    # Call calculation function
    # Treat input as 1D (total_elements,), only tiling on dim 0
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9,
        dtype_size=4,
        memory_multiplier=memory_multiplier,
        shapes=((total_elements,),),
        tiling_dims=(0,),
    )

    # Parse result
    if tile_shapes and len(tile_shapes) > 0:
        block_size = tile_shapes[0][0]
        return max(256, block_size)
    else:
        return 2048


def geglu_forward(a, b):
    """
    High-performance GEGLU forward pass for NPU using flatten 1D approach.
    """
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    total_elements = a.numel()
    c = torch.empty_like(a)

    block_size = get_optimal_block_size(total_elements, is_backward=False)

    num_cores = get_npu_core_count()
    grid_size = min(num_cores, (total_elements + block_size - 1) // block_size)

    _geglu_forward_kernel_flat[(grid_size,)](a, b, c, total_elements, BLOCK_SIZE=block_size, NUM_STAGES=3, num_warps=4)
    return c


def geglu_backward(a, b, dc):
    """
    High-performance GEGLU backward pass for NPU using flatten 1D approach.
    """
    if not dc.is_contiguous():
        dc = dc.contiguous()
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    total_elements = dc.numel()
    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)

    block_size = get_optimal_block_size(total_elements, is_backward=True)

    num_cores = get_npu_core_count()
    grid_size = min(num_cores, (total_elements + block_size - 1) // block_size)

    _geglu_backward_kernel_flat[(grid_size,)](
        dc, a, b, grad_a, grad_b, total_elements, BLOCK_SIZE=block_size, NUM_STAGES=3, num_warps=4
    )
    return grad_a, grad_b


class LigerGELUMulFunction(torch.autograd.Function):
    """High-performance GEGLU function for Ascend NPU."""

    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        c = geglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        grad_a, grad_b = geglu_backward(a, b, dc)
        return grad_a, grad_b
