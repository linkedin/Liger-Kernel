import torch
import triton
import triton.language as tl

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count

# -----------------------------------------------------------------------------
# Kernels (High-performance 1D Flatten Implementation)
# -----------------------------------------------------------------------------


@triton.jit
def _swiglu_forward_kernel_flat(a_ptr, b_ptr, c_ptr, total_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Grid-Stride Loop
    start_idx = pid * BLOCK_SIZE
    stride = num_progs * BLOCK_SIZE

    for idx in tl.range(start_idx, total_elements, stride):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        a_val = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b_val = tl.load(b_ptr + offsets, mask=mask, other=0.0)

        sig_a = tl.sigmoid(a_val)
        silu_a = a_val * sig_a
        res = silu_a.cast(b_val.dtype) * b_val
        tl.store(c_ptr + offsets, res, mask=mask)


@triton.jit
def _swiglu_backward_kernel_flat(dc_ptr, a_ptr, b_ptr, da_ptr, db_ptr, total_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    start_idx = pid * BLOCK_SIZE
    stride = num_progs * BLOCK_SIZE

    for idx in tl.range(start_idx, total_elements, stride):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        dc = tl.load(dc_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        sig_a = tl.sigmoid(a)
        silu_a = a * sig_a
        term1 = silu_a * (1.0 - sig_a) + sig_a

        db = dc * silu_a
        da = dc * b * term1

        tl.store(da_ptr + offsets, da, mask=mask)
        tl.store(db_ptr + offsets, db, mask=mask)


# -----------------------------------------------------------------------------
# Helper: Call compute_default_tiling_strategy
# -----------------------------------------------------------------------------


def get_optimal_block_size(total_elements, is_backward: bool = False):
    # Align UB assumptions with ascend GEGLU; SwiLU backward uses fewer temporaries than GELU.
    memory_multiplier = 6.0 if is_backward else 3.0
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9,
        dtype_size=4,
        memory_multiplier=memory_multiplier,
        shapes=((total_elements,),),
        tiling_dims=(0,),
    )

    if tile_shapes and len(tile_shapes) > 0:
        block_size = tile_shapes[0][0]
        return max(256, block_size)
    return 2048


def swiglu_forward(a, b):
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    total_elements = a.numel()
    c = torch.empty_like(a)

    block_size = get_optimal_block_size(total_elements, is_backward=False)

    num_cores = get_npu_core_count()
    # Avoid grid_size == 0 when num_vectorcore reports 0 (would fault the NPU).
    grid_size = max(1, min(num_cores, (total_elements + block_size - 1) // block_size))

    _swiglu_forward_kernel_flat[(grid_size,)](a, b, c, total_elements, BLOCK_SIZE=block_size)
    return c


def swiglu_backward(a, b, dc):
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
    grid_size = max(1, min(num_cores, (total_elements + block_size - 1) // block_size))

    _swiglu_backward_kernel_flat[(grid_size,)](dc, a, b, grad_a, grad_b, total_elements, BLOCK_SIZE=block_size)
    return grad_a, grad_b


class LigerSiLUMulFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b, gate_multiplier: float = 1.0, down_multiplier: float = 1.0):
        gate_multiplier = float(gate_multiplier)
        down_multiplier = float(down_multiplier)
        ctx.gate_multiplier = gate_multiplier
        ctx.down_multiplier = down_multiplier

        a_in = a if gate_multiplier == 1.0 else a * gate_multiplier
        c = swiglu_forward(a_in, b)
        if down_multiplier != 1.0:
            c = c * down_multiplier
        ctx.save_for_backward(a_in, b)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a_in, b = ctx.saved_tensors
        gate_multiplier = ctx.gate_multiplier
        down_multiplier = ctx.down_multiplier

        if down_multiplier != 1.0:
            dc = dc * down_multiplier
        grad_a_in, grad_b = swiglu_backward(a_in, b, dc)
        grad_a = grad_a_in if gate_multiplier == 1.0 else grad_a_in * gate_multiplier
        return grad_a, grad_b, None, None
