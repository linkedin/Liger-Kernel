import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.utils import is_npu_available

if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh


@triton.jit
def _geglu_tanh_forward_kernel(
    a, b, c, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)
    base_offset = program_id * stride
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)

    for sub_block_idx in range(num_sub_blocks):
        col_offsets = tl.arange(0, BLOCK_SIZE_SUB) + sub_block_idx * BLOCK_SIZE_SUB
        mask = col_offsets < n_cols

        a_row = tl.load(a + base_offset + col_offsets, mask=mask, other=0).to(tl.float32)
        b_row = tl.load(b + base_offset + col_offsets, mask=mask, other=0)

        # GELU tanh approximation (same as original)
        sqrt_2_over_pi = 0.7978845608028654
        a_cubed = a_row**3
        tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
        c_row = 0.5 * a_row * (1 + tanh(tanh_arg)) * b_row

        tl.store(c + base_offset + col_offsets, c_row, mask=mask)


@triton.jit
def _geglu_tanh_backward_kernel(
    dc, a, b, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc += program_id * stride
    a += program_id * stride
    b += program_id * stride

    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)

    for sub_block_idx in range(num_sub_blocks):
        col_offsets = tl.arange(0, BLOCK_SIZE_SUB) + sub_block_idx * BLOCK_SIZE_SUB
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

        db_row = dc_row * geglu_a

        # Gradient w.r.t. a can be computed with:
        # b * (0.5 * (1 + tanh(z)) + 0.5 * a * (1 - tanh(z)^2) * (sqrt(2/pi) * (1 + 3 * 0.044715 * a^2)))
        # where z = sqrt(2/pi) * (a + 0.044715 * a^3)
        term1 = 0.5 * (1 + tanh_result)
        tanh_sq = tanh_result * tanh_result
        term2 = 0.5 * a_row * (1 - tanh_sq) * (sqrt_2_over_pi * (1 + 3 * 0.044715 * a_row * a_row))
        da_row = dc_row * b_row * (term1 + term2)

        tl.store(a + col_offsets, da_row, mask=mask)
        tl.store(b + col_offsets, db_row, mask=mask)


def geglu_forward(a, b):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)

    n_rows = a.shape[0]
    # TODO: support larger n_rows for NPU. (currently limited by launch grid size)
    # Each NPU core can handle up to 65535 program ids. Solution: elementwise kernel?
    assert n_rows <= 65535, "Number of rows exceeds the maximum limit for NPU devices."

    # TODO: autotune to find the best configuration instead of hardcoding
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    BLOCK_SIZE_SUB = 1024  # sub-block size to avoid ub overflow on NPU

    _geglu_tanh_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_SIZE_SUB=BLOCK_SIZE_SUB,
        num_warps=num_warps,
    )
    return a, b, c.view(*ori_shape), BLOCK_SIZE, BLOCK_SIZE_SUB, num_warps


def geglu_backward(a, b, dc, BLOCK_SIZE, BLOCK_SIZE_SUB, num_warps):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    _geglu_tanh_backward_kernel[(n_rows,)](
        dc,
        a,
        b,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_SIZE_SUB=BLOCK_SIZE_SUB,
        num_warps=num_warps,
    )

    return a.view(*ori_shape), b.view(*ori_shape)


class LigerGELUMulFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        a, b, c, BLOCK_SIZE, BLOCK_SIZE_SUB, num_warps = geglu_forward(a, b)
        ctx.save_for_backward(a, b)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.BLOCK_SIZE_SUB = BLOCK_SIZE_SUB
        ctx.num_warps = num_warps
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        BLOCK_SIZE = ctx.BLOCK_SIZE
        BLOCK_SIZE_SUB = ctx.BLOCK_SIZE_SUB
        num_warps = ctx.num_warps
        a, b = geglu_backward(a, b, dc, BLOCK_SIZE, BLOCK_SIZE_SUB, num_warps)
        return a, b
