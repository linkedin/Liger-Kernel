import math
import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import ensure_contiguous


@triton.jit
def _gelu_forward_kernel(a, c, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a += program_id * stride
    c += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)

    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    c_row = 0.5 * a_row * (1 + tl.erf(a_row / tl.sqrt(2.0)))
    tl.store(c + col_offsets, c_row, mask=mask)


@triton.jit
def _gelu_backward_kernel(
    dc, a, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc += program_id * stride
    a += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)

    # GELU derivative: 0.5 * (1 + erf(x / sqrt(2))) + 0.5 * x * (1 / sqrt(2 * pi)) * exp(-0.5 * x^2)
    da_row = 0.5 * (1 + tl.erf(a_row / tl.sqrt(2.0))) + 0.5 * a_row * (
        1 / tl.sqrt(2 * math.pi)
    ) * tl.exp(-0.5 * a_row * a_row)
    da_row = da_row * dc_row
    tl.store(a + col_offsets, da_row, mask=mask)


def gelu_forward(a):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _gelu_forward_kernel[(n_rows,)](
        a,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a, c.view(*ori_shape)


def gelu_backward(a, dc):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _gelu_backward_kernel[(n_rows,)](
        dc,
        a,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return a.view(*ori_shape)


class LigerGELUFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a):
        a, c = gelu_forward(a)
        ctx.save_for_backward(a)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        (a,) = ctx.saved_tensors
        da = gelu_backward(a, dc)
        return da
