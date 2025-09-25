import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import ensure_contiguous


@triton.jit
def _relu_forward_kernel(a, c, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a += program_id * stride
    c += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a + col_offsets, mask=mask, other=0)

    # ReLU: max(0, x)
    c_row = tl.maximum(a_row, 0.0)
    tl.store(c + col_offsets, c_row, mask=mask)


@triton.jit
def _relu_backward_kernel(
    dc, a, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc += program_id * stride
    a += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
    a_row = tl.load(a + col_offsets, mask=mask, other=0)

    # ReLU derivative: 1 if x > 0, else 0
    da_row = tl.where(a_row > 0.0, dc_row, 0.0)

    tl.store(a + col_offsets, da_row, mask=mask)


def relu_forward(a):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _relu_forward_kernel[(n_rows,)](
        a,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a, c.view(*ori_shape)


def relu_backward(a, dc):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _relu_backward_kernel[(n_rows,)](
        dc,
        a,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return a.view(*ori_shape)


class LigerReLUFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a):
        a, c = relu_forward(a)
        ctx.save_for_backward(a)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        (a,) = ctx.saved_tensors
        da = relu_backward(a, dc)
        return da
