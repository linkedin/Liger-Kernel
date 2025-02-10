import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import ensure_contiguous





@triton.jit
def _squared_relu_forward_kernel(a_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # load data
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)

    # apply squared ReLU: relu(a) * relu(a)
    relu_a = tl.maximum(a_row, 0)  # ReLU
    squared_relu_a = relu_a * relu_a  # Squared ReLU

    # store result
    tl.store(c_ptr + col_offsets, squared_relu_a, mask=mask)


@triton.jit
def _squared_relu_backward_kernel(dc_ptr, a_ptr, da_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    da_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # load data
    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)

    # compute gradient of squared ReLU: 2 * relu(a) * (a > 0)
    relu_a = tl.maximum(a_row, 0)  # ReLU
    relu_grad = (a_row > 0).to(tl.float32)  # Gradient of ReLU (indicator function)
    da_row = 2 * relu_a * relu_grad * dc_row  # Gradient of squared ReLU

    # store gradient
    tl.store(da_ptr + col_offsets, da_row, mask=mask)

def relu2_forward(a):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _squared_relu_forward_kernel[(n_rows,)](
        a,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a, b, c.view(*ori_shape)


def relu2_backward(a, b, dc):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _squared_relu_backward_kernel[(n_rows,)](
        dc,
        a,
        b,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a.view(*ori_shape), b.view(*ori_shape)


class LigerReLU2MulFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a):
        a, c = relu2_forward(a)
        ctx.save_for_backward(a)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a = ctx.saved_tensors
        a = relu2_backward(a, dc)
        return a
