import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import ensure_contiguous


@triton.jit
def _relu_squared_forward_kernel(
    Y_ptr,
    Y_stride,
    X_ptr,
    X_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    X_ptr += row_idx * X_stride
    Y_ptr += row_idx * Y_stride

    x_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    # relu(x) = max(0, x), then square
    relu_x = tl.maximum(x_row, 0)
    y_row = relu_x * relu_x

    tl.store(Y_ptr + col_offsets, y_row, mask=mask)


@triton.jit
def _relu_squared_backward_kernel(
    dX_ptr,
    dX_stride,
    dY_ptr,
    dY_stride,
    X_ptr,
    X_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dX_ptr += row_idx * dX_stride
    dY_ptr += row_idx * dY_stride
    X_ptr += row_idx * X_stride

    dy_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0)
    x_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)

    # d/dx[relu(x)^2] = 2 * relu(x) = 2 * x * (x > 0)
    relu_x = tl.maximum(x_row, 0)
    dx_row = dy_row * 2 * relu_x

    tl.store(dX_ptr + col_offsets, dx_row, mask=mask)


def relu_squared_forward(X):
    ori_shape = X.shape
    n_cols = ori_shape[-1]
    X_2d = X.view(-1, n_cols)
    n_rows = X_2d.shape[0]

    Y = torch.empty_like(X_2d)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _relu_squared_forward_kernel[(n_rows,)](
        Y,
        Y.stride(-2),
        X_2d,
        X_2d.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return Y.view(*ori_shape)


def relu_squared_backward(X, dY):
    ori_shape = dY.shape
    n_cols = ori_shape[-1]
    X_2d = X.view(-1, n_cols)
    dY_2d = dY.view(-1, n_cols)
    n_rows = dY_2d.shape[0]

    dX = torch.empty_like(dY_2d)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _relu_squared_backward_kernel[(n_rows,)](
        dX,
        dX.stride(-2),
        dY_2d,
        dY_2d.stride(-2),
        X_2d,
        X_2d.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return dX.view(*ori_shape)


class LigerReLUSquaredFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(X):
        Y = relu_squared_forward(X)
        return Y, X.view_as(X)

    @staticmethod
    def setup_context(ctx, inputs, output):
        Y, X = output
        ctx.save_for_backward(X)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY, _dX):
        (X,) = ctx.saved_tensors
        dX = relu_squared_backward(X, dY)
        return dX
