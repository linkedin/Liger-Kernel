import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count


@triton.jit
def _softmax_forward_kernel(Y_ptr, X_ptr, X_row_stride, Y_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = X_ptr + row_idx * X_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        X_ptrs = row_start_ptr + col_offsets

        mask = col_offsets < n_cols
        row = tl.load(X_ptrs, mask=mask, other=-float("inf"))

        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        output_row_start_ptr = Y_ptr + row_idx * Y_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


@triton.jit
def _softmax_multi_block_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = X_ptr + row_idx * X_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        m = tl.full((), -float("inf"), tl.float32)
        d = tl.full((), 0.0, tl.float32)

        for start in tl.range(0, n_cols, BLOCK_SIZE):
            idx = start + col_offsets
            mask = idx < n_cols
            xblk = tl.load(row_start_ptr + idx, mask=mask, other=-float("inf"))
            blk_max = tl.max(xblk, axis=0)
            new_m = tl.maximum(m, blk_max)
            d = d * tl.exp(m - new_m) + tl.sum(tl.exp(xblk - new_m), axis=0)
            m = new_m

        for start in tl.range(0, n_cols, BLOCK_SIZE):
            idx = start + col_offsets
            mask = idx < n_cols
            xblk = tl.load(row_start_ptr + idx, mask=mask, other=-float("inf"))
            yblk = tl.exp(xblk - m) / d
            tl.store(Y_ptr + row_idx * Y_row_stride + idx, yblk, mask=mask)


@triton.jit
def _softmax_backward_kernel(
    dX_ptr,
    dY_ptr,
    Y_ptr,
    dY_row_stride,
    Y_row_stride,
    dX_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        dy_row_ptr = dY_ptr + row_idx * dY_row_stride
        y_row_ptr = Y_ptr + row_idx * Y_row_stride
        dx_row_ptr = dX_ptr + row_idx * dX_row_stride

        dy_ptrs = dy_row_ptr + col_offsets
        y_ptrs = y_row_ptr + col_offsets
        dx_ptrs = dx_row_ptr + col_offsets

        dy = tl.load(dy_ptrs, mask=mask, other=0.0)
        y = tl.load(y_ptrs, mask=mask, other=0.0)

        dot = tl.sum(dy * y, axis=0)
        dx = y * (dy - dot)
        tl.store(dx_ptrs, dx, mask=mask)


@triton.jit
def _softmax_multi_block_backward_kernel(
    dy_ptr,
    dy_stride,
    y_ptr,
    y_stride,
    dx_ptr,
    dx_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.full((), 0.0, tl.float32)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step):
        dy_start_ptr = dy_ptr + row_idx * dy_stride
        y_start_ptr = y_ptr + row_idx * y_stride
        for start in tl.range(0, n_cols, BLOCK_SIZE):
            idx = start + col_offsets
            mask = idx < n_cols
            dy_blk = tl.load(dy_start_ptr + idx, mask=mask, other=0.0)
            y_blk = tl.load(y_start_ptr + idx, mask=mask, other=0.0)
            acc += tl.sum(dy_blk * y_blk, axis=0)

        for start in tl.range(0, n_cols, BLOCK_SIZE):
            idx = start + col_offsets
            mask = idx < n_cols
            dy_blk = tl.load(dy_start_ptr + idx, mask=mask, other=0.0)
            y_blk = tl.load(y_start_ptr + idx, mask=mask, other=0.0)
            dx_blk = y_blk * (dy_blk - acc)
            tl.store(dx_ptr + row_idx * dx_stride + idx, dx_blk, mask=mask)


def softmax_forward(x):
    *batch, n_cols = x.shape
    x2d = x.contiguous().view(-1, n_cols)
    n_rows = x2d.shape[0]
    MAX_FUSED_BLOCK_SIZE = 8192

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, MAX_FUSED_BLOCK_SIZE)

    y2d = torch.empty_like(x2d)
    num_cores = get_npu_core_count()
    num_programs = min(num_cores, n_rows)

    if n_cols <= BLOCK_SIZE:
        _softmax_forward_kernel[(num_programs,)](y2d, x2d, x2d.stride(0), y2d.stride(0), n_rows, n_cols, BLOCK_SIZE)
        multi_block_launch = False
    else:
        _softmax_multi_block_forward_kernel[(num_programs,)](
            y2d, y2d.stride(0), x2d, x2d.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE
        )
        multi_block_launch = True
    return y2d.view(*batch, n_cols), BLOCK_SIZE, multi_block_launch


def softmax_backward(
    dy: torch.Tensor,
    y: torch.Tensor,
    BLOCK_SIZE: int,
    multi_block_launch: bool,
) -> torch.Tensor:
    *batch, n_cols = dy.shape
    dy2d = dy.contiguous().view(-1, n_cols)
    y2d = y.contiguous().view(-1, n_cols)
    n_rows = dy2d.shape[0]
    dx2d = torch.empty_like(dy2d)

    num_cores = get_npu_core_count()
    num_programs = min(num_cores, n_rows)

    if not multi_block_launch and n_cols <= BLOCK_SIZE:
        _softmax_backward_kernel[(num_programs,)](
            dx2d,
            dy2d,
            y2d,
            dy2d.stride(0),
            y2d.stride(0),
            dx2d.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        _softmax_multi_block_backward_kernel[(num_programs,)](
            dy2d,
            dy2d.stride(0),
            y2d,
            y2d.stride(0),
            dx2d,
            dx2d.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return dx2d.view(*batch, n_cols)


class LigerSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, input_: torch.Tensor):
        y, BLOCK_SIZE, multi_block_launch = softmax_forward(input_)
        ctx.save_for_backward(y)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.multi_block_launch = multi_block_launch
        return y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        dx = softmax_backward(
            grad_output,
            y,
            ctx.BLOCK_SIZE,
            ctx.multi_block_launch,
        )
        return dx
