import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count


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
    """
    Multi-block softmax forward kernel using two-pass algorithm.

    First pass computes max and sum for numerical stability.
    Second pass normalizes and writes output.

    Args:
        Y_ptr: Output tensor pointer
        Y_row_stride: Stride for output rows
        X_ptr: Input tensor pointer
        X_row_stride: Stride for input rows
        n_rows: Number of rows to process
        n_cols: Number of columns per row
        BLOCK_SIZE: Block size for column processing
    """
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = X_ptr + row_idx * X_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        m = float("-inf")
        d = 0.0

        for start in tl.range(0, n_cols, BLOCK_SIZE):
            idx = start + col_offsets
            mask = idx < n_cols
            xblk = tl.load(
                row_start_ptr + idx, mask=mask, other=float("-inf"), eviction_policy="evict_first", cache_modifier=".ca"
            )
            blk_max = tl.max(xblk, axis=0)
            new_m = tl.maximum(m, blk_max)
            d = d * tl.exp(m - new_m) + tl.sum(tl.exp(xblk - new_m), axis=0)
            m = new_m

        for start in tl.range(0, n_cols, BLOCK_SIZE):
            idx = start + col_offsets
            mask = idx < n_cols
            xblk = tl.load(
                row_start_ptr + idx, mask=mask, other=float("-inf"), eviction_policy="evict_first", cache_modifier=".ca"
            )
            yblk = tl.exp(xblk - m) / d
            tl.store(Y_ptr + row_idx * Y_row_stride + idx, yblk, mask=mask, cache_modifier=".cs")


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
    """
    Multi-block softmax backward kernel using two-pass algorithm.

    Computes gradient: dx = y * (dy - sum(dy * y))

    Args:
        dy_ptr: Gradient output pointer
        dy_stride: Stride for gradient output rows
        y_ptr: Forward output pointer
        y_stride: Stride for forward output rows
        dx_ptr: Gradient input pointer
        dx_stride: Stride for gradient input rows
        n_rows: Number of rows to process
        n_cols: Number of columns per row
        BLOCK_SIZE: Block size for column processing
    """
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    acc = 0.0
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step):
        dy_start_ptr = dy_ptr + row_idx * dy_stride
        y_start_ptr = y_ptr + row_idx * y_stride

        for start in tl.range(0, n_cols, BLOCK_SIZE):
            idx = start + col_offsets
            mask = idx < n_cols
            dy_blk = tl.load(dy_start_ptr + idx, mask=mask, other=0.0, eviction_policy="evict_first")
            y_blk = tl.load(
                y_start_ptr + idx, mask=mask, other=0.0, eviction_policy="evict_first", cache_modifier=".ca"
            )
            acc += tl.sum(dy_blk * y_blk, axis=0)

        for start in tl.range(0, n_cols, BLOCK_SIZE):
            idx = start + col_offsets
            mask = idx < n_cols
            dy_blk = tl.load(dy_start_ptr + idx, mask=mask, other=0.0)
            y_blk = tl.load(y_start_ptr + idx, mask=mask, other=0.0, cache_modifier=".ca")
            dx_blk = y_blk * (dy_blk - acc)
            tl.store(dx_ptr + row_idx * dx_stride + idx, dx_blk, mask=mask, cache_modifier=".wb")


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

    _softmax_multi_block_forward_kernel[(num_programs,)](
        y2d, y2d.stride(0), x2d, x2d.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE
    )

    return y2d.view(*batch, n_cols), BLOCK_SIZE


def softmax_backward(
    dy: torch.Tensor,
    y: torch.Tensor,
    BLOCK_SIZE: int,
) -> torch.Tensor:
    *batch, n_cols = dy.shape
    dy2d = dy.contiguous().view(-1, n_cols)
    y2d = y.contiguous().view(-1, n_cols)
    n_rows = dy2d.shape[0]
    dx2d = torch.empty_like(dy2d)

    num_cores = get_npu_core_count()
    num_programs = min(num_cores, n_rows)

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
        y, BLOCK_SIZE = softmax_forward(input_)
        ctx.save_for_backward(y)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        dx = softmax_backward(
            grad_output,
            y,
            ctx.BLOCK_SIZE,
        )
        return dx
