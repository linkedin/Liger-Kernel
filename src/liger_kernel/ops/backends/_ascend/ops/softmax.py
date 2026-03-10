import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count


@triton.jit
def _softmax_single_block_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    """
    Single-block softmax forward kernel for small column sizes.

    Processes entire row in one block when n_cols <= BLOCK_SIZE.
    Uses 2D tensor to process multiple rows simultaneously for better UB utilization.

    Args:
        Y_ptr: Output tensor pointer
        Y_row_stride: Stride for output rows
        X_ptr: Input tensor pointer
        X_row_stride: Stride for input rows
        n_rows: Number of rows to process
        n_cols: Number of columns per row
        BLOCK_SIZE: Block size for column processing
        ROWS_PER_BLOCK: Number of rows to process simultaneously
    """
    row_block_start = tl.program_id(0) * ROWS_PER_BLOCK
    row_block_step = tl.num_programs(0) * ROWS_PER_BLOCK

    row_offsets = tl.arange(0, ROWS_PER_BLOCK)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    for row_block_idx in tl.range(row_block_start, n_rows, row_block_step):
        row_idx = row_block_idx + row_offsets
        row_mask = row_idx < n_rows
        col_mask = col_offsets < n_cols

        # 2D mask: [ROWS_PER_BLOCK, BLOCK_SIZE]
        mask = row_mask[:, None] & col_mask[None, :]

        # Load 2D block: [ROWS_PER_BLOCK, BLOCK_SIZE]
        offsets = row_idx[:, None] * X_row_stride + col_offsets[None, :]
        x = tl.load(X_ptr + offsets, mask=mask, other=float("-inf"))

        # Compute softmax per row (axis=1)
        m = tl.max(x, axis=1)
        e = tl.exp(x - m[:, None])
        d = tl.sum(e, axis=1)
        y = e / d[:, None]

        # Store 2D block
        offsets = row_idx[:, None] * Y_row_stride + col_offsets[None, :]
        tl.store(Y_ptr + offsets, y, mask=mask)


@triton.jit
def _softmax_multi_block_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
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
    col_offsets = tl.arange(0, BLOCK_SIZE)

    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = X_ptr + row_idx * X_row_stride
        m = tl.float32(float("-inf"))
        d = tl.float32(0.0)

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
def _softmax_single_block_backward_kernel(
    dy_ptr,
    dy_stride,
    y_ptr,
    y_stride,
    dx_ptr,
    dx_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    """
    Single-block softmax backward kernel for small column sizes.

    Computes gradient: dx = y * (dy - sum(dy * y))
    Uses 2D tensor to process multiple rows simultaneously for better UB utilization.

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
        ROWS_PER_BLOCK: Number of rows to process simultaneously
    """
    row_block_start = tl.program_id(0) * ROWS_PER_BLOCK
    row_block_step = tl.num_programs(0) * ROWS_PER_BLOCK

    row_offsets = tl.arange(0, ROWS_PER_BLOCK)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    for row_block_idx in tl.range(row_block_start, n_rows, row_block_step):
        row_idx = row_block_idx + row_offsets
        row_mask = row_idx < n_rows
        col_mask = col_offsets < n_cols

        # 2D mask: [ROWS_PER_BLOCK, BLOCK_SIZE]
        mask = row_mask[:, None] & col_mask[None, :]

        # Load 2D blocks: [ROWS_PER_BLOCK, BLOCK_SIZE]
        dy_offsets = row_idx[:, None] * dy_stride + col_offsets[None, :]
        y_offsets = row_idx[:, None] * y_stride + col_offsets[None, :]

        dy = tl.load(dy_ptr + dy_offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + y_offsets, mask=mask, other=0.0)

        # Compute dot product per row (axis=1)
        dot = tl.sum(dy * y, axis=1)
        dx = y * (dy - dot[:, None])

        # Store 2D block
        dx_offsets = row_idx[:, None] * dx_stride + col_offsets[None, :]
        tl.store(dx_ptr + dx_offsets, dx, mask=mask)


@triton.jit
def _softmax_multi_block_backward_kernel(
    dy_ptr,
    dy_stride,
    y_ptr,
    y_stride,
    dx_ptr,
    dx_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
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

    if n_cols <= BLOCK_SIZE:
        # Calculate optimal ROWS_PER_BLOCK to utilize UB efficiently
        # Target: ROWS_PER_BLOCK * BLOCK_SIZE <= MAX_FUSED_BLOCK_SIZE
        ROWS_PER_BLOCK = min(MAX_FUSED_BLOCK_SIZE // BLOCK_SIZE, 32)
        ROWS_PER_BLOCK = triton.next_power_of_2(ROWS_PER_BLOCK)

        # Calculate number of programs needed
        num_row_blocks = (n_rows + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
        num_programs = min(num_cores, num_row_blocks)

        _softmax_single_block_forward_kernel[(num_programs,)](
            y2d, y2d.stride(0), x2d, x2d.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, ROWS_PER_BLOCK=ROWS_PER_BLOCK
        )
        multi_block_launch = False
    else:
        num_programs = min(num_cores, n_rows)
        ROWS_PER_BLOCK = 1  # Not used in multi-block

        _softmax_multi_block_forward_kernel[(num_programs,)](
            y2d, y2d.stride(0), x2d, x2d.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE
        )
        multi_block_launch = True

    return y2d.view(*batch, n_cols), BLOCK_SIZE, ROWS_PER_BLOCK, multi_block_launch


def softmax_backward(
    dy: torch.Tensor,
    y: torch.Tensor,
    BLOCK_SIZE: int,
    ROWS_PER_BLOCK: int,
    multi_block_launch: bool,
) -> torch.Tensor:
    *batch, n_cols = dy.shape
    dy2d = dy.contiguous().view(-1, n_cols)
    y2d = y.contiguous().view(-1, n_cols)
    n_rows = dy2d.shape[0]
    dx2d = torch.empty_like(dy2d)

    num_cores = get_npu_core_count()

    if not multi_block_launch and n_cols <= BLOCK_SIZE:
        num_row_blocks = (n_rows + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
        num_programs = min(num_cores, num_row_blocks)
        _softmax_single_block_backward_kernel[(num_programs,)](
            dy2d,
            dy2d.stride(0),
            y2d,
            y2d.stride(0),
            dx2d,
            dx2d.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            ROWS_PER_BLOCK=ROWS_PER_BLOCK,
        )
    else:
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
        y, BLOCK_SIZE, ROWS_PER_BLOCK, multi_block_launch = softmax_forward(input_)
        ctx.save_for_backward(y)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.ROWS_PER_BLOCK = ROWS_PER_BLOCK
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
            ctx.ROWS_PER_BLOCK,
            ctx.multi_block_launch,
        )
        return dx
