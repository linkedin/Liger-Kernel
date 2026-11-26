from typing import Tuple

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import ensure_contiguous

MAX_FUSED_SIZE = 65536


@triton.jit
def _softmax_single_block_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x = tl.load(X_ptr + row_id * X_row_stride + offs, mask=mask, other=-float("inf"), cache_modifier=".ca")
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    d = tl.sum(e, axis=0)
    y = e / d
    tl.store(Y_ptr + row_id * Y_row_stride + offs, y, mask=mask, cache_modifier=".cs")


@triton.jit
def _softmax_multi_block_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_SIZE)

    m = float("-inf")
    d = 0.0
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        xblk = tl.load(X_ptr + row_id * X_row_stride + idx, mask=mask, other=-float("inf"), cache_modifier=".ca")
        blk_max = tl.max(xblk, axis=0)
        new_m = tl.maximum(m, blk_max)
        d = d * tl.exp(m - new_m) + tl.sum(tl.exp(xblk - new_m), axis=0)
        m = new_m

    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        xblk = tl.load(X_ptr + row_id * X_row_stride + idx, mask=mask, other=-float("inf"), cache_modifier=".ca")
        yblk = tl.exp(xblk - m) / d
        tl.store(Y_ptr + row_id * Y_row_stride + idx, yblk, mask=mask, cache_modifier=".cs")


@triton.jit
def _softmax_single_block_backward_kernel(
    dy_ptr,
    dy_stride,
    y_ptr,
    y_stride,
    dx_ptr,
    dx_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    dy = tl.load(dy_ptr + row_id * dy_stride + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + row_id * y_stride + offs, mask=mask, other=0.0, cache_modifier=".ca")
    dot = tl.sum(dy * y, axis=0)
    dx = y * (dy - dot)
    # At the argmax k, dy - dot cancels to 0 once y_k rounds to 1 (arXiv 2610.31642). Using 1 - y_k = sum of the
    # other y gives dx_k = y_k * sum_{j != k} y_j * (dy_k - dy_j), which doesn't cancel.
    is_max = offs == tl.argmax(tl.where(mask, y, -1.0), axis=0)
    y_k = tl.sum(tl.where(is_max, y, 0.0), axis=0)
    dy_k = tl.sum(tl.where(is_max, dy, 0.0), axis=0)
    dx_k = y_k * tl.sum(tl.where(is_max, 0.0, y * (dy_k - dy)), axis=0)
    dx = tl.where(is_max, dx_k, dx)
    tl.store(dx_ptr + row_id * dx_stride + offs, dx, mask=mask, cache_modifier=".wb")


@triton.jit
def _softmax_multi_block_backward_kernel(
    dy_ptr,
    dy_stride,
    y_ptr,
    y_stride,
    dx_ptr,
    dx_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_SIZE)
    acc = 0.0
    y_k = -1.0
    argmax = 0
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        dy_blk = tl.load(dy_ptr + row_id * dy_stride + idx, mask=mask, other=0.0)
        y_blk = tl.load(y_ptr + row_id * y_stride + idx, mask=mask, other=0.0, cache_modifier=".ca")
        acc += tl.sum(dy_blk * y_blk, axis=0)
        blk_max = tl.max(tl.where(mask, y_blk, -1.0), axis=0)
        argmax = tl.where(blk_max > y_k, tl.min(tl.where(y_blk == blk_max, idx, n_cols)), argmax)
        y_k = tl.maximum(y_k, blk_max)

    dy_k = tl.load(dy_ptr + row_id * dy_stride + argmax).to(tl.float32)
    dx_k = 0.0
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        dy_blk = tl.load(dy_ptr + row_id * dy_stride + idx, mask=mask, other=0.0)
        y_blk = tl.load(y_ptr + row_id * y_stride + idx, mask=mask, other=0.0, cache_modifier=".ca")
        dx_blk = y_blk * (dy_blk - acc)
        dx_k += tl.sum(tl.where(idx == argmax, 0.0, y_blk * (dy_k - dy_blk)), axis=0)
        tl.store(dx_ptr + row_id * dx_stride + idx, dx_blk, mask=mask, cache_modifier=".wb")

    # Same argmax correction as the single-block kernel, overwriting what the loop stored there.
    tl.debug_barrier()
    tl.store(dx_ptr + row_id * dx_stride + argmax, y_k * dx_k)


def _softmax_forward(x: torch.Tensor) -> Tuple[torch.Tensor, int, int, bool]:
    *batch, n_cols = x.shape
    x2d = x.contiguous().view(-1, n_cols)
    n_rows = x2d.shape[0]

    # Rows wider than the largest block stream through the multi-block kernels.
    BLOCK_SIZE, num_warps = calculate_settings(min(n_cols, MAX_FUSED_SIZE))
    y2d = torch.empty_like(x2d)

    if n_cols <= BLOCK_SIZE:
        _softmax_single_block_forward_kernel[(n_rows,)](
            y2d, y2d.stride(0), x2d, x2d.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
        multi_block_launch = False
    else:
        _softmax_multi_block_forward_kernel[(n_rows,)](
            y2d, y2d.stride(0), x2d, x2d.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
        multi_block_launch = True

    return y2d.view(*batch, n_cols), BLOCK_SIZE, num_warps, multi_block_launch


def _softmax_backward(
    dy: torch.Tensor,
    y: torch.Tensor,
    BLOCK_SIZE: int,
    num_warps: int,
    multi_block_launch: bool,
) -> torch.Tensor:
    *batch, n_cols = dy.shape
    dy2d = dy.contiguous().view(-1, n_cols)
    y2d = y.contiguous().view(-1, n_cols)
    n_rows = dy2d.shape[0]
    dx2d = torch.empty_like(dy2d)

    if not multi_block_launch and n_cols <= BLOCK_SIZE:
        _softmax_single_block_backward_kernel[(n_rows,)](
            dy2d,
            dy2d.stride(0),
            y2d,
            y2d.stride(0),
            dx2d,
            dx2d.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        _softmax_multi_block_backward_kernel[(n_rows,)](
            dy2d,
            dy2d.stride(0),
            y2d,
            y2d.stride(0),
            dx2d,
            dx2d.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return dx2d.view(*batch, n_cols)


class LigerSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, input_: torch.Tensor):
        y, BLOCK_SIZE, num_warps, multi_block_launch = _softmax_forward(input_)
        ctx.save_for_backward(y)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.multi_block_launch = multi_block_launch
        return y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        dx = _softmax_backward(
            grad_output,
            y,
            ctx.BLOCK_SIZE,
            ctx.num_warps,
            ctx.multi_block_launch,
        )
        return dx
