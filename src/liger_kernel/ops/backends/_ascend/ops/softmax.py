from typing import Tuple

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous


@triton.jit
def _softmax_single_block_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
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
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,  # 固定为 48
):
    program_id = tl.program_id(0)
    
    # 计算每个 program 需要处理的行数
    rows_per_program = tl.cdiv(n_rows, NUM_PROGRAMS)
    
    # 计算当前 program 处理的行范围
    row_start = program_id * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, n_rows)
    
    offs = tl.arange(0, BLOCK_SIZE)
    
    # 循环处理分配给当前 program 的所有行
    for row_id in range(row_start, row_end):
        # 第一遍：计算 max 和 sum
        m = float("-inf")
        d = 0.0
        
        for start in tl.range(0, n_cols, BLOCK_SIZE):
            idx = start + offs
            mask = idx < n_cols
            xblk = tl.load(
                X_ptr + row_id * X_row_stride + idx, 
                mask=mask, 
                other=float("-inf"), 
                cache_modifier=".ca"
            )
            blk_max = tl.max(xblk, axis=0)
            new_m = tl.maximum(m, blk_max)
            d = d * tl.exp(m - new_m) + tl.sum(tl.exp(xblk - new_m), axis=0)
            m = new_m
        
        # 第二遍：归一化并写入
        for start in tl.range(0, n_cols, BLOCK_SIZE):
            idx = start + offs
            mask = idx < n_cols
            xblk = tl.load(
                X_ptr + row_id * X_row_stride + idx, 
                mask=mask, 
                other=float("-inf"), 
                cache_modifier=".ca"
            )
            yblk = tl.exp(xblk - m) / d
            tl.store(
                Y_ptr + row_id * Y_row_stride + idx, 
                yblk, 
                mask=mask, 
                cache_modifier=".cs"
            )

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
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    dy = tl.load(dy_ptr + row_id * dy_stride + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + row_id * y_stride + offs, mask=mask, other=0.0, cache_modifier=".ca")
    # dot = tl.sum(dy * y, axis=1)
    dot = tl.sum(dy * y, axis=0)
    dx = y * (dy - dot)
    tl.store(dx_ptr + row_id * dx_stride + offs, dx, mask=mask, cache_modifier=".wb")

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
    NUM_PROGRAMS: tl.constexpr,
):
    program_id = tl.program_id(0)
    
    # 计算每个 program 需要处理的行数
    rows_per_program = tl.cdiv(n_rows, NUM_PROGRAMS)
    
    # 计算当前 program 处理的行范围
    row_start = program_id * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, n_rows)
    
    offs = tl.arange(0, BLOCK_SIZE)
    
    # 循环处理分配给当前 program 的所有行
    for row_id in range(row_start, row_end):
        # 第一遍：计算 sum(dy * y)
        acc = 0.0
        
        for start in tl.range(0, n_cols, BLOCK_SIZE):
            idx = start + offs
            mask = idx < n_cols
            dy_blk = tl.load(dy_ptr + row_id * dy_stride + idx, mask=mask, other=0.0)
            y_blk = tl.load(y_ptr + row_id * y_stride + idx, mask=mask, other=0.0, cache_modifier=".ca")
            acc += tl.sum(dy_blk * y_blk, axis=0)
        
        # 第二遍：计算 dx = y * (dy - acc)
        for start in tl.range(0, n_cols, BLOCK_SIZE):
            idx = start + offs
            mask = idx < n_cols
            dy_blk = tl.load(dy_ptr + row_id * dy_stride + idx, mask=mask, other=0.0)
            y_blk = tl.load(y_ptr + row_id * y_stride + idx, mask=mask, other=0.0, cache_modifier=".ca")
            dx_blk = y_blk * (dy_blk - acc)
            tl.store(dx_ptr + row_id * dx_stride + idx, dx_blk, mask=mask, cache_modifier=".wb")

def _softmax_forward(x: torch.Tensor) -> Tuple[torch.Tensor, int, int, bool]:
    *batch, n_cols = x.shape
    x2d = x.contiguous().view(-1, n_cols)
    n_rows = x2d.shape[0]

    BLOCK_SIZE = 1024
    num_cores = 48
    y2d = torch.empty_like(x2d)

    if n_cols <= BLOCK_SIZE:
        _softmax_single_block_forward_kernel[(n_rows,)](
            y2d, y2d.stride(0), x2d, x2d.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE, 
        )
        multi_block_launch = False
    else:
        _softmax_multi_block_forward_kernel[(num_cores,)](
            y2d, y2d.stride(0), x2d, x2d.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, NUM_PROGRAMS=num_cores
        )
        multi_block_launch = True

    return y2d.view(*batch, n_cols), BLOCK_SIZE, multi_block_launch


def _softmax_backward(
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
    num_cores = 48

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
        )
    else:
        _softmax_multi_block_backward_kernel[(num_cores,)](
            dy2d,
            dy2d.stride(0),
            y2d,
            y2d.stride(0),
            dx2d,
            dx2d.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_PROGRAMS=num_cores,
        )

    return dx2d.view(*batch, n_cols)


class LigerSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, input_: torch.Tensor):
        y, BLOCK_SIZE, multi_block_launch = _softmax_forward(input_)
        ctx.save_for_backward(y)
        ctx.BLOCK_SIZE = BLOCK_SIZE
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
            ctx.multi_block_launch,
        )
        return dx
