import math
import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import (
    calculate_settings,
    compare_version,
    ensure_contiguous,
)

if compare_version("triton", operator.ge, "3.0.0"):
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import rsqrt
else:
    from triton.language.math import rsqrt


@triton.jit
def _group_norm_forward_kernel(
    Y_ptr,  # pointer to output, shape (n_rows, n_groups, hidden_size)
    Y_row_stride,  # stride of each row in output
    Y_col_stride, # stride of each column in output
    X_ptr,  # pointer to input, shape (n_rows, n_groups, hidden_size)
    X_row_stride,  # stride of each row in input
    X_col_stride,  # stride of each column in input
    Mean_ptr,  # pointer to mean, shape (n_rows, n_groups)
    Mean_row_stride,  # stride of each row in mean
    Mean_col_stride,  # stride of each column in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows, n_groups)
    RSTD_row_stride,  # stride of each row in rstd
    RSTD_col_stride,  # stride of each column in rstd
    W_ptr,  # pointer to weights, shape (n_groups)
    B_ptr,  # pointer to bias, shape (n_groups)
    hidden_size,
    num_channels,
    num_rows,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    """
    References:
    https://nn.labml.ai/normalization/group_norm/index.html
    """
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)

    X_ptr += batch_idx * X_row_stride + group_idx * X_col_stride
    Y_ptr += batch_idx * Y_row_stride + group_idx * Y_col_stride
    
    # Compute mean
    sum = 0.0
    for i in range(0, hidden_size, BLOCK_SIZE):
        hidden_size_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = hidden_size_offsets < hidden_size
        X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=0.0)
        sum += tl.sum(X)
    
    mean = sum / hidden_size
    tl.store(Mean_ptr + batch_idx * Mean_row_stride + group_idx * Mean_col_stride, mean)
    
    # Compute variance
    variance = 0.0
    for i in range(0, hidden_size, BLOCK_SIZE):
        hidden_size_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = hidden_size_offsets < hidden_size
        X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=0.0)
        diff = X - mean
        variance += tl.sum(diff * diff)
    
    variance = variance / (hidden_size)
    std = tl.sqrt(variance + eps)

    tl.store(RSTD_ptr + batch_idx * RSTD_row_stride + group_idx * RSTD_col_stride, variance)

    for i in range(0, hidden_size, BLOCK_SIZE):
        hidden_size_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = hidden_size_offsets < hidden_size
        X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=0.0)
        Y = (X - mean) / std
        tl.store(Y_ptr + hidden_size_offsets, Y, mask=mask)

        
@triton.jit
def _group_norm_backward_kernel(
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    W_ptr,  # pointer to weights, shape (n_cols,)
    Mean_ptr,  # pointer to mean, shape (n_rows,)
    RSTD_ptr,  # pointer to rstd, shape (n_rows,)
    DX_ptr,  # pointer to input grad, shape (n_rows, n_cols)
    DW_ptr,  # pointer to weights grad, shape (n_cols,)
    DB_ptr,  # pointer to bias grad, shape (n_cols,)
    DY_ptr,  # pointer to output grad, shape (n_rows, n_cols)
    stride_x,  # stride of each row in input
    stride_dx,  # stride of each row in input grad
    stride_dw,  # stride of each row in weights grad
    stride_db,  # stride of each row in bias grad
    stride_dy,  # stride of each row in output grad
    n_rows,
    n_cols,
    rows_per_program: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """
    References:
    https://nn.labml.ai/normalization/group_norm/index.html
    """
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    dw_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    db_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    X_ptr += row_start * stride_x
    Mean_ptr += row_start
    RSTD_ptr += row_start
    DX_ptr += row_start * stride_dx
    DY_ptr += row_start * stride_dy

    for _ in range(row_start, row_end):
        x = tl.load(X_ptr + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0)
        dy = tl.load(DY_ptr + cols, mask=mask, other=0.0)
        mean = tl.load(Mean_ptr)
        rstd = tl.load(RSTD_ptr)

        x_hat = (x - mean) * rstd
        wdy = w * dy
        c1 = tl.sum(x_hat * wdy, axis=0) / n_cols
        c2 = tl.sum(wdy, axis=0) / n_cols
        dx = (wdy - (x_hat * c1 + c2)) * rstd
        tl.store(DX_ptr + cols, dx.to(dtype), mask=mask)

        dw_row += dy * x_hat
        db_row += dy

        X_ptr += stride_x
        Mean_ptr += 1
        RSTD_ptr += 1
        DX_ptr += stride_dx
        DY_ptr += stride_dy

    tl.store(DW_ptr + row_block_id * stride_dw + cols, dw_row.to(dtype), mask=mask)
    tl.store(DB_ptr + row_block_id * stride_db + cols, db_row.to(dtype), mask=mask)


def group_norm_forward(X, num_channels, num_groups, W, B, eps):
    shape = X.shape
    batch_size = shape[0]
    # Reshape X so that the mean and std are computed across the groups
    X = X.view(batch_size, num_groups, -1)
    hidden_size = X.shape[-1]
    BLOCK_SIZE, num_warps = calculate_settings(hidden_size)
    Y = torch.empty((batch_size, num_groups, hidden_size), dtype=X.dtype, device=X.device)
    Mean = torch.empty((batch_size, num_groups), dtype=X.dtype, device=X.device)
    RSTD = torch.empty((batch_size, num_groups), dtype=X.dtype, device=X.device)
    
    _group_norm_forward_kernel[(batch_size, num_groups)](
        Y,
        Y.stride(0),
        Y.stride(1),
        X,
        X.stride(0),
        X.stride(1),
        Mean,
        Mean.stride(0),
        Mean.stride(1),
        RSTD,
        RSTD.stride(0),
        RSTD.stride(1),
        W,
        B,
        hidden_size,
        num_channels,
        batch_size,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
     
    Y = Y.view(*shape)
    affine_shape = [1] * len(shape)
    affine_shape[1] = num_channels
    Y = Y * W.view(affine_shape) + B.view(affine_shape)

    return Y, X, Mean, RSTD, BLOCK_SIZE, num_warps


def group_norm_backward(dY, X, W, B, Mean, RSTD):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    DX = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    _DW = torch.empty((sm_count, n_cols), dtype=W.dtype, device=W.device)
    _DB = torch.empty((sm_count, n_cols), dtype=W.dtype, device=W.device)

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    if n_cols > BLOCK_SIZE:
        raise RuntimeError("This group norm doesn't support feature dim >= 64KB.")

    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count,)
    triton_dtype = tl.float32 if X.dtype == torch.float32 else tl.bfloat16
    _group_norm_backward_kernel[grid](
        X,
        W,
        Mean,
        RSTD,
        DX,
        _DW,
        _DB,
        dY,
        X.stride(0),
        DX.stride(0),
        _DW.stride(0),
        _DB.stride(0),
        dY.stride(0),
        n_rows,
        n_cols,
        rows_per_program,
        BLOCK_SIZE=BLOCK_SIZE,
        dtype=triton_dtype,
    )

    DW = _DW.sum(dim=0).to(W.dtype)
    DB = _DB.sum(dim=0).to(W.dtype)

    DX = DX.view(*shape)
    return DX, DW, DB


class LigerGroupNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, num_channels, num_groups, affine_scaling_weight, affine_shifting_bias, eps):
        Y, X, Mean, RSTD, BLOCK_SIZE, num_warps = group_norm_forward(X, num_channels, num_groups, affine_scaling_weight, affine_shifting_bias, eps)
        ctx.save_for_backward(X, affine_scaling_weight, affine_shifting_bias, Mean, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X, W, B, Mean, RSTD = ctx.saved_tensors
        DX, DW, DB = group_norm_backward(dY, X, W, B, Mean, RSTD)
        return DX, DW, DB, None
