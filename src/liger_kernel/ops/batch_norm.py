import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import ensure_contiguous

# Import rsqrt function based on Triton version
if compare_version("triton", operator.ge, "3.0.0"):
    try:
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import rsqrt
else:
    from triton.language.math import rsqrt


@triton.jit
def _batch_norm_forward_kernel(
    Y_ptr,  # pointer to output, shape (N, C)
    Y_row_stride,  # stride between rows in output (usually C)
    X_ptr,  # pointer to input, shape (N, C)
    X_row_stride,  # stride between rows in input
    gamma_ptr,  # pointer to scale, shape (C,)
    beta_ptr,  # pointer to bias, shape (C,)
    mean_ptr,  # pointer to mean, shape (C,)
    rstd_ptr,  # pointer to rstd, shape (C,)
    n_rows: tl.constexpr,  # batch size N
    n_channels: tl.constexpr,  # feature dim C
    eps,  # small constant
    BLOCK_SIZE: tl.constexpr,  # the number of rows processed in each block (for vectorization)
):
    """
    BatchNorm Forward kernel
    Each program instance handles one channel (i.e., feature c),
    and performs reduction on all batch elements (rows) of that channel to compute mean, variance, and produce normalized output.
    """
    # Each program instance processes one channel
    channel_idx = tl.program_id(0)
    if channel_idx >= n_channels:
        return

    # --- First pass: compute mean and variance ---
    sum_val = tl.zeros([], dtype=tl.float32)
    sum_sq_val = tl.zeros([], dtype=tl.float32)
    for row_offset in range(0, n_rows, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < (n_rows - row_offset)
        # Element address: X_ptr + (row_offset + offset)*X_row_stride + channel_idx
        base_ptr = X_ptr + (row_offset * X_row_stride) + channel_idx
        x_block = tl.load(base_ptr + offsets * X_row_stride, mask=mask, other=0.0)
        sum_val += tl.sum(x_block, axis=0)
        sum_sq_val += tl.sum(x_block * x_block, axis=0)
    mean = sum_val / n_rows
    var = sum_sq_val / n_rows - mean * mean
    rstd = rsqrt(var + eps)
    tl.store(mean_ptr + channel_idx, mean)
    tl.store(rstd_ptr + channel_idx, rstd)

    # --- Second pass: compute normalized output ---
    gamma_val = tl.load(gamma_ptr + channel_idx)
    beta_val = tl.load(beta_ptr + channel_idx)
    for row_offset in range(0, n_rows, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < (n_rows - row_offset)
        base_ptr_in = X_ptr + (row_offset * X_row_stride) + channel_idx
        x_block = tl.load(base_ptr_in + offsets * X_row_stride, mask=mask, other=0.0)
        # Normalization: xhat = (x - mean)*rstd
        y_val = gamma_val * ((x_block - mean) * rstd) + beta_val
        base_ptr_out = Y_ptr + (row_offset * Y_row_stride) + channel_idx
        tl.store(base_ptr_out + offsets * Y_row_stride, y_val, mask=mask)


@triton.jit
def _batch_norm_backward_kernel(
    X_ptr,  # pointer to input X, shape (N, C)
    dY_ptr,  # pointer to upstream gradient dY, shape (N, C)
    DX_ptr,  # pointer to output gradient dX, shape (N, C)
    gamma_ptr,  # pointer to scale, shape (C,)
    mean_ptr,  # pointer to mean, shape (C,)
    rstd_ptr,  # pointer to rstd, shape (C,)
    dgamma_ptr,  # pointer to dgamma, shape (C,)
    dbeta_ptr,  # pointer to dbeta, shape (C,)
    n_rows: tl.constexpr,  # batch size
    n_channels: tl.constexpr,  # feature dim C
    BLOCK_SIZE: tl.constexpr,  # the number of rows processed in each block
    stride_x,  # stride between rows in X (usually C)
    stride_dy,  # stride between rows in dY
    stride_dx,  # stride between rows in dX
):
    """
    BatchNorm Backward kernel
    Each program instance processes one channel, performing two passes over the batch:
    The first pass computes dgamma and dbeta;
    The second pass computes dX.
    """
    channel_idx = tl.program_id(0)
    if channel_idx >= n_channels:
        return

    gamma_val = tl.load(gamma_ptr + channel_idx)
    mean_val = tl.load(mean_ptr + channel_idx)
    rstd_val = tl.load(rstd_ptr + channel_idx)

    # --- First pass: compute dgamma and dbeta ---
    dgamma_acc = tl.zeros([], dtype=tl.float32)
    dbeta_acc = tl.zeros([], dtype=tl.float32)
    for row_offset in range(0, n_rows, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < (n_rows - row_offset)
        base_ptr_dy = dY_ptr + (row_offset * stride_dy) + channel_idx
        dy_block = tl.load(base_ptr_dy + offsets * stride_dy, mask=mask, other=0.0)
        base_ptr_x = X_ptr + (row_offset * stride_x) + channel_idx
        x_block = tl.load(base_ptr_x + offsets * stride_x, mask=mask, other=0.0)
        # Compute xhat = (x - mean)*rstd
        xhat_block = (x_block - mean_val) * rstd_val
        dgamma_acc += tl.sum(dy_block * xhat_block, axis=0)
        dbeta_acc += tl.sum(dy_block, axis=0)
    tl.store(dgamma_ptr + channel_idx, dgamma_acc)
    tl.store(dbeta_ptr + channel_idx, dbeta_acc)

    # --- Second pass: compute dX ---
    # Note: since n_rows is constexpr, we can convert it to float for division
    N_float = float(n_rows)
    for row_offset in range(0, n_rows, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < (n_rows - row_offset)
        base_ptr_dy = dY_ptr + (row_offset * stride_dy) + channel_idx
        dy_block = tl.load(base_ptr_dy + offsets * stride_dy, mask=mask, other=0.0)
        base_ptr_x = X_ptr + (row_offset * stride_x) + channel_idx
        x_block = tl.load(base_ptr_x + offsets * stride_x, mask=mask, other=0.0)
        xhat_block = (x_block - mean_val) * rstd_val
        # dx = gamma * rstd * [dy - dbeta/N - xhat*(dgamma/N)]
        dx_block = gamma_val * rstd_val * (dy_block - (dbeta_acc / N_float) - xhat_block * (dgamma_acc / N_float))
        base_ptr_dx = DX_ptr + (row_offset * stride_dx) + channel_idx
        tl.store(base_ptr_dx + offsets * stride_dx, dx_block, mask=mask)


def batch_norm_forward(X, gamma, beta, eps):
    """
    Forward pass:
      X: shape (N, C)
      gamma, beta: shape (C,)
    Returns:
      Y, as well as intermediate variables X, Mean, RSTD for backward pass
    """
    shape = X.shape
    assert len(shape) == 2, "Currently, BatchNorm only supports 2D input (N, C)"
    n_rows, n_channels = shape
    # Choose BLOCK_SIZE based on the dimension of the batch
    BLOCK_SIZE, num_warps = calculate_settings(n_rows)
    Y = torch.empty((n_rows, n_channels), dtype=X.dtype, device=X.device)
    # Mean and rstd saved per channel
    Mean = torch.empty(n_channels, dtype=X.dtype, device=X.device)
    RSTD = torch.empty(n_channels, dtype=X.dtype, device=X.device)
    # Check gamma shape
    assert gamma.shape[0] == n_channels, "gamma dimension should match input feature dimension"
    grid = (n_channels,)
    _batch_norm_forward_kernel[grid](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        gamma,
        beta,
        Mean,
        RSTD,
        n_rows,
        n_channels,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return Y, X, Mean, RSTD, BLOCK_SIZE, num_warps


def batch_norm_backward(dY, X, gamma, beta, Mean, RSTD):
    """
    Backward pass:
      dY: upstream gradient, shape (N, C)
    Returns:
      dX, dgamma, dbeta
    """
    shape = dY.shape
    assert len(shape) == 2, "Currently, BatchNorm only supports 2D input (N, C)"
    n_rows, n_channels = shape
    DX = torch.empty((n_rows, n_channels), dtype=X.dtype, device=X.device)
    # dgamma, dbeta are both (C,)
    dgamma = torch.empty(n_channels, dtype=gamma.dtype, device=gamma.device)
    dbeta = torch.empty(n_channels, dtype=gamma.dtype, device=gamma.device)
    BLOCK_SIZE, num_warps = calculate_settings(n_rows)
    grid = (n_channels,)
    _batch_norm_backward_kernel[grid](
        X,
        dY,
        DX,
        gamma,
        Mean,
        RSTD,
        dgamma,
        dbeta,
        n_rows,
        n_channels,
        BLOCK_SIZE=BLOCK_SIZE,
        stride_x=X.stride(0),
        stride_dy=dY.stride(0),
        stride_dx=DX.stride(0),
    )
    return DX, dgamma, dbeta


class LigerBatchNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, gamma, beta, eps):
        Y, X_saved, Mean, RSTD, BLOCK_SIZE, num_warps = batch_norm_forward(X, gamma, beta, eps)
        ctx.save_for_backward(X_saved, gamma, beta, Mean, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X, gamma, beta, Mean, RSTD = ctx.saved_tensors
        DX, dgamma, dbeta = batch_norm_backward(dY, X, gamma, beta, Mean, RSTD)
        return DX, dgamma, dbeta, None

