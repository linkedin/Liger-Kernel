import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import ensure_contiguous

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
def _layer_norm_forward_kernel(
    Y_ptr,  # pointer to output, shape (n_rows, n_cols)
    Y_row_stride,  # stride of each row in output
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    X_row_stride,  # stride of each row in input
    W_ptr,  # pointer to weights, shape (n_cols,)
    W_row_stride,  # stride of each row in weights
    B_ptr,  # pointer to bias, shape (n_cols,)
    B_row_stride,  # stride of each row in bias
    Mean_ptr,  # pointer to mean, shape (n_rows,)
    Mean_row_stride,  # stride of each row in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows,)
    RSTD_row_stride,  # stride of each row in rstd
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Pre-load weights and bias in fp32 to avoid repeated conversions
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0.0)
    W_f32 = W_row.to(tl.float32)
    B_f32 = B_row.to(tl.float32)

    # Calculate pointers for this row
    row_X_ptr = X_ptr + row_idx * X_row_stride
    row_Y_ptr = Y_ptr + row_idx * Y_row_stride
    row_Mean_ptr = Mean_ptr + row_idx * Mean_row_stride
    row_RSTD_ptr = RSTD_ptr + row_idx * RSTD_row_stride

    # Load input data and convert to fp32 for numerical stability
    X_row = tl.load(row_X_ptr + col_offsets, mask=mask, other=0.0)
    X_f32 = X_row.to(tl.float32)

    # Compute statistics in fp32 for numerical stability
    n_cols_f32 = n_cols.to(tl.float32)
    mean = tl.sum(X_f32, axis=0) / n_cols_f32
    X_centered = X_f32 - mean
    # Apply mask to variance calculation to exclude contributions from masked elements
    X_centered_masked = tl.where(mask, X_centered, 0.0)
    var = tl.sum(X_centered_masked * X_centered_masked, axis=0) / n_cols_f32
    rstd = rsqrt(var + eps)

    # Store statistics (convert back to original dtype only once)
    tl.store(row_Mean_ptr, mean.to(X_row.dtype))
    tl.store(row_RSTD_ptr, rstd.to(X_row.dtype))

    # Fused normalization and affine transformation
    # Y = (X - mean) * rstd * W + B = X_centered * rstd * W + B
    Y_f32 = X_centered * rstd * W_f32 + B_f32

    # Store output (single conversion back to original dtype)
    tl.store(row_Y_ptr + col_offsets, Y_f32.to(X_row.dtype), mask=mask)


@triton.jit
def _layer_norm_backward_kernel(
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
    stride_dy,  # stride of each row in output grad
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr,
    atomic_dtype: tl.constexpr,
):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    """
    row_idx = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # Pre-load weights once (same optimization as forward pass)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    w_f32 = w.to(tl.float32)
    n_cols_f32 = n_cols.to(tl.float32)

    # Calculate pointers for this specific row
    row_X_ptr = X_ptr + row_idx * stride_x
    row_DX_ptr = DX_ptr + row_idx * stride_dx
    row_DY_ptr = DY_ptr + row_idx * stride_dy
    row_Mean_ptr = Mean_ptr + row_idx
    row_RSTD_ptr = RSTD_ptr + row_idx

    # Load data for this row
    x = tl.load(row_X_ptr + cols, mask=mask, other=0.0)
    dy = tl.load(row_DY_ptr + cols, mask=mask, other=0.0)
    mean = tl.load(row_Mean_ptr)
    rstd = tl.load(row_RSTD_ptr)

    # Convert to fp32 for numerical stability
    x_f32 = x.to(tl.float32)
    dy_f32 = dy.to(tl.float32)
    mean_f32 = mean.to(tl.float32)
    rstd_f32 = rstd.to(tl.float32)

    # Compute backward pass for this row
    x_hat = (x_f32 - mean_f32) * rstd_f32
    wdy = w_f32 * dy_f32
    c1 = tl.sum(x_hat * wdy, axis=0) / n_cols_f32
    c2 = tl.sum(wdy, axis=0) / n_cols_f32
    dx = (wdy - (x_hat * c1 + c2)) * rstd_f32

    # Store input gradient
    tl.store(row_DX_ptr + cols, dx.to(dtype), mask=mask)

    # Accumulate weight and bias gradients using atomic operations
    dw = dy_f32 * x_hat
    db = dy_f32
    tl.atomic_add(DW_ptr + cols, dw.to(atomic_dtype), mask=mask)
    tl.atomic_add(DB_ptr + cols, db.to(atomic_dtype), mask=mask)


def layer_norm_forward(X, W, B, eps):
    """
    Args:
        X: Input tensor of shape (..., hidden_size)
        W: Weight tensor of shape (hidden_size,)
        B: Bias tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        Tuple of (output, input, mean, rstd, block_size, num_warps)
    """
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape

    # Calculate optimal block size and warp configuration
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    # Allocate output tensors
    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    Mean = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    RSTD = torch.empty(n_rows, dtype=X.dtype, device=X.device)

    # Validate input dimensions
    if X.shape[1] != W.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: input feature size (X.shape[1]={X.shape[1]}) "
            f"must match weight size (W.shape[0]={W.shape[0]})"
        )

    # XPU-specific optimization
    kernel_args = {}
    if X.device.type == "xpu":
        kernel_args["grf_mode"] = "large"

    # Launch kernel with one thread block per row for optimal performance
    grid = (n_rows,)
    _layer_norm_forward_kernel[grid](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        W,
        W.stride(0),
        B,
        B.stride(0),
        Mean,
        Mean.stride(0),
        RSTD,
        RSTD.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        **kernel_args,
    )

    return Y.view(*shape), X, Mean, RSTD, BLOCK_SIZE, num_warps


def layer_norm_backward(dY, X, W, B, Mean, RSTD):
    """
    Args:
        dY: Gradient of output
        X: Input tensor
        W: Weight tensor
        B: Bias tensor
        Mean: Pre-computed mean
        RSTD: Pre-computed reciprocal standard deviation

    Returns:
        Tuple of (input_grad, weight_grad, bias_grad)
    """
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    # Allocate gradient tensors
    DX = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    # Use float32 for weight/bias gradients if bfloat16 (due to atomic_add limitation)
    grad_dtype = torch.float32 if W.dtype == torch.bfloat16 else W.dtype
    DW = torch.zeros(n_cols, dtype=grad_dtype, device=W.device)
    DB = torch.zeros(n_cols, dtype=grad_dtype, device=W.device)

    # Calculate optimal block size and warp configuration
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    if n_cols > BLOCK_SIZE:
        raise RuntimeError(f"Feature dimension {n_cols} exceeds maximum supported size of {BLOCK_SIZE}.")

    # Determine dtype for triton operations
    triton_dtype = (
        tl.float32
        if X.dtype == torch.float32
        else tl.bfloat16
        if X.dtype == torch.bfloat16
        else tl.float16
        if X.dtype == torch.float16
        else tl.float32  # fallback
    )

    # Use float32 for atomic operations if bfloat16 is not supported
    atomic_dtype = tl.float32 if triton_dtype == tl.bfloat16 else triton_dtype

    kernel_args = {"num_warps": num_warps}
    # XPU-specific optimization
    if X.device.type == "xpu":
        kernel_args.update({"grf_mode": "large", "num_warps": 32, "num_stages": 4})

    # Launch kernel with one thread block per row for optimal performance
    grid = (n_rows,)
    _layer_norm_backward_kernel[grid](
        X,
        W,
        Mean,
        RSTD,
        DX,
        DW,
        DB,
        dY,
        X.stride(0),
        DX.stride(0),
        dY.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        dtype=triton_dtype,
        atomic_dtype=atomic_dtype,
        **kernel_args,
    )

    DX = DX.view(*shape)
    return DX, DW.to(W.dtype), DB.to(W.dtype)


class LigerLayerNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, B, eps):
        Y, X, Mean, RSTD, BLOCK_SIZE, num_warps = layer_norm_forward(X, W, B, eps)
        ctx.save_for_backward(X, W, B, Mean, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X, W, B, Mean, RSTD = ctx.saved_tensors
        DX, DW, DB = layer_norm_backward(dY, X, W, B, Mean, RSTD)
        return DX, DW, DB, None
