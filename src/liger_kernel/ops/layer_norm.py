import math
import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import device_context
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count
from liger_kernel.ops.utils import set_large_grf_mode
from liger_kernel.utils import is_npu_available

_BLOCK_ROW_MAX_BLOCK_SIZE = 256
_BLOCK_ROW_MIN_ROWS = 4096
_BLOCK_ROW_ELEMENTS = 4096
_BLOCK_ROW_MAX_ROWS = 64
_BLOCK_ROW_BWD_PROGRAMS_PER_SM = 4

if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
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
    mean = tl.sum(X_f32, axis=0) / n_cols
    X_centered = X_f32 - mean
    # Apply mask to variance calculation to exclude contributions from masked elements
    X_centered_masked = tl.where(mask, X_centered, 0.0)
    var = tl.sum(X_centered_masked * X_centered_masked, axis=0) / n_cols
    rstd = rsqrt(var + tl.cast(eps, tl.float32))

    # Preserve the fp32 statistics used by forward for the backward pass.
    tl.store(row_Mean_ptr, mean)
    tl.store(row_RSTD_ptr, rstd)

    # Fused normalization and affine transformation
    # Y = (X - mean) * rstd * W + B = X_centered * rstd * W + B
    Y_f32 = X_centered * rstd * W_f32 + B_f32

    # Store output (single conversion back to original dtype)
    tl.store(row_Y_ptr + col_offsets, Y_f32.to(X_row.dtype), mask=mask)


@triton.jit
def _layer_norm_backward_kernel(
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    stride_x,  # stride of each row in input
    W_ptr,  # pointer to weights, shape (n_cols,)
    Mean_ptr,  # pointer to mean, shape (n_rows,)
    stride_mean,  # stride of each row in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows,)
    stride_rstd,  # stride of each row in rstd
    DX_ptr,  # pointer to input grad, shape (n_rows, n_cols)
    stride_dx,  # stride of each row in input grad
    DW_ptr,  # pointer to weights grad, shape (n_cols,)
    stride_dw,  # stride of each row in weights grad
    DB_ptr,  # pointer to bias grad, shape (n_cols,)
    stride_db,  # stride of each row in bias grad
    DY_ptr,  # pointer to output grad, shape (n_rows, n_cols)
    stride_dy,  # stride of each row in output grad
    n_rows,
    n_cols,
    rows_per_program: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_DW: tl.constexpr = True,
    COMPUTE_DB: tl.constexpr = True,
):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    """
    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    if COMPUTE_DW:
        dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    if COMPUTE_DB:
        db_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Pre-load weights once (same optimization as forward pass)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    w_f32 = w.to(tl.float32)

    for row_idx in range(row_start, row_end):
        # Calculate pointers for this specific row
        row_X_ptr = X_ptr + row_idx * stride_x
        row_DX_ptr = DX_ptr + row_idx * stride_dx
        row_DY_ptr = DY_ptr + row_idx * stride_dy
        row_Mean_ptr = Mean_ptr + row_idx * stride_mean
        row_RSTD_ptr = RSTD_ptr + row_idx * stride_rstd

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
        c1 = tl.sum(x_hat * wdy, axis=0) / n_cols
        c2 = tl.sum(wdy, axis=0) / n_cols
        dx = (wdy - (x_hat * c1 + c2)) * rstd_f32

        # Store input gradient
        tl.store(row_DX_ptr + cols, dx, mask=mask)

        # Accumulate weight and bias gradients for this thread block's assigned rows
        if COMPUTE_DW:
            dw = dy_f32 * x_hat
            dW_row += dw
        if COMPUTE_DB:
            db = dy_f32
            db_row += db

    if COMPUTE_DW:
        tl.store(DW_ptr + row_block_id * stride_dw + cols, dW_row, mask=mask)
    if COMPUTE_DB:
        tl.store(DB_ptr + row_block_id * stride_db + cols, db_row, mask=mask)


@triton.jit
def _block_layer_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    B_ptr,
    B_row_stride,
    Mean_ptr,
    Mean_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    """
    Block-row LayerNorm forward kernel: tiles BLOCK_ROW rows per program for narrow feature widths
    (BLOCK_SIZE <= 256) with many rows.
    """
    row_idx = tl.program_id(0).to(tl.int64) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = row_idx < n_rows
    col_mask = col_offsets < n_cols
    mask = row_mask[:, None] & col_mask[None, :]

    W_f32 = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    B_f32 = tl.load(B_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    X_row = tl.load(
        X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
        mask=mask,
        other=0.0,
    )
    X_f32 = X_row.to(tl.float32)

    mean = tl.sum(X_f32, axis=1) / n_cols
    X_centered = X_f32 - mean[:, None]
    X_centered_masked = tl.where(mask, X_centered, 0.0)
    var = tl.sum(X_centered_masked * X_centered_masked, axis=1) / n_cols
    rstd = rsqrt(var + tl.cast(eps, tl.float32))

    tl.store(Mean_ptr + row_idx * Mean_row_stride, mean, mask=row_mask)
    tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd, mask=row_mask)

    Y_f32 = X_centered * rstd[:, None] * W_f32[None, :] + B_f32[None, :]

    tl.store(
        Y_ptr + row_idx[:, None] * Y_row_stride + col_offsets[None, :],
        Y_f32.to(X_row.dtype),
        mask=mask,
    )


@triton.jit
def _block_layer_norm_backward_kernel(
    X_ptr,
    stride_x,
    W_ptr,
    Mean_ptr,
    stride_mean,
    RSTD_ptr,
    stride_rstd,
    DX_ptr,
    stride_dx,
    DW_ptr,
    stride_dw,
    DB_ptr,
    stride_db,
    DY_ptr,
    stride_dy,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    COMPUTE_DW: tl.constexpr = True,
    COMPUTE_DB: tl.constexpr = True,
):
    """
    Block-row persistent LayerNorm backward kernel for narrow feature widths (BLOCK_SIZE <= 256).
    """
    pid = tl.program_id(0).to(tl.int64)
    num_programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_SIZE)
    col_mask = cols < n_cols

    if COMPUTE_DW:
        dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    if COMPUTE_DB:
        db_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    w_f32 = tl.load(W_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    for start in range(pid * BLOCK_ROW, n_rows, num_programs * BLOCK_ROW):
        row_idx = start + tl.arange(0, BLOCK_ROW)
        row_mask = row_idx < n_rows
        mask = row_mask[:, None] & col_mask[None, :]

        x_f32 = tl.load(X_ptr + row_idx[:, None] * stride_x + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        dy_f32 = tl.load(DY_ptr + row_idx[:, None] * stride_dy + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        mean_f32 = tl.load(Mean_ptr + row_idx * stride_mean, mask=row_mask, other=0.0).to(tl.float32)
        rstd_f32 = tl.load(RSTD_ptr + row_idx * stride_rstd, mask=row_mask, other=0.0).to(tl.float32)

        x_hat = tl.where(mask, (x_f32 - mean_f32[:, None]) * rstd_f32[:, None], 0.0)
        wdy = w_f32[None, :] * dy_f32
        c1 = tl.sum(x_hat * wdy, axis=1) / n_cols
        c2 = tl.sum(wdy, axis=1) / n_cols
        dx = (wdy - (x_hat * c1[:, None] + c2[:, None])) * rstd_f32[:, None]

        tl.store(DX_ptr + row_idx[:, None] * stride_dx + cols[None, :], dx, mask=mask)

        if COMPUTE_DW:
            dW_row += tl.sum(tl.where(mask, dy_f32 * x_hat, 0.0), axis=0)
        if COMPUTE_DB:
            db_row += tl.sum(tl.where(mask, dy_f32, 0.0), axis=0)

    if COMPUTE_DW:
        tl.store(DW_ptr + pid * stride_dw + cols, dW_row, mask=col_mask)
    if COMPUTE_DB:
        tl.store(DB_ptr + pid * stride_db + cols, db_row, mask=col_mask)


def layer_norm_forward(X, W, B, eps, row_mode=None):
    """
    Args:
        X: Input tensor of shape (..., hidden_size)
        W: Weight tensor of shape (hidden_size,)
        B: Bias tensor of shape (hidden_size,)
        eps: Small constant for numerical stability
        row_mode: If truthy, force the single-row-per-program path

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
    Mean = torch.empty(n_rows, dtype=torch.float32, device=X.device)
    RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)

    # Validate input dimensions
    if X.shape[1] != W.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: input feature size (X.shape[1]={X.shape[1]}) "
            f"must match weight size (W.shape[0]={W.shape[0]})"
        )

    # XPU-specific optimization
    kernel_args = {}
    if X.device.type == "xpu":
        set_large_grf_mode(kernel_args)

    with device_context(X.device):
        if BLOCK_SIZE > _BLOCK_ROW_MAX_BLOCK_SIZE or n_rows < _BLOCK_ROW_MIN_ROWS or row_mode:
            _layer_norm_forward_kernel[(n_rows,)](
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
        else:
            BLOCK_ROW = min(_BLOCK_ROW_MAX_ROWS, _BLOCK_ROW_ELEMENTS // max(1, BLOCK_SIZE))
            _block_layer_norm_forward_kernel[(triton.cdiv(n_rows, BLOCK_ROW),)](
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
                n_rows,
                n_cols,
                eps,
                BLOCK_SIZE=BLOCK_SIZE,
                BLOCK_ROW=BLOCK_ROW,
                num_warps=num_warps,
                **kernel_args,
            )

    return Y.view(*shape), X, Mean, RSTD, BLOCK_SIZE, num_warps


def layer_norm_backward(dY, X, W, B, Mean, RSTD, row_mode=None, needs_dw=None, needs_db=None):
    """
    Args:
        dY: Gradient of output
        X: Input tensor
        W: Weight tensor
        B: Bias tensor
        Mean: Pre-computed mean
        RSTD: Pre-computed reciprocal standard deviation
        row_mode: If truthy, force the single-row-per-program path
        needs_dw: Whether to compute gradient of weight
        needs_db: Whether to compute gradient of bias

    Returns:
        Tuple of (input_grad, weight_grad, bias_grad)
    """
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    compute_dw = True if needs_dw is None else needs_dw
    compute_db = True if needs_db is None else needs_db

    sm_count = 1
    if X.device.type == "cuda":
        sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    elif X.device.type == "xpu":
        sm_count = torch.xpu.get_device_properties(X.device).gpu_eu_count
    elif X.device.type == "npu":
        sm_count = get_npu_core_count()

    # Calculate optimal block size and warp configuration
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    if n_cols > BLOCK_SIZE:
        raise RuntimeError(f"Feature dimension {n_cols} exceeds maximum supported size of {BLOCK_SIZE}.")

    use_single_row = BLOCK_SIZE > _BLOCK_ROW_MAX_BLOCK_SIZE or n_rows < _BLOCK_ROW_MIN_ROWS or bool(row_mode)
    if use_single_row:
        n_programs = sm_count
    else:
        BLOCK_ROW = min(_BLOCK_ROW_MAX_ROWS, _BLOCK_ROW_ELEMENTS // max(1, BLOCK_SIZE))
        n_programs = max(1, min(sm_count * _BLOCK_ROW_BWD_PROGRAMS_PER_SM, triton.cdiv(n_rows, BLOCK_ROW)))

    # fp32 for numerical stability especially.
    _DW = torch.empty((n_programs, n_cols), dtype=torch.float32, device=W.device) if compute_dw else None
    _DB = torch.empty((n_programs, n_cols), dtype=torch.float32, device=W.device) if compute_db else None

    # Allocate gradient tensors
    DX = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)

    kernel_args = {"num_warps": num_warps}
    # XPU-specific optimization
    if X.device.type == "xpu":
        kernel_args.update({"num_warps": 32, "num_stages": 4})
        set_large_grf_mode(kernel_args)

    with device_context(X.device):
        if use_single_row:
            rows_per_program = math.ceil(n_rows / sm_count)
            _layer_norm_backward_kernel[(n_programs,)](
                X,
                X.stride(0),
                W,
                Mean,
                Mean.stride(0),
                RSTD,
                RSTD.stride(0),
                DX,
                DX.stride(0),
                _DW,
                _DW.stride(0) if compute_dw else 0,
                _DB,
                _DB.stride(0) if compute_db else 0,
                dY,
                dY.stride(0),
                n_rows,
                n_cols,
                rows_per_program=rows_per_program,
                BLOCK_SIZE=BLOCK_SIZE,
                COMPUTE_DW=compute_dw,
                COMPUTE_DB=compute_db,
                **kernel_args,
            )
        else:
            _block_layer_norm_backward_kernel[(n_programs,)](
                X,
                X.stride(0),
                W,
                Mean,
                Mean.stride(0),
                RSTD,
                RSTD.stride(0),
                DX,
                DX.stride(0),
                _DW,
                _DW.stride(0) if compute_dw else 0,
                _DB,
                _DB.stride(0) if compute_db else 0,
                dY,
                dY.stride(0),
                n_rows,
                n_cols,
                BLOCK_SIZE=BLOCK_SIZE,
                BLOCK_ROW=BLOCK_ROW,
                COMPUTE_DW=compute_dw,
                COMPUTE_DB=compute_db,
                **kernel_args,
            )

    DX = DX.view(*shape)
    DW = _DW.sum(dim=0).to(W.dtype) if compute_dw else None
    DB = _DB.sum(dim=0).to(B.dtype) if compute_db else None

    return DX, DW, DB


class LigerLayerNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, B, eps, row_mode=None):
        Y, X, Mean, RSTD, BLOCK_SIZE, num_warps = layer_norm_forward(X, W, B, eps, row_mode=row_mode)
        ctx.row_mode = row_mode
        ctx.save_for_backward(X, W, B, Mean, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X, W, B, Mean, RSTD = ctx.saved_tensors
        needs_dw = ctx.needs_input_grad[1]
        needs_db = ctx.needs_input_grad[2]
        DX, DW, DB = layer_norm_backward(
            dY, X, W, B, Mean, RSTD, row_mode=ctx.row_mode, needs_dw=needs_dw, needs_db=needs_db
        )
        return DX, DW, DB, None, None
