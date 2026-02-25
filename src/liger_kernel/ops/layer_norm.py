import math
import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count
from liger_kernel.ops.utils import set_large_grf_mode
from liger_kernel.utils import is_npu_available

if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
    try:
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import rsqrt
else:
    from triton.language.math import rsqrt


@triton.jit
def _layer_norm_forward_kernel(
    Y_ptr, Y_row_stride,
    X_ptr, X_row_stride,
    W_ptr, W_row_stride,
    B_ptr, B_row_stride,
    Mean_ptr, Mean_row_stride,
    RSTD_ptr, RSTD_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    col_offsets = tl.multiple_of(col_offsets, BLOCK_SIZE)
    mask = col_offsets < n_cols

    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0.0)
    W_f32 = W_row.to(tl.float32)
    B_f32 = B_row.to(tl.float32)

    row_X_ptr = X_ptr + row_idx * X_row_stride
    row_Y_ptr = Y_ptr + row_idx * Y_row_stride
    row_Mean_ptr = Mean_ptr + row_idx * Mean_row_stride
    row_RSTD_ptr = RSTD_ptr + row_idx * RSTD_row_stride

    X_row = tl.load(row_X_ptr + col_offsets, mask=mask, other=0.0)
    X_f32 = X_row.to(tl.float32)

    mean = tl.sum(X_f32, axis=0) / n_cols
    X_centered = X_f32 - mean
    X_centered_masked = tl.where(mask, X_centered, 0.0)
    var = tl.sum(X_centered_masked * X_centered_masked, axis=0) / n_cols
    rstd = rsqrt(var + eps)

    tl.store(row_Mean_ptr, mean.to(X_row.dtype))
    tl.store(row_RSTD_ptr, rstd.to(X_row.dtype))

    Y_f32 = X_centered * rstd * W_f32 + B_f32
    tl.store(row_Y_ptr + col_offsets, Y_f32.to(X_row.dtype), mask=mask)


@triton.jit
def _layer_norm_backward_kernel(
    X_ptr, stride_x,
    W_ptr,
    Mean_ptr, stride_mean,
    RSTD_ptr, stride_rstd,
    DX_ptr, stride_dx,
    DW_ptr,  # shape (n_cols,) — atomic accumulation target
    DB_ptr,  # shape (n_cols,) — atomic accumulation target
    DY_ptr, stride_dy,
    n_rows, n_cols,
    rows_per_program: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    cols = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    cols = tl.multiple_of(cols, BLOCK_SIZE)
    mask = cols < n_cols

    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    db_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    w_f32 = w.to(tl.float32)

    for row_idx in range(row_start, row_end):
        row_X_ptr = X_ptr + row_idx * stride_x
        row_DX_ptr = DX_ptr + row_idx * stride_dx
        row_DY_ptr = DY_ptr + row_idx * stride_dy
        row_Mean_ptr = Mean_ptr + row_idx * stride_mean
        row_RSTD_ptr = RSTD_ptr + row_idx * stride_rstd

        x = tl.load(row_X_ptr + cols, mask=mask, other=0.0)
        dy = tl.load(row_DY_ptr + cols, mask=mask, other=0.0)
        mean = tl.load(row_Mean_ptr)
        rstd = tl.load(row_RSTD_ptr)

        x_f32 = x.to(tl.float32)
        dy_f32 = dy.to(tl.float32)
        mean_f32 = mean.to(tl.float32)
        rstd_f32 = rstd.to(tl.float32)

        x_hat = (x_f32 - mean_f32) * rstd_f32
        wdy = w_f32 * dy_f32
        c1 = tl.sum(x_hat * wdy, axis=0) / n_cols
        c2 = tl.sum(wdy, axis=0) / n_cols
        dx = (wdy - (x_hat * c1 + c2)) * rstd_f32

        tl.store(row_DX_ptr + cols, dx, mask=mask)

        dw = dy_f32 * x_hat
        db = dy_f32
        dW_row += dw
        db_row += db

    # Atomic add partial DW/DB directly to the output — eliminates post-kernel reduction
    tl.atomic_add(DW_ptr + cols, dW_row, mask=mask)
    tl.atomic_add(DB_ptr + cols, db_row, mask=mask)


def layer_norm_forward(X, W, B, eps):
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    Mean = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    RSTD = torch.empty(n_rows, dtype=X.dtype, device=X.device)

    if X.shape[1] != W.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: input feature size (X.shape[1]={X.shape[1]}) "
            f"must match weight size (W.shape[0]={W.shape[0]})"
        )

    kernel_args = {}
    if X.device.type == "xpu":
        set_large_grf_mode(kernel_args)

    grid = (n_rows,)
    _layer_norm_forward_kernel[grid](
        Y, Y.stride(0),
        X, X.stride(0),
        W, W.stride(0),
        B, B.stride(0),
        Mean, Mean.stride(0),
        RSTD, RSTD.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        **kernel_args,
    )

    return Y.view(*shape), X, Mean, RSTD, BLOCK_SIZE, num_warps


def layer_norm_backward(dY, X, W, B, Mean, RSTD):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    sm_count = 1
    if X.device.type == "cuda":
        sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    elif X.device.type == "xpu":
        sm_count = torch.xpu.get_device_properties(X.device).gpu_eu_count
    elif X.device.type == "npu":
        sm_count = get_npu_core_count()

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    if n_cols > BLOCK_SIZE:
        raise RuntimeError(f"Feature dimension {n_cols} exceeds maximum supported size of {BLOCK_SIZE}.")
    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count,)

    DX = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    # Atomic accumulation targets — must be zeroed, single (n_cols,) shape
    DW = torch.zeros(n_cols, dtype=torch.float32, device=W.device)
    DB = torch.zeros(n_cols, dtype=torch.float32, device=W.device)

    kernel_args = {"num_warps": num_warps}
    if X.device.type == "xpu":
        kernel_args.update({"num_warps": 32, "num_stages": 4})
        set_large_grf_mode(kernel_args)

    _layer_norm_backward_kernel[grid](
        X, X.stride(0),
        W,
        Mean, Mean.stride(0),
        RSTD, RSTD.stride(0),
        DX, DX.stride(0),
        DW,
        DB,
        dY, dY.stride(0),
        n_rows, n_cols,
        rows_per_program=rows_per_program,
        BLOCK_SIZE=BLOCK_SIZE,
        **kernel_args,
    )

    DX = DX.view(*shape)
    return DX, DW.to(W.dtype), DB.to(B.dtype)


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
