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
def _batch_norm_forward_kernel(
        X_ptr,
        X_batch_stride,
        X_row_stride,
        Y_ptr,
        Y_batch_stride,
        Y_row_stride,
        # MEAN_ptr,
        # MEAN_row_stride,
        # VARIANCE_ptr,
        # VARIANCE_row_stride,
        # AXIS_ptr,
        # AXIS_row_stride,
        n_cols,
        eps,
        mean,
        axis,
        scale,
        offset,
        variance,
        BLOCK_SIZE: tl.constexpr):
    pass
    row_idx = tl.program_id(1)
    batch_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE) * batch_idx

    mask = True  # col_offsets < n_cols
    X_ptr += (X_row_stride) * (row_idx)  # +X_batch_stride
    Y_ptr += (Y_row_stride) * (row_idx)  # +Y_batch_stride
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)

    inv = tl.rsqrt(tl.load(variance) + eps)

    if scale is not None:
        inv = inv * tl.load(scale)

    res = -tl.load(mean) * inv
    if offset is not None:
        res = res + tl.load(offset)

    tl.store(Y_ptr + col_offsets, X_row * inv + res)


def batch_norm_forward(X, axis, offset, scale, eps, mean, variance):
    batch, n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.zeros((batch, n_rows, n_cols), dtype=X.dtype, device=X.device)

    _batch_norm_forward_kernel[(batch * n_cols * n_rows, n_rows,)](  # [(n_rows,)]
        X,
        X.stride(0),
        X.stride(1),
        Y,
        Y.stride(0),
        Y.stride(1),
        n_cols,
        eps,
        mean,
        axis,
        scale,
        offset,
        variance,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return Y


class LigerBatchNormFunction(torch.autograd.Function):
    variance = torch.tensor([1] * 32 * 8).to('cuda')
    scale = torch.tensor([1] * 32 * 8).to('cuda')
    offset = torch.tensor([0] * 32 * 8).to('cuda')
    mean = torch.tensor([0] * 32 * 8).to('cuda')

    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, axis, eps):
        X = batch_norm_forward(X, axis, LigerBatchNormFunction.offset, LigerBatchNormFunction.scale,
                               eps, LigerBatchNormFunction.mean, LigerBatchNormFunction.variance)

        return X
