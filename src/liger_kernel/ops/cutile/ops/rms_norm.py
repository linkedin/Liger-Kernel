# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import is_dtensor

ConstBool = ct.Constant[bool]
ConstFloat = ct.Constant[float]
ConstInt = ct.Constant[int]

_CASTING_MODE_NONE = -1
_CASTING_MODE_LLAMA = 0
_CASTING_MODE_GEMMA = 1


def _calculate_settings(n_cols):
    block_size = _next_power_of_2(n_cols)
    if block_size > 65536:
        raise RuntimeError(f"Hidden dimension {n_cols} exceeds maximum supported size of 65536.")
    return block_size


@ct.kernel(occupancy=1)
def _rms_norm_fwd_kernel_ct(
    x,  # (n_rows, n_cols)
    y,  # (n_rows, n_cols)
    w,  # (n_cols,) or dummy
    rstd_out,  # (n_rows,)
    n_cols: ConstInt,
    eps: ConstFloat,
    offset: ConstFloat,
    BLOCK_SIZE: ConstInt,
    CHECK_BOUNDS: ConstBool,
    CASTING_MODE: ConstInt,
    ELEMENTWISE_AFFINE: ConstBool,
):
    row_idx = ct.bid(0)
    num_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    sum_sq = ct.full((1,), 0.0, dtype=ct.float32)
    for ci in range(num_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        x_tile = ct.gather(x, (row_idx, col_idx), check_bounds=CHECK_BOUNDS, padding_value=0.0)

        if CASTING_MODE == _CASTING_MODE_NONE:
            x_acc = ct.astype(x_tile, ct.float32)
        else:
            x_acc = ct.astype(x_tile, ct.float32)

        chunk_sum_sq = ct.sum(x_acc * x_acc, 0, keepdims=False)
        sum_sq = ct.full((1,), ct.sum(sum_sq, 0, keepdims=False) + chunk_sum_sq, dtype=ct.float32)

    mean_square = ct.sum(sum_sq, 0, keepdims=False) / n_cols
    rstd = ct.rsqrt(mean_square + eps)
    ct.scatter(rstd_out, row_idx, ct.astype(rstd, rstd_out.dtype))

    for ci in range(num_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        x_tile = ct.gather(x, (row_idx, col_idx), check_bounds=CHECK_BOUNDS, padding_value=0.0)

        if CASTING_MODE == _CASTING_MODE_GEMMA:
            x_norm = ct.astype(x_tile, ct.float32) * rstd
            if ELEMENTWISE_AFFINE:
                w_tile = ct.astype(ct.gather(w, col_idx, check_bounds=CHECK_BOUNDS, padding_value=0.0), ct.float32)
                y_tile = x_norm * (offset + w_tile)
            else:
                y_tile = x_norm
            y_tile = ct.astype(y_tile, y.dtype)

        elif CASTING_MODE == _CASTING_MODE_LLAMA:
            x_norm = ct.astype(x_tile, ct.float32) * rstd
            x_norm = ct.astype(x_norm, x_tile.dtype)
            if ELEMENTWISE_AFFINE:
                w_tile = ct.gather(w, col_idx, check_bounds=CHECK_BOUNDS, padding_value=0.0)
                y_tile = x_norm * (ct.astype(offset, x_norm.dtype) + w_tile)
            else:
                y_tile = x_norm

        else:
            x_norm = ct.astype(ct.astype(x_tile, ct.float32) * rstd, x_tile.dtype)
            if ELEMENTWISE_AFFINE:
                w_tile = ct.gather(w, col_idx, check_bounds=CHECK_BOUNDS, padding_value=0.0)
                y_tile = x_norm * (ct.astype(offset, x_norm.dtype) + w_tile)
            else:
                y_tile = x_norm

        ct.scatter(y, (row_idx, col_idx), ct.astype(y_tile, y.dtype), check_bounds=CHECK_BOUNDS)


@ct.kernel(occupancy=1)
def _rms_norm_bwd_combined_kernel_ct(
    x,  # (n_rows, n_cols)
    dy,  # (n_rows, n_cols)
    w,  # (n_cols,) or dummy
    rstd,  # (n_rows,)
    dx,  # (n_rows, n_cols)
    dw_partial,  # (num_programs, n_cols) or dummy
    n_rows: ConstInt,
    n_cols: ConstInt,
    rows_per_program: ConstInt,
    offset: ConstFloat,
    BLOCK_SIZE: ConstInt,
    CHECK_BOUNDS: ConstBool,
    CASTING_MODE: ConstInt,
    ELEMENTWISE_AFFINE: ConstBool,
):
    block_id = ct.bid(0)
    col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32)

    if ELEMENTWISE_AFFINE:
        w_row = ct.gather(w, col_idx, check_bounds=CHECK_BOUNDS, padding_value=0.0)
        if CASTING_MODE == _CASTING_MODE_GEMMA:
            w_row = ct.astype(w_row, ct.float32)
        w_row = w_row + offset
        dw_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

    for ri in range(rows_per_program):
        row_idx = block_id * rows_per_program + ri
        if row_idx < n_rows:
            x_row = ct.gather(x, (row_idx, col_idx), check_bounds=CHECK_BOUNDS, padding_value=0.0)
            dy_row = ct.gather(dy, (row_idx, col_idx), check_bounds=CHECK_BOUNDS, padding_value=0.0)
            rstd_row = ct.astype(ct.load(rstd, row_idx, shape=()), ct.float32)

            x_row_f32 = ct.astype(x_row, ct.float32)

            if CASTING_MODE == _CASTING_MODE_LLAMA:
                if ELEMENTWISE_AFFINE:
                    m = ct.astype(dy_row * w_row, ct.float32)
                else:
                    m = ct.astype(dy_row, ct.float32)
            elif CASTING_MODE == _CASTING_MODE_GEMMA:
                dy_row = ct.astype(dy_row, ct.float32)
                if ELEMENTWISE_AFFINE:
                    m = dy_row * w_row
                else:
                    m = dy_row
            else:
                if ELEMENTWISE_AFFINE:
                    m = ct.astype(dy_row * w_row, ct.float32)
                else:
                    m = ct.astype(dy_row, ct.float32)

            dx_row = rstd_row * m
            dx_row = dx_row + (
                rstd_row * (-(1.0 / n_cols) * rstd_row * rstd_row * ct.sum(m * x_row_f32, 0, keepdims=False) * x_row_f32)
            )
            ct.scatter(dx, (row_idx, col_idx), ct.astype(dx_row, dx.dtype), check_bounds=CHECK_BOUNDS)

            if ELEMENTWISE_AFFINE:
                if CASTING_MODE == _CASTING_MODE_LLAMA:
                    dw_acc = dw_acc + ct.astype(dy_row * ct.astype(x_row_f32 * rstd_row, x.dtype), ct.float32)
                else:
                    dw_acc = dw_acc + ct.astype(dy_row, ct.float32) * (x_row_f32 * rstd_row)

    if ELEMENTWISE_AFFINE:
        ct.scatter(dw_partial, (block_id, col_idx), dw_acc, check_bounds=CHECK_BOUNDS)


_str_to_casting_mode = {
    "llama": _CASTING_MODE_LLAMA,
    "gemma": _CASTING_MODE_GEMMA,
    "none": _CASTING_MODE_NONE,
}


def rms_norm_forward(X, W, eps, offset, casting_mode, row_mode):
    del row_mode

    if not isinstance(casting_mode, int):
        assert casting_mode in _str_to_casting_mode, f"Invalid casting mode: {casting_mode}"
        casting_mode = _str_to_casting_mode[casting_mode]
    else:
        assert casting_mode in _str_to_casting_mode.values(), f"Invalid casting mode: {casting_mode}"

    shape = X.shape
    dim = shape[-1]
    X2d = X.view(-1, dim).contiguous()
    n_rows, n_cols = X2d.shape
    block_size = _calculate_settings(n_cols)
    check_bounds = n_cols % block_size != 0

    Y = torch.empty_like(X2d)
    rstd_dtype = torch.float32 if casting_mode in (_CASTING_MODE_LLAMA, _CASTING_MODE_GEMMA) else X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

    if W is not None:
        assert X2d.shape[1] == W.shape[0], (
            "Incompatible hidden size dimension between tensor1.shape[1] and tensor2.shape[0]"
        )
        W_arg = W.contiguous()
        elementwise_affine = True
    else:
        W_arg = torch.empty(1, dtype=X.dtype, device=X.device)
        elementwise_affine = False

    ct.launch(
        torch.cuda.current_stream(),
        (n_rows, 1, 1),
        _rms_norm_fwd_kernel_ct,
        (
            X2d,
            Y,
            W_arg,
            RSTD,
            int(n_cols),
            float(eps),
            float(offset),
            int(block_size),
            bool(check_bounds),
            int(casting_mode),
            bool(elementwise_affine),
        ),
    )

    # Keep return shape compatible with the default Triton implementation.
    return Y.view(*shape), X2d, RSTD, block_size, None, casting_mode


def rms_norm_backward(dY, X, W, RSTD, offset, casting_mode, BLOCK_SIZE, num_warps, in_place, row_mode):
    del num_warps
    del row_mode

    shape = dY.shape
    dim = shape[-1]
    dY2d = dY.view(-1, dim).contiguous()
    n_rows, n_cols = dY2d.shape

    if n_cols > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    if in_place:
        dX = dY2d
    else:
        dX = torch.empty_like(dY2d)

    sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    rows_per_program = max(1, math.ceil(n_rows / sm_count))
    check_bounds = n_cols % BLOCK_SIZE != 0

    if W is not None:
        elementwise_affine = True
        W_arg = W.contiguous()
        dW_partial = torch.empty((sm_count, n_cols), dtype=torch.float32, device=W.device)
    else:
        elementwise_affine = False
        W_arg = torch.empty(1, dtype=X.dtype, device=X.device)
        dW_partial = torch.empty((1, 1), dtype=torch.float32, device=X.device)

    ct.launch(
        torch.cuda.current_stream(),
        (sm_count, 1, 1),
        _rms_norm_bwd_combined_kernel_ct,
        (
            X.contiguous(),
            dY2d,
            W_arg,
            RSTD,
            dX,
            dW_partial,
            int(n_rows),
            int(n_cols),
            int(rows_per_program),
            float(offset),
            int(BLOCK_SIZE),
            bool(check_bounds),
            int(casting_mode),
            bool(elementwise_affine),
        ),
    )

    if elementwise_affine:
        dW = dW_partial.sum(dim=0).to(W.dtype)
    else:
        dW = None

    return dX.view(*shape), dW


class LigerRMSNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, eps, offset=0.0, casting_mode="llama", in_place=True, row_mode=None):
        if is_dtensor(X):
            X = X.full_tensor()

        Y, X2d, RSTD, BLOCK_SIZE, num_warps, casting_mode = rms_norm_forward(
            X, W, eps, offset, casting_mode, row_mode
        )
        ctx.offset = offset
        ctx.casting_mode = casting_mode
        ctx.in_place = in_place
        ctx.row_mode = row_mode
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.elementwise_affine = W is not None
        if W is not None:
            ctx.save_for_backward(X2d, W, RSTD)
        else:
            ctx.save_for_backward(X2d, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        if ctx.elementwise_affine:
            X, W, RSTD = ctx.saved_tensors
        else:
            X, RSTD = ctx.saved_tensors
            W = None

        if is_dtensor(dY):
            dY = dY.full_tensor()

        dX, dW = rms_norm_backward(
            dY,
            X,
            W,
            RSTD,
            ctx.offset,
            ctx.casting_mode,
            ctx.BLOCK_SIZE,
            ctx.num_warps,
            ctx.in_place,
            ctx.row_mode,
        )
        return dX, dW, None, None, None, None, None