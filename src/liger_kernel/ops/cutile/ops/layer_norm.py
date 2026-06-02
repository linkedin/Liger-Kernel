# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Layer Normalization kernel (CuTile backend).

Forward: row-parallel (one block per row). Single-pass computes sum and sum_sq
  via fold trick → mean and variance. Second pass applies normalization:
  Y = (X - mean) * rstd * W + B. Mean and RSTD cached for backward.

Backward: For combined backward path, use single combined kernel, grid=(sm_count, 1, 1), persistent loop.
  Block b processes rows [b*rpp, min((b+1)*rpp, n_rows)).
  Per row: loads X, DY, W, Mean[row], RSTD[row]; computes x_hat, wdy, c1, c2;
  writes DX[row] inline; accumulates dW+=dy*x_hat, dB+=dy.
  After loop: writes DW_partial[block_id] and DB_partial[block_id].
"""

import math

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.utils import ensure_contiguous

ConstBool = ct.Constant[bool]
ConstFloat = ct.Constant[float]
ConstInt = ct.Constant[int]


def _calculate_settings(n_cols):
    BLOCK_SIZE = _next_power_of_2(n_cols)
    if BLOCK_SIZE > 65536:
        raise RuntimeError(f"Hidden dimension {n_cols} exceeds maximum supported size of 65536.")
    return BLOCK_SIZE


@ct.kernel(occupancy=1)
def _layer_norm_fwd_kernel_ct(
    x,  # (n_rows, n_cols) input
    y,  # (n_rows, n_cols) output
    weight,  # (n_cols,) scale
    bias,  # (n_cols,) bias
    mean_out,  # (n_rows,) cached mean
    rstd_out,  # (n_rows,) cached reciprocal std
    n_cols: ConstInt,
    eps: ConstFloat,
    BLOCK_SIZE: ConstInt,
    ALIGNED: ConstBool,
):
    """
    Layer norm forward.

    Row-parallel: one block per row. Grid=(n_rows,1,1): every row_idx in [0,n_rows).
    When ALIGNED=True (n_cols is power-of-2), BLOCK_SIZE==n_cols and column accesses
    are exactly in-bounds → check_bounds=False enables hardware TMA path.
    """
    row_idx = ct.bid(0)
    n_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    check_bounds = not ALIGNED

    # ---- Pass 1: compute sum(x) and sum(x^2) via fold trick ----
    # OOB column positions (when ALIGNED=False, last chunk) padded with 0.0 → correct
    sum_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    sum_sq_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        x_tile = ct.astype(ct.gather(x, (row_idx, col_idx), check_bounds=check_bounds, padding_value=0.0), ct.float32)
        sum_tile = ct.add(sum_tile, x_tile)
        sum_sq_tile = ct.add(sum_sq_tile, x_tile * x_tile)

    total_sum = ct.sum(sum_tile, 0, keepdims=False)  # scalar
    total_sum_sq = ct.sum(sum_sq_tile, 0, keepdims=False)  # scalar
    mean = total_sum / n_cols
    var = total_sum_sq / n_cols - mean * mean
    rstd = ct.rsqrt(var + eps)  # scalar

    # Cache mean and rstd for backward
    ct.scatter(mean_out, row_idx, ct.astype(mean, mean_out.dtype))
    ct.scatter(rstd_out, row_idx, ct.astype(rstd, rstd_out.dtype))

    # ---- Pass 2: Y = (X - mean) * rstd * W + B ----
    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        x_tile = ct.astype(ct.gather(x, (row_idx, col_idx), check_bounds=check_bounds, padding_value=0.0), ct.float32)
        weight_tile = ct.astype(ct.gather(weight, col_idx, check_bounds=check_bounds, padding_value=0.0), ct.float32)
        bias_tile = ct.astype(ct.gather(bias, col_idx, check_bounds=check_bounds, padding_value=0.0), ct.float32)
        y_tile = (x_tile - mean) * rstd * weight_tile + bias_tile
        ct.scatter(y, (row_idx, col_idx), ct.astype(y_tile, y.dtype), check_bounds=check_bounds)


@ct.kernel(occupancy=1)
def _layer_norm_bwd_dx_kernel_ct(
    x,  # (n_rows, n_cols) saved input
    dy,  # (n_rows, n_cols) upstream gradient
    weight,  # (n_cols,) scale
    mean,  # (n_rows,) saved mean
    rstd,  # (n_rows,) saved rstd
    dx,  # (n_rows, n_cols) output gradient w.r.t. input
    n_cols: ConstInt,
    BLOCK_SIZE: ConstInt,
    ALIGNED: ConstBool,
):
    """
    Layer norm backward — DX kernel only.

    Grid: (n_rows, 1, 1). One block per row → high concurrency hides latency.
    Each block: load X[row], DY[row], W, Mean[row], RSTD[row]; compute DX[row].
    Does NOT compute DW/DB — handled by separate _layer_norm_bwd_dw_ct kernel.

    When ALIGNED=True: check_bounds=False → hardware TMA path for all accesses.
    """
    row_idx = ct.bid(0)
    col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32)
    check_bounds = not ALIGNED

    mean_row = ct.astype(ct.load(mean, row_idx, shape=(), latency=3), ct.float32)
    rstd_row = ct.astype(ct.load(rstd, row_idx, shape=(), latency=3), ct.float32)

    weight_tile = ct.astype(ct.gather(weight, col_idx, check_bounds=check_bounds, padding_value=0.0), ct.float32)
    _lat = 3 if BLOCK_SIZE <= 1024 else 6
    x_tile = ct.astype(
        ct.gather(x, (row_idx, col_idx), check_bounds=check_bounds, padding_value=0.0, latency=_lat), ct.float32
    )
    dy_tile = ct.astype(
        ct.gather(dy, (row_idx, col_idx), check_bounds=check_bounds, padding_value=0.0, latency=_lat), ct.float32
    )

    inv_n_cols = 1.0 / n_cols
    x_hat = (x_tile - mean_row) * rstd_row
    wdy = weight_tile * dy_tile
    c1 = ct.sum(x_hat * wdy, 0, keepdims=False) * inv_n_cols
    c2 = ct.sum(wdy, 0, keepdims=False) * inv_n_cols
    dx_tile = (wdy - (x_hat * c1 + c2)) * rstd_row
    ct.scatter(dx, (row_idx, col_idx), ct.astype(dx_tile, dx.dtype), check_bounds=check_bounds)


@ct.kernel(occupancy=1)
def _layer_norm_bwd_combined_kernel_ct(
    x,  # (n_rows, n_cols) saved input
    dy,  # (n_rows, n_cols) upstream gradient
    weight,  # (n_cols,) scale
    mean,  # (n_rows,) saved mean
    rstd,  # (n_rows,) saved rstd
    dx,  # (n_rows, n_cols) output gradient w.r.t. input
    dw_partial,  # (num_programs, n_cols) partial DW accumulator
    db_partial,  # (num_programs, n_cols) partial DB accumulator
    n_rows: ConstInt,
    n_cols: ConstInt,
    num_programs: ConstInt,
    rows_per_program: ConstInt,
    BLOCK_SIZE: ConstInt,
    ALIGNED: ConstBool,
):
    """
    Layer norm backward — combined DX + DW/DB kernel (one-stream path).

    Grid: (num_programs, 1, 1). Block b processes rows [b*rpp, min((b+1)*rpp, n_rows)).
    Computes DX inline (written per-row) and accumulates DW/DB partials over all rows.

    W is loaded once per block outside the row loop to amortize the load cost.
    When ALIGNED=True: check_bounds=False → hardware TMA for all accesses.
    """
    block_id = ct.bid(0)
    col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32)
    check_bounds = not ALIGNED

    # Load W once per block — shared across all rows this block processes.
    weight_tile = ct.astype(
        ct.gather(weight, col_idx, check_bounds=check_bounds, padding_value=0.0, latency=2), ct.float32
    )

    dw_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    db_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

    for ri in range(rows_per_program):
        row_idx = block_id * rows_per_program + ri
        if row_idx < n_rows:
            mean_row = ct.astype(ct.load(mean, row_idx, shape=(), latency=3), ct.float32)
            rstd_row = ct.astype(ct.load(rstd, row_idx, shape=(), latency=3), ct.float32)
            x_tile = ct.astype(
                ct.gather(x, (row_idx, col_idx), check_bounds=check_bounds, padding_value=0.0, latency=2), ct.float32
            )
            dy_tile = ct.astype(
                ct.gather(dy, (row_idx, col_idx), check_bounds=check_bounds, padding_value=0.0, latency=2),
                ct.float32,
            )

            x_hat = (x_tile - mean_row) * rstd_row
            wdy = weight_tile * dy_tile
            inv_n_cols = 1.0 / n_cols
            c1 = ct.sum(x_hat * wdy, 0, keepdims=False) * inv_n_cols
            c2 = ct.sum(wdy, 0, keepdims=False) * inv_n_cols
            dx_tile = (wdy - (x_hat * c1 + c2)) * rstd_row
            ct.scatter(dx, (row_idx, col_idx), ct.astype(dx_tile, dx.dtype), check_bounds=check_bounds)

            dw_acc = ct.add(dw_acc, dy_tile * x_hat)
            db_acc = ct.add(db_acc, dy_tile)

    # Write per-block DW/DB partials.
    # ALIGNED=True: BLOCK_SIZE==n_cols, ct.store is safe (no OOB).
    # ALIGNED=False: BLOCK_SIZE>n_cols, scatter with check_bounds=True to avoid OOB writes.
    if ALIGNED:
        ct.store(dw_partial, index=(block_id, 0), tile=dw_acc.reshape((1, BLOCK_SIZE)))
        ct.store(db_partial, index=(block_id, 0), tile=db_acc.reshape((1, BLOCK_SIZE)))
    else:
        ct.scatter(dw_partial, (block_id, col_idx), dw_acc, check_bounds=True)
        ct.scatter(db_partial, (block_id, col_idx), db_acc, check_bounds=True)


def _layer_norm_forward_ct(X, W, B, eps):
    shape = X.shape
    dim = shape[-1]
    X2d = X.view(-1, dim).contiguous()
    n_rows, n_cols = X2d.shape

    BLOCK_SIZE = _calculate_settings(n_cols)
    aligned = (n_cols & (n_cols - 1)) == 0  # True when n_cols is a power of 2

    Y = torch.empty_like(X2d)
    Mean = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    RSTD = torch.empty(n_rows, dtype=X.dtype, device=X.device)

    grid = (n_rows, 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _layer_norm_fwd_kernel_ct,
        (
            X2d,
            Y,
            W.contiguous(),
            B.contiguous(),
            Mean,
            RSTD,
            int(n_cols),
            float(eps),
            int(BLOCK_SIZE),
            bool(aligned),
        ),
    )

    return Y.view(*shape), X2d, Mean, RSTD, BLOCK_SIZE


def _layer_norm_backward_ct(dY, X, W, B, Mean, RSTD, BLOCK_SIZE, compute_dW=True, compute_dB=True):
    shape = dY.shape
    dim = shape[-1]
    dY2d = dY.view(-1, dim).contiguous()
    n_rows, n_cols = dY2d.shape

    sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    aligned = (n_cols & (n_cols - 1)) == 0  # True when n_cols is a power of 2

    X_contig = X.contiguous()
    W_contig = W.contiguous()
    DX = torch.empty_like(X)

    # Fast path: skip DW/DB entirely if neither W nor B needs gradients.
    if not compute_dW and not compute_dB:
        ct.launch(
            torch.cuda.current_stream(),
            (n_rows, 1, 1),
            _layer_norm_bwd_dx_kernel_ct,
            (
                X_contig,
                dY2d,
                W_contig,
                Mean,
                RSTD,
                DX,
                int(n_cols),
                int(BLOCK_SIZE),
                bool(aligned),
            ),
        )
        DX = DX.view(*shape)
        DW = torch.zeros_like(W)
        DB = torch.zeros_like(B)
        return DX, DW, DB

    num_programs = sm_count
    rows_per_program = math.ceil(n_rows / num_programs)

    DW_partial = torch.empty((num_programs, n_cols), dtype=torch.float32, device=W.device)
    DB_partial = torch.empty((num_programs, n_cols), dtype=torch.float32, device=W.device)

    ct.launch(
        torch.cuda.current_stream(),
        (num_programs, 1, 1),
        _layer_norm_bwd_combined_kernel_ct,
        (
            X_contig,
            dY2d,
            W_contig,
            Mean,
            RSTD,
            DX,
            DW_partial,
            DB_partial,
            int(n_rows),
            int(n_cols),
            int(num_programs),
            int(rows_per_program),
            int(BLOCK_SIZE),
            bool(aligned),
        ),
    )

    DX = DX.view(*shape)
    DW = DW_partial.sum(dim=0).to(W.dtype)
    DB = DB_partial.sum(dim=0).to(B.dtype)
    return DX, DW, DB


def layer_norm_forward(X, W, B, eps):
    Y, X2d, Mean, RSTD, BLOCK_SIZE = _layer_norm_forward_ct(X, W, B, eps)
    return Y, X2d, Mean, RSTD, BLOCK_SIZE, None


def layer_norm_backward(dY, X, W, B, Mean, RSTD):
    BLOCK_SIZE = _calculate_settings(X.shape[-1])
    return _layer_norm_backward_ct(dY, X, W, B, Mean, RSTD, BLOCK_SIZE)


class LigerLayerNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, B, eps):
        Y, X, Mean, RSTD, BLOCK_SIZE, _ = layer_norm_forward(X, W, B, eps)
        ctx.save_for_backward(X, W, B, Mean, RSTD)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X, W, B, Mean, RSTD = ctx.saved_tensors
        need_dW = ctx.needs_input_grad[1]
        need_dB = ctx.needs_input_grad[2]
        DX, DW, DB = _layer_norm_backward_ct(
            dY, X, W, B, Mean, RSTD, ctx.BLOCK_SIZE, compute_dW=need_dW, compute_dB=need_dB
        )
        return DX, DW if need_dW else None, DB if need_dB else None, None
