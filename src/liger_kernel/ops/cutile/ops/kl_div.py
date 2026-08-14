# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
KL Divergence loss kernel (CuTile backend).

Computes KL(y_true || y_pred) where y_pred is in log-space.
"""

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2

MAX_FUSED_SIZE = 4096

_REDUCTION_MODE_NONE = 0
_REDUCTION_MODE_SUM = 1
_REDUCTION_MODE_MEAN = 2
_REDUCTION_MODE_BATCHMEAN = 3

_str_to_reduction_mode = {
    "none": _REDUCTION_MODE_NONE,
    "sum": _REDUCTION_MODE_SUM,
    "mean": _REDUCTION_MODE_MEAN,
    "batchmean": _REDUCTION_MODE_BATCHMEAN,
}


@ct.kernel
def _kldiv_fwd_none_kernel_ct(
    Y,
    GT,
    LOSS,
    n_cols: ct.Constant[int],
    eps: ct.Constant[float],
    BLOCK_SIZE: ct.Constant[int],
    LOG_TARGET: ct.Constant[int],
    N_FULL_CHUNKS: ct.Constant[int],
):
    row_idx = ct.bid(0)
    eps_tile = ct.full((BLOCK_SIZE,), eps, dtype=ct.float32)

    for ci in range(N_FULL_CHUNKS):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        y = ct.astype(ct.gather(Y, (row_idx, col_idx), check_bounds=False), ct.float32)
        gt = ct.astype(ct.gather(GT, (row_idx, col_idx), check_bounds=False), ct.float32)

        if LOG_TARGET:
            loss = ct.exp(gt) * (gt - y)
        else:
            gt_clipped = ct.maximum(gt, eps_tile)
            loss = gt * (ct.log(gt_clipped) - y)

        ct.scatter(LOSS, (row_idx, col_idx), ct.astype(loss, LOSS.dtype), check_bounds=False)

    if N_FULL_CHUNKS * BLOCK_SIZE < n_cols:
        ci = N_FULL_CHUNKS
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        y = ct.astype(ct.gather(Y, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        gt = ct.astype(ct.gather(GT, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)

        if LOG_TARGET:
            loss = ct.exp(gt) * (gt - y)
        else:
            gt_clipped = ct.maximum(gt, eps_tile)
            loss = gt * (ct.log(gt_clipped) - y)

        ct.scatter(LOSS, (row_idx, col_idx), ct.astype(loss, LOSS.dtype), check_bounds=True)


@ct.kernel
def _kldiv_fwd_reduce_kernel_ct(
    Y,
    GT,
    LOSS,
    n_cols: ct.Constant[int],
    eps: ct.Constant[float],
    BLOCK_SIZE: ct.Constant[int],
    LOG_TARGET: ct.Constant[int],
    N_FULL_CHUNKS: ct.Constant[int],
):
    row_idx = ct.bid(0)

    loss_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    eps_tile = ct.full((BLOCK_SIZE,), eps, dtype=ct.float32)

    for ci in range(N_FULL_CHUNKS):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        y = ct.astype(ct.gather(Y, (row_idx, col_idx), check_bounds=False), ct.float32)
        gt = ct.astype(ct.gather(GT, (row_idx, col_idx), check_bounds=False), ct.float32)

        if LOG_TARGET:
            loss = ct.exp(gt) * (gt - y)
        else:
            gt_clipped = ct.maximum(gt, eps_tile)
            loss = gt * (ct.log(gt_clipped) - y)

        loss_acc = ct.add(loss_acc, loss)

    if N_FULL_CHUNKS * BLOCK_SIZE < n_cols:
        ci = N_FULL_CHUNKS
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        y = ct.astype(ct.gather(Y, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        gt = ct.astype(ct.gather(GT, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)

        if LOG_TARGET:
            loss = ct.exp(gt) * (gt - y)
        else:
            gt_clipped = ct.maximum(gt, eps_tile)
            loss = gt * (ct.log(gt_clipped) - y)

        loss_acc = ct.add(loss_acc, loss)

    row_sum = ct.sum(loss_acc, 0, keepdims=False)
    ct.scatter(LOSS, row_idx, ct.astype(row_sum, LOSS.dtype))


@ct.kernel
def _kldiv_bwd_kernel_ct(
    GT,
    GRADS,
    n_cols: ct.Constant[int],
    scale,  # runtime: depends on grad_output.item() — making this constexpr triggers a
    # JIT recompile on every backward when the upstream grad changes (e.g. in autograd
    # benchmarks that draw a fresh torch.randn_like(loss) per iteration).
    BLOCK_SIZE: ct.Constant[int],
    LOG_TARGET: ct.Constant[int],
    N_FULL_CHUNKS: ct.Constant[int],
):
    row_idx = ct.bid(0)

    for ci in range(N_FULL_CHUNKS):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        gt = ct.astype(ct.gather(GT, (row_idx, col_idx), check_bounds=False), ct.float32)

        if LOG_TARGET:
            res = -ct.exp(gt) * scale
        else:
            res = -gt * scale

        ct.scatter(GRADS, (row_idx, col_idx), ct.astype(res, GRADS.dtype), check_bounds=False)

    if N_FULL_CHUNKS * BLOCK_SIZE < n_cols:
        ci = N_FULL_CHUNKS
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        gt = ct.astype(ct.gather(GT, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)

        if LOG_TARGET:
            res = -ct.exp(gt) * scale
        else:
            res = -gt * scale

        ct.scatter(GRADS, (row_idx, col_idx), ct.astype(res, GRADS.dtype), check_bounds=True)


def _kldiv_forward(y_pred, y_true, log_target, reduction, eps):
    BT, V = y_pred.shape
    BLOCK_SIZE = min(MAX_FUSED_SIZE, _next_power_of_2(V))
    reduction_int = _str_to_reduction_mode[reduction]
    n_full_chunks = V // BLOCK_SIZE

    grid = (BT, 1, 1)

    if reduction_int == _REDUCTION_MODE_NONE:
        output_tensor = torch.zeros(BT, V, device=y_pred.device, dtype=torch.float32)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _kldiv_fwd_none_kernel_ct,
            (
                y_pred,
                y_true,
                output_tensor,
                int(V),
                float(eps),
                int(BLOCK_SIZE),
                int(log_target),
                int(n_full_chunks),
            ),
        )
        return output_tensor
    else:
        row_sums = torch.zeros(BT, device=y_pred.device, dtype=torch.float32)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _kldiv_fwd_reduce_kernel_ct,
            (
                y_pred,
                y_true,
                row_sums,
                int(V),
                float(eps),
                int(BLOCK_SIZE),
                int(log_target),
                int(n_full_chunks),
            ),
        )
        if reduction_int == _REDUCTION_MODE_BATCHMEAN:
            return row_sums.sum() / BT
        elif reduction_int == _REDUCTION_MODE_SUM:
            return row_sums.sum(dim=0)
        else:  # mean
            return row_sums.sum() / (BT * V)


def _kldiv_backward(y_true, scale, log_target):
    BT, V = y_true.shape
    BLOCK_SIZE = min(MAX_FUSED_SIZE, _next_power_of_2(V))
    n_full_chunks = V // BLOCK_SIZE

    new_grads = torch.empty_like(y_true)
    grid = (BT, 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _kldiv_bwd_kernel_ct,
        (
            y_true,
            new_grads,
            int(V),
            float(scale),
            int(BLOCK_SIZE),
            int(log_target),
            int(n_full_chunks),
        ),
    )

    return new_grads


class LigerKLDivLossFunction(torch.autograd.Function):
    """CuTile autograd wrapper for KL divergence loss."""

    @staticmethod
    def forward(ctx, y_pred, y_true, reduction, log_target, eps):
        y_pred = y_pred.contiguous()
        y_true = y_true.contiguous()
        ctx.save_for_backward(y_true)
        ctx.reduction = reduction
        ctx.log_target = log_target
        return _kldiv_forward(y_pred, y_true, log_target, reduction, eps)

    @staticmethod
    def backward(ctx, grad_output):
        (y_true,) = ctx.saved_tensors
        BT, V = y_true.shape

        if grad_output.numel() == 1:
            scale = grad_output.item()
        else:
            scale = 1.0

        if ctx.reduction == "batchmean":
            scale /= BT
        elif ctx.reduction == "mean":
            scale /= BT * V

        derivative = _kldiv_backward(y_true, scale, ctx.log_target)

        if grad_output.numel() != 1:
            derivative = derivative * grad_output

        return derivative, None, None, None, None
