# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Group Normalization kernel (CuTile backend).
"""

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.utils import ensure_contiguous

MAX_FUSED_SIZE = 65536


@ct.kernel
def _group_norm_fwd_kernel_ct(
    x_input,
    y_output,
    weight,
    bias,
    mean_stats,
    rstd_stats,
    NUM_CHANNELS: ct.Constant[int],
    NUM_GROUPS: ct.Constant[int],
    CHANNELS_PER_GROUP: ct.Constant[int],
    TOTAL_HIDDEN_SIZE: ct.Constant[int],
    eps,
    BLOCK_SIZE: ct.Constant[int],
):
    batch_idx = ct.bid(0)
    group_idx = ct.bid(1)

    group_row = batch_idx * NUM_GROUPS + group_idx

    N = CHANNELS_PER_GROUP * TOTAL_HIDDEN_SIZE
    inv_N = 1.0 / N

    num_h_chunks = (TOTAL_HIDDEN_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE

    sum_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    sum_sq_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

    for c_in_group in range(CHANNELS_PER_GROUP):
        channel_idx = group_idx * CHANNELS_PER_GROUP + c_in_group
        row_idx = batch_idx * NUM_CHANNELS + channel_idx

        for hi in range(num_h_chunks):
            col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32) + hi * BLOCK_SIZE
            x_tile = ct.astype(
                ct.gather(x_input, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
                ct.float32,
            )
            sum_tile = sum_tile + x_tile
            sum_sq_tile = sum_sq_tile + x_tile * x_tile

    s = ct.sum(sum_tile, 0, keepdims=False)
    sq = ct.sum(sum_sq_tile, 0, keepdims=False)
    mean = s * inv_N
    variance = sq * inv_N - mean * mean
    rstd = ct.rsqrt(variance + eps)

    ct.scatter(mean_stats, group_row, ct.astype(mean, mean_stats.dtype))
    ct.scatter(rstd_stats, group_row, ct.astype(rstd, rstd_stats.dtype))

    for c_in_group in range(CHANNELS_PER_GROUP):
        channel_idx = group_idx * CHANNELS_PER_GROUP + c_in_group
        row_idx = batch_idx * NUM_CHANNELS + channel_idx

        w_scalar = ct.astype(ct.load(weight, channel_idx, shape=()), ct.float32)
        b_scalar = ct.astype(ct.load(bias, channel_idx, shape=()), ct.float32)

        for hi in range(num_h_chunks):
            col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32) + hi * BLOCK_SIZE
            x_tile = ct.astype(
                ct.gather(x_input, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
                ct.float32,
            )
            y_tile = (x_tile - mean) * rstd * w_scalar + b_scalar
            ct.scatter(y_output, (row_idx, col_idx), ct.astype(y_tile, y_output.dtype), check_bounds=True)


@ct.kernel
def _group_norm_bwd_kernel_ct(
    x_input,
    upstream,
    weight,
    mean_stats,
    rstd_stats,
    dx_output,
    dw_partial,
    db_partial,
    NUM_CHANNELS: ct.Constant[int],
    NUM_GROUPS: ct.Constant[int],
    CHANNELS_PER_GROUP: ct.Constant[int],
    TOTAL_HIDDEN_SIZE: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
):
    batch_idx = ct.bid(0)
    group_idx = ct.bid(1)

    group_row = batch_idx * NUM_GROUPS + group_idx

    mean = ct.astype(ct.load(mean_stats, group_row, shape=()), ct.float32)
    rstd = ct.astype(ct.load(rstd_stats, group_row, shape=()), ct.float32)

    N = CHANNELS_PER_GROUP * TOTAL_HIDDEN_SIZE
    inv_N = 1.0 / N

    num_h_chunks = (TOTAL_HIDDEN_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE

    c1_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    c2_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

    for c_in_group in range(CHANNELS_PER_GROUP):
        channel_idx = group_idx * CHANNELS_PER_GROUP + c_in_group
        row_idx = batch_idx * NUM_CHANNELS + channel_idx

        w_scalar = ct.astype(ct.load(weight, channel_idx, shape=()), ct.float32)

        dW_acc_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
        dB_acc_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

        for hi in range(num_h_chunks):
            col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32) + hi * BLOCK_SIZE
            x_tile = ct.astype(
                ct.gather(x_input, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
                ct.float32,
            )
            upstream_tile = ct.astype(
                ct.gather(upstream, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
                ct.float32,
            )
            x_hat = (x_tile - mean) * rstd
            wdy = w_scalar * upstream_tile
            c1_tile = c1_tile + x_hat * wdy
            c2_tile = c2_tile + wdy
            dW_acc_tile = dW_acc_tile + upstream_tile * x_hat
            dB_acc_tile = dB_acc_tile + upstream_tile

        dW_val = ct.sum(dW_acc_tile, 0, keepdims=False)
        dB_val = ct.sum(dB_acc_tile, 0, keepdims=False)
        ct.scatter(dw_partial, (batch_idx, channel_idx), ct.astype(dW_val, dw_partial.dtype))
        ct.scatter(db_partial, (batch_idx, channel_idx), ct.astype(dB_val, db_partial.dtype))

    c1 = ct.sum(c1_tile, 0, keepdims=False) * inv_N
    c2 = ct.sum(c2_tile, 0, keepdims=False) * inv_N

    for c_in_group in range(CHANNELS_PER_GROUP):
        channel_idx = group_idx * CHANNELS_PER_GROUP + c_in_group
        row_idx = batch_idx * NUM_CHANNELS + channel_idx

        w_scalar = ct.astype(ct.load(weight, channel_idx, shape=()), ct.float32)

        for hi in range(num_h_chunks):
            col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32) + hi * BLOCK_SIZE
            x_tile = ct.astype(
                ct.gather(x_input, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
                ct.float32,
            )
            upstream_tile = ct.astype(
                ct.gather(upstream, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
                ct.float32,
            )
            x_hat = (x_tile - mean) * rstd
            wdy = w_scalar * upstream_tile
            dx = (wdy - (x_hat * c1 + c2)) * rstd
            ct.scatter(dx_output, (row_idx, col_idx), ct.astype(dx, dx_output.dtype), check_bounds=True)


def group_norm_forward(X, num_channels, num_groups, W, B, eps):
    shape = X.shape
    batch_size = shape[0]
    channels_per_group = num_channels // num_groups
    hidden_size = X.shape[-1]

    BLOCK_SIZE = min(MAX_FUSED_SIZE, _next_power_of_2(hidden_size))

    X_2d = X.view(batch_size * num_channels, hidden_size).contiguous()
    Y_2d = torch.empty_like(X_2d)
    # Stats kept in fp32 (matches upstream Liger). bf16 stats round-trip through
    # forward → backward and lose precision in the (x - mean) * rstd step.
    mean_stats = torch.empty(batch_size * num_groups, dtype=torch.float32, device=X.device)
    rstd_stats = torch.empty(batch_size * num_groups, dtype=torch.float32, device=X.device)

    grid = (batch_size, num_groups, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _group_norm_fwd_kernel_ct,
        (
            X_2d,
            Y_2d,
            W,
            B,
            mean_stats,
            rstd_stats,
            int(num_channels),
            int(num_groups),
            int(channels_per_group),
            int(hidden_size),
            float(eps),
            int(BLOCK_SIZE),
        ),
    )

    return Y_2d.view(*shape), X_2d, mean_stats, rstd_stats, BLOCK_SIZE


def group_norm_backward(dY, X_2d, W, B, Mean, RSTD, num_channels, num_groups):
    shape = dY.shape
    batch_size = shape[0]
    hidden_size = shape[-1]
    channels_per_group = num_channels // num_groups
    BLOCK_SIZE = min(MAX_FUSED_SIZE, _next_power_of_2(hidden_size))

    dY_2d = dY.contiguous().view(batch_size * num_channels, hidden_size).contiguous()

    dx_2d = torch.empty_like(X_2d)
    dw_partial = torch.zeros(batch_size, num_channels, dtype=W.dtype, device=W.device)
    db_partial = torch.zeros(batch_size, num_channels, dtype=B.dtype, device=B.device)

    grid = (batch_size, num_groups, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _group_norm_bwd_kernel_ct,
        (
            X_2d,
            dY_2d,
            W,
            Mean,
            RSTD,
            dx_2d,
            dw_partial,
            db_partial,
            int(num_channels),
            int(num_groups),
            int(channels_per_group),
            int(hidden_size),
            int(BLOCK_SIZE),
        ),
    )

    dw = dw_partial.sum(dim=0)
    db = db_partial.sum(dim=0)
    return dx_2d.view(*shape), dw, db


class LigerGroupNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        X,
        affine_scaling_weight,
        affine_shifting_bias,
        num_channels,
        num_groups,
        eps,
    ):
        Y, X_2d, Mean, RSTD, BLOCK_SIZE = group_norm_forward(
            X,
            num_channels,
            num_groups,
            affine_scaling_weight,
            affine_shifting_bias,
            eps,
        )
        ctx.num_channels = num_channels
        ctx.num_groups = num_groups
        ctx.save_for_backward(X_2d, affine_scaling_weight, affine_shifting_bias, Mean, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X_2d, W, B, Mean, RSTD = ctx.saved_tensors
        DX, DW, DB = group_norm_backward(dY, X_2d, W, B, Mean, RSTD, ctx.num_channels, ctx.num_groups)
        return DX, DW, DB, None, None, None
