# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Dynamic Tanh (DyT) activation kernel (cuTile backend).

Formula: y = tanh(alpha * x) * gamma + beta

Forward uses a 2D grid (num_col_blocks, M). Each block handles BLOCK_N columns for one row
using gather/scatter with check_bounds=True to handle partial last chunks. The aligned fast
path (BLOCK_N == N, power-of-2, single col-block) compiles with check_bounds=False.

Backward uses a persistent 2D grid (num_col_blocks, NUM_SMS): each block handles BLOCK_N
columns and strides over rows (start_row_id, start_row_id+NUM_SMS, ...), matching Triton for
row parallelism and occupancy. Per-block partials are reduced on the host (no atomics).
"""

import cuda.tile as ct
import torch

from cuda.tile import RoundingMode as RMd

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2

MAX_FUSED_SIZE = 65536


@ct.kernel
def _dyt_fwd_kernel(
    x_input,  # (M, N) input
    y_output,  # (M, N) output
    alpha_tensor,  # (1,) scalar alpha
    gamma_tensor,  # (N,) per-channel scale
    beta_tensor,  # (N,) per-channel bias (or dummy 1-element tensor when HAVE_BETA=0)
    HAVE_BETA: ct.Constant[int],
    CHECK_BOUNDS: ct.Constant[bool],
    BLOCK_N: ct.Constant[int],
):
    """
    DyT forward kernel.

    Grid: (num_col_blocks, M, 1).
    CHECK_BOUNDS=False fast path (BLOCK_N==N, power-of-2, num_col_blocks==1) avoids
    predicate-mask instructions on every gather/scatter — ~9-11% faster.
    """
    row_id = ct.bid(1)
    # Aligned path (CHECK_BOUNDS=False): single col-block, no col offset, no padding_value.
    # CuTile DCE collapses the unused branch since CHECK_BOUNDS is a compile-time Constant.
    if CHECK_BOUNDS:
        col_indices = ct.arange(BLOCK_N, dtype=ct.int32) + ct.bid(0) * BLOCK_N
        gamma = ct.astype(ct.gather(gamma_tensor, col_indices, check_bounds=True, padding_value=0.0), ct.float32)
        x = ct.astype(ct.gather(x_input, (row_id, col_indices), check_bounds=True, padding_value=0.0), ct.float32)
    else:
        col_indices = ct.arange(BLOCK_N, dtype=ct.int32)
        gamma = ct.astype(ct.gather(gamma_tensor, col_indices, check_bounds=False), ct.float32)
        x = ct.astype(ct.gather(x_input, (row_id, col_indices), check_bounds=False), ct.float32)

    alpha = ct.astype(ct.load(alpha_tensor, 0, shape=()), ct.float32)
    tanh_x = ct.tanh(alpha * x)
    y = tanh_x * gamma
    if HAVE_BETA:
        if CHECK_BOUNDS:
            beta = ct.astype(ct.gather(beta_tensor, col_indices, check_bounds=True, padding_value=0.0), ct.float32)
        else:
            beta = ct.astype(ct.gather(beta_tensor, col_indices, check_bounds=False), ct.float32)
        y = y + beta

    ct.scatter(y_output, (row_id, col_indices), ct.astype(y, y_output.dtype), check_bounds=CHECK_BOUNDS)


@ct.kernel
def _dyt_bwd_kernel(
    dy_input,  # (M, N) upstream gradient
    dx_output,  # (M, N) gradient w.r.t. x
    da_partial,  # (NUM_SMS, num_col_blocks) partial d_alpha per block, host reduces
    dg_partial,  # (NUM_SMS, N) partial d_gamma per block row, host reduces
    db_partial,  # (NUM_SMS, N) partial d_beta when HAVE_BETA, host reduces
    x_input,  # (M, N) saved input
    alpha_tensor,  # (1,) scalar alpha
    gamma_tensor,  # (N,) per-channel scale
    HAVE_BETA: ct.Constant[int],
    M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    NUM_SMS: ct.Constant[int],
):
    """
    DyT backward kernel (persistent 2D grid, aligned with Triton).

    Grid: (num_col_blocks, NUM_SMS)
    Block (col_block, start_row_id) strides over rows: start_row_id, start_row_id+NUM_SMS, ...
    Writes DG/DB/DA to unique (start_row_id, col) so no atomics; host sums.
    """
    col_block = ct.bid(0)
    start_row_id = ct.bid(1)
    col_start = col_block * BLOCK_N
    col_indices = ct.arange(BLOCK_N, dtype=ct.int32) + col_start

    alpha = ct.astype(ct.load(alpha_tensor, 0, shape=()), ct.float32)
    gamma = ct.astype(ct.gather(gamma_tensor, col_indices, check_bounds=True, padding_value=0.0), ct.float32)

    da_acc = ct.full((BLOCK_N,), 0.0, dtype=ct.float32)  # tile accumulator, reduce once at end
    dg_acc = ct.full((BLOCK_N,), 0.0, dtype=ct.float32)
    if HAVE_BETA:
        db_acc = ct.full((BLOCK_N,), 0.0, dtype=ct.float32)

    # Stride over rows assigned to this block (same as Triton: start_row_id, start_row_id+NUM_SMS, ...)
    num_iters = (M + NUM_SMS - 1) // NUM_SMS
    for i in range(num_iters):
        row_id = start_row_id + i * NUM_SMS
        if row_id < M:
            x = ct.astype(ct.gather(x_input, (row_id, col_indices), check_bounds=True, padding_value=0.0), ct.float32)
            dy = ct.astype(ct.gather(dy_input, (row_id, col_indices), check_bounds=True, padding_value=0.0), ct.float32)

            # APPROX tanh: ~1.6x faster, 2-4 ULP off; well within bwd tolerance 1e-2.
            tanh_x = ct.tanh(alpha * x, rounding_mode=RMd.APPROX)
            sech2_x = ct.full((BLOCK_N,), 1.0, dtype=ct.float32) - tanh_x * tanh_x

            if HAVE_BETA:
                db_acc = db_acc + dy

            dg_acc = dg_acc + dy * tanh_x

            tmp = sech2_x * dy * gamma
            da_acc = da_acc + x * tmp
            dx = alpha * tmp
            ct.scatter(dx_output, (row_id, col_indices), ct.astype(dx, dx_output.dtype), check_bounds=True)

    # Write to unique (start_row_id, col) so host can sum over dim 0
    row_idx_tile = ct.full((BLOCK_N,), start_row_id, dtype=ct.int32)
    ct.scatter(dg_partial, (row_idx_tile, col_indices), dg_acc, check_bounds=True)
    if HAVE_BETA:
        ct.scatter(db_partial, (row_idx_tile, col_indices), db_acc, check_bounds=True)
    # DA: one scalar per block at (start_row_id, col_block) — single reduction at end
    da_scalar = ct.full((1,), ct.sum(da_acc, 0, keepdims=False), dtype=ct.float32)
    ct.scatter(
        da_partial,
        (ct.full((1,), start_row_id, dtype=ct.int32), ct.full((1,), col_block, dtype=ct.int32)),
        da_scalar,
    )


# nww=8 matches Triton's num_warps=8 on this bwd kernel.
_dyt_bwd_kernel_nww8 = _dyt_bwd_kernel.replace_hints(num_worker_warps=8)


def _dyt_forward_ct(x, alpha, gamma, beta):
    HAVE_BETA = beta is not None
    input_shape = x.shape
    x_2d = x.view(-1, input_shape[-1])
    M, N = x_2d.shape

    BLOCK_N = min(MAX_FUSED_SIZE, _next_power_of_2(N))
    num_col_blocks = (N + BLOCK_N - 1) // BLOCK_N
    # Aligned fast path: single col-block + N is power-of-2 → all gathers/scatters
    # are in-bounds, kernel compiles with check_bounds=False (no predicate masks).
    check_bounds = not ((BLOCK_N == N) and (num_col_blocks == 1))

    y = torch.empty_like(x_2d)
    beta_tensor = beta if HAVE_BETA else torch.empty(1, device=x.device, dtype=x.dtype)

    ct.launch(
        torch.cuda.current_stream(),
        (num_col_blocks, M, 1),
        _dyt_fwd_kernel,
        (x_2d, y, alpha, gamma, beta_tensor, int(HAVE_BETA), bool(check_bounds), int(BLOCK_N)),
    )
    return y.view(input_shape)


def _dyt_backward_ct(dy, x, alpha, gamma, beta):
    HAVE_BETA = beta is not None
    input_shape = x.shape
    x_2d = x.view(-1, input_shape[-1])
    dy_2d = dy.view(-1, input_shape[-1])
    M, N = x_2d.shape

    NUM_SMS = torch.cuda.get_device_properties(x.device).multi_processor_count
    BLOCK_N = min(_next_power_of_2(N), 1024)
    num_col_blocks = (N + BLOCK_N - 1) // BLOCK_N

    dx = torch.empty_like(dy_2d)
    # Per-block partials (match Triton): host reduces over dim 0
    da_partial = torch.zeros(NUM_SMS, num_col_blocks, dtype=torch.float32, device=x.device)
    dg_partial = torch.empty(NUM_SMS, N, dtype=torch.float32, device=x.device)
    db_partial = torch.empty(NUM_SMS, N, dtype=torch.float32, device=x.device) if HAVE_BETA else None

    db_tensor = db_partial if HAVE_BETA else torch.empty(1, device=x.device, dtype=torch.float32)

    ct.launch(
        torch.cuda.current_stream(),
        (num_col_blocks, NUM_SMS, 1),
        _dyt_bwd_kernel_nww8,
        (
            dy_2d,
            dx,
            da_partial,
            dg_partial,
            db_tensor,
            x_2d,
            alpha,
            gamma,
            int(HAVE_BETA),
            int(M),
            int(BLOCK_N),
            int(NUM_SMS),
        ),
    )

    da = da_partial.sum().to(x.dtype).unsqueeze(0)
    dg = dg_partial.sum(0).to(gamma.dtype)
    db = db_partial.sum(0).to(x.dtype) if HAVE_BETA else None
    return dx.view(input_shape), da, dg, db


class LigerDyTFunction(torch.autograd.Function):
    """CuTile autograd wrapper for Dynamic Tanh: y = tanh(alpha * x) * gamma + beta."""

    @staticmethod
    def forward(ctx, x, alpha, gamma, beta):
        x = x.contiguous()
        alpha = alpha.contiguous()
        gamma = gamma.contiguous()
        if beta is not None:
            beta = beta.contiguous()
        y = _dyt_forward_ct(x, alpha, gamma, beta)
        ctx.save_for_backward(x, alpha, gamma, beta)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, alpha, gamma, beta = ctx.saved_tensors
        dy = dy.contiguous()
        dx, dalpha, dgamma, dbeta = _dyt_backward_ct(dy, x, alpha, gamma, beta)
        return dx, dalpha, dgamma, dbeta
