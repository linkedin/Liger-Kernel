# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
PolyNorm normalization kernel (cuTile backend).

Formula: y = w0*norm(x^3) + w1*norm(x^2) + w2*norm(x) + b,  norm(u) = u / sqrt(mean(u^2) + eps)

Forward is a 2-pass row-parallel kernel: pass 1 accumulates sum-of-squares per power (using
x^6 = x^4 * x^2 inline) and caches rstd; pass 2 applies the output in Horner form
(y = x2*(w0r3*x + w1r2) + w2r1*x + b) to avoid materialising x^3. Backward computes the
closed-form gradient and atomically reduces dW/dB into a 4-element buffer (no host .sum pass).
Aligned (power-of-2 n_cols) paths use check_bounds=False.
"""

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2

MAX_FUSED_SIZE = 65536


@ct.kernel(occupancy=4)
def _poly_norm_fwd_kernel(
    x_input,  # (n_rows, n_cols) input
    y_output,  # (n_rows, n_cols) output
    weights,  # (3,) weights [w0, w1, w2]
    bias,  # (1,) scalar bias
    rstd3,  # (n_rows,) cached rstd for x^3 power
    rstd2,  # (n_rows,) cached rstd for x^2 power
    rstd1,  # (n_rows,) cached rstd for x^1 power
    N_COLS: ct.Constant[int],
    eps,
    BLOCK_SIZE: ct.Constant[int],
    ALIGNED: ct.Constant[bool],
):
    """
    PolyNorm forward kernel (row-parallel).

    Two passes per row:
    1. Accumulate sum-of-squares for each power, compute rstd, cache it.
    2. Compute output y = w0*norm(x^3) + w1*norm(x^2) + w2*norm(x) + b.

    ALIGNED=True: n_cols is a power of 2, so BLOCK_SIZE==n_cols and no partial
    chunk exists. Uses check_bounds=False (hardware TMA path, ~10% faster).
    ALIGNED=False: general case with software bounds checking.
    """
    row_idx = ct.bid(0)

    # Load scalar weights and bias
    w0 = ct.astype(ct.load(weights, 0, shape=()), ct.float32)
    w1 = ct.astype(ct.load(weights, 1, shape=()), ct.float32)
    w2 = ct.astype(ct.load(weights, 2, shape=()), ct.float32)
    b = ct.astype(ct.load(bias, 0, shape=()), ct.float32)

    # Pass 1: accumulate sum-of-squares for each power using "fold" trick
    # sum_sq_p[i] accumulates sum over col-chunks at position i within BLOCK_SIZE
    sum_sq_3 = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    sum_sq_2 = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    sum_sq_1 = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

    num_chunks = (N_COLS + BLOCK_SIZE - 1) // BLOCK_SIZE
    for ci in range(num_chunks):
        col_indices = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
        if ALIGNED:
            x_tile = ct.gather(x_input, (row_idx, col_indices), check_bounds=False)
        else:
            x_tile = ct.gather(x_input, (row_idx, col_indices), check_bounds=True, padding_value=0.0)
        x_f32 = ct.astype(x_tile, ct.float32)
        # x4-path: reuse x2 for sum_sq_1 and x4 for sum_sq_2; x^6 = x4*x2 inline.
        # Saves 2 FMUL/element vs computing x3=x^3 separately.
        x2 = x_f32 * x_f32  # x^2
        x4 = x2 * x2  # x^4
        sum_sq_1 = sum_sq_1 + x2  # sum x^2
        sum_sq_2 = sum_sq_2 + x4  # sum x^4
        sum_sq_3 = sum_sq_3 + x4 * x2  # sum x^6 = x^4 * x^2

    # Compute rstd = rsqrt(mean(u^2) + eps) for each power
    rstd_3 = ct.rsqrt(ct.sum(sum_sq_3, axis=0, keepdims=False) / N_COLS + eps)
    rstd_2 = ct.rsqrt(ct.sum(sum_sq_2, axis=0, keepdims=False) / N_COLS + eps)
    rstd_1 = ct.rsqrt(ct.sum(sum_sq_1, axis=0, keepdims=False) / N_COLS + eps)

    # Cache rstd values for backward pass
    ct.scatter(rstd3, row_idx, rstd_3)
    ct.scatter(rstd2, row_idx, rstd_2)
    ct.scatter(rstd1, row_idx, rstd_1)

    # Precompute loop-invariant scalar products (w × rstd) before pass 2.
    # Horner form: y = x2*(w0r3*x + w1r2) + w2r1*x + b  — avoids materialising x3 tile.
    w0r3 = w0 * rstd_3
    w1r2 = w1 * rstd_2
    w2r1 = w2 * rstd_1

    # Pass 2: compute output
    for ci in range(num_chunks):
        col_indices = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
        if ALIGNED:
            x_tile = ct.gather(x_input, (row_idx, col_indices), check_bounds=False)
        else:
            x_tile = ct.gather(x_input, (row_idx, col_indices), check_bounds=True, padding_value=0.0)
        x_f32 = ct.astype(x_tile, ct.float32)
        x2 = x_f32 * x_f32
        # Horner: y = x2*(w0r3*x + w1r2) + w2r1*x + b
        inner = w0r3 * x_f32 + w1r2
        y_f32 = x2 * inner + w2r1 * x_f32 + b
        if ALIGNED:
            ct.scatter(y_output, (row_idx, col_indices), ct.astype(y_f32, x_tile.dtype), check_bounds=False)
        else:
            ct.scatter(y_output, (row_idx, col_indices), ct.astype(y_f32, x_tile.dtype), check_bounds=True)


_poly_norm_fwd_kernel_occ8 = _poly_norm_fwd_kernel.replace_hints(occupancy=8)


@ct.kernel(occupancy=4)
def _poly_norm_fwd_kernel_sc_large(
    x_input,
    y_output,
    weights,
    bias,
    rstd3,
    rstd2,
    rstd1,
    N_COLS: ct.Constant[int],
    eps,
    BLOCK_SIZE: ct.Constant[int],
    ALIGNED: ct.Constant[bool],
):
    """Single-chunk: re-gather x in pass2, latency=2 on both gathers."""
    row_idx = ct.bid(0)

    w0 = ct.astype(ct.load(weights, 0, shape=()), ct.float32)
    w1 = ct.astype(ct.load(weights, 1, shape=()), ct.float32)
    w2 = ct.astype(ct.load(weights, 2, shape=()), ct.float32)
    b = ct.astype(ct.load(bias, 0, shape=()), ct.float32)

    col_indices = ct.arange(BLOCK_SIZE, dtype=ct.int32)
    if ALIGNED:
        x_tile = ct.gather(x_input, (row_idx, col_indices), check_bounds=False, latency=2)
    else:
        x_tile = ct.gather(x_input, (row_idx, col_indices), check_bounds=True, padding_value=0.0, latency=2)
    x_f32 = ct.astype(x_tile, ct.float32)
    x2 = x_f32 * x_f32
    x4 = x2 * x2

    rstd_3 = ct.rsqrt(ct.sum(x4 * x2, axis=0, keepdims=False) / N_COLS + eps)
    rstd_2 = ct.rsqrt(ct.sum(x4, axis=0, keepdims=False) / N_COLS + eps)
    rstd_1 = ct.rsqrt(ct.sum(x2, axis=0, keepdims=False) / N_COLS + eps)

    ct.scatter(rstd3, row_idx, rstd_3)
    ct.scatter(rstd2, row_idx, rstd_2)
    ct.scatter(rstd1, row_idx, rstd_1)

    w0r3 = w0 * rstd_3
    w1r2 = w1 * rstd_2
    w2r1 = w2 * rstd_1

    if ALIGNED:
        x_tile2 = ct.gather(x_input, (row_idx, col_indices), check_bounds=False, latency=2)
    else:
        x_tile2 = ct.gather(x_input, (row_idx, col_indices), check_bounds=True, padding_value=0.0, latency=2)
    x2_f32 = ct.astype(x_tile2, ct.float32)
    x2sq = x2_f32 * x2_f32
    inner = w0r3 * x2_f32 + w1r2
    y_f32 = x2sq * inner + w2r1 * x2_f32 + b
    if ALIGNED:
        ct.scatter(y_output, (row_idx, col_indices), ct.astype(y_f32, x_tile2.dtype), check_bounds=False)
    else:
        ct.scatter(y_output, (row_idx, col_indices), ct.astype(y_f32, x_tile2.dtype), check_bounds=True)


_poly_norm_fwd_kernel_sc_large_occ4 = _poly_norm_fwd_kernel_sc_large.replace_hints(occupancy=4)
_poly_norm_fwd_kernel_sc_large_occ8 = _poly_norm_fwd_kernel_sc_large.replace_hints(occupancy=8)
_poly_norm_fwd_kernel_sc_large_occ16 = _poly_norm_fwd_kernel_sc_large.replace_hints(occupancy=16)


@ct.kernel(occupancy=2)
def _poly_norm_bwd_kernel(
    dy,  # (n_rows, n_cols) output gradient
    dx,  # (n_rows, n_cols) input gradient (output)
    x_input,  # (n_rows, n_cols) saved input
    weights,  # (3,) weights [w0, w1, w2]
    rstd3,  # (n_rows,) cached rstd for x^3 power
    rstd2,  # (n_rows,) cached rstd for x^2 power
    rstd1,  # (n_rows,) cached rstd for x^1 power
    dwdb_output,  # (4,) global atomic reduction target [dW0, dW1, dW2, dB]
    N_COLS: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    ALIGNED: ct.Constant[bool],
):
    """
    PolyNorm backward kernel (row-parallel).

    Two passes per row:
    1. Compute S_p = sum(dy * x^p) for each power p, and dB = sum(dy).
    2. Compute dx using closed-form gradient formula.

    dW/dB contributions are atomically reduced into a single (4,) output tensor
    directly inside the kernel — eliminates a separate host-side .sum(dim=1)
    + .to(W.dtype) launch chain.

    ALIGNED=True uses check_bounds=False (hardware TMA, ~10% faster) when
    n_cols is a power of 2 and BLOCK_SIZE==n_cols.
    """
    row_idx = ct.bid(0)

    # Load weights and cached rstd values
    w0 = ct.astype(ct.load(weights, 0, shape=()), ct.float32)
    w1 = ct.astype(ct.load(weights, 1, shape=()), ct.float32)
    w2 = ct.astype(ct.load(weights, 2, shape=()), ct.float32)

    rstd_3 = ct.load(rstd3, row_idx, shape=())
    rstd_2 = ct.load(rstd2, row_idx, shape=())
    rstd_1 = ct.load(rstd1, row_idx, shape=())

    S_3_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    S_2_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    S_1_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    dB_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

    num_chunks = (N_COLS + BLOCK_SIZE - 1) // BLOCK_SIZE
    for ci in range(num_chunks):
        col_indices = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
        if ALIGNED:
            dy_t = ct.astype(ct.gather(dy, (row_idx, col_indices), check_bounds=False), ct.float32)
            x = ct.astype(ct.gather(x_input, (row_idx, col_indices), check_bounds=False), ct.float32)
        else:
            dy_t = ct.astype(ct.gather(dy, (row_idx, col_indices), check_bounds=True, padding_value=0.0), ct.float32)
            x = ct.astype(ct.gather(x_input, (row_idx, col_indices), check_bounds=True, padding_value=0.0), ct.float32)
        S_3_acc = S_3_acc + dy_t * (x * x * x)
        S_2_acc = S_2_acc + dy_t * (x * x)
        S_1_acc = S_1_acc + dy_t * x
        dB_acc = dB_acc + dy_t

    S_3 = ct.sum(S_3_acc, axis=0, keepdims=False)
    S_2 = ct.sum(S_2_acc, axis=0, keepdims=False)
    S_1 = ct.sum(S_1_acc, axis=0, keepdims=False)
    dB_row = ct.sum(dB_acc, axis=0, keepdims=False)

    ct.atomic_add(dwdb_output, 0, rstd_3 * S_3, check_bounds=False)
    ct.atomic_add(dwdb_output, 1, rstd_2 * S_2, check_bounds=False)
    ct.atomic_add(dwdb_output, 2, rstd_1 * S_1, check_bounds=False)
    ct.atomic_add(dwdb_output, 3, dB_row, check_bounds=False)

    for ci in range(num_chunks):
        col_indices = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
        if ALIGNED:
            dy_t = ct.astype(ct.gather(dy, (row_idx, col_indices), check_bounds=False), ct.float32)
            x = ct.astype(ct.gather(x_input, (row_idx, col_indices), check_bounds=False), ct.float32)
        else:
            dy_t = ct.astype(ct.gather(dy, (row_idx, col_indices), check_bounds=True, padding_value=0.0), ct.float32)
            x = ct.astype(ct.gather(x_input, (row_idx, col_indices), check_bounds=True, padding_value=0.0), ct.float32)
        x2 = x * x
        x3 = x2 * x

        rstd3_cu = rstd_3 * rstd_3 * rstd_3
        grad_3 = w0 * (3.0 * x2 * rstd_3 * dy_t - (3.0 / N_COLS) * x2 * x3 * rstd3_cu * S_3)

        rstd2_cu = rstd_2 * rstd_2 * rstd_2
        grad_2 = w1 * (2.0 * x * rstd_2 * dy_t - (2.0 / N_COLS) * x3 * rstd2_cu * S_2)

        rstd1_cu = rstd_1 * rstd_1 * rstd_1
        grad_1 = w2 * (rstd_1 * dy_t - (1.0 / N_COLS) * x * rstd1_cu * S_1)

        dx_t = grad_3 + grad_2 + grad_1
        if ALIGNED:
            ct.scatter(dx, (row_idx, col_indices), ct.astype(dx_t, dx.dtype), check_bounds=False)
        else:
            ct.scatter(dx, (row_idx, col_indices), ct.astype(dx_t, dx.dtype), check_bounds=True)


# Large-N variant (n_cols >= 8192): higher occupancy + 8 worker warps to widen
# parallelism on latency-bound 2-pass multi-chunk path.
_poly_norm_bwd_kernel_large = _poly_norm_bwd_kernel.replace_hints(occupancy=4, num_worker_warps=8)


class LigerPolyNormFunction(torch.autograd.Function):
    """
    PolyNorm autograd function with CuTile forward and backward kernels.

    Formula: y = w0·norm(x^3) + w1·norm(x^2) + w2·norm(x) + b, norm(u) = u / sqrt(mean(u^2) + eps).
    in_place is accepted for signature parity with the Triton implementation but ignored (the
    cuTile forward always writes a fresh output).
    """

    @staticmethod
    def forward(ctx, X, W, B, eps=1e-6, in_place=True):
        shape = X.shape
        dim = shape[-1]
        X_2d = X.contiguous().view(-1, dim)
        n_rows, n_cols = X_2d.shape

        # B is a scalar bias — accept 0-dim (torch.tensor(1.0)) or (1,); the kernel reads
        # bias[0], so flatten to a 1-element tensor and restore B's shape for the dB gradient.
        bias_shape = B.shape
        B = B.reshape(1)

        # Per-shape BLOCK_SIZE & kernel variant chosen to match OAIT/NVT autotune
        # picks. Single-chunk uses the SC kernel (re-gather, lower live-tile
        # pressure); multi-chunk uses the fold-accumulator kernel.
        # n_cols=2048 keeps BLOCK=1024 (multi-chunk) because BLOCK=2048 single-chunk
        # spills the 3 fp32 accumulators (sum_sq_3/2/1).
        if n_cols <= 1024:
            BLOCK_SIZE = _next_power_of_2(n_cols)
        elif n_cols == 2048:
            BLOCK_SIZE = 1024
        elif n_cols <= 8192:
            BLOCK_SIZE = 4096
        else:
            BLOCK_SIZE = min(MAX_FUSED_SIZE, _next_power_of_2(n_cols))
        aligned = (n_cols % BLOCK_SIZE) == 0
        single_chunk = n_cols <= BLOCK_SIZE

        if n_cols <= 1024:
            fwd_kernel = _poly_norm_fwd_kernel_sc_large_occ16
        elif n_cols == 2048:
            fwd_kernel = _poly_norm_fwd_kernel
        elif n_cols == 4096:
            fwd_kernel = _poly_norm_fwd_kernel_sc_large_occ8
        elif n_cols <= 8192:
            fwd_kernel = _poly_norm_fwd_kernel_occ8
        elif single_chunk:
            fwd_kernel = _poly_norm_fwd_kernel_sc_large_occ4
        else:
            fwd_kernel = _poly_norm_fwd_kernel

        Y = torch.empty_like(X_2d)
        RSTD3 = torch.empty(n_rows, dtype=torch.float32, device=X.device)
        RSTD2 = torch.empty(n_rows, dtype=torch.float32, device=X.device)
        RSTD1 = torch.empty(n_rows, dtype=torch.float32, device=X.device)

        grid = (n_rows, 1, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            fwd_kernel,
            (
                X_2d,
                Y,
                W.contiguous(),
                B.contiguous(),
                RSTD3,
                RSTD2,
                RSTD1,
                int(n_cols),
                float(eps),
                int(BLOCK_SIZE),
                bool(aligned),
            ),
        )

        ctx.save_for_backward(X_2d, W, RSTD3, RSTD2, RSTD1)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.aligned = aligned
        ctx.shape = shape
        ctx.bias_shape = bias_shape

        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dy):
        X_2d, W, RSTD3, RSTD2, RSTD1 = ctx.saved_tensors
        shape = ctx.shape

        dim = shape[-1]
        dY_2d = dy.contiguous().view(-1, dim)
        n_rows, n_cols = dY_2d.shape

        # Bwd has 4 fp32 accumulators (S_3, S_2, S_1, dB) of BLOCK_SIZE each.
        # Cap BLOCK_SIZE at 4096 to keep accumulators in registers (16KB per tile)
        # and avoid spills observed at BLOCK_SIZE >= 16384 on norm-like bwd kernels.
        BWD_MAX_BLOCK = 4096
        BLOCK_SIZE = min(BWD_MAX_BLOCK, _next_power_of_2(n_cols))
        aligned = (n_cols % BLOCK_SIZE) == 0

        dx = torch.empty_like(dY_2d)
        # The kernel atomically accumulates dW/dB contributions across all rows
        # directly into this 4-element fp32 buffer — eliminates the host-side
        # .sum(dim=1) + .to(W.dtype) launch chain.
        dwdb_output = torch.zeros(4, dtype=torch.float32, device=W.device)

        # Medium/large-N (n_cols>=8192) is latency-bound on the 2-pass multi-chunk
        # IR; the occ=4/nww=8 variant closes the gap vs OAIT.
        kernel_choice = _poly_norm_bwd_kernel_large if n_cols >= 8192 else _poly_norm_bwd_kernel

        grid = (n_rows, 1, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            kernel_choice,
            (
                dY_2d,
                dx,
                X_2d,
                W,
                RSTD3,
                RSTD2,
                RSTD1,
                dwdb_output,
                int(n_cols),
                int(BLOCK_SIZE),
                bool(aligned),
            ),
        )

        sums = dwdb_output.to(W.dtype)  # (4,) cast to W.dtype
        dW = sums[:3]
        dB = sums[3:4].reshape(ctx.bias_shape)  # match B's original shape (scalar or (1,))

        return dx.view(*shape), dW, dB, None, None
