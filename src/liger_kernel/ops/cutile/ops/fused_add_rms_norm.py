# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Fused residual-add + RMSNorm (cuTile backend).

Forward: S = X + R, Y = S / RMS(S) * (W + offset). Both Y and the updated residual S are
returned. Row-parallel; single-pass for one chunk, two-pass for multi-chunk; forward occupancy
is autotuned per shape.

Backward: persistent grid (one block per SM), each block accumulates dW in registers across its
rows and writes it once. A 2-chunk variant (splitting n_cols into lo/hi halves) caps per-thread
register pressure when BLOCK_SIZE exceeds _BWD_MAX_CHUNK_SIZE. dR == dX (gradient flows equally
to X and R).

Casting modes match the Triton implementation: "llama" (fp32 RMS only), "gemma" (full fp32),
"none" (compute in input dtype). in_place is accepted for signature parity but ignored.
"""

import math

from types import SimpleNamespace

import cuda.tile as ct
import torch

from cuda.tile.tune import exhaustive_search

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.utils import ensure_contiguous

_CASTING_MODE_NONE = -1
_CASTING_MODE_LLAMA = 0
_CASTING_MODE_GEMMA = 1

_STR_TO_CASTING_MODE = {
    "llama": _CASTING_MODE_LLAMA,
    "gemma": _CASTING_MODE_GEMMA,
    "none": _CASTING_MODE_NONE,
}

# _BWD_MAX_CHUNK_SIZE: threshold for switching to the 2-chunk persistent kernel.
# When BLOCK_SIZE (= next_power_of_2(n_cols)) exceeds this value, the backward
# dispatch uses _fused_add_rms_norm_bwd_persistent_2c_ct with CHUNK_SIZE = BLOCK_SIZE//2.
# This caps per-thread register usage at CHUNK_SIZE/128 elements per tile (32 f32/thread
# at CHUNK_SIZE=4096), reducing peak pressure from 87% -> 56% of the B200 budget.
_BWD_MAX_CHUNK_SIZE = 4096


def calculate_settings(n_cols):
    BLOCK_SIZE = _next_power_of_2(n_cols)
    if BLOCK_SIZE > 65536:
        raise RuntimeError(f"Hidden dimension {n_cols} exceeds maximum supported size of 65536.")
    return BLOCK_SIZE


@ct.kernel
def _fused_add_rms_norm_fwd_ct(
    Y,  # (n_rows, n_cols) normalized output
    S,  # (n_rows, n_cols) updated residual (X + R)
    X,  # (n_rows, n_cols) hidden states input
    R,  # (n_rows, n_cols) residual input
    W,  # (n_cols,) RMSNorm weight
    RSTD,  # (n_rows,) cached rstd (scalar per row)
    n_cols: ct.Constant[int],
    eps: ct.Constant[float],
    offset: ct.Constant[float],
    BLOCK_SIZE: ct.Constant[int],
    casting_mode: ct.Constant[int],
):
    """
    Forward: S = X + R, Y = S * rstd * (W + offset).

    Row-parallel: one block per row. Two passes over columns.

    Casting modes:
      _CASTING_MODE_LLAMA  (0): fp32 RMS; S*rstd cast back to X.dtype before W multiply
      _CASTING_MODE_GEMMA  (1): full fp32, Y cast back to X.dtype at end (original behavior)
      _CASTING_MODE_NONE  (-1): no casting, compute in X.dtype throughout
    """
    row_idx = ct.bid(0)
    num_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    if num_chunks == 1:
        # Single-pass: mirrors Triton's structure — S stays in registers across the
        # rstd compute, so pass 2 doesn't re-load it from DRAM. Reassigning S_tile
        # through dtype transitions drops the previous register footprint.
        col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32)
        S_tile = ct.add(
            ct.gather(X, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
            ct.gather(R, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
        )
        ct.scatter(S, (row_idx, col_idx), S_tile, check_bounds=True)
        W_tile = ct.gather(W, col_idx, check_bounds=True, padding_value=0.0)

        if casting_mode == _CASTING_MODE_LLAMA:
            S_tile = ct.astype(S_tile, ct.float32)
            rstd = ct.rsqrt(ct.sum(ct.mul(S_tile, S_tile), 0, keepdims=False) / n_cols + eps)
            ct.scatter(RSTD, row_idx, rstd)
            S_tile = ct.astype(ct.mul(S_tile, rstd), X.dtype)
            # W may be higher precision than X (e.g. fp32 norm weight, bf16 activations);
            # the multiply promotes to W's dtype, so cast back to Y.dtype before storing.
            ct.scatter(Y, (row_idx, col_idx), ct.astype(ct.mul(S_tile, ct.add(W_tile, offset)), Y.dtype), check_bounds=True)
        elif casting_mode == _CASTING_MODE_GEMMA:
            S_tile = ct.astype(S_tile, ct.float32)
            rstd = ct.rsqrt(ct.sum(ct.mul(S_tile, S_tile), 0, keepdims=False) / n_cols + eps)
            ct.scatter(RSTD, row_idx, rstd)
            W_shifted = ct.add(ct.astype(W_tile, ct.float32), offset)
            Y_f32 = ct.mul(ct.mul(S_tile, rstd), W_shifted)
            ct.scatter(Y, (row_idx, col_idx), ct.astype(Y_f32, Y.dtype), check_bounds=True)
        else:
            # NONE: compute mean_sq in X.dtype (then promote to fp32 at division),
            # store rstd as X.dtype.
            mean_sq = ct.sum(ct.mul(S_tile, S_tile), 0, keepdims=False) / n_cols
            rstd = ct.rsqrt(mean_sq + eps)
            ct.scatter(RSTD, row_idx, rstd)
            ct.scatter(
                Y,
                (row_idx, col_idx),
                ct.astype(ct.mul(ct.mul(S_tile, rstd), ct.add(W_tile, offset)), Y.dtype),
                check_bounds=True,
            )
        return

    # ---- Multi-chunk path (BLOCK_SIZE chunked from n_cols, num_chunks >= 2) ----
    # Two-pass loop: pass 1 computes S, scatters S, accumulates sum(S^2).
    # Pass 2 re-loads S to compute Y (unavoidable: tiles don't fit in registers across chunks).
    if casting_mode == _CASTING_MODE_NONE:
        sum_sq_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=X.dtype)
    else:
        sum_sq_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

    for ci in range(num_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        X_tile = ct.gather(X, (row_idx, col_idx), check_bounds=True, padding_value=0.0)
        R_tile = ct.gather(R, (row_idx, col_idx), check_bounds=True, padding_value=0.0)
        S_tile = ct.add(X_tile, R_tile)
        ct.scatter(S, (row_idx, col_idx), S_tile, check_bounds=True)
        if casting_mode == _CASTING_MODE_NONE:
            sum_sq_tile = ct.add(sum_sq_tile, ct.mul(S_tile, S_tile))
        else:
            S_tile_f32 = ct.astype(S_tile, ct.float32)
            sum_sq_tile = ct.add(sum_sq_tile, ct.mul(S_tile_f32, S_tile_f32))

    mean_sq = ct.sum(sum_sq_tile, 0, keepdims=False) / n_cols
    rstd = ct.rsqrt(mean_sq + eps)
    ct.scatter(RSTD, row_idx, rstd)

    for ci in range(num_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        S_tile = ct.gather(S, (row_idx, col_idx), check_bounds=True, padding_value=0.0)
        W_tile = ct.gather(W, col_idx, check_bounds=True, padding_value=0.0)

        if casting_mode == _CASTING_MODE_LLAMA:
            S_tile = ct.astype(S_tile, ct.float32)
            S_normed = ct.astype(ct.mul(S_tile, rstd), X.dtype)
            ct.scatter(Y, (row_idx, col_idx), ct.astype(ct.mul(S_normed, ct.add(W_tile, offset)), Y.dtype), check_bounds=True)
        elif casting_mode == _CASTING_MODE_GEMMA:
            S_tile = ct.astype(S_tile, ct.float32)
            W_shifted = ct.add(ct.astype(W_tile, ct.float32), offset)
            Y_f32 = ct.mul(ct.mul(S_tile, rstd), W_shifted)
            ct.scatter(Y, (row_idx, col_idx), ct.astype(Y_f32, Y.dtype), check_bounds=True)
        else:
            ct.scatter(
                Y,
                (row_idx, col_idx),
                ct.astype(ct.mul(ct.mul(S_tile, rstd), ct.add(W_tile, offset)), Y.dtype),
                check_bounds=True,
            )


_fused_add_rms_norm_fwd_large_ct = _fused_add_rms_norm_fwd_ct.replace_hints(num_worker_warps=8)
# Small BLOCK_SIZE: Triton's calculate_settings picks num_warps=4 for BLOCK_SIZE<2048.
# Matching that avoids over-warping when each thread only has 4 fp32/tile of work.
_fused_add_rms_norm_fwd_small_ct = _fused_add_rms_norm_fwd_ct.replace_hints(num_worker_warps=4)


# Per-shape autotune for fwd occupancy. Higher occupancy forces the compiler to budget
# fewer regs/thread (target: 1/occ of the SM register file). Sweet spot is shape-
# dependent — tiny tiles at small N benefit from occ>=8, large tiles spill above occ=1.
def _fwd_autotune_configs():
    for occ in (None, 2, 3, 4, 5, 6, 8):
        yield SimpleNamespace(occupancy=occ)


_fwd_tune_cache: dict = {}


def _autotune_fwd_kernel(base_kernel, args, n_rows, cache_key, stream):
    if cache_key in _fwd_tune_cache:
        return _fwd_tune_cache[cache_key]
    result = exhaustive_search(
        list(_fwd_autotune_configs()),
        stream,
        lambda cfg: (n_rows, 1, 1),
        base_kernel,
        lambda cfg: args,
        lambda cfg: {"occupancy": cfg.occupancy} if cfg.occupancy is not None else {},
        quiet=True,
    )
    best = result.best.config
    tuned = base_kernel.replace_hints(occupancy=best.occupancy) if best.occupancy is not None else base_kernel
    _fwd_tune_cache[cache_key] = tuned
    return tuned


@ct.kernel(occupancy=1)
def _fused_add_rms_norm_bwd_persistent_ct(
    dY,  # (n_rows, n_cols) gradient of Y
    dS_out,  # (n_rows, n_cols) gradient of S flowing from downstream
    dX,  # (n_rows, n_cols) output gradient (also used for dR)
    S,  # (n_rows, n_cols) saved residual S = X + R from forward
    W,  # (n_cols,) RMSNorm weight
    RSTD,  # (n_rows,) cached rstd from forward
    dW_partial,  # (sm_count, n_cols) per-SM dW, host reduces with sum(dim=0)
    n_cols: ct.Constant[int],
    offset: ct.Constant[float],
    num_iters: ct.Constant[int],
    sm_count: ct.Constant[int],
    n_rows: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],  # = next_power_of_2(n_cols) — no cap
    casting_mode: ct.Constant[int],
):
    """
    Persistent backward: grid = (sm_count,).
    Each block handles ceil(n_rows / sm_count) rows in a blocked loop:
      row_idx = sm_id * num_iters + i   for i in range(num_iters)

    Casting modes:
      _CASTING_MODE_LLAMA  (0): m = (dY * W).to(f32); dW uses cast-back intermediate
      _CASTING_MODE_GEMMA  (1): dY cast to f32 first, then full fp32 (original behavior)
      _CASTING_MODE_NONE  (-1): compute in S.dtype throughout
    """
    sm_id = ct.bid(0)
    col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32)

    # Hoist W load outside the row loop: W is row-invariant, load it once per SM.
    W_tile = ct.gather(W, col_idx, check_bounds=True, padding_value=0.0)
    if casting_mode == _CASTING_MODE_NONE:
        W_shifted = ct.add(W_tile, offset)
    else:
        W_shifted = ct.add(ct.astype(W_tile, ct.float32), offset)

    # Single register-resident dW accumulator (always float32 for numerical stability).
    dW_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    inv_n_cols = 1.0 / n_cols

    row_start = sm_id * num_iters
    for i in range(num_iters):
        row_idx = row_start + i

        if row_idx < n_rows:
            rstd = ct.load(RSTD, row_idx, shape=())  # scalar

            dY_tile = ct.gather(dY, (row_idx, col_idx), check_bounds=True, padding_value=0.0, latency=3)
            S_tile = ct.gather(S, (row_idx, col_idx), check_bounds=True, padding_value=0.0, latency=3)

            if casting_mode == _CASTING_MODE_LLAMA:
                # m = (dY * W_orig_dtype).to(f32)
                m_tile = ct.astype(ct.mul(dY_tile, ct.astype(W_shifted, dY.dtype)), ct.float32)
                S_tile_f32 = ct.astype(S_tile, ct.float32)
                sum_mS = ct.sum(ct.mul(m_tile, S_tile_f32), 0, keepdims=False)
                # dW: dY * (S * rstd).to(S.dtype)
                dW_acc = ct.add(
                    dW_acc, ct.astype(ct.mul(dY_tile, ct.astype(ct.mul(S_tile_f32, rstd), S.dtype)), ct.float32)
                )
                dS_out_tile = ct.astype(
                    ct.gather(dS_out, (row_idx, col_idx), check_bounds=True, padding_value=0.0, latency=3), ct.float32
                )
                rstd3_coeff = rstd * rstd * rstd * inv_n_cols * sum_mS
                dX_tile = ct.add(ct.sub(ct.mul(rstd, m_tile), ct.mul(rstd3_coeff, S_tile_f32)), dS_out_tile)
                ct.scatter(dX, (row_idx, col_idx), ct.astype(dX_tile, dX.dtype), check_bounds=True)
            elif casting_mode == _CASTING_MODE_GEMMA:
                dY_tile_f32 = ct.astype(dY_tile, ct.float32)
                S_tile_f32 = ct.astype(S_tile, ct.float32)
                m_tile = ct.mul(dY_tile_f32, W_shifted)
                sum_mS = ct.sum(ct.mul(m_tile, S_tile_f32), 0, keepdims=False)
                dW_acc = ct.add(dW_acc, ct.mul(dY_tile_f32, ct.mul(S_tile_f32, rstd)))
                dS_out_tile = ct.astype(
                    ct.gather(dS_out, (row_idx, col_idx), check_bounds=True, padding_value=0.0, latency=3), ct.float32
                )
                rstd3_coeff = rstd * rstd * rstd * inv_n_cols * sum_mS
                dX_tile = ct.add(ct.sub(ct.mul(rstd, m_tile), ct.mul(rstd3_coeff, S_tile_f32)), dS_out_tile)
                ct.scatter(dX, (row_idx, col_idx), ct.astype(dX_tile, dX.dtype), check_bounds=True)
            else:
                # _CASTING_MODE_NONE: compute in S.dtype
                m_tile = ct.mul(dY_tile, W_shifted)
                sum_mS = ct.sum(ct.astype(ct.mul(m_tile, S_tile), ct.float32), 0, keepdims=False)
                dW_acc = ct.add(dW_acc, ct.astype(ct.mul(dY_tile, ct.mul(S_tile, rstd)), ct.float32))
                dS_out_tile = ct.gather(dS_out, (row_idx, col_idx), check_bounds=True, padding_value=0.0, latency=3)
                rstd3_coeff = rstd * rstd * rstd * inv_n_cols * sum_mS
                dX_tile = ct.add(
                    ct.sub(
                        ct.mul(rstd, m_tile), ct.astype(ct.mul(rstd3_coeff, ct.astype(S_tile, ct.float32)), S.dtype)
                    ),
                    dS_out_tile,
                )
                ct.scatter(dX, (row_idx, col_idx), ct.astype(dX_tile, dX.dtype), check_bounds=True)

    # Write register-accumulated dW to global memory — exactly ONCE per SM.
    ct.scatter(dW_partial, (sm_id, col_idx), dW_acc, check_bounds=True)


@ct.kernel(occupancy=1)
def _fused_add_rms_norm_bwd_persistent_2c_ct(
    dY,  # (n_rows, n_cols) gradient of Y
    dS_out,  # (n_rows, n_cols) gradient of S flowing from downstream
    dX,  # (n_rows, n_cols) output gradient (also used for dR)
    S,  # (n_rows, n_cols) saved residual S = X + R from forward
    W,  # (n_cols,) RMSNorm weight
    RSTD,  # (n_rows,) cached rstd from forward
    dW_partial,  # (sm_count, n_cols) per-SM dW, host reduces with sum(dim=0)
    n_cols: ct.Constant[int],
    offset: ct.Constant[float],
    num_iters: ct.Constant[int],
    sm_count: ct.Constant[int],
    n_rows: ct.Constant[int],
    CHUNK_SIZE: ct.Constant[int],  # = BLOCK_SIZE // 2 = next_power_of_2(n_cols) // 2
    casting_mode: ct.Constant[int],
):
    """
    2-chunk persistent backward: used when BLOCK_SIZE > _BWD_MAX_CHUNK_SIZE.

    Splits the n_cols dimension into two halves (lo=0..CHUNK_SIZE-1,
    hi=CHUNK_SIZE..2*CHUNK_SIZE-1) processed together in a single pass per row.
    Each half uses a separate set of register tiles, capping per-thread register
    usage at CHUNK_SIZE/128 elements (32 f32/thread at CHUNK_SIZE=4096) vs.
    BLOCK_SIZE/128 in the 1-chunk kernel (64 f32/thread for n_cols=8192).
    """
    sm_id = ct.bid(0)

    # Column index tiles for each half
    col_idx_lo = ct.arange(CHUNK_SIZE, dtype=ct.int32)
    col_idx_hi = ct.add(ct.arange(CHUNK_SIZE, dtype=ct.int32), CHUNK_SIZE)

    # Hoist W loads outside the row loop (W is row-invariant).
    W_tile_lo_raw = ct.gather(W, col_idx_lo, check_bounds=True, padding_value=0.0)
    W_tile_hi_raw = ct.gather(W, col_idx_hi, check_bounds=True, padding_value=0.0)
    if casting_mode == _CASTING_MODE_NONE:
        W_shifted_lo = ct.add(W_tile_lo_raw, offset)
        W_shifted_hi = ct.add(W_tile_hi_raw, offset)
    else:
        W_shifted_lo = ct.add(ct.astype(W_tile_lo_raw, ct.float32), offset)
        W_shifted_hi = ct.add(ct.astype(W_tile_hi_raw, ct.float32), offset)

    # Separate dW accumulators for lo and hi halves.
    dW_acc_lo = ct.full((CHUNK_SIZE,), 0.0, dtype=ct.float32)
    dW_acc_hi = ct.full((CHUNK_SIZE,), 0.0, dtype=ct.float32)
    inv_n_cols = 1.0 / n_cols

    row_start = sm_id * num_iters
    for i in range(num_iters):
        row_idx = row_start + i

        if row_idx < n_rows:
            rstd = ct.load(RSTD, row_idx, shape=())  # scalar

            dY_lo_raw = ct.gather(dY, (row_idx, col_idx_lo), check_bounds=True, padding_value=0.0, latency=3)
            dY_hi_raw = ct.gather(dY, (row_idx, col_idx_hi), check_bounds=True, padding_value=0.0, latency=3)
            S_lo_raw = ct.gather(S, (row_idx, col_idx_lo), check_bounds=True, padding_value=0.0, latency=3)
            S_hi_raw = ct.gather(S, (row_idx, col_idx_hi), check_bounds=True, padding_value=0.0, latency=3)

            if casting_mode == _CASTING_MODE_LLAMA:
                m_lo = ct.astype(ct.mul(dY_lo_raw, ct.astype(W_shifted_lo, dY.dtype)), ct.float32)
                m_hi = ct.astype(ct.mul(dY_hi_raw, ct.astype(W_shifted_hi, dY.dtype)), ct.float32)
                S_lo = ct.astype(S_lo_raw, ct.float32)
                S_hi = ct.astype(S_hi_raw, ct.float32)
                sum_mS = ct.sum(ct.mul(m_lo, S_lo), 0, keepdims=False) + ct.sum(ct.mul(m_hi, S_hi), 0, keepdims=False)
                dW_acc_lo = ct.add(
                    dW_acc_lo, ct.astype(ct.mul(dY_lo_raw, ct.astype(ct.mul(S_lo, rstd), S.dtype)), ct.float32)
                )
                dW_acc_hi = ct.add(
                    dW_acc_hi, ct.astype(ct.mul(dY_hi_raw, ct.astype(ct.mul(S_hi, rstd), S.dtype)), ct.float32)
                )
                rstd3_coeff = rstd * rstd * rstd * inv_n_cols * sum_mS
                dS_out_lo = ct.astype(
                    ct.gather(dS_out, (row_idx, col_idx_lo), check_bounds=True, padding_value=0.0, latency=3),
                    ct.float32,
                )
                dX_lo = ct.add(ct.sub(ct.mul(rstd, m_lo), ct.mul(rstd3_coeff, S_lo)), dS_out_lo)
                ct.scatter(dX, (row_idx, col_idx_lo), ct.astype(dX_lo, dX.dtype), check_bounds=True)
                dS_out_hi = ct.astype(
                    ct.gather(dS_out, (row_idx, col_idx_hi), check_bounds=True, padding_value=0.0, latency=3),
                    ct.float32,
                )
                dX_hi = ct.add(ct.sub(ct.mul(rstd, m_hi), ct.mul(rstd3_coeff, S_hi)), dS_out_hi)
                ct.scatter(dX, (row_idx, col_idx_hi), ct.astype(dX_hi, dX.dtype), check_bounds=True)
            elif casting_mode == _CASTING_MODE_GEMMA:
                dY_lo = ct.astype(dY_lo_raw, ct.float32)
                dY_hi = ct.astype(dY_hi_raw, ct.float32)
                S_lo = ct.astype(S_lo_raw, ct.float32)
                S_hi = ct.astype(S_hi_raw, ct.float32)
                m_lo = ct.mul(dY_lo, W_shifted_lo)
                m_hi = ct.mul(dY_hi, W_shifted_hi)
                sum_mS = ct.sum(ct.mul(m_lo, S_lo), 0, keepdims=False) + ct.sum(ct.mul(m_hi, S_hi), 0, keepdims=False)
                dW_acc_lo = ct.add(dW_acc_lo, ct.mul(dY_lo, ct.mul(S_lo, rstd)))
                dW_acc_hi = ct.add(dW_acc_hi, ct.mul(dY_hi, ct.mul(S_hi, rstd)))
                rstd3_coeff = rstd * rstd * rstd * inv_n_cols * sum_mS
                dS_out_lo = ct.astype(
                    ct.gather(dS_out, (row_idx, col_idx_lo), check_bounds=True, padding_value=0.0, latency=3),
                    ct.float32,
                )
                dX_lo = ct.add(ct.sub(ct.mul(rstd, m_lo), ct.mul(rstd3_coeff, S_lo)), dS_out_lo)
                ct.scatter(dX, (row_idx, col_idx_lo), ct.astype(dX_lo, dX.dtype), check_bounds=True)
                dS_out_hi = ct.astype(
                    ct.gather(dS_out, (row_idx, col_idx_hi), check_bounds=True, padding_value=0.0, latency=3),
                    ct.float32,
                )
                dX_hi = ct.add(ct.sub(ct.mul(rstd, m_hi), ct.mul(rstd3_coeff, S_hi)), dS_out_hi)
                ct.scatter(dX, (row_idx, col_idx_hi), ct.astype(dX_hi, dX.dtype), check_bounds=True)
            else:
                # _CASTING_MODE_NONE: compute in S.dtype
                m_lo = ct.mul(dY_lo_raw, W_shifted_lo)
                m_hi = ct.mul(dY_hi_raw, W_shifted_hi)
                sum_mS = ct.sum(ct.astype(ct.mul(m_lo, S_lo_raw), ct.float32), 0, keepdims=False) + ct.sum(
                    ct.astype(ct.mul(m_hi, S_hi_raw), ct.float32), 0, keepdims=False
                )
                dW_acc_lo = ct.add(dW_acc_lo, ct.astype(ct.mul(dY_lo_raw, ct.mul(S_lo_raw, rstd)), ct.float32))
                dW_acc_hi = ct.add(dW_acc_hi, ct.astype(ct.mul(dY_hi_raw, ct.mul(S_hi_raw, rstd)), ct.float32))
                rstd3_coeff = rstd * rstd * rstd * inv_n_cols * sum_mS
                dS_out_lo = ct.gather(dS_out, (row_idx, col_idx_lo), check_bounds=True, padding_value=0.0, latency=3)
                dX_lo = ct.add(
                    ct.sub(
                        ct.mul(rstd, m_lo), ct.astype(ct.mul(rstd3_coeff, ct.astype(S_lo_raw, ct.float32)), S.dtype)
                    ),
                    dS_out_lo,
                )
                ct.scatter(dX, (row_idx, col_idx_lo), ct.astype(dX_lo, dX.dtype), check_bounds=True)
                dS_out_hi = ct.gather(dS_out, (row_idx, col_idx_hi), check_bounds=True, padding_value=0.0, latency=3)
                dX_hi = ct.add(
                    ct.sub(
                        ct.mul(rstd, m_hi), ct.astype(ct.mul(rstd3_coeff, ct.astype(S_hi_raw, ct.float32)), S.dtype)
                    ),
                    dS_out_hi,
                )
                ct.scatter(dX, (row_idx, col_idx_hi), ct.astype(dX_hi, dX.dtype), check_bounds=True)

    # Write accumulated dW to global memory — ONCE per SM, both halves.
    ct.scatter(dW_partial, (sm_id, col_idx_lo), dW_acc_lo, check_bounds=True)
    ct.scatter(dW_partial, (sm_id, col_idx_hi), dW_acc_hi, check_bounds=True)


_fused_add_rms_norm_bwd_persistent_ct_nww8 = _fused_add_rms_norm_bwd_persistent_ct.replace_hints(num_worker_warps=8)
_fused_add_rms_norm_bwd_persistent_2c_ct_nww8 = _fused_add_rms_norm_bwd_persistent_2c_ct.replace_hints(
    num_worker_warps=8
)


def _fused_add_rms_norm_forward_ct(X, R, W, eps, offset, casting_mode):
    if isinstance(casting_mode, str):
        casting_mode = _STR_TO_CASTING_MODE[casting_mode]

    shape = X.shape
    dim = shape[-1]
    X2d = X.view(-1, dim)
    R2d = R.view(-1, dim)
    n_rows, n_cols = X2d.shape
    BLOCK_SIZE = calculate_settings(n_cols)

    Y = torch.empty_like(X2d)
    S = torch.empty_like(X2d)
    # RSTD dtype: float32 for llama/gemma (fp32 rstd computation), X.dtype for none
    rstd_dtype = torch.float32 if casting_mode in (_CASTING_MODE_LLAMA, _CASTING_MODE_GEMMA) else X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

    # Fwd register pressure: for n_cols > 4096, 1 huge chunk spills; 2 chunks is the
    # sweet spot (3+ adds per-iteration overhead with no register benefit).
    FWD_BLOCK_SIZE = BLOCK_SIZE // 2 if BLOCK_SIZE > 4096 else BLOCK_SIZE
    base_kernel = _fused_add_rms_norm_fwd_small_ct if BLOCK_SIZE < 2048 else _fused_add_rms_norm_fwd_large_ct

    stream = torch.cuda.current_stream()
    args = (
        Y,
        S,
        X2d.contiguous(),
        R2d.contiguous(),
        W.contiguous(),
        RSTD,
        int(n_cols),
        float(eps),
        float(offset),
        int(FWD_BLOCK_SIZE),
        int(casting_mode),
    )
    cache_key = (n_cols, FWD_BLOCK_SIZE, casting_mode, X.dtype, str(X.device))
    tuned_kernel = _autotune_fwd_kernel(base_kernel, args, n_rows, cache_key, stream)

    ct.launch(stream, (n_rows, 1, 1), tuned_kernel, args)

    return Y.view(*shape), S.view(*shape), RSTD, BLOCK_SIZE, casting_mode


def _fused_add_rms_norm_backward_ct(dY, dS_out, S, W, RSTD, offset, casting_mode, BLOCK_SIZE):
    shape = dY.shape
    dim = shape[-1]
    dY2d = dY.view(-1, dim)
    dS_out2d = dS_out.view(-1, dim)
    S2d = S.view(-1, dim)
    n_rows, n_cols = dY2d.shape

    # Persistent kernel: one block per SM, each handles ceil(n_rows/sm_count) rows.
    # Grid = (sm_count,) -> exactly 1 block/SM -> full 256KB register file available.
    # When BLOCK_SIZE > _BWD_MAX_CHUNK_SIZE, use the 2-chunk kernel to reduce
    # register pressure (CHUNK_SIZE/128 elements/thread vs BLOCK_SIZE/128).
    sm_count = torch.cuda.get_device_properties(W.device).multi_processor_count
    num_iters = math.ceil(n_rows / sm_count)

    dX = torch.empty_like(dY2d)
    # Per-SM partial dW: shape (sm_count, n_cols).
    dW_partial = torch.empty(sm_count, n_cols, dtype=torch.float32, device=W.device)

    grid = (sm_count, 1, 1)
    # gather+latency=3 gives explicit cp.async-style pipelining that outperforms TMA
    # at every test shape on this kernel. 2-chunk variant caps per-thread register
    # usage when BLOCK_SIZE > _BWD_MAX_CHUNK_SIZE (splits cols into lo/hi halves).
    if BLOCK_SIZE > _BWD_MAX_CHUNK_SIZE:
        CHUNK_SIZE = BLOCK_SIZE // 2
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _fused_add_rms_norm_bwd_persistent_2c_ct_nww8,
            (
                dY2d.contiguous(),
                dS_out2d.contiguous(),
                dX,
                S2d.contiguous(),
                W.contiguous(),
                RSTD,
                dW_partial,
                int(n_cols),
                float(offset),
                int(num_iters),
                int(sm_count),
                int(n_rows),
                int(CHUNK_SIZE),
                int(casting_mode),
            ),
        )
    else:
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _fused_add_rms_norm_bwd_persistent_ct_nww8,
            (
                dY2d.contiguous(),
                dS_out2d.contiguous(),
                dX,
                S2d.contiguous(),
                W.contiguous(),
                RSTD,
                dW_partial,
                int(n_cols),
                float(offset),
                int(num_iters),
                int(sm_count),
                int(n_rows),
                int(BLOCK_SIZE),
                int(casting_mode),
            ),
        )

    dX = dX.view(*shape)
    dW = dW_partial.sum(dim=0).to(W.dtype)
    return dX, dX, dW  # dR == dX (gradient flows equally to X and R)


class LigerFusedAddRMSNormFunction(torch.autograd.Function):
    """CuTile autograd wrapper for fused residual-add + RMSNorm.

    Signature-compatible with the Triton LigerFusedAddRMSNormFunction. in_place is accepted
    for parity but ignored (this backend always writes fresh output buffers).
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, R, W, eps, offset=0.0, casting_mode="llama", in_place=False):
        Y, S, RSTD, BLOCK_SIZE, casting_mode_int = _fused_add_rms_norm_forward_ct(X, R, W, eps, offset, casting_mode)
        ctx.offset = offset
        ctx.casting_mode = casting_mode_int
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.save_for_backward(S, W, RSTD)
        return Y, S

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY, dS_out):
        S, W, RSTD = ctx.saved_tensors
        dX, dR, dW = _fused_add_rms_norm_backward_ct(
            dY, dS_out, S, W, RSTD, ctx.offset, ctx.casting_mode, ctx.BLOCK_SIZE
        )
        return dX, dR, dW, None, None, None, None
