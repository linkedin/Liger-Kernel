"""cuTile (NVIDIA cuda.tile) backend for ``rms_norm``.

This module implements RMSNorm with NVIDIA's cuTile DSL and registers it via
``@register_op``. It exposes three forward kernel variants, selected by the
``mode`` argument:

- ``"standard"``        — one row per program block; good for very wide rows.
- ``"static_persistent"`` — NUM_SMS persistent blocks each striding over rows;
  best when ``M`` is much larger than ``NUM_SMS``. The forward variant uses
  multi-row tiles (``TILE_SIZE_M × TILE_SIZE_N``) when ``M`` is large enough
  to keep all SMs busy with full tiles; this is the dominant perf win on
  Blackwell over the row-by-row legacy variant (``_fwd_persistent_*_singlerow``)
  which is kept as a correctness fallback when ``M`` is not a multiple of
  ``TILE_SIZE_M`` (padding-mode=ZERO handles that too, but we keep both for
  bench triangulation).
- ``"multi_wave_cached"`` — single-tile-per-row with the weight vector cached
  in registers across multiple rows; best for narrow rows.

Backward uses one persistent kernel per (casting_mode, has_W) pair. Partial
dW accumulators are reduced to the final ``dW`` on the host (matching the
Triton kernel's ``_dW.sum(dim=0).to(W.dtype)`` pattern).

Casting modes mirror the Triton version exactly:

- ``"llama"`` — only the RMS reduction runs in fp32; the weight multiply
  stays in the input dtype.
- ``"gemma"`` — every operation runs in fp32; the result is cast back at the
  end. ``W`` is also cast to fp32 inside the kernel.
- ``"none"``  — everything runs in the input dtype (faster but less precise).

References
----------
- cuTile RMSNorm reference: ``cutile-python/test/kernels/rms_norm.py`` —
  source of the ``@ct.kernel(occupancy=ct.ByTarget(sm_100=16))`` hint,
  ``ct.load(..., allow_tma=False, latency=N)`` recipes, and the static
  persistent loop structure.
- Triton reference: ``liger_kernel.ops.rms_norm`` — defines the computational
  semantics we must reproduce bit-for-bit (RSTD cache, casting modes, dX with
  the second-order correction, dW row-sum).
"""

# NOTE: do NOT use `from __future__ import annotations` here.
# cuTile's @ct.kernel introspects the wrapped function's __annotations__ to
# discover which parameters are Constants (i.e., compile-time specialized).
# PEP-563 future annotations stringifies all annotations, which makes cuTile
# silently miss the Constant flags, so a *single* compiled binary gets reused
# across all callers — yielding catastrophically wrong results when N or
# TILE_SIZE changes between launches. (Repro: shape0=(32,256) passes; shape1=
# (128,1024) after it returns max_diff~10 because the second launch reuses
# the binary specialized for the first launch.)

import math

from typing import Optional

import cuda.tile as ct
import numpy as np
import torch

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op

# Casting-mode constants + next_pow2 / num_sms helpers — single source of truth
# at liger_kernel.ops._nvidia_shared.
from liger_kernel.ops._nvidia_shared import CASTING_MODE_GEMMA as _CASTING_MODE_GEMMA
from liger_kernel.ops._nvidia_shared import CASTING_MODE_LLAMA as _CASTING_MODE_LLAMA
from liger_kernel.ops._nvidia_shared import CASTING_MODE_NONE as _CASTING_MODE_NONE
from liger_kernel.ops._nvidia_shared import STR_TO_CASTING_MODE as _str_to_casting_mode
from liger_kernel.ops._nvidia_shared import cutile_compiler_available as _cutile_compiler_available
from liger_kernel.ops._nvidia_shared import next_pow2 as _next_pow2
from liger_kernel.ops._nvidia_shared import num_sms as _num_sms
from liger_kernel.ops._nvidia_shared import to_local_if_dtensor as _to_local_if_dtensor

# Backward kernels above this tile width risk spilling on Blackwell (the live
# working set is ~22 B/element × 8 vectors ≈ 180 KB at 8192 lanes, which is
# right at the limit of the 256 KB register file). Keep N <= this; we
# additionally fail-fast in ``rms_norm_backward`` if the user feeds a wider
# row, mirroring the Triton kernel's BLOCK_SIZE assertion.
_BWD_MAX_TILE = 8192


def _select_mode(mode: Optional[str], n_rows: int, n_cols: int) -> str:
    """Pick a kernel variant when ``mode`` is ``None``.

    Heuristic (matches the cuTile RMSNorm reference + production TileGym):
      - If M > NUM_SMS * 2, use single-row ``static_persistent`` —
        each block strides over rows, amortising launch overhead and
        keeping W register-resident across many rows.
      - Else narrow rows (<= 4096) prefer the cached weight variant.
      - Else fall back to the standard one-row-per-program kernel.

    Note: an earlier in-progress multi-row kernel variant was removed because
    its tileiras compile time exceeded 12 minutes per specialisation on B200.
    The single-row persistent kernel matches NVIDIA TileGym's measured
    perf and compiles in <1s — kept as the only ``static_persistent``
    variant.

    Validates positive dimensions so a shape bug in the caller surfaces here
    as a clear ValueError rather than a cryptic CUDA grid error inside the
    kernel launch.
    """
    if mode is not None:
        # Explicit mode skips dim validation (caller's responsibility — same
        # contract as the legacy backend API).
        return mode

    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(f"rms_norm_cutile: invalid shape ({n_rows}, {n_cols}); both dims must be positive.")

    sms = _num_sms()
    if n_rows > sms * 2:
        return "static_persistent"
    if n_cols <= 4096:
        return "multi_wave_cached"
    return "standard"


# ===========================================================================
# Forward kernels — one row per program (``standard`` mode)
#
# These are the simplest variant: each block handles a single row, loading
# X[row, :] and W[:] once. They're correctness-anchor kernels and the right
# default when each row dominates a full block of work (wide rows, small M).
# ===========================================================================
@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _fwd_standard_llama(
    X,
    W,
    Y,
    RSTD,
    N: ct.Constant[int],
    eps: ct.Constant[float],
    offset: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    """Llama casting: fp32 reduction, weight multiply in input dtype."""
    row = ct.bid(0)
    x = ct.load(X, index=(row, 0), shape=(1, TILE_SIZE), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    w = ct.load(W, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)

    x_f32 = ct.astype(x, np.float32)
    mean_sq = ct.sum(ct.mul(x_f32, x_f32)) / N
    rstd = ct.rsqrt(mean_sq + eps)
    ct.store(RSTD, index=(row,), tile=ct.reshape(rstd, (1,)))

    x_norm = ct.astype(ct.mul(x_f32, rstd), x.dtype)
    w_shifted = ct.astype(ct.astype(w, np.float32) + ct.full((TILE_SIZE,), offset, np.float32), x.dtype)
    y = ct.mul(ct.reshape(x_norm, (TILE_SIZE,)), w_shifted)
    ct.store(Y, index=(row, 0), tile=ct.reshape(y, (1, TILE_SIZE)), allow_tma=False, latency=3)


@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _fwd_standard_gemma(
    X,
    W,
    Y,
    RSTD,
    N: ct.Constant[int],
    eps: ct.Constant[float],
    offset: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    """Gemma casting: everything fp32 inside, output cast back at the end."""
    row = ct.bid(0)
    x = ct.load(X, index=(row, 0), shape=(1, TILE_SIZE), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    w = ct.load(W, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    x_dtype = x.dtype

    x_f32 = ct.astype(x, np.float32)
    w_f32 = ct.astype(w, np.float32) + ct.full((TILE_SIZE,), offset, np.float32)

    mean_sq = ct.sum(ct.mul(x_f32, x_f32)) / N
    rstd = ct.rsqrt(mean_sq + eps)
    ct.store(RSTD, index=(row,), tile=ct.reshape(rstd, (1,)))

    y_f32 = ct.mul(ct.reshape(ct.mul(x_f32, rstd), (TILE_SIZE,)), w_f32)
    y = ct.astype(y_f32, x_dtype)
    ct.store(Y, index=(row, 0), tile=ct.reshape(y, (1, TILE_SIZE)), allow_tma=False, latency=3)


@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _fwd_standard_none(
    X,
    W,
    Y,
    RSTD,
    N: ct.Constant[int],
    eps: ct.Constant[float],
    offset: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    """No casting: stay in the input dtype throughout."""
    row = ct.bid(0)
    x = ct.load(X, index=(row, 0), shape=(1, TILE_SIZE), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    w = ct.load(W, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)

    mean_sq = ct.sum(ct.mul(x, x)) / N
    rstd = ct.rsqrt(mean_sq + eps)
    ct.store(RSTD, index=(row,), tile=ct.reshape(rstd, (1,)))

    x_norm = ct.mul(x, rstd)
    w_shifted = w + ct.full((TILE_SIZE,), offset, w.dtype)
    y = ct.mul(ct.reshape(x_norm, (TILE_SIZE,)), w_shifted)
    ct.store(Y, index=(row, 0), tile=ct.reshape(y, (1, TILE_SIZE)), allow_tma=False, latency=3)


# ===========================================================================
# Forward kernels — static persistent, single-row fallback
#
# These are the LEGACY persistent kernels: one row per loop iteration. They
# remain as a correctness fallback for cases where M does not divide the
# multi-row TILE_SIZE_M (e.g. M=37). The dispatcher routes here when
# ``_select_mode`` returns ``tile_m == 1``.
#
# The perf-tier multi-row kernels live just below this block.
# ===========================================================================
@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _fwd_persistent_llama_singlerow(
    X,
    W,
    Y,
    RSTD,
    n_rows: ct.Constant[int],
    N: ct.Constant[int],
    eps: ct.Constant[float],
    offset: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    """Llama casting, persistent (legacy single-row): one row at a time, striding by NUM_SMS.

    Kept as the correctness fallback when M is too small to amortise a multi-row tile,
    or when the dispatcher explicitly opts out of the multi-row kernel.
    """
    pid = ct.bid(0)
    num_blocks = ct.num_blocks(0)

    w = ct.load(W, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    w_shifted = ct.astype(ct.astype(w, np.float32) + ct.full((TILE_SIZE,), offset, np.float32), w.dtype)

    row_idx = pid
    while row_idx < n_rows:
        x = ct.load(
            X, index=(row_idx, 0), shape=(1, TILE_SIZE), allow_tma=False, latency=3, padding_mode=ct.PaddingMode.ZERO
        )
        x_f32 = ct.astype(x, np.float32)
        mean_sq = ct.sum(ct.mul(x_f32, x_f32)) / N
        rstd = ct.rsqrt(mean_sq + eps)
        ct.store(RSTD, index=(row_idx,), tile=ct.reshape(rstd, (1,)))
        x_norm = ct.astype(ct.mul(x_f32, rstd), x.dtype)
        y = ct.mul(ct.reshape(x_norm, (TILE_SIZE,)), w_shifted)
        ct.store(Y, index=(row_idx, 0), tile=ct.reshape(y, (1, TILE_SIZE)), allow_tma=False, latency=3)
        row_idx = row_idx + num_blocks


@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _fwd_persistent_gemma_singlerow(
    X,
    W,
    Y,
    RSTD,
    n_rows: ct.Constant[int],
    N: ct.Constant[int],
    eps: ct.Constant[float],
    offset: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    """Gemma casting, persistent (legacy single-row)."""
    pid = ct.bid(0)
    num_blocks = ct.num_blocks(0)

    w = ct.load(W, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    w_f32 = ct.astype(w, np.float32) + ct.full((TILE_SIZE,), offset, np.float32)

    row_idx = pid
    while row_idx < n_rows:
        x = ct.load(
            X, index=(row_idx, 0), shape=(1, TILE_SIZE), allow_tma=False, latency=3, padding_mode=ct.PaddingMode.ZERO
        )
        x_dtype = x.dtype
        x_f32 = ct.astype(x, np.float32)
        mean_sq = ct.sum(ct.mul(x_f32, x_f32)) / N
        rstd = ct.rsqrt(mean_sq + eps)
        ct.store(RSTD, index=(row_idx,), tile=ct.reshape(rstd, (1,)))
        y_f32 = ct.mul(ct.reshape(ct.mul(x_f32, rstd), (TILE_SIZE,)), w_f32)
        y = ct.astype(y_f32, x_dtype)
        ct.store(Y, index=(row_idx, 0), tile=ct.reshape(y, (1, TILE_SIZE)), allow_tma=False, latency=3)
        row_idx = row_idx + num_blocks


@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _fwd_persistent_none_singlerow(
    X,
    W,
    Y,
    RSTD,
    n_rows: ct.Constant[int],
    N: ct.Constant[int],
    eps: ct.Constant[float],
    offset: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    """No casting, persistent (legacy single-row)."""
    pid = ct.bid(0)
    num_blocks = ct.num_blocks(0)

    w = ct.load(W, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    w_shifted = w + ct.full((TILE_SIZE,), offset, w.dtype)

    row_idx = pid
    while row_idx < n_rows:
        x = ct.load(
            X, index=(row_idx, 0), shape=(1, TILE_SIZE), allow_tma=False, latency=3, padding_mode=ct.PaddingMode.ZERO
        )
        mean_sq = ct.sum(ct.mul(x, x)) / N
        rstd = ct.rsqrt(mean_sq + eps)
        ct.store(RSTD, index=(row_idx,), tile=ct.reshape(rstd, (1,)))
        x_norm = ct.mul(x, rstd)
        y = ct.mul(ct.reshape(x_norm, (TILE_SIZE,)), w_shifted)
        ct.store(Y, index=(row_idx, 0), tile=ct.reshape(y, (1, TILE_SIZE)), allow_tma=False, latency=3)
        row_idx = row_idx + num_blocks


# ---------------------------------------------------------------------------
# Multi-row static_persistent kernels were removed.
#
# They lived here previously and triggered a 12+ minute tileiras compile
# per specialization on B200 — a pathological compile-cost cliff caused by
# the register-pressure pattern of TILE_SIZE_M*TILE_SIZE_N tiles with
# inline persistent loops. The single-row static_persistent variants
# (defined above) match TileGym's reference perf and compile in <1s.
#
# If a future cuTile compiler resolves the cliff the multi-row kernels
# can be revived from git history of this file.
# ---------------------------------------------------------------------------


# ===========================================================================
# Forward kernels — multi-wave with cached weight (``multi_wave_cached`` mode)
#
# Same structure as standard, but with a higher latency hint on the X load so
# the scheduler issues more outstanding loads per wave. Useful for narrow
# rows where standard underutilises memory BW. We pull W via a single small
# load so the compiler can keep it in registers across the kernel.
# ===========================================================================
@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _fwd_cached_llama(
    X,
    W,
    Y,
    RSTD,
    N: ct.Constant[int],
    eps: ct.Constant[float],
    offset: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    """Llama casting, single row per program, latency-hinted X load."""
    row = ct.bid(0)
    w = ct.load(W, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    w_shifted = ct.astype(ct.astype(w, np.float32) + ct.full((TILE_SIZE,), offset, np.float32), w.dtype)

    x = ct.load(X, index=(row, 0), shape=(1, TILE_SIZE), allow_tma=False, latency=3, padding_mode=ct.PaddingMode.ZERO)
    x_f32 = ct.astype(x, np.float32)
    mean_sq = ct.sum(ct.mul(x_f32, x_f32)) / N
    rstd = ct.rsqrt(mean_sq + eps)
    ct.store(RSTD, index=(row,), tile=ct.reshape(rstd, (1,)))
    x_norm = ct.astype(ct.mul(x_f32, rstd), x.dtype)
    y = ct.mul(ct.reshape(x_norm, (TILE_SIZE,)), w_shifted)
    ct.store(Y, index=(row, 0), tile=ct.reshape(y, (1, TILE_SIZE)), allow_tma=False, latency=3)


@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _fwd_cached_gemma(
    X,
    W,
    Y,
    RSTD,
    N: ct.Constant[int],
    eps: ct.Constant[float],
    offset: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    """Gemma casting, single row per program, latency-hinted X load."""
    row = ct.bid(0)
    w = ct.load(W, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    w_f32 = ct.astype(w, np.float32) + ct.full((TILE_SIZE,), offset, np.float32)

    x = ct.load(X, index=(row, 0), shape=(1, TILE_SIZE), allow_tma=False, latency=3, padding_mode=ct.PaddingMode.ZERO)
    x_dtype = x.dtype
    x_f32 = ct.astype(x, np.float32)
    mean_sq = ct.sum(ct.mul(x_f32, x_f32)) / N
    rstd = ct.rsqrt(mean_sq + eps)
    ct.store(RSTD, index=(row,), tile=ct.reshape(rstd, (1,)))
    y_f32 = ct.mul(ct.reshape(ct.mul(x_f32, rstd), (TILE_SIZE,)), w_f32)
    y = ct.astype(y_f32, x_dtype)
    ct.store(Y, index=(row, 0), tile=ct.reshape(y, (1, TILE_SIZE)), allow_tma=False, latency=3)


@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _fwd_cached_none(
    X,
    W,
    Y,
    RSTD,
    N: ct.Constant[int],
    eps: ct.Constant[float],
    offset: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    """No casting, single row per program, latency-hinted X load."""
    row = ct.bid(0)
    w = ct.load(W, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    w_shifted = w + ct.full((TILE_SIZE,), offset, w.dtype)

    x = ct.load(X, index=(row, 0), shape=(1, TILE_SIZE), allow_tma=False, latency=3, padding_mode=ct.PaddingMode.ZERO)
    mean_sq = ct.sum(ct.mul(x, x)) / N
    rstd = ct.rsqrt(mean_sq + eps)
    ct.store(RSTD, index=(row,), tile=ct.reshape(rstd, (1,)))
    x_norm = ct.mul(x, rstd)
    y = ct.mul(ct.reshape(x_norm, (TILE_SIZE,)), w_shifted)
    ct.store(Y, index=(row, 0), tile=ct.reshape(y, (1, TILE_SIZE)), allow_tma=False, latency=3)


# Forward kernels without W (rare path; HF RMSNorm always has W, but we
# preserve interface parity with the Triton backend).
@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _fwd_no_w_llama(
    X,
    Y,
    RSTD,
    N: ct.Constant[int],
    eps: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(X, index=(row, 0), shape=(1, TILE_SIZE), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    x_f32 = ct.astype(x, np.float32)
    mean_sq = ct.sum(ct.mul(x_f32, x_f32)) / N
    rstd = ct.rsqrt(mean_sq + eps)
    ct.store(RSTD, index=(row,), tile=ct.reshape(rstd, (1,)))
    y = ct.astype(ct.mul(x_f32, rstd), x.dtype)
    ct.store(Y, index=(row, 0), tile=ct.reshape(y, (1, TILE_SIZE)), allow_tma=False, latency=3)


@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _fwd_no_w_gemma(
    X,
    Y,
    RSTD,
    N: ct.Constant[int],
    eps: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(X, index=(row, 0), shape=(1, TILE_SIZE), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    x_dtype = x.dtype
    x_f32 = ct.astype(x, np.float32)
    mean_sq = ct.sum(ct.mul(x_f32, x_f32)) / N
    rstd = ct.rsqrt(mean_sq + eps)
    ct.store(RSTD, index=(row,), tile=ct.reshape(rstd, (1,)))
    y = ct.astype(ct.mul(x_f32, rstd), x_dtype)
    ct.store(Y, index=(row, 0), tile=ct.reshape(y, (1, TILE_SIZE)), allow_tma=False, latency=3)


@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _fwd_no_w_none(
    X,
    Y,
    RSTD,
    N: ct.Constant[int],
    eps: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(X, index=(row, 0), shape=(1, TILE_SIZE), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    mean_sq = ct.sum(ct.mul(x, x)) / N
    rstd = ct.rsqrt(mean_sq + eps)
    ct.store(RSTD, index=(row,), tile=ct.reshape(rstd, (1,)))
    y = ct.mul(x, rstd)
    ct.store(Y, index=(row, 0), tile=ct.reshape(y, (1, TILE_SIZE)), allow_tma=False, latency=3)


# ===========================================================================
# Backward kernels — persistent, one block per SM, rows_per_program rows each
#
# Math (matches the Triton kernel):
#   m  = dY * (offset + W)
#         (llama: keep dY in input dtype, cast m to fp32 after weight mul)
#         (gemma: cast dY to fp32 first, m is fp32)
#         (none:  m stays in input dtype)
#   dX = rstd * m + rstd * (-(1/N) * rstd^2 * sum(m * X) * X)
#   dW_partial += dY * (X * rstd)
#         (llama: keep dY in input dtype; intermediate cast to input dtype)
#         (gemma/none: dY in fp32)
#
# dW partials are reduced cross-SM on the host: ``_dW.sum(0).to(W.dtype)``.
# ===========================================================================
@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _bwd_llama(
    dY,
    X,
    W,
    RSTD,
    dX,
    dW_partial,
    n_rows: ct.Constant[int],
    N: ct.Constant[int],
    offset: ct.Constant[float],
    rows_per_program: ct.Constant[int],
    TILE_SIZE: ct.Constant[int],
):
    """Backward for casting_mode='llama' (with W)."""
    pid = ct.bid(0)
    row_start = pid * rows_per_program
    row_end_val = (pid + 1) * rows_per_program

    w = ct.load(W, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    w_shifted_orig = ct.astype(ct.astype(w, np.float32) + ct.full((TILE_SIZE,), offset, np.float32), w.dtype)

    dw_accum = ct.full((TILE_SIZE,), 0.0, np.float32)

    row_idx = row_start
    while row_idx < row_end_val and row_idx < n_rows:
        dy_row = ct.reshape(
            ct.load(
                dY,
                index=(row_idx, 0),
                shape=(1, TILE_SIZE),
                allow_tma=False,
                latency=3,
                padding_mode=ct.PaddingMode.ZERO,
            ),
            (TILE_SIZE,),
        )
        x_row = ct.reshape(
            ct.load(
                X,
                index=(row_idx, 0),
                shape=(1, TILE_SIZE),
                allow_tma=False,
                latency=3,
                padding_mode=ct.PaddingMode.ZERO,
            ),
            (TILE_SIZE,),
        )
        rstd_val = ct.load(
            RSTD, index=(row_idx,), shape=(1,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO
        )

        x_f32 = ct.astype(x_row, np.float32)
        rstd_f32 = ct.astype(rstd_val, np.float32)

        # Llama: (dY * W_shifted) in input dtype, THEN cast to fp32.
        m = ct.astype(ct.mul(dy_row, w_shifted_orig), np.float32)

        # dX = rstd * m - (rstd^3 / N) * sum(m*x) * x
        inner = ct.sum(ct.mul(m, x_f32))
        correction = ct.mul(ct.mul(rstd_f32, ct.mul(rstd_f32, rstd_f32)), inner) / N
        dx_row = ct.mul(rstd_f32, m) - ct.mul(correction, x_f32)
        ct.store(
            dX,
            index=(row_idx, 0),
            tile=ct.reshape(ct.astype(dx_row, x_row.dtype), (1, TILE_SIZE)),
            allow_tma=False,
            latency=3,
        )

        # dW partial: dY * (x * rstd) — dY kept in input dtype per Triton llama.
        x_norm = ct.astype(ct.mul(x_f32, rstd_f32), x_row.dtype)
        dw_accum = dw_accum + ct.astype(ct.mul(dy_row, x_norm), np.float32)

        row_idx = row_idx + 1

    ct.store(dW_partial, index=(pid, 0), tile=ct.reshape(dw_accum, (1, TILE_SIZE)), allow_tma=False, latency=3)


@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _bwd_gemma(
    dY,
    X,
    W,
    RSTD,
    dX,
    dW_partial,
    n_rows: ct.Constant[int],
    N: ct.Constant[int],
    offset: ct.Constant[float],
    rows_per_program: ct.Constant[int],
    TILE_SIZE: ct.Constant[int],
):
    """Backward for casting_mode='gemma' (with W)."""
    pid = ct.bid(0)
    row_start = pid * rows_per_program
    row_end_val = (pid + 1) * rows_per_program

    w = ct.load(W, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    w_f32 = ct.astype(w, np.float32) + ct.full((TILE_SIZE,), offset, np.float32)

    dw_accum = ct.full((TILE_SIZE,), 0.0, np.float32)

    row_idx = row_start
    while row_idx < row_end_val and row_idx < n_rows:
        dy_row = ct.reshape(
            ct.load(
                dY,
                index=(row_idx, 0),
                shape=(1, TILE_SIZE),
                allow_tma=False,
                latency=3,
                padding_mode=ct.PaddingMode.ZERO,
            ),
            (TILE_SIZE,),
        )
        x_row = ct.reshape(
            ct.load(
                X,
                index=(row_idx, 0),
                shape=(1, TILE_SIZE),
                allow_tma=False,
                latency=3,
                padding_mode=ct.PaddingMode.ZERO,
            ),
            (TILE_SIZE,),
        )
        rstd_val = ct.load(
            RSTD, index=(row_idx,), shape=(1,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO
        )

        x_f32 = ct.astype(x_row, np.float32)
        dy_f32 = ct.astype(dy_row, np.float32)
        rstd_f32 = ct.astype(rstd_val, np.float32)

        m = ct.mul(dy_f32, w_f32)

        inner = ct.sum(ct.mul(m, x_f32))
        correction = ct.mul(ct.mul(rstd_f32, ct.mul(rstd_f32, rstd_f32)), inner) / N
        dx_row = ct.mul(rstd_f32, m) - ct.mul(correction, x_f32)
        ct.store(
            dX,
            index=(row_idx, 0),
            tile=ct.reshape(ct.astype(dx_row, x_row.dtype), (1, TILE_SIZE)),
            allow_tma=False,
            latency=3,
        )

        dw_accum = dw_accum + ct.mul(dy_f32, ct.mul(x_f32, rstd_f32))
        row_idx = row_idx + 1

    ct.store(dW_partial, index=(pid, 0), tile=ct.reshape(dw_accum, (1, TILE_SIZE)), allow_tma=False, latency=3)


@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _bwd_none(
    dY,
    X,
    W,
    RSTD,
    dX,
    dW_partial,
    n_rows: ct.Constant[int],
    N: ct.Constant[int],
    offset: ct.Constant[float],
    rows_per_program: ct.Constant[int],
    TILE_SIZE: ct.Constant[int],
):
    """Backward for casting_mode='none' (with W). All work stays in input dtype
    except the second-order correction term, which we keep in fp32 to avoid
    silent under/overflow at small rstd.
    """
    pid = ct.bid(0)
    row_start = pid * rows_per_program
    row_end_val = (pid + 1) * rows_per_program

    w = ct.load(W, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    w_shifted = w + ct.full((TILE_SIZE,), offset, w.dtype)

    dw_accum = ct.full((TILE_SIZE,), 0.0, np.float32)

    row_idx = row_start
    while row_idx < row_end_val and row_idx < n_rows:
        dy_row = ct.reshape(
            ct.load(
                dY,
                index=(row_idx, 0),
                shape=(1, TILE_SIZE),
                allow_tma=False,
                latency=3,
                padding_mode=ct.PaddingMode.ZERO,
            ),
            (TILE_SIZE,),
        )
        x_row = ct.reshape(
            ct.load(
                X,
                index=(row_idx, 0),
                shape=(1, TILE_SIZE),
                allow_tma=False,
                latency=3,
                padding_mode=ct.PaddingMode.ZERO,
            ),
            (TILE_SIZE,),
        )
        rstd_val = ct.load(
            RSTD, index=(row_idx,), shape=(1,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO
        )

        x_f32 = ct.astype(x_row, np.float32)
        rstd_f32 = ct.astype(rstd_val, np.float32)

        m = ct.mul(dy_row, w_shifted)
        m_f32 = ct.astype(m, np.float32)

        inner = ct.sum(ct.mul(m_f32, x_f32))
        correction = ct.mul(ct.mul(rstd_f32, ct.mul(rstd_f32, rstd_f32)), inner) / N
        dx_row_f32 = ct.astype(ct.mul(rstd_val, m), np.float32) - ct.mul(correction, x_f32)
        ct.store(
            dX,
            index=(row_idx, 0),
            tile=ct.reshape(ct.astype(dx_row_f32, x_row.dtype), (1, TILE_SIZE)),
            allow_tma=False,
            latency=3,
        )

        dw_accum = dw_accum + ct.mul(ct.astype(dy_row, np.float32), ct.mul(x_f32, rstd_f32))
        row_idx = row_idx + 1

    ct.store(dW_partial, index=(pid, 0), tile=ct.reshape(dw_accum, (1, TILE_SIZE)), allow_tma=False, latency=3)


# Backward kernels without W — dX only, no dW partial.
@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _bwd_no_w_llama(
    dY,
    X,
    RSTD,
    dX,
    n_rows: ct.Constant[int],
    N: ct.Constant[int],
    rows_per_program: ct.Constant[int],
    TILE_SIZE: ct.Constant[int],
):
    pid = ct.bid(0)
    row_start = pid * rows_per_program
    row_end_val = (pid + 1) * rows_per_program

    row_idx = row_start
    while row_idx < row_end_val and row_idx < n_rows:
        dy_row = ct.reshape(
            ct.load(
                dY,
                index=(row_idx, 0),
                shape=(1, TILE_SIZE),
                allow_tma=False,
                latency=3,
                padding_mode=ct.PaddingMode.ZERO,
            ),
            (TILE_SIZE,),
        )
        x_row = ct.reshape(
            ct.load(
                X,
                index=(row_idx, 0),
                shape=(1, TILE_SIZE),
                allow_tma=False,
                latency=3,
                padding_mode=ct.PaddingMode.ZERO,
            ),
            (TILE_SIZE,),
        )
        rstd_val = ct.load(
            RSTD, index=(row_idx,), shape=(1,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO
        )

        x_f32 = ct.astype(x_row, np.float32)
        m = ct.astype(dy_row, np.float32)
        rstd_f32 = ct.astype(rstd_val, np.float32)

        inner = ct.sum(ct.mul(m, x_f32))
        correction = ct.mul(ct.mul(rstd_f32, ct.mul(rstd_f32, rstd_f32)), inner) / N
        dx_row = ct.mul(rstd_f32, m) - ct.mul(correction, x_f32)
        ct.store(
            dX,
            index=(row_idx, 0),
            tile=ct.reshape(ct.astype(dx_row, x_row.dtype), (1, TILE_SIZE)),
            allow_tma=False,
            latency=3,
        )
        row_idx = row_idx + 1


@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _bwd_no_w_gemma(
    dY,
    X,
    RSTD,
    dX,
    n_rows: ct.Constant[int],
    N: ct.Constant[int],
    rows_per_program: ct.Constant[int],
    TILE_SIZE: ct.Constant[int],
):
    pid = ct.bid(0)
    row_start = pid * rows_per_program
    row_end_val = (pid + 1) * rows_per_program

    row_idx = row_start
    while row_idx < row_end_val and row_idx < n_rows:
        dy_row = ct.reshape(
            ct.load(
                dY,
                index=(row_idx, 0),
                shape=(1, TILE_SIZE),
                allow_tma=False,
                latency=3,
                padding_mode=ct.PaddingMode.ZERO,
            ),
            (TILE_SIZE,),
        )
        x_row = ct.reshape(
            ct.load(
                X,
                index=(row_idx, 0),
                shape=(1, TILE_SIZE),
                allow_tma=False,
                latency=3,
                padding_mode=ct.PaddingMode.ZERO,
            ),
            (TILE_SIZE,),
        )
        rstd_val = ct.load(
            RSTD, index=(row_idx,), shape=(1,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO
        )

        x_f32 = ct.astype(x_row, np.float32)
        dy_f32 = ct.astype(dy_row, np.float32)
        rstd_f32 = ct.astype(rstd_val, np.float32)

        inner = ct.sum(ct.mul(dy_f32, x_f32))
        correction = ct.mul(ct.mul(rstd_f32, ct.mul(rstd_f32, rstd_f32)), inner) / N
        dx_row = ct.mul(rstd_f32, dy_f32) - ct.mul(correction, x_f32)
        ct.store(
            dX,
            index=(row_idx, 0),
            tile=ct.reshape(ct.astype(dx_row, x_row.dtype), (1, TILE_SIZE)),
            allow_tma=False,
            latency=3,
        )
        row_idx = row_idx + 1


@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _bwd_no_w_none(
    dY,
    X,
    RSTD,
    dX,
    n_rows: ct.Constant[int],
    N: ct.Constant[int],
    rows_per_program: ct.Constant[int],
    TILE_SIZE: ct.Constant[int],
):
    pid = ct.bid(0)
    row_start = pid * rows_per_program
    row_end_val = (pid + 1) * rows_per_program

    row_idx = row_start
    while row_idx < row_end_val and row_idx < n_rows:
        dy_row = ct.reshape(
            ct.load(
                dY,
                index=(row_idx, 0),
                shape=(1, TILE_SIZE),
                allow_tma=False,
                latency=3,
                padding_mode=ct.PaddingMode.ZERO,
            ),
            (TILE_SIZE,),
        )
        x_row = ct.reshape(
            ct.load(
                X,
                index=(row_idx, 0),
                shape=(1, TILE_SIZE),
                allow_tma=False,
                latency=3,
                padding_mode=ct.PaddingMode.ZERO,
            ),
            (TILE_SIZE,),
        )
        rstd_val = ct.load(
            RSTD, index=(row_idx,), shape=(1,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO
        )

        x_f32 = ct.astype(x_row, np.float32)
        m_f32 = ct.astype(dy_row, np.float32)
        rstd_f32 = ct.astype(rstd_val, np.float32)

        inner = ct.sum(ct.mul(m_f32, x_f32))
        correction = ct.mul(ct.mul(rstd_f32, ct.mul(rstd_f32, rstd_f32)), inner) / N
        dx_row_f32 = ct.astype(ct.mul(rstd_val, dy_row), np.float32) - ct.mul(correction, x_f32)
        ct.store(
            dX,
            index=(row_idx, 0),
            tile=ct.reshape(ct.astype(dx_row_f32, x_row.dtype), (1, TILE_SIZE)),
            allow_tma=False,
            latency=3,
        )
        row_idx = row_idx + 1


# ---------------------------------------------------------------------------
# Forward kernel dispatch tables: (mode, casting_mode) -> kernel
# ---------------------------------------------------------------------------
_FWD = {
    ("standard", _CASTING_MODE_LLAMA): _fwd_standard_llama,
    ("standard", _CASTING_MODE_GEMMA): _fwd_standard_gemma,
    ("standard", _CASTING_MODE_NONE): _fwd_standard_none,
    # static_persistent: the multi-row variant was removed (pathological
    # tileiras compile time). The "_singlerow" suffix is preserved on the
    # function names below for git-history continuity but these are the
    # single (and only) static_persistent kernels.
    ("static_persistent", _CASTING_MODE_LLAMA): _fwd_persistent_llama_singlerow,
    ("static_persistent", _CASTING_MODE_GEMMA): _fwd_persistent_gemma_singlerow,
    ("static_persistent", _CASTING_MODE_NONE): _fwd_persistent_none_singlerow,
    ("multi_wave_cached", _CASTING_MODE_LLAMA): _fwd_cached_llama,
    ("multi_wave_cached", _CASTING_MODE_GEMMA): _fwd_cached_gemma,
    ("multi_wave_cached", _CASTING_MODE_NONE): _fwd_cached_none,
}

# Kept as a no-op alias to the same single-row functions for any older code
# path that consulted it; new code should index ``_FWD`` directly.
_FWD_PERSISTENT_SINGLEROW = {
    _CASTING_MODE_LLAMA: _fwd_persistent_llama_singlerow,
    _CASTING_MODE_GEMMA: _fwd_persistent_gemma_singlerow,
    _CASTING_MODE_NONE: _fwd_persistent_none_singlerow,
}

_FWD_NO_W = {
    _CASTING_MODE_LLAMA: _fwd_no_w_llama,
    _CASTING_MODE_GEMMA: _fwd_no_w_gemma,
    _CASTING_MODE_NONE: _fwd_no_w_none,
}

_BWD = {
    _CASTING_MODE_LLAMA: _bwd_llama,
    _CASTING_MODE_GEMMA: _bwd_gemma,
    _CASTING_MODE_NONE: _bwd_none,
}

_BWD_NO_W = {
    _CASTING_MODE_LLAMA: _bwd_no_w_llama,
    _CASTING_MODE_GEMMA: _bwd_no_w_gemma,
    _CASTING_MODE_NONE: _bwd_no_w_none,
}

_VALID_MODES = ("standard", "static_persistent", "multi_wave_cached")


# ---------------------------------------------------------------------------
# Host-side launchers
# ---------------------------------------------------------------------------
def _rms_norm_forward(X, W, eps, offset, casting_mode, mode):
    """Launch the forward kernel and return saved tensors for backward."""
    if not isinstance(casting_mode, int):
        if casting_mode not in _str_to_casting_mode:
            raise ValueError(f"Invalid casting mode: {casting_mode}")
        casting_mode = _str_to_casting_mode[casting_mode]

    shape = X.shape
    n_cols = shape[-1]
    X_flat = X.view(-1, n_cols)
    n_rows = X_flat.shape[0]

    TILE_SIZE = _next_pow2(n_cols)

    Y = torch.empty_like(X_flat)
    rstd_dtype = torch.float32 if casting_mode in (_CASTING_MODE_LLAMA, _CASTING_MODE_GEMMA) else X_flat.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X_flat.device)

    elementwise_affine = W is not None
    if elementwise_affine:
        assert X_flat.shape[1] == W.shape[0], f"Hidden size mismatch: X has {X_flat.shape[1]}, W has {W.shape[0]}"
        # cuTile's @ct.kernel can't implicitly promote a mixed-dtype multiply
        # (e.g., fp32 weight × bf16 activation) before the bf16 store. Triton
        # and CuTe DSL both auto-cast internally; we cast the weight to the
        # activation dtype on the host so the kernel sees same-dtype operands
        # — matches Liger's existing semantic for "llama"-mode casting where
        # the weight multiply stays in input dtype.
        if W.dtype != X_flat.dtype:
            W = W.to(X_flat.dtype)

    stream = torch.cuda.current_stream()

    if not elementwise_affine:
        # Without W there's only one kernel layout (standard one-row-per-program);
        # mode selection is a no-op in this branch.
        kernel = _FWD_NO_W[casting_mode]
        ct.launch(stream, (n_rows,), kernel, (X_flat, Y, RSTD, n_cols, eps, TILE_SIZE))
        return Y.view(*shape), X_flat, RSTD, TILE_SIZE, casting_mode, mode

    mode = _select_mode(mode, n_rows, n_cols)
    if mode not in _VALID_MODES:
        raise ValueError(f"cuTile rms_norm: unknown mode {mode!r}; expected one of {_VALID_MODES}")

    kernel = _FWD[(mode, casting_mode)]
    if mode == "static_persistent":
        # Persistent grid: NUM_SMS blocks each strides over rows.
        grid = (_num_sms(),)
        ct.launch(stream, grid, kernel, (X_flat, W, Y, RSTD, n_rows, n_cols, eps, offset, TILE_SIZE))
    else:
        # standard / multi_wave_cached: one block per row.
        grid = (n_rows,)
        ct.launch(stream, grid, kernel, (X_flat, W, Y, RSTD, n_cols, eps, offset, TILE_SIZE))

    return Y.view(*shape), X_flat, RSTD, TILE_SIZE, casting_mode, mode


def _rms_norm_backward(dY, X, W, RSTD, offset, casting_mode, TILE_SIZE, in_place):
    """Launch the backward kernel; return (dX, dW)."""
    shape = dY.shape
    n_cols = shape[-1]
    dY_flat = dY.view(-1, n_cols)
    n_rows = dY_flat.shape[0]

    if n_cols > _BWD_MAX_TILE:
        raise RuntimeError(
            f"cuTile rms_norm backward only supports hidden dim <= {_BWD_MAX_TILE}; "
            f"got {n_cols}. Use the Triton backend for wider rows."
        )

    sms = _num_sms()
    elementwise_affine = W is not None

    # in_place reuses dY's storage for dX. We honour the request best-effort:
    # the kernel reads dY and X before overwriting dY, which is safe so long
    # as no aliasing tile is read after being written (it isn't — each row is
    # processed atomically by one block).
    if in_place:
        dX_flat = dY_flat
    else:
        dX_flat = torch.empty_like(dY_flat)

    stream = torch.cuda.current_stream()
    rows_per_program = math.ceil(n_rows / sms)
    grid = (sms,)

    if elementwise_affine:
        # Partial dW: one fp32 row per SM, reduced on the host after launch.
        _dW = torch.empty((sms, n_cols), dtype=torch.float32, device=W.device)
        kernel = _BWD[casting_mode]
        ct.launch(
            stream,
            grid,
            kernel,
            (dY_flat, X, W, RSTD, dX_flat, _dW, n_rows, n_cols, offset, rows_per_program, TILE_SIZE),
        )
        dW = _dW.sum(dim=0).to(W.dtype)
    else:
        kernel = _BWD_NO_W[casting_mode]
        ct.launch(
            stream,
            grid,
            kernel,
            (dY_flat, X, RSTD, dX_flat, n_rows, n_cols, rows_per_program, TILE_SIZE),
        )
        dW = None

    return dX_flat.view(*shape), dW


# ---------------------------------------------------------------------------
# autograd.Function — variant-specific dispatch happens BEFORE we get here,
# so the Function stays plain (no mode kwarg, .apply() compatible).
# ---------------------------------------------------------------------------
class _LigerRMSNormCuTileFunction(torch.autograd.Function):
    """cuTile RMSNorm. ``mode`` is passed via a sticky attribute on the
    function class rather than a kwarg because ``Function.apply`` rejects
    unknown kwargs and we want to keep this class itself dispatcher-agnostic.
    """

    @staticmethod
    def forward(ctx, X, W, eps, offset, casting_mode, in_place, row_mode, mode):
        # row_mode is the legacy Triton tuning knob; cuTile ignores it.
        del row_mode

        X = _to_local_if_dtensor(X)

        X = X.contiguous()
        if W is not None:
            W = W.contiguous()

        Y, X_flat, RSTD, TILE_SIZE, casting_mode_int, _selected_mode = _rms_norm_forward(
            X, W, eps, offset, casting_mode, mode
        )
        ctx.offset = offset
        ctx.casting_mode = casting_mode_int
        ctx.in_place = in_place
        ctx.TILE_SIZE = TILE_SIZE
        ctx.elementwise_affine = W is not None
        if W is not None:
            ctx.save_for_backward(X_flat, W, RSTD)
        else:
            ctx.save_for_backward(X_flat, RSTD)
        return Y

    @staticmethod
    def backward(ctx, dY):
        dY = _to_local_if_dtensor(dY).contiguous()

        if ctx.elementwise_affine:
            X, W, RSTD = ctx.saved_tensors
        else:
            X, RSTD = ctx.saved_tensors
            W = None

        dX, dW = _rms_norm_backward(dY, X, W, RSTD, ctx.offset, ctx.casting_mode, ctx.TILE_SIZE, ctx.in_place)
        # Match forward arity: (X, W, eps, offset, casting_mode, in_place, row_mode, mode)
        return dX, dW, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Public registration
# ---------------------------------------------------------------------------
@register_op(
    "rms_norm",
    impl_name="nvidia-cutile",
    capability=Capability(
        min_cc=(10, 0),
        modules=["cuda.tile", "torch"],
        predicate=_cutile_compiler_available,
    ),
    modes=("standard", "static_persistent", "multi_wave_cached"),
    default_mode="static_persistent",
    # Measured on B200 sm_100 (torch 2.12, tileiras v13.2): cuTile RMSNorm
    # loses to both Triton and CuTeDSL on every shape we sweep (e.g., 32K×4096
    # bwd: cuTile 0.77ms vs Triton 0.38ms vs CuTeDSL 0.17ms). Keep cuTile as
    # the *last* fallback so users only get it when explicitly requested or
    # when no other impl is usable. The kernel ships because it's still useful
    # for hidden_dim > 32K where CuTeDSL is range-capped.
    preference_rank=80,
    tolerances={
        torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
        torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
        torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
    },
    notes="cuTile static-persistent + multi-wave RMSNorm for Blackwell.",
)
def rms_norm_cutile(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    offset: float = 0.0,
    casting_mode: str = "llama",
    in_place: bool = False,
    row_mode: Optional[bool] = None,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """Dispatch entry-point for the cuTile RMSNorm backend.

    The wrapper resolves ``mode`` (auto if ``None``), then forwards everything
    to :class:`_LigerRMSNormCuTileFunction`. It exists because the dispatcher
    passes ``mode`` as a kwarg, and ``torch.autograd.Function.apply`` does not
    accept unknown kwargs.
    """
    if mode is not None and mode not in _VALID_MODES:
        raise ValueError(f"cuTile rms_norm: unknown mode {mode!r}; valid modes are {_VALID_MODES}.")
    return _LigerRMSNormCuTileFunction.apply(x, weight, eps, offset, casting_mode, in_place, row_mode, mode)
