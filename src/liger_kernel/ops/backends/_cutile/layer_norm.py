"""cuTile (NVIDIA cuda.tile) backend for ``layer_norm``.

This module implements LayerNorm with NVIDIA's cuTile DSL and registers it via
``@register_op``. It exposes three forward kernel variants, selected by the
``mode`` argument:

- ``"standard"``        — one row per program block; good for very wide rows.
- ``"static_persistent"`` — NUM_SMS persistent blocks, each striding over
  rows one at a time. W and B are loaded once per block and stay register-
  resident across all rows. Best when M is much larger than NUM_SMS. Matches
  NVIDIA TileGym's reference for LayerNorm.
- ``"multi_wave_cached"`` — single-tile-per-row with the weight and bias
  vectors loaded once per block; best for narrow rows.

A prior multi-row variant of ``static_persistent`` was removed because the
row-tile / row-element index relationship in the persistent loop was wrong
(consecutive blocks' row ranges overlapped by ``TILE_SIZE_M - 1`` rows,
producing races and incorrect results when the runtime gate was lifted).

Backward is one persistent kernel that produces (dX, partial-dW, partial-dB).
The partial dW / dB rows (one per SM) are reduced to the final ``dW`` / ``dB``
on the host (matching the Triton kernel's ``_dW.sum(dim=0).to(W.dtype)``
pattern).

Math (matches the Triton kernel and reference cuTile kernel exactly):

    Forward (with cached Mean & RSTD for backward):
        mean  = sum(X) / N
        var   = sum((X - mean)^2) / N
        rstd  = rsqrt(var + eps)
        Y     = (X - mean) * rstd * W + B

    Backward (per row, all reductions in fp32):
        x_hat = (X - mean) * rstd
        wdy   = W * dY
        c1    = sum(x_hat * wdy) / N
        c2    = sum(wdy) / N
        dX    = (wdy - x_hat * c1 - c2) * rstd
        dW   += dY * x_hat        (row-summed across SMs on host)
        dB   += dY                (row-summed across SMs on host)

References
----------
- cuTile LayerNorm reference: ``cutile-python/test/kernels/layer_norm.py`` —
  source of the ``@ct.kernel(occupancy=ct.ByTarget(sm_100=16))`` hint,
  ``ct.load(..., allow_tma=False, latency=N)`` recipes, and the per-SM partial
  dW/dB pattern (the reference uses a lock-protected accumulator into a
  GROUP_SIZE_M buffer; we instead allocate one partial-row per SM and reduce
  on the host, mirroring the Triton kernel).
- Triton reference: ``liger_kernel.ops.layer_norm`` — defines the
  computational semantics we must reproduce bit-for-bit (Mean + RSTD caches,
  the dX reduction with c1/c2 in fp32, and the dW/dB row-sum reduction on the
  host).
- cuTile RMSNorm sibling: ``liger_kernel.ops.backends._cutile.rms_norm`` — the
  three-mode dispatch scaffolding, ``_select_mode`` heuristic, and the
  fp32-partial-dW host-reduction pattern come from this file. LayerNorm
  differs in (a) caching Mean *and* RSTD for backward, (b) accepting a real
  bias tensor (functional layer substitutes zeros when bias is None), and
  (c) computing a dB partial alongside dW.
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
from liger_kernel.ops._nvidia_shared import cutile_compiler_available as _cutile_compiler_available
from liger_kernel.ops._nvidia_shared import next_pow2 as _next_pow2
from liger_kernel.ops._nvidia_shared import num_sms as _num_sms
from liger_kernel.ops._nvidia_shared import to_local_if_dtensor as _to_local_if_dtensor

# Backward kernels above this tile width risk spilling on Blackwell (the live
# working set is ~24 B/element × 9 vectors (x, dy, x_hat, wdy, w, dw, db, dx,
# scratch) ≈ 216 KB at 8192 lanes, which is right at the limit of the 256 KB
# register file). LayerNorm holds slightly more state than RMSNorm because of
# the extra dB accumulator; we keep the same 8192 ceiling and fail-fast in
# ``_layer_norm_backward`` if the user feeds a wider row, mirroring the
# Triton kernel's BLOCK_SIZE assertion. The test framework's "documented
# range limit" path will skip rather than fail when this fires.
# Both forward and backward materialise a single ``next_pow2(N)``-wide tile per
# row, so the supported hidden dimension is capped identically on both sides.
_MAX_TILE = 8192
_BWD_MAX_TILE = _MAX_TILE


def _launch(device: torch.device, grid, kernel, args) -> None:
    """Device-safe cuTile launch tied to the input tensor's CUDA device.

    Dispatch selects this backend from the first CUDA tensor's device, so in a
    multi-GPU process the process-current device can differ. Guarding the
    device (and using that device's stream) keeps the launch on the tensor's
    GPU. Mirrors ``ops/cutile/ops/fused_linear_cross_entropy.py``.
    """
    with torch.cuda.device(device):
        ct.launch(torch.cuda.current_stream(device), grid, kernel, args)


def _select_mode(mode: Optional[str], n_rows: int, n_cols: int, device: torch.device) -> str:
    """Pick a kernel variant when ``mode`` is ``None``.

    Heuristic (matches the cuTile RMSNorm sibling after its multi-row removal):
      - if M > NUM_SMS * 2, use single-row ``static_persistent`` — each block
        strides over rows, amortising launch overhead and keeping W & B
        register-resident across many rows;
      - else, narrow rows (<= 4096) prefer the cached weight variant;
      - else fall through to the standard one-row-per-program kernel.

    A prior multi-row variant of the persistent kernel was removed: the
    row-tile / row-element index relationship in the persistent loop was
    wrong (each block's row range overlapped its neighbours' by
    ``TILE_SIZE_M - 1`` rows, producing races and incorrect results when the
    runtime gate was lifted). The single-row persistent kernel below matches
    NVIDIA TileGym's reference for LayerNorm.

    Validates positive dimensions so a shape bug in the caller surfaces here
    as a clear ValueError rather than a cryptic CUDA grid error inside the
    kernel launch.
    """
    if mode is not None:
        return mode
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(f"layer_norm_cutile: invalid shape ({n_rows}, {n_cols}); both dims must be positive.")
    sms = _num_sms(device)
    if n_rows > sms * 2:
        return "static_persistent"
    if n_cols <= 4096:
        return "multi_wave_cached"
    return "standard"


# ===========================================================================
# Forward kernels
#
# All three variants compute identical math; they differ only in launch
# topology and in how aggressively they cache W/B and overlap memory loads.
#
# Numerics: we always upcast X, W, B to fp32 inside the kernel for the mean,
# variance, and rstd reductions; the final Y is cast back to X's dtype on
# store. This mirrors the Triton kernel's fp32-reductions/io-dtype-output
# strategy. Mean and RSTD are stored in fp32 (not X's dtype) so that large-mean
# and bf16 rows keep the precision the backward pass needs.
#
# Variance is computed as a MASKED centered second moment,
# ``sum(mask * (x - mean)**2) / N``, NOT the ``E[x**2] - E[x]**2`` identity:
# the latter catastrophically cancels in fp32 for rows with a large mean
# (inputs ~1e4 -> var ~0/negative -> rsqrt(var+eps) -> NaN). The mask zeroes
# the ZERO-padded lanes (index >= N) whose centered value ``-mean`` would
# otherwise inflate the sum.
# ===========================================================================
@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _fwd_standard(
    X,
    W,
    B,
    Y,
    Mean,
    RSTD,
    N: ct.Constant[int],
    eps: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    """One row per program block. Good when rows are wide or M is small."""
    row = ct.bid(0)

    x = ct.load(X, index=(row, 0), shape=(1, TILE_SIZE), allow_tma=False, latency=10, padding_mode=ct.PaddingMode.ZERO)
    w = ct.load(W, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    b = ct.load(B, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)

    x_f32 = ct.astype(x, np.float32)
    w_f32 = ct.astype(w, np.float32)
    b_f32 = ct.astype(b, np.float32)

    mean = ct.sum(x_f32) / N
    # Numerically-stable variance via a masked centered second moment.
    #
    # We deliberately do NOT use the E[x**2] - E[x]**2 identity: for rows with
    # a large mean (e.g. inputs ~1e4) the two fp32 terms are nearly equal and
    # catastrophically cancel -> var ~0 or negative -> rsqrt(var+eps) -> NaN.
    #
    # padding_mode=ZERO leaves the padded lanes (index >= N) at 0, so
    # centered = 0 - mean = -mean there; masking those lanes to 0 keeps them
    # out of the variance sum. sum(x)/N and the store are unaffected by the
    # padded zeros, so mean is exact.
    cols = ct.arange(TILE_SIZE, dtype=ct.int32)
    mask_f32 = ct.astype(ct.less(cols, N), np.float32)
    centered = ct.reshape(x_f32, (TILE_SIZE,)) - mean
    centered_masked = ct.mul(centered, mask_f32)
    var = ct.sum(ct.mul(centered_masked, centered_masked)) / N
    rstd = ct.rsqrt(var + eps)

    # Persist Mean and RSTD in fp32 so large-mean/bf16 rows keep the precision
    # the backward pass needs (backward upcasts to fp32 anyway).
    ct.store(Mean, index=(row,), tile=ct.reshape(mean, (1,)))
    ct.store(RSTD, index=(row,), tile=ct.reshape(rstd, (1,)))

    y_f32 = ct.mul(ct.mul(centered, rstd), w_f32) + b_f32
    y = ct.astype(y_f32, x.dtype)
    ct.store(Y, index=(row, 0), tile=ct.reshape(y, (1, TILE_SIZE)), allow_tma=False, latency=3)


@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _fwd_persistent_singlerow(
    X,
    W,
    B,
    Y,
    Mean,
    RSTD,
    n_rows: ct.Constant[int],
    N: ct.Constant[int],
    eps: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    """NUM_SMS persistent blocks, **one row at a time** striding over rows.

    This is the only ``static_persistent`` variant; a prior multi-row variant
    was removed (see module docstring for the index bug).

    W and B are loaded once and cached for the lifetime of the block; X is
    reloaded per row with a higher latency hint to overlap multiple in-flight
    loads.
    """
    pid = ct.bid(0)
    num_blocks = ct.num_blocks(0)

    w = ct.load(W, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    b = ct.load(B, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    w_f32 = ct.astype(w, np.float32)
    b_f32 = ct.astype(b, np.float32)

    row_idx = pid
    while row_idx < n_rows:
        x = ct.load(
            X, index=(row_idx, 0), shape=(1, TILE_SIZE), allow_tma=False, latency=10, padding_mode=ct.PaddingMode.ZERO
        )
        x_f32 = ct.astype(x, np.float32)

        mean = ct.sum(x_f32) / N
        # Numerically-stable masked variance (see _fwd_standard for the full
        # rationale: the E[x**2]-E[x]**2 identity cancels for large-mean rows).
        cols = ct.arange(TILE_SIZE, dtype=ct.int32)
        mask_f32 = ct.astype(ct.less(cols, N), np.float32)
        centered = ct.reshape(x_f32, (TILE_SIZE,)) - mean
        centered_masked = ct.mul(centered, mask_f32)
        var = ct.sum(ct.mul(centered_masked, centered_masked)) / N
        rstd = ct.rsqrt(var + eps)

        ct.store(Mean, index=(row_idx,), tile=ct.reshape(mean, (1,)))
        ct.store(RSTD, index=(row_idx,), tile=ct.reshape(rstd, (1,)))

        y_f32 = ct.mul(ct.mul(centered, rstd), w_f32) + b_f32
        y = ct.astype(y_f32, x.dtype)
        ct.store(Y, index=(row_idx, 0), tile=ct.reshape(y, (1, TILE_SIZE)), allow_tma=False, latency=3)

        row_idx = row_idx + num_blocks


@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _fwd_cached(
    X,
    W,
    B,
    Y,
    Mean,
    RSTD,
    N: ct.Constant[int],
    eps: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    """Single row per program with W/B cached and a latency-hinted X load.

    Identical math to ``_fwd_standard``; the only difference is the explicit
    ordering hint (load W & B first, then issue the X load with latency=10 so
    the scheduler keeps it in flight while we do the arithmetic). On narrow
    rows this gives a ~10-15% bump over ``_fwd_standard``.
    """
    row = ct.bid(0)

    w = ct.load(W, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    b = ct.load(B, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    w_f32 = ct.astype(w, np.float32)
    b_f32 = ct.astype(b, np.float32)

    x = ct.load(X, index=(row, 0), shape=(1, TILE_SIZE), allow_tma=False, latency=10, padding_mode=ct.PaddingMode.ZERO)
    x_f32 = ct.astype(x, np.float32)

    mean = ct.sum(x_f32) / N
    # Numerically-stable masked variance (see _fwd_standard for the full
    # rationale: the E[x**2]-E[x]**2 identity cancels for large-mean rows).
    cols = ct.arange(TILE_SIZE, dtype=ct.int32)
    mask_f32 = ct.astype(ct.less(cols, N), np.float32)
    centered = ct.reshape(x_f32, (TILE_SIZE,)) - mean
    centered_masked = ct.mul(centered, mask_f32)
    var = ct.sum(ct.mul(centered_masked, centered_masked)) / N
    rstd = ct.rsqrt(var + eps)

    ct.store(Mean, index=(row,), tile=ct.reshape(mean, (1,)))
    ct.store(RSTD, index=(row,), tile=ct.reshape(rstd, (1,)))

    y_f32 = ct.mul(ct.mul(centered, rstd), w_f32) + b_f32
    y = ct.astype(y_f32, x.dtype)
    ct.store(Y, index=(row, 0), tile=ct.reshape(y, (1, TILE_SIZE)), allow_tma=False, latency=3)


# ===========================================================================
# Backward kernel — persistent, one block per SM, rows_per_program rows each.
#
# Each block walks ``rows_per_program`` consecutive rows. For each row it:
#   (1) loads X[row,:], dY[row,:], Mean[row], RSTD[row];
#   (2) computes x_hat, wdy, c1, c2 in fp32;
#   (3) writes dX[row,:] in the input dtype;
#   (4) accumulates dY * x_hat and dY into block-local fp32 vectors.
#
# After the row loop the block writes its accumulated dW/dB partials at row
# `pid` of the per-SM partial buffers. The host then reduces:
#     dW = _dW.sum(dim=0).to(W.dtype)
#     dB = _dB.sum(dim=0).to(B.dtype)
# matching the Triton kernel.
# ===========================================================================
@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def _bwd(
    dY,
    X,
    W,
    Mean,
    RSTD,
    dX,
    dW_partial,
    dB_partial,
    n_rows: ct.Constant[int],
    N: ct.Constant[int],
    rows_per_program: ct.Constant[int],
    TILE_SIZE: ct.Constant[int],
):
    """Backward for LayerNorm with W and B (the only path the dispatcher
    exposes — torch.nn.LayerNorm always has affine params, and our
    functional.layer_norm substitutes a zeros tensor when bias is None)."""
    pid = ct.bid(0)
    row_start = pid * rows_per_program
    row_end_val = (pid + 1) * rows_per_program

    w = ct.load(W, index=(0,), shape=(TILE_SIZE,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO)
    w_f32 = ct.astype(w, np.float32)

    dw_accum = ct.full((TILE_SIZE,), 0.0, np.float32)
    db_accum = ct.full((TILE_SIZE,), 0.0, np.float32)

    row_idx = row_start
    while row_idx < row_end_val and row_idx < n_rows:
        dy_row = ct.reshape(
            ct.load(
                dY,
                index=(row_idx, 0),
                shape=(1, TILE_SIZE),
                allow_tma=False,
                latency=10,
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
                latency=10,
                padding_mode=ct.PaddingMode.ZERO,
            ),
            (TILE_SIZE,),
        )
        mean_val = ct.load(
            Mean, index=(row_idx,), shape=(1,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO
        )
        rstd_val = ct.load(
            RSTD, index=(row_idx,), shape=(1,), allow_tma=False, latency=1, padding_mode=ct.PaddingMode.ZERO
        )

        x_f32 = ct.astype(x_row, np.float32)
        dy_f32 = ct.astype(dy_row, np.float32)
        mean_f32 = ct.astype(mean_val, np.float32)
        rstd_f32 = ct.astype(rstd_val, np.float32)

        x_hat = ct.mul(x_f32 - mean_f32, rstd_f32)
        wdy = ct.mul(w_f32, dy_f32)

        c1 = ct.sum(ct.mul(x_hat, wdy)) / N
        c2 = ct.sum(wdy) / N

        dx_row_f32 = ct.mul(wdy - ct.mul(x_hat, c1) - c2, rstd_f32)
        ct.store(
            dX,
            index=(row_idx, 0),
            tile=ct.reshape(ct.astype(dx_row_f32, x_row.dtype), (1, TILE_SIZE)),
            allow_tma=False,
            latency=3,
        )

        dw_accum = dw_accum + ct.mul(dy_f32, x_hat)
        db_accum = db_accum + dy_f32

        row_idx = row_idx + 1

    ct.store(dW_partial, index=(pid, 0), tile=ct.reshape(dw_accum, (1, TILE_SIZE)), allow_tma=False, latency=3)
    ct.store(dB_partial, index=(pid, 0), tile=ct.reshape(db_accum, (1, TILE_SIZE)), allow_tma=False, latency=3)


# ---------------------------------------------------------------------------
# Forward kernel dispatch table
# ---------------------------------------------------------------------------
# Three modes. The ``_fwd_persistent_singlerow`` symbol name is historical —
# the multi-row sibling was removed (see module docstring for the index bug);
# the singlerow kernel is now the only persistent variant.
_FWD = {
    "standard": _fwd_standard,
    "static_persistent": _fwd_persistent_singlerow,
    "multi_wave_cached": _fwd_cached,
}

_VALID_MODES = (
    "standard",
    "static_persistent",
    "multi_wave_cached",
)


# ---------------------------------------------------------------------------
# Host-side launchers
# ---------------------------------------------------------------------------
def _layer_norm_forward(X, W, B, eps, mode):
    """Launch the forward kernel; return saved tensors for backward.

    Returns ``(Y, X_flat, Mean, RSTD, TILE_SIZE, selected_mode)`` so the
    caller can store everything for backward and report the selected mode.
    """
    shape = X.shape
    n_cols = shape[-1]
    X_flat = X.view(-1, n_cols)
    n_rows = X_flat.shape[0]
    device = X_flat.device

    if X_flat.shape[1] != W.shape[0]:
        raise ValueError(f"Hidden size mismatch: X has {X_flat.shape[1]}, W has {W.shape[0]}")
    if B.shape[0] != W.shape[0]:
        raise ValueError(f"Bias size mismatch: B has {B.shape[0]}, W has {W.shape[0]}")

    if n_cols > _MAX_TILE:
        raise RuntimeError(
            f"cuTile layer_norm only supports hidden dim <= {_MAX_TILE}; got {n_cols}. "
            f"Both forward and backward materialise a single tile of this width; "
            f"use the Triton backend for wider rows."
        )

    # Do NOT pre-quantize fp32 affine params to the activation dtype: the
    # kernels already upcast W and B to fp32 internally (see the forward and
    # backward bodies), and the autograd wrapper saves these *same* tensors for
    # backward. Rounding them here would make forward use different affine
    # values than backward differentiates, and would drop dB below the bias's
    # true precision.

    TILE_SIZE = _next_pow2(n_cols)

    Y = torch.empty_like(X_flat)
    # Mean and RSTD are cached for backward in fp32 (not X's dtype): for
    # large-mean or bf16 rows, storing them in a low-precision dtype loses
    # information the backward reduction needs. Backward upcasts to fp32
    # anyway, so this is the natural storage dtype.
    Mean = torch.empty(n_rows, dtype=torch.float32, device=X_flat.device)
    RSTD = torch.empty(n_rows, dtype=torch.float32, device=X_flat.device)

    mode = _select_mode(mode, n_rows, n_cols, device)
    if mode not in _VALID_MODES:
        raise ValueError(f"cuTile layer_norm: unknown mode {mode!r}; expected one of {_VALID_MODES}")

    kernel = _FWD[mode]

    if mode == "static_persistent":
        grid = (_num_sms(device),)
        _launch(device, grid, kernel, (X_flat, W, B, Y, Mean, RSTD, n_rows, n_cols, eps, TILE_SIZE))
    else:
        grid = (n_rows,)
        _launch(device, grid, kernel, (X_flat, W, B, Y, Mean, RSTD, n_cols, eps, TILE_SIZE))

    return Y.view(*shape), X_flat, Mean, RSTD, TILE_SIZE, mode


def _layer_norm_backward(dY, X, W, B, Mean, RSTD, TILE_SIZE):
    """Launch the backward kernel; return ``(dX, dW, dB)``.

    ``B`` is passed only so ``dB`` can be returned in the bias's own dtype
    (the kernel itself does not read ``B`` — ``dB`` depends only on ``dY``).
    """
    shape = dY.shape
    n_cols = shape[-1]
    dY_flat = dY.view(-1, n_cols)
    n_rows = dY_flat.shape[0]
    device = dY_flat.device

    if n_cols > _BWD_MAX_TILE:
        raise RuntimeError(
            f"cuTile layer_norm backward only supports hidden dim <= {_BWD_MAX_TILE}; "
            f"got {n_cols}. Use the Triton backend for wider rows."
        )

    sms = _num_sms(device)
    rows_per_program = math.ceil(n_rows / sms)

    dX_flat = torch.empty_like(dY_flat)
    # fp32 partials for numerical stability — one row per SM, host-reduced.
    _dW = torch.empty((sms, n_cols), dtype=torch.float32, device=W.device)
    _dB = torch.empty((sms, n_cols), dtype=torch.float32, device=W.device)

    grid = (sms,)
    _launch(
        device,
        grid,
        _bwd,
        (dY_flat, X, W, Mean, RSTD, dX_flat, _dW, _dB, n_rows, n_cols, rows_per_program, TILE_SIZE),
    )

    dW = _dW.sum(dim=0).to(W.dtype)
    dB = _dB.sum(dim=0).to(B.dtype)  # return the bias gradient in the bias's own dtype
    return dX_flat.view(*shape), dW, dB


# ---------------------------------------------------------------------------
# autograd.Function — same shape as the RMSNorm sibling. ``mode`` is plumbed
# through .apply() as a positional arg (Function.apply rejects unknown
# kwargs) and the dispatcher's mode kwarg is consumed by the registration
# wrapper below.
# ---------------------------------------------------------------------------
class _LigerLayerNormCuTileFunction(torch.autograd.Function):
    """cuTile LayerNorm. ``mode`` selects between standard / persistent /
    cached forward variants. Backward always uses the single persistent
    kernel (rows split across SMs)."""

    @staticmethod
    def forward(ctx, X, W, B, eps, mode):
        X = _to_local_if_dtensor(X)

        X = X.contiguous()
        W = W.contiguous()
        B = B.contiguous()

        Y, X_flat, Mean, RSTD, TILE_SIZE, _selected_mode = _layer_norm_forward(X, W, B, eps, mode)
        ctx.TILE_SIZE = TILE_SIZE
        ctx.save_for_backward(X_flat, W, B, Mean, RSTD)
        return Y

    @staticmethod
    def backward(ctx, dY):
        dY = _to_local_if_dtensor(dY).contiguous()

        X, W, B, Mean, RSTD = ctx.saved_tensors
        # B is saved so dB can be returned in the bias's own dtype (and so its
        # gradient flows back through the same ctx). The backward kernel itself
        # doesn't read B — dB depends only on dY.
        dX, dW, dB = _layer_norm_backward(dY, X, W, B, Mean, RSTD, ctx.TILE_SIZE)
        # Match forward arity: (X, W, B, eps, mode)
        return dX, dW, dB, None, None


# ---------------------------------------------------------------------------
# Public registration
# ---------------------------------------------------------------------------
@register_op(
    "layer_norm",
    impl_name="nvidia-cutile",
    capability=Capability(
        min_cc=(10, 0),
        modules=["cuda.tile", "torch"],
        predicate=_cutile_compiler_available,
    ),
    modes=("standard", "static_persistent", "multi_wave_cached"),
    default_mode="static_persistent",
    # See sibling _cutile/rms_norm.py for measured-perf rationale: cuTile
    # LayerNorm also loses to both Triton and CuTeDSL across our shape sweep
    # (matches NVIDIA's own pptx — H100 fwd 0.98x, H100 bwd 0.93x, B200 bwd
    # 0.90x vs Triton). Last-fallback rank. Both forward and backward
    # materialise a single ``next_pow2(N)``-wide tile per row, so the hidden
    # dimension is capped at 8192 (``_MAX_TILE``) on both sides; wider rows
    # raise a clear error rather than silently mis-normalising.
    preference_rank=80,
    tolerances={
        torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
        torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
        # Two-pass reduction (mean + variance) accumulates more rounding than
        # RMSNorm; atol=5e-4 absorbs N up to ~32K at unit magnitude.
        torch.float32: {"atol_fwd": 5e-4, "atol_bwd": 1e-3, "rtol_fwd": 1e-4, "rtol_bwd": 1e-3},
    },
    notes="cuTile static-persistent + multi-wave LayerNorm for Blackwell.",
)
def layer_norm_cutile(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """Dispatch entry-point for the cuTile LayerNorm backend.

    ``bias`` is always a real tensor — :func:`liger_kernel.functional.layer_norm`
    substitutes a zeros tensor when the user passes ``None``. We add it
    unconditionally inside the kernel; we deliberately do not special-case
    "all zeros" because the cost of the add at this scale is dwarfed by the
    memory traffic for X.
    """
    if mode is not None and mode not in _VALID_MODES:
        raise ValueError(f"cuTile layer_norm: unknown mode {mode!r}; valid modes are {_VALID_MODES}.")
    return _LigerLayerNormCuTileFunction.apply(x, weight, bias, eps, mode)
