# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
RMS Normalization kernel (CuTile backend).

Y = X / RMS(X) * (W + offset), RMS = sqrt(mean(x^2) + eps).

Forward kernel: row-parallel (one block per row), single pass.
  col_idx = arange(BLOCK_SIZE); gather X (check_bounds=True pads OOB to 0),
  compute rstd, scatter Y.

Backward kernel: SM-count partitioned, single DRAM pass per row (all BLOCK_SIZE).
  - W loaded once per block; dW accumulated in registers, scattered once at end.
  - dW_partial shape: (sm_count, n_cols) instead of (n_rows, n_cols).

Casting modes:
  - "llama" (0): X cast to fp32 for RMS; X*rstd cast BACK to X.dtype before W multiply.
                 RSTD stored as fp32.
  - "gemma" (1): Both X and W cast to fp32; Y cast back to X.dtype.
                 RSTD stored as fp32.
  - "none" (-1): No casting. Everything in X.dtype. RSTD stored in X.dtype.

Uses gather/scatter with check_bounds=True for arbitrary n_cols.

row_mode is accepted for signature parity with the Triton LigerRMSNormFunction but
is ignored: this backend always uses the row-parallel path.
"""

import math

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2

_CASTING_MODE_NONE = -1
_CASTING_MODE_LLAMA = 0
_CASTING_MODE_GEMMA = 1

_str_to_casting_mode = {
    "llama": _CASTING_MODE_LLAMA,
    "gemma": _CASTING_MODE_GEMMA,
    "none": _CASTING_MODE_NONE,
}


def _calculate_settings(n_cols):
    BLOCK_SIZE = _next_power_of_2(n_cols)
    if BLOCK_SIZE > 65536:
        raise RuntimeError(f"Feature dimension {n_cols} exceeds maximum supported (65536)")
    return BLOCK_SIZE


# ---------------------------------------------------------------------------
# Forward kernel (row-parallel)
# ---------------------------------------------------------------------------


@ct.kernel
def _rms_norm_fwd_ct(
    Y,  # (n_rows, n_cols) output
    X,  # (n_rows, n_cols) input
    W,  # (n_cols,) affine weight (dummy 1-element tensor when elementwise_affine=False)
    RSTD,  # (n_rows,) cached rstd
    n_cols,
    eps,  # runtime float
    offset,  # runtime float
    BLOCK_SIZE: ct.Constant[int],
    casting_mode: ct.Constant[int],
    elementwise_affine: ct.Constant[bool],
):
    """
    RMS norm forward (unified, single pass).

    Row-parallel forward pass:
      col_idx = arange(BLOCK_SIZE)   # BLOCK_SIZE = next_power_of_2(n_cols)
      load X (check_bounds=True → OOB elements zero-padded, harmless for RMS sum)
      compute rstd, store RSTD
      scale X; optionally multiply by (W + offset)
      store Y

    elementwise_affine is a compile-time constant — the W gather/multiply is
    dead-code-eliminated when False, so the no-weight path has zero overhead.

    casting_mode:
      llama (0): X cast to fp32 for RMS; X*rstd cast BACK to X.dtype before W multiply.
      gemma (1): Both X and W cast to fp32; Y cast back to X.dtype.
      none (-1): Compute in X.dtype (no upcast). x*x accumulated in X.dtype;
        division by n_cols promotes to fp32. eps/offset rounded to X.dtype before
        arithmetic.
    """
    row_idx = ct.bid(0)
    col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32)

    if casting_mode == _CASTING_MODE_NONE:
        x_val = ct.gather(X, (row_idx, col_idx), check_bounds=True, padding_value=0.0)
        if elementwise_affine:
            w_val = ct.gather(W, col_idx, check_bounds=True, padding_value=0.0)
        mean_sq = ct.astype(ct.sum(x_val * x_val, 0, keepdims=False), ct.float32) / n_cols
        eps_rounded = ct.astype(ct.astype(eps, x_val.dtype), ct.float32)
        rstd = ct.rsqrt(mean_sq + eps_rounded)  # fp32
        ct.scatter(RSTD, row_idx, ct.astype(rstd, x_val.dtype), check_bounds=False)
        x_scaled = ct.astype(x_val, ct.float32) * rstd  # fp32 (upcast x for RMS computation)
        if elementwise_affine:
            offset_native = ct.astype(offset, x_val.dtype)  # round offset to X.dtype precision
            w_plus_offset_f32 = ct.astype(w_val + offset_native, ct.float32)
            ct.scatter(Y, (row_idx, col_idx), ct.astype(x_scaled * w_plus_offset_f32, x_val.dtype), check_bounds=True)
        else:
            ct.scatter(Y, (row_idx, col_idx), ct.astype(x_scaled, x_val.dtype), check_bounds=True)

    elif casting_mode == _CASTING_MODE_LLAMA:
        x_val = ct.gather(X, (row_idx, col_idx), check_bounds=True, padding_value=0.0)
        if elementwise_affine:
            w_val = ct.gather(W, col_idx, check_bounds=True, padding_value=0.0)
        x_f32 = ct.astype(x_val, ct.float32)
        mean_sq = ct.sum(ct.mul(x_f32, x_f32, flush_to_zero=True), 0, keepdims=False) / n_cols
        rstd = ct.rsqrt(mean_sq + eps)
        ct.scatter(RSTD, row_idx, rstd, check_bounds=False)
        # Cast X*rstd back to X.dtype before W multiply (llama behaviour)
        x_scaled = ct.astype(x_f32 * rstd, X.dtype)
        if elementwise_affine:
            ct.scatter(Y, (row_idx, col_idx), ct.astype(x_scaled * (w_val + offset), Y.dtype), check_bounds=True)
        else:
            ct.scatter(Y, (row_idx, col_idx), ct.astype(x_scaled, Y.dtype), check_bounds=True)

    else:
        # gemma: both X and W to fp32, Y cast back to X.dtype
        x_f32 = ct.astype(ct.gather(X, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        if elementwise_affine:
            w_f32 = ct.astype(ct.gather(W, col_idx, check_bounds=True, padding_value=0.0), ct.float32)
        mean_sq = ct.sum(x_f32 * x_f32, 0, keepdims=False) / n_cols
        rstd = ct.rsqrt(mean_sq + eps)
        ct.scatter(RSTD, row_idx, rstd, check_bounds=False)
        x_scaled = x_f32 * rstd
        if elementwise_affine:
            ct.scatter(Y, (row_idx, col_idx), ct.astype(x_scaled * (w_f32 + offset), Y.dtype), check_bounds=True)
        else:
            ct.scatter(Y, (row_idx, col_idx), ct.astype(x_scaled, Y.dtype), check_bounds=True)


_rms_norm_fwd_large_ct = _rms_norm_fwd_ct.replace_hints(num_worker_warps=8)


# ---------------------------------------------------------------------------
# Backward kernels — SM-count grid, single DRAM pass
# ---------------------------------------------------------------------------


@ct.kernel
def _rms_norm_bwd_large_ct(
    dY,  # (n_rows, n_cols) upstream gradient
    dX,  # (n_rows, n_cols) output gradient
    X,  # (n_rows, n_cols) saved input
    RSTD,  # (n_rows,) cached rstd; OOB-safe via bounds-checked gather
    n_cols: ct.Constant[int],
    rows_per_program: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    casting_mode: ct.Constant[int],
):
    """
    RMS norm backward without affine weight. SM-count partitioned, single DRAM pass.

    Grid: (sm_count,). Block b processes rows [b*rpp, (b+1)*rpp).
    Single pass: load dY and X once; sum_mX via register reduction; no re-read.
    OOB rows return 0 via check_bounds; RSTD zero-padded for safe scalar load.
    """
    block_id = ct.bid(0)
    col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32)
    inv_n_cols = 1.0 / n_cols

    for ri in range(rows_per_program):
        row_idx = block_id * rows_per_program + ri

        rstd = ct.astype(ct.gather(RSTD, (row_idx,), padding_value=0.0).item(), ct.float32)
        dy_f32 = ct.astype(
            ct.gather(dY, (row_idx, col_idx), check_bounds=True, padding_value=0.0, latency=3), ct.float32
        )
        x_f32 = ct.astype(ct.gather(X, (row_idx, col_idx), check_bounds=True, padding_value=0.0, latency=3), ct.float32)

        sum_mX = ct.sum(dy_f32 * x_f32, 0, keepdims=False)
        coeff = rstd * rstd * rstd * inv_n_cols * sum_mX
        dx_f32 = rstd * dy_f32 - coeff * x_f32
        ct.scatter(dX, (row_idx, col_idx), ct.astype(dx_f32, dX.dtype), check_bounds=True)


@ct.kernel
def _rms_norm_bwd_w_large_ct(
    dY,  # (n_rows, n_cols) upstream gradient
    dX,  # (n_rows, n_cols) output gradient
    X,  # (n_rows, n_cols) saved input
    W,  # (n_cols,) affine weight
    RSTD,  # (n_rows,) cached rstd; OOB-safe via bounds-checked gather
    dW_partial,  # (sm_count, n_cols) per-block dW accumulation (host reduces)
    n_cols: ct.Constant[int],
    offset: ct.Constant[float],
    rows_per_program: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    casting_mode: ct.Constant[int],
):
    """
    RMS norm backward with affine weight. SM-count partitioned, single DRAM pass.

    Grid: (sm_count,). Block b processes rows [b*rpp, (b+1)*rpp).
    W loaded once per block and reused across all rows.
    dW accumulated in registers throughout the row loop; scattered once at the end.
    dW_partial shape: (sm_count, n_cols) — vastly smaller than (n_rows, n_cols).
    OOB rows return 0 via check_bounds; RSTD zero-padded.

    casting_mode:
      llama (0): load W in original dtype once; per row: dY in orig dtype,
                 m = (dY*(W+offset)) cast to fp32; dW += dy_orig*(X*rstd cast to X.dtype).
      gemma (1): W loaded in fp32; per row: dY in fp32, m = dy_f32*(w_f32+offset);
                 dW += dy_f32 * x_f32 * rstd.
      none (-1): load W in original dtype once; per row: dY in orig dtype,
                 m = dy_orig*(w_orig+offset) without cast to fp32 (cast for sum);
                 dW += dy_orig * (x_orig * rstd) without extra fp32 cast.
    """
    block_id = ct.bid(0)
    col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32)

    # Per-block dW accumulator in registers; scattered to dW_partial once at end
    dW_acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    inv_n_cols = 1.0 / n_cols

    # Load W once; dtype depends on mode (gemma keeps fp32; llama/none keep original).
    if casting_mode == _CASTING_MODE_GEMMA:
        w_f32 = ct.astype(ct.gather(W, col_idx, check_bounds=True, padding_value=0.0), ct.float32)
    else:
        w_orig = ct.gather(W, col_idx, check_bounds=True, padding_value=0.0)

    for ri in range(rows_per_program):
        row_idx = block_id * rows_per_program + ri

        # Bounds-checked scalar read on RSTD (avoids host-side cat-padding).
        rstd = ct.astype(ct.gather(RSTD, (row_idx,), padding_value=0.0).item(), ct.float32)
        x_f32 = ct.astype(ct.gather(X, (row_idx, col_idx), check_bounds=True, padding_value=0.0, latency=3), ct.float32)

        if casting_mode == _CASTING_MODE_GEMMA:
            dy_f32 = ct.astype(
                ct.gather(dY, (row_idx, col_idx), check_bounds=True, padding_value=0.0, latency=3), ct.float32
            )
            m_f32 = dy_f32 * (w_f32 + offset)
            dW_term_f32 = dy_f32 * x_f32 * rstd
        else:
            dy_orig = ct.gather(dY, (row_idx, col_idx), check_bounds=True, padding_value=0.0, latency=3)
            m_f32 = ct.astype(dy_orig * (w_orig + offset), ct.float32)
            if casting_mode == _CASTING_MODE_LLAMA:
                # x*rstd computed in fp32, then downcast for the dy multiply
                dW_term_f32 = ct.astype(dy_orig * ct.astype(x_f32 * rstd, dY.dtype), ct.float32)
            else:
                # none: downcast x first, then multiply by rstd (mixed-precision)
                x_orig = ct.astype(x_f32, dY.dtype)
                dW_term_f32 = ct.astype(dy_orig * (x_orig * rstd), ct.float32)

        sum_mX = ct.sum(m_f32 * x_f32, 0, keepdims=False)
        coeff = rstd * rstd * rstd * inv_n_cols * sum_mX
        dx_f32 = rstd * m_f32 - coeff * x_f32
        ct.scatter(dX, (row_idx, col_idx), ct.astype(dx_f32, dX.dtype), check_bounds=True)

        # OOB rows contribute 0 (dy=0, x=0 via check_bounds)
        dW_acc = ct.add(dW_acc, dW_term_f32)

    # Write this block's partial dW once (block_id < sm_count, always in-bounds)
    ct.scatter(dW_partial, (block_id, col_idx), dW_acc, check_bounds=True)


_rms_norm_bwd_large_ct_nww8 = _rms_norm_bwd_large_ct.replace_hints(num_worker_warps=8)
_rms_norm_bwd_w_large_ct_nww8 = _rms_norm_bwd_w_large_ct.replace_hints(num_worker_warps=8)


# ---------------------------------------------------------------------------
# Python launch wrappers
# ---------------------------------------------------------------------------


def _rms_norm_forward_ct(X, W, eps, offset, casting_mode_int):
    shape = X.shape
    dim = shape[-1]
    X2d = X.view(-1, dim).contiguous()
    n_rows, n_cols = X2d.shape
    BLOCK_SIZE = _calculate_settings(n_cols)

    Y = torch.empty_like(X2d)
    # RSTD dtype: fp32 for llama/gemma, X.dtype for none
    rstd_dtype = torch.float32 if casting_mode_int in (_CASTING_MODE_LLAMA, _CASTING_MODE_GEMMA) else X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)
    elementwise_affine = W is not None

    grid = (n_rows, 1, 1)
    # When no weight, pass a 1-element dummy tensor; elementwise_affine=False causes the compiler
    # to dead-code-eliminate every ct.gather(W, ...) so the dummy is never accessed.
    W_tensor = W.contiguous() if elementwise_affine else X2d.new_empty(1)
    fwd_kernel = _rms_norm_fwd_large_ct if BLOCK_SIZE >= 16384 else _rms_norm_fwd_ct
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fwd_kernel,
        (
            Y,
            X2d,
            W_tensor,
            RSTD,
            int(n_cols),
            float(eps),
            float(offset) if elementwise_affine else 0.0,
            int(BLOCK_SIZE),
            int(casting_mode_int),
            bool(elementwise_affine),
        ),
    )

    return Y.view(*shape), X2d, RSTD, int(BLOCK_SIZE)


def _rms_norm_backward_ct(dY, X, W, RSTD, offset, BLOCK_SIZE, casting_mode_int, in_place):
    shape = dY.shape
    dim = shape[-1]
    dY2d = dY.view(-1, dim).contiguous()
    n_rows, n_cols = dY2d.shape
    elementwise_affine = W is not None

    sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count, 1, 1)

    # When in_place=True, reuse dY2d buffer in-place for dX (safe: each row is processed
    # independently, and within a row the load precedes the store in every kernel pass).
    dX = dY2d if in_place else torch.zeros_like(dY2d)

    if elementwise_affine:
        # Every (block_id, col) in dW_partial is written exactly once by the scatter at
        # kernel end, so zero-init is unnecessary.
        dW_partial = torch.empty(sm_count, n_cols, dtype=torch.float32, device=W.device)
        # Larger hidden dims spill registers under the default nww=4; nww=8 fixes that.
        bwd_w_kernel = _rms_norm_bwd_w_large_ct_nww8 if BLOCK_SIZE >= 8192 else _rms_norm_bwd_w_large_ct
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            bwd_w_kernel,
            (
                dY2d,
                dX,
                X.contiguous(),
                W.contiguous(),
                RSTD,
                dW_partial,
                int(n_cols),
                float(offset),
                int(rows_per_program),
                int(BLOCK_SIZE),
                int(casting_mode_int),
            ),
        )
        dW = dW_partial.sum(dim=0).to(W.dtype)
    else:
        bwd_kernel = _rms_norm_bwd_large_ct_nww8 if BLOCK_SIZE >= 8192 else _rms_norm_bwd_large_ct
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            bwd_kernel,
            (
                dY2d,
                dX,
                X.contiguous(),
                RSTD,
                int(n_cols),
                int(rows_per_program),
                int(BLOCK_SIZE),
                int(casting_mode_int),
            ),
        )
        dW = None

    return dX.view(*shape), dW


class LigerRMSNormFunction(torch.autograd.Function):
    """CuTile autograd wrapper for RMS normalization.

    Signature-compatible with the Triton ``LigerRMSNormFunction`` so the cuTile backend
    swaps in transparently. ``row_mode`` is accepted but ignored (row-parallel only).
    """

    @staticmethod
    def forward(ctx, X, W, eps, offset=0.0, casting_mode="llama", in_place=True, row_mode=None):
        X = X.contiguous()
        if W is not None:
            W = W.contiguous()

        # Resolve casting_mode string → int
        if isinstance(casting_mode, int):
            casting_mode_int = casting_mode
        else:
            assert casting_mode in _str_to_casting_mode, f"Invalid casting_mode: {casting_mode}"
            casting_mode_int = _str_to_casting_mode[casting_mode]

        Y, X_saved, RSTD, BLOCK_SIZE = _rms_norm_forward_ct(X, W, eps, offset, casting_mode_int)

        ctx.offset = offset
        ctx.casting_mode = casting_mode_int
        ctx.in_place = in_place
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.elementwise_affine = W is not None
        if W is not None:
            ctx.save_for_backward(X_saved, W, RSTD)
        else:
            ctx.save_for_backward(X_saved, RSTD)
        return Y

    @staticmethod
    def backward(ctx, dY):
        dY = dY.contiguous()
        if ctx.elementwise_affine:
            X, W, RSTD = ctx.saved_tensors
        else:
            X, RSTD = ctx.saved_tensors
            W = None

        dX, dW = _rms_norm_backward_ct(dY, X, W, RSTD, ctx.offset, ctx.BLOCK_SIZE, ctx.casting_mode, ctx.in_place)
        return dX, dW, None, None, None, None, None
