# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
SwiGLU activation kernel (CuTile backend).

Computes: c = silu(a * gate_multiplier) * b  where silu(x) = x * sigmoid(x)

Row-parallel: grid = (n_rows, 1, 1). Each block handles one row.
Backward writes da into A and db into B in-place (memory optimization).

gate_multiplier: applied inside the kernel as ct.Constant[float] (compile-time
  constant; scales a before SiLU; chain rule applies extra factor in backward).
down_multiplier: applied at the Python wrapper level only (multiplied onto output
  in forward; multiplied onto dc before backward kernel dispatch). Not in kernel.

Two kernel variants per direction (fwd/bwd):
  *_aligned: check_bounds=False — used when n_cols % BLOCK_SIZE == 0
             (all power-of-2 n_cols up to MAX_FUSED_SIZE, e.g. 4096, 8192),
             ~17-20% faster vs check_bounds=True on B200.
  *_ct:      check_bounds=True  — fallback for non-aligned n_cols.

Forward uses @ct.kernel(occupancy=1) → 8 warps and the exp2 trick:
  sigmoid via exp2(-a * LOG2E) → FMUL+EX2 on Blackwell. occupancy=1 is required
  for the exp2→EX2 lowering. Backward does NOT set occupancy=1 (scatter inside a
  backward loop risks hangs), so it uses exp(-a) instead of exp2.
"""

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2

MAX_FUSED_SIZE_FWD = 4096  # Forward: larger tile fits; forward is compute-bound, no register spill observed
MAX_FUSED_SIZE_BWD = 1024  # Backward: 14 chunks at n_cols=14336 (vs 28 at 512); stable without occupancy=1

# exp2 trick: sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + exp2(-x * LOG2E))
# Using exp2(x * LOG2E) instead of exp(x) avoids Cody-Waite range reduction
# and maps to FMUL+EX2 on Blackwell (same as Triton's native sigmoid).
# CRITICAL: Only effective with @ct.kernel(occupancy=1) — without it, ct.exp2 calls exp internally.
LOG2E: float = 1.4426950408889634  # log2(e) = 1/ln(2)


@ct.kernel(occupancy=1, num_worker_warps=8)
def _swiglu_fwd_ct_aligned(
    A,  # (n_rows, n_cols) input a
    B,  # (n_rows, n_cols) input b
    C,  # (n_rows, n_cols) output c
    n_cols: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    gate_multiplier: ct.Constant[float],
):
    """
    SwiGLU forward — aligned fast path (check_bounds=False).

    Safe only when n_cols % BLOCK_SIZE == 0 (no out-of-bounds accesses).
    ~17-20% faster than the bounds-checked variant on B200.
    Computes: c = silu(a * gate_multiplier) * b
    """
    row_idx = ct.bid(0)
    n_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)

        a = ct.astype(ct.gather(A, (row_idx, col_idx), check_bounds=False, padding_value=0.0), ct.float32)
        b = ct.gather(B, (row_idx, col_idx), check_bounds=False, padding_value=0.0)

        # Apply gate_multiplier before SiLU (Liger convention)
        a_scaled = a * gate_multiplier

        # exp2 trick + flush_to_zero: sigmoid via exp2(-a*LOG2E) — FMUL+EX2 (avoids Cody-Waite range reduction).
        # flush_to_zero=True skips denormal handling; sigmoid range is well above the denormal threshold.
        # Requires occupancy=1 for correct exp2→EX2 lowering.
        sig_a = ct.truediv(
            1.0,
            1.0 + ct.exp2(ct.mul(-a_scaled, LOG2E), flush_to_zero=True),
            flush_to_zero=True,
            rounding_mode=ct.RoundingMode.APPROX,
        )
        silu_a = a_scaled * sig_a

        c = ct.astype(silu_a, b.dtype) * b
        ct.scatter(C, (row_idx, col_idx), c, check_bounds=False)


@ct.kernel(occupancy=1, num_worker_warps=8)
def _swiglu_fwd_ct(
    A,  # (n_rows, n_cols) input a
    B,  # (n_rows, n_cols) input b
    C,  # (n_rows, n_cols) output c
    n_cols: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    gate_multiplier: ct.Constant[float],
):
    """
    SwiGLU forward — general path (check_bounds=True).

    Handles arbitrary n_cols. Used as fallback when n_cols % BLOCK_SIZE != 0.
    Computes: c = silu(a * gate_multiplier) * b
    """
    row_idx = ct.bid(0)
    n_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)

        a = ct.astype(ct.gather(A, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        b = ct.gather(B, (row_idx, col_idx), check_bounds=True, padding_value=0.0)

        # Apply gate_multiplier before SiLU (Liger convention)
        a_scaled = a * gate_multiplier

        # exp2 trick + flush_to_zero: sigmoid via exp2(-a*LOG2E) — FMUL+EX2 (avoids Cody-Waite range reduction).
        # flush_to_zero=True skips denormal handling; sigmoid range is well above the denormal threshold.
        # Requires occupancy=1 for correct exp2→EX2 lowering.
        sig_a = ct.truediv(
            1.0,
            1.0 + ct.exp2(ct.mul(-a_scaled, LOG2E), flush_to_zero=True),
            flush_to_zero=True,
            rounding_mode=ct.RoundingMode.APPROX,
        )
        silu_a = a_scaled * sig_a

        c = ct.astype(silu_a, b.dtype) * b
        ct.scatter(C, (row_idx, col_idx), c, check_bounds=True)


@ct.kernel
def _swiglu_bwd_ct_aligned(
    DC,  # (n_rows, n_cols) upstream gradient
    A,  # (n_rows, n_cols) saved input a — DA written in-place
    B,  # (n_rows, n_cols) saved input b — DB written in-place
    n_cols: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    gate_multiplier: ct.Constant[float],
):
    """
    SwiGLU backward — aligned fast path (check_bounds=False).

    Safe only when n_cols % BLOCK_SIZE == 0. da/db written in-place to A/B.
    NOTE: No occupancy=1 — scatter inside a backward loop risks hangs.

    Chain rule: fwd computes c = silu(a * gm) * b
      db = dc * silu(a * gm)
      da = dc * d_silu(a*gm)/d(a*gm) * gm * b
         = dc * (silu(a*gm) * (1 - sig(a*gm)) + sig(a*gm)) * gm * b
    """
    row_idx = ct.bid(0)
    n_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)

        dc = ct.astype(ct.gather(DC, (row_idx, col_idx), check_bounds=False, padding_value=0.0), ct.float32)
        # A holds original a (forward did not write back); reapply gate_multiplier
        a = ct.astype(ct.gather(A, (row_idx, col_idx), check_bounds=False, padding_value=0.0), ct.float32)
        b = ct.astype(ct.gather(B, (row_idx, col_idx), check_bounds=False, padding_value=0.0), ct.float32)

        a_scaled = a * gate_multiplier
        sig_a = ct.truediv(1.0, 1.0 + ct.exp(0.0 - a_scaled), rounding_mode=ct.RoundingMode.APPROX)
        silu_a = a_scaled * sig_a

        db = dc * silu_a
        da = dc * (silu_a * (1.0 - sig_a) + sig_a) * b * gate_multiplier

        ct.scatter(A, (row_idx, col_idx), ct.astype(da, A.dtype), check_bounds=False)
        ct.scatter(B, (row_idx, col_idx), ct.astype(db, B.dtype), check_bounds=False)


@ct.kernel
def _swiglu_bwd_ct(
    DC,  # (n_rows, n_cols) upstream gradient
    A,  # (n_rows, n_cols) saved input a — DA written in-place
    B,  # (n_rows, n_cols) saved input b — DB written in-place
    n_cols: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    gate_multiplier: ct.Constant[float],
):
    """
    SwiGLU backward — general path (check_bounds=True).

    Recomputes sigmoid for memory efficiency (no saved activations).
    da/db written in-place to A/B. Grid: (n_rows, 1, 1).
    NOTE: No occupancy=1 — scatter inside a backward loop risks hangs.
    """
    row_idx = ct.bid(0)
    n_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)

        dc = ct.astype(ct.gather(DC, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        # A holds original a (forward did not write back); reapply gate_multiplier
        a = ct.astype(ct.gather(A, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        b = ct.astype(ct.gather(B, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)

        a_scaled = a * gate_multiplier
        # Recompute sigmoid (APPROX division: no FCHK + software-fallback CALL)
        sig_a = ct.truediv(1.0, 1.0 + ct.exp(0.0 - a_scaled), rounding_mode=ct.RoundingMode.APPROX)
        silu_a = a_scaled * sig_a

        db = dc * silu_a
        da = dc * (silu_a * (1.0 - sig_a) + sig_a) * b * gate_multiplier

        ct.scatter(A, (row_idx, col_idx), ct.astype(da, A.dtype), check_bounds=True)
        ct.scatter(B, (row_idx, col_idx), ct.astype(db, B.dtype), check_bounds=True)


def _calculate_block_size(n_cols, max_fused_size):
    # Cap the tile at max_fused_size (or next_pow2(n_cols) if smaller).
    block = max(min(_next_power_of_2(n_cols), max_fused_size), 128)
    # Largest power-of-2 tile <= block that evenly divides n_cols — this enables the
    # check_bounds=False aligned fast path (which dispatch selects when block % n_cols == 0).
    aligned = block
    while aligned > 128 and n_cols % aligned != 0:
        aligned //= 2
    # Prefer the aligned block only when it stays large (>= half the cap). For sizes with small
    # odd factors (e.g. 11008 = 256*43, 13824 = 512*27) the largest aligned block collapses to a
    # tiny tile with dozens of chunks; there, keep the full block and let the masked
    # (check_bounds=True) kernel cover the remainder in far fewer chunks (~10% faster).
    # 14336 = 2048*7 keeps a large aligned block (2048) and stays on the fast path.
    if n_cols % aligned == 0 and aligned >= block // 2:
        return aligned
    return block


class LigerSiLUMulFunction(torch.autograd.Function):
    """CuTile autograd wrapper for SwiGLU (silu(a * gate_multiplier) * b * down_multiplier).

    gate_multiplier is applied inside the kernel (consistent with Liger-Kernel).
    down_multiplier is applied at the Python wrapper level.
    """

    @staticmethod
    def forward(ctx, a, b, gate_multiplier: float = 1.0, down_multiplier: float = 1.0):
        gate_multiplier = float(gate_multiplier)
        down_multiplier = float(down_multiplier)
        ori_shape = a.shape
        n_cols = ori_shape[-1]
        a = a.view(-1, n_cols).contiguous()
        b = b.view(-1, n_cols).contiguous()
        n_rows = a.shape[0]

        c = torch.empty_like(a)
        BLOCK_SIZE = _calculate_block_size(n_cols, MAX_FUSED_SIZE_FWD)
        fwd_kernel = _swiglu_fwd_ct_aligned if n_cols % BLOCK_SIZE == 0 else _swiglu_fwd_ct

        ct.launch(
            torch.cuda.current_stream(),
            (n_rows, 1, 1),
            fwd_kernel,
            (a, b, c, int(n_cols), int(BLOCK_SIZE), gate_multiplier),
        )
        c_out = c.view(*ori_shape)
        if down_multiplier != 1.0:
            c_out = c_out * down_multiplier
        ctx.save_for_backward(a, b)
        ctx.ori_shape = ori_shape
        ctx.gate_multiplier = gate_multiplier
        ctx.down_multiplier = down_multiplier
        return c_out

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        ori_shape = ctx.ori_shape
        n_cols = ori_shape[-1]
        dc = dc.view(-1, n_cols).contiguous()
        n_rows = dc.shape[0]
        if ctx.down_multiplier != 1.0:
            dc = dc * ctx.down_multiplier
        BLOCK_SIZE = _calculate_block_size(n_cols, MAX_FUSED_SIZE_BWD)
        bwd_kernel = _swiglu_bwd_ct_aligned if n_cols % BLOCK_SIZE == 0 else _swiglu_bwd_ct

        ct.launch(
            torch.cuda.current_stream(),
            (n_rows, 1, 1),
            bwd_kernel,
            (dc, a, b, int(n_cols), int(BLOCK_SIZE), ctx.gate_multiplier),
        )
        return a.view(*ori_shape), b.view(*ori_shape), None, None
