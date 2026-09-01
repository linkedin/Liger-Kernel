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

Both directions use ct.gather / ct.scatter with a contiguous ct.arange index —
the leanest, fully-coalesced lowering on B200 (block ct.load / ct.store emit ~3x
more address-arithmetic instructions for identical memory traffic and are slower
here for this row-wise pattern).

Forward — exact-fit power-of-2 tiling (removes the non-power-of-2 cliff)
-----------------------------------------------------------------------
The forward is memory-bound and highly sensitive to the tiling. Whenever no large
power-of-2 tile divides n_cols evenly, a single uniform-tile kernel must fall back
to check_bounds=True (masking) for the overhang tile, which was ~1.4-1.6x slower
on the non-power-of-2 intermediate sizes of e.g. Qwen2.5 (11008, 13824, 18944).

Instead we decompose each row into exact-fit descending power-of-2 chunks so every
chunk is fully in-bounds and uses the fast check_bounds=False path:

  * n_cols // BASE full BASE chunks via a uniform ``for`` loop, then
  * the n_cols % BASE remainder as a fixed ladder of ``if <present>:`` guards for
    chunk widths BASE/2 .. 1 (each guard is constant-folded away when absent).

``ct.arange`` requires a power-of-2 size and cuTile compiles ``for`` / ``while`` as
*device* loops (the loop variable is a runtime value), so a single loop cannot vary
the chunk width. The kernel is therefore built by a factory keyed on n_cols
(cached) that bakes the decomposition as compile-time scalars. BASE = 2048 gives
the best DRAM utilisation on B200 (4096 tiles become issue-bound; see git history).

Forward uses @ct.kernel(occupancy=1) → 8 warps and the exp2 trick:
  sigmoid via exp2(-a * LOG2E) → FMUL+EX2 on Blackwell. occupancy=1 is required
  for the exp2→EX2 lowering.

Backward — uniform tiling with adaptive block size
--------------------------------------------------
The backward is more compute-heavy per element, so the check_bounds=True predicate
is fully hidden behind the math: measured backward throughput is flat across
aligned and non-aligned n_cols (no cliff). It therefore keeps the simpler uniform
kernel with an adaptive block size and does NOT set occupancy=1 (uses exp(-a)).
"""

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.utils import device_context

# Forward base tile for the exact-fit decomposition (occupancy=1 + exp2). 2048
# gives the best DRAM utilisation on B200; larger tiles (4096) become issue-bound.
MAX_FUSED_SIZE_FWD = 2048
# Backward base tile cap. Backward reads dc/a/b and writes da/db in-place, so it
# runs at a smaller cap to keep register pressure low.
MAX_FUSED_SIZE_BWD = 1024

# exp2 trick: sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + exp2(-x * LOG2E))
# Using exp2(x * LOG2E) instead of exp(x) avoids Cody-Waite range reduction
# and maps to FMUL+EX2 on Blackwell (same as Triton's native sigmoid).
# CRITICAL: Only effective with @ct.kernel(occupancy=1) — without it, ct.exp2 calls exp internally.
LOG2E: float = 1.4426950408889634  # log2(e) = 1/ln(2)


# ---------------------------------------------------------------------------
# Forward — exact-fit power-of-2 gather/scatter
# ---------------------------------------------------------------------------
def _swiglu_fwd_chunk(A, B, C, row_idx, col_offset, BLOCK, gate_multiplier):
    """Forward SwiGLU on one exact-fit chunk of ``BLOCK`` contiguous columns.

    Inlined into the generated forward kernel; ``BLOCK`` is a compile-time literal
    power of two and ``[col_offset, col_offset + BLOCK)`` is fully in-bounds, so
    ``check_bounds=False`` is safe.
    """
    col_idx = ct.add(ct.arange(BLOCK, dtype=ct.int32), col_offset)

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


def _remainder_offsets(n_cols, base):
    """Column offsets for the exact-fit descending power-of-2 remainder chunks.

    Returns ``{chunk_width: col_offset}`` for each active power in ``base//2 .. 1``
    covering ``n_cols % base`` columns after the uniform ``base`` chunks.
    """
    plan = {}
    offset = (n_cols // base) * base
    remaining = n_cols % base
    size = base // 2
    while size >= 1:
        if remaining >= size:
            plan[size] = offset
            offset += size
            remaining -= size
        size //= 2
    return plan


_FWD_KERNEL_CACHE = {}


def _make_fwd_kernel(n_cols):
    base = MAX_FUSED_SIZE_FWD
    n_full = n_cols // base
    plan = _remainder_offsets(n_cols, base)
    # Presence + column offset per pow2 level (base//2 .. 1), baked as compile-time scalars.
    p1024, c1024 = 1024 in plan, plan.get(1024, 0)
    p512, c512 = 512 in plan, plan.get(512, 0)
    p256, c256 = 256 in plan, plan.get(256, 0)
    p128, c128 = 128 in plan, plan.get(128, 0)
    p64, c64 = 64 in plan, plan.get(64, 0)
    p32, c32 = 32 in plan, plan.get(32, 0)
    p16, c16 = 16 in plan, plan.get(16, 0)
    p8, c8 = 8 in plan, plan.get(8, 0)
    p4, c4 = 4 in plan, plan.get(4, 0)
    p2, c2 = 2 in plan, plan.get(2, 0)
    p1, c1 = 1 in plan, plan.get(1, 0)

    @ct.kernel(occupancy=1, num_worker_warps=8)
    def _swiglu_fwd_ct(
        A,  # (n_rows, n_cols) input a
        B,  # (n_rows, n_cols) input b
        C,  # (n_rows, n_cols) output c
        gate_multiplier: ct.Constant[float],
    ):
        row_idx = ct.bid(0)
        for ci in range(n_full):
            _swiglu_fwd_chunk(A, B, C, row_idx, ci * base, base, gate_multiplier)
        # Exact-fit remainder ladder (absent powers constant-fold away).
        if p1024:
            _swiglu_fwd_chunk(A, B, C, row_idx, c1024, 1024, gate_multiplier)
        if p512:
            _swiglu_fwd_chunk(A, B, C, row_idx, c512, 512, gate_multiplier)
        if p256:
            _swiglu_fwd_chunk(A, B, C, row_idx, c256, 256, gate_multiplier)
        if p128:
            _swiglu_fwd_chunk(A, B, C, row_idx, c128, 128, gate_multiplier)
        if p64:
            _swiglu_fwd_chunk(A, B, C, row_idx, c64, 64, gate_multiplier)
        if p32:
            _swiglu_fwd_chunk(A, B, C, row_idx, c32, 32, gate_multiplier)
        if p16:
            _swiglu_fwd_chunk(A, B, C, row_idx, c16, 16, gate_multiplier)
        if p8:
            _swiglu_fwd_chunk(A, B, C, row_idx, c8, 8, gate_multiplier)
        if p4:
            _swiglu_fwd_chunk(A, B, C, row_idx, c4, 4, gate_multiplier)
        if p2:
            _swiglu_fwd_chunk(A, B, C, row_idx, c2, 2, gate_multiplier)
        if p1:
            _swiglu_fwd_chunk(A, B, C, row_idx, c1, 1, gate_multiplier)

    return _swiglu_fwd_ct


def _get_fwd_kernel(n_cols):
    kernel = _FWD_KERNEL_CACHE.get(n_cols)
    if kernel is None:
        kernel = _make_fwd_kernel(n_cols)
        _FWD_KERNEL_CACHE[n_cols] = kernel
    return kernel


# ---------------------------------------------------------------------------
# Backward — uniform tiling with adaptive block size (no cliff in backward)
# ---------------------------------------------------------------------------
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
        with device_context(a.device):
            gate_multiplier = float(gate_multiplier)
            down_multiplier = float(down_multiplier)
            ori_shape = a.shape
            n_cols = ori_shape[-1]
            a = a.view(-1, n_cols).contiguous()
            b = b.view(-1, n_cols).contiguous()
            n_rows = a.shape[0]

            c = torch.empty_like(a)
            fwd_kernel = _get_fwd_kernel(int(n_cols))

            ct.launch(
                torch.cuda.current_stream(),
                (n_rows, 1, 1),
                fwd_kernel,
                (a, b, c, gate_multiplier),
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
        with device_context(dc.device):
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
