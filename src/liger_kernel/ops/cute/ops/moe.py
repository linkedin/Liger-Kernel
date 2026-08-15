"""Fused expert-parallel MoE autograd op for the ``cute`` backend.

Thin autograd wrapper around the native fused MoE fwd/bwd kernels shipped by the
separate ``liger_cute_kernels`` (lck) wheel. Ported from LigerCommKernels'
``liger_comm_kernels/moe_ops.py``; the kernel ABI is identical, only the package
plumbing differs:

  - the TVM FFI facade is reached through the parent package's
    ``_load_tvm_ffi()`` (``liger_cute_kernels.tvm_ffi``), and
  - the expert-parallel ``ProcessGroup`` -> NVSHMEM team translation reuses
    ``liger_cute_kernels.nvshmem.resolve_team`` (which already caches per-PG).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Optional

import torch

from liger_kernel.ops.cute import _load_tvm_ffi

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

__all__ = ["LigerExpertParallelFusedMoEFunction", "moe_fused"]

# Resolve the TVM FFI facade once at import. cute/ops is imported only when the
# "cute" implementation is actively selected, so a missing lck wheel surfaces as
# a clear ImportError to the user who asked for it (see _load_tvm_ffi).
tvm_ffi = _load_tvm_ffi()


def _resolve_team(pg: Optional["ProcessGroup"]) -> int:
    """Return the NVSHMEM team prepared for the expert-parallel process group.

    ``pg=None`` and a process group spanning the NVSHMEM bootstrap team map to
    ``NVSHMEM_TEAM_WORLD`` without a collective. A proper subgroup must have
    been cached explicitly with ``liger_cute_kernels.nvshmem.resolve_team(pg)``
    during distributed setup; forward never creates a team collectively.
    """
    from liger_cute_kernels.nvshmem import resolve_team

    return resolve_team(pg, create=False)


class LigerExpertParallelFusedMoEFunction(torch.autograd.Function):
    """Autograd wrapper around the fused expert-parallel MoE fwd/bwd kernels.

    Takes the expert-parallel ``ProcessGroup`` directly and looks up its
    previously prepared NVSHMEM team inside ``forward``. The handle is stashed
    on ``ctx`` so ``backward`` reuses it.

    The forward always uses the with-intermediates kernel and saves every tensor
    bwd consumes on ``ctx``. The no-grad fast path is handled in ``moe_fused``
    *outside* this Function — checking ``torch.is_grad_enabled()`` inside
    ``forward`` is unsafe because PyTorch disables grad globally while running
    ``Function.forward``.

    Backward calls the bwd kernel and then ``moe_pop_fwd`` so the symmetric stack
    returns to its pre-fwd depth and the buffers are available for the next
    iteration.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        X: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        all_B: torch.Tensor,
        all_C: torch.Tensor,
        all_A: torch.Tensor,
        num_experts: int,
        top_k: int,
        pg: Optional["ProcessGroup"],
    ) -> torch.Tensor:
        if not all_B.is_contiguous() or not all_C.is_contiguous():
            raise ValueError(
                "Strided MoE gate/up weights are supported only when gradients "
                "are disabled; backward currently requires contiguous weights."
            )
        team_handle = _resolve_team(pg)

        (
            Y,
            x_sorted,
            y_buf,
            expert_offsets,
            token_expert_slots,
            tile_expert_ids,
            chosen_tile_m,
        ) = tvm_ffi.moe_fused_fwd_bf16(
            X,
            expert_indices,
            expert_weights,
            all_B,
            all_C,
            all_A,
            num_experts,
            top_k,
            team_handle,
        )

        ctx.save_for_backward(
            expert_indices,
            expert_weights,
            all_B,
            all_C,
            all_A,
            x_sorted,
            y_buf,
            token_expert_slots,
            tile_expert_ids,
            expert_offsets,
        )
        ctx.num_experts = num_experts
        ctx.top_k = top_k
        # Reuse the team resolved in forward — re-translating in backward would
        # re-run team_from_pg's collective (and is unsafe under graph capture).
        ctx.team_handle = team_handle
        # TileM the fwd autotuner picked. Bwd MUST run with the same TileM:
        # tile_expert_ids, expert_offsets, and x_sorted are all laid out at FWD's
        # TileM granularity. The C++ bwd auto entry enforces this via the
        # dispatch filter.
        ctx.fwd_tile_m = chosen_tile_m
        return Y

    @staticmethod
    def backward(ctx, dY: torch.Tensor):  # type: ignore[override]
        (
            expert_indices,
            expert_weights,
            all_B,
            all_C,
            all_A,
            x_sorted,
            y_buf,
            token_expert_slots,
            tile_expert_ids,
            expert_offsets,
        ) = ctx.saved_tensors

        dX, dB, dC, dA, dW = tvm_ffi.moe_fused_bwd_bf16(
            dY.contiguous(),
            y_buf,
            x_sorted,
            token_expert_slots,
            tile_expert_ids,
            expert_offsets,
            expert_indices,
            expert_weights,
            all_B,
            all_C,
            all_A,
            ctx.num_experts,
            ctx.top_k,
            ctx.team_handle,
            ctx.fwd_tile_m,
        )
        tvm_ffi.moe_pop_fwd()

        # Argument order matches forward: X, expert_indices, expert_weights,
        # all_B, all_C, all_A, num_experts, top_k, pg.
        return dX, None, dW, dB, dC, dA, None, None, None


def moe_fused(
    X: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    all_B: torch.Tensor,
    all_C: torch.Tensor,
    all_A: torch.Tensor,
    num_experts: int,
    top_k: int,
    pg: Optional["ProcessGroup"] = None,
) -> torch.Tensor:
    """Fused expert-parallel MoE forward with autograd support.

    ``pg`` is the expert-parallel process group whose ranks hold the remote
    experts. The default ``pg=None`` uses ``NVSHMEM_TEAM_WORLD``. Before passing
    a proper subgroup, every NVSHMEM PE must collectively prepare it during
    distributed setup with ``liger_cute_kernels.nvshmem.resolve_team(pg)``.
    Forward only reads that cache and never starts a process-group collective.

    Returns the combined output ``Y`` of shape ``X.shape``. Under
    ``torch.no_grad()`` (or when no input requires grad) the symmetric memory
    used by fwd is popped immediately before returning; otherwise the
    intermediates stay alive on the autograd context until the matching
    ``backward`` consumes them and pops the stack there.

    The no-grad path accepts gate/up views with contiguous inner dimensions and
    a larger stride between experts, such as slices of packed ``w13`` storage.
    The grad path currently requires contiguous gate/up tensors because the
    backward kernel does not yet support an expert stride.
    """
    # Decide which path to take BEFORE entering the Function:
    # torch.is_grad_enabled() is forced to False inside
    # autograd.Function.forward, so the check has to live here.
    needs_grad = torch.is_grad_enabled() and any(t.requires_grad for t in (X, expert_weights, all_B, all_C, all_A))

    if not needs_grad:
        # Inference / torch.no_grad(): there is no backward to consume the fwd's
        # symmetric buffers, so pop them right away to return the symmetric stack
        # to its pre-fwd depth (rather than leaking until a backward that never
        # runs).
        team_handle = _resolve_team(pg)
        Y, _x_sorted, _y_buf, _all_eo, _tes, _tei, _tile_m = tvm_ffi.moe_fused_fwd_bf16(
            X,
            expert_indices,
            expert_weights,
            all_B,
            all_C,
            all_A,
            num_experts,
            top_k,
            team_handle,
        )
        tvm_ffi.moe_pop_fwd()
        return Y

    return LigerExpertParallelFusedMoEFunction.apply(
        X,
        expert_indices,
        expert_weights,
        all_B,
        all_C,
        all_A,
        num_experts,
        top_k,
        pg,
    )
