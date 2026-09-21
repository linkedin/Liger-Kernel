"""cuTile (NVIDIA cuda.tile) backend for ``jsd``.

This file ports NVIDIA's cuTile JSD kernel from
`linkedin/Liger-Kernel#1228 <https://github.com/linkedin/Liger-Kernel/pull/1228>`_
(SPDX-License-Identifier: MIT — NVIDIA CORPORATION & AFFILIATES, 2025) and
wraps it in PR #4's per-(op, impl) registry so it coexists with the Triton
implementation under ``Capability``-based gating.

Two ops registered here:

1. ``jsd`` (autograd-aware) — :class:`_LigerJSDCuTileFunction` matches the
   API surface of :class:`liger_kernel.ops.jsd.LigerJSDFunction`.

2. ``jsd_loss_and_grad`` (non-autograd primitive) — returns per-row loss + dx
   directly, used by ``fused_linear_jsd`` so it picks up cuTile when the
   user sets ``impl="nvidia-cutile"``.

Capability: requires ``cuda.tile`` runtime + ``tileiras`` compiler reachable
from the current process. We probe both at registration time so partial
installs (runtime present, compiler missing) gate out gracefully.

References
----------
- NVIDIA TileGym Liger suite — ``src/tilegym/suites/liger/cutile/jsd.py``
- In-repo reproduction on f2e7013f B200 sm_100 (torch 2.12, tileiras v13.2):
  **8.12× forward vs Triton** at llama_3_8b shape (BT=4096, V=128256, bf16).
"""

# NOTE: same anti-PEP-563 caveat as the cuTile rms_norm / layer_norm siblings.
# ``@ct.kernel`` introspects ``__annotations__`` to discover ``Constant``
# parameters; ``from __future__ import annotations`` stringifies them and
# silently breaks specialization across launches.

import math

from typing import Optional
from typing import Tuple

import cuda.tile as ct
import torch

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.ops._nvidia_shared import cutile_compiler_available as _cutile_compiler_available
from liger_kernel.ops._nvidia_shared import next_pow2 as _next_pow2
from liger_kernel.ops._nvidia_shared import to_local_if_dtensor as _to_local_if_dtensor
from liger_kernel.ops.utils import ensure_contiguous

_ConstFloat = ct.Constant[float]
_ConstInt = ct.Constant[int]
_JSD_BLOCK_SIZE = 4096


@ct.kernel(occupancy=ct.ByTarget(sm_100=4))
def _jsd_kernel_ct(
    x,  # (BT, V) log Q (student)
    y,  # (BT, V) log P (teacher)
    loss,  # (BT, V) float32 loss accumulator
    dx,  # (BT, V) gradient output
    label,  # (BT,) label tensor, or dummy when HAS_LABEL=0
    beta: _ConstFloat,
    inv_n_non_ignore,  # runtime fp32 scalar tensor
    ignore_index: _ConstInt,
    n_cols: _ConstInt,
    BLOCK_SIZE: _ConstInt,
    HAS_LABEL: _ConstInt,
):
    """One row per program. Fused loss + dx; fp32 inside, dtype-cast on store."""
    row_idx = ct.bid(0)
    scale = ct.load(inv_n_non_ignore, 0, shape=())

    if HAS_LABEL:
        lbl = ct.load(label, row_idx, shape=())
        if lbl == ignore_index:
            # Zero this row's gradient and return — no loss contribution.
            num_chunks_early = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
            for ci in range(num_chunks_early):
                col_indices = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
                ct.scatter(
                    dx,
                    (row_idx, col_indices),
                    ct.full((BLOCK_SIZE,), 0.0, dtype=dx.dtype),
                    check_bounds=True,
                )
            return

    num_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    for chunk_idx in range(num_chunks):
        col_indices = ct.arange(BLOCK_SIZE, dtype=ct.int32) + chunk_idx * BLOCK_SIZE

        x_tile = ct.gather(x, (row_idx, col_indices), check_bounds=True, padding_value=-math.inf)
        y_tile = ct.gather(y, (row_idx, col_indices), check_bounds=True, padding_value=-math.inf)

        x_f32 = ct.astype(x_tile, ct.float32)
        y_f32 = ct.astype(y_tile, ct.float32)

        loss_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
        dx_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

        if beta == 0.0:
            # Forward KL: P || M, M -> P
            y_max = ct.max(y_f32, 0, keepdims=True)
            y_prob = ct.exp(y_f32 - y_max) * ct.exp(y_max)
            loss_tile = y_prob * (y_f32 - x_f32)
            dx_tile = -y_prob
        elif beta == 1.0:
            # Reverse KL: Q || M, M -> Q
            x_max = ct.max(x_f32, 0, keepdims=True)
            x_prob = ct.exp(x_f32 - x_max) * ct.exp(x_max)
            loss_tile = x_prob * (x_f32 - y_f32)
            dx_tile = loss_tile + x_prob
        else:
            # Generalized JSD with mixing coefficient beta.
            x_max = ct.max(x_f32, 0, keepdims=True)
            y_max = ct.max(y_f32, 0, keepdims=True)
            max_val = ct.maximum(x_max, y_max)
            exp_max = ct.exp(max_val)
            q_prob = ct.exp(x_f32 - max_val) * exp_max
            p_prob = ct.exp(y_f32 - max_val) * exp_max
            beta_p = beta * p_prob
            one_minus_beta_q = (1.0 - beta) * q_prob
            m_prob = beta_p + one_minus_beta_q
            log_m = ct.log(m_prob)
            loss_tile = beta_p * y_f32 + one_minus_beta_q * x_f32 - m_prob * log_m
            dx_tile = one_minus_beta_q * (x_f32 - log_m)

        loss_tile = loss_tile * scale
        dx_tile = dx_tile * scale

        ct.scatter(loss, (row_idx, col_indices), loss_tile, check_bounds=True)
        ct.scatter(dx, (row_idx, col_indices), ct.astype(dx_tile, dx.dtype), check_bounds=True)


def _launch_jsd_kernel(
    student_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    shift_labels: Optional[torch.Tensor],
    beta: float,
    ignore_index: int,
    n_non_ignore: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate ``loss`` + ``dx`` and launch :func:`_jsd_kernel_ct`.

    Returns ``(loss, dx)`` with shapes ``(BT, V)`` (fp32) and ``(BT, V)`` (input dtype).
    """
    BT, V = student_prob.shape
    BLOCK_SIZE = min(_JSD_BLOCK_SIZE, _next_pow2(V))
    has_label = shift_labels is not None

    loss = torch.zeros((BT, V), dtype=torch.float32, device=student_prob.device)
    dx = torch.empty_like(student_prob)

    inv_n_non_ignore = torch.tensor(
        [0.0 if n_non_ignore == 0 else (1.0 / n_non_ignore)],
        dtype=torch.float32,
        device=student_prob.device,
    )
    label_arg = shift_labels if has_label else torch.empty(1, dtype=torch.int64, device=student_prob.device)

    # Device-safe launch: dispatch selects this backend from the input tensor's
    # device, so tie the launch (and stream) to that device rather than the
    # process-current one. Mirrors ops/cutile/ops/fused_linear_cross_entropy.py.
    device = student_prob.device
    with torch.cuda.device(device):
        ct.launch(
            torch.cuda.current_stream(device),
            (BT, 1, 1),
            _jsd_kernel_ct,
            (
                student_prob,
                teacher_prob,
                loss,
                dx,
                label_arg,
                float(beta),
                inv_n_non_ignore,
                int(ignore_index),
                int(V),
                int(BLOCK_SIZE),
                int(has_label),
            ),
        )
    return loss, dx


# ---------------------------------------------------------------------------
# Autograd-aware wrapper — matches LigerJSDFunction's API.
# ---------------------------------------------------------------------------


class _LigerJSDCuTileFunction(torch.autograd.Function):
    """cuTile autograd wrapper for generalized Jensen-Shannon Divergence."""

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        _input: torch.Tensor,
        target: torch.Tensor,
        shift_labels: Optional[torch.Tensor],
        beta: float,
        ignore_index: int,
    ) -> torch.Tensor:
        _input = _to_local_if_dtensor(_input)
        target = _to_local_if_dtensor(target)
        if shift_labels is not None:
            shift_labels = _to_local_if_dtensor(shift_labels)

        has_label = False
        if shift_labels is not None:
            assert shift_labels.shape == (_input.shape[0],), (
                f"shift_labels must have shape (BT,). Got: {shift_labels.shape}"
            )
            shift_labels = shift_labels.contiguous()
            has_label = True

        BT = _input.shape[0]
        if has_label:
            n_non_ignore = int((shift_labels != ignore_index).sum().item())
        else:
            n_non_ignore = BT

        if n_non_ignore == 0:
            ctx.save_for_backward(torch.zeros_like(_input))
            return torch.tensor(0.0, device=_input.device, dtype=_input.dtype)

        loss, dx = _launch_jsd_kernel(_input, target, shift_labels, beta, ignore_index, float(n_non_ignore))
        ctx.save_for_backward(dx)
        return torch.sum(loss).to(_input.dtype)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor):
        (dx,) = ctx.saved_tensors
        # If grad_output is the scalar 1.0 (typical when JSD is the last loss),
        # skip the elementwise multiply entirely.
        if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
            dx = grad_output * dx
        return (dx, None, None, None, None)


_CUTILE_JSD_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-3, "rtol_fwd": 1e-3, "rtol_bwd": 1e-3},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 2e-2, "rtol_fwd": 1e-2, "rtol_bwd": 1e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-5, "rtol_fwd": 1e-5, "rtol_bwd": 1e-5},
}


@register_op(
    "jsd",
    impl_name="nvidia-cutile",
    capability=Capability(
        min_cc=(10, 0),
        modules=["cuda.tile", "torch"],
        predicate=_cutile_compiler_available,
    ),
    modes=("default",),
    default_mode="default",
    # Ranked ABOVE cutedsl JSD (rank 20 on sm_100 / 2 on sm_103) so cutedsl stays the
    # auto-default where it is faster. Measured B200 (fwd+bwd, V=128256, rel-vs-triton=0):
    # cutedsl 3.04ms < cuTile 4.09ms < Triton 6.06ms @ BT=8192 (cuTile ~1.5x vs Triton but
    # ~35% slower than cutedsl). cuTile stays below Triton (50) so it still wins as a fallback
    # when cutedsl is unavailable, and remains selectable via explicit impl="nvidia-cutile".
    preference_rank=25,
    tolerances=_CUTILE_JSD_TOLERANCES,
    notes=(
        "cuTile JSD ported from NVIDIA TileGym + PR #1228 (MIT). B200 fwd+bwd ~1.5x vs Triton "
        "but slower than cutedsl JSD, so ranked below cutedsl (opt-in / fallback above Triton)."
    ),
)
def jsd_cutile(
    _input: torch.Tensor,
    target: torch.Tensor,
    shift_labels: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    ignore_index: int = -100,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    if mode not in (None, "default"):
        raise ValueError(f"cuTile jsd has only mode='default'; got {mode!r}.")
    return _LigerJSDCuTileFunction.apply(_input, target, shift_labels, beta, ignore_index)


@register_op(
    "jsd_loss_and_grad",
    impl_name="nvidia-cutile",
    capability=Capability(
        min_cc=(10, 0),
        modules=["cuda.tile", "torch"],
        predicate=_cutile_compiler_available,
    ),
    modes=("default",),
    default_mode="default",
    # Inner primitive the fused-linear JSD training path composes. Ranked above cutedsl's
    # jsd_loss_and_grad (rank 20 on sm_100 / 2 on sm_103) so the faster cutedsl inner kernel
    # stays the auto-default there (cutedsl > cuTile on B200; see _cutile/jsd.py notes).
    preference_rank=25,
    notes="cuTile per-chunk JSD primitive for fused_linear_jsd (opt-in; cutedsl inner stays default).",
)
def jsd_loss_and_grad_cutile(
    student_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    shift_labels: Optional[torch.Tensor],
    beta: float,
    ignore_index: int,
    n_non_ignore: float,
    *,
    mode: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-chunk JSD primitive: returns ``(loss, dx)`` with shapes ``(BT, V)``.

    Unlike the Triton sibling, ``dx`` is a **freshly allocated tensor** (not
    an in-place overwrite of ``student_prob``). Callers must rebind:
    ``student_prob = dx`` if they want the old in-place semantics.
    """
    if mode not in (None, "default"):
        raise ValueError(f"jsd_loss_and_grad_cutile: only mode='default'; got {mode!r}")
    return _launch_jsd_kernel(student_prob, teacher_prob, shift_labels, beta, ignore_index, n_non_ignore)
