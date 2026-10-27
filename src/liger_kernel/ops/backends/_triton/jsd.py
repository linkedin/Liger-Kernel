"""Triton backend registration for ``jsd``.

Two ops registered here:

1. ``jsd`` — autograd-aware wrapper. Defers to the existing
   :class:`liger_kernel.ops.jsd.LigerJSDFunction`.

2. ``jsd_loss_and_grad`` — non-autograd primitive that returns the per-row
   loss tile and per-element ``dx`` directly. Used by composed ops like
   :class:`liger_kernel.ops.fused_linear_jsd.LigerFusedLinearJSDFunction`
   so they pick up the same impl as the standalone JSD when the user sets
   ``impl="nvidia-cutile"``. (Without this, composed ops would silently
   bypass the dispatcher — the same class of bug we hit reproducing
   linkedin/Liger-Kernel#1228.)

Capability: requires the ``triton`` package. No compute-capability gate — the
existing kernels target sm_80 through sm_100.
"""

from __future__ import annotations

from typing import Optional
from typing import Tuple

import torch

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.ops.jsd import LigerJSDFunction
from liger_kernel.ops.jsd import _jsd_kernel

_TRITON_JSD_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-3, "rtol_fwd": 1e-3, "rtol_bwd": 1e-3},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 2e-2, "rtol_fwd": 1e-2, "rtol_bwd": 1e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-5, "rtol_fwd": 1e-5, "rtol_bwd": 1e-5},
}


@register_op(
    "jsd",
    impl_name="nvidia-triton",
    capability=Capability(modules=["triton"]),
    modes=("default",),
    default_mode="default",
    preference_rank=50,
    tolerances=_TRITON_JSD_TOLERANCES,
    notes="Liger's original Triton JSD kernel; default cross-arch fallback.",
)
def jsd_triton(
    _input: torch.Tensor,
    target: torch.Tensor,
    shift_labels: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    ignore_index: int = -100,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """Triton JSD via the existing ``LigerJSDFunction`` autograd wrapper."""
    if mode not in (None, "default"):
        raise ValueError(
            f"Triton jsd has only mode='default'; got mode={mode!r}. "
            f"Pass impl='nvidia-cutile' to access cuTile's kernel variants."
        )
    return LigerJSDFunction.apply(
        _input,
        target,
        shift_labels,
        beta,
        ignore_index,
        "nvidia-triton",
        mode,
    )


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


_JSD_TRITON_BLOCK_SIZE = 4096


@register_op(
    "jsd_loss_and_grad",
    impl_name="nvidia-triton",
    capability=Capability(modules=["triton"]),
    modes=("default",),
    default_mode="default",
    preference_rank=50,
    notes=(
        "Per-chunk JSD primitive (returns per-row loss tile + per-element dx). "
        "Used by fused_linear_jsd so composed ops route through the dispatcher."
    ),
)
def jsd_loss_and_grad_triton(
    student_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    shift_labels: Optional[torch.Tensor],
    beta: float,
    ignore_index: int,
    n_non_ignore: float,
    *,
    mode: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-element loss + dx for one chunk.

    Triton path: writes dx **in-place into** ``student_prob`` (matches the
    existing fused_linear_jsd behaviour — no extra allocation). Caller is
    responsible for ``loss.sum()`` at the outer level.

    Args:
        student_prob: ``(BT, V)`` log Q (student). Will be overwritten with dx.
        teacher_prob: ``(BT, V)`` log P (teacher).
        shift_labels: optional ``(BT,)`` mask; rows where the label equals
            ``ignore_index`` contribute zero loss and zero gradient.
        beta: mixing coefficient in [0, 1].
        ignore_index: label value to ignore.
        n_non_ignore: pre-computed count of non-ignored rows (caller's job).

    Returns:
        ``(loss, dx)`` where ``loss.shape == (BT, V)`` (fp32, summed elsewhere)
        and ``dx is student_prob`` (the in-place write).
    """
    if mode not in (None, "default"):
        raise ValueError(f"jsd_loss_and_grad_triton: only mode='default'; got {mode!r}")

    BT, V = student_prob.shape
    BLOCK_SIZE = min(_JSD_TRITON_BLOCK_SIZE, _next_pow2(V))
    has_label = shift_labels is not None

    # NOTE: this must be a standalone contiguous buffer, *not* a view into a caller-owned
    # loss tensor. `torch.compile` functionalizes the kernel's mutation by cloning the pointer
    # argument via `clone_preserve_strides`, which clones `storage_offset + numel` elements out
    # of a buffer Inductor may have sized to the view alone -- an out-of-bounds read whose
    # garbage survives into the loss on rows where the kernel returns early (ignore_index).
    loss = torch.zeros((BT, V), dtype=torch.float32, device=student_prob.device)
    label_arg = shift_labels if has_label else torch.empty(1, device=student_prob.device)

    _jsd_kernel[(BT,)](
        X_ptr=student_prob,
        X_stride=student_prob.stride(-2),
        Y_ptr=teacher_prob,
        Y_stride=teacher_prob.stride(-2),
        loss_ptr=loss,
        loss_stride=loss.stride(-2),
        dX_ptr=student_prob,  # write in place
        dX_stride=student_prob.stride(-2),
        label_ptr=label_arg,
        beta=beta,
        # Triton kernel expects n_non_ignore as int; helper API takes float
        # for cross-impl portability (cuTile uses 1/n in fp32 internally).
        n_non_ignore=int(round(n_non_ignore)),
        ignore_index=ignore_index,
        n_cols=V,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_LABEL=has_label,
    )
    return loss, student_prob
