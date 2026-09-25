"""Triton backend registration for ``kl_div``.

Two ops registered here:

1. ``kl_div`` — autograd-aware wrapper. Defers to the existing
   :class:`liger_kernel.ops.kl_div.LigerKLDivLossFunction`.

2. ``kl_loss_and_grad`` — non-autograd primitive wrapping the *fused-linear*
   KL kernel (temperature scaling + log-softmax fused over raw logits), from
   :mod:`liger_kernel.ops.fused_linear_kl_div`. Used by the composed op
   :class:`liger_kernel.ops.fused_linear_kl_div.LigerFusedLinearKLDivFunction`
   so it routes through the dispatcher the same way ``fused_linear_jsd`` does
   for ``jsd_loss_and_grad`` — composed ops that launch a backend directly
   would silently bypass impl selection. Future KL backends (e.g. CuTe DSL)
   register under the same primitive name.

Capability: requires the ``triton`` package. No compute-capability gate — the
existing kernels target sm_80 through sm_100.
"""

from __future__ import annotations

from typing import Optional
from typing import Tuple

import torch
import triton

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.ops.fused_linear_kl_div import MAX_FUSED_SIZE
from liger_kernel.ops.fused_linear_kl_div import _kl_div_kernel
from liger_kernel.ops.fused_linear_kl_div import get_num_warps
from liger_kernel.ops.kl_div import LigerKLDivLossFunction

_TRITON_KLDIV_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


@register_op(
    "kl_div",
    impl_name="nvidia-triton",
    capability=Capability(modules=["triton"]),
    modes=("default",),
    default_mode="default",
    preference_rank=50,
    tolerances=_TRITON_KLDIV_TOLERANCES,
    notes="Liger's original Triton KL-divergence loss kernel; default cross-arch fallback.",
)
def kl_div_triton(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduction: str = "batchmean",
    log_target: bool = False,
    eps: float = 1e-10,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """Triton KL-divergence loss. ``mode`` is accepted for API parity and must
    be one of ``None``/``"default"``; anything else is rejected with a clear
    error.
    """
    if mode not in (None, "default"):
        raise ValueError(
            f"Triton kl_div has only mode='default'; got mode={mode!r}. "
            f"Pass impl='nvidia-cutedsl' to use the CuTe DSL variant."
        )
    return LigerKLDivLossFunction.apply(y_pred, y_true, reduction, log_target, eps)


@register_op(
    "kl_loss_and_grad",
    impl_name="nvidia-triton",
    capability=Capability(modules=["triton"]),
    modes=("default",),
    default_mode="default",
    preference_rank=50,
    notes=(
        "Per-chunk fused KL primitive (per-row loss + dx written in place into "
        "logits). Used by fused_linear_kl_div so the composed op routes through "
        "the dispatcher."
    ),
)
def kl_loss_and_grad_triton(
    logits_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
    shift_labels: Optional[torch.Tensor],
    ignore_index: int,
    temperature: float,
    eps: float,
    scale: float,
    *,
    mode: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row loss + dL/dlogits for one chunk of the fused-linear KL.

    Triton path: writes dx **in-place into** ``logits_chunk`` (no extra
    allocation). Callers must rebind the returned ``dx`` — other impls may
    return a freshly allocated tensor instead.

    Args:
        logits_chunk: ``(chunk_BT, V)`` student logits in native precision
            (the GEMM output). Overwritten with dL/dlogits.
        target_chunk: ``(chunk_BT, V)`` target probabilities.
        shift_labels: optional ``(chunk_BT,)`` mask; rows whose label equals
            ``ignore_index`` contribute zero loss and zero gradient.
        ignore_index: label value to ignore.
        temperature: temperature applied to the logits before log-softmax.
        eps: clamp for the target before the log (0 * log 0 = 0).
        scale: pre-computed reduction scale (caller's job), fused into both
            the loss and the gradients.

    Returns:
        ``(loss_rows, dx)`` where ``loss_rows.shape == (chunk_BT,)`` (fp32,
        already scaled) and ``dx is logits_chunk`` (the in-place write).
    """
    if mode not in (None, "default"):
        raise ValueError(f"kl_loss_and_grad_triton: only mode='default'; got {mode!r}")

    chunk_n_rows, V = logits_chunk.shape
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    has_label = shift_labels is not None
    # Pre-zeroed: rows ignored via ``ignore_index`` return early from the
    # kernel without writing the loss, and the caller sums every row.
    loss_rows = torch.zeros((chunk_n_rows,), dtype=torch.float32, device=logits_chunk.device)
    label_arg = shift_labels if has_label else torch.empty(1, device=logits_chunk.device)

    _kl_div_kernel[(chunk_n_rows,)](
        X_ptr=logits_chunk,
        X_stride=logits_chunk.stride(-2),
        Q_ptr=target_chunk,
        Q_stride=target_chunk.stride(-2),
        loss_ptr=loss_rows,
        loss_stride=loss_rows.stride(0),
        label_ptr=label_arg,  # dummy ptr if no label
        ignore_index=ignore_index,
        n_cols=V,
        temperature=temperature,
        eps=eps,
        scale=scale,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_LABEL=has_label,
        num_warps=get_num_warps(BLOCK_SIZE),
    )
    return loss_rows, logits_chunk
