"""Triton backend registration for ``cross_entropy``.

Registers Liger's original Triton cross-entropy kernel under the multi-DSL
dispatcher so it coexists with the CuTe DSL implementation. This is the
universal cross-architecture fallback — no compute-capability gate (the
existing kernels target sm_80 through sm_100).

The actual kernels live in :mod:`liger_kernel.ops.cross_entropy`; this module
only wraps :class:`liger_kernel.ops.cross_entropy.LigerCrossEntropyFunction`
so the dispatcher can route to it via ``impl="nvidia-triton"``. Mirrors the
``_triton/softmax.py`` sibling.
"""

from __future__ import annotations

from typing import Optional
from typing import Tuple

import torch
import triton

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
from liger_kernel.ops.utils import device_context
from liger_kernel.ops.utils import is_hip
from liger_kernel.utils import infer_device

_TRITON_CE_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


@register_op(
    "cross_entropy",
    impl_name="nvidia-triton",
    capability=Capability(modules=["triton"]),
    modes=("default",),
    default_mode="default",
    preference_rank=50,
    tolerances=_TRITON_CE_TOLERANCES,
    notes="Liger's original Triton cross-entropy kernel; default cross-arch fallback.",
)
def cross_entropy_triton(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
    *,
    mode: Optional[str] = None,
):
    """Triton cross-entropy via the existing ``LigerCrossEntropyFunction``.

    ``mode`` is accepted for API parity with the other backends; the only valid
    value is ``"default"`` (or ``None``).
    """
    if mode not in (None, "default"):
        raise ValueError(
            f"Triton cross_entropy has only mode='default'; got mode={mode!r}. "
            f"Pass impl='nvidia-cutedsl' to use the CuTe DSL variant."
        )
    return LigerCrossEntropyFunction.apply(
        input,
        target,
        weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
        return_token_accuracy,
        return_predicted_tokens,
    )


# ---------------------------------------------------------------------------
# Per-chunk CE primitive — used by fused_linear_cross_entropy so the composed
# op routes through the dispatcher and picks up CuTe DSL on Hopper+.
# Mirrors the ``jsd_loss_and_grad`` primitive pattern.
# ---------------------------------------------------------------------------
_MAX_FUSED_SIZE = 2048 if infer_device() == "npu" else 65536 // 2


@register_op(
    "cross_entropy_loss_and_grad",
    impl_name="nvidia-triton",
    capability=Capability(modules=["triton"]),
    modes=("default",),
    default_mode="default",
    preference_rank=50,
    notes=(
        "Per-chunk CE primitive (returns per-row loss + in-place grad logits). "
        "Used by fused_linear_cross_entropy so composed ops route through the dispatcher."
    ),
)
def cross_entropy_loss_and_grad_triton(
    logits_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
    ce_weight: Optional[torch.Tensor],
    ignore_index: int,
    lse_square_scale: float,
    label_smoothing: float,
    reduction: str,
    softcap: Optional[float],
    n_non_ignore: int,
    sum_non_ignore_weight: float,
    weight_sum: float,
    return_z_loss: bool,
    return_token_accuracy: bool,
    return_predicted_tokens: bool,
    has_gradients: bool,
    *,
    mode: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """Compute per-row loss + in-place grad for one chunk of logits.

    Triton path: writes the gradient **in-place into** ``logits_chunk``
    (matching the existing fused_linear_cross_entropy behaviour — no extra
    allocation). Caller is responsible for accumulating the per-row loss.

    Returns:
        ``(loss_1d, z_loss_1d, token_accuracy_1d, predicted_tokens_1d, grad_logits)``
        where ``grad_logits is logits_chunk`` (the in-place write).
    """
    with device_context(logits_chunk.device):
        if mode not in (None, "default"):
            raise ValueError(f"cross_entropy_loss_and_grad_triton: only mode='default'; got {mode!r}")

        n_rows, V = logits_chunk.shape
        BLOCK_SIZE = min(_MAX_FUSED_SIZE, triton.next_power_of_2(V))

        loss_1d = torch.zeros(n_rows, dtype=torch.float32, device=logits_chunk.device)
        z_loss_1d = torch.zeros(n_rows, dtype=logits_chunk.dtype, device=logits_chunk.device) if return_z_loss else None
        token_accuracy_1d = (
            torch.zeros(n_rows, dtype=torch.float32, device=logits_chunk.device) if return_token_accuracy else None
        )
        predicted_tokens_1d = (
            torch.full((n_rows,), -1, dtype=torch.int64, device=logits_chunk.device)
            if return_predicted_tokens
            else None
        )

        liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),
            weight_ptr=ce_weight,
            loss_ptr=loss_1d,
            z_loss_ptr=z_loss_1d,
            loss_stride=loss_1d.stride(-1),
            token_accuracy_ptr=token_accuracy_1d,
            token_accuracy_stride=token_accuracy_1d.stride(-1) if return_token_accuracy else 0,
            predicted_tokens_ptr=predicted_tokens_1d,
            predicted_tokens_stride=predicted_tokens_1d.stride(-1) if return_predicted_tokens else 0,
            n_cols=V,
            n_non_ignore=n_non_ignore,
            sum_non_ignore_weight=sum_non_ignore_weight,
            weight_sum=weight_sum,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            RETURN_Z_LOSS=return_z_loss,
            RETURN_TOKEN_ACCURACY=return_token_accuracy,
            RETURN_PREDICTED_TOKENS=return_predicted_tokens,
            HAS_WEIGHT=ce_weight is not None,
            HAS_SOFTCAPPING=softcap is not None,
            HAS_GRADIENTS=has_gradients,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        return loss_1d, z_loss_1d, token_accuracy_1d, predicted_tokens_1d, logits_chunk
