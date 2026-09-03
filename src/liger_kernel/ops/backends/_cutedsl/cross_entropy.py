"""CuTe DSL backend registration for ``cross_entropy``.

Thin adapter: the kernel and ``LigerCrossEntropyFunction`` live in
:mod:`liger_kernel.ops.cutedsl.ops.cross_entropy` (the CuTe DSL operator
package).  This file only wraps it in a :func:`register_op`-decorated callable
that the multi-DSL dispatcher can find — mirroring the
:mod:`liger_kernel.ops.backends._triton.cross_entropy` sibling.

Capability: requires the ``cutlass.cute`` package and compute capability >=
sm_90 (Hopper or newer), matching the rms_norm / softmax CuTe DSL siblings.

The underlying CuTe DSL cross-entropy kernel uses an online-softmax reduction
adapted from Quack (Apache-2.0) and is numerically identical to the Triton
kernel (:mod:`liger_kernel.ops.cross_entropy`).
"""

from __future__ import annotations

from typing import Optional
from typing import Tuple

import torch

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.backends.dispatch import emit_fallback_warning
from liger_kernel.ops.cutedsl.ops.cross_entropy import LigerCrossEntropyFunction

_CUTEDSL_CE_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


@register_op(
    "cross_entropy",
    impl_name="nvidia-cutedsl",
    capability=Capability(
        min_cc=(9, 0),
        modules=["cutlass.cute", "torch"],
    ),
    modes=("default",),
    default_mode="default",
    # Ranked BELOW Triton (50) for the standalone cross_entropy op: large-vocab CE is
    # HBM-bandwidth-bound, and on Blackwell (B200/B300) the CuTe DSL kernel measured
    # tied-to-slower than Triton (bf16 ~=, fp32 ~-5% @ T8192/V128256). Triton therefore
    # stays the auto-default here; the CuTe DSL path remains available via explicit
    # impl="nvidia-cutedsl". (The rank-10 cross_entropy_loss_and_grad primitive below is
    # unchanged — it is the inner kernel the fused-linear CE path composes.)
    preference_rank=60,
    tolerances=_CUTEDSL_CE_TOLERANCES,
    notes="CuTe DSL cross-entropy for Hopper+ (sm_90+); online-softmax reduction. Opt-in on Blackwell.",
)
def cross_entropy_cutedsl(
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
    """CuTe DSL cross-entropy dispatch entry point.

    ``mode`` is accepted only for API parity with the other backends; the only
    valid value is ``"default"`` (or ``None``).
    """
    if mode not in (None, "default"):
        raise ValueError(f"CuTe DSL cross_entropy has only mode='default'; got mode={mode!r}.")
    vector_width = 16 // input.element_size()
    if input.shape[-1] % vector_width:
        from liger_kernel.ops.backends._triton.cross_entropy import cross_entropy_triton

        emit_fallback_warning(
            "cross_entropy",
            "nvidia-cutedsl",
            "nvidia-triton",
            f"vocab size {input.shape[-1]} is not divisible by vector width {vector_width}",
        )
        return cross_entropy_triton(
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
    # The CuTe DSL forward's 128-bit vectorized loader requires every row to
    # start on a 16-byte boundary (asserted in `_launch_ce_fwd`). A row-padded
    # view (e.g. `base[:, :V]` where `base` has V+1 columns) is contiguous in
    # the inner dim but not row-aligned, so materialize a contiguous copy that
    # the kernel can consume. The Triton path accepts such views via its
    # explicit row stride, so this keeps behavioral parity.
    if input.stride(0) * input.element_size() % 16 != 0:
        input = input.contiguous()
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
@register_op(
    "cross_entropy_loss_and_grad",
    impl_name="nvidia-cutedsl",
    capability=Capability(
        min_cc=(9, 0),
        modules=["cutlass.cute", "torch"],
    ),
    modes=("default",),
    default_mode="default",
    preference_rank=10,
    notes=(
        "Per-chunk CE primitive (returns per-row loss + in-place grad logits). "
        "CuTe DSL online-softmax kernel for Hopper+ (sm_90+). Used by fused_linear_cross_entropy."
    ),
)
def cross_entropy_loss_and_grad_cutedsl(
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
    """Compute per-row loss + in-place grad for one chunk of logits via CuTe DSL.

    Delegates to the CuTe DSL CE kernel (``_launch_ce_fwd``) which uses an
    online-softmax reduction with native HW exp2. Writes the gradient
    **in-place into** ``logits_chunk`` (matching the Triton primitive's contract).

    Returns:
        ``(loss_1d, z_loss_1d, token_accuracy_1d, predicted_tokens_1d, grad_logits)``
        where ``grad_logits is logits_chunk`` (the in-place write).
    """
    if mode not in (None, "default"):
        raise ValueError(f"cross_entropy_loss_and_grad_cutedsl: only mode='default'; got {mode!r}")
    vector_width = 16 // logits_chunk.element_size()
    if logits_chunk.shape[-1] % vector_width:
        from liger_kernel.ops.backends._triton.cross_entropy import cross_entropy_loss_and_grad_triton

        emit_fallback_warning(
            "cross_entropy_loss_and_grad",
            "nvidia-cutedsl",
            "nvidia-triton",
            f"vocab size {logits_chunk.shape[-1]} is not divisible by vector width {vector_width}",
        )
        return cross_entropy_loss_and_grad_triton(
            logits_chunk,
            target_chunk,
            ce_weight,
            ignore_index,
            lse_square_scale,
            label_smoothing,
            reduction,
            softcap,
            n_non_ignore,
            sum_non_ignore_weight,
            weight_sum,
            return_z_loss,
            return_token_accuracy,
            return_predicted_tokens,
            has_gradients,
        )

    from liger_kernel.ops.cutedsl.ops.cross_entropy import _launch_ce_fwd

    n_rows, V = logits_chunk.shape

    loss_1d = torch.zeros(n_rows, dtype=torch.float32, device=logits_chunk.device)
    z_loss_1d = torch.zeros(n_rows, dtype=logits_chunk.dtype, device=logits_chunk.device) if return_z_loss else None
    token_accuracy_1d = (
        torch.zeros(n_rows, dtype=torch.float32, device=logits_chunk.device) if return_token_accuracy else None
    )
    predicted_tokens_1d = (
        torch.full((n_rows,), -1, dtype=torch.int64, device=logits_chunk.device) if return_predicted_tokens else None
    )

    # Normalizers (matching cross_entropy_forward in cutedsl ops).
    if reduction == "mean" and n_non_ignore > 0:
        if ce_weight is not None and sum_non_ignore_weight > 0:
            inv_n_loss = 1.0 / sum_non_ignore_weight
        else:
            inv_n_loss = 1.0 / n_non_ignore
        inv_n_z = 1.0 / n_non_ignore
    else:
        inv_n_loss = 1.0
        inv_n_z = 1.0

    # The CuTe DSL kernel reads weight as fp32; upcast here (exact parity).
    weight_fp32 = None
    if ce_weight is not None:
        weight_fp32 = ce_weight.to(torch.float32)
        if weight_fp32.stride(-1) != 1:
            weight_fp32 = weight_fp32.contiguous()

    _launch_ce_fwd(
        logits_chunk,
        target_chunk,
        loss_1d,
        inv_n_loss,
        ignore_index,
        has_gradients,
        lse_square_scale,
        z_loss_1d,
        return_z_loss,
        softcap,
        label_smoothing=label_smoothing,
        weight=weight_fp32,
        weight_sum=weight_sum,
        return_token_accuracy=return_token_accuracy,
        return_predicted_tokens=return_predicted_tokens,
        token_acc_out=token_accuracy_1d,
        pred_tok_out=predicted_tokens_1d,
        inv_n_z=inv_n_z,
    )

    return loss_1d, z_loss_1d, token_accuracy_1d, predicted_tokens_1d, logits_chunk
