"""CuTe DSL backend registration for ``fused_linear_cross_entropy``.

Thin adapter: the composed-op logic and ``LigerFusedLinearCrossEntropyFunction``
live in :mod:`liger_kernel.ops.fused_linear_cross_entropy`. This file only wraps
them in a :func:`register_op`-decorated callable that the multi-DSL dispatcher
can find.

The CuTe DSL acceleration happens **inside** the composed op: the
``LigerFusedLinearCrossEntropyFunction.forward`` method calls
``dispatch("cross_entropy_loss_and_grad", ...)`` for the inner CE computation,
which auto-selects the CuTe DSL :mod:`._cutedsl.cross_entropy` primitive on
Hopper+ (preference_rank=10 < Triton's 50). The matmul and accumulation
portions use cuBLAS / PyTorch primitives, so no additional CuTe DSL kernel is
needed here.

Capability: requires the ``cutlass.cute`` package and compute capability >=
sm_90 (Hopper or newer), matching the ``cross_entropy_loss_and_grad`` CuTeDSL
primitive.
"""

from __future__ import annotations

from typing import Optional
from typing import Tuple

import torch

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

_CUTEDSL_FLCE_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


@register_op(
    "fused_linear_cross_entropy",
    impl_name="nvidia-cutedsl",
    capability=Capability(
        min_cc=(9, 0),
        modules=["cutlass.cute", "torch"],
    ),
    modes=("default",),
    default_mode="default",
    # Auto-preferred when satisfied (lowest rank). The inner
    # cross_entropy_loss_and_grad primitive uses the CuTe DSL online-softmax
    # kernel; matmuls use cuBLAS.
    preference_rank=10,
    tolerances=_CUTEDSL_FLCE_TOLERANCES,
    notes="CuTe DSL fused_linear_cross_entropy for Hopper+ (sm_90+); inner CE via CuTe DSL online-softmax kernel.",
)
def fused_linear_cross_entropy_cutedsl(
    _input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ce_weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    accum_dtype: Optional[torch.dtype] = None,
    use_token_scaling: bool = False,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
    *,
    mode: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """CuTe DSL fused_linear_cross_entropy dispatch entry point.

    Delegates to ``LigerFusedLinearCrossEntropyFunction``, which internally
    dispatches the ``cross_entropy_loss_and_grad`` primitive through the
    dispatcher — picking up the CuTe DSL kernel on Hopper+.

    ``mode`` is accepted only for API parity with the other backends; the only
    valid value is ``"default"`` (or ``None``).
    """
    if mode not in (None, "default"):
        raise ValueError(f"CuTe DSL fused_linear_cross_entropy has only mode='default'; got mode={mode!r}.")
    return LigerFusedLinearCrossEntropyFunction.apply(
        _input,
        weight,
        target,
        bias,
        ce_weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
        accum_dtype,
        use_token_scaling,
        return_token_accuracy,
        return_predicted_tokens,
        "nvidia-cutedsl",
        mode,
    )
