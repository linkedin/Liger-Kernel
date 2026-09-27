"""Ascend Triton backend registration for ``fused_linear_cross_entropy``."""

from __future__ import annotations

from typing import Optional
from typing import Tuple

import torch

from liger_kernel.backends import register_op
from liger_kernel.ops.backends._ascend.dispatch_common import ASCEND_TRITON_CAPABILITY
from liger_kernel.ops.backends._ascend.dispatch_common import ASCEND_TRITON_RANK
from liger_kernel.ops.backends._ascend.dispatch_common import DEFAULT_TOLERANCES
from liger_kernel.ops.backends._ascend.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction


@register_op(
    "fused_linear_cross_entropy",
    impl_name="ascend-triton",
    capability=ASCEND_TRITON_CAPABILITY,
    modes=("default",),
    default_mode="default",
    preference_rank=ASCEND_TRITON_RANK,
    tolerances=DEFAULT_TOLERANCES,
    notes="NPU-tuned fused linear + CE. Preferred over nvidia-triton on Ascend.",
)
def fused_linear_cross_entropy_ascend(
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
    if mode not in (None, "default"):
        raise ValueError(f"Ascend fused_linear_cross_entropy has only mode='default'; got mode={mode!r}.")
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
    )
