"""cuTile backend registration for ``cross_entropy``."""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import register_op
from liger_kernel.ops.backends._cutile._common import CUTILE_CAPABILITY
from liger_kernel.ops.backends._cutile._common import CUTILE_TOLERANCES
from liger_kernel.ops.backends._cutile._common import validate_default_mode
from liger_kernel.ops.cutile.ops.cross_entropy import LigerCrossEntropyFunction


@register_op(
    "cross_entropy",
    impl_name="nvidia-cutile",
    capability=CUTILE_CAPABILITY,
    modes=("default",),
    default_mode="default",
    preference_rank=80,
    tolerances=CUTILE_TOLERANCES,
    notes="OSS cuTile cross-entropy for Blackwell; explicit or last-fallback selection.",
)
def cross_entropy_cutile(
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
    validate_default_mode("cross_entropy", mode)
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
