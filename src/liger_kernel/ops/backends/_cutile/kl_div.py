"""cuTile backend registration for ``kl_div``."""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import register_op
from liger_kernel.ops.backends._cutile._common import CUTILE_CAPABILITY
from liger_kernel.ops.backends._cutile._common import CUTILE_TOLERANCES
from liger_kernel.ops.backends._cutile._common import validate_default_mode
from liger_kernel.ops.cutile.ops.kl_div import LigerKLDivLossFunction


@register_op(
    "kl_div",
    impl_name="nvidia-cutile",
    capability=CUTILE_CAPABILITY,
    modes=("default",),
    default_mode="default",
    preference_rank=80,
    tolerances=CUTILE_TOLERANCES,
    notes="OSS cuTile KL divergence for Blackwell; explicit or last-fallback selection.",
)
def kl_div_cutile(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduction: str = "batchmean",
    log_target: bool = False,
    eps: float = 1e-10,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    validate_default_mode("kl_div", mode)
    return LigerKLDivLossFunction.apply(y_pred, y_true, reduction, log_target, eps)
