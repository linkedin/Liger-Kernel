"""cuTile backend registration for ``fused_add_rms_norm``."""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import register_op
from liger_kernel.ops.backends._cutile._common import CUTILE_CAPABILITY
from liger_kernel.ops.backends._cutile._common import CUTILE_TOLERANCES
from liger_kernel.ops.backends._cutile._common import validate_default_mode
from liger_kernel.ops.cutile.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction


@register_op(
    "fused_add_rms_norm",
    impl_name="nvidia-cutile",
    capability=CUTILE_CAPABILITY,
    modes=("default",),
    default_mode="default",
    preference_rank=80,
    tolerances=CUTILE_TOLERANCES,
    notes="OSS cuTile fused add + RMSNorm for Blackwell; explicit or last-fallback selection.",
)
def fused_add_rms_norm_cutile(
    X: torch.Tensor,
    R: torch.Tensor,
    W: torch.Tensor,
    eps: float = 1e-6,
    offset: float = 0.0,
    casting_mode: str = "llama",
    in_place: bool = False,
    *,
    mode: Optional[str] = None,
):
    validate_default_mode("fused_add_rms_norm", mode)
    return LigerFusedAddRMSNormFunction.apply(X, R, W, eps, offset, casting_mode, in_place)
