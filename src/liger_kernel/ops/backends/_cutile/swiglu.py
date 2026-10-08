"""cuTile backend registration for ``swiglu``."""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import register_op
from liger_kernel.ops.backends._cutile._common import CUTILE_CAPABILITY
from liger_kernel.ops.backends._cutile._common import CUTILE_TOLERANCES
from liger_kernel.ops.backends._cutile._common import validate_default_mode
from liger_kernel.ops.cutile.ops.swiglu import LigerSiLUMulFunction


@register_op(
    "swiglu",
    impl_name="nvidia-cutile",
    capability=CUTILE_CAPABILITY,
    modes=("default",),
    default_mode="default",
    preference_rank=80,
    tolerances=CUTILE_TOLERANCES,
    notes="OSS cuTile SwiGLU for Blackwell; explicit or last-fallback selection.",
)
def swiglu_cutile(
    a: torch.Tensor,
    b: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    validate_default_mode("swiglu", mode)
    return LigerSiLUMulFunction.apply(a, b, gate_multiplier, down_multiplier)
