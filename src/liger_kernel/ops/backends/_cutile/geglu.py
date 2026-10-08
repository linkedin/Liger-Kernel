"""cuTile backend registration for ``geglu``."""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import register_op
from liger_kernel.ops.backends._cutile._common import CUTILE_CAPABILITY
from liger_kernel.ops.backends._cutile._common import CUTILE_TOLERANCES
from liger_kernel.ops.backends._cutile._common import validate_default_mode
from liger_kernel.ops.cutile.ops.geglu import LigerGELUMulFunction


@register_op(
    "geglu",
    impl_name="nvidia-cutile",
    capability=CUTILE_CAPABILITY,
    modes=("default",),
    default_mode="default",
    preference_rank=80,
    tolerances=CUTILE_TOLERANCES,
    notes="OSS cuTile GeGLU for Blackwell; explicit or last-fallback selection.",
)
def geglu_cutile(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    validate_default_mode("geglu", mode)
    return LigerGELUMulFunction.apply(a, b)
