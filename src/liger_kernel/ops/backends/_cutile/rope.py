"""cuTile backend registration for ``rope``."""

from __future__ import annotations

from typing import Optional
from typing import Tuple

import torch

from liger_kernel.backends import register_op
from liger_kernel.ops.backends._cutile._common import CUTILE_CAPABILITY
from liger_kernel.ops.backends._cutile._common import CUTILE_TOLERANCES
from liger_kernel.ops.backends._cutile._common import validate_default_mode
from liger_kernel.ops.cutile.ops.rope import LigerRopeFunction


@register_op(
    "rope",
    impl_name="nvidia-cutile",
    capability=CUTILE_CAPABILITY,
    modes=("default",),
    default_mode="default",
    preference_rank=80,
    tolerances=CUTILE_TOLERANCES,
    notes="OSS cuTile RoPE for Blackwell; explicit or last-fallback selection.",
)
def rope_cutile(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
    *,
    mode: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_default_mode("rope", mode)
    return LigerRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)
