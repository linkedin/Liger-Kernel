"""Ascend Triton backend registration for ``rope``."""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import register_op
from liger_kernel.ops.backends._ascend.dispatch_common import ASCEND_TRITON_CAPABILITY
from liger_kernel.ops.backends._ascend.dispatch_common import ASCEND_TRITON_RANK
from liger_kernel.ops.backends._ascend.dispatch_common import DEFAULT_TOLERANCES
from liger_kernel.ops.backends._ascend.ops.rope import LigerRopeFunction


@register_op(
    "rope",
    impl_name="ascend-triton",
    capability=ASCEND_TRITON_CAPABILITY,
    modes=("default",),
    default_mode="default",
    preference_rank=ASCEND_TRITON_RANK,
    tolerances=DEFAULT_TOLERANCES,
    notes="NPU-tuned RoPE. Preferred over nvidia-triton on Ascend.",
)
def rope_ascend(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
    *,
    mode: Optional[str] = None,
):
    if mode not in (None, "default"):
        raise ValueError(f"Ascend rope has only mode='default'; got mode={mode!r}.")
    return LigerRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)
