"""Ascend Triton backend registration for ``swiglu``."""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import register_op
from liger_kernel.ops.backends._ascend.dispatch_common import ASCEND_TRITON_CAPABILITY
from liger_kernel.ops.backends._ascend.dispatch_common import ASCEND_TRITON_RANK
from liger_kernel.ops.backends._ascend.dispatch_common import DEFAULT_TOLERANCES
from liger_kernel.ops.backends._ascend.ops.swiglu import LigerSiLUMulFunction


@register_op(
    "swiglu",
    impl_name="ascend-triton",
    capability=ASCEND_TRITON_CAPABILITY,
    modes=("default",),
    default_mode="default",
    preference_rank=ASCEND_TRITON_RANK,
    tolerances=DEFAULT_TOLERANCES,
    notes="NPU-tuned SwiGLU. Preferred over nvidia-triton on Ascend.",
)
def swiglu_ascend(
    a: torch.Tensor,
    b: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    if mode not in (None, "default"):
        raise ValueError(f"Ascend swiglu has only mode='default'; got mode={mode!r}.")
    return LigerSiLUMulFunction.apply(a, b, float(gate_multiplier), float(down_multiplier))
