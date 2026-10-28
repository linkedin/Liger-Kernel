"""Ascend TileLang backend registration for ``swiglu``."""

from typing import Optional

import torch

from liger_kernel.backends import register_op
from liger_kernel.ops.tilelang.ascend.dispatch_common import ASCEND_TILELANG_CAPABILITY
from liger_kernel.ops.tilelang.ascend.dispatch_common import ASCEND_TILELANG_RANK
from liger_kernel.ops.tilelang.ascend.dispatch_common import DEFAULT_TOLERANCES
from liger_kernel.ops.tilelang.ascend.ops.swiglu import LigerSiLUMulFunction


@register_op(
    "swiglu",
    impl_name="ascend-tilelang",
    capability=ASCEND_TILELANG_CAPABILITY,
    modes=("default",),
    default_mode="default",
    preference_rank=ASCEND_TILELANG_RANK,
    tolerances=DEFAULT_TOLERANCES,
    notes="TileLang SIMD SwiGLU forward/backward for Ascend 950 (A5).",
)
def swiglu_ascend_tilelang(
    a: torch.Tensor,
    b: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """Functional dispatcher entry point over the same public Function."""
    if mode not in (None, "default"):
        raise ValueError(f"ascend-tilelang SwiGLU only supports mode='default', got {mode!r}")
    return LigerSiLUMulFunction.apply(a, b, gate_multiplier, down_multiplier)
