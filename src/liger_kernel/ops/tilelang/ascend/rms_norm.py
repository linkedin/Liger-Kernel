"""Ascend TileLang backend registration for ``rms_norm``."""

from typing import Optional

import torch

from liger_kernel.backends import register_op
from liger_kernel.ops.tilelang.ascend.dispatch_common import ASCEND_TILELANG_CAPABILITY
from liger_kernel.ops.tilelang.ascend.dispatch_common import ASCEND_TILELANG_RANK
from liger_kernel.ops.tilelang.ascend.dispatch_common import DEFAULT_TOLERANCES


@register_op(
    "rms_norm",
    impl_name="ascend-tilelang",
    capability=ASCEND_TILELANG_CAPABILITY,
    modes=("default",),
    default_mode="default",
    preference_rank=ASCEND_TILELANG_RANK,
    tolerances=DEFAULT_TOLERANCES,
    notes="TileLang full-width and column-tiled RMSNorm forward/backward for Ascend 950.",
)
def rms_norm_ascend_tilelang(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    offset: float = 0.0,
    casting_mode: str = "llama",
    in_place: bool = False,
    row_mode: Optional[bool] = None,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    if mode not in (None, "default"):
        raise ValueError(f"ascend-tilelang RMSNorm only supports mode='default', got {mode!r}")
    # Import the numerical implementation only after this backend is selected.
    from liger_kernel.ops.tilelang.ascend.ops.rms_norm import LigerRMSNormFunction

    return LigerRMSNormFunction.apply(x, weight, eps, offset, casting_mode, in_place, row_mode)
