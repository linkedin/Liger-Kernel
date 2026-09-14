"""Ascend Triton backend registration for ``layer_norm``.

``LigerLayerNorm`` and ``functional.layer_norm`` dispatch through
:func:`liger_kernel.backends.dispatch`. Without this adapter, auto-select only
sees ``nvidia-triton`` (the CUDA-oriented kernel in ``ops.layer_norm``) and
never the UB-aware NPU kernels in ``ops.backends._ascend.ops.layer_norm``.
"""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import register_op
from liger_kernel.ops.backends._ascend.dispatch_common import ASCEND_TRITON_CAPABILITY
from liger_kernel.ops.backends._ascend.dispatch_common import ASCEND_TRITON_RANK
from liger_kernel.ops.backends._ascend.dispatch_common import DEFAULT_TOLERANCES
from liger_kernel.ops.backends._ascend.ops.layer_norm import LigerLayerNormFunction


@register_op(
    "layer_norm",
    impl_name="ascend-triton",
    capability=ASCEND_TRITON_CAPABILITY,
    modes=("default",),
    default_mode="default",
    preference_rank=ASCEND_TRITON_RANK,
    tolerances=DEFAULT_TOLERANCES,
    notes="NPU-tuned LayerNorm (UB-aware fused 2D / row / tiled). Preferred on Ascend.",
)
def layer_norm_ascend(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    if mode not in (None, "default"):
        raise ValueError(f"Ascend layer_norm has only mode='default'; got mode={mode!r}.")
    return LigerLayerNormFunction.apply(x, weight, bias, eps)
