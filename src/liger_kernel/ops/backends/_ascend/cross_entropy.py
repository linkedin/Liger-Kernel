"""Ascend Triton backend registration for ``cross_entropy``.

``LigerCrossEntropyLoss`` and ``functional.cross_entropy`` dispatch through
:func:`liger_kernel.backends.dispatch`. Without this adapter, auto-select only
sees ``nvidia-triton`` (gradient-in-forward, ``BLOCK_SIZE=2048``) which is
~3–4× slower than CANN ``npu_cross_entropy_loss`` on 910B.
"""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import register_op
from liger_kernel.ops.backends._ascend.dispatch_common import ASCEND_TRITON_CAPABILITY
from liger_kernel.ops.backends._ascend.dispatch_common import ASCEND_TRITON_RANK
from liger_kernel.ops.backends._ascend.dispatch_common import DEFAULT_TOLERANCES
from liger_kernel.ops.backends._ascend.ops.cross_entropy import LigerCrossEntropyFunction


@register_op(
    "cross_entropy",
    impl_name="ascend-triton",
    capability=ASCEND_TRITON_CAPABILITY,
    modes=("default",),
    default_mode="default",
    preference_rank=ASCEND_TRITON_RANK,
    tolerances=DEFAULT_TOLERANCES,
    notes="NPU-tuned CE (48-core persistent tiles, split fwd/bwd). Preferred on Ascend.",
)
def cross_entropy_ascend(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
    *,
    mode: Optional[str] = None,
):
    if mode not in (None, "default"):
        raise ValueError(f"Ascend cross_entropy has only mode='default'; got mode={mode!r}.")
    return LigerCrossEntropyFunction.apply(
        input,
        target,
        weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
        return_token_accuracy,
        return_predicted_tokens,
    )
