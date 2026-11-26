"""Optional Ascend TileLang registration for CrossEntropy."""

from typing import Optional

import torch

from liger_kernel.backends import register_op
from liger_kernel.ops.tilelang.ascend.dispatch_common import ASCEND_TILELANG_CAPABILITY
from liger_kernel.ops.tilelang.ascend.dispatch_common import ASCEND_TILELANG_RANK


@register_op(
    "cross_entropy",
    impl_name="ascend-tilelang",
    capability=ASCEND_TILELANG_CAPABILITY,
    modes=("default",),
    default_mode="default",
    preference_rank=ASCEND_TILELANG_RANK,
    tolerances={
        torch.bfloat16: {"atol_fwd": 1e-8, "rtol_fwd": 5e-2, "atol_bwd": 1e-8, "rtol_bwd": 5e-2},
        torch.float32: {"atol_fwd": 1e-8, "rtol_fwd": 1e-6, "atol_bwd": 1e-8, "rtol_bwd": 1e-6},
    },
    notes="Ascend950 TileLang/PTO CrossEntropy for BF16/FP32 with first-order gradients.",
)
def cross_entropy_ascend_tilelang(
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
        raise ValueError(f"ascend-tilelang CrossEntropy only supports mode='default', got {mode!r}")
    # Keep backend discovery usable when TileLang is absent or not selected.
    from liger_kernel.ops.tilelang.ascend.ops.cross_entropy import _validate_return_flags
    from liger_kernel.ops.tilelang.ascend.ops.cross_entropy import cross_entropy

    _validate_return_flags(return_z_loss, return_token_accuracy, return_predicted_tokens)
    result = cross_entropy(
        input,
        target,
        weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss=return_z_loss,
    )
    loss, z_loss = result if return_z_loss else (result, None)
    return loss, z_loss, None, None
