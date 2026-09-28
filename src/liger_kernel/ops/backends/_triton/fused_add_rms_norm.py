"""Triton backend registration for ``fused_add_rms_norm``.

Thin adapter: the kernels and ``LigerFusedAddRMSNormFunction`` live in
``liger_kernel.ops.fused_add_rms_norm``. This file only wraps them in a
:func:`register_op`-decorated callable that the multi-DSL dispatcher can find.

Capability: requires the ``triton`` package. No compute-capability gate — the
existing kernels target sm_80 through sm_100.
"""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction

_TRITON_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


@register_op(
    "fused_add_rms_norm",
    impl_name="nvidia-triton",
    capability=Capability(modules=["triton"]),
    modes=("default",),
    default_mode="default",
    preference_rank=50,
    tolerances=_TRITON_TOLERANCES,
    notes="Liger's original Triton fused_add_rms_norm kernel; default cross-arch fallback.",
)
def fused_add_rms_norm_triton(
    X: torch.Tensor,
    R: torch.Tensor,
    W: torch.Tensor,
    eps: float,
    offset: float = 0.0,
    casting_mode: str = "llama",
    in_place: bool = False,
    *,
    mode: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton fused_add_rms_norm. ``mode`` is accepted for API parity and must
    be one of ``None``/``"default"``; anything else is rejected with a clear
    error.
    """
    if mode not in (None, "default"):
        raise ValueError(
            f"Triton fused_add_rms_norm has only mode='default'; got mode={mode!r}. "
            f"Pass impl='nvidia-cutedsl' to use the CuTe DSL variant."
        )
    return LigerFusedAddRMSNormFunction.apply(X, R, W, eps, offset, casting_mode, in_place)
