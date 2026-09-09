"""Triton backend registration for ``rms_norm``.

Thin adapter: the kernels and ``LigerRMSNormFunction`` live in
``liger_kernel.ops.rms_norm``. This file only wraps them in a
:func:`register_op`-decorated callable that the multi-DSL dispatcher can find.

Capability: requires the ``triton`` package. No compute-capability gate — the
existing kernels target sm_80 through sm_100.
"""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.ops.rms_norm import LigerRMSNormFunction

_TRITON_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


@register_op(
    "rms_norm",
    impl_name="nvidia-triton",
    capability=Capability(modules=["triton"]),
    modes=("default",),
    default_mode="default",
    preference_rank=50,
    tolerances=_TRITON_TOLERANCES,
    notes="Liger's original Triton RMSNorm kernel; default cross-arch fallback.",
)
def rms_norm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    offset: float = 0.0,
    casting_mode: str = "llama",
    in_place: bool = False,
    row_mode: Optional[bool] = None,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """Triton RMSNorm. ``mode`` is accepted for API parity and must be one of
    ``None``/``"default"``; anything else is rejected with a clear error.
    """
    if mode not in (None, "default"):
        raise ValueError(
            f"Triton rms_norm has only mode='default'; got mode={mode!r}. "
            f"Pass backend='cutile' to use mode='static_persistent' or 'multi_wave_cached'."
        )
    return LigerRMSNormFunction.apply(x, weight, eps, offset, casting_mode, in_place, row_mode)
