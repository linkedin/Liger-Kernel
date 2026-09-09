"""Triton backend registration for ``kl_div``.

Thin adapter: the kernels and ``LigerKLDivLossFunction`` live in
``liger_kernel.ops.kl_div``. This file only wraps them in a
:func:`register_op`-decorated callable that the multi-DSL dispatcher can find.

Capability: requires the ``triton`` package. No compute-capability gate — the
existing kernels target sm_80 through sm_100.
"""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.ops.kl_div import LigerKLDivLossFunction

_TRITON_KLDIV_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


@register_op(
    "kl_div",
    impl_name="nvidia-triton",
    capability=Capability(modules=["triton"]),
    modes=("default",),
    default_mode="default",
    preference_rank=50,
    tolerances=_TRITON_KLDIV_TOLERANCES,
    notes="Liger's original Triton KL-divergence loss kernel; default cross-arch fallback.",
)
def kl_div_triton(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduction: str = "batchmean",
    log_target: bool = False,
    eps: float = 1e-10,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """Triton KL-divergence loss. ``mode`` is accepted for API parity and must
    be one of ``None``/``"default"``; anything else is rejected with a clear
    error.
    """
    if mode not in (None, "default"):
        raise ValueError(
            f"Triton kl_div has only mode='default'; got mode={mode!r}. "
            f"Pass impl='nvidia-cutedsl' to use the CuTe DSL variant."
        )
    return LigerKLDivLossFunction.apply(y_pred, y_true, reduction, log_target, eps)
