"""Triton backend registration for ``layer_norm``.

Thin adapter: the kernels and ``LigerLayerNormFunction`` live in
``liger_kernel.ops.layer_norm``. This file only wraps them in a
:func:`register_op`-decorated callable that the multi-DSL dispatcher can find.

Capability: requires the ``triton`` package. No compute-capability gate — the
existing kernels target sm_80 through sm_100.
"""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.ops.layer_norm import LigerLayerNormFunction

_TRITON_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    # LayerNorm has TWO reductions (mean + variance), so fp32 accumulated
    # error is ~sqrt(N) larger than RMSNorm. atol=5e-4 absorbs N up to ~32K.
    torch.float32: {"atol_fwd": 5e-4, "atol_bwd": 1e-3, "rtol_fwd": 1e-4, "rtol_bwd": 1e-3},
}


@register_op(
    "layer_norm",
    impl_name="nvidia-triton",
    capability=Capability(modules=["triton"]),
    modes=("default",),
    default_mode="default",
    preference_rank=50,
    tolerances=_TRITON_TOLERANCES,
    notes="Liger's original Triton LayerNorm kernel; default cross-arch fallback.",
)
def layer_norm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """Triton LayerNorm. ``mode`` is accepted for API parity; only ``None``
    or ``"default"`` are valid.
    """
    if mode not in (None, "default"):
        raise ValueError(
            f"Triton layer_norm has only mode='default'; got mode={mode!r}. "
            f"Pass backend='cutile' or backend='cutedsl' to use kernel-variant modes."
        )
    return LigerLayerNormFunction.apply(x, weight, bias, eps)
