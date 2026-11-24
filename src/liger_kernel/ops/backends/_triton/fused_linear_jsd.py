"""Triton backend registration for ``fused_linear_jsd``.

Thin adapter: the kernels and ``LigerFusedLinearJSDFunction`` live in
``liger_kernel.ops.fused_linear_jsd``. This file only wraps them in a
:func:`register_op`-decorated callable that the multi-DSL dispatcher can find.

Capability: requires the ``triton`` package. No compute-capability gate — the
existing kernels target sm_80 through sm_100.
"""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction

_TRITON_FLJSD_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


@register_op(
    "fused_linear_jsd",
    impl_name="nvidia-triton",
    capability=Capability(modules=["triton"]),
    modes=("default",),
    default_mode="default",
    preference_rank=50,
    tolerances=_TRITON_FLJSD_TOLERANCES,
    notes="Liger's original Triton fused_linear_jsd kernel; default cross-arch fallback.",
)
def fused_linear_jsd_triton(
    student_input: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_input: torch.Tensor,
    teacher_weight: torch.Tensor,
    shift_labels: Optional[torch.Tensor] = None,
    jsd_beta: float = 0.5,
    ignore_index: int = -100,
    temperature: float = 1.0,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """Triton fused_linear_jsd. ``mode`` is accepted for API parity and must
    be one of ``None``/``"default"``; anything else is rejected with a clear
    error.
    """
    if mode not in (None, "default"):
        raise ValueError(
            f"Triton fused_linear_jsd has only mode='default'; got mode={mode!r}. "
            f"Pass impl='nvidia-cutedsl' to use the CuTe DSL variant."
        )
    return LigerFusedLinearJSDFunction.apply(
        student_input,
        student_weight,
        teacher_input,
        teacher_weight,
        shift_labels,
        jsd_beta,
        ignore_index,
        temperature,
        None,
        "nvidia-triton",
        mode,
    )
