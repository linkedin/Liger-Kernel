"""CuTe DSL backend registration for ``fused_linear_jsd``.

Thin adapter: the composed-op logic and ``LigerFusedLinearJSDFunction`` live
in :mod:`liger_kernel.ops.fused_linear_jsd`.  This file only wraps them in a
:func:`register_op`-decorated callable that the multi-DSL dispatcher can find.

The composed op computes logits and log-softmax in fp32; the CuTe DSL JSD
primitive (b02ab91) handles fp32 natively via trace-time precise exp/log, so
the whole composed op now runs on the CuTe DSL inner primitive.

Capability: requires the ``cutlass.cute`` package and compute capability >=
sm_90 (Hopper or newer), matching the ``jsd_loss_and_grad`` CuTeDSL primitive.
"""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction

_CUTEDSL_FLJSD_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


@register_op(
    "fused_linear_jsd",
    impl_name="nvidia-cutedsl",
    capability=Capability(
        min_cc=(9, 0),
        modules=["cutlass.cute", "torch"],
    ),
    modes=("default",),
    default_mode="default",
    # Auto-select: fp32 inner JSD runs CuTe DSL natively (b02ab91), so the
    # composed op beats cuTile (rank 5) by ~2-8% at V>=65536 and ties it at
    # V=32000 on B200 (A/B flake-level at 32000, 1.02-1.08x at wide vocab);
    # it also never loses to Triton (rank 50). Rank 4 sits below both so the
    # dispatcher picks CuTe DSL on sm_90+; pre-Hopper falls to Triton via the
    # min_cc gate.
    preference_rank=4,
    tolerances=_CUTEDSL_FLJSD_TOLERANCES,
    notes="CuTe DSL FLJSD (fp32 inner JSD via precise exp/log); auto-selected on sm_90+, beats cuTile/Triton at wide vocab.",
)
def fused_linear_jsd_cutedsl(
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
    """CuTe DSL fused_linear_jsd dispatch entry point.

    Delegates to ``LigerFusedLinearJSDFunction`` with the inner
    ``jsd_loss_and_grad`` primitive pinned to the CuTe DSL impl.

    ``mode`` is accepted only for API parity with the other backends; the only
    valid value is ``"default"`` (or ``None``).
    """
    if mode not in (None, "default"):
        raise ValueError(f"CuTe DSL fused_linear_jsd has only mode='default'; got mode={mode!r}.")
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
        "nvidia-cutedsl",
        mode,
    )
