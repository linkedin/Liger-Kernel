"""cuTile backend registration for ``fused_linear_jsd``."""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.ops.backends._cutile.jsd import _cutile_compiler_available
from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction

_CUTILE_FLJSD_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


@register_op(
    "fused_linear_jsd",
    impl_name="nvidia-cutile",
    capability=Capability(
        min_cc=(10, 0),
        modules=["cuda.tile", "torch"],
        predicate=_cutile_compiler_available,
    ),
    modes=("default",),
    default_mode="default",
    # Ranked above cutedsl fused_linear_jsd (rank 4) so cutedsl stays the auto-default; the
    # fused-linear path is GEMM-bound so backends are ~tied, and cuTile composes the cuTile
    # JSD primitive which is slower than cutedsl on B200 (see _cutile/jsd.py). Opt-in / fallback.
    preference_rank=25,
    tolerances=_CUTILE_FLJSD_TOLERANCES,
    notes="cuTile fused_linear_jsd for Blackwell (opt-in; cutedsl stays default). Inner JSD uses the cuTile primitive.",
)
def fused_linear_jsd_cutile(
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
    if mode not in (None, "default"):
        raise ValueError(f"cuTile fused_linear_jsd has only mode='default'; got mode={mode!r}.")
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
        "nvidia-cutile",
        mode,
    )
