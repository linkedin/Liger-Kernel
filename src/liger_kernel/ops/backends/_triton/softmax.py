"""Triton backend registration for ``softmax``.

Registers Liger's original Triton softmax kernel under the multi-DSL
dispatcher so it coexists with the cuTile and CuTe DSL implementations. This is
the universal cross-architecture fallback — no compute-capability gate (the
existing kernels target sm_80 through sm_100).

The actual kernels live in :mod:`liger_kernel.ops.softmax`; this module only
wraps :class:`liger_kernel.ops.softmax.LigerSoftmaxFunction` so the dispatcher
can route to it via ``impl="nvidia-triton"``. Mirrors the ``_triton/jsd.py``
sibling.
"""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.ops.softmax import LigerSoftmaxFunction

_TRITON_SOFTMAX_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


@register_op(
    "softmax",
    impl_name="nvidia-triton",
    capability=Capability(modules=["triton"]),
    modes=("default",),
    default_mode="default",
    preference_rank=50,
    tolerances=_TRITON_SOFTMAX_TOLERANCES,
    notes="Liger's original Triton softmax kernel; default cross-arch fallback.",
)
def softmax_triton(
    x: torch.Tensor,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """Triton softmax via the existing ``LigerSoftmaxFunction`` autograd wrapper."""
    if mode not in (None, "default"):
        raise ValueError(
            f"Triton softmax has only mode='default'; got mode={mode!r}. "
            f"Pass impl='nvidia-cutile' to access cuTile's kernel variants."
        )
    return LigerSoftmaxFunction.apply(x)
