"""Triton backend registration for ``rope``.

Registers Liger's original Triton RoPE (Rotary Positional Embedding) kernel
under the multi-DSL dispatcher so it coexists with the CuTe DSL implementation.
This is the universal cross-architecture fallback — no compute-capability gate
(the existing kernels target sm_80 through sm_100).

The actual kernels live in :mod:`liger_kernel.ops.rope`; this module only
wraps :class:`liger_kernel.ops.rope.LigerRopeFunction` so the dispatcher can
route to it via ``impl="nvidia-triton"``. Mirrors the ``_triton/softmax.py``
sibling.
"""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.ops.rope import LigerRopeFunction

_TRITON_ROPE_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


@register_op(
    "rope",
    impl_name="nvidia-triton",
    capability=Capability(modules=["triton"]),
    modes=("default",),
    default_mode="default",
    preference_rank=50,
    tolerances=_TRITON_ROPE_TOLERANCES,
    notes="Liger's original Triton RoPE kernel; default cross-arch fallback.",
)
def rope_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
    *,
    mode: Optional[str] = None,
):
    """Triton RoPE via the existing ``LigerRopeFunction`` autograd wrapper.

    ``mode`` is accepted for API parity with the other backends; the only valid
    value is ``"default"`` (or ``None``).
    """
    if mode not in (None, "default"):
        raise ValueError(
            f"Triton rope has only mode='default'; got mode={mode!r}. "
            f"Pass impl='nvidia-cutedsl' to use the CuTe DSL variant."
        )
    return LigerRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)
