"""Triton backend registration for ``geglu``.

Registers Liger's original Triton GeGLU (GELU-mul) kernel under the
multi-DSL dispatcher so it coexists with the CuTe DSL implementation. This is
the universal cross-architecture fallback — no compute-capability gate (the
existing kernels target sm_80 through sm_100).

The actual kernels live in :mod:`liger_kernel.ops.geglu`; this module only
wraps :class:`liger_kernel.ops.geglu.LigerGELUMulFunction` so the dispatcher
can route to it via ``impl="nvidia-triton"``. Mirrors the ``_triton/swiglu.py``
sibling.
"""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.ops.geglu import LigerGELUMulFunction

_TRITON_GELU_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


@register_op(
    "geglu",
    impl_name="nvidia-triton",
    capability=Capability(modules=["triton"]),
    modes=("default",),
    default_mode="default",
    preference_rank=50,
    tolerances=_TRITON_GELU_TOLERANCES,
    notes="Liger's original Triton GeGLU kernel; default cross-arch fallback.",
)
def geglu_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """Triton GeGLU via the existing ``LigerGELUMulFunction`` autograd wrapper.

    ``mode`` is accepted for API parity with the other backends; the only valid
    value is ``"default"`` (or ``None``).
    """
    if mode not in (None, "default"):
        raise ValueError(
            f"Triton geglu has only mode='default'; got mode={mode!r}. "
            f"Pass impl='nvidia-cutedsl' to use the CuTe DSL variant."
        )
    return LigerGELUMulFunction.apply(a, b)
