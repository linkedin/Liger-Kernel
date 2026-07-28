"""
Shared helpers for the CuTe DSL backend ops.
"""

import torch

from cutlass.cute.runtime import from_dlpack


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def to_cute_tensor(t, leading_dim=None, assumed_align=16):
    """torch.Tensor -> cute.Tensor via DLPack, with a dynamic (runtime) layout."""
    if t is None:
        return None
    ct = from_dlpack(t.detach(), assumed_align=assumed_align)
    ld = (t.ndim - 1) if leading_dim is None else leading_dim
    return ct.mark_layout_dynamic(leading_dim=ld)


# Process-wide cache of compiled CuteDSL callables, shared across ops.
#
# Sharing one dict across ops is safe *only* because every op namespaces its
# keys: entries carry a trailing string tag (e.g. ``"tok"`` / ``"tma_qk"`` for
# rope, ``"fwd"`` / ``"bwd"`` for swiglu) and/or differ in tuple arity, so keys
# from different ops can never collide. New ops MUST keep their keys namespaced
# the same way.
_COMPILE_CACHE: dict = {}


def _dyn(t: torch.Tensor):
    # ``from_dlpack`` refuses tensors that require grad; the kernels operate on
    # raw storage inside ``autograd.Function`` so detaching is safe. The dynamic
    # layout lets one compiled object serve both contiguous forward tensors and
    # transposed backward views.
    return from_dlpack(t.detach()).mark_layout_dynamic()
