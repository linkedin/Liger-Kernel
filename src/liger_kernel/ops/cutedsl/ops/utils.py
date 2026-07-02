"""Shared helpers for the CuteDSL backend operators.

Small utilities factored out of the individual CuteDSL operator modules (``rope``,
``swiglu``, ...) so a new op does not have to re-derive them. This module only
depends on ``torch`` and ``cutlass``'s ``from_dlpack`` -- both hard dependencies
of any op that imports it -- so it introduces no extra optional requirements.
"""

import cutlass
import torch

from cutlass.cute.runtime import from_dlpack
from cutlass.cute.runtime import make_fake_tensor as _cute_make_fake_tensor

# log2(e). Lets a kernel compute ``exp(x)`` / ``sigmoid(x)`` via the hardware
# ``exp2`` SFU op: ``exp(x) == exp2(x * _LOG2E)``.
_LOG2E = 1.4426950408889634

# dtypes every CuteDSL op currently supports.
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# torch dtype -> CuteDSL numeric type. Used by ops that compile against an
# abstract (fake) tensor for the TVM-FFI fast path.
torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def make_fake_tensor(dtype, shape, divisibility: int = 1):
    """Build an abstract (fake) CuteDSL tensor for ahead-of-time ``cute.compile``.

    Thin wrapper over ``cutlass.cute.runtime.make_fake_tensor`` that mirrors the
    signature the CuteDSL ops were written against: a trailing ``divisibility``
    (in elements) instead of an explicit stride tuple. A contiguous, row-major
    stride is derived from ``shape`` (dynamic ``SymInt`` leading dims are fine
    because only the *inner*, static extents feed a stride), and ``divisibility``
    is translated into a byte ``assumed_align`` so the compiler may emit
    vectorized (e.g. 128-bit) accesses.
    """
    stride = []
    acc = 1
    for extent in reversed(shape):
        stride.append(acc)
        # Only static (int) extents accumulate into an inner stride; once a
        # dynamic dim is hit the remaining (outer) strides stay symbolic-free
        # because callers only ever place the dynamic dim outermost.
        acc = acc * extent if isinstance(extent, int) else acc
    stride = tuple(reversed(stride))

    assumed_align = None
    width_bits = getattr(dtype, "width", None)
    if width_bits:
        assumed_align = max(1, divisibility * (width_bits // 8))

    return _cute_make_fake_tensor(dtype, tuple(shape), stride, assumed_align=assumed_align)


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


def _vec_for_dtype(dtype: torch.dtype) -> int:
    """Number of elements in a 128-bit vectorized access for ``dtype``."""
    return max(1, 128 // (torch.finfo(dtype).bits))


def _validate_supported_dtype(dtype: torch.dtype, op_name: str = "cutedsl"):
    if dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{op_name} supports only {_SUPPORTED_DTYPES}; got {dtype}.")
