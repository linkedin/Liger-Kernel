"""
Shared helpers for the CuTe DSL backend ops.
"""

from typing import Optional

import cutlass.cute as cute
import torch

from cutlass import BFloat16
from cutlass import Float16
from cutlass import Float32
from cutlass import Int32
from cutlass import Int64
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


# ---------------------------------------------------------------------------
# torch -> cute dtype mapping and abstract-tensor builder.
#
# These back the TVM-FFI fast path (``swiglu``), which compiles a kernel once per
# dtype against an abstract tensor so PyTorch tensors can be passed straight
# through, skipping the per-call ``from_dlpack`` / memref-construction host
# overhead.
# ---------------------------------------------------------------------------
torch2cute_dtype_map = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


def make_fake_tensor(dtype, shape, divisibility=1, leading_dim=-1) -> Optional[cute.Tensor]:
    """Build an abstract ``cute.Tensor`` for ``cute.compile`` (TVM-FFI fast path).

    Every non-leading stride is symbolic (runtime) with the given ``divisibility``;
    the ``leading_dim`` is contiguous (stride 1). Returns ``None`` for ``dtype=None``.
    """
    if leading_dim < 0:
        leading_dim = len(shape) + leading_dim
    if dtype is None:
        return None
    stride = tuple(cute.sym_int64(divisibility=divisibility) if i != leading_dim else 1 for i in range(len(shape)))
    return cute.runtime.make_fake_tensor(dtype, shape, stride=stride, assumed_align=divisibility * dtype.width // 8)
