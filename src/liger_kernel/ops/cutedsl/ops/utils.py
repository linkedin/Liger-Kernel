"""
Shared helpers for the CuTe DSL backend ops.
"""

import torch

from cutlass import BFloat16
from cutlass import Float16
from cutlass import Float32
from cutlass import Int32
from cutlass import Int64
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

torch2cute_dtype_map = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


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


def make_fake_tensor(dtype, shape, divisibility=1, leading_dim=-1):
    """Build an abstract tensor for shape-polymorphic ``cute.compile`` calls."""
    if dtype is None:
        return None
    if leading_dim < 0:
        leading_dim += len(shape)
    stride = tuple(cute.sym_int64(divisibility=divisibility) if dim != leading_dim else 1 for dim in range(len(shape)))
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=stride,
        assumed_align=divisibility * dtype.width // 8,
    )


def to_cute_tensor(t, leading_dim=None, assumed_align=16):
    """torch.Tensor -> cute.Tensor via DLPack, with a dynamic (runtime) layout."""
    if t is None:
        return None
    ct = from_dlpack(t.detach(), assumed_align=assumed_align)
    ld = (t.ndim - 1) if leading_dim is None else leading_dim
    return ct.mark_layout_dynamic(leading_dim=ld)
