"""
Shared helpers for the CuTe DSL backend ops.
"""

import torch
import cutlass
from cutlass.cute.runtime import from_dlpack
from quack.compile_utils import make_fake_tensor  # noqa: F401

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
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


def to_cute_tensor(t, leading_dim=None, assumed_align=16):
    """torch.Tensor -> cute.Tensor via DLPack, with a dynamic (runtime) layout."""
    if t is None:
        return None
    ct = from_dlpack(t.detach(), assumed_align=assumed_align)
    ld = (t.ndim - 1) if leading_dim is None else leading_dim
    return ct.mark_layout_dynamic(leading_dim=ld)
