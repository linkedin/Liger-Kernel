"""
Shared helpers for the CuTe DSL backend ops.
"""

from typing import Optional

import cuda.bindings.driver as driver
import cutlass.cute as cute
import torch

from cutlass import BFloat16
from cutlass import Float16
from cutlass import Float32
from cutlass import Int32
from cutlass import Int64
from cutlass.cute.runtime import from_dlpack

torch2cute_dtype_map = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


def ensure_cuda_context():
    """Make the CUDA primary context current on the calling thread.

    ``torch.autograd`` runs backward nodes on worker threads.  PyTorch binds
    those threads through the CUDA *runtime* API, which leaves the *driver*
    API's current context unset, so CuTe DSL helpers that call the driver
    directly (notably ``cutlass.utils.HardwareInfo``) fail with
    ``CUDA_ERROR_INVALID_CONTEXT``.  Retaining and setting the device's primary
    context -- the very context PyTorch itself uses -- fixes that and is a no-op
    once a context is already current.
    """
    err, ctx = driver.cuCtxGetCurrent()
    if err == driver.CUresult.CUDA_SUCCESS and ctx and int(ctx) != 0:
        return
    err, device = driver.cuDeviceGet(torch.cuda.current_device())
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGet failed: {err}")
    err, primary = driver.cuDevicePrimaryCtxRetain(device)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDevicePrimaryCtxRetain failed: {err}")
    (err,) = driver.cuCtxSetCurrent(primary)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxSetCurrent failed: {err}")


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


# ---------------------------------------------------------------------------
# Abstract-tensor builder for the TVM-FFI fast path.
#
# This backs the TVM-FFI fast path (``swiglu``), which compiles a kernel once per
# dtype against an abstract tensor so PyTorch tensors can be passed straight
# through, skipping the per-call ``from_dlpack`` / memref-construction host
# overhead.
# ---------------------------------------------------------------------------
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
