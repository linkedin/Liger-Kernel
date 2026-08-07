"""
Shared helpers for the CuTe DSL backend ops.
"""

import cuda.bindings.driver as driver
import torch

from cutlass.cute.runtime import from_dlpack


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
