"""Shared host and DSL helpers for the SM90 scaled cross entropy forward."""

import inspect

import cuda.bindings.driver as cuda
import cutlass
import torch

from cutlass import Float32
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op

LOG2_E = 1.4426950408889634
LN2 = 0.6931471805599453
NEG_INF_F32 = -1.0e38
MASK_F32 = -3.0e38
HOPPER_MAX_SMEM_BYTES = 227 * 1024

_stream_cache = {}
_max_active_clusters_cache = {}

try:
    _FMAX_NEEDS_RESULT_TYPE = next(iter(inspect.signature(nvvm.fmax).parameters)) == "res"
except Exception:
    _FMAX_NEEDS_RESULT_TYPE = False


@dsl_user_op
def _fmax(a, b, *, loc=None, ip=None) -> Float32:
    av = Float32(a).ir_value(loc=loc, ip=ip)
    bv = Float32(b).ir_value(loc=loc, ip=ip)
    if _FMAX_NEEDS_RESULT_TYPE:
        return Float32(nvvm.fmax(T.f32(), av, bv, loc=loc, ip=ip))
    return Float32(nvvm.fmax(av, bv, loc=loc, ip=ip))


def _cute_stream():
    raw = torch.cuda.current_stream().cuda_stream
    stream = _stream_cache.get(raw)
    if stream is None:
        stream = cuda.CUstream(raw)
        _stream_cache[raw] = stream
    return stream


def _max_active_clusters(cluster_size):
    key = (torch.cuda.current_device(), cluster_size)
    value = _max_active_clusters_cache.get(key)
    if value is None:
        torch.cuda.current_stream()
        value = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size)
        _max_active_clusters_cache[key] = value
    return value


def _validate(x, weight, target):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(x.device) != (9, 0):
        raise RuntimeError("SM90 fused scaled cross entropy forward requires a Hopper (sm90) GPU")
    if x.device != weight.device or x.device != target.device:
        raise ValueError("input, weight, and target must be on the same CUDA device")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("SM90 fused scaled cross entropy forward supports BF16 input and weight only")
    if x.ndim != 2 or weight.ndim != 2 or target.ndim != 1:
        raise ValueError("expected input[BT,H], weight[V,H], and target[BT]")
    if x.shape[0] != target.shape[0] or x.shape[1] != weight.shape[1]:
        raise ValueError("input, weight, and target shapes are incompatible")


def _pad_hidden(x, weight, tile_k):
    hidden_size = x.shape[1]
    padded = (hidden_size + tile_k - 1) // tile_k * tile_k
    if padded == hidden_size:
        return x.contiguous(), weight.contiguous(), hidden_size
    return (
        torch.nn.functional.pad(x, (0, padded - hidden_size)),
        torch.nn.functional.pad(weight, (0, padded - hidden_size)),
        hidden_size,
    )
