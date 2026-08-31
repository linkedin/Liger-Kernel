"""LCK adapter for tensor-parallel fused linear scaled cross entropy."""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

_HIDDEN_ALIGNMENT = 8
_RUNTIME = None


@dataclass
class _Runtime:
    group_ranks: tuple[int, ...]
    device_index: int
    team_handle: int
    nccl_comm_handle: int
    pool_generation: int
    max_tokens: int = 0
    max_hidden: int = 0
    max_local_vocab: int = 0
    max_tiles_per_reduce: int = 0


def _get_tvm_ffi():
    from liger_kernel.ops.cute import _load_tvm_ffi

    return _load_tvm_ffi()


def _get_nvshmem():
    from liger_cute_kernels import nvshmem

    return nvshmem


def is_available() -> bool:
    try:
        tvm_ffi = _get_tvm_ffi()
        nvshmem = _get_nvshmem()
    except ImportError:
        return False
    required_tvm_ffi = (
        "fused_linear_scaled_cross_entropy_configure_backward",
        "fused_linear_scaled_cross_entropy_configure_forward",
        "fused_linear_scaled_cross_entropy_backward",
        "fused_linear_scaled_cross_entropy_forward",
        "nccl_comm_destroy",
        "nccl_comm_init_rank",
        "nccl_get_unique_id",
        "nccl_unique_id_nbytes",
        "pool_generation",
    )
    if not all(callable(getattr(tvm_ffi, name, None)) for name in required_tvm_ffi):
        return False
    return all(
        callable(getattr(nvshmem, name, None)) for name in ("ensure_initialized", "is_initialized", "resolve_team")
    )


def _validate_temperature(temperature) -> None:
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a real number")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")


def _validate_inputs(x, weight, target, vocab_start, tiles_per_reduce, return_entropy) -> None:
    if x.device != weight.device or x.device != target.device:
        raise ValueError("input, weight, and target must be on the same CUDA device")
    if not x.is_cuda:
        raise RuntimeError("the LCK tensor-parallel path requires CUDA tensors")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("the LCK tensor-parallel path supports BF16 input and weight only")
    if target.dtype != torch.int64:
        raise TypeError("target must be an int64 tensor of global vocabulary indices")
    if x.ndim != 2 or weight.ndim != 2 or target.ndim != 1:
        raise ValueError("expected input[M,H], weight[V_local,H], and target[M]")
    if x.shape[0] != target.shape[0] or x.shape[1] != weight.shape[1]:
        raise ValueError(
            f"input {tuple(x.shape)}, weight {tuple(weight.shape)}, and target {tuple(target.shape)} are incompatible"
        )
    if x.shape[0] == 0 or x.shape[1] == 0 or weight.shape[0] == 0:
        raise ValueError("token, hidden, and local vocabulary dimensions must be non-empty")
    if not isinstance(vocab_start, int) or isinstance(vocab_start, bool):
        raise TypeError("vocab_start must be an int")
    if vocab_start < 0:
        raise ValueError("vocab_start must be non-negative")
    if tiles_per_reduce not in (1, 2, 4):
        raise ValueError("tiles_per_reduce must be one of 1, 2, or 4")
    if not isinstance(return_entropy, bool):
        raise TypeError("return_entropy must be a bool")


def _pad_hidden(x, weight):
    hidden = x.shape[1]
    padded_hidden = (hidden + _HIDDEN_ALIGNMENT - 1) // _HIDDEN_ALIGNMENT * _HIDDEN_ALIGNMENT
    if padded_hidden == hidden:
        return x.contiguous(), weight.contiguous(), hidden
    padding = (0, padded_hidden - hidden)
    return (
        torch.nn.functional.pad(x.contiguous(), padding),
        torch.nn.functional.pad(weight.contiguous(), padding),
        hidden,
    )


def _group_ranks(process_group: "ProcessGroup") -> tuple[int, ...]:
    return tuple(dist.get_global_rank(process_group, rank) for rank in range(dist.get_world_size(process_group)))


def _create_nccl_communicator(process_group: "ProcessGroup", device: torch.device) -> int:
    tvm_ffi = _get_tvm_ffi()
    rank = dist.get_rank(process_group)
    group_ranks = _group_ranks(process_group)
    unique_id = torch.empty(tvm_ffi.nccl_unique_id_nbytes(), dtype=torch.uint8, device=device)
    if rank == 0:
        unique_id.copy_(tvm_ffi.nccl_get_unique_id(), non_blocking=False)
    dist.broadcast(unique_id, src=group_ranks[0], group=process_group)
    return tvm_ffi.nccl_comm_init_rank(rank, len(group_ranks), unique_id.cpu().contiguous())


def _configure_runtime(runtime: _Runtime, tokens: int, hidden: int, local_vocab: int, tiles_per_reduce: int) -> None:
    if (
        tokens <= runtime.max_tokens
        and hidden <= runtime.max_hidden
        and local_vocab <= runtime.max_local_vocab
        and tiles_per_reduce <= runtime.max_tiles_per_reduce
    ):
        return

    tvm_ffi = _get_tvm_ffi()
    tvm_ffi.fused_linear_scaled_cross_entropy_configure_backward(
        max(tokens, runtime.max_tokens),
        max(hidden, runtime.max_hidden),
        max(local_vocab, runtime.max_local_vocab),
        max(tiles_per_reduce, runtime.max_tiles_per_reduce),
        runtime.team_handle,
    )
    tvm_ffi.fused_linear_scaled_cross_entropy_configure_forward(
        max(tokens, runtime.max_tokens),
        max(local_vocab, runtime.max_local_vocab),
    )
    runtime.max_tokens = max(tokens, runtime.max_tokens)
    runtime.max_hidden = max(hidden, runtime.max_hidden)
    runtime.max_local_vocab = max(local_vocab, runtime.max_local_vocab)
    runtime.max_tiles_per_reduce = max(tiles_per_reduce, runtime.max_tiles_per_reduce)


def _prepare_runtime(process_group: "ProcessGroup", device, tokens, hidden, local_vocab, tiles_per_reduce) -> _Runtime:
    global _RUNTIME

    nvshmem = _get_nvshmem()
    tvm_ffi = _get_tvm_ffi()
    if _RUNTIME is not None and not nvshmem.is_initialized():
        tvm_ffi.nccl_comm_destroy(_RUNTIME.nccl_comm_handle)
        _RUNTIME = None

    ranks = _group_ranks(process_group)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    if _RUNTIME is not None:
        if _RUNTIME.group_ranks != ranks or _RUNTIME.device_index != device_index:
            raise RuntimeError(
                "the LCK tensor-parallel runtime is already bound to another process group or CUDA device"
            )
        pool_generation = tvm_ffi.pool_generation()
        if _RUNTIME.pool_generation != pool_generation:
            _RUNTIME.pool_generation = pool_generation
            _RUNTIME.max_tokens = 0
            _RUNTIME.max_hidden = 0
            _RUNTIME.max_local_vocab = 0
            _RUNTIME.max_tiles_per_reduce = 0
        _configure_runtime(_RUNTIME, tokens, hidden, local_vocab, tiles_per_reduce)
        return _RUNTIME

    with torch.cuda.device(device):
        # Share one WORLD bootstrap with LigerMoE; EP and TP remain separate
        # cached NVSHMEM teams underneath that process-wide runtime.
        nvshmem.ensure_initialized()
        team_handle = nvshmem.resolve_team(process_group)
        runtime = _Runtime(
            group_ranks=ranks,
            device_index=device_index,
            team_handle=team_handle,
            nccl_comm_handle=0,
            pool_generation=tvm_ffi.pool_generation(),
        )
        _configure_runtime(runtime, tokens, hidden, local_vocab, tiles_per_reduce)
        runtime.nccl_comm_handle = _create_nccl_communicator(process_group, device)
    _RUNTIME = runtime
    return runtime


class LigerFusedLinearScaledCrossEntropyLckTPFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        target,
        vocab_start,
        temperature,
        ignore_index,
        tiles_per_reduce,
        return_entropy,
        tp_group,
    ):
        ctx.set_materialize_grads(False)
        _validate_temperature(temperature)
        _validate_inputs(x, weight, target, vocab_start, tiles_per_reduce, return_entropy)

        x_padded, weight_padded, hidden = _pad_hidden(x, weight)
        target = target.contiguous()
        runtime = _prepare_runtime(
            tp_group,
            x.device,
            target.shape[0],
            x_padded.shape[1],
            weight.shape[0],
            tiles_per_reduce,
        )
        tvm_ffi = _get_tvm_ffi()
        nll, lse, entropy = tvm_ffi.fused_linear_scaled_cross_entropy_forward(
            x_padded,
            weight_padded,
            target,
            vocab_start,
            ignore_index,
            1.0 / temperature,
            runtime.nccl_comm_handle,
            return_entropy,
        )

        ctx.save_for_backward(x_padded, weight_padded, target, lse, entropy)
        ctx.hidden = hidden
        ctx.vocab_start = vocab_start
        ctx.ignore_index = ignore_index
        ctx.inverse_temperature = 1.0 / temperature
        ctx.team_handle = runtime.team_handle
        ctx.tiles_per_reduce = tiles_per_reduce
        ctx.return_entropy = return_entropy
        return (nll, entropy) if return_entropy else nll

    @staticmethod
    def backward(ctx, grad_nll, grad_entropy=None):
        x, weight, target, lse, entropy = ctx.saved_tensors
        tokens = target.shape[0]
        if grad_nll is None:
            grad_nll = torch.zeros(tokens, dtype=torch.float32, device=target.device)
        else:
            grad_nll = grad_nll.detach().to(torch.float32).contiguous()
        if grad_nll.shape != (tokens,):
            raise ValueError(f"expected per-token NLL gradients with shape {(tokens,)}, got {tuple(grad_nll.shape)}")

        if grad_entropy is None:
            grad_entropy = torch.zeros(tokens, dtype=torch.float32, device=target.device)
        else:
            grad_entropy = grad_entropy.detach().to(torch.float32).contiguous()
        if grad_entropy.shape != (tokens,):
            raise ValueError(
                f"expected per-token entropy gradients with shape {(tokens,)}, got {tuple(grad_entropy.shape)}"
            )

        grad_input, grad_weight = _get_tvm_ffi().fused_linear_scaled_cross_entropy_backward(
            grad_nll,
            grad_entropy,
            x,
            weight,
            target,
            lse,
            entropy,
            ctx.vocab_start,
            ctx.ignore_index,
            ctx.inverse_temperature,
            ctx.team_handle,
            ctx.tiles_per_reduce,
            ctx.return_entropy,
        )
        if ctx.hidden != x.shape[1]:
            grad_input = grad_input[:, : ctx.hidden].contiguous()
            grad_weight = grad_weight[:, : ctx.hidden].contiguous()
        return grad_input, grad_weight, None, None, None, None, None, None, None


__all__ = ["LigerFusedLinearScaledCrossEntropyLckTPFunction", "is_available"]
