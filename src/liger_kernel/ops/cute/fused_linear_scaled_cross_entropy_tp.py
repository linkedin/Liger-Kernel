"""LCK adapter for tensor-parallel fused linear scaled cross entropy."""

from __future__ import annotations

import math

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

_HIDDEN_ALIGNMENT = 8


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
        native_module = tvm_ffi._load_module()
    except ImportError:
        return False
    required_native = (
        "fused_linear_scaled_cross_entropy_configure_backward",
        "fused_linear_scaled_cross_entropy_configure_forward",
        "fused_linear_scaled_cross_entropy_backward",
        "fused_linear_scaled_cross_entropy_forward",
    )
    if not all(hasattr(native_module, name) for name in required_native):
        return False
    return callable(getattr(nvshmem, "resolve_team", None))


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


@contextmanager
def _cuda_device_context(tensor):
    if not tensor.is_cuda:
        yield
        return
    with torch.cuda.device(tensor.device):
        torch.cuda.set_device(tensor.device)
        yield


def _prepare_lck_call(process_group: "ProcessGroup", device, tokens, hidden, local_vocab, tiles_per_reduce) -> int:
    tvm_ffi = _get_tvm_ffi()
    with torch.cuda.device(device):
        team_handle = _get_nvshmem().resolve_team(process_group, create=False)
        tvm_ffi.fused_linear_scaled_cross_entropy_configure_backward(
            tokens,
            hidden,
            local_vocab,
            tiles_per_reduce,
            team_handle,
        )
        tvm_ffi.fused_linear_scaled_cross_entropy_configure_forward(
            tokens,
            local_vocab,
        )
    return team_handle


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
        team_handle = _prepare_lck_call(
            tp_group,
            x.device,
            target.shape[0],
            x_padded.shape[1],
            weight.shape[0],
            tiles_per_reduce,
        )
        with _cuda_device_context(x_padded):
            nll, lse, entropy = _get_tvm_ffi().fused_linear_scaled_cross_entropy_forward(
                x_padded,
                weight_padded,
                target,
                vocab_start,
                ignore_index,
                1.0 / temperature,
                team_handle,
                return_entropy,
            )

        ctx.save_for_backward(x_padded, weight_padded, target, lse, entropy)
        ctx.hidden = hidden
        ctx.vocab_start = vocab_start
        ctx.ignore_index = ignore_index
        ctx.inverse_temperature = 1.0 / temperature
        ctx.team_handle = team_handle
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

        with _cuda_device_context(x):
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
