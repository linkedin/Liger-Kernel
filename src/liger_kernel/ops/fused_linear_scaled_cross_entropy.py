"""Device-dispatching frontend for fused scaled cross entropy."""

import functools
import logging
import math
import os

from importlib.metadata import version

import torch
import torch.distributed as dist

from packaging.version import InvalidVersion
from packaging.version import Version

# The fallback formulas and 512-token chunking are adapted from Verl's
# Apache-2.0 FusedLinearForPPOFunction:
# https://github.com/verl-project/verl/blob/main/verl/utils/experimental/torch_functional.py
_FALLBACK_CHUNK_SIZE = 512
_log = logging.getLogger(__name__)
_SUPPORTS_OUT_DTYPE = Version(torch.__version__.split("+")[0]) >= Version("2.8.0")


def _validate_accum_dtype(accum_dtype, weight_dtype):
    if accum_dtype is None:
        return weight_dtype
    if not isinstance(accum_dtype, torch.dtype):
        raise TypeError("accum_dtype must be a torch.dtype or None")
    if accum_dtype not in (weight_dtype, torch.float32):
        raise ValueError("accum_dtype must be the weight dtype or torch.float32")
    return accum_dtype


def _accumulate_fp32_grad_weight(grad_weight, grad_logits, hidden_states):
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        if (
            _SUPPORTS_OUT_DTYPE
            and hidden_states.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(hidden_states.device)[0] >= 8
            and grad_logits.dtype in (torch.float16, torch.bfloat16)
        ):
            # FP32-output addmm bypasses autocast's operand conversion.
            if hidden_states.dtype != grad_logits.dtype:
                hidden_states = hidden_states.to(grad_logits.dtype)
            torch.addmm(grad_weight, grad_logits.t(), hidden_states, out_dtype=torch.float32, out=grad_weight)
        else:
            grad_weight.addmm_(grad_logits.float().t(), hidden_states.float())


def _load_sm90_function():
    from liger_kernel.ops.cutedsl.ops.fused_scaled_cross_entropy_sm90 import LigerFusedScaledCrossEntropySM90Function

    return LigerFusedScaledCrossEntropySM90Function


@functools.lru_cache(maxsize=1)
def _load_sm100_function():
    try:
        installed_version = version("nvidia-cutlass-dsl")
        if Version(installed_version) < Version("4.6.0"):
            raise ImportError(f"nvidia-cutlass-dsl>=4.6.0 is required, found {installed_version}")
        import cutlass.cute  # noqa: F401

        from liger_kernel.ops.cutedsl.ops.fused_scaled_cross_entropy_sm100 import (
            LigerFusedScaledCrossEntropySM100Function,
        )
    except (ImportError, InvalidVersion) as exc:
        _log.warning(
            "Native Blackwell scaled cross entropy is unavailable (%s); using the PyTorch fallback. "
            "Install nvidia-cutlass-dsl>=4.6.0 with cutlass.cute, or set LIGER_DISABLE_SM100_SCALED_CE=1 "
            "to disable native dispatch.",
            exc,
        )
        return None

    return LigerFusedScaledCrossEntropySM100Function


@functools.lru_cache(maxsize=1)
def _load_cutile_function():
    try:
        from liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy import (
            LigerFusedLinearScaledCrossEntropyFunction,
        )
    except ImportError as exc:
        _log.warning(
            "cuTile scaled cross entropy is unavailable (%s); falling back to the default dispatch. "
            "Install cuda-tile with the tileiras compiler (pip install 'liger-kernel[cutile-tileiras]'), "
            "or unset LIGER_ENABLE_CUTILE_SCALED_CE to silence this warning.",
            exc,
        )
        return None

    return LigerFusedLinearScaledCrossEntropyFunction


def _load_native_tp_function():
    from liger_kernel.ops.cute.fused_linear_scaled_cross_entropy_tp import (
        LigerFusedLinearScaledCrossEntropyNativeTPFunction,
    )
    from liger_kernel.ops.cute.fused_linear_scaled_cross_entropy_tp import is_available

    return LigerFusedLinearScaledCrossEntropyNativeTPFunction if is_available() else None


def _validate_temperature(temperature):
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a real number")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")


def _validate_selected_token_ids(selected_token_ids, weight, *, token_count=None):
    if selected_token_ids is None:
        return None
    if not isinstance(selected_token_ids, torch.Tensor):
        raise TypeError("selected_token_ids must be an int64 tensor")
    if token_count is not None:
        if selected_token_ids.ndim != 1 and not (
            selected_token_ids.ndim == 2 and selected_token_ids.shape[0] == token_count
        ):
            raise ValueError("selected_token_ids must have shape [K] or [M, K] with one row per token")
    elif selected_token_ids.ndim != 1:
        raise ValueError("selected_token_ids must have shape [K]")
    if selected_token_ids.dtype != torch.int64:
        raise TypeError("selected_token_ids must be an int64 tensor")
    if selected_token_ids.device != weight.device:
        raise ValueError("selected_token_ids and weight must be on the same device")
    if bool(((selected_token_ids < 0) | (selected_token_ids >= weight.shape[0])).any()):
        raise ValueError(f"selected_token_ids must be in [0, {weight.shape[0]})")
    return selected_token_ids.contiguous()


def _scaled_ce_outputs(nll, entropy, selected_log_probs):
    outputs = (nll,) if entropy is None else (nll, entropy)
    if selected_log_probs is not None:
        outputs += (selected_log_probs,)
    return outputs[0] if len(outputs) == 1 else outputs


def _validate_fallback_inputs(_input, weight, target, ignore_index):
    if _input.ndim != 2 or weight.ndim != 2 or target.ndim != 1:
        raise ValueError("expected input[M,H], weight[V,H], and target[M]")
    if _input.shape[0] != target.shape[0] or _input.shape[1] != weight.shape[1]:
        raise ValueError(
            f"input {tuple(_input.shape)}, weight {tuple(weight.shape)} and target "
            f"{tuple(target.shape)} shapes are incompatible"
        )
    if _input.shape[0] == 0 or _input.shape[1] == 0 or weight.shape[0] == 0:
        raise ValueError("input, hidden, and vocabulary dimensions must be non-empty")
    if _input.device != weight.device or _input.device != target.device:
        raise ValueError("input, weight, and target must be on the same device")
    if _input.dtype != weight.dtype or not torch.is_floating_point(_input):
        raise TypeError("input and weight must have the same floating-point dtype")
    if target.dtype != torch.int64:
        raise TypeError("target must be an int64 tensor")

    valid_targets = target[target != ignore_index]
    if valid_targets.numel() and (bool((valid_targets < 0).any()) or bool((valid_targets >= weight.shape[0]).any())):
        raise ValueError(
            f"target contains values outside [0, {weight.shape[0]}) that are not ignore_index={ignore_index}"
        )


def _tp_group_info(process_group):
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before using tensor-parallel scaled cross entropy")
    process_group = process_group if process_group is not None else dist.group.WORLD
    return process_group, dist.get_rank(process_group)


def _validate_tp_inputs(_input, weight, target):
    if _input.ndim != 2 or weight.ndim != 2 or target.ndim != 1:
        raise ValueError("expected input[M,H], weight[V_local,H], and target[M]")
    if _input.shape[0] != target.shape[0] or _input.shape[1] != weight.shape[1]:
        raise ValueError(
            f"input {tuple(_input.shape)}, weight {tuple(weight.shape)}, and target {tuple(target.shape)} "
            "shapes are incompatible"
        )
    if _input.shape[0] == 0 or _input.shape[1] == 0 or weight.shape[0] == 0:
        raise ValueError("token, hidden, and local vocabulary dimensions must be non-empty")
    if _input.device != weight.device or _input.device != target.device:
        raise ValueError("input, weight, and target must be on the same device")
    if not _input.is_cuda:
        raise RuntimeError("tensor-parallel fused linear scaled cross entropy requires CUDA tensors")
    if _input.dtype != weight.dtype or not torch.is_floating_point(_input):
        raise TypeError("input and weight must have the same floating-point dtype")
    if target.dtype != torch.int64:
        raise TypeError("target must be an int64 tensor")


def _supports_native_tp_architecture(device):
    if device.type != "cuda" or not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return (major, minor) == (9, 0) or major == 10


class _TensorParallelFusedLinearPPOFallbackFunction(torch.autograd.Function):
    """Chunked TP fallback adapted from Verl's fused linear PPO formulas."""

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        vocab_weights,
        input_ids,
        temperature,
        ignore_index,
        return_entropy,
        process_group,
        accum_dtype=None,
    ):
        ctx.accum_dtype = _validate_accum_dtype(accum_dtype, vocab_weights.dtype)
        ctx.set_materialize_grads(False)
        rank = dist.get_rank(process_group)
        world_size = dist.get_world_size(process_group)
        local_vocab = vocab_weights.shape[0]
        vocab_start = rank * local_vocab
        valid = input_ids != ignore_index
        safe_input_ids = input_ids.masked_fill(~valid, 0)
        token_count = hidden_states.shape[0]

        nll = torch.empty(token_count, dtype=torch.float32, device=hidden_states.device)
        lse = torch.empty_like(nll)
        entropy = torch.empty_like(nll) if return_entropy else torch.zeros_like(nll)

        for start in range(0, token_count, _FALLBACK_CHUNK_SIZE):
            end = min(start + _FALLBACK_CHUNK_SIZE, token_count)
            logits = (hidden_states[start:end] @ vocab_weights.t()).float() / temperature
            global_max = logits.amax(dim=-1)
            if world_size > 1:
                dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=process_group)

            exp_logits = torch.exp(logits - global_max[:, None])
            local_sum = exp_logits.sum(dim=-1)
            local_target = safe_input_ids[start:end] - vocab_start
            owns_target = valid[start:end] & (local_target >= 0) & (local_target < local_vocab)
            target_index = local_target.clamp(0, local_vocab - 1)
            target_logit = torch.where(
                owns_target,
                logits.gather(-1, target_index[:, None]).squeeze(-1),
                torch.zeros_like(local_sum),
            )

            if return_entropy:
                local_weighted_sum = (exp_logits * logits).sum(dim=-1)
                reduced = torch.stack((local_sum, local_weighted_sum, target_logit))
            else:
                reduced = torch.stack((local_sum, target_logit))
            if world_size > 1:
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=process_group)

            global_lse = global_max + torch.log(reduced[0])
            chunk_nll = global_lse - reduced[-1]
            nll[start:end] = torch.where(valid[start:end], chunk_nll, torch.zeros_like(chunk_nll))
            lse[start:end] = global_lse
            if return_entropy:
                chunk_entropy = global_lse - reduced[1] / reduced[0]
                entropy[start:end] = torch.where(
                    valid[start:end],
                    chunk_entropy,
                    torch.zeros_like(chunk_entropy),
                )

        ctx.save_for_backward(hidden_states, vocab_weights, safe_input_ids, valid, lse, entropy)
        ctx.temperature = temperature
        ctx.vocab_start = vocab_start
        ctx.local_vocab = local_vocab
        ctx.return_entropy = return_entropy
        ctx.process_group = process_group
        ctx.world_size = world_size
        return (nll, entropy) if return_entropy else nll

    @staticmethod
    def backward(ctx, grad_nll, grad_entropy=None):
        hidden_states, vocab_weights, input_ids, valid, lse, entropy = ctx.saved_tensors
        token_count = hidden_states.shape[0]
        if grad_nll is None:
            grad_nll = torch.zeros(token_count, dtype=torch.float32, device=hidden_states.device)
        else:
            grad_nll = grad_nll.to(torch.float32)
        if grad_entropy is None:
            grad_entropy = torch.zeros_like(grad_nll)
        else:
            grad_entropy = grad_entropy.to(torch.float32)

        grad_hidden_states = torch.zeros_like(hidden_states) if ctx.needs_input_grad[0] else None
        grad_vocab_weights = torch.zeros_like(vocab_weights, dtype=ctx.accum_dtype) if ctx.needs_input_grad[1] else None
        inverse_temperature = 1.0 / ctx.temperature

        for start in range(0, token_count, _FALLBACK_CHUNK_SIZE):
            end = min(start + _FALLBACK_CHUNK_SIZE, token_count)
            hidden_chunk = hidden_states[start:end]
            logits = (hidden_chunk @ vocab_weights.t()).float() * inverse_temperature
            probabilities = torch.exp(logits - lse[start:end, None])
            nll_scale = grad_nll[start:end] * valid[start:end]
            grad_logits = probabilities * nll_scale[:, None]

            local_target = input_ids[start:end] - ctx.vocab_start
            owns_target = valid[start:end] & (local_target >= 0) & (local_target < ctx.local_vocab)
            target_index = local_target.clamp(0, ctx.local_vocab - 1)
            rows = torch.arange(end - start, device=hidden_states.device)
            grad_logits[rows, target_index] -= nll_scale * owns_target

            if ctx.return_entropy:
                entropy_scale = grad_entropy[start:end] * valid[start:end]
                grad_logits += (
                    probabilities * (lse[start:end, None] - entropy[start:end, None] - logits) * entropy_scale[:, None]
                )

            grad_logits = (grad_logits * inverse_temperature).to(hidden_states.dtype)
            if grad_hidden_states is not None:
                grad_hidden_states[start:end] = grad_logits @ vocab_weights
            if grad_vocab_weights is not None:
                if grad_vocab_weights.dtype != grad_logits.dtype:
                    _accumulate_fp32_grad_weight(grad_vocab_weights, grad_logits, hidden_chunk)
                else:
                    grad_vocab_weights.add_(grad_logits.t() @ hidden_chunk)

        if grad_hidden_states is not None and ctx.world_size > 1:
            dist.all_reduce(grad_hidden_states, op=dist.ReduceOp.SUM, group=ctx.process_group)
        if grad_vocab_weights is not None:
            grad_vocab_weights = grad_vocab_weights.to(vocab_weights.dtype)
        return grad_hidden_states, grad_vocab_weights, None, None, None, None, None, None


def _apply_tp_fallback(
    _input, weight, target, temperature, ignore_index, return_entropy, process_group, accum_dtype=None
):
    return _TensorParallelFusedLinearPPOFallbackFunction.apply(
        _input.contiguous(),
        weight.contiguous(),
        target.contiguous(),
        temperature,
        ignore_index,
        return_entropy,
        process_group,
        accum_dtype,
    )


def _fallback_forward_chunk(hidden_states, vocab_weights, input_ids, valid, temperature, selected_token_ids=None):
    logits = (hidden_states @ vocab_weights.t()) / temperature
    output_dtype = logits.dtype
    logits = logits.to(torch.float32)

    probs = logits.softmax(dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    log_probs = logits.log_softmax(dim=-1)
    token_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

    outputs = (
        torch.where(valid, -token_log_probs, torch.zeros_like(token_log_probs)),
        torch.where(valid, entropy, torch.zeros_like(entropy)).to(output_dtype),
    )
    if selected_token_ids is not None:
        selected = log_probs.index_select(1, selected_token_ids).masked_fill(~valid[:, None], 0.0)
        outputs += (selected,)
    return outputs


def _fallback_backward_chunk(
    grad_nll,
    grad_entropy,
    hidden_states,
    vocab_weights,
    input_ids,
    valid,
    temperature,
    selected_token_ids=None,
    grad_selected=None,
    grad_weight=None,
):
    logits = (hidden_states @ vocab_weights.t()) / temperature
    output_dtype = logits.dtype
    logits = logits.to(torch.float32)
    probs = logits.softmax(dim=-1)
    grad_logits = torch.zeros_like(logits)

    if grad_nll is not None:
        grad_log_probs = -grad_nll.to(torch.float32) * valid
        one_hot = torch.zeros_like(logits).scatter_(-1, input_ids.unsqueeze(-1), 1)
        grad_logits.add_(grad_log_probs.unsqueeze(-1) * (one_hot - probs))

    if grad_entropy is not None:
        grad_entropy = grad_entropy.to(torch.float32) * valid
        log_probs = logits.log_softmax(dim=-1)
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
        grad_logits.add_(probs * (log_probs + entropy.unsqueeze(-1)) * (-grad_entropy.unsqueeze(-1)))

    if grad_selected is not None:
        selected_grad = grad_selected.float().masked_fill(~valid[:, None], 0.0)
        grad_logits.sub_(probs * selected_grad.sum(-1, keepdim=True))
        grad_logits.scatter_add_(1, selected_token_ids.expand(hidden_states.shape[0], -1), selected_grad)

    grad_logits = grad_logits.to(output_dtype) / temperature
    if grad_weight is not None:
        if grad_weight.dtype != grad_logits.dtype:
            _accumulate_fp32_grad_weight(grad_weight, grad_logits, hidden_states)
        else:
            grad_weight.add_(grad_logits.t() @ hidden_states)
    return grad_logits @ vocab_weights


class _FusedLinearPPOFallbackFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states,
        vocab_weights,
        input_ids,
        temperature,
        ignore_index,
        return_entropy,
        selected_token_ids=None,
        accum_dtype=None,
    ):
        ctx.accum_dtype = _validate_accum_dtype(accum_dtype, vocab_weights.dtype)
        ctx.set_materialize_grads(False)
        _validate_fallback_inputs(hidden_states, vocab_weights, input_ids, ignore_index)
        _validate_temperature(temperature)
        selected_token_ids = _validate_selected_token_ids(selected_token_ids, vocab_weights)

        valid = input_ids != ignore_index
        safe_input_ids = input_ids.masked_fill(~valid, 0)
        token_count = hidden_states.shape[0]
        nll = torch.empty(token_count, device=hidden_states.device, dtype=torch.float32)
        entropy = hidden_states.new_empty(token_count) if return_entropy else None
        selected = nll.new_empty(token_count, selected_token_ids.numel()) if selected_token_ids is not None else None

        for start in range(0, token_count, _FALLBACK_CHUNK_SIZE):
            end = min(start + _FALLBACK_CHUNK_SIZE, token_count)
            chunk_outputs = _fallback_forward_chunk(
                hidden_states[start:end],
                vocab_weights,
                safe_input_ids[start:end],
                valid[start:end],
                temperature,
                selected_token_ids,
            )
            nll[start:end] = chunk_outputs[0]
            if return_entropy:
                entropy[start:end] = chunk_outputs[1]
            if selected is not None:
                selected[start:end] = chunk_outputs[2]

        ctx.save_for_backward(hidden_states, vocab_weights, safe_input_ids, valid, selected_token_ids)
        ctx.temperature = temperature
        ctx.return_entropy = return_entropy
        return _scaled_ce_outputs(nll, entropy, selected)

    @staticmethod
    def backward(ctx, grad_nll, grad_entropy=None, grad_selected=None):
        hidden_states, vocab_weights, input_ids, valid, selected_token_ids = ctx.saved_tensors
        if not ctx.return_entropy:
            grad_selected, grad_entropy = grad_entropy, None
        grad_hidden_states = torch.zeros_like(hidden_states) if ctx.needs_input_grad[0] else None
        grad_vocab_weights = torch.zeros_like(vocab_weights, dtype=ctx.accum_dtype) if ctx.needs_input_grad[1] else None

        for start in range(0, hidden_states.shape[0], _FALLBACK_CHUNK_SIZE):
            end = min(start + _FALLBACK_CHUNK_SIZE, hidden_states.shape[0])
            chunk_grad_hidden = _fallback_backward_chunk(
                grad_nll[start:end] if grad_nll is not None else None,
                grad_entropy[start:end] if grad_entropy is not None else None,
                hidden_states[start:end],
                vocab_weights,
                input_ids[start:end],
                valid[start:end],
                ctx.temperature,
                selected_token_ids,
                grad_selected[start:end] if grad_selected is not None else None,
                grad_vocab_weights,
            )
            if grad_hidden_states is not None:
                grad_hidden_states[start:end].add_(chunk_grad_hidden)
        if grad_vocab_weights is not None:
            grad_vocab_weights = grad_vocab_weights.to(vocab_weights.dtype)
        return grad_hidden_states, grad_vocab_weights, None, None, None, None, None, None


def _resolve_implementation(device, dtype=None):
    if device.type != "cuda" or not torch.cuda.is_available() or torch.version.hip is not None:
        return _FusedLinearPPOFallbackFunction

    capability = torch.cuda.get_device_capability(device)
    if (
        dtype == torch.bfloat16
        and capability in ((9, 0), (10, 0), (10, 3))
        and os.environ.get("LIGER_ENABLE_CUTILE_SCALED_CE") == "1"
    ):
        cutile_implementation = _load_cutile_function()
        if cutile_implementation is not None:
            return cutile_implementation
    if capability == (9, 0):
        return _load_sm90_function()
    if capability in ((10, 0), (10, 3)) and dtype == torch.bfloat16:
        if os.environ.get("LIGER_DISABLE_SM100_SCALED_CE") == "1":
            return _FusedLinearPPOFallbackFunction
        implementation = _load_sm100_function()
        if implementation is not None:
            return implementation
    return _FusedLinearPPOFallbackFunction


class LigerFusedLinearScaledCrossEntropyFunction:
    """Per-token NLL and optional entropy, dispatched by input device.

    Supplying int64 ``selected_token_ids[K]`` appends FP32 full-vocabulary
    log-probabilities ``[M,K]`` to the outputs. Ignored rows are zero; ID order
    and duplicates are preserved. The cuTile backend also accepts per-token IDs
    of shape ``[M,K]``. All floating-point outputs are differentiable.

    ``accum_dtype=None`` accumulates weight gradients in the weight dtype.
    Use ``torch.float32`` to accumulate dW across token chunks in FP32, then
    cast once to the weight dtype. Forward, dZ and dX precision are unchanged.
    """

    @staticmethod
    def apply(
        _input,
        weight,
        target,
        temperature=1.0,
        ignore_index=-100,
        m_tiles_per_cluster=1,
        return_entropy=False,
        selected_token_ids=None,
        accum_dtype=None,
    ):
        if not isinstance(m_tiles_per_cluster, int) or isinstance(m_tiles_per_cluster, bool):
            raise TypeError("m_tiles_per_cluster must be an int")
        if m_tiles_per_cluster < 1:
            raise ValueError("m_tiles_per_cluster must be >= 1")
        if not isinstance(return_entropy, bool):
            raise TypeError("return_entropy must be a bool")
        if accum_dtype is not None:
            _validate_accum_dtype(accum_dtype, weight.dtype)

        implementation = _resolve_implementation(_input.device, _input.dtype)
        selected_args = (
            (selected_token_ids, accum_dtype)
            if accum_dtype is not None
            else (selected_token_ids,)
            if selected_token_ids is not None
            else ()
        )
        if implementation is _FusedLinearPPOFallbackFunction:
            return implementation.apply(
                _input,
                weight,
                target,
                temperature,
                ignore_index,
                return_entropy,
                *selected_args,
            )
        return implementation.apply(
            _input,
            weight,
            target,
            temperature,
            ignore_index,
            m_tiles_per_cluster,
            return_entropy,
            *selected_args,
        )


class LigerFusedLinearScaledCrossEntropyTPFunction:
    """Tensor-parallel fused projection and scaled cross entropy.

    ``weight`` is the calling rank's equally sized contiguous vocabulary shard,
    while ``target`` contains global vocabulary indices. ``tp_group`` must be
    initialized before the call and must match the NVSHMEM team used by the
    native backend. BF16 Hopper/Blackwell inputs use the native CUTLASS +
    NVSHMEM path when available; otherwise the operation uses the repository's
    512-token Verl-derived fallback.

    Returns per-token NLL, or ``(nll, entropy)`` when ``return_entropy=True``.
    Autograd returns a tensor-parallel reduced input gradient and a rank-local
    gradient for the vocabulary shard.

    ``accum_dtype=None`` uses the weight dtype for dW accumulation.
    ``torch.float32`` retains FP32 dW partials and accumulates token waves in
    FP32 before casting to the weight dtype. TP dX reduction is unchanged.
    """

    @staticmethod
    def apply(
        _input,
        weight,
        target,
        tp_group,
        temperature=1.0,
        ignore_index=-100,
        tiles_per_reduce=1,
        return_entropy=False,
        accum_dtype=None,
    ):
        _validate_temperature(temperature)
        if not isinstance(tiles_per_reduce, int) or isinstance(tiles_per_reduce, bool):
            raise TypeError("tiles_per_reduce must be an int")
        if tiles_per_reduce not in (1, 2, 4):
            raise ValueError("tiles_per_reduce must be one of 1, 2, or 4")
        if not isinstance(return_entropy, bool):
            raise TypeError("return_entropy must be a bool")
        if accum_dtype is not None:
            _validate_accum_dtype(accum_dtype, weight.dtype)
        accum_args = (accum_dtype,) if accum_dtype is not None else ()

        process_group, rank = _tp_group_info(tp_group)
        _validate_tp_inputs(_input, weight, target)
        vocab_start = rank * weight.shape[0]

        native_function = None
        if _input.dtype == torch.bfloat16 and _supports_native_tp_architecture(_input.device):
            try:
                native_function = _load_native_tp_function()
            except ImportError:
                native_function = None
        if native_function is not None:
            return native_function.apply(
                _input,
                weight,
                target,
                vocab_start,
                temperature,
                ignore_index,
                tiles_per_reduce,
                return_entropy,
                process_group,
                *accum_args,
            )

        return _apply_tp_fallback(
            _input,
            weight,
            target,
            temperature,
            ignore_index,
            return_entropy,
            process_group,
            *accum_args,
        )


__all__ = [
    "LigerFusedLinearScaledCrossEntropyFunction",
    "LigerFusedLinearScaledCrossEntropyTPFunction",
]
