"""Device-dispatching frontend for fused scaled cross entropy."""

import math

import torch
import torch.distributed as dist

# The fallback formulas and 512-token chunking are adapted from Verl's
# Apache-2.0 FusedLinearForPPOFunction:
# https://github.com/verl-project/verl/blob/main/verl/utils/experimental/torch_functional.py
_FALLBACK_CHUNK_SIZE = 512


def _load_sm90_function():
    from liger_kernel.ops.cutedsl.ops.fused_scaled_cross_entropy_sm90 import LigerFusedScaledCrossEntropySM90Function

    return LigerFusedScaledCrossEntropySM90Function


def _load_lck_tp_function(process_group, device):
    from liger_kernel.ops.cute.fused_linear_scaled_cross_entropy_tp import (
        LigerFusedLinearScaledCrossEntropyLckTPFunction,
    )
    from liger_kernel.ops.cute.fused_linear_scaled_cross_entropy_tp import is_available
    from liger_kernel.ops.cute.fused_linear_scaled_cross_entropy_tp import supports_process_group

    if not is_available() or not supports_process_group(process_group, device):
        return None
    return LigerFusedLinearScaledCrossEntropyLckTPFunction


def _validate_temperature(temperature):
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a real number")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")


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
    if str(dist.get_backend(process_group)).lower() != "nccl":
        raise RuntimeError("tensor-parallel fused linear scaled cross entropy requires an NCCL process group")
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


def _is_hopper(device):
    if device.type != "cuda" or not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    return torch.cuda.get_device_capability(device) == (9, 0)


class _TensorParallelFusedLinearPPOFallbackFunction(torch.autograd.Function):
    """Chunked TP fallback adapted from Verl's fused linear PPO formulas."""

    @staticmethod
    def forward(ctx, hidden_states, vocab_weights, input_ids, temperature, ignore_index, return_entropy, process_group):
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
        grad_vocab_weights = torch.zeros_like(vocab_weights) if ctx.needs_input_grad[1] else None
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
                grad_vocab_weights.add_(grad_logits.t() @ hidden_chunk)

        if grad_hidden_states is not None and ctx.world_size > 1:
            dist.all_reduce(grad_hidden_states, op=dist.ReduceOp.SUM, group=ctx.process_group)
        return grad_hidden_states, grad_vocab_weights, None, None, None, None, None


def _apply_tp_fallback(_input, weight, target, temperature, ignore_index, return_entropy, process_group):
    return _TensorParallelFusedLinearPPOFallbackFunction.apply(
        _input.contiguous(),
        weight.contiguous(),
        target.contiguous(),
        temperature,
        ignore_index,
        return_entropy,
        process_group,
    )


def _fallback_forward_chunk(hidden_states, vocab_weights, input_ids, valid, temperature):
    logits = (hidden_states @ vocab_weights.t()) / temperature
    output_dtype = logits.dtype
    logits = logits.to(torch.float32)

    probs = logits.softmax(dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    log_probs = logits.log_softmax(dim=-1)
    token_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

    return (
        torch.where(valid, -token_log_probs, torch.zeros_like(token_log_probs)),
        torch.where(valid, entropy, torch.zeros_like(entropy)).to(output_dtype),
    )


def _fallback_backward_chunk(
    grad_nll,
    grad_entropy,
    hidden_states,
    vocab_weights,
    input_ids,
    valid,
    temperature,
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

    grad_logits = grad_logits.to(output_dtype) / temperature
    return grad_logits @ vocab_weights, grad_logits.t() @ hidden_states


class _FusedLinearPPOFallbackFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, vocab_weights, input_ids, temperature, ignore_index, return_entropy):
        ctx.set_materialize_grads(False)
        _validate_fallback_inputs(hidden_states, vocab_weights, input_ids, ignore_index)
        _validate_temperature(temperature)

        valid = input_ids != ignore_index
        safe_input_ids = input_ids.masked_fill(~valid, 0)
        token_count = hidden_states.shape[0]
        nll = torch.empty(token_count, device=hidden_states.device, dtype=torch.float32)
        entropy = hidden_states.new_empty(token_count) if return_entropy else None

        for start in range(0, token_count, _FALLBACK_CHUNK_SIZE):
            end = min(start + _FALLBACK_CHUNK_SIZE, token_count)
            chunk_nll, chunk_entropy = _fallback_forward_chunk(
                hidden_states[start:end],
                vocab_weights,
                safe_input_ids[start:end],
                valid[start:end],
                temperature,
            )
            nll[start:end] = chunk_nll
            if return_entropy:
                entropy[start:end] = chunk_entropy

        ctx.save_for_backward(hidden_states, vocab_weights, safe_input_ids, valid)
        ctx.temperature = temperature
        return (nll, entropy) if return_entropy else nll

    @staticmethod
    def backward(ctx, grad_nll, grad_entropy=None):
        hidden_states, vocab_weights, input_ids, valid = ctx.saved_tensors
        grad_hidden_states = torch.zeros_like(hidden_states) if ctx.needs_input_grad[0] else None
        grad_vocab_weights = torch.zeros_like(vocab_weights) if ctx.needs_input_grad[1] else None

        for start in range(0, hidden_states.shape[0], _FALLBACK_CHUNK_SIZE):
            end = min(start + _FALLBACK_CHUNK_SIZE, hidden_states.shape[0])
            chunk_grad_hidden, chunk_grad_weight = _fallback_backward_chunk(
                grad_nll[start:end] if grad_nll is not None else None,
                grad_entropy[start:end] if grad_entropy is not None else None,
                hidden_states[start:end],
                vocab_weights,
                input_ids[start:end],
                valid[start:end],
                ctx.temperature,
            )
            if grad_hidden_states is not None:
                grad_hidden_states[start:end].add_(chunk_grad_hidden)
            if grad_vocab_weights is not None:
                grad_vocab_weights.add_(chunk_grad_weight)

        return grad_hidden_states, grad_vocab_weights, None, None, None, None


def _resolve_implementation(device):
    if device.type != "cuda" or not torch.cuda.is_available() or torch.version.hip is not None:
        return _FusedLinearPPOFallbackFunction

    capability = torch.cuda.get_device_capability(device)
    if capability == (9, 0):
        return _load_sm90_function()
    return _FusedLinearPPOFallbackFunction


class LigerFusedLinearScaledCrossEntropyFunction:
    """Dispatch fused scaled cross entropy to the implementation for ``input.device``."""

    @staticmethod
    def apply(
        _input,
        weight,
        target,
        temperature=1.0,
        ignore_index=-100,
        m_tiles_per_cluster=1,
        return_entropy=False,
    ):
        if not isinstance(m_tiles_per_cluster, int) or isinstance(m_tiles_per_cluster, bool):
            raise TypeError("m_tiles_per_cluster must be an int")
        if m_tiles_per_cluster < 1:
            raise ValueError("m_tiles_per_cluster must be >= 1")
        if not isinstance(return_entropy, bool):
            raise TypeError("return_entropy must be a bool")

        implementation = _resolve_implementation(_input.device)
        if implementation is _FusedLinearPPOFallbackFunction:
            return implementation.apply(
                _input,
                weight,
                target,
                temperature,
                ignore_index,
                return_entropy,
            )
        return implementation.apply(
            _input,
            weight,
            target,
            temperature,
            ignore_index,
            m_tiles_per_cluster,
            return_entropy,
        )


class LigerFusedLinearScaledCrossEntropyTPFunction:
    """Tensor-parallel frontend using LCK on Hopper and a Liger fallback otherwise.

    ``weight`` is the calling rank's equally sized contiguous vocabulary shard,
    while ``target`` contains global vocabulary indices.
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
    ):
        _validate_temperature(temperature)
        if not isinstance(tiles_per_reduce, int) or isinstance(tiles_per_reduce, bool):
            raise TypeError("tiles_per_reduce must be an int")
        if tiles_per_reduce not in (1, 2, 4):
            raise ValueError("tiles_per_reduce must be one of 1, 2, or 4")
        if not isinstance(return_entropy, bool):
            raise TypeError("return_entropy must be a bool")

        process_group, rank = _tp_group_info(tp_group)
        _validate_tp_inputs(_input, weight, target)
        vocab_start = rank * weight.shape[0]

        lck_function = None
        if _input.dtype == torch.bfloat16 and _is_hopper(_input.device):
            try:
                lck_function = _load_lck_tp_function(process_group, _input.device)
            except ImportError:
                lck_function = None
        if lck_function is not None:
            return lck_function.apply(
                _input,
                weight,
                target,
                vocab_start,
                temperature,
                ignore_index,
                tiles_per_reduce,
                return_entropy,
                process_group,
            )

        return _apply_tp_fallback(
            _input,
            weight,
            target,
            temperature,
            ignore_index,
            return_entropy,
            process_group,
        )


__all__ = [
    "LigerFusedLinearScaledCrossEntropyFunction",
    "LigerFusedLinearScaledCrossEntropyTPFunction",
]
