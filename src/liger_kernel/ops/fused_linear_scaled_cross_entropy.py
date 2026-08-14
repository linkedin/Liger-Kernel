"""Device-dispatching frontend for fused scaled cross entropy."""

import math

import torch

# The fallback formulas and 512-token chunking are adapted from Verl's
# Apache-2.0 FusedLinearForPPOFunction:
# https://github.com/verl-project/verl/blob/main/verl/utils/experimental/torch_functional.py
_FALLBACK_CHUNK_SIZE = 512


def _load_sm90_function():
    from liger_kernel.ops.cutedsl.ops.fused_scaled_cross_entropy_sm90 import LigerFusedScaledCrossEntropySM90Function

    return LigerFusedScaledCrossEntropySM90Function


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


__all__ = ["LigerFusedLinearScaledCrossEntropyFunction"]
