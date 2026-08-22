"""Device-dispatching frontend for fused scaled cross entropy."""

import math

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import device_context
from liger_kernel.utils import infer_device_arch

# The fallback formulas and 512-token chunking are adapted from Verl's
# Apache-2.0 FusedLinearForPPOFunction:
# https://github.com/verl-project/verl/blob/main/verl/utils/experimental/torch_functional.py
_FALLBACK_CHUNK_SIZE = 512
_MIB = 1024 * 1024
_SM103_LOGITS_WORKSPACE_BYTES = 512 * _MIB
_SM103_BLOCK_SIZE = 8192
_SM103_NUM_WARPS = 8
_LOG2_E = tl.constexpr(1.4426950408889634)


@triton.jit
def _scaled_cross_entropy_forward_kernel(
    logits,
    target,
    nll,
    entropy,
    lse,
    logits_row_stride: tl.constexpr,
    vocab_size,
    inv_temperature: tl.constexpr,
    ignore_index: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RETURN_ENTROPY: tl.constexpr,
):
    row = tl.program_id(0)
    row_logits = logits + row * logits_row_stride
    target_idx = tl.load(target + row)
    valid = target_idx != ignore_index
    safe_target = tl.where(valid, target_idx, 0)
    target_logit = tl.load(row_logits + safe_target).to(tl.float32) * inv_temperature

    running_max = -float("inf")
    running_sum = 0.0
    running_weighted_sum = 0.0
    offsets = tl.arange(0, BLOCK_SIZE)
    for start in range(0, vocab_size, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < vocab_size
        values = tl.load(row_logits + cols, mask=mask, other=-float("inf")).to(tl.float32)
        values *= inv_temperature
        block_max = tl.max(values, axis=0)
        block_exp = tl.exp2((values - block_max) * _LOG2_E)
        block_sum = tl.sum(block_exp, axis=0)
        new_max = tl.maximum(running_max, block_max)
        previous_scale = tl.exp2((running_max - new_max) * _LOG2_E)
        block_scale = tl.exp2((block_max - new_max) * _LOG2_E)
        running_sum = running_sum * previous_scale + block_sum * block_scale
        if RETURN_ENTROPY:
            safe_values = tl.where(mask, values, 0.0)
            block_weighted_sum = tl.sum(block_exp * safe_values, axis=0)
            running_weighted_sum = running_weighted_sum * previous_scale + block_weighted_sum * block_scale
        running_max = new_max

    row_lse = running_max + tl.log(running_sum)
    tl.store(nll + row, tl.where(valid, row_lse - target_logit, 0.0))
    tl.store(lse + row, tl.where(valid, row_lse, 0.0))
    if RETURN_ENTROPY:
        row_entropy = row_lse - running_weighted_sum / running_sum
        tl.store(entropy + row, tl.where(valid, row_entropy, 0.0))


@triton.jit
def _scaled_cross_entropy_backward_kernel(
    logits,
    target,
    lse,
    entropy,
    grad_nll,
    grad_entropy,
    logits_row_stride: tl.constexpr,
    vocab_size,
    inv_temperature: tl.constexpr,
    ignore_index: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_NLL_GRAD: tl.constexpr,
    HAS_ENTROPY_GRAD: tl.constexpr,
):
    row = tl.program_id(0)
    row_logits = logits + row * logits_row_stride
    target_idx = tl.load(target + row)
    valid = target_idx != ignore_index
    safe_target = tl.where(valid, target_idx, 0)
    row_lse = tl.load(lse + row)

    row_grad_nll = 0.0
    if HAS_NLL_GRAD:
        row_grad_nll = tl.load(grad_nll + row).to(tl.float32)
    row_entropy = 0.0
    row_grad_entropy = 0.0
    if HAS_ENTROPY_GRAD:
        row_entropy = tl.load(entropy + row).to(tl.float32)
        row_grad_entropy = tl.load(grad_entropy + row).to(tl.float32)

    offsets = tl.arange(0, BLOCK_SIZE)
    for start in range(0, vocab_size, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < vocab_size
        raw_logits = tl.load(row_logits + cols, mask=mask, other=0.0).to(tl.float32)
        scaled_logits = raw_logits * inv_temperature
        probabilities = tl.exp2((scaled_logits - row_lse) * _LOG2_E)

        grad_scaled_logits = probabilities * row_grad_nll
        grad_scaled_logits = tl.where(cols == safe_target, grad_scaled_logits - row_grad_nll, grad_scaled_logits)
        if HAS_ENTROPY_GRAD:
            log_probabilities = scaled_logits - row_lse
            grad_scaled_logits -= row_grad_entropy * probabilities * (log_probabilities + row_entropy)
        grad_raw_logits = tl.where(valid & mask, grad_scaled_logits * inv_temperature, 0.0)
        tl.store(row_logits + cols, grad_raw_logits, mask=mask)


def _calculate_sm103_token_chunk_size(token_count, vocab_size, element_size):
    max_tokens_per_chunk = max(1, _SM103_LOGITS_WORKSPACE_BYTES // (vocab_size * element_size))
    if token_count <= max_tokens_per_chunk:
        return token_count
    minimum_chunks = (token_count + max_tokens_per_chunk - 1) // max_tokens_per_chunk
    return (token_count + minimum_chunks - 1) // minimum_chunks


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


def _sm103_forward(hidden_states, vocab_weights, input_ids, temperature, ignore_index, return_entropy):
    token_count = hidden_states.shape[0]
    vocab_size = vocab_weights.shape[0]
    chunk_size = _calculate_sm103_token_chunk_size(
        token_count,
        vocab_size,
        hidden_states.element_size(),
    )
    nll = torch.empty(token_count, dtype=torch.float32, device=hidden_states.device)
    entropy = hidden_states.new_empty(token_count) if return_entropy else None
    lse = torch.empty(token_count, dtype=torch.float32, device=hidden_states.device)
    logits_workspace = torch.empty(
        min(token_count, chunk_size),
        vocab_size,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    dummy_entropy = hidden_states.new_empty(1)

    with device_context(hidden_states.device):
        for start in range(0, token_count, chunk_size):
            end = min(start + chunk_size, token_count)
            logits_chunk = logits_workspace[: end - start]
            torch.mm(hidden_states[start:end], vocab_weights.t(), out=logits_chunk)
            _scaled_cross_entropy_forward_kernel[(end - start,)](
                logits_chunk,
                input_ids[start:end],
                nll[start:end],
                entropy[start:end] if entropy is not None else dummy_entropy,
                lse[start:end],
                logits_chunk.stride(0),
                vocab_size,
                1.0 / temperature,
                ignore_index,
                BLOCK_SIZE=_SM103_BLOCK_SIZE,
                RETURN_ENTROPY=return_entropy,
                num_warps=_SM103_NUM_WARPS,
            )

    return nll, entropy, lse


def _sm103_backward(
    grad_nll,
    grad_entropy,
    hidden_states,
    vocab_weights,
    input_ids,
    lse,
    entropy,
    temperature,
    ignore_index,
    needs_input_grad,
    needs_weight_grad,
):
    token_count = hidden_states.shape[0]
    vocab_size = vocab_weights.shape[0]
    chunk_size = _calculate_sm103_token_chunk_size(
        token_count,
        vocab_size,
        hidden_states.element_size(),
    )
    logits_workspace = torch.empty(
        min(token_count, chunk_size),
        vocab_size,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    grad_hidden_states = torch.empty_like(hidden_states) if needs_input_grad else None
    grad_vocab_weights = torch.empty_like(vocab_weights) if needs_weight_grad else None
    dummy_f32 = torch.empty(1, dtype=torch.float32, device=hidden_states.device)
    dummy_input_dtype = hidden_states.new_empty(1)

    grad_nll = grad_nll.contiguous() if grad_nll is not None else None
    grad_entropy = grad_entropy.contiguous() if grad_entropy is not None else None

    with device_context(hidden_states.device):
        for start in range(0, token_count, chunk_size):
            end = min(start + chunk_size, token_count)
            hidden_chunk = hidden_states[start:end]
            grad_logits_chunk = logits_workspace[: end - start]
            torch.mm(hidden_chunk, vocab_weights.t(), out=grad_logits_chunk)
            _scaled_cross_entropy_backward_kernel[(end - start,)](
                grad_logits_chunk,
                input_ids[start:end],
                lse[start:end],
                entropy[start:end] if entropy is not None else dummy_input_dtype,
                grad_nll[start:end] if grad_nll is not None else dummy_f32,
                grad_entropy[start:end] if grad_entropy is not None else dummy_input_dtype,
                grad_logits_chunk.stride(0),
                vocab_size,
                1.0 / temperature,
                ignore_index,
                BLOCK_SIZE=_SM103_BLOCK_SIZE,
                HAS_NLL_GRAD=grad_nll is not None,
                HAS_ENTROPY_GRAD=grad_entropy is not None,
                num_warps=_SM103_NUM_WARPS,
            )

            if grad_hidden_states is not None:
                torch.mm(grad_logits_chunk, vocab_weights, out=grad_hidden_states[start:end])
            if grad_vocab_weights is not None:
                if start == 0:
                    torch.mm(grad_logits_chunk.t(), hidden_chunk, out=grad_vocab_weights)
                else:
                    torch.addmm(
                        grad_vocab_weights,
                        grad_logits_chunk.t(),
                        hidden_chunk,
                        out=grad_vocab_weights,
                    )

    return grad_hidden_states, grad_vocab_weights


class _FusedLinearScaledCrossEntropySM103Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, vocab_weights, input_ids, temperature, ignore_index, return_entropy):
        ctx.set_materialize_grads(False)
        _validate_fallback_inputs(hidden_states, vocab_weights, input_ids, ignore_index)
        _validate_temperature(temperature)

        input_ids = input_ids.contiguous()
        nll, entropy, lse = _sm103_forward(
            hidden_states,
            vocab_weights,
            input_ids,
            temperature,
            ignore_index,
            return_entropy,
        )
        saved_entropy = entropy if entropy is not None else hidden_states.new_empty(0)
        ctx.save_for_backward(hidden_states, vocab_weights, input_ids, lse, saved_entropy)
        ctx.temperature = temperature
        ctx.ignore_index = ignore_index
        ctx.return_entropy = return_entropy
        return (nll, entropy) if return_entropy else nll

    @staticmethod
    def backward(ctx, grad_nll, grad_entropy=None):
        hidden_states, vocab_weights, input_ids, lse, entropy = ctx.saved_tensors
        grad_hidden_states, grad_vocab_weights = _sm103_backward(
            grad_nll,
            grad_entropy,
            hidden_states,
            vocab_weights,
            input_ids,
            lse,
            entropy if ctx.return_entropy else None,
            ctx.temperature,
            ctx.ignore_index,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
        )
        return grad_hidden_states, grad_vocab_weights, None, None, None, None


def _device_id(device):
    return device.index if device.index is not None else torch.cuda.current_device()


def _resolve_implementation(device, dtype=None):
    if device.type != "cuda" or not torch.cuda.is_available() or torch.version.hip is not None:
        return _FusedLinearPPOFallbackFunction

    capability = torch.cuda.get_device_capability(device)
    if capability == (9, 0):
        return _load_sm90_function()
    if (
        capability == (10, 3)
        and dtype in (torch.float16, torch.bfloat16, torch.float32)
        and infer_device_arch(_device_id(device)) == "blackwell_ultra"
    ):
        return _FusedLinearScaledCrossEntropySM103Function
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

        implementation = _resolve_implementation(_input.device, _input.dtype)
        if implementation in (_FusedLinearPPOFallbackFunction, _FusedLinearScaledCrossEntropySM103Function):
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
