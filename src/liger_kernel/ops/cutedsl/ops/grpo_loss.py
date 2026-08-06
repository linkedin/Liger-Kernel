import torch

from liger_kernel.ops.cutedsl.ops._sm100_gemm import K_ALIGNMENT
from liger_kernel.ops.cutedsl.ops._sm100_gemm import row_logsumexp
from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import _addmm_fp32_out
from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import _bwd_chunk_size
from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import _dx_correct
from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import _mm_out
from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import _native_sm100_supported
from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import _recompute_softmax
from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import _scatter_target_grad_rowscaled
from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import _target_logit
from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import _vocab_chunk_size
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd


def native_fused_linear_selective_logprob_supported(_input, weight, temperature=1.0):
    """Return whether the SM100 selective-logprob path supports these inputs."""
    return (
        _input.ndim == 2
        and weight.ndim == 2
        and _input.shape[0] > 0
        and _input.shape[1] > 0
        and weight.shape[0] > 0
        and _input.shape[1] == weight.shape[1]
        and _input.dtype == weight.dtype
        and _input.dtype in (torch.bfloat16, torch.float16)
        and isinstance(temperature, (int, float))
        and temperature == 1.0
        and _native_sm100_supported(_input)
    )


def _validate_inputs(_input, weight, target, bias, temperature):
    if _input.ndim != 2 or weight.ndim != 2:
        raise ValueError(f"_input and weight must be 2D, got {_input.shape} and {weight.shape}.")
    if target.ndim != 1 or target.shape[0] != _input.shape[0]:
        raise ValueError(f"target must have shape ({_input.shape[0]},), got {target.shape}.")
    if _input.shape[1] != weight.shape[1]:
        raise ValueError(f"Input and weight hidden dimensions must match, got {_input.shape[1]} and {weight.shape[1]}.")
    if _input.shape[0] == 0 or _input.shape[1] == 0 or weight.shape[0] == 0:
        raise ValueError("CuTe DSL selective logprob requires non-empty token, hidden, and vocabulary dimensions.")
    if _input.device != weight.device or _input.device != target.device:
        raise ValueError(
            f"_input, weight, and target must share a device, got {_input.device}, {weight.device}, and {target.device}."
        )
    if _input.dtype != weight.dtype or _input.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"_input and weight must share a BF16 or FP16 dtype, got {_input.dtype} and {weight.dtype}.")
    if target.dtype != torch.long:
        raise TypeError(f"target must have dtype torch.long, got {target.dtype}.")
    if temperature != 1.0:
        raise ValueError(f"CuTe DSL selective logprob currently requires temperature=1.0, got {temperature}.")
    if not _native_sm100_supported(_input):
        raise RuntimeError("CuTe DSL selective logprob requires an NVIDIA SM100 GPU.")
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise ValueError(f"bias must have shape ({weight.shape[0]},), got {bias.shape}.")
        if bias.device != _input.device or not torch.is_floating_point(bias):
            raise TypeError("bias must be a floating-point tensor on the same device as _input.")


class LigerFusedLinearSelectiveLogProbFunction(torch.autograd.Function):
    """SM100 fused-linear selected-token log probabilities for GRPO-style losses."""

    @staticmethod
    @amp_custom_fwd
    def forward(ctx, _input, weight, target, bias=None, temperature=1.0):
        _validate_inputs(_input, weight, target, bias, temperature)
        vocab_size = weight.shape[0]
        if target.numel() > 0 and (target.min() < 0 or target.max() >= vocab_size):
            raise AssertionError(f"Target out of bounds. Expected values in [0, {vocab_size}).")

        hidden_size = weight.shape[1]
        ctx.h_orig = None
        if hidden_size % K_ALIGNMENT != 0:
            pad = (-hidden_size) % K_ALIGNMENT
            _input = torch.nn.functional.pad(_input, (0, pad))
            weight = torch.nn.functional.pad(weight, (0, pad))
            ctx.h_orig = hidden_size

        x = _input.detach().contiguous()
        w = weight.detach().contiguous()
        target = target.detach().contiguous()
        bias_f = bias.detach().float().contiguous() if bias is not None else None

        lse = row_logsumexp(x, w, bias_f)
        target_logit = _target_logit(x, w, target)
        if bias_f is not None:
            target_logit = target_logit + bias_f[target]
        logp = target_logit - lse

        saved_bias = bias_f if bias_f is not None else x.new_empty(0, dtype=torch.float32)
        ctx.save_for_backward(x, w, target, lse, saved_bias)
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return logp

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_logp):
        x, w, target, lse, bias_f = ctx.saved_tensors
        grad_logp = grad_logp.reshape(-1).float()
        token_count, hidden_size = x.shape
        vocab_size = w.shape[0]
        token_chunk = _bwd_chunk_size(token_count, hidden_size, vocab_size)
        vocab_chunk = _vocab_chunk_size(vocab_size, token_count, token_chunk)
        col_bias = -lse

        grad_input_softmax = torch.zeros(
            token_count,
            hidden_size,
            device=x.device,
            dtype=torch.float32,
        )
        grad_weight = torch.empty_like(w)
        grad_bias = torch.empty(vocab_size, device=x.device, dtype=torch.float32) if ctx.has_bias else None

        for start in range(0, vocab_size, vocab_chunk):
            end = min(start + vocab_chunk, vocab_size)
            weight_slice = w[start:end]
            vocab_bias = bias_f[start:end].reshape(1, -1) if ctx.has_bias else None
            softmax = _recompute_softmax(x, weight_slice, col_bias, vocab_bias)
            _addmm_fp32_out(grad_input_softmax, softmax, weight_slice)
            softmax.mul_(-grad_logp[:, None])
            _mm_out(grad_weight[start:end], softmax.t(), x)
            if grad_bias is not None:
                grad_bias[start:end] = softmax.sum(0, dtype=torch.float32)

        del softmax
        valid = torch.ones_like(target, dtype=torch.bool)
        grad_input = _dx_correct(
            grad_input_softmax,
            w[target],
            valid,
            -grad_logp[:, None],
            x.dtype,
        )
        _scatter_target_grad_rowscaled(
            grad_weight,
            x,
            target,
            valid,
            grad_logp,
            1.0,
        )
        if grad_bias is not None:
            grad_bias.index_add_(0, target, grad_logp)
            grad_bias = grad_bias.to(ctx.bias_dtype)

        if ctx.h_orig is not None:
            grad_input = grad_input[:, : ctx.h_orig].contiguous()
            grad_weight = grad_weight[:, : ctx.h_orig].contiguous()
        return grad_input, grad_weight, None, grad_bias, None


def fused_linear_selective_logprob(_input, weight, target, bias=None, temperature=1.0):
    if not native_fused_linear_selective_logprob_supported(_input, weight, temperature):
        from liger_kernel.ops.grpo_loss import fused_linear_selective_logprob as default_selective_logprob

        return default_selective_logprob(_input, weight, target, bias, temperature)
    return LigerFusedLinearSelectiveLogProbFunction.apply(_input, weight, target, bias, temperature)
