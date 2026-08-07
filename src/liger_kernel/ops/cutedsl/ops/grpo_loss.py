import torch

from liger_kernel.ops.cutedsl.ops._sm100_gemm import K_ALIGNMENT
from liger_kernel.ops.cutedsl.ops._sm100_gemm import run_epilogue_gemm
from liger_kernel.ops.cutedsl.ops.cross_entropy import _launch_ce_fwd
from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import _accum_grad_weight
from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import _identity_epilogue
from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import _mm_out
from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import _native_sm100_supported
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd

_MAX_LOGITS_CHUNK_SIZE = 1024


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


def _logits_storage(token_count, vocab_size, device, dtype):
    chunk_size = min(token_count, _MAX_LOGITS_CHUNK_SIZE)
    vector_width = 16 // dtype.itemsize
    storage_width = ((vocab_size + vector_width - 1) // vector_width) * vector_width
    return torch.empty(chunk_size, storage_width, device=device, dtype=dtype), chunk_size


def _fill_logits(storage, x, weight, bias):
    rows = x.shape[0]
    vocab_size = weight.shape[0]
    logits = storage[:rows, :vocab_size]
    run_epilogue_gemm(x, weight, logits, _identity_epilogue)
    if bias is not None:
        logits.add_(bias)
    if storage.shape[1] != vocab_size:
        storage[:rows, vocab_size:].zero_()
    return storage[:rows]


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
        native_bias = bias.detach().contiguous() if bias is not None else None

        token_count = x.shape[0]
        vocab_size = w.shape[0]
        logits_storage, chunk_size = _logits_storage(token_count, vocab_size, x.device, x.dtype)
        logp = torch.empty(token_count, device=x.device, dtype=torch.float32)
        for start in range(0, token_count, chunk_size):
            end = min(start + chunk_size, token_count)
            logits = _fill_logits(logits_storage, x[start:end], w, native_bias)
            _launch_ce_fwd(
                logits,
                target[start:end],
                logp[start:end],
                -1.0,
                -100,
                False,
                logical_vocab_size=vocab_size,
            )

        saved_bias = native_bias if native_bias is not None else x.new_empty(0)
        ctx.save_for_backward(x, w, target, saved_bias)
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return logp

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_logp):
        x, w, target, bias = ctx.saved_tensors
        grad_logp = grad_logp.reshape(-1).float().contiguous()
        token_count, hidden_size = x.shape
        vocab_size = w.shape[0]
        needs_input, needs_weight, _, needs_bias, _ = ctx.needs_input_grad
        logits_storage, chunk_size = _logits_storage(token_count, vocab_size, x.device, x.dtype)
        loss = torch.empty(chunk_size, device=x.device, dtype=torch.float32)
        grad_input = torch.empty_like(x) if needs_input else None
        grad_weight = torch.empty_like(w) if needs_weight else None
        grad_bias_acc = (
            torch.zeros(vocab_size, device=x.device, dtype=torch.float32) if ctx.has_bias and needs_bias else None
        )

        for start in range(0, token_count, chunk_size):
            end = min(start + chunk_size, token_count)
            x_chunk = x[start:end]
            dlogits = _fill_logits(logits_storage, x_chunk, w, bias if ctx.has_bias else None)
            _launch_ce_fwd(
                dlogits,
                target[start:end],
                loss[: end - start],
                -1.0,
                -100,
                True,
                logical_vocab_size=vocab_size,
            )
            dlogits = dlogits[:, :vocab_size]
            dlogits.mul_(grad_logp[start:end, None])
            if grad_input is not None:
                _mm_out(grad_input[start:end], dlogits, w)
            if grad_weight is not None:
                if start == 0:
                    _mm_out(grad_weight, dlogits.t(), x_chunk)
                else:
                    _accum_grad_weight(grad_weight, dlogits.t(), x_chunk)
            if grad_bias_acc is not None:
                grad_bias_acc.add_(dlogits.sum(0, dtype=torch.float32))

        grad_bias = grad_bias_acc.to(ctx.bias_dtype) if grad_bias_acc is not None else None

        if ctx.h_orig is not None:
            if grad_input is not None:
                grad_input = grad_input[:, : ctx.h_orig].contiguous()
            if grad_weight is not None:
                grad_weight = grad_weight[:, : ctx.h_orig].contiguous()
        return grad_input, grad_weight, None, grad_bias, None


def fused_linear_selective_logprob(_input, weight, target, bias=None, temperature=1.0):
    if not native_fused_linear_selective_logprob_supported(_input, weight, temperature):
        from liger_kernel.ops.grpo_loss import fused_linear_selective_logprob as default_selective_logprob

        return default_selective_logprob(_input, weight, target, bias, temperature)
    return LigerFusedLinearSelectiveLogProbFunction.apply(_input, weight, target, bias, temperature)
