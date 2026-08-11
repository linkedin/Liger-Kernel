"""CuTe DSL tensor-parallel fused linear cross entropy for Megatron.

The SM100 path uses a persistent CuTe DSL GEMM for the local vocabulary
projection, Triton for vocabulary-parallel cross entropy, and NCCL for
tensor-parallel collectives. Shifted exponentials overwrite the projection
buffer and are reused by backward.
"""

from __future__ import annotations

import operator

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.nn.functional as F

from liger_kernel.ops.cutedsl.ops._sm100_gemm import K_ALIGNMENT
from liger_kernel.ops.cutedsl.ops._sm100_gemm import run_epilogue_gemm
from liger_kernel.ops.megatron_fused_linear_cross_entropy import _tp_rank_and_world
from liger_kernel.ops.megatron_fused_linear_cross_entropy import (
    liger_megatron_fused_linear_cross_entropy as default_megatron_fused_linear_cross_entropy,
)
from liger_kernel.ops.utils import compare_version

_SUPPORTS_OUT_DTYPE = compare_version("torch", operator.ge, "2.8.0")


@cute.jit
def _identity_epilogue(accumulator, output):
    output_dtype = output.element_type
    for element in cutlass.range_constexpr(cute.size(accumulator)):
        output[element] = accumulator[element].to(output_dtype)


def _native_cutedsl_supported(hidden: torch.Tensor, weight: torch.Tensor) -> bool:
    if hidden.device.type != "cuda" or hidden.dtype not in (torch.bfloat16, torch.float16):
        return False
    if weight.device != hidden.device or weight.dtype != hidden.dtype:
        return False
    try:
        return torch.cuda.get_device_capability(hidden.device)[0] >= 10
    except (AssertionError, RuntimeError):
        return False


def _cutedsl_projection(
    hidden: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    padding = (-hidden.shape[1]) % K_ALIGNMENT
    if padding:
        hidden = F.pad(hidden, (0, padding))
        weight = F.pad(weight, (0, padding))
    logits = torch.empty(
        hidden.shape[0],
        weight.shape[0],
        device=hidden.device,
        dtype=hidden.dtype,
    )
    run_epilogue_gemm(hidden, weight, logits, _identity_epilogue)
    return logits


def _materialized_backward(ctx, grad_output: torch.Tensor):
    from liger_kernel.ops.vocab_parallel_cross_entropy import _get_num_warps
    from liger_kernel.ops.vocab_parallel_cross_entropy import liger_vocab_parallel_ce_backward_kernel

    hidden, weight, exp_buffer, sum_exp, target = ctx.saved_tensors
    grad_output_1d = grad_output.contiguous().reshape(-1).float()
    num_warps = _get_num_warps(ctx.ce_block_size)
    liger_vocab_parallel_ce_backward_kernel[(hidden.shape[0],)](
        EXP_ptr=exp_buffer,
        EXP_stride=exp_buffer.stride(0),
        sum_exp_ptr=sum_exp,
        Y_ptr=target,
        grad_out_ptr=grad_output_1d,
        vocab_start=ctx.vocab_start,
        n_cols=weight.shape[0],
        ignore_index=ctx.ignore_index,
        alpha_eff=0.0,
        eps_eff=0.0,
        HAS_LABEL_SMOOTHING=False,
        BLOCK_SIZE=ctx.ce_block_size,
        num_warps=num_warps,
    )

    if _SUPPORTS_OUT_DTYPE:
        grad_hidden = torch.mm(exp_buffer, weight, out_dtype=torch.float32)
    else:
        grad_hidden = exp_buffer.float() @ weight.float()
    reduce_work = (
        dist.all_reduce(
            grad_hidden,
            op=dist.ReduceOp.SUM,
            group=ctx.tp_group,
            async_op=True,
        )
        if ctx.tp_world > 1
        else None
    )
    grad_weight = exp_buffer.t() @ hidden
    grad_bias = exp_buffer.sum(dim=0, dtype=torch.float32).to(ctx.bias_dtype) if ctx.has_bias else None
    if reduce_work is not None:
        reduce_work.wait()

    grad_hidden = grad_hidden.to(ctx.hidden_dtype).reshape(ctx.original_hidden_shape)
    return grad_hidden, grad_weight, grad_bias


class LigerMegatronFusedLinearCrossEntropyFunction(torch.autograd.Function):
    """Megatron FLCE using a persistent CuTe DSL SM100 projection."""

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: torch.Tensor | None,
        tp_group,
        ignore_index: int,
    ) -> torch.Tensor:
        if hidden.ndim < 2:
            raise ValueError(f"hidden must have at least 2 dimensions, got shape {tuple(hidden.shape)}.")
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2-D [V_local, H], got shape {tuple(weight.shape)}.")
        if tuple(target.shape) != tuple(hidden.shape[:-1]):
            raise ValueError(
                f"target shape must equal hidden.shape[:-1]; got target={tuple(target.shape)}, "
                f"hidden={tuple(hidden.shape)}."
            )
        if hidden.shape[-1] != weight.shape[1]:
            raise ValueError(f"hidden size mismatch: hidden has H={hidden.shape[-1]}, weight has H={weight.shape[1]}.")
        if hidden.dtype != weight.dtype:
            raise TypeError(f"hidden and weight must have the same dtype, got {hidden.dtype} and {weight.dtype}.")
        if hidden.device != weight.device or hidden.device != target.device:
            raise ValueError("hidden, weight, and target must be on the same device.")
        if bias is not None:
            if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
                raise ValueError(f"bias must have shape ({weight.shape[0]},), got {tuple(bias.shape)}.")
            if bias.device != hidden.device or bias.dtype != hidden.dtype:
                raise TypeError("bias must have the same device and dtype as hidden.")

        tp_rank, tp_world = _tp_rank_and_world(tp_group)
        vocab_local = weight.shape[0]
        vocab_global = vocab_local * tp_world
        vocab_start = tp_rank * vocab_local
        flat_target = target.reshape(-1).to(torch.int64).contiguous()
        valid = flat_target != ignore_index
        invalid = valid & ((flat_target < 0) | (flat_target >= vocab_global))
        valid_targets = ~torch.any(invalid)
        if hasattr(torch, "_assert_async"):
            torch._assert_async(valid_targets, f"non-ignored targets must be in [0, {vocab_global}).")
        elif not valid_targets.item():
            raise ValueError(f"non-ignored targets must be in [0, {vocab_global}).")

        original_hidden_shape = hidden.shape
        hidden_2d = hidden.reshape(-1, hidden.shape[-1]).contiguous()
        weight_2d = weight.contiguous()
        bias_1d = bias.contiguous() if bias is not None else None
        logits = _cutedsl_projection(hidden_2d, weight_2d)
        if bias_1d is not None:
            logits.add_(bias_1d)

        logits_max = logits.amax(dim=-1).float()
        if tp_world > 1:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        from liger_kernel.ops.vocab_parallel_cross_entropy import _get_num_warps
        from liger_kernel.ops.vocab_parallel_cross_entropy import _select_block_size
        from liger_kernel.ops.vocab_parallel_cross_entropy import liger_vocab_parallel_ce_forward_kernel

        exp_buffer = logits
        stats = torch.empty((2, hidden_2d.shape[0]), device=hidden.device, dtype=torch.float32)
        predicted_logit = stats[0]
        sum_exp = stats[1]
        ce_block_size = _select_block_size(vocab_local)
        num_warps = _get_num_warps(ce_block_size)
        liger_vocab_parallel_ce_forward_kernel[(hidden_2d.shape[0],)](
            X_ptr=logits,
            X_stride=logits.stride(0),
            EXP_ptr=exp_buffer,
            EXP_stride=exp_buffer.stride(0),
            logits_max_ptr=logits_max,
            Y_ptr=flat_target,
            pred_ptr=predicted_logit,
            sum_exp_ptr=sum_exp,
            vocab_start=vocab_start,
            n_cols=vocab_local,
            ignore_index=ignore_index,
            BLOCK_SIZE=ce_block_size,
            num_warps=num_warps,
        )
        if tp_world > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=tp_group)

        loss = torch.log(sum_exp) - predicted_logit
        loss = torch.where(valid, loss, torch.zeros_like(loss))
        ctx.save_for_backward(hidden_2d, weight_2d, exp_buffer, sum_exp, flat_target)
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.tp_group = tp_group
        ctx.tp_world = tp_world
        ctx.vocab_start = vocab_start
        ctx.ignore_index = ignore_index
        ctx.ce_block_size = ce_block_size
        ctx.original_hidden_shape = original_hidden_shape
        ctx.hidden_dtype = hidden.dtype
        return loss.reshape(target.shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_hidden, grad_weight, grad_bias = _materialized_backward(ctx, grad_output)
        return grad_hidden, grad_weight, None, grad_bias, None, None


def liger_megatron_fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor | None = None,
    tp_group=None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute Megatron FLCE with a CuTe DSL projection and NCCL TP collectives."""
    if not _native_cutedsl_supported(hidden, weight):
        return default_megatron_fused_linear_cross_entropy(
            hidden,
            weight,
            target,
            bias=bias,
            tp_group=tp_group,
            ignore_index=ignore_index,
        )
    return LigerMegatronFusedLinearCrossEntropyFunction.apply(
        hidden,
        weight,
        target,
        bias,
        tp_group,
        ignore_index,
    )
