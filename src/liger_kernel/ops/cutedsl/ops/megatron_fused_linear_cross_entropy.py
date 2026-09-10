"""CuTe DSL tensor-parallel fused linear cross entropy for Megatron.

The SM100 path uses a persistent CuTe DSL GEMM for the local vocabulary
projection, Triton for vocabulary-parallel cross entropy, and NCCL for
tensor-parallel collectives. Backward converts the saved projection buffer to
dlogits in-place.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.nn.functional as F

from liger_kernel.ops.cutedsl.ops._sm100_gemm import K_ALIGNMENT
from liger_kernel.ops.cutedsl.ops._sm100_gemm import run_epilogue_gemm
from liger_kernel.ops.megatron_fused_linear_cross_entropy import _ce_backward_from_logits
from liger_kernel.ops.megatron_fused_linear_cross_entropy import _ce_forward_stats
from liger_kernel.ops.megatron_fused_linear_cross_entropy import _tp_rank_and_world
from liger_kernel.ops.megatron_fused_linear_cross_entropy import _validate_megatron_flce_inputs


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
        return torch.cuda.get_device_capability(hidden.device) == (10, 0)
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
    hidden, weight, logits, logits_max, sum_exp, target = ctx.saved_tensors
    grad_output_1d = grad_output.contiguous().reshape(-1).float()
    _ce_backward_from_logits(
        logits,
        logits_max,
        sum_exp,
        target,
        grad_output_1d,
        ctx.vocab_start,
        ctx.ignore_index,
        ctx.ce_block_size,
    )

    grad_hidden = torch.mm(logits, weight)
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
    grad_weight = logits.t() @ hidden
    grad_bias = logits.sum(dim=0, dtype=torch.float32).to(ctx.bias_dtype) if ctx.has_bias else None
    if reduce_work is not None:
        reduce_work.wait()

    grad_hidden = grad_hidden.reshape(ctx.original_hidden_shape)
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
        _validate_megatron_flce_inputs(hidden, weight, target, bias)
        if not _native_cutedsl_supported(hidden, weight):
            raise RuntimeError("CuTe DSL Megatron FLCE requires an SM100 GPU and float16 or bfloat16 inputs.")

        tp_rank, tp_world = _tp_rank_and_world(tp_group)
        vocab_local = weight.shape[0]
        vocab_global = vocab_local * tp_world
        vocab_start = tp_rank * vocab_local
        flat_target = target.reshape(-1).contiguous()
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

        from liger_kernel.ops.vocab_parallel_cross_entropy import _select_block_size

        ce_block_size = _select_block_size(vocab_local)
        stats = _ce_forward_stats(
            logits,
            logits_max,
            flat_target,
            vocab_start,
            ignore_index,
            ce_block_size,
        )
        predicted_logit = stats[0]
        sum_exp = stats[1]
        if tp_world > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=tp_group)

        loss = torch.log(sum_exp) - predicted_logit
        loss = torch.where(valid, loss, torch.zeros_like(loss))
        ctx.save_for_backward(hidden_2d, weight_2d, logits, logits_max, sum_exp, flat_target)
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
    return LigerMegatronFusedLinearCrossEntropyFunction.apply(
        hidden,
        weight,
        target,
        bias,
        tp_group,
        ignore_index,
    )
