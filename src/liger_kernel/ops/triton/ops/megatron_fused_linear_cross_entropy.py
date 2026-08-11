"""Portable all-Triton tensor-parallel fused linear cross entropy for Megatron.

Each tensor-parallel rank owns a contiguous vocabulary shard. Forward performs
one Triton projection GEMM, computes globally normalized cross entropy, and
saves shifted exponentials in the projection dtype. Backward converts that
buffer to dlogits in-place before Triton dX and dW GEMMs. Tensor-parallel
collectives remain NCCL/RCCL calls between architecture-independent kernels.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import triton
import triton.language as tl


def _matmul_autotune_configs():
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4},
            num_stages=3,
            num_warps=4,
        ),
    ]


def _split_k_matmul_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "SPLIT_K": 2,
            },
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "SPLIT_K": 4,
            },
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "SPLIT_K": 8,
            },
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "SPLIT_K": 4,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "SPLIT_K": 4,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "SPLIT_K": 4,
            },
            num_stages=3,
            num_warps=8,
        ),
    ]


@triton.autotune(configs=_matmul_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k_start * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        accumulator += bias[None, :]

    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(output_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.autotune(
    configs=_split_k_matmul_autotune_configs(),
    key=["M", "N", "K"],
    reset_to_zero=["output_ptr"],
)
@triton.jit
def _split_k_matmul_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    split_k_id = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = split_k_id * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        k_remaining = K - k_start * BLOCK_SIZE_K * SPLIT_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.atomic_add(output_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _row_max_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    input_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_ptr = input_ptr + row * input_row_stride
    row_max = -float("inf")
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        values = tl.load(row_ptr + offsets, mask=offsets < n_cols, other=-float("inf")).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(values))
    tl.store(output_ptr + row, row_max)


@triton.jit
def _loss_kernel(
    sum_exp_ptr,
    predicted_logit_ptr,
    target_ptr,
    loss_ptr,
    n_rows,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_rows
    sum_exp = tl.load(sum_exp_ptr + offsets, mask=mask, other=1.0)
    predicted_logit = tl.load(predicted_logit_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=ignore_index)
    loss = tl.log(sum_exp) - predicted_logit
    loss = tl.where(target == ignore_index, 0.0, loss)
    tl.store(loss_ptr + offsets, loss, mask=mask)


@triton.jit
def _column_sum_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    input_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    col = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    total = 0.0
    for start in range(0, n_rows, BLOCK_SIZE):
        rows = start + offsets
        values = tl.load(input_ptr + rows * input_row_stride + col, mask=rows < n_rows, other=0.0)
        total += tl.sum(values.to(tl.float32))
    tl.store(output_ptr + col, total)


def _triton_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul expects [M, K] @ [K, N], got {tuple(a.shape)} and {tuple(b.shape)}.")
    m, k = a.shape
    n = b.shape[1]
    output = torch.empty((m, n), device=a.device, dtype=output_dtype or a.dtype)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),)
    _matmul_kernel[grid](
        a,
        b,
        bias if bias is not None else output,
        output,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
        HAS_BIAS=bias is not None,
    )
    return output


def _triton_dx_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    if m > 1024 or k < 4096:
        return _triton_matmul(a, b, output_dtype=torch.float32)

    output = torch.zeros((m, n), device=a.device, dtype=torch.float32)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
        meta["SPLIT_K"],
    )
    _split_k_matmul_kernel[grid](
        a,
        b,
        output,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
    )
    return output


def _triton_row_max(input: torch.Tensor, block_size: int) -> torch.Tensor:
    from liger_kernel.ops.vocab_parallel_cross_entropy import _get_num_warps

    output = torch.empty(input.shape[0], device=input.device, dtype=torch.float32)
    _row_max_kernel[(input.shape[0],)](
        input,
        output,
        input.shape[1],
        input.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=_get_num_warps(block_size),
    )
    return output


def _triton_loss(
    sum_exp: torch.Tensor,
    predicted_logit: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    output = torch.empty_like(sum_exp)
    block_size = 256
    _loss_kernel[(triton.cdiv(target.numel(), block_size),)](
        sum_exp,
        predicted_logit,
        target,
        output,
        target.numel(),
        ignore_index,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output


def _triton_column_sum(input: torch.Tensor) -> torch.Tensor:
    from liger_kernel.ops.vocab_parallel_cross_entropy import _get_num_warps
    from liger_kernel.ops.vocab_parallel_cross_entropy import _select_block_size

    block_size = _select_block_size(input.shape[0])
    output = torch.empty(input.shape[1], device=input.device, dtype=torch.float32)
    _column_sum_kernel[(input.shape[1],)](
        input,
        output,
        input.shape[0],
        input.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=_get_num_warps(block_size),
    )
    return output


def _tp_rank_and_world(tp_group) -> tuple[int, int]:
    if tp_group is None:
        return 0, 1
    world = dist.get_world_size(tp_group)
    if world == 1:
        return 0, 1
    return dist.get_rank(tp_group), world


def _materialized_backward(ctx, grad_output: torch.Tensor):
    """Convert saved CE state to dlogits and form projection gradients."""
    from liger_kernel.ops.vocab_parallel_cross_entropy import _get_num_warps
    from liger_kernel.ops.vocab_parallel_cross_entropy import liger_vocab_parallel_ce_backward_kernel

    hidden, weight, exp_buf, sum_exp_global, target = ctx.saved_tensors
    grad_out = grad_output.contiguous().reshape(-1).float()
    num_warps = _get_num_warps(ctx.ce_block_size)
    liger_vocab_parallel_ce_backward_kernel[(hidden.shape[0],)](
        EXP_ptr=exp_buf,
        EXP_stride=exp_buf.stride(0),
        sum_exp_ptr=sum_exp_global,
        Y_ptr=target,
        grad_out_ptr=grad_out,
        vocab_start=ctx.vocab_start,
        n_cols=weight.shape[0],
        ignore_index=ctx.ignore_index,
        alpha_eff=0.0,
        eps_eff=0.0,
        HAS_LABEL_SMOOTHING=False,
        BLOCK_SIZE=ctx.ce_block_size,
        num_warps=num_warps,
    )

    grad_hidden = _triton_dx_matmul(exp_buf, weight)
    grad_weight = _triton_matmul(exp_buf.t(), hidden)
    grad_bias = _triton_column_sum(exp_buf).to(ctx.bias_dtype) if ctx.has_bias else None

    if ctx.tp_world > 1:
        dist.all_reduce(grad_hidden, op=dist.ReduceOp.SUM, group=ctx.tp_group)
    grad_hidden = grad_hidden.to(ctx.hidden_dtype).reshape(ctx.original_hidden_shape)
    return grad_hidden, grad_weight, grad_bias


class LigerMegatronFusedLinearCrossEntropyFunction(torch.autograd.Function):
    """Hidden-to-loss tensor-parallel FLCE with saved low-precision CE state."""

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
        if hidden.device.type != "cuda" or hidden.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError("Megatron FLCE requires a CUDA GPU and float16 or bfloat16 inputs.")

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

        from liger_kernel.ops.vocab_parallel_cross_entropy import _get_num_warps
        from liger_kernel.ops.vocab_parallel_cross_entropy import _select_block_size
        from liger_kernel.ops.vocab_parallel_cross_entropy import liger_vocab_parallel_ce_forward_kernel

        logits = _triton_matmul(hidden_2d, weight_2d.t(), bias=bias_1d)
        ce_block_size = _select_block_size(vocab_local)
        logits_max = _triton_row_max(logits, ce_block_size)
        if tp_world > 1:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        exp_buf = torch.empty(
            hidden_2d.shape[0],
            vocab_local,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        predicted_logit = torch.empty(hidden_2d.shape[0], device=hidden.device, dtype=torch.float32)
        sum_exp = torch.empty_like(predicted_logit)
        num_warps = _get_num_warps(ce_block_size)
        liger_vocab_parallel_ce_forward_kernel[(hidden_2d.shape[0],)](
            X_ptr=logits,
            X_stride=logits.stride(0),
            EXP_ptr=exp_buf,
            EXP_stride=exp_buf.stride(0),
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
            dist.all_reduce(predicted_logit, op=dist.ReduceOp.SUM, group=tp_group)
            dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)

        loss = _triton_loss(sum_exp, predicted_logit, flat_target, ignore_index)

        ctx.save_for_backward(hidden_2d, weight_2d, exp_buf, sum_exp, flat_target)
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
    """Compute per-token loss from replicated hidden states and a local vocab shard."""
    return LigerMegatronFusedLinearCrossEntropyFunction.apply(
        hidden,
        weight,
        target,
        bias,
        tp_group,
        ignore_index,
    )
