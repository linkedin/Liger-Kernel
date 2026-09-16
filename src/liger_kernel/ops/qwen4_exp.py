"""Qwen4Exp text-specific n-gram and hyper-connection kernels."""

import math

import torch
import triton
import triton.language as tl

from liger_kernel.ops.rms_norm import _rms_group_norm_backward_row
from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import device_context
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_device_multiprocessor_count
from liger_kernel.ops.utils import is_hip
from liger_kernel.ops.utils import torch_to_triton_dtype
from liger_kernel.utils import infer_device_arch


@triton.jit
def _qwen4_exp_ngram_hash_kernel(
    previous_context,
    previous_context_row_stride,
    input_ids,
    input_ids_row_stride,
    multipliers,
    vocab_sizes,
    head_offsets,
    output,
    output_token_stride,
    seq_len,
    context_len: tl.constexpr,
    eos_token_id: tl.constexpr,
    heads_per_ngram: tl.constexpr,
    ngram_size: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    token_offsets = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    token_mask = token_offsets < seq_len
    base_token = tl.load(
        input_ids + batch_idx * input_ids_row_stride + token_offsets,
        mask=token_mask,
        other=eos_token_id,
    ).to(tl.int64)
    mix_prefix = base_token * tl.load(multipliers).to(tl.int64)
    blocked_by_eos = tl.zeros((BLOCK_T,), dtype=tl.int1)

    # Construct each shifted token directly from the cached context while
    # preserving Hugging Face's EOS-delimited segment semantics.
    for current_ngram_size in tl.static_range(2, ngram_size + 1):
        shift = current_ngram_size - 1
        history_offsets = context_len + token_offsets - shift
        from_context = history_offsets < context_len
        context_token = tl.load(
            previous_context + batch_idx * previous_context_row_stride + history_offsets,
            mask=token_mask & from_context,
            other=eos_token_id,
        ).to(tl.int64)
        input_token = tl.load(
            input_ids + batch_idx * input_ids_row_stride + history_offsets - context_len,
            mask=token_mask & ~from_context,
            other=eos_token_id,
        ).to(tl.int64)
        shifted_token = tl.where(from_context, context_token, input_token)

        # Accumulating the boundary flag from nearest to farthest exactly matches
        # HF: the current token is included, while any earlier EOS blocks all
        # tokens at or before that boundary.
        blocked_by_eos |= shifted_token == eos_token_id
        shifted_token = tl.where(blocked_by_eos, eos_token_id, shifted_token)
        mix_prefix ^= shifted_token * tl.load(multipliers + shift).to(tl.int64)

        head_start = (current_ngram_size - 2) * heads_per_ngram
        for head_offset in tl.static_range(0, heads_per_ngram):
            head_idx = head_start + head_offset
            vocab_size = tl.load(vocab_sizes + head_idx).to(tl.int64)
            # Triton's signed integer `%` can return a negative remainder after
            # Qwen4's intentional int64 overflow; torch.remainder is non-negative.
            hashed_id = ((mix_prefix % vocab_size) + vocab_size) % vocab_size
            hashed_id += tl.load(head_offsets + head_idx).to(tl.int64)
            tl.store(
                output + (batch_idx * seq_len + token_offsets) * output_token_stride + head_idx,
                hashed_id,
                mask=token_mask,
            )


def qwen4_exp_ngram_hash(previous_context, input_ids, multipliers, vocab_sizes, offsets, eos_token_id):
    """Build EOS-aware Qwen4Exp PLE n-gram embedding IDs on NVIDIA CUDA.

    ``previous_context`` contains the cached tokens immediately preceding
    ``input_ids``. The returned int64 tensor has shape
    ``[batch, sequence, ngram_heads]`` and already includes each head's embedding
    table offset.
    """
    if previous_context.ndim != 2 or input_ids.ndim != 2:
        raise ValueError(
            "previous_context and input_ids must both be [batch, sequence], "
            f"got {tuple(previous_context.shape)} and {tuple(input_ids.shape)}"
        )
    if previous_context.shape[0] != input_ids.shape[0]:
        raise ValueError("previous_context and input_ids must have the same batch size")
    if previous_context.device != input_ids.device:
        raise ValueError("previous_context and input_ids must be on the same device")
    if input_ids.device.type != "cuda" or is_hip():
        raise ValueError("Qwen4Exp n-gram hashing requires NVIDIA CUDA tensors; ROCm is not supported")
    if eos_token_id is None:
        raise ValueError("eos_token_id must be set when Qwen4Exp PLE is enabled")
    tensors = (previous_context, input_ids, multipliers, vocab_sizes, offsets)
    if any(tensor.dtype != torch.long for tensor in tensors):
        raise TypeError("Qwen4Exp n-gram token and metadata tensors must use torch.long")
    ngram_size = previous_context.shape[-1] + 1
    n_heads = vocab_sizes.numel()
    if ngram_size < 2:
        raise ValueError(f"ngram_size must be at least 2, got {ngram_size}")
    if n_heads % (ngram_size - 1) != 0:
        raise ValueError(f"n_heads={n_heads} must be divisible by ngram_size - 1={ngram_size - 1}")
    if offsets.numel() != n_heads:
        raise ValueError(f"offsets must contain {n_heads} entries, got {offsets.numel()}")
    if multipliers.numel() < ngram_size:
        raise ValueError(f"multipliers must contain at least {ngram_size} entries, got {multipliers.numel()}")

    if any(tensor.device != input_ids.device for tensor in (multipliers, vocab_sizes, offsets)):
        raise ValueError("hash metadata and token tensors must be on the same device")

    batch_size, seq_len = input_ids.shape
    output = torch.empty((batch_size, seq_len, n_heads), device=input_ids.device, dtype=torch.long)
    if batch_size == 0 or seq_len == 0:
        return output

    previous_context = previous_context.contiguous()
    input_ids = input_ids.contiguous()
    multipliers = multipliers.contiguous()
    vocab_sizes = vocab_sizes.contiguous()
    offsets = offsets.contiguous()
    heads_per_ngram = n_heads // (ngram_size - 1)
    block_t = 64
    device_index = input_ids.device.index
    architecture = infer_device_arch(0 if device_index is None else device_index)
    num_stages = 4 if architecture.startswith("blackwell") else 3 if architecture == "hopper" else 2
    with device_context(input_ids.device):
        _qwen4_exp_ngram_hash_kernel[(batch_size, triton.cdiv(seq_len, block_t))](
            previous_context,
            previous_context.stride(0),
            input_ids,
            input_ids.stride(0),
            multipliers,
            vocab_sizes,
            offsets,
            output,
            output.stride(-2),
            seq_len,
            context_len=previous_context.shape[1],
            eos_token_id=int(eos_token_id),
            heads_per_ngram=heads_per_ngram,
            ngram_size=ngram_size,
            BLOCK_T=block_t,
            num_warps=4,
            num_stages=num_stages,
        )
    return output


@triton.jit
def _qwen4_exp_hyper_connection_pre_forward_kernel(
    output,
    output_row_stride,
    mix_logits,
    mix_logits_row_stride,
    normalized_input,
    normalized_input_row_stride,
    hidden_size,
    n_groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    mixed = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for group_idx in tl.static_range(n_groups):
        group_offsets = group_idx * hidden_size + offsets
        logits = tl.load(mix_logits + row_idx * mix_logits_row_stride + group_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        values = tl.load(normalized_input + row_idx * normalized_input_row_stride + group_offsets, mask=mask, other=0.0)
        mixed += tl.sigmoid(logits) * values
    tl.store(output + row_idx * output_row_stride + offsets, mixed / n_groups, mask=mask)


@triton.jit
def _qwen4_exp_hyper_connection_pre_backward_kernel(
    grad_output,
    grad_output_row_stride,
    mix_logits,
    mix_logits_row_stride,
    normalized_input,
    normalized_input_row_stride,
    grad_mix_logits,
    grad_mix_logits_row_stride,
    grad_normalized_input,
    grad_normalized_input_row_stride,
    hidden_size,
    n_groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    dy = tl.load(grad_output + row_idx * grad_output_row_stride + offsets, mask=mask, other=0.0).to(tl.float32)
    for group_idx in tl.static_range(n_groups):
        group_offsets = group_idx * hidden_size + offsets
        logits = tl.load(mix_logits + row_idx * mix_logits_row_stride + group_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        values = tl.load(
            normalized_input + row_idx * normalized_input_row_stride + group_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        sigmoid = tl.sigmoid(logits)
        scaled_dy = dy / n_groups
        tl.store(
            grad_mix_logits + row_idx * grad_mix_logits_row_stride + group_offsets,
            scaled_dy * values * sigmoid * (1.0 - sigmoid),
            mask=mask,
        )
        tl.store(
            grad_normalized_input + row_idx * grad_normalized_input_row_stride + group_offsets,
            scaled_dy * sigmoid,
            mask=mask,
        )


class LigerQwen4ExpHyperConnectionPreFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, mix_logits, normalized_input, n_groups):
        if mix_logits.shape != normalized_input.shape:
            raise ValueError("mix_logits and normalized_input must have identical shapes")
        if n_groups <= 0:
            raise ValueError(f"n_groups must be positive, got {n_groups}")
        if mix_logits.shape[-1] % n_groups != 0:
            raise ValueError(f"last dimension {mix_logits.shape[-1]} must be divisible by n_groups={n_groups}")
        hidden_size = mix_logits.shape[-1] // n_groups
        original_shape = mix_logits.shape
        rows = mix_logits.numel() // mix_logits.shape[-1]
        mix_logits_2d = mix_logits.view(rows, -1)
        normalized_input_2d = normalized_input.view(rows, -1)
        output = torch.empty((rows, hidden_size), device=mix_logits.device, dtype=normalized_input.dtype)
        block_size, num_warps = calculate_settings(hidden_size)
        with device_context(mix_logits.device):
            _qwen4_exp_hyper_connection_pre_forward_kernel[(rows,)](
                output,
                output.stride(0),
                mix_logits_2d,
                mix_logits_2d.stride(0),
                normalized_input_2d,
                normalized_input_2d.stride(0),
                hidden_size,
                n_groups=n_groups,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        ctx.save_for_backward(mix_logits_2d, normalized_input_2d)
        ctx.original_shape = original_shape
        ctx.hidden_size = hidden_size
        ctx.n_groups = n_groups
        return output.view(*original_shape[:-1], hidden_size)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        mix_logits, normalized_input = ctx.saved_tensors
        grad_output = grad_output.view(-1, ctx.hidden_size)
        grad_mix_logits = torch.empty_like(mix_logits)
        grad_normalized_input = torch.empty_like(normalized_input)
        block_size, num_warps = calculate_settings(ctx.hidden_size)
        with device_context(mix_logits.device):
            _qwen4_exp_hyper_connection_pre_backward_kernel[(mix_logits.shape[0],)](
                grad_output,
                grad_output.stride(0),
                mix_logits,
                mix_logits.stride(0),
                normalized_input,
                normalized_input.stride(0),
                grad_mix_logits,
                grad_mix_logits.stride(0),
                grad_normalized_input,
                grad_normalized_input.stride(0),
                ctx.hidden_size,
                n_groups=ctx.n_groups,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        return grad_mix_logits.view(*ctx.original_shape), grad_normalized_input.view(*ctx.original_shape), None


class LigerGroupRMSNormFusedFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, hyper_input, weight, eps, offset, casting_mode, n_groups):
        from liger_kernel.ops.rms_norm import rms_norm_forward

        Y, X, RSTD, BLOCK_SIZE, num_warps, casting_mode_ret = rms_norm_forward(
            hyper_input, weight, eps, offset, casting_mode, row_mode=None, n_groups=n_groups
        )
        ctx.save_for_backward(X, weight, RSTD)
        ctx.offset = offset
        ctx.casting_mode = casting_mode_ret
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.n_groups = n_groups
        ctx.hyper_input_shape = hyper_input.shape
        # Preserve undefined gradients for unused aliases so two-consumer paths do not allocate a
        # full-size zero tensor before entering this custom backward.
        ctx.set_materialize_grads(False)
        # Return three views so autograd tracks the consumers separately while they share storage,
        # avoiding activation-sized clones for every consumer.
        return Y, Y.view(Y.shape), Y.view(Y.shape)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad0, grad1, grad2):
        from liger_kernel.ops.rms_norm import rms_group_norm_backward_add2
        from liger_kernel.ops.rms_norm import rms_group_norm_backward_add3
        from liger_kernel.ops.rms_norm import rms_norm_backward

        X, weight, RSTD = ctx.saved_tensors
        gradients = [grad for grad in (grad0, grad1, grad2) if grad is not None]
        if not gradients:
            return None, None, None, None, None, None

        if len(gradients) >= 2:
            backward_fn = rms_group_norm_backward_add3 if len(gradients) == 3 else rms_group_norm_backward_add2
            dX, dW = backward_fn(
                *gradients,
                X,
                weight,
                RSTD,
                ctx.offset,
                ctx.casting_mode,
                ctx.BLOCK_SIZE,
                ctx.num_warps,
                ctx.n_groups,
            )
            return dX.view(*ctx.hyper_input_shape), dW, None, None, None, None

        grad_sum = gradients[0]
        dX, dW = rms_norm_backward(
            grad_sum,
            X,
            weight,
            RSTD,
            ctx.offset,
            ctx.casting_mode,
            ctx.BLOCK_SIZE,
            ctx.num_warps,
            in_place=False,
            row_mode=None,
            n_groups=ctx.n_groups,
        )
        if weight is None:
            return dX.view(*ctx.hyper_input_shape), None, None, None, None, None
        return dX.view(*ctx.hyper_input_shape), dW, None, None, None, None


@triton.jit
def _qwen4_exp_group_rms_norm_backward_write4_kernel(
    dY0_ptr,
    dY2_ptr,
    dResidual_ptr,
    dY_row_stride,
    grad_write_ptr,
    grad_write_row_stride,
    write_weight_ptr,
    write_weight_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_groups: tl.constexpr,
    n_cols,
    offset,
    rows_per_program,
    casting_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    """Qwen4Exp W_write contribution fused into generic grouped-RMSNorm backward math."""
    pid = tl.program_id(0).to(tl.int64)
    row_block_id = pid // n_groups
    group_id = pid % n_groups

    row_start = row_block_id * rows_per_program
    row_end = min(row_start + rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    W_row = tl.load(W_ptr + group_id * n_cols + col_offsets, mask=mask, other=0.0)
    W_row = W_row + tl.cast(offset, tl.float32)

    for row_idx in range(row_start, row_end):
        flat_row = row_idx * n_groups + group_id
        dy_offsets = flat_row * dY_row_stride + col_offsets
        dx_base = dX_ptr + flat_row * dX_row_stride
        x_base = X_ptr + flat_row * X_row_stride
        rstd_base = RSTD_ptr + flat_row * RSTD_row_stride

        dY0_row = tl.load(dY0_ptr + dy_offsets, mask=mask, other=0.0)
        dY_dtype = dY0_row.dtype
        dY2_row = tl.load(dY2_ptr + dy_offsets, mask=mask, other=0.0)

        # Match the vendor BF16 input-gradient GEMM: accumulate G=4 in FP32 and round its
        # materialized output before the left-associated low-precision add3.
        write_contribution = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        write_col_offsets = group_id * n_cols + col_offsets
        for write_group in tl.static_range(n_groups):
            grad_write = tl.load(grad_write_ptr + row_idx * grad_write_row_stride + write_group)
            write_weight = tl.load(
                write_weight_ptr + write_group * write_weight_row_stride + write_col_offsets,
                mask=mask,
                other=0.0,
            )
            write_contribution += grad_write.to(tl.float32) * write_weight.to(tl.float32)
        write_contribution = write_contribution.to(dY_dtype)
        dY_row = (dY0_row + write_contribution + dY2_row).to(dY_dtype)

        X_row = tl.load(x_base + col_offsets, mask=mask, other=0.0)
        rstd_row = tl.load(rstd_base)
        dX_row, dW_update = _rms_group_norm_backward_row(dY_row, X_row, W_row, rstd_row, n_cols, casting_mode, X_dtype)
        dW_row += dW_update
        if HAS_RESIDUAL:
            dResidual_row = tl.load(dResidual_ptr + flat_row * n_cols + col_offsets, mask=mask, other=0.0)
            # The row helper has already rounded dX_norm to the input dtype. Preserve that
            # rounding before the residual add; never add to the unrounded FP32 RMS result.
            dX_row = (dX_row.to(tl.float32) + dResidual_row.to(tl.float32)).to(X_dtype)
        tl.store(dx_base + col_offsets, dX_row, mask=mask)

    tl.store(dW_ptr + pid * dW_row_stride + col_offsets, dW_row, mask=mask)


def _qwen4_exp_group_rms_norm_backward_write4(
    grad0,
    grad2,
    grad_write,
    write_weight,
    X,
    W,
    RSTD,
    offset,
    casting_mode,
    BLOCK_SIZE,
    num_warps,
    n_groups,
    grad_residual=None,
):
    """Run Qwen4Exp's G=4 W_write-gradient fused grouped-RMSNorm backward."""
    if n_groups != 4:
        raise ValueError(f"The fused W_write path requires n_groups=4, got {n_groups}.")

    shape = grad0.shape
    dim = shape[-1]
    group_size = dim // n_groups
    grad0 = grad0.view(-1, group_size)
    grad2 = grad2.view(-1, group_size)
    grad_write = grad_write.view(-1, n_groups)
    write_weight = write_weight.view(n_groups, dim)
    n_rows, n_cols = grad0.shape
    n_token_rows = n_rows // n_groups
    sm_count = get_device_multiprocessor_count(X.device)

    if n_cols > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    dX = torch.empty_like(X)
    partial_dW = torch.empty((sm_count * n_groups, group_size), dtype=torch.float32, device=W.device)
    rows_per_program = math.ceil(n_token_rows / sm_count)

    with device_context(X.device):
        _qwen4_exp_group_rms_norm_backward_write4_kernel[(sm_count * n_groups,)](
            grad0,
            grad2,
            grad_residual,
            grad0.stride(0),
            grad_write,
            grad_write.stride(0),
            write_weight,
            write_weight.stride(0),
            dX,
            dX.stride(0),
            X,
            X.stride(0),
            torch_to_triton_dtype[X.dtype],
            W,
            RSTD,
            RSTD.stride(0),
            partial_dW,
            partial_dW.stride(0),
            n_token_rows,
            n_groups,
            n_cols,
            offset,
            rows_per_program,
            casting_mode,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_RESIDUAL=grad_residual is not None,
            num_warps=num_warps,
        )

    dW = partial_dW.view(sm_count, n_groups, group_size).sum(dim=0).reshape(dim).to(W.dtype)
    return dX.view(*shape), dW


class LigerGroupRMSNormWrite4Function(torch.autograd.Function):
    """Qwen4Exp G=4 grouped RMSNorm plus vendor W_write forward/dWeight and fused input backward."""

    @staticmethod
    def forward(ctx, hyper_input, rms_weight, write_weight, eps, offset, casting_mode, n_groups, return_residual=False):
        from liger_kernel.ops.rms_norm import rms_norm_forward

        # Keep the external residual view on the original input, even when normalization needs
        # contiguous storage. Decorating this forward with ensure_contiguous would lose it.
        residual = hyper_input
        hyper_input = hyper_input.contiguous()
        rms_weight = rms_weight.contiguous()
        write_weight = write_weight.contiguous()
        if n_groups != 4:
            raise ValueError(f"The fused W_write path requires n_groups=4, got {n_groups}.")
        Y, X, RSTD, BLOCK_SIZE, num_warps, casting_mode_ret = rms_norm_forward(
            hyper_input, rms_weight, eps, offset, casting_mode, row_mode=None, n_groups=n_groups
        )
        # Keep both production GEMMs vendor-backed. Only the input-gradient GEMM is replaced in
        # backward; forward and dWeight remain torch.mm/cuBLAS operations.
        Y_2d = Y.view(-1, Y.shape[-1])
        write_logits = torch.mm(Y_2d, write_weight.transpose(0, 1)) / n_groups
        ctx.save_for_backward(X, rms_weight, RSTD, Y_2d, write_weight)
        ctx.offset = offset
        ctx.casting_mode = casting_mode_ret
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.n_groups = n_groups
        ctx.hyper_input_shape = hyper_input.shape
        ctx.set_materialize_grads(False)
        write_logits_shape = (*hyper_input.shape[:-1], n_groups)
        outputs = (Y, Y.view(Y.shape), write_logits.view(*write_logits_shape))
        if return_residual:
            return (*outputs, residual.view_as(residual))
        return outputs

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad0, grad2, grad_write_logits, grad_residual=None):
        from liger_kernel.ops.rms_norm import rms_group_norm_backward_add3

        X, rms_weight, RSTD, normalized_input, write_weight = ctx.saved_tensors
        if grad0 is None and grad2 is None and grad_write_logits is None:
            # A residual-only consumer does not use any of the normalization/write parameters.
            return (grad_residual, None, None, None, None, None, None, None)[: len(ctx.needs_input_grad)]
        grad_write = None if grad_write_logits is None else grad_write_logits.view(-1, ctx.n_groups) / ctx.n_groups
        d_write_weight = None if grad_write is None else torch.mm(grad_write.transpose(0, 1), normalized_input)

        if grad0 is not None and grad2 is not None and grad_write is not None:
            dX, d_rms_weight = _qwen4_exp_group_rms_norm_backward_write4(
                grad0,
                grad2,
                grad_write,
                write_weight,
                X,
                rms_weight,
                RSTD,
                ctx.offset,
                ctx.casting_mode,
                ctx.BLOCK_SIZE,
                ctx.num_warps,
                ctx.n_groups,
                grad_residual,
            )
            return (dX.view(*ctx.hyper_input_shape), d_rms_weight, d_write_weight, None, None, None, None, None)[
                : len(ctx.needs_input_grad)
            ]

        if grad0 is None:
            grad0 = torch.zeros_like(normalized_input).view(*ctx.hyper_input_shape)
        if grad2 is None:
            grad2 = torch.zeros_like(grad0)
        d_norm_write = (
            torch.zeros_like(grad0) if grad_write is None else torch.mm(grad_write, write_weight).view_as(grad0)
        )
        dX, d_rms_weight = rms_group_norm_backward_add3(
            grad0,
            d_norm_write,
            grad2,
            X,
            rms_weight,
            RSTD,
            ctx.offset,
            ctx.casting_mode,
            ctx.BLOCK_SIZE,
            ctx.num_warps,
            ctx.n_groups,
        )
        dX = dX.view(*ctx.hyper_input_shape)
        if grad_residual is not None:
            dX = dX + grad_residual
        return (dX, d_rms_weight, d_write_weight, None, None, None, None, None)[: len(ctx.needs_input_grad)]


@triton.jit
def _qwen4_exp_gr_write_forward_kernel(
    output,
    output_row_stride,
    block_output,
    block_output_row_stride,
    residual,
    residual_row_stride,
    write_logits,
    write_logits_row_stride,
    hidden_size,
    n_groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    block = tl.load(block_output + row_idx * block_output_row_stride + offsets, mask=mask, other=0.0)
    for group_idx in tl.static_range(n_groups):
        group_offsets = group_idx * hidden_size + offsets
        residual_values = tl.load(residual + row_idx * residual_row_stride + group_offsets, mask=mask, other=0.0)
        write_logit = tl.load(write_logits + row_idx * write_logits_row_stride + group_idx).to(tl.float32)
        write_scale = (2.0 * tl.sigmoid(write_logit)).to(block.dtype)
        # Match HF's materialized low-precision multiplication before the residual add.
        injected = (write_scale * block).to(block.dtype)
        tl.store(output + row_idx * output_row_stride + group_offsets, residual_values + injected, mask=mask)


@triton.jit
def _qwen4_exp_gr_write_backward_kernel(
    grad_output,
    grad_output_row_stride,
    block_output,
    block_output_row_stride,
    write_logits,
    write_logits_row_stride,
    grad_block_output,
    grad_block_output_row_stride,
    grad_write_logits,
    grad_write_logits_row_stride,
    hidden_size,
    n_groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    block = tl.load(block_output + row_idx * block_output_row_stride + offsets, mask=mask, other=0.0)
    grad_block = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for group_idx in tl.static_range(n_groups):
        group_offsets = group_idx * hidden_size + offsets
        grad = tl.load(grad_output + row_idx * grad_output_row_stride + group_offsets, mask=mask, other=0.0)
        write_logit = tl.load(write_logits + row_idx * write_logits_row_stride + group_idx).to(tl.float32)
        sigmoid = tl.sigmoid(write_logit).to(block.dtype)
        write_scale = (2.0 * sigmoid).to(block.dtype)

        # Cast each product to the input dtype before reducing, matching the
        # low-precision tensors materialized by the eager PyTorch graph.
        grad_block += (grad * write_scale).to(block.dtype).to(tl.float32)
        grad_scale = tl.sum((grad * block).to(block.dtype).to(tl.float32), axis=0).to(block.dtype)
        grad_sigmoid = (2.0 * grad_scale).to(block.dtype)
        grad_logit = grad_sigmoid.to(tl.float32) * sigmoid.to(tl.float32) * (1.0 - sigmoid.to(tl.float32))
        tl.store(grad_write_logits + row_idx * grad_write_logits_row_stride + group_idx, grad_logit)

    tl.store(
        grad_block_output + row_idx * grad_block_output_row_stride + offsets,
        grad_block,
        mask=mask,
    )


class LigerQwen4ExpGRWriteFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, block_output, residual, write_logits):
        n_groups = write_logits.shape[-1]
        hidden_size = block_output.shape[-1]
        if n_groups <= 0 or hidden_size <= 0:
            raise ValueError(f"groups and hidden size must be positive, got groups={n_groups}, hidden={hidden_size}")
        if (
            write_logits.shape[:-1] != block_output.shape[:-1]
            or residual.shape[:-1] != block_output.shape[:-1]
            or residual.shape[-1] != n_groups * hidden_size
        ):
            raise ValueError("residual, block_output, and write_logits shapes are incompatible")

        rows = block_output.numel() // hidden_size
        block_output_2d = block_output.view(rows, hidden_size)
        residual_2d = residual.view(rows, n_groups * hidden_size)
        write_logits_2d = write_logits.view(rows, n_groups)
        output = torch.empty_like(residual_2d)
        block_size, num_warps = calculate_settings(hidden_size)
        with device_context(block_output.device):
            _qwen4_exp_gr_write_forward_kernel[(rows,)](
                output,
                output.stride(0),
                block_output_2d,
                block_output_2d.stride(0),
                residual_2d,
                residual_2d.stride(0),
                write_logits_2d,
                write_logits_2d.stride(0),
                hidden_size,
                n_groups=n_groups,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )

        ctx.save_for_backward(block_output_2d, write_logits_2d)
        ctx.block_output_shape = block_output.shape
        ctx.residual_shape = residual.shape
        ctx.write_logits_shape = write_logits.shape
        ctx.hidden_size = hidden_size
        ctx.n_groups = n_groups
        return output.view(*residual.shape)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        block_output, write_logits = ctx.saved_tensors
        grad_output_2d = grad_output.view(-1, ctx.n_groups * ctx.hidden_size)
        grad_block_output = torch.empty_like(block_output)
        grad_write_logits = torch.empty_like(write_logits)
        block_size, num_warps = calculate_settings(ctx.hidden_size)
        with device_context(block_output.device):
            _qwen4_exp_gr_write_backward_kernel[(block_output.shape[0],)](
                grad_output_2d,
                grad_output_2d.stride(0),
                block_output,
                block_output.stride(0),
                write_logits,
                write_logits.stride(0),
                grad_block_output,
                grad_block_output.stride(0),
                grad_write_logits,
                grad_write_logits.stride(0),
                ctx.hidden_size,
                n_groups=ctx.n_groups,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )

        # Returning grad_output as dResidual is a semantic alias only. The kernel
        # never mutates it or treats it as a shared accumulation buffer.
        return (
            grad_block_output.view(*ctx.block_output_shape),
            grad_output.view(*ctx.residual_shape),
            grad_write_logits.view(*ctx.write_logits_shape),
        )
