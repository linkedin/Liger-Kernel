import warnings

import torch
import triton

from liger_kernel.ops.backends._ascend.ops.cross_entropy import _make_ce_stats_buffer
from liger_kernel.ops.backends._ascend.ops.cross_entropy import liger_cross_entropy_backward_kernel
from liger_kernel.ops.backends._ascend.ops.cross_entropy import liger_cross_entropy_backward_kernel_no_weight
from liger_kernel.ops.backends._ascend.ops.cross_entropy import liger_cross_entropy_forward_kernel
from liger_kernel.ops.backends._ascend.ops.cross_entropy import liger_cross_entropy_forward_kernel_plain
from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import get_npu_core_count


def get_optimal_block_size(n_cols, has_gradients=True):
    """
    Calculate optimal Block Size using compute_default_tiling_strategy
    """
    multiplier = 12.0 if has_gradients else 8.0
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9, dtype_size=4, memory_multiplier=multiplier, shapes=((n_cols,),), tiling_dims=(0,)
    )

    if tile_shapes:
        return tile_shapes[0][0]
    return 2048


def _get_logits_save_limit_bytes(device):
    # Reusing the forward logits avoids an extra backward GEMM on large plain CE
    # shapes. Cap the retained tensor so the fast path does not silently become
    # a full-vocabulary cache for every possible model size.
    gib = 1024 * 1024 * 1024
    default_limit = 4 * gib
    try:
        device_index = device.index if device.index is not None else torch.npu.current_device()
        total_memory = torch.npu.get_device_properties(device_index).total_memory
    except Exception:
        return default_limit

    return max(gib, min(default_limit, total_memory // 8))


_PLAIN_CE_FAST_PATH_WARN_EMITTED = False


def _warn_once_plain_cross_entropy_preferred():
    """Plain CE fast path materializes full logits; nudge users who only need vanilla CE."""
    global _PLAIN_CE_FAST_PATH_WARN_EMITTED
    if _PLAIN_CE_FAST_PATH_WARN_EMITTED:
        return
    _PLAIN_CE_FAST_PATH_WARN_EMITTED = True
    warnings.warn(
        "Ascend fused_linear_cross_entropy is using the plain cross-entropy path (full BT×V "
        "logits are materialized). If you only need standard cross-entropy, consider "
        "cross_entropy instead.",
        UserWarning,
        stacklevel=3,
    )


def fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    ce_weight=None,
    bias=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
    return_z_loss=False,
    accum_dtype=None,
    use_token_scaling=False,
    return_token_accuracy=False,
    return_predicted_tokens=False,
):
    assert isinstance(return_z_loss, bool), f"return_z_loss must be True or False. Got: {return_z_loss}"
    assert isinstance(return_token_accuracy, bool), (
        f"return_token_accuracy must be True or False. Got: {return_token_accuracy}"
    )
    assert isinstance(return_predicted_tokens, bool), (
        f"return_predicted_tokens must be True or False. Got: {return_predicted_tokens}"
    )
    device = _input.device
    input_requires_grad = _input.requires_grad
    BT, H = _input.shape
    V = weight.shape[0]
    forward_block_size = get_optimal_block_size(V, has_gradients=False)

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))  # (BT + inc_factor - 1) // inc_factor
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    z_loss_1d = torch.zeros(BT, dtype=_input.dtype, device=_input.device) if return_z_loss else None
    token_accuracy_1d = torch.zeros(BT, dtype=torch.float32, device=device) if return_token_accuracy else None
    predicted_tokens_1d = torch.full((BT,), -1, dtype=torch.int64, device=device) if return_predicted_tokens else None

    target_mask = target != ignore_index
    if ce_weight is not None:
        assert ce_weight.shape[0] == V, f"If given, weight has to be a Tensor of size V. Got: {ce_weight.shape}"
        assert torch.is_floating_point(ce_weight), (
            f"If given, weight has to be a Tensor of floating point dtype. Got: {ce_weight.dtype}"
        )
        if ce_weight.stride(-1) != 1:
            ce_weight = ce_weight.contiguous()
    ce_stats = _make_ce_stats_buffer(target, ignore_index, ce_weight, reduction, target_mask=target_mask)

    num_cores = get_npu_core_count()
    ls_eps = float(label_smoothing) / float(V) if label_smoothing else 0.0
    # Keep a single large matmul for performance; avoid materializing/storing full grad_logits.
    logits = _input @ weight.t()  # BT x V

    plain_fast_path = (
        ce_weight is None
        and softcap is None
        and float(label_smoothing) == 0.0
        and float(lse_square_scale) == 0.0
        and (not return_z_loss)
        and (not return_token_accuracy)
        and (not return_predicted_tokens)
        and (not use_token_scaling)
    )

    # For plain CE, prefer a larger BLOCK_SIZE (fewer loop iters) using the same
    # NPU-oriented tuning as `cross_entropy_forward`'s plain_lm path.
    if plain_fast_path:
        tile_shapes = compute_default_tiling_strategy(
            safety_margin=0.9,
            dtype_size=4,
            memory_multiplier=2.5,
            shapes=((V,),),
            tiling_dims=(0,),
        )
        forward_block_size = max(256, tile_shapes[0][0]) if tile_shapes else 8192

    # If we're in the plain path and bias-free, launch forward once for all rows
    # to reduce Python overhead and kernel launch count.
    scaling_factors_full = None
    logits_for_backward = None
    if input_requires_grad and plain_fast_path and bias is None:
        # Save logits for backward when memory allows, avoiding an extra
        # input @ weight.T in backward. This is especially important once BT is
        # large enough that the GEMM dominates the plain CE backward path.
        bytes_per_elem = logits.element_size()
        if BT * V * bytes_per_elem <= _get_logits_save_limit_bytes(device):
            logits_for_backward = logits

    if plain_fast_path and bias is None:
        _warn_once_plain_cross_entropy_preferred()
        if not logits.is_contiguous():
            logits = logits.contiguous()
        if target.stride(-1) != 1:
            target = target.contiguous()
        liger_cross_entropy_forward_kernel_plain[(BT,)](
            X_ptr=logits,
            X_stride=logits.stride(-2),
            Y_ptr=target,
            loss_ptr=loss_1d,
            n_cols=V,
            n_rows=BT,
            ce_stats_ptr=ce_stats,
            ignore_index=ignore_index,
            reduction=reduction,
            BLOCK_SIZE=forward_block_size,
        )
        loss = loss_1d if reduction == "none" else torch.sum(loss_1d)
        return (
            loss,
            None,
            None,
            None,
            loss_1d,
            ce_stats,
            None,
            plain_fast_path,
            logits_for_backward,
        )

    # Forward-only: compute per-row loss (and optional metrics). Gradients are computed in backward.
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        # # when doing matmul, use the original precision

        logits_chunk = logits[start_idx:end_idx]  # chunk_size x V
        if bias is not None:
            logits_chunk = logits_chunk + bias

        target_chunk = target[start_idx:end_idx]  # chunk_size,

        n_rows = logits_chunk.shape[0]

        # Compute predicted probabilities for token scaling if needed
        if use_token_scaling:
            # Compute softmax probabilities for scaling
            # We compute token scaling from the forward logits before any gradient kernel runs.
            logits_for_softmax = logits_chunk.detach().clone()  # Detach to avoid gradient flow
            if softcap is not None:
                logits_for_softmax = softcap * torch.tanh(logits_for_softmax / softcap)

            # Compute softmax to get predicted probabilities
            probs = torch.softmax(logits_for_softmax, dim=-1)

            # Get predicted probabilities for token scaling, handling ignored targets
            valid_target_mask = target_chunk != ignore_index
            safe_targets = torch.where(valid_target_mask, target_chunk, torch.zeros_like(target_chunk))
            pred_probs = torch.gather(probs, -1, safe_targets.unsqueeze(-1)).squeeze(-1)
            pred_probs = pred_probs * valid_target_mask.to(pred_probs.dtype)

            # Store the scaling factors
            scaling_factors = pred_probs.detach()  # Detach to ensure no gradient flow
            if scaling_factors_full is None:
                scaling_factors_full = torch.empty(BT, dtype=scaling_factors.dtype, device=scaling_factors.device)
            scaling_factors_full[start_idx:end_idx] = scaling_factors

        # unreduced loss
        loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size,
        z_loss_1d_slice = z_loss_1d[start_idx:end_idx] if return_z_loss else None
        token_accuracy_1d_slice = token_accuracy_1d[start_idx:end_idx] if return_token_accuracy else None
        predicted_tokens_1d_slice = predicted_tokens_1d[start_idx:end_idx] if return_predicted_tokens else None

        # Avoid unconditional materialization: slices are typically contiguous already.
        if not logits_chunk.is_contiguous():
            logits_chunk = logits_chunk.contiguous()
        if not target_chunk.is_contiguous():
            target_chunk = target_chunk.contiguous()

        if plain_fast_path:
            liger_cross_entropy_forward_kernel_plain[(n_rows,)](
                X_ptr=logits_chunk,
                X_stride=logits_chunk.stride(-2),
                Y_ptr=target_chunk,
                loss_ptr=loss_1d_slice,
                n_cols=V,
                n_rows=n_rows,
                ce_stats_ptr=ce_stats,
                ignore_index=ignore_index,
                reduction=reduction,
                BLOCK_SIZE=forward_block_size,
            )
        else:
            liger_cross_entropy_forward_kernel[(min(n_rows, num_cores),)](
                X_ptr=logits_chunk,
                X_stride=logits_chunk.stride(-2),
                Y_ptr=target_chunk,
                weight_ptr=ce_weight,
                loss_ptr=loss_1d_slice,
                z_loss_ptr=z_loss_1d_slice,
                lse_ptr=loss_1d_slice,
                token_accuracy_ptr=token_accuracy_1d_slice,
                token_accuracy_stride=token_accuracy_1d_slice.stride(-1)
                if return_token_accuracy
                else 0,  # always 1 if accuracy is enabled
                predicted_tokens_ptr=predicted_tokens_1d_slice,
                predicted_tokens_stride=predicted_tokens_1d_slice.stride(-1)
                if return_predicted_tokens
                else 0,  # always 1 if predicted tokens is enabled
                n_cols=V,
                n_rows=n_rows,
                ce_stats_ptr=ce_stats,
                ignore_index=ignore_index,
                ls_eps=ls_eps,
                lse_square_scale=lse_square_scale,
                label_smoothing=label_smoothing,
                reduction=reduction,
                softcap=softcap,
                RETURN_Z_LOSS=return_z_loss,
                RETURN_LSE=False,
                RETURN_TOKEN_ACCURACY=return_token_accuracy,
                RETURN_PREDICTED_TOKENS=return_predicted_tokens,
                HAS_WEIGHT=True if ce_weight is not None else False,
                HAS_SOFTCAPPING=True if softcap is not None else False,
                BLOCK_SIZE=forward_block_size,
            )

        # Apply token scaling if requested
        if use_token_scaling:
            loss_1d_slice = loss_1d_slice * scaling_factors
            if return_z_loss:
                z_loss_1d_slice = z_loss_1d_slice * scaling_factors

        loss_1d[start_idx:end_idx] = loss_1d_slice
        if return_z_loss:
            z_loss_1d[start_idx:end_idx] = z_loss_1d_slice
        if return_token_accuracy:
            token_accuracy_1d[start_idx:end_idx] = token_accuracy_1d_slice
        if return_predicted_tokens:
            predicted_tokens_1d[start_idx:end_idx] = predicted_tokens_1d_slice
        # No gradient work in forward anymore.

    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
        token_accuracy = token_accuracy_1d if return_token_accuracy else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        token_accuracy = (
            torch.sum(token_accuracy_1d) / target_mask.sum(dtype=torch.float32).clamp(min=1.0)
            if return_token_accuracy
            else None
        )

    predicted_tokens = predicted_tokens_1d if return_predicted_tokens else None

    return (
        loss,
        z_loss,
        token_accuracy,
        predicted_tokens,
        loss_1d,  # saved for backward (plain path) / metrics
        ce_stats,
        scaling_factors_full,
        plain_fast_path,
        logits_for_backward,
    )


def fused_linear_cross_entropy_backward(ctx, grad_output):
    (_input, weight, target, loss_1d, ce_stats, saved_logits) = ctx.saved_tensors
    bias = ctx.bias
    ce_weight = ctx.ce_weight
    scaling_factors_full = ctx.scaling_factors_full

    device = _input.device
    BT = _input.shape[0]
    V = weight.shape[0]

    forward_block_size = get_optimal_block_size(V, has_gradients=False)
    backward_block_size = get_optimal_block_size(V, has_gradients=True)
    if ctx.plain_fast_path and 32768 < V <= 131072:
        backward_block_size = 4096

    # Use larger row tiles for GEMM efficiency.
    # If forward saved logits (small/medium BT), process everything in one chunk
    # to minimize launch overhead and maximize GEMM utilization.
    has_saved_logits = saved_logits.numel() != 0
    chunk_size = BT if has_saved_logits else min(BT, 4096)
    num_chunks = triton.cdiv(BT, chunk_size)

    num_cores = get_npu_core_count()
    use_no_weight_backward = (
        V > 4096
        and ctx.plain_fast_path
        and ce_weight is None
        and ctx.softcap is None
        and ctx.label_smoothing == 0.0
        and ctx.lse_square_scale == 0.0
    )
    ls_eps = float(ctx.label_smoothing) / float(V) if ctx.label_smoothing else 0.0

    # Prepare output grads. fp32 accumulation only buys precision when we
    # accumulate multiple chunk GEMMs; for a single chunk it just adds a
    # large dtype-conversion buffer on the hot path.
    grad_accum_dtype = ctx.accum_dtype if num_chunks > 1 else None
    grad_input = torch.empty_like(_input)
    grad_weight = torch.empty_like(weight, dtype=grad_accum_dtype or weight.dtype, device=device)
    grad_bias = (
        torch.empty_like(bias, dtype=grad_accum_dtype or bias.dtype, device=device) if bias is not None else None
    )

    # grad_output is scalar for mean/sum, vector for none
    has_grad_output_vector = ctx.reduction == "none"
    if has_grad_output_vector and grad_output.stride(-1) != 1:
        grad_output = grad_output.contiguous()
    grad_output_stride = grad_output.stride(-1) if has_grad_output_vector else 0

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        input_chunk = _input[start_idx:end_idx]
        target_chunk = target[start_idx:end_idx]
        n_rows = end_idx - start_idx

        if has_saved_logits:
            logits_chunk = saved_logits[start_idx:end_idx]
        else:
            logits_chunk = input_chunk @ weight.t()
            if bias is not None:
                logits_chunk = logits_chunk + bias

        if not logits_chunk.is_contiguous():
            logits_chunk = logits_chunk.contiguous()
        if not target_chunk.is_contiguous():
            target_chunk = target_chunk.contiguous()

        grad_logits_chunk = torch.empty_like(logits_chunk)

        if use_no_weight_backward:
            # Derive lse from per-row loss + x[y]
            loss_1d_slice = loss_1d[start_idx:end_idx]
            liger_cross_entropy_backward_kernel_no_weight[(min(n_rows, num_cores),)](
                X_ptr=logits_chunk,
                X_stride=logits_chunk.stride(-2),
                Y_ptr=target_chunk,
                lse_ptr=loss_1d_slice,
                grad_output_ptr=grad_output,
                grad_output_stride=grad_output_stride,
                dX_ptr=grad_logits_chunk,
                dX_stride=grad_logits_chunk.stride(-2),
                n_cols=V,
                n_rows=n_rows,
                ce_stats_ptr=ce_stats,
                ignore_index=ctx.ignore_index,
                reduction=ctx.reduction,
                BLOCK_SIZE=backward_block_size,
                HAS_LSE=False,
            )
        else:
            # General path needs LSE; recompute it for this chunk (forward kernel with RETURN_LSE).
            lse_chunk = torch.empty(n_rows, dtype=torch.float32, device=device)
            loss_tmp = torch.empty(n_rows, dtype=torch.float32, device=device)
            liger_cross_entropy_forward_kernel[(min(n_rows, num_cores),)](
                X_ptr=logits_chunk,
                X_stride=logits_chunk.stride(-2),
                Y_ptr=target_chunk,
                weight_ptr=ce_weight,
                loss_ptr=loss_tmp,
                z_loss_ptr=loss_tmp,
                lse_ptr=lse_chunk,
                token_accuracy_ptr=loss_tmp,
                token_accuracy_stride=0,
                predicted_tokens_ptr=target_chunk,
                predicted_tokens_stride=0,
                n_cols=V,
                n_rows=n_rows,
                ce_stats_ptr=ce_stats,
                ignore_index=ctx.ignore_index,
                ls_eps=ls_eps,
                lse_square_scale=ctx.lse_square_scale,
                label_smoothing=ctx.label_smoothing,
                reduction=ctx.reduction,
                softcap=ctx.softcap,
                RETURN_Z_LOSS=False,
                RETURN_LSE=True,
                RETURN_TOKEN_ACCURACY=False,
                RETURN_PREDICTED_TOKENS=False,
                HAS_WEIGHT=True if ce_weight is not None else False,
                HAS_SOFTCAPPING=True if ctx.softcap is not None else False,
                BLOCK_SIZE=forward_block_size,
            )

            liger_cross_entropy_backward_kernel[(min(n_rows, num_cores),)](
                X_ptr=logits_chunk,
                X_stride=logits_chunk.stride(-2),
                Y_ptr=target_chunk,
                weight_ptr=ce_weight,
                lse_ptr=lse_chunk,
                grad_output_ptr=grad_output,
                grad_output_stride=grad_output_stride,
                dX_ptr=grad_logits_chunk,
                dX_stride=grad_logits_chunk.stride(-2),
                n_cols=V,
                n_rows=n_rows,
                ce_stats_ptr=ce_stats,
                ignore_index=ctx.ignore_index,
                lse_square_scale=ctx.lse_square_scale,
                label_smoothing=ctx.label_smoothing,
                reduction=ctx.reduction,
                softcap=ctx.softcap,
                BLOCK_SIZE=backward_block_size,
                HAS_WEIGHT=True if ce_weight is not None else False,
                HAS_SOFTCAPPING=True if ctx.softcap is not None else False,
            )

        if ctx.use_token_scaling and scaling_factors_full is not None:
            grad_logits_chunk = grad_logits_chunk * scaling_factors_full[start_idx:end_idx].unsqueeze(-1)

        grad_input[start_idx:end_idx] = grad_logits_chunk @ weight
        grad_weight_ = grad_logits_chunk.t() @ input_chunk
        if chunk_id == 0:
            grad_weight.copy_(grad_weight_)
        else:
            grad_weight.add_(grad_weight_)
        if grad_bias is not None:
            grad_bias_ = grad_logits_chunk.sum(dim=0)
            if chunk_id == 0:
                grad_bias.copy_(grad_bias_)
            else:
                grad_bias.add_(grad_bias_)

    # Keep fp32 accumulation results in fp32 when requested and actually used.
    if grad_accum_dtype is None:
        grad_weight = grad_weight.to(weight.dtype)
        grad_bias = grad_bias.to(bias.dtype) if grad_bias is not None else None

    return (
        grad_input,
        grad_weight,
        None,
        grad_bias,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,  # use_token_scaling
        None,  # return_token_accuracy
        None,  # return_predicted_tokens
    )


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ce_weight=None,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        softcap=None,
        return_z_loss: bool = False,
        accum_dtype=None,
        use_token_scaling: bool = False,
        return_token_accuracy: bool = False,
        return_predicted_tokens: bool = False,
    ):
        """
        Fusing the last linear layer with cross-entropy loss
            Reference: https://github.com/mgmalek/efficient_cross_entropy

        Handle the forward and backward pass of the final linear layer via cross-entropy loss. On Ascend,
        gradients are computed in backward so that the real autograd grad_output is honored. The optimized
        plain CE path may materialize logits internally; if logits are already available to the caller, use
        the cross entropy operator directly.

        _input: (B*T, H) where B is batch size, T is sequence length, H is hidden dimension.
        target: (B*T) where each value is in [0, V-1]
        weight: (V, H) where V is the number of classes
        bias: (V) where V is the number of classes
        ce_weight: a manual rescaling weight given to each class. If given, has to be a Tensor of size V and floating point dtype
        ignore_index: the index to ignore in the target
        label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction: reduction to apply
        accum_dtype (torch.dtype): the dtype of intermediate result buffers for weight and bias gradient accumulations.
            Recommended to set `accum_dtype` to higher precision, e.g. `torch.float32`, if the training is unstable with original dtype. Default: `None`, performing accumulations in original dtype
        use_token_scaling (bool): whether to scale each token's loss by its predicted probability (detached).
            When True, each token's loss is multiplied by the model's predicted probability for that token's true class.
            Default: False.
        return_token_accuracy (bool): When `return_token_accuracy` is `True`, computes and returns per-token accuracy without materializing logits. Default: `False`
        return_predicted_tokens (bool): When `return_predicted_tokens` is `True`, returns per-token predicted class indices (argmax) without materializing logits. Default: `False`
        """
        (
            loss,
            z_loss,
            token_accuracy,
            predicted_tokens,
            loss_1d,
            ce_stats,
            scaling_factors_full,
            plain_fast_path,
            logits_for_backward,
        ) = fused_linear_cross_entropy_forward(
            _input=_input,
            weight=weight,
            target=target,
            bias=bias,
            ce_weight=ce_weight,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            return_z_loss=return_z_loss,
            accum_dtype=accum_dtype,
            use_token_scaling=use_token_scaling,
            return_token_accuracy=return_token_accuracy,
            return_predicted_tokens=return_predicted_tokens,
        )

        # Save minimal tensors; optional tensors are stored on ctx attributes
        # (save_for_backward only accepts Tensors).
        to_save = [_input.detach(), weight.detach(), target.detach(), loss_1d, ce_stats]
        if logits_for_backward is not None:
            to_save.append(logits_for_backward.detach())
        else:
            to_save.append(torch.empty(0, device=_input.device, dtype=_input.dtype))
        ctx.save_for_backward(*to_save)
        ctx.bias = bias.detach() if bias is not None else None
        ctx.ce_weight = ce_weight.detach() if ce_weight is not None else None
        ctx.scaling_factors_full = scaling_factors_full if scaling_factors_full is not None else None

        ctx.ignore_index = ignore_index
        ctx.lse_square_scale = float(lse_square_scale)
        ctx.label_smoothing = float(label_smoothing)
        ctx.reduction = reduction
        ctx.softcap = softcap
        ctx.accum_dtype = accum_dtype
        ctx.use_token_scaling = bool(use_token_scaling)
        ctx.plain_fast_path = bool(plain_fast_path)
        ctx.return_z_loss = return_z_loss
        ctx.return_token_accuracy = return_token_accuracy
        ctx.return_predicted_tokens = return_predicted_tokens
        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2, grad_output3, grad_output4):
        if ctx.return_z_loss:
            del grad_output2  # z_loss is only for logging
        if ctx.return_token_accuracy:
            del grad_output3  # token_accuracy is only for metrics
        if ctx.return_predicted_tokens:
            del grad_output4  # predicted_tokens is only for metrics
        return fused_linear_cross_entropy_backward(ctx, grad_output)
