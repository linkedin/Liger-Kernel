import torch
import triton

from liger_kernel.ops.backends._ascend.ops.cross_entropy import _make_ce_stats_buffer
from liger_kernel.ops.backends._ascend.ops.cross_entropy import _plain_ce_need_mask
from liger_kernel.ops.backends._ascend.ops.cross_entropy import get_optimal_block_size
from liger_kernel.ops.backends._ascend.ops.cross_entropy import get_plain_ce_block_size
from liger_kernel.ops.backends._ascend.ops.cross_entropy import liger_cross_entropy_backward_kernel
from liger_kernel.ops.backends._ascend.ops.cross_entropy import liger_cross_entropy_backward_kernel_no_weight
from liger_kernel.ops.backends._ascend.ops.cross_entropy import liger_cross_entropy_forward_kernel
from liger_kernel.ops.backends._ascend.ops.cross_entropy import liger_cross_entropy_forward_kernel_plain
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import get_npu_core_count


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


def _is_plain_flce(
    ce_weight,
    softcap,
    label_smoothing,
    lse_square_scale,
    return_z_loss,
    return_token_accuracy,
    return_predicted_tokens,
    use_token_scaling,
):
    return (
        ce_weight is None
        and softcap is None
        and float(label_smoothing) == 0.0
        and float(lse_square_scale) == 0.0
        and (not return_z_loss)
        and (not return_token_accuracy)
        and (not return_predicted_tokens)
        and (not use_token_scaling)
    )


def _launch_ce_forward(
    logits,
    target,
    loss_1d,
    ce_stats,
    ce_weight,
    n_rows,
    V,
    ignore_index,
    ls_eps,
    lse_square_scale,
    label_smoothing,
    reduction,
    softcap,
    return_z_loss,
    return_lse,
    return_token_accuracy,
    return_predicted_tokens,
    z_loss_1d,
    lse_ptr,
    token_accuracy_1d,
    predicted_tokens_1d,
    plain_fast_path,
    forward_block_size,
    num_cores,
):
    cores = min(n_rows, num_cores)
    if not logits.is_contiguous():
        logits = logits.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    if plain_fast_path:
        liger_cross_entropy_forward_kernel_plain[(cores,)](
            X_ptr=logits,
            X_stride=logits.stride(-2),
            Y_ptr=target,
            loss_ptr=loss_1d,
            lse_ptr=lse_ptr,
            n_cols=V,
            n_rows=n_rows,
            ce_stats_ptr=ce_stats,
            ignore_index=ignore_index,
            reduction=reduction,
            BLOCK_SIZE=forward_block_size,
            HAS_TAIL=_plain_ce_need_mask(V, forward_block_size),
            RETURN_LSE=return_lse,
        )
        return logits

    z_loss_ptr = z_loss_1d if z_loss_1d is not None else loss_1d
    token_accuracy_ptr = token_accuracy_1d if token_accuracy_1d is not None else loss_1d
    predicted_tokens_ptr = predicted_tokens_1d if predicted_tokens_1d is not None else target
    liger_cross_entropy_forward_kernel[(cores,)](
        X_ptr=logits,
        X_stride=logits.stride(-2),
        Y_ptr=target,
        weight_ptr=ce_weight,
        loss_ptr=loss_1d,
        z_loss_ptr=z_loss_ptr,
        lse_ptr=lse_ptr,
        token_accuracy_ptr=token_accuracy_ptr,
        token_accuracy_stride=token_accuracy_ptr.stride(-1) if return_token_accuracy else 0,
        predicted_tokens_ptr=predicted_tokens_ptr,
        predicted_tokens_stride=predicted_tokens_ptr.stride(-1) if return_predicted_tokens else 0,
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
        RETURN_LSE=return_lse,
        RETURN_TOKEN_ACCURACY=return_token_accuracy,
        RETURN_PREDICTED_TOKENS=return_predicted_tokens,
        HAS_WEIGHT=ce_weight is not None,
        HAS_SOFTCAPPING=softcap is not None,
        BLOCK_SIZE=forward_block_size,
    )
    return logits


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
    BT = _input.shape[0]
    V = weight.shape[0]

    plain_fast_path = _is_plain_flce(
        ce_weight,
        softcap,
        label_smoothing,
        lse_square_scale,
        return_z_loss,
        return_token_accuracy,
        return_predicted_tokens,
        use_token_scaling,
    )
    if plain_fast_path:
        forward_block_size = get_plain_ce_block_size(V, has_gradients=False)
    else:
        forward_block_size = get_optimal_block_size(V, has_gradients=False)

    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    z_loss_1d = torch.zeros(BT, dtype=_input.dtype, device=device) if return_z_loss else None
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

    # One Cube GEMM: NPU prefers a single large matmul over many token chunks.
    logits = _input @ weight.t()
    if bias is not None:
        logits = logits + bias

    return_lse = input_requires_grad and not plain_fast_path
    lse_1d = torch.empty(BT, dtype=torch.float32, device=device) if return_lse else None
    lse_ptr = lse_1d if return_lse else loss_1d

    scaling_factors_full = None

    if use_token_scaling:
        logits_for_softmax = logits.detach()
        if softcap is not None:
            logits_for_softmax = softcap * torch.tanh(logits_for_softmax / softcap)
        probs = torch.softmax(logits_for_softmax, dim=-1)
        safe_targets = torch.where(target_mask, target, torch.zeros_like(target))
        scaling_factors_full = torch.gather(probs, -1, safe_targets.unsqueeze(-1)).squeeze(-1)
        scaling_factors_full = (scaling_factors_full * target_mask.to(scaling_factors_full.dtype)).detach()

    logits = _launch_ce_forward(
        logits=logits,
        target=target,
        loss_1d=loss_1d,
        ce_stats=ce_stats,
        ce_weight=ce_weight,
        n_rows=BT,
        V=V,
        ignore_index=ignore_index,
        ls_eps=ls_eps,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        reduction=reduction,
        softcap=softcap,
        return_z_loss=return_z_loss,
        return_lse=return_lse,
        return_token_accuracy=return_token_accuracy,
        return_predicted_tokens=return_predicted_tokens,
        z_loss_1d=z_loss_1d,
        lse_ptr=lse_ptr,
        token_accuracy_1d=token_accuracy_1d,
        predicted_tokens_1d=predicted_tokens_1d,
        plain_fast_path=plain_fast_path,
        forward_block_size=forward_block_size,
        num_cores=num_cores,
    )

    logits_for_backward = None
    if input_requires_grad:
        bytes_per_elem = logits.element_size()
        if BT * V * bytes_per_elem <= _get_logits_save_limit_bytes(device):
            logits_for_backward = logits

    if use_token_scaling:
        loss_1d = loss_1d * scaling_factors_full
        if return_z_loss:
            z_loss_1d = z_loss_1d * scaling_factors_full

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
        loss_1d,
        ce_stats,
        scaling_factors_full,
        plain_fast_path,
        logits_for_backward,
        lse_1d,
    )


def _launch_ce_backward(
    logits,
    target,
    loss_1d,
    lse,
    ce_stats,
    ce_weight,
    grad_output,
    grad_output_stride,
    n_rows,
    V,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    reduction,
    softcap,
    plain_fast_path,
    backward_block_size,
    num_cores,
):
    """Write CE logits-grad in-place into ``logits`` (no extra BT×V buffer)."""
    cores = min(n_rows, num_cores)
    if plain_fast_path:
        liger_cross_entropy_backward_kernel_no_weight[(cores,)](
            X_ptr=logits,
            X_stride=logits.stride(-2),
            Y_ptr=target,
            lse_ptr=loss_1d,
            grad_output_ptr=grad_output,
            grad_output_stride=grad_output_stride,
            dX_ptr=logits,
            dX_stride=logits.stride(-2),
            n_cols=V,
            n_rows=n_rows,
            ce_stats_ptr=ce_stats,
            ignore_index=ignore_index,
            reduction=reduction,
            BLOCK_SIZE=backward_block_size,
            HAS_LSE=False,
            HAS_TAIL=_plain_ce_need_mask(V, backward_block_size),
        )
        return

    liger_cross_entropy_backward_kernel[(cores,)](
        X_ptr=logits,
        X_stride=logits.stride(-2),
        Y_ptr=target,
        weight_ptr=ce_weight,
        lse_ptr=lse,
        grad_output_ptr=grad_output,
        grad_output_stride=grad_output_stride,
        dX_ptr=logits,
        dX_stride=logits.stride(-2),
        n_cols=V,
        n_rows=n_rows,
        ce_stats_ptr=ce_stats,
        ignore_index=ignore_index,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        reduction=reduction,
        softcap=softcap,
        BLOCK_SIZE=backward_block_size,
        HAS_WEIGHT=ce_weight is not None,
        HAS_SOFTCAPPING=softcap is not None,
    )


def fused_linear_cross_entropy_backward(ctx, grad_output):
    (_input, weight, target, loss_1d, ce_stats, saved_logits) = ctx.saved_tensors
    bias = ctx.bias
    ce_weight = ctx.ce_weight
    scaling_factors_full = ctx.scaling_factors_full
    lse = ctx.lse

    device = _input.device
    BT = _input.shape[0]
    V = weight.shape[0]

    if ctx.plain_fast_path:
        backward_block_size = get_plain_ce_block_size(V, has_gradients=True)
    else:
        backward_block_size = get_optimal_block_size(V, has_gradients=True)

    has_saved_logits = saved_logits.numel() != 0
    chunk_size = BT if has_saved_logits else min(BT, 4096)
    num_chunks = triton.cdiv(BT, chunk_size)
    num_cores = get_npu_core_count()

    grad_accum_dtype = ctx.accum_dtype if num_chunks > 1 else None
    grad_input = torch.empty_like(_input)
    grad_weight = torch.empty_like(weight, dtype=grad_accum_dtype or weight.dtype, device=device)
    grad_bias = (
        torch.empty_like(bias, dtype=grad_accum_dtype or bias.dtype, device=device) if bias is not None else None
    )

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
        loss_1d_slice = loss_1d[start_idx:end_idx]
        lse_chunk = lse[start_idx:end_idx] if lse is not None else loss_1d_slice
        go_chunk = grad_output[start_idx:end_idx] if has_grad_output_vector else grad_output

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

        _launch_ce_backward(
            logits=logits_chunk,
            target=target_chunk,
            loss_1d=loss_1d_slice,
            lse=lse_chunk,
            ce_stats=ce_stats,
            ce_weight=ce_weight,
            grad_output=go_chunk,
            grad_output_stride=grad_output_stride,
            n_rows=n_rows,
            V=V,
            ignore_index=ctx.ignore_index,
            lse_square_scale=ctx.lse_square_scale,
            label_smoothing=ctx.label_smoothing,
            reduction=ctx.reduction,
            softcap=ctx.softcap,
            plain_fast_path=ctx.plain_fast_path,
            backward_block_size=backward_block_size,
            num_cores=num_cores,
        )

        if ctx.use_token_scaling and scaling_factors_full is not None:
            logits_chunk = logits_chunk * scaling_factors_full[start_idx:end_idx].unsqueeze(-1)

        # Project dlogits → dX / dW without an extra V×H workspace when dtypes match.
        grad_input_chunk = grad_input[start_idx:end_idx]
        if grad_input_chunk.is_contiguous() and logits_chunk.dtype == weight.dtype == grad_input_chunk.dtype:
            torch.mm(logits_chunk, weight, out=grad_input_chunk)
        else:
            grad_input_chunk.copy_(logits_chunk @ weight)

        dlogits_t = logits_chunk.t()
        if dlogits_t.dtype == grad_weight.dtype and input_chunk.dtype == grad_weight.dtype:
            if chunk_id == 0:
                torch.mm(dlogits_t, input_chunk, out=grad_weight)
            else:
                grad_weight.addmm_(dlogits_t, input_chunk)
        else:
            grad_weight_ = torch.mm(dlogits_t, input_chunk)
            if grad_weight_.dtype != grad_weight.dtype:
                grad_weight_ = grad_weight_.to(grad_weight.dtype)
            if chunk_id == 0:
                grad_weight.copy_(grad_weight_)
            else:
                grad_weight.add_(grad_weight_)
        if grad_bias is not None:
            grad_bias_ = logits_chunk.sum(dim=0)
            if grad_bias_.dtype != grad_bias.dtype:
                grad_bias_ = grad_bias_.to(grad_bias.dtype)
            if chunk_id == 0:
                grad_bias.copy_(grad_bias_)
            else:
                grad_bias.add_(grad_bias_)

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
            None,
            None,
            None,
            None,  # ce_impl (NVIDIA dispatcher slots; unused on Ascend)
            None,  # ce_mode
        )


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    # NVIDIA FLCE forwards inner CE through dispatch(ce_impl, ce_mode).
    # The Ascend kernel is self-contained; extra slots are accepted and ignored.
    supports_inner_impl_dispatch = False

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
        ce_impl=None,
        ce_mode=None,
    ):
        """
        Fusing the last linear layer with cross-entropy loss
            Reference: https://github.com/mgmalek/efficient_cross_entropy

        Handle the forward and backward pass of the final linear layer via cross-entropy loss. On Ascend,
        a single Cube GEMM materializes logits, the optimized CE kernels compute loss (and optional
        metrics), and backward overwrites those logits in-place with ``dlogits`` before the two
        projection GEMMs. If logits are already available to the caller, use the cross entropy operator.

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
        ce_impl / ce_mode: accepted for NVIDIA FLCE signature parity; unused on Ascend.
        """
        _ = (ce_impl, ce_mode)
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
            lse_1d,
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
            use_token_scaling=use_token_scaling,
            return_token_accuracy=return_token_accuracy,
            return_predicted_tokens=return_predicted_tokens,
        )

        to_save = [_input.detach(), weight.detach(), target.detach(), loss_1d, ce_stats]
        if logits_for_backward is not None:
            to_save.append(logits_for_backward.detach())
        else:
            to_save.append(torch.empty(0, device=_input.device, dtype=_input.dtype))
        ctx.save_for_backward(*to_save)
        ctx.bias = bias.detach() if bias is not None else None
        ctx.ce_weight = ce_weight.detach() if ce_weight is not None else None
        ctx.scaling_factors_full = scaling_factors_full
        ctx.lse = lse_1d

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
            del grad_output2
        if ctx.return_token_accuracy:
            del grad_output3
        if ctx.return_predicted_tokens:
            del grad_output4
        return fused_linear_cross_entropy_backward(ctx, grad_output)
