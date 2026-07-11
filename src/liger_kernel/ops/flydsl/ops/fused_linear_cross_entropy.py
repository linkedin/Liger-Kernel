"""
FlyDSL fused linear + cross-entropy.

Never materializes full BT×V logits: chunks over the flattened batch×seq dim
(BT), runs the FlyDSL CE kernel on each chunk×V tile (in-place grads), then
projects gradients back to H. Chunk sizing matches Triton FLCE / PyTorch
``LinearCrossEntropyOptions(chunking_method="aspect_ratio")``.
"""

import torch

from liger_kernel.ops.flydsl.ops.cross_entropy import launch_ce_on_logits
from liger_kernel.ops.flydsl.ops.cross_entropy import make_norm
from liger_kernel.ops.flydsl.ops.utils import next_power_of_2
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def compute_flce_chunk_size(BT: int, H: int, V: int) -> int:
    """BT rows per logits tile (aspect_ratio heuristic)."""
    inc_factor = _ceil_div(V, H)
    target = _ceil_div(BT, inc_factor)
    return min(next_power_of_2(target), BT)


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
    token_grad_output=None,
    compute_gradients=None,
    weight_requires_grad=None,
):
    assert isinstance(return_z_loss, bool), f"return_z_loss must be True or False. Got: {return_z_loss}"
    assert isinstance(return_token_accuracy, bool), (
        f"return_token_accuracy must be True or False. Got: {return_token_accuracy}"
    )
    assert isinstance(return_predicted_tokens, bool), (
        f"return_predicted_tokens must be True or False. Got: {return_predicted_tokens}"
    )
    assert reduction in ("mean", "sum", "none"), f"Unsupported reduction: {reduction}"

    if ce_weight is not None:
        raise NotImplementedError("flydsl FLCE does not yet support class weights")
    if return_token_accuracy:
        raise NotImplementedError("flydsl FLCE does not yet support return_token_accuracy")
    if return_predicted_tokens:
        raise NotImplementedError("flydsl FLCE does not yet support return_predicted_tokens")
    if use_token_scaling:
        raise NotImplementedError("flydsl FLCE does not yet support use_token_scaling")

    device = _input.device
    # ``compute_gradients`` / ``weight_requires_grad`` let backward re-enter this
    # function with *detached* saved tensors and still get gradients out.
    # ``token_grad_output`` is the per-token upstream gradient for reduction="none";
    # it must be folded into the logits gradient before that gradient is summed into
    # grad_weight / grad_bias.
    input_requires_grad = _input.requires_grad if compute_gradients is None else compute_gradients
    weight_needs_grad = weight.requires_grad if weight_requires_grad is None else weight_requires_grad
    BT, H = _input.shape
    V = weight.shape[0]

    if target.stride(-1) != 1:
        target = target.contiguous()

    chunk_size = compute_flce_chunk_size(BT, H, V)
    num_chunks = _ceil_div(BT, chunk_size)

    # Always allocate like Triton so backward can save a real tensor.
    grad_input = torch.zeros_like(_input, device=device)
    if input_requires_grad:
        if accum_dtype is None:
            grad_weight = torch.zeros_like(weight, device=device) if weight_needs_grad else None
            grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None
        else:
            grad_weight = torch.zeros_like(weight, dtype=accum_dtype, device=device) if weight_needs_grad else None
            grad_bias = torch.zeros_like(bias, dtype=accum_dtype, device=device) if bias is not None else None
    else:
        grad_weight = None
        grad_bias = None

    # fp32 accumulators: these hold per-row losses that are summed below, and a
    # 16-bit buffer overflows once BT * loss exceeds the dtype range (fp16 caps at
    # 65504). Triton FLCE and the cutedsl backend use fp32 here for the same reason.
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    z_loss_1d = torch.zeros(BT, dtype=torch.float32, device=device) if return_z_loss else None

    # Keep the non-ignore count on-device — no .item()/.tolist() D2H sync. The
    # reciprocal is passed to the kernel as a tensor, so the mean is folded into
    # the loss and the logits gradient *inside* the kernel. Everything derived
    # from the logits gradient (grad_input/weight/bias) is therefore already
    # normalized, and no post-hoc rescale pass is needed.
    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum()
    norm = make_norm(n_non_ignore, reduction, device)

    target_i32 = target.to(torch.int32)

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        input_chunk = _input[start_idx:end_idx]

        logits_chunk = input_chunk @ weight.t()
        if bias is not None:
            logits_chunk = logits_chunk + bias
        logits_chunk = logits_chunk.contiguous()

        target_chunk = target_i32[start_idx:end_idx]
        loss_slice = loss_1d[start_idx:end_idx]
        z_slice = z_loss_1d[start_idx:end_idx] if return_z_loss else None

        launch_ce_on_logits(
            logits_chunk,
            target_chunk,
            loss_slice,
            norm,
            ignore_index=ignore_index,
            has_grad=input_requires_grad,
            lse_square_scale=lse_square_scale,
            softcap=softcap,
            label_smoothing=label_smoothing,
            return_z_loss=return_z_loss,
            z_loss_1d=z_slice,
        )

        if input_requires_grad:
            grad_logits_chunk = logits_chunk  # in-place CE grads

            # reduction="none": fold the per-token upstream gradient in HERE, before
            # the projections below sum over the token dimension. grad_weight is
            # sum_i go_i * (g_i outer x_i); once that sum has happened the per-token
            # weights cannot be recovered, so rescaling grad_weight afterwards is
            # mathematically incapable of being correct.
            if token_grad_output is not None:
                grad_logits_chunk = grad_logits_chunk * token_grad_output[start_idx:end_idx].unsqueeze(-1).to(
                    grad_logits_chunk.dtype
                )

            grad_input[start_idx:end_idx] = grad_logits_chunk @ weight
            if grad_weight is not None:
                gw = torch.mm(grad_logits_chunk.t(), input_chunk)
                if grad_weight.dtype != gw.dtype:
                    grad_weight += gw.to(grad_weight.dtype)
                else:
                    grad_weight += gw
            if grad_bias is not None:
                gb = grad_logits_chunk.sum(dim=0)
                if grad_bias.dtype != gb.dtype:
                    grad_bias += gb.to(grad_bias.dtype)
                else:
                    grad_bias += gb

    # The kernel already applied inv_n to both the loss and the logits gradient,
    # so there is nothing left to rescale here.
    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None

    if grad_weight is not None:
        grad_weight = grad_weight.to(weight.dtype)
    if grad_bias is not None:
        grad_bias = grad_bias.to(bias.dtype)

    return loss, z_loss, None, None, grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight, grad_bias):
    """Scale precomputed grads by a *scalar* upstream ``grad_output`` (usually 1.0).

    Always multiply — skipping via ``torch.equal`` would force a D2H sync.

    Only valid for reduction in {mean, sum}, where ``grad_output`` is a scalar. The
    per-token ``reduction="none"`` case cannot be handled by post-hoc scaling (see
    ``LigerFusedLinearCrossEntropyFunction.backward``) and never reaches here.
    """
    grad_input = grad_input * grad_output
    if grad_weight is not None:
        grad_weight = grad_weight * grad_output
    if grad_bias is not None:
        grad_bias = grad_bias * grad_output
    return grad_input, grad_weight, grad_bias


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    """
    FlyDSL fused last-linear + cross-entropy.

    Signature-compatible with ``liger_kernel.ops.fused_linear_cross_entropy``.
    """

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
        # With reduction="none" the loss is per-token, so backward receives a
        # per-token grad_output that has to be folded into the logits gradient BEFORE
        # it is summed into grad_weight / grad_bias. That weight is not known yet, so
        # defer the gradient to backward rather than produce an unscaled one we could
        # never correctly rescale. mean/sum keep the compute-in-forward fast path.
        needs_grad = _input.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
        ctx.defer_grads = reduction == "none" and needs_grad

        loss, z_loss, token_accuracy, predicted_tokens, grad_input, grad_weight, grad_bias = (
            fused_linear_cross_entropy_forward(
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
                compute_gradients=False if ctx.defer_grads else None,
            )
        )

        if ctx.defer_grads:
            ctx.save_for_backward(_input.detach(), weight.detach(), target, bias.detach() if bias is not None else None)
            ctx.fwd_kwargs = dict(
                ce_weight=ce_weight,
                ignore_index=ignore_index,
                lse_square_scale=lse_square_scale,
                label_smoothing=label_smoothing,
                reduction=reduction,
                softcap=softcap,
                accum_dtype=accum_dtype,
                use_token_scaling=use_token_scaling,
            )
            ctx.weight_requires_grad = weight.requires_grad
        else:
            ctx.save_for_backward(
                grad_input.detach(),
                grad_weight.detach() if grad_weight is not None else None,
                grad_bias.detach() if grad_bias is not None else None,
            )
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

        if ctx.defer_grads:
            # reduction="none": grad_output is per-token. Recompute the chunked
            # logits gradient and fold grad_output into each row before projecting,
            # which is the only point at which the per-token weights can be applied.
            _input, weight, target, bias = ctx.saved_tensors
            _, _, _, _, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
                _input=_input,
                weight=weight,
                target=target,
                bias=bias,
                token_grad_output=grad_output,
                compute_gradients=True,
                weight_requires_grad=ctx.weight_requires_grad,
                **ctx.fwd_kwargs,
            )
        else:
            grad_input, grad_weight, grad_bias = ctx.saved_tensors
            grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
                grad_output, grad_input, grad_weight, grad_bias
            )
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
        )
