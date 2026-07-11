"""
FlyDSL fused linear + cross-entropy.

Never materializes full BT×V logits: chunks over the flattened batch×seq dim
(BT), runs the FlyDSL CE kernel on each chunk×V tile (in-place grads), then
projects gradients back to H. Chunk sizing matches Triton FLCE / PyTorch
``LinearCrossEntropyOptions(chunking_method="aspect_ratio")``.
"""

import torch

from liger_kernel.ops.flydsl.ops.cross_entropy import launch_ce_on_logits
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
    input_requires_grad = _input.requires_grad
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
            grad_weight = torch.zeros_like(weight, device=device) if weight.requires_grad else None
            grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None
        else:
            grad_weight = torch.zeros_like(weight, dtype=accum_dtype, device=device) if weight.requires_grad else None
            grad_bias = torch.zeros_like(bias, dtype=accum_dtype, device=device) if bias is not None else None
    else:
        grad_weight = None
        grad_bias = None

    loss_1d = torch.zeros(BT, dtype=_input.dtype, device=device)
    z_loss_1d = torch.zeros(BT, dtype=_input.dtype, device=device) if return_z_loss else None

    # Keep the non-ignore count on-device — no .item()/.tolist() D2H sync. The CE
    # kernel always runs with inv_n=1 (sum-style); mean reduction is applied below
    # by scaling loss/grads with a device reciprocal.
    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum()

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
            inv_n_loss=1.0,
            inv_n_z=1.0,
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

    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        if reduction == "mean":
            inv = torch.reciprocal(n_non_ignore.to(torch.float32).clamp(min=1.0))
            loss = loss * inv.to(dtype=loss.dtype)
            if z_loss is not None:
                z_loss = z_loss * inv.to(dtype=z_loss.dtype)
            if input_requires_grad:
                scale = inv.to(dtype=grad_input.dtype)
                grad_input = grad_input * scale
                if grad_weight is not None:
                    grad_weight = grad_weight * scale.to(dtype=grad_weight.dtype)
                if grad_bias is not None:
                    grad_bias = grad_bias * scale.to(dtype=grad_bias.dtype)

    if grad_weight is not None:
        grad_weight = grad_weight.to(weight.dtype)
    if grad_bias is not None:
        grad_bias = grad_bias.to(bias.dtype)

    return loss, z_loss, None, None, grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight, grad_bias):
    """Scale precomputed grads by upstream ``grad_output`` (usually 1.0).

    Always multiply — skipping via ``torch.equal`` would force a D2H sync.
    """
    if grad_output.ndim == 0:
        grad_input = grad_input * grad_output
        if grad_weight is not None:
            grad_weight = grad_weight * grad_output
        if grad_bias is not None:
            grad_bias = grad_bias * grad_output
    else:
        # reduction="none": per-token upstream grad on the input path.
        grad_input = grad_input * grad_output.unsqueeze(-1)
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
            )
        )
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
        (grad_input, grad_weight, grad_bias) = ctx.saved_tensors
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
