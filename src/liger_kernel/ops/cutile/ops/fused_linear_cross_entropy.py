# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Fused linear + cross-entropy (cuTile backend).

Fuses the final linear projection with the cross-entropy loss, computing gradients during the
forward pass (backward-in-forward) so the full (BT, V) logit tensor is never materialised. The
token dimension BT is chunked; each chunk does one GEMM (input_chunk @ weight.T), one cuTile CE
kernel (writes d_logits in-place), then folds grad_input / grad_weight / grad_bias.

Feature parity with the Triton LigerFusedLinearCrossEntropyFunction: ce_weight, ignore_index,
lse_square_scale, label_smoothing, reduction, softcap, return_z_loss, accum_dtype,
use_token_scaling, return_token_accuracy, return_predicted_tokens. The cuTile CE kernel
(liger_cross_entropy_kernel_ct) implements all of these; this wrapper plumbs them through.
"""

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.cross_entropy import _select_cross_entropy_block_size
from liger_kernel.ops.cutile.ops.cross_entropy import liger_cross_entropy_kernel_ct
from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd

# Cap on one chunk's logit tensor so peak logit memory stays O(chunk_size x V) rather than O(BT x V).
MAX_FUSED_SIZE = 65536 // 2


def _launch_ce(
    logits,
    target,
    ce_weight,
    loss_slice,
    z_loss_slice,
    token_acc_slice,
    pred_slice,
    dummies,
    V,
    BLOCK_SIZE,
    inv_n_non_ignore,
    sum_non_ignore_weight,
    weight_sum,
    ignore_index,
    label_smoothing,
    lse_square_scale,
    softcap,
    has_grad,
    reduction_mean,
    has_weight,
    has_softcap,
    return_z_loss,
    return_token_accuracy,
    return_predicted_tokens,
):
    """Launch the cuTile CE kernel for one chunk, substituting dummy tensors for disabled outputs."""
    dummy_f32, dummy_i64 = dummies
    n_rows = logits.shape[0]
    ct.launch(
        torch.cuda.current_stream(),
        (n_rows, 1, 1),
        liger_cross_entropy_kernel_ct,
        (
            logits,
            target,
            ce_weight if has_weight else dummy_f32,
            loss_slice,
            z_loss_slice if return_z_loss else dummy_f32,
            token_acc_slice if return_token_accuracy else dummy_f32,
            pred_slice if return_predicted_tokens else dummy_i64,
            int(V),
            float(inv_n_non_ignore),
            float(sum_non_ignore_weight),
            float(weight_sum),
            int(ignore_index),
            float(label_smoothing),
            float(lse_square_scale),
            float(softcap if softcap is not None else 0.0),
            int(BLOCK_SIZE),
            int(has_grad),
            int(reduction_mean),
            int(has_weight),
            int(has_softcap),
            int(return_z_loss),
            int(return_token_accuracy),
            int(return_predicted_tokens),
        ),
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
    assert isinstance(return_token_accuracy, bool), f"return_token_accuracy must be bool. Got: {return_token_accuracy}"
    assert isinstance(return_predicted_tokens, bool), (
        f"return_predicted_tokens must be bool. Got: {return_predicted_tokens}"
    )
    device = _input.device
    input_requires_grad = _input.requires_grad

    BT, H = _input.shape
    V = weight.shape[0]
    BLOCK_SIZE = _select_cross_entropy_block_size(V)

    # Chunk BT so one chunk's logit tensor is ~O(BT x H): inc_factor = ceil(V/H).
    inc_factor = (V + H - 1) // H
    chunk_size = _next_power_of_2((BT + inc_factor - 1) // inc_factor)
    num_chunks = (BT + chunk_size - 1) // chunk_size

    grad_input = torch.zeros_like(_input, device=device)

    if input_requires_grad:
        acc = weight.dtype if accum_dtype is None else accum_dtype
        grad_weight = torch.zeros_like(weight, dtype=acc, device=device) if weight.requires_grad else None
        grad_bias = torch.zeros_like(bias, dtype=acc, device=device) if bias is not None else None
    else:
        grad_weight = None
        grad_bias = None

    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    z_loss_1d = torch.zeros(BT, dtype=_input.dtype, device=device) if return_z_loss else None
    token_accuracy_1d = torch.zeros(BT, dtype=torch.float32, device=device) if return_token_accuracy else None
    predicted_tokens_1d = torch.full((BT,), -1, dtype=torch.int64, device=device) if return_predicted_tokens else None

    # Normalization counts (matches the Triton path).
    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()
    inv_n_non_ignore = 1.0 / max(total_n_non_ignore, 1)
    reduction_mean = int(reduction == "mean")

    sum_non_ignore_weight = float(total_n_non_ignore)
    weight_sum = 0.0
    has_weight = ce_weight is not None
    if has_weight:
        assert ce_weight.shape[0] == V, f"If given, weight has to be a Tensor of size V. Got: {ce_weight.shape}"
        assert torch.is_floating_point(ce_weight), (
            f"If given, weight has to be of floating point dtype. Got: {ce_weight.dtype}"
        )
        sum_non_ignore_weight = float(
            torch.gather(ce_weight, dim=0, index=target.masked_select(target_mask)).sum().item()
        )
        weight_sum = float(ce_weight.sum().item())
        ce_weight = ce_weight.contiguous().float()

    has_softcap = softcap is not None
    # Dummy tensors for disabled kernel outputs (cuTile requires valid tensor args).
    dummies = (
        torch.zeros(1, dtype=torch.float32, device=device),
        torch.zeros(1, dtype=torch.int64, device=device),
    )

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]  # chunk_size x H

        # Matmul in the input precision.
        logits_chunk = _input_chunk @ weight.t()  # chunk_size x V
        if bias is not None:
            logits_chunk = logits_chunk + bias
        target_chunk = target[start_idx:end_idx]

        # Predicted-probability scaling factors (computed before the CE kernel overwrites logits).
        if use_token_scaling:
            logits_for_softmax = logits_chunk.detach().clone()
            if softcap is not None:
                logits_for_softmax = softcap * torch.tanh(logits_for_softmax / softcap)
            probs = torch.softmax(logits_for_softmax, dim=-1)
            valid_target_mask = target_chunk != ignore_index
            valid_targets = target_chunk[valid_target_mask]
            pred_probs = torch.zeros_like(target_chunk, dtype=probs.dtype, device=device)
            if valid_targets.numel() > 0:
                valid_probs = probs[valid_target_mask]
                pred_probs[valid_target_mask] = torch.gather(valid_probs, -1, valid_targets.unsqueeze(-1)).squeeze(-1)
            scaling_factors = pred_probs.detach()

        loss_1d_slice = loss_1d[start_idx:end_idx]
        z_loss_1d_slice = z_loss_1d[start_idx:end_idx] if return_z_loss else None
        token_accuracy_1d_slice = token_accuracy_1d[start_idx:end_idx] if return_token_accuracy else None
        predicted_tokens_1d_slice = predicted_tokens_1d[start_idx:end_idx] if return_predicted_tokens else None

        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        # CE kernel: computes loss (and z_loss/token_accuracy/predicted_tokens) and writes
        # d_logits in-place into logits_chunk when input_requires_grad.
        _launch_ce(
            logits_chunk,
            target_chunk,
            ce_weight,
            loss_1d_slice,
            z_loss_1d_slice,
            token_accuracy_1d_slice,
            predicted_tokens_1d_slice,
            dummies,
            V,
            BLOCK_SIZE,
            inv_n_non_ignore,
            sum_non_ignore_weight,
            weight_sum,
            ignore_index,
            label_smoothing,
            lse_square_scale,
            softcap,
            input_requires_grad,
            reduction_mean,
            has_weight,
            has_softcap,
            return_z_loss,
            return_token_accuracy,
            return_predicted_tokens,
        )

        # Token scaling on loss / z_loss.
        if use_token_scaling:
            loss_1d_slice = loss_1d_slice * scaling_factors
            loss_1d[start_idx:end_idx] = loss_1d_slice
            if return_z_loss:
                z_loss_1d[start_idx:end_idx] = z_loss_1d_slice * scaling_factors

        grad_logits_chunk = logits_chunk  # now holds d_logits
        if use_token_scaling:
            grad_logits_chunk = grad_logits_chunk * scaling_factors.unsqueeze(-1)

        if input_requires_grad:
            grad_input[start_idx:end_idx] = grad_logits_chunk.to(_input.dtype) @ weight
            if grad_weight is not None:
                grad_weight += torch.mm(grad_logits_chunk.t(), _input_chunk).to(grad_weight.dtype)
            if grad_bias is not None:
                grad_bias += grad_logits_chunk.sum(dim=0).to(grad_bias.dtype)

    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
        token_accuracy = token_accuracy_1d if return_token_accuracy else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        token_accuracy = torch.sum(token_accuracy_1d) / total_n_non_ignore if return_token_accuracy else None

    predicted_tokens = predicted_tokens_1d if return_predicted_tokens else None

    grad_weight = grad_weight.to(weight.dtype) if grad_weight is not None else None
    grad_bias = grad_bias.to(bias.dtype) if grad_bias is not None else None

    return loss, z_loss, token_accuracy, predicted_tokens, grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight, grad_bias):
    """Scale the pre-computed grads by grad_output. Out-of-place (safe for repeated backward)."""
    if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        grad_input = grad_input * grad_output
        if grad_weight is not None:
            grad_weight = grad_weight * grad_output
        if grad_bias is not None:
            grad_bias = grad_bias * grad_output
    return grad_input, grad_weight, grad_bias


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
        loss, z_loss, token_accuracy, predicted_tokens, grad_input, grad_weight, grad_bias = (
            fused_linear_cross_entropy_forward(
                _input,
                weight,
                target,
                ce_weight,
                bias,
                ignore_index,
                lse_square_scale,
                label_smoothing,
                reduction,
                softcap,
                return_z_loss,
                accum_dtype,
                use_token_scaling,
                return_token_accuracy,
                return_predicted_tokens,
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
        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )
        return (
            grad_input,
            grad_weight,
            None,  # target
            grad_bias,
            None,  # ce_weight
            None,  # ignore_index
            None,  # lse_square_scale
            None,  # label_smoothing
            None,  # reduction
            None,  # softcap
            None,  # return_z_loss
            None,  # accum_dtype
            None,  # use_token_scaling
            None,  # return_token_accuracy
            None,  # return_predicted_tokens
        )
