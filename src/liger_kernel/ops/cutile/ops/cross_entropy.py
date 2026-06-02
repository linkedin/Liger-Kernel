# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
from typing import Optional

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2

ConstFloat = ct.Constant[float]
ConstInt = ct.Constant[int]

MAX_FUSED_SIZE = 65536 // 2
LOG2E = 1.4426950408889634


def _select_cross_entropy_block_size(vocab_size: int) -> int:
    return min(4096, min(MAX_FUSED_SIZE, _next_power_of_2(vocab_size)))


@ct.kernel(occupancy=4)
def liger_cross_entropy_kernel_ct(
    input,
    target,
    weight,
    loss,
    z_loss,
    token_accuracy,
    predicted_tokens,
    n_cols,
    inv_n_non_ignore,
    sum_non_ignore_weight,
    weight_sum,
    ignore_index,
    label_smoothing: ConstFloat,
    lse_square_scale: ConstFloat,
    softcap,
    BLOCK_SIZE: ConstInt,
    HAS_GRADIENTS: ConstInt,
    REDUCTION_MEAN: ConstInt,
    HAS_WEIGHT: ConstInt,
    HAS_SOFTCAPPING: ConstInt,
    RETURN_Z_LOSS: ConstInt,
    RETURN_TOKEN_ACCURACY: ConstInt,
    RETURN_PREDICTED_TOKENS: ConstInt,
):
    """
    Compute cross-entropy loss and optionally write input gradients in-place.

    input: Logit tensor of shape (num_rows, n_cols). Gradients are written in-place when HAS_GRADIENTS=1.
    target: Target tensor of shape (num_rows,), dtype int64.
    weight: Per-class weight tensor of shape (n_cols,), dtype float32. Ignored when HAS_WEIGHT=0.
    loss: Output loss tensor of shape (num_rows,).
    z_loss: Output z_loss tensor of shape (num_rows,). Ignored when RETURN_Z_LOSS=0.
    token_accuracy: Output token accuracy tensor of shape (num_rows,). Ignored when RETURN_TOKEN_ACCURACY=0.
    predicted_tokens: Output predicted token indices of shape (num_rows,). Ignored when RETURN_PREDICTED_TOKENS=0.
    n_cols: Vocabulary size.
    inv_n_non_ignore: Reciprocal of the number of non-ignored elements in the batch.
    sum_non_ignore_weight: Sum of non-ignored target weights in the batch.
    weight_sum: Sum of the weight tensor.
    ignore_index: Target index to ignore.
    label_smoothing: Smoothing value for the target distribution.
    lse_square_scale: Scale for the z-loss term, logsumexp(input) ** 2.
    softcap: Threshold for soft-capping logits into (-softcap, +softcap).
    BLOCK_SIZE: Tile width for chunked column iteration.
    HAS_GRADIENTS: Whether to compute and write gradients in-place.
    REDUCTION_MEAN: Whether to apply mean reduction.
    HAS_WEIGHT: Whether per-class weights are enabled.
    HAS_SOFTCAPPING: Whether soft-capping is enabled.
    RETURN_Z_LOSS: Whether to store z loss.
    RETURN_TOKEN_ACCURACY: Whether to store per-token accuracy.
    RETURN_PREDICTED_TOKENS: Whether to store predicted tokens.
    """
    row_idx = ct.bid(0)

    # Load target label (scalar from int64 tensor)
    y = ct.load(target, row_idx, shape=())
    y_int32 = ct.astype(y, ct.int32)

    # Handle ignore_index: zero gradient and store 0 loss
    if y == ignore_index:
        if HAS_GRADIENTS:
            num_chunks_early = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
            for ci in range(num_chunks_early):
                col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
                zero_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=input.dtype)
                ct.scatter(input, (row_idx, col_idx), zero_tile, check_bounds=True)
        ct.scatter(loss, row_idx, ct.astype(0.0, loss.dtype))
        if RETURN_TOKEN_ACCURACY:
            ct.scatter(token_accuracy, row_idx, ct.astype(0.0, token_accuracy.dtype))
        if RETURN_PREDICTED_TOKENS:
            ct.scatter(predicted_tokens, row_idx, ct.astype(-1, predicted_tokens.dtype))
        return

    # Load weight for target class y (when HAS_WEIGHT)
    weight_y = 1.0  # placeholder; overwritten below when HAS_WEIGHT
    if HAS_WEIGHT:
        y_w_idx = ct.add(ct.arange(1, dtype=ct.int32), y_int32)
        w_y_tile = ct.astype(ct.gather(weight, (y_w_idx,), check_bounds=False), ct.float32)
        weight_y = ct.sum(w_y_tile, 0, keepdims=False)

    # Load input[row, y] once.
    y_idx = ct.add(ct.arange(1, dtype=ct.int32), y_int32)
    target_logit_tile = ct.gather(input, (row_idx, y_idx), check_bounds=False)
    target_logit = ct.sum(ct.astype(target_logit_tile, ct.float32), 0, keepdims=False)
    if HAS_SOFTCAPPING:
        target_logit = ct.mul(softcap, ct.tanh(ct.mul(target_logit, 1.0 / softcap)))

    eps = label_smoothing / n_cols
    num_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    # ---- Single-pass online numerically stable logsumexp ----
    m_tile = ct.full((1,), -math.inf, dtype=ct.float32)
    d_tile = ct.full((1,), 0.0, dtype=ct.float32)
    scaled_x_sum_tile = ct.full((1,), 0.0, dtype=ct.float32)
    # For argmax tracking (token accuracy / predicted tokens)
    argmax_val_tile = ct.full((1,), -math.inf, dtype=ct.float32)
    argmax_idx_tile = ct.full((1,), 0, dtype=ct.int32)

    for ci in range(num_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        input_raw = ct.astype(
            ct.gather(input, (row_idx, col_idx), check_bounds=True, padding_value=-math.inf, latency=3),
            ct.float32,
        )
        if HAS_SOFTCAPPING:
            input_tile = ct.mul(softcap, ct.tanh(ct.mul(input_raw, 1.0 / softcap)))
        else:
            input_tile = input_raw

        block_max = ct.max(input_tile, 0, keepdims=False)
        neg_bmax_log2e = -block_max * LOG2E
        block_sum_exp = ct.sum(ct.exp2(input_tile * LOG2E + neg_bmax_log2e, flush_to_zero=True), 0, keepdims=False)

        # Track argmax for token accuracy / predicted tokens
        if RETURN_TOKEN_ACCURACY or RETURN_PREDICTED_TOKENS:
            cur_argmax_val = ct.sum(argmax_val_tile, 0, keepdims=False)
            cur_argmax_idx = ct.sum(argmax_idx_tile, 0, keepdims=False)
            # Find the index of the maximum value in this block
            is_max_mask = ct.equal(input_tile, block_max)
            in_bounds_mask = ct.less(col_idx, n_cols)
            # For invalid (out-of-bounds) positions, use n_cols (larger than any valid index)
            valid_max_col_idx = ct.where(
                ct.bitwise_and(is_max_mask, in_bounds_mask),
                col_idx,
                ct.full((BLOCK_SIZE,), n_cols, dtype=ct.int32),
            )
            block_argmax_idx = ct.min(valid_max_col_idx, 0, keepdims=False)
            # Update global argmax: if block_max > cur_argmax_val, use block_argmax_idx
            new_argmax_val = ct.maximum(cur_argmax_val, block_max)
            new_argmax_idx = ct.where(
                ct.greater(block_max, cur_argmax_val),
                block_argmax_idx,
                cur_argmax_idx,
            )
            argmax_val_tile = ct.full((1,), new_argmax_val, dtype=ct.float32)
            argmax_idx_tile = ct.full((1,), new_argmax_idx, dtype=ct.int32)

        if label_smoothing > 0:
            in_bounds = ct.less(col_idx, n_cols)
            if HAS_WEIGHT:
                w_tile = ct.astype(ct.gather(weight, (col_idx,), check_bounds=True, padding_value=0.0), ct.float32)
                chunk_scaled = ct.sum(
                    ct.where(in_bounds, ct.mul(-eps, ct.mul(input_tile, w_tile)), 0.0),
                    0,
                    keepdims=False,
                )
            else:
                chunk_scaled = ct.sum(ct.where(in_bounds, ct.mul(-eps, input_tile), 0.0), 0, keepdims=False)
            scaled_x_sum_tile = ct.full(
                (1,),
                ct.sum(scaled_x_sum_tile, 0, keepdims=False) + chunk_scaled,
                dtype=ct.float32,
            )

        m_prev = ct.sum(m_tile, 0, keepdims=False)
        d_prev = ct.sum(d_tile, 0, keepdims=False)
        m_new = ct.maximum(m_prev, block_max)
        neg_m_new_log2e = -m_new * LOG2E
        block_sum_exp_scaled = block_sum_exp * ct.exp2(block_max * LOG2E + neg_m_new_log2e, flush_to_zero=True)
        d_new = d_prev * ct.exp2(m_prev * LOG2E + neg_m_new_log2e, flush_to_zero=True) + block_sum_exp_scaled
        m_tile = ct.full((1,), m_new, dtype=ct.float32)
        d_tile = ct.full((1,), d_new, dtype=ct.float32)

    m = ct.sum(m_tile, 0, keepdims=False)
    d = ct.sum(d_tile, 0, keepdims=False)
    lse = m + ct.log(d)

    # ---- Write gradient in-place (fused forward+backward) ----
    if HAS_GRADIENTS:
        inv_d = 1.0 / d
        neg_m_log2e = -m * LOG2E
        for ci in range(num_chunks):
            col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
            input_raw = ct.astype(
                ct.gather(input, (row_idx, col_idx), check_bounds=True, padding_value=-math.inf),
                ct.float32,
            )
            if HAS_SOFTCAPPING:
                intermediate = ct.tanh(ct.mul(input_raw, 1.0 / softcap))
                input_tile = ct.mul(softcap, intermediate)
            else:
                input_tile = input_raw

            softmax_tile = ct.exp2(input_tile * LOG2E + neg_m_log2e, flush_to_zero=True) * inv_d
            is_y = ct.equal(col_idx, y_int32)

            if not HAS_WEIGHT:
                grad_tile = ct.add(softmax_tile, ct.mul(2.0 * lse_square_scale * lse, softmax_tile))
                grad_tile = ct.sub(grad_tile, eps)
                grad_tile = ct.where(is_y, ct.sub(grad_tile, 1.0 - label_smoothing), grad_tile)
                if REDUCTION_MEAN:
                    grad_tile = ct.mul(grad_tile, inv_n_non_ignore)
            else:
                # Weighted gradient
                w_tile = ct.astype(ct.gather(weight, (col_idx,), check_bounds=True, padding_value=0.0), ct.float32)
                dloss_ori = ct.mul(1.0 - label_smoothing, softmax_tile)
                dloss_ori = ct.where(is_y, ct.sub(dloss_ori, 1.0 - label_smoothing), dloss_ori)
                dloss_ori = ct.mul(dloss_ori, weight_y)
                dloss_smooth = ct.mul(eps, ct.sub(ct.mul(softmax_tile, weight_sum), w_tile))
                dz_loss = ct.mul(2.0 * lse_square_scale * lse, softmax_tile)
                if REDUCTION_MEAN:
                    inv_sum_w = 1.0 / sum_non_ignore_weight
                    dloss_ori = ct.mul(dloss_ori, inv_sum_w)
                    dloss_smooth = ct.mul(dloss_smooth, inv_sum_w)
                    dz_loss = ct.mul(dz_loss, inv_n_non_ignore)
                grad_tile = ct.add(ct.add(dloss_ori, dloss_smooth), dz_loss)

            # Chain rule for softcapping: multiply by (1 - tanh^2(x/softcap))
            if HAS_SOFTCAPPING:
                sech2 = ct.sub(1.0, ct.mul(intermediate, intermediate))
                grad_tile = ct.mul(grad_tile, sech2)

            ct.scatter(input, (row_idx, col_idx), ct.astype(grad_tile, input.dtype), check_bounds=True)

    # ---- Compute loss ----
    loss_val = lse - target_logit
    if HAS_WEIGHT:
        loss_val = ct.mul(loss_val, weight_y)

    if label_smoothing > 0:
        scaled_x_sum = ct.sum(scaled_x_sum_tile, 0, keepdims=False)
        if HAS_WEIGHT:
            smooth_loss = ct.add(scaled_x_sum, ct.mul(eps * weight_sum, lse))
        else:
            smooth_loss = scaled_x_sum + label_smoothing * lse
        loss_val = loss_val * (1.0 - label_smoothing) + smooth_loss

    z_loss_val = ct.mul(ct.mul(lse_square_scale, lse), lse)

    if REDUCTION_MEAN:
        if HAS_WEIGHT:
            inv_sum_w = 1.0 / sum_non_ignore_weight
            loss_val = ct.mul(loss_val, inv_sum_w)
        else:
            loss_val = ct.mul(loss_val, inv_n_non_ignore)
        z_loss_val = ct.mul(z_loss_val, inv_n_non_ignore)

    loss_val = ct.add(loss_val, z_loss_val)
    ct.scatter(loss, row_idx, ct.astype(loss_val, loss.dtype))

    if RETURN_Z_LOSS:
        ct.scatter(z_loss, row_idx, ct.astype(z_loss_val, z_loss.dtype))

    if RETURN_TOKEN_ACCURACY or RETURN_PREDICTED_TOKENS:
        argmax_idx = ct.sum(argmax_idx_tile, 0, keepdims=False)
        if RETURN_TOKEN_ACCURACY:
            is_correct = ct.astype(ct.equal(argmax_idx, y_int32), ct.float32)
            ct.scatter(token_accuracy, row_idx, is_correct)
        if RETURN_PREDICTED_TOKENS:
            ct.scatter(predicted_tokens, row_idx, ct.astype(argmax_idx, predicted_tokens.dtype))


def cross_entropy_forward(
    _input,
    target,
    weight,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    reduction,
    softcap,
    return_z_loss,
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

    input_requires_grad = _input.requires_grad
    num_rows, vocab_size = _input.shape

    BLOCK_SIZE = _select_cross_entropy_block_size(vocab_size)

    loss_1d = torch.zeros(num_rows, dtype=_input.dtype, device=_input.device)
    z_loss_1d = torch.zeros(num_rows, dtype=_input.dtype, device=_input.device) if return_z_loss else None
    token_accuracy_1d = (
        torch.zeros(num_rows, dtype=torch.float32, device=_input.device) if return_token_accuracy else None
    )
    predicted_tokens_1d = (
        torch.full((num_rows,), -1, dtype=torch.int64, device=_input.device) if return_predicted_tokens else None
    )

    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum().item()
    assert (target * target_mask).max() < _input.shape[-1], (
        f"Target {target.max()} is out of bounds. Expected < {_input.shape[-1]}"
    )
    assert (target * target_mask).min() >= 0, f"Target {target.min()} is out of bounds. Expected >= 0"
    inv_n_non_ignore = 1.0 / max(n_non_ignore, 1)
    reduction_mean = int(reduction == "mean")

    has_weight = weight is not None
    sum_non_ignore_weight = float(n_non_ignore)
    weight_sum = 0.0
    if has_weight:
        assert weight.shape[0] == vocab_size, f"If given, weight has to be a Tensor of size V. Got: {weight.shape}"
        assert torch.is_floating_point(weight), (
            f"If given, weight has to be a Tensor of floating point dtype. Got: {weight.dtype}"
        )
        sum_non_ignore_weight = (
            torch.gather(weight.float(), dim=0, index=target.masked_select(target_mask)).sum().item()
            if n_non_ignore > 0
            else 1.0
        )
        weight_sum = weight.float().sum().item()
        if not weight.is_contiguous():
            weight = weight.contiguous()

    has_softcapping = softcap is not None
    softcap_val = float(softcap) if softcap is not None else 0.0

    if not _input.is_contiguous():
        _input = _input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()

    # CuTile requires valid tensor arguments even for disabled outputs
    dummy_f32 = torch.zeros(1, dtype=torch.float32, device=_input.device)
    dummy_i64 = torch.zeros(1, dtype=torch.int64, device=_input.device)
    dummy_weight = torch.zeros(1, dtype=torch.float32, device=_input.device)

    ct.launch(
        torch.cuda.current_stream(),
        (num_rows, 1, 1),
        liger_cross_entropy_kernel_ct,
        (
            _input,
            target,
            weight.float() if has_weight else dummy_weight,
            loss_1d,
            z_loss_1d if return_z_loss else dummy_f32,
            token_accuracy_1d if return_token_accuracy else dummy_f32,
            predicted_tokens_1d if return_predicted_tokens else dummy_i64,
            int(vocab_size),
            float(inv_n_non_ignore),
            float(sum_non_ignore_weight),
            float(weight_sum),
            int(ignore_index),
            float(label_smoothing),
            float(lse_square_scale),
            float(softcap_val),
            int(BLOCK_SIZE),
            int(input_requires_grad),
            int(reduction_mean),
            int(has_weight),
            int(has_softcapping),
            int(return_z_loss),
            int(return_token_accuracy),
            int(return_predicted_tokens),
        ),
    )

    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
        token_accuracy = token_accuracy_1d if return_token_accuracy else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        token_accuracy = torch.sum(token_accuracy_1d) / max(n_non_ignore, 1) if return_token_accuracy else None

    predicted_tokens = predicted_tokens_1d if return_predicted_tokens else None

    return loss, z_loss, token_accuracy, predicted_tokens, _input


def cross_entropy_backward(_input, grad_output):
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return _input
    if grad_output.ndim > 0:
        return _input * grad_output.unsqueeze(dim=1)
    return _input * grad_output


class LigerCrossEntropyFunction(torch.autograd.Function):
    """
    cuTile autograd wrapper for the fused cross-entropy loss.
    """

    @staticmethod
    def forward(
        ctx,
        _input: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.FloatTensor],
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        return_token_accuracy: bool = False,
        return_predicted_tokens: bool = False,
    ):
        input_requires_grad = _input.requires_grad

        loss, z_loss, token_accuracy, predicted_tokens, _input = cross_entropy_forward(
            _input,
            target,
            weight,
            ignore_index,
            lse_square_scale,
            label_smoothing,
            reduction,
            softcap,
            return_z_loss,
            return_token_accuracy,
            return_predicted_tokens,
        )

        if input_requires_grad:
            ctx.save_for_backward(_input.detach())
        ctx.return_z_loss = return_z_loss
        ctx.return_token_accuracy = return_token_accuracy
        ctx.return_predicted_tokens = return_predicted_tokens

        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    def backward(ctx, grad_output, grad_output2, grad_output3, grad_output4):
        if ctx.return_z_loss:
            del grad_output2
        if ctx.return_token_accuracy:
            del grad_output3
        if ctx.return_predicted_tokens:
            del grad_output4

        (_input,) = ctx.saved_tensors
        _input = cross_entropy_backward(_input, grad_output)
        return (
            _input,
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
