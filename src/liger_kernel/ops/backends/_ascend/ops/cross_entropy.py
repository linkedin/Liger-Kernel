from typing import Optional

import torch
import triton
import triton.language as tl

from triton.language.math import tanh

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import element_mul_kernel
from liger_kernel.ops.utils import get_npu_core_count


@triton.jit
def liger_cross_entropy_forward_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    weight_ptr,
    loss_ptr,
    z_loss_ptr,
    lse_ptr,
    loss_stride,
    token_accuracy_ptr,
    token_accuracy_stride,
    predicted_tokens_ptr,
    predicted_tokens_stride,
    n_cols,
    n_rows,
    n_non_ignore,
    sum_non_ignore_weight,
    weight_sum,
    ignore_index,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,  # set it as constexpr since reduction is always known at compile time
    softcap,
    RETURN_Z_LOSS: tl.constexpr,
    RETURN_TOKEN_ACCURACY: tl.constexpr,
    RETURN_PREDICTED_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    """
    This kernel computes the cross entropy loss (forward pass only, no gradient computation).
    Compared to the original fused kernel, separating forward from backward eliminates the
    second read pass over X that gradient computation requires, halving memory traffic in
    forward. On NPU where Triton dispatch overhead is significant, this brings forward
    latency down from ~20x to ~2x vs the PyTorch baseline.

    lse (log-sum-exp) is written to lse_ptr when STORE_LSE is True so the backward pass
    can reuse it without recomputing the full softmax pass.

    Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Parameters:
    X_ptr: Pointer to input tensor. Read-only in this kernel; gradients are NOT written here.
    X_stride (int): The stride of the input tensor.
    Y_ptr: Pointer to target tensor.
    Y_stride (int): The stride of the target tensor.
    weight_ptr: Pointer to weight tensor.
    loss_ptr: Pointer to tensor to store the loss.
    z_loss_ptr: Pointer to tensor to store the z loss. No operation if RETURN_Z_LOSS is 0.
    lse_ptr: Pointer to tensor to store per-row log-sum-exp for reuse in backward. No operation if STORE_LSE is 0.
    loss_stride (int): The stride of the loss tensor.
    token_accuracy_ptr: Pointer to tensor to store the per-token accuracy. No operation if RETURN_TOKEN_ACCURACY is 0.
    token_accuracy_stride (int): The stride of the token accuracy tensor.
    n_cols (int): The number of columns in the input tensor.
    n_rows (int): The total number of rows to process.
    n_non_ignore (float): The number of non-ignored elements in the batch.
    sum_non_ignore_weight (float): The sum of non-ignored target's weights in the batch.
    weight_sum (float): The sum of weight tensor.
    ignore_index (int): The index to ignore in the target.
    label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
    lse_square_scale (float): The scaler of (logsumexp(_input)) ^ 2 adding to the loss for the stability of training.
    reduction (str): The string for the reduction to apply.
    softcap (float): The upper threshold for scaling logits to the range (-softcap, +softcap).
    RETURN_Z_LOSS (int): The boolean value to decide whether to store z loss to z_loss_ptr or not. It must be 0 or 1.
    RETURN_TOKEN_ACCURACY (int): The boolean value to decide whether to store per-token accuracy to token_accuracy_ptr or not. It must be 0 or 1.
    RETURN_PREDICTED_TOKENS (int): The boolean value to decide whether to store per-token predicted class indices. It must be 0 or 1.
    BLOCK_SIZE (int): The block size for Triton operations.
    HAS_WEIGHT (bool): The boolean value to determine whether assigning weight to each of the classes.
    HAS_SOFTCAPPING (bool): The boolean value to determine whether applying soft-capping or not.
    STORE_LSE (bool): The boolean value to determine whether storing lse to lse_ptr for backward reuse.
    """

    # Grid-Stride Loop: each program processes multiple rows
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    start_row = pid
    stride = num_progs

    for row_idx in range(start_row, n_rows, stride):
        # https://github.com/triton-lang/triton/issues/1058
        # If B*T*V is too large, program_id * stride will overflow out of int32, so we convert to int64
        program_id = row_idx.to(tl.int64)

        # 1. Load Y_ptr first because if the target is ignore_index, we can return right away
        Y_ptr_offset = program_id * Y_stride
        y = tl.load(Y_ptr + Y_ptr_offset)

        # 2. locate the start index
        X_ptr_offset = program_id * X_stride

        is_ignored = y == ignore_index

        if is_ignored:
            # For ignored tokens, zero out lse so backward can detect and skip the row cleanly
            if STORE_LSE:
                tl.store(lse_ptr + program_id, 0.0)
            # For ignored tokens, set token accuracy to 0
            if RETURN_TOKEN_ACCURACY:
                token_accuracy_ptr_offset = program_id * token_accuracy_stride
                tl.store(token_accuracy_ptr + token_accuracy_ptr_offset, 0.0)
            if RETURN_PREDICTED_TOKENS:
                predicted_tokens_ptr_offset = program_id * predicted_tokens_stride
                tl.store(predicted_tokens_ptr + predicted_tokens_ptr_offset, -1)
        else:
            loss_ptr_offset = program_id * loss_stride
            if RETURN_Z_LOSS:
                z_loss_ptr_offset = program_id * loss_stride
            if RETURN_TOKEN_ACCURACY:
                token_accuracy_ptr_offset = program_id * token_accuracy_stride
            if RETURN_PREDICTED_TOKENS:
                predicted_tokens_ptr_offset = program_id * predicted_tokens_stride

            if HAS_WEIGHT:
                weight_y = tl.load(weight_ptr + y).cast(tl.float32)

            # Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)
            # Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867

            # 3. [Online softmax] first pass: find max + sum
            m = float("-inf")  # m is the max value. use the notation from the paper
            d = 0.0  # d is the sum. use the notation from the paper
            argmax_idx = 0  # Track the index of the maximum value for token accuracy / predicted tokens computation
            ori_X_y = tl.load(X_ptr + X_ptr_offset + y).cast(
                tl.float32
            )  # we need to store the original value of X_y for the loss calculation
            if HAS_SOFTCAPPING:
                ori_X_y = softcap * tanh(ori_X_y / softcap)

            # Label smoothing is a general case of normal cross entropy
            # See the full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issue-2503665310
            scaled_x_sum = 0.0
            eps = label_smoothing / n_cols

            for i in range(0, n_cols, BLOCK_SIZE):
                X_offsets = i + tl.arange(0, BLOCK_SIZE)
                X_block = tl.load(
                    X_ptr + X_ptr_offset + X_offsets,
                    mask=X_offsets < n_cols,
                    other=float("-inf"),
                    # Ensure float32 precision for softmax calculation
                ).cast(tl.float32)
                if HAS_SOFTCAPPING:
                    X_block = softcap * tanh(X_block / softcap)
                block_max = tl.max(X_block)

                # Track argmax for accuracy / predicted tokens computation
                if RETURN_TOKEN_ACCURACY or RETURN_PREDICTED_TOKENS:
                    # Find the index of the maximum value in this block
                    is_max_mask = X_block == block_max
                    # Mask out invalid indices with a value larger than n_cols
                    masked_offsets = tl.where(is_max_mask, X_offsets, n_cols)
                    # Get the first (smallest) index where max occurs
                    current_block_argmax_idx = tl.min(masked_offsets)

                    is_new_max = block_max > m
                    argmax_idx = tl.where(is_new_max, current_block_argmax_idx, argmax_idx)

                if label_smoothing > 0:
                    # scale X beforehand to avoid overflow
                    if HAS_WEIGHT:
                        weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                        scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block * weight_block, 0.0))
                    else:
                        scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))
                m_new = tl.maximum(m, block_max)
                d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
                m = m_new

            # log (sum(e^(X_i))) = log (sum(e ^ (max(X) * e ^ (X_i - max(X)))))
            #                    = log (e^(max(X)) * sum(e ^ (X_i - max(X))))
            #                    = max(X) + log (sum(e ^ (X_i - max(X)))) = m + log d
            lse = m + tl.log(d)

            # Store lse for backward reuse: avoids recomputing the full softmax pass in the backward pass.
            # The backward pass reads lse_ptr to reconstruct softmax(x_i) = exp(x_i - lse) cheaply.
            if STORE_LSE:
                tl.store(lse_ptr + program_id, lse)

            # 4. Calculate the loss

            # loss = log (softmax(X_y)) = log ((e ^ (X_y - max(X)) / sum(e ^ (X - max(X))))
            #      = (X_y - max(X)) - log(sum(e ^ (X - max(X))))
            #      = X_y - m - log d = X_y - lse
            # sum(e ^ (X - max(X))) must >= 1 because the max term is e ^ 0 = 1
            # So we can safely calculate log (softmax(X_y)) without overflow
            loss = lse - ori_X_y
            if HAS_WEIGHT:
                loss = weight_y * loss

            # Original loss = H(q, p),  with label smoothing regularization = H(q', p) and (label_smoothing / V) = eps
            # H(q', p) = (1 - label_smoothing) * H(q, p) + label_smoothing * H(u, p)
            #          = (1 - label_smoothing) * H(q, p) + eps * sum(logsoftmax(x_i))
            # By using m (global max of xi) and d (sum of e^(xi-m)), we can simplify as:
            #          = (1 - label_smoothing) * H(q, p) + (sum(-eps * x_i) + label_smoothing * (m + logd))
            # Refer to H(q', p) in section 7 of the paper: https://arxiv.org/pdf/1512.00567
            # pytorch: https://github.com/pytorch/pytorch/blob/2981534f54d49fa3a9755c9b0855e7929c2527f0/aten/src/ATen/native/LossNLL.cpp#L516
            # See full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issuecomment-2333753087
            if label_smoothing > 0:
                if HAS_WEIGHT:
                    smooth_loss = scaled_x_sum + eps * lse * weight_sum
                else:
                    smooth_loss = scaled_x_sum + label_smoothing * lse
                loss = loss * (1 - label_smoothing) + smooth_loss

            # An auxiliary loss, z_loss
            # Refer to Page14 Loss function section in the paper PaLM: https://www.jmlr.org/papers/v24/22-1144.html
            z_loss = lse_square_scale * lse * lse
            # Normalize the loss by the number of non-ignored elements if reduction is "mean"
            if reduction == "mean":
                if HAS_WEIGHT:
                    loss = loss / sum_non_ignore_weight
                else:
                    loss = loss / n_non_ignore
                # TODO: Implement weighted z_loss. Currently, z_loss is not scaled by weight.
                z_loss = z_loss / n_non_ignore
            loss += z_loss

            tl.store(loss_ptr + loss_ptr_offset, loss)
            if RETURN_Z_LOSS:
                tl.store(z_loss_ptr + z_loss_ptr_offset, z_loss)
            if RETURN_TOKEN_ACCURACY:
                # Store 1.0 if prediction is correct, 0.0 otherwise
                is_correct = 1.0 if argmax_idx == y else 0.0
                tl.store(token_accuracy_ptr + token_accuracy_ptr_offset, is_correct)
            if RETURN_PREDICTED_TOKENS:
                tl.store(predicted_tokens_ptr + predicted_tokens_ptr_offset, argmax_idx)


def get_optimal_block_size(n_cols):
    """
    Calculate optimal Block Size for the forward kernel using compute_default_tiling_strategy.
    Forward-only: uses a lighter memory multiplier (no gradient intermediate buffers needed).
    """
    # Cross entropy forward needs online softmax (max + sum accumulation).
    # 10.0 is an empirical multiplier calibrated to Atlas 800I A2 UB (192 KB).
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9, dtype_size=4, memory_multiplier=10.0, shapes=((n_cols,),), tiling_dims=(0,)
    )

    if tile_shapes and len(tile_shapes) > 0:
        return tile_shapes[0][0]
    return 2048


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

    BT, V = _input.shape
    n_rows = BT

    BLOCK_SIZE = get_optimal_block_size(V)

    # unreduced loss
    loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device)
    z_loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device) if return_z_loss else None
    token_accuracy_1d = (
        torch.zeros(n_rows, dtype=torch.float32, device=_input.device) if return_token_accuracy else None
    )
    predicted_tokens_1d = (
        torch.full((n_rows,), -1, dtype=torch.int64, device=_input.device) if return_predicted_tokens else None
    )

    # lse is stored for backward reuse when gradients are needed, avoiding a full softmax recompute
    store_lse = _input.requires_grad
    lse_1d = torch.zeros(n_rows, dtype=torch.float32, device=_input.device) if store_lse else None

    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum().item()
    assert (target * target_mask).max() < _input.shape[-1], (
        f"Target {target.max()} is out of bounds. Expected < {_input.shape[-1]}"
    )
    assert (target * target_mask).min() >= 0, f"Target {target.min()} is out of bounds. Expected >= 0"
    sum_non_ignore_weight = n_non_ignore
    weight_sum = 0.0
    if weight is not None:
        assert weight.shape[0] == V, f"If given, weight has to be a Tensor of size V. Got: {weight.shape}"
        assert torch.is_floating_point(weight), (
            f"If given, weight has to be a Tensor of floating point dtype. Got: {weight.dtype}"
        )
        sum_non_ignore_weight = torch.gather(weight, dim=0, index=target.masked_select(target_mask)).sum().item()
        weight_sum = weight.sum().item()
        # ensure weight is contiguous
        if weight.stride(-1) != 1:
            weight = weight.contiguous()

    # ensure _input and target are contiguous in the last dimension
    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    # NPU-optimized grid configuration
    num_cores = get_npu_core_count()
    grid_size = min(num_cores, n_rows)

    liger_cross_entropy_forward_kernel[(grid_size,)](
        X_ptr=_input,
        X_stride=_input.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),  # always 1
        weight_ptr=weight,  # dummy if None
        loss_ptr=loss_1d,
        z_loss_ptr=z_loss_1d,
        lse_ptr=lse_1d,
        loss_stride=loss_1d.stride(-1),  # always 1
        token_accuracy_ptr=token_accuracy_1d,
        token_accuracy_stride=token_accuracy_1d.stride(-1)
        if return_token_accuracy
        else 0,  # always 1 if accuracy is enabled
        predicted_tokens_ptr=predicted_tokens_1d,
        predicted_tokens_stride=predicted_tokens_1d.stride(-1)
        if return_predicted_tokens
        else 0,  # always 1 if predicted tokens is enabled
        n_cols=V,
        n_rows=n_rows,
        n_non_ignore=n_non_ignore,
        sum_non_ignore_weight=sum_non_ignore_weight,
        ignore_index=ignore_index,
        weight_sum=weight_sum,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        reduction=reduction,
        softcap=softcap,
        RETURN_Z_LOSS=return_z_loss,
        RETURN_TOKEN_ACCURACY=return_token_accuracy,
        RETURN_PREDICTED_TOKENS=return_predicted_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=True if weight is not None else False,
        HAS_SOFTCAPPING=True if softcap is not None else False,
        STORE_LSE=store_lse,
    )

    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
        token_accuracy = token_accuracy_1d if return_token_accuracy else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        # For accuracy, we compute the mean across all non-ignored tokens
        token_accuracy = torch.sum(token_accuracy_1d) / n_non_ignore if return_token_accuracy else None

    predicted_tokens = predicted_tokens_1d if return_predicted_tokens else None

    return loss, z_loss, token_accuracy, predicted_tokens, lse_1d


def cross_entropy_backward_kernel(
    _input,
    target,
    lse_1d,
    weight,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    reduction,
    softcap,
    n_non_ignore,
    sum_non_ignore_weight,
    weight_sum,
):
    """
    Compute input gradients for cross entropy loss using native PyTorch operators.

    Reads lse_1d stored by the forward kernel to reconstruct softmax(x_i) = exp(x_i - lse)
    without a second full scan over X.

    Memory strategy: a single (BT, V) buffer is allocated for the softmax reconstruction and
    reused in-place as the gradient output. Per-row scaling uses (BT,) vectors broadcast as
    (BT, 1), and target-position corrections use index operations instead of constructing a
    full y_onehot matrix. Together these eliminate 2–3 extra (BT, V) intermediate allocations
    compared to a naive implementation.

    Gradient formulas (consistent with the original fused Triton kernel):

    No weight, no label_smoothing, no z_loss, 'mean' reduction:
        dx_i = softmax(x_i) / N              (i != y)
        dx_y = (softmax(x_y) - 1) / N

    With label_smoothing (eps = label_smoothing / V):
        dx_i = (softmax(x_i) - eps) / N      (i != y)
        dx_y = dx_i - (1 - label_smoothing) / N

    With z_loss (lse_square_scale > 0):
        softmax term scaled by (1 + 2 * lse_square_scale * lse), computed as two-step
        addition to avoid float32 precision loss when lse_square_scale is tiny (e.g. 1e-8).

    With weight:
        dloss_ori  = (1 - label_smoothing) * softmax(x_i) * weight_y   (adjust at target)
        dloss_smooth = eps * (-weight_i + softmax(x_i) * weight_sum)
        dz_loss    = 2 * lse_square_scale * lse * softmax(x_i)
        each normalized separately under 'mean' reduction

    For 'sum' reduction N = 1 (no normalization applied).
    Ignored rows (target == ignore_index) receive zero gradient.

    Parameters:
    _input: The original input tensor of shape (BT, V), read-only.
    target: The target tensor of shape (BT,).
    lse_1d: Per-row log-sum-exp values stored by the forward kernel, shape (BT,).
    weight: Optional per-class weight tensor of shape (V,).
    ignore_index (int): The index to ignore in the target.
    lse_square_scale (float): Scale factor for the z-loss gradient term.
    label_smoothing (float): Label smoothing factor in [0, 1).
    reduction (str): 'mean' | 'sum' | 'none'.
    softcap (Optional[float]): Softcap value used in forward; triggers chain-rule correction.
    n_non_ignore (int): Number of non-ignored tokens for 'mean' normalization.
    sum_non_ignore_weight (float): Sum of weights for non-ignored tokens.
    weight_sum (float): Sum of the full weight tensor.

    Returns:
    grad_input (Tensor): Gradient tensor of shape (BT, V), same dtype as _input.
    """
    BT, V = _input.shape
    eps = label_smoothing / V

    # --- softcap forward pass (needed for chain-rule correction) ---
    # intermediate is kept for the chain-rule multiply at the end; x is released early.
    x = _input.float()
    if softcap is not None:
        intermediate = torch.tanh(x / softcap)  # (BT, V), held for chain rule
        x = softcap * intermediate  # softcapped logits (BT, V)

    # --- valid-row mask and safe target ---
    valid_mask = target != ignore_index  # (BT,)  bool
    safe_target = target.clone()
    safe_target[~valid_mask] = 0  # clamp ignored rows to avoid OOB index

    # Indices of non-ignored rows: used for scatter-style point corrections instead of y_onehot.
    valid_rows = torch.where(valid_mask)[0]  # (N_valid,)  — tiny, no (BT,V) alloc
    valid_tgt = safe_target[valid_rows]  # (N_valid,)

    # --- reconstruct softmax(x) = exp(x_i - lse) in-place, reuse as grad buffer ---
    # Single (BT, V) allocation. All subsequent ops are in-place on this buffer.
    lse = lse_1d.float()  # (BT,)
    grad = torch.exp(x - lse.unsqueeze(1))  # (BT, V)  — only large alloc
    del x  # free softcapped logits immediately

    if weight is None:
        # -----------------------------------------------------------------------
        # grad[i, j] = (1 + 2·lse_sq·lse_i) · softmax_ij  −  eps
        # grad[i, y_i] −= (1 − label_smoothing)            (point correction)
        # Then divide by N for 'mean' reduction.
        #
        # Everything is in-place on `grad` to avoid extra (BT, V) allocations.
        # -----------------------------------------------------------------------

        # 1. Z-loss scaling: grad += 2·lse_sq·lse_i · grad  (two-step form)
        #
        # DO NOT compute (1 + c) first: when lse_square_scale is tiny (e.g. 1e-8),
        # c = 2·lse_sq·lse ≈ 2e-8 < float32 epsilon (1.19e-7), so float32(1+c) == 1.0
        # and the entire z-loss term vanishes.
        #
        # addcmul_(t1, t2) does: self += t1 * t2  — fused, no extra (BT,V) allocation,
        # and exactly replicates the Triton: X_block += 2 * lse_sq * lse * X_block
        if lse_square_scale != 0.0:
            c = (2.0 * lse_square_scale * lse).to(grad.dtype).unsqueeze(1)  # (BT, 1)
            grad.addcmul_(grad, c)  # grad[i,j] += grad[i,j] * c[i]

        # 2. Label-smoothing uniform term: subtract eps from every element
        if label_smoothing > 0.0:
            grad.add_(-eps)

        # 3. Target-position correction: subtract (1 − ls) at y_i using index op.
        #    Replaces constructing a full (BT, V) one-hot matrix.
        grad[valid_rows, valid_tgt] -= 1.0 - label_smoothing

        # 4. 'mean' normalization
        if reduction == "mean":
            grad.mul_(1.0 / n_non_ignore)

    else:
        # -----------------------------------------------------------------------
        # With per-class weights the three loss components (original, smooth, z)
        # use different row-wise normalizers, so we handle them via two passes:
        #
        # Pass A — scale softmax_ij by a per-row scalar (no new (BT,V) tensor):
        #   row_scale[i] = (1−ls)·w_y[i]·norm_ori + eps·weight_sum·norm_ori
        #                  + 2·lse_sq·lse[i]·norm_z
        #
        # Pass B — subtract column-broadcast terms:
        #   −eps·norm_ori·w_j        (from smooth loss, shape (V,))
        #   −(1−ls)·w_y[i]·norm_ori  at position y_i  (from one-hot term, scalar per row)
        # -----------------------------------------------------------------------
        weight_f = weight.float()  # (V,)
        weight_y = weight_f[safe_target]  # (BT,)  per-row target weight

        norm_ori = 1.0 / sum_non_ignore_weight if reduction == "mean" else 1.0
        norm_z = 1.0 / n_non_ignore if reduction == "mean" else 1.0

        # Pass A: per-row scale as a (BT,) vector → (BT, 1) broadcast, in-place
        row_scale = (
            (1.0 - label_smoothing) * weight_y * norm_ori  # (BT,)
            + eps * weight_sum * norm_ori  # scalar → broadcast
            + 2.0 * lse_square_scale * lse * norm_z  # (BT,)
        )
        grad.mul_(row_scale.unsqueeze(1))  # in-place, single pass over (BT, V)

        # Pass B-1: subtract eps·norm_ori·weight_j column-wise (label-smooth term)
        if label_smoothing > 0.0:
            grad.add_(-eps * norm_ori * weight_f)  # (V,) broadcasts across rows, in-place

        # Pass B-2: subtract one-hot term at target positions via index op
        # correction[i] = (1−ls)·weight_y[i]·norm_ori  (per valid row, scalar)
        correction = (1.0 - label_smoothing) * weight_y[valid_rows] * norm_ori  # (N_valid,)
        grad[valid_rows, valid_tgt] -= correction

    # --- chain rule for softcapping: d/dx[softcap·tanh(x/softcap)] = 1 − tanh²(x/softcap) ---
    if softcap is not None:
        grad.mul_(1.0 - intermediate * intermediate)  # in-place, reuses intermediate buffer

    # --- zero out ignored rows (in-place, avoids allocating a float mask tensor) ---
    if not valid_mask.all():
        grad[~valid_mask] = 0.0

    return grad.to(_input.dtype)


def cross_entropy_backward(_input, grad_output):
    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        pass
    # If reduction is 'none'
    elif grad_output.ndim > 0:
        _input = _input * grad_output.unsqueeze(dim=1)
    # If reduction is ['mean', 'sum'], grad_output is just a scalar
    # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
    # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
    else:
        BT, V = _input.shape
        n_rows = BT
        BLOCK_SIZE = min(2048, triton.next_power_of_2(V))

        element_mul_kernel[(n_rows,)](
            _input,
            _input.stride(-2),
            grad_output,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return _input


class LigerCrossEntropyFunction(torch.autograd.Function):
    """
    This class implements a custom autograd function for the Liger Cross Entropy loss.
    It overrides the forward and backward methods of the torch.autograd.Function class.
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
        """
        The forward pass of the Liger Cross Entropy loss.

        Parameters:
        ctx : The context object.
        _input (tensor): The input tensor of shape (BT, V) where B is batch size, T is sequence length, V is vocab size.
        target (tensor): The target tensor of shape (BT) where each value is in [0, V-1].
        weight(Tensor, optional): a manual rescaling weight given to each class. If given, has to be a Tensor of size V and floating point dtype
        ignore_index (int): The index to ignore in the target.
        lse_square_scale (float): The scaler of (logsumexp(_input)) ^ 2 adding to the loss for the stability of training.
        label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction (str): The reduction to apply to the output: "none" | "mean | "sum".
        softcap (Optional[float]): The upper threshold for scaling logits to the range (-softcap, +softcap).
        return_z_loss (bool): When `return_z_loss` is `True`, returns (loss, z_loss, token_accuracy, predicted_tokens) instead of (loss, None, None, None). Default: `False`
        return_token_accuracy (bool): When `return_token_accuracy` is `True`, computes and returns per-token accuracy without materializing logits. Default: `False`
        return_predicted_tokens (bool): When `return_predicted_tokens` is `True`, returns per-token predicted class indices (argmax) without materializing logits. Default: `False`

        Returns:
        tuple: A tuple with the computed losses, accuracy, and predicted tokens: (loss, z_loss, token_accuracy, predicted_tokens). z_loss, token_accuracy, and predicted_tokens are None if not requested.
        """
        loss, z_loss, token_accuracy, predicted_tokens, lse_1d = cross_entropy_forward(
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
        if _input.requires_grad:
            # Save original _input (not gradient-modified) and lse for the backward pass.
            # lse_1d is a lightweight (BT,) float32 tensor; much cheaper to store than the full (BT, V) softmax.
            ctx.save_for_backward(_input.detach(), target, lse_1d, weight)
            ctx.ignore_index = ignore_index
            ctx.lse_square_scale = lse_square_scale
            ctx.label_smoothing = label_smoothing
            ctx.reduction = reduction
            ctx.softcap = softcap
            # Pre-compute normalization stats so backward doesn't need to recompute them
            target_mask = target != ignore_index
            ctx.n_non_ignore = target_mask.sum().item()
            ctx.sum_non_ignore_weight = ctx.n_non_ignore
            ctx.weight_sum = 0.0
            if weight is not None:
                ctx.sum_non_ignore_weight = (
                    torch.gather(weight, dim=0, index=target.masked_select(target_mask)).sum().item()
                )
                ctx.weight_sum = weight.sum().item()

        ctx.return_z_loss = return_z_loss
        ctx.return_token_accuracy = return_token_accuracy
        ctx.return_predicted_tokens = return_predicted_tokens

        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    def backward(ctx, grad_output, grad_output2, grad_output3, grad_output4):
        """
        The backward pass of the Liger Cross Entropy loss.

        Parameters:
        ctx : The context object with saved tensors.
        grad_output (tensor): The tensor containing the gradient of the loss with respect to the output.
        grad_output2 (tensor): No use. Gradient for z_loss (not used as z_loss is only for logging).
        grad_output3 (tensor): No use. Gradient for token_accuracy (not used as token_accuracy is only for metrics).
        grad_output4 (tensor): No use. Gradient for predicted_tokens (not used as predicted_tokens is only for metrics).
        Returns:
        tuple: A tuple with the gradients with respect to the inputs. The elements are tensors or None.
        """
        if ctx.return_z_loss:
            del grad_output2  # z_loss is only for logging
        if ctx.return_token_accuracy:
            del grad_output3  # token_accuracy is only for metrics
        if ctx.return_predicted_tokens:
            del grad_output4  # predicted_tokens is only for metrics

        _input, target, lse_1d, weight = ctx.saved_tensors

        # Compute input gradients via the native PyTorch backward implementation.
        # lse_1d lets the backward reconstruct softmax(x_i) = exp(x_i - lse) without
        # a full rescan of X, keeping peak memory at O(BT * V) rather than 2x that.
        grad_input = cross_entropy_backward_kernel(
            _input,
            target,
            lse_1d,
            weight,
            ctx.ignore_index,
            ctx.lse_square_scale,
            ctx.label_smoothing,
            ctx.reduction,
            ctx.softcap,
            ctx.n_non_ignore,
            ctx.sum_non_ignore_weight,
            ctx.weight_sum,
        )

        # Scale gradient by upstream grad_output
        grad_input = cross_entropy_backward(grad_input, grad_output)

        return (
            grad_input,
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
