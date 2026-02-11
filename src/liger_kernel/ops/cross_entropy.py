import operator

from typing import Optional

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import element_mul_kernel
from liger_kernel.ops.utils import is_hip
from liger_kernel.utils import infer_device
from liger_kernel.utils import is_npu_available

if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh


@triton.jit
def liger_cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    weight_ptr,
    loss_ptr,
    z_loss_ptr,
    loss_stride,
    token_accuracy_ptr,
    token_accuracy_stride,
    n_cols,
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
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
    HAS_GRADIENTS: tl.constexpr,
):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now. Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Parameters:
    X_ptr: Pointer to input tensor.
    X_stride (int): The stride of the input tensor.
    Y_ptr: Pointer to target tensor.
    Y_stride (int): The stride of the target tensor.
    weight_ptr: Pointer to weight tensor.
    loss_ptr: Pointer to tensor to store the loss.
    z_loss_ptr: Pointer to tensor to store the z loss. No operation if RETURN_Z_LOSS is 0.
    loss_stride (int): The stride of the loss tensor.
    token_accuracy_ptr: Pointer to tensor to store the per-token accuracy. No operation if RETURN_TOKEN_ACCURACY is 0.
    token_accuracy_stride (int): The stride of the token accuracy tensor.
    n_cols (int): The number of columns in the input tensor.
    n_non_ignore (float): The number of non-ignored elements in the batch.
    sum_non_ignore_weight (float): The sum of non-ignored target's weights in the batch.
    weight_sum (float): The sum of weight tensor.
    ignore_index (int): The index to ignore in the target.
    label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
    lse_square_scale (float): The scaler of (logsumexp(_input)) ^ 2 adding to the loss for the stability of training.
    reduction (str): The string for the reduction to apply
    softcap (float): The upper threshold for scaling logits to the range (-softcap, +softcap).
    RETURN_Z_LOSS (int): The boolean value to decide whether to store z loss to z_loss_ptr or not. It must be 0 or 1.
    RETURN_TOKEN_ACCURACY (int): The boolean value to decide whether to store per-token accuracy to token_accuracy_ptr or not. It must be 0 or 1.
    BLOCK_SIZE (int): The block size for Triton operations.
    HAS_WEIGHT (bool): The boolean value to determine whether assigning weight to each of the classes.
    HAS_SOFTCAPPING (bool): The boolean value to determine whether applying soft-capping or not.
    HAS_GRADIENTS (bool): The boolean value to determine whether calculating gradients in forward pass.
    """

    # https://github.com/triton-lang/triton/issues/1058
    # If B*T*V is too large, program_id * stride will overflow out of int32, so we convert to int64
    program_id = tl.program_id(0).to(tl.int64)

    # 1. Load Y_ptr first because if the target is ignore_index, we can return right away
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    # 2. locate the start index
    X_ptr += program_id * X_stride

    if y == ignore_index:
        # set all X_ptr as 0
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        # For ignored tokens, set token accuracy to 0
        if RETURN_TOKEN_ACCURACY:
            token_accuracy_ptr += program_id * token_accuracy_stride
            tl.store(token_accuracy_ptr, 0.0)
        return

    loss_ptr += program_id * loss_stride
    if RETURN_Z_LOSS:
        z_loss_ptr += program_id * loss_stride
    if RETURN_TOKEN_ACCURACY:
        token_accuracy_ptr += program_id * token_accuracy_stride

    if HAS_WEIGHT:
        weight_y = tl.load(weight_ptr + y).cast(tl.float32)

    # Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)
    # Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867

    # 3. [Online softmax] first pass: find max + sum
    m = float("-inf")  # m is the max value. use the notation from the paper
    d = 0.0  # d is the sum. use the notation from the paper
    argmax_idx = 0  # Track the index of the maximum value for token accuracy computation
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)  # we need to store the original value of X_y for the loss calculation
    if HAS_SOFTCAPPING:
        ori_X_y = softcap * tanh(ori_X_y / softcap)

    # Label smoothing is a general case of normal cross entropy
    # See the full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issue-2503665310
    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
            # Ensure float32 precision for softmax calculation
        ).cast(tl.float32)
        if HAS_SOFTCAPPING:
            X_block = softcap * tanh(X_block / softcap)
        block_max = tl.max(X_block)

        # Track argmax for accuracy computation
        if RETURN_TOKEN_ACCURACY:
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

    # 4. [Online Softmax] Second pass: compute gradients
    # For 'mean' reduction, gradients are normalized by number of non-ignored elements (N)
    # dx_y = (softmax(x_y) - 1) / N
    # dx_i = softmax(x_i) / N, i != y
    # For label smoothing:
    # dx_i = (softmax(x_i) - label_smoothing / V) / N, V = n_cols, i != y
    # dx_y = (softmax(x_y) - label_smoothing / V - (1 - label_smoothing)) / N
    #      = dx_i - (1 - label_smoothing) / N
    # With Z loss:
    # dx_i = ((1 + 2 * lse_square_scale * lse) * softmax(x_i) - label_smoothing / V) / N, i != y
    # dx_y = dx_i - (1 - label_smoothing) / N
    # For 'sum' reduction, no normalization is applied:
    # dx_y = softmax(x_y) - 1
    # dx_i = softmax(x_i), for i ≠ y
    if HAS_GRADIENTS:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            X_block = tl.load(
                X_ptr + X_offsets,
                mask=X_offsets < n_cols,
                other=float("-inf"),
                # Ensure float32 precision for softmax calculation
            ).cast(tl.float32)
            if HAS_SOFTCAPPING:
                intermediate = tanh(X_block / softcap)
                X_block = softcap * intermediate

            if not HAS_WEIGHT:
                # softmax(x_i)
                X_block = tl.exp(X_block - m) / d
                # derivative of z-loss: 2 * lse_square_scale * lse * softmax(x_i)
                X_block += 2 * lse_square_scale * lse * X_block
                # smoothing term
                X_block += -eps
                # special handle dx_y
                X_block = tl.where(X_offsets != y, X_block, X_block - (1 - label_smoothing))
                # reduction scale
                if reduction == "mean":
                    X_block = X_block / n_non_ignore
            else:
                weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                softmax_X = tl.exp(X_block - m) / d
                # derivative of original_loss
                dloss_ori = (1 - label_smoothing) * softmax_X
                # specially handle dx_y
                dloss_ori = tl.where(X_offsets != y, dloss_ori, dloss_ori - (1 - label_smoothing))
                dloss_ori = dloss_ori * weight_y
                # derivative of smooth_loss
                dloss_smooth = eps * (-weight_block + softmax_X * weight_sum)
                # derivative of z-loss
                dz_loss = 2 * lse_square_scale * lse * softmax_X
                # reduction scale
                if reduction == "mean":
                    dloss_ori = dloss_ori / sum_non_ignore_weight
                    dloss_smooth = dloss_smooth / sum_non_ignore_weight
                    # TODO: Implement weighted z_loss. Currently, z_loss is not scaled by weight.
                    dz_loss = dz_loss / n_non_ignore
                # derivative of total_loss
                X_block = dloss_ori + dloss_smooth + dz_loss

            # chain rule softcapping
            # d(softcap * tanh(x / softcap)) = (1 - tanh^2(x / softcap))
            if HAS_SOFTCAPPING:
                X_block = X_block * (1 - intermediate * intermediate)

            tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    # We need tl.debug_barrier() to ensure the new result of X_ptr is written as mentioned in
    # https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/ops/cross_entropy.py#L34
    tl.debug_barrier()

    # 5. Calculate the loss

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

    tl.store(loss_ptr, loss)
    if RETURN_Z_LOSS:
        tl.store(z_loss_ptr, z_loss)
    if RETURN_TOKEN_ACCURACY:
        # Store 1.0 if prediction is correct, 0.0 otherwise
        is_correct = 1.0 if argmax_idx == y else 0.0
        tl.store(token_accuracy_ptr, is_correct)


# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
# the best size we found by manually tuning on xpu and npu.
if infer_device() == "xpu":
    MAX_FUSED_SIZE = 4096
elif infer_device() == "npu":
    MAX_FUSED_SIZE = 2048
else:
    MAX_FUSED_SIZE = 65536 // 2


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
):
    assert isinstance(return_z_loss, bool), f"return_z_loss must be True or False. Got: {return_z_loss}"
    assert isinstance(return_token_accuracy, bool), (
        f"return_token_accuracy must be True or False. Got: {return_token_accuracy}"
    )

    BT, V = _input.shape
    n_rows = BT

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    # unreduced loss
    loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device)
    z_loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device) if return_z_loss else None
    token_accuracy_1d = (
        torch.zeros(n_rows, dtype=torch.float32, device=_input.device) if return_token_accuracy else None
    )

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

    # Here we use a trick to store X_ptr gradient in X_ptr so we can save memory
    liger_cross_entropy_kernel[(n_rows,)](
        X_ptr=_input,
        X_stride=_input.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),  # always 1
        weight_ptr=weight,  # dummy if None
        loss_ptr=loss_1d,
        z_loss_ptr=z_loss_1d,
        loss_stride=loss_1d.stride(-1),  # always 1
        token_accuracy_ptr=token_accuracy_1d,
        token_accuracy_stride=token_accuracy_1d.stride(-1)
        if return_token_accuracy
        else 0,  # always 1 if accuracy is enabled
        n_cols=V,
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
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=True if weight is not None else False,
        HAS_SOFTCAPPING=True if softcap is not None else False,
        HAS_GRADIENTS=_input.requires_grad,
        # TODO: 32 seems to give the best performance
        # Performance is quite sensitive to num_warps
        num_warps=32 if not is_hip() else 16,
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

    return loss, z_loss, token_accuracy, _input


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
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

        element_mul_kernel[(n_rows,)](
            _input,
            _input.stride(-2),
            grad_output,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
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
        return_z_loss (bool): When `return_z_loss` is `True`, returns (loss, z_loss, token_accuracy) instead of (loss, None, None). Default: `False`
        return_token_accuracy (bool): When `return_token_accuracy` is `True`, computes and returns per-token accuracy without materializing logits. Default: `False`

        Returns:
        tuple: A tuple with the computed losses and accuracy: (loss, z_loss, token_accuracy). z_loss and token_accuracy are None if not requested.
        """
        input_requires_grad = _input.requires_grad

        loss, z_loss, token_accuracy, _input = cross_entropy_forward(
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
        )
        # TODO: investigation
        # If we don't detach the _input tensor, the memory will double
        # Not sure why but seems that there will be a time both grad and value exist but in different location
        if input_requires_grad:
            ctx.save_for_backward(_input.detach())
        ctx.return_z_loss = return_z_loss
        ctx.return_token_accuracy = return_token_accuracy

        return loss, z_loss, token_accuracy

    @staticmethod
    def backward(ctx, grad_output, grad_output2, grad_output3):
        """
        The backward pass of the Liger Cross Entropy loss.

        Parameters:
        ctx : The context object with saved tensors.
        grad_output (tensor): The tensor containing the gradient of the loss with respect to the output.
        grad_output2 (tensor): No use. Gradient for z_loss (not used as z_loss is only for logging).
        grad_output3 (tensor): No use. Gradient for token_accuracy (not used as token_accuracy is only for metrics).
        Returns:
        tuple: A tuple with the gradients with respect to the inputs. The elements are tensors or None.
        """
        if ctx.return_z_loss:
            del grad_output2  # z_loss is only for logging
        if ctx.return_token_accuracy:
            del grad_output3  # token_accuracy is only for metrics

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
        )
