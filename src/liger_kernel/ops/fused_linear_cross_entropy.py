import torch
import triton

from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import element_mul_kernel
from liger_kernel.ops.utils import is_hip
from liger_kernel.utils import infer_device

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 2048 if infer_device() == "npu" else 65536 // 2


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
):
    assert isinstance(return_z_loss, bool), f"return_z_loss must be True or False. Got: {return_z_loss}"
    assert isinstance(return_token_accuracy, bool), (
        f"return_token_accuracy must be True or False. Got: {return_token_accuracy}"
    )
    device = _input.device

    input_requires_grad = _input.requires_grad

    # inputs have shape: BT x H
    # materialized activations will have shape: BT x V
    # the increase in memory = BT x V
    # reduction can be achieved by partitioning the number of tokens BT into smaller chunks.
    # for ex: if we were to achieve the same memory consumption as BT x H, then the chunk size should be:
    # inc_factor = (V+H-1)//H, chunk_size = (BT + inc_factor - 1)//inc_factor
    # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048
    BT, H = _input.shape
    V = weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))  # (BT + inc_factor - 1) // inc_factor
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    grad_input = torch.zeros_like(_input, device=device)

    # we use fp32 for loss and gradients accumulator
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

    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    z_loss_1d = torch.zeros(BT, dtype=_input.dtype, device=_input.device) if return_z_loss else None
    token_accuracy_1d = torch.zeros(BT, dtype=torch.float32, device=device) if return_token_accuracy else None

    # TODO: evaluate how CUDA synchronization caused by .item() affects the speed
    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()
    total_sum_non_ignore_ce_weight = total_n_non_ignore
    ce_weight_sum = 0.0
    if ce_weight is not None:
        assert ce_weight.shape[0] == V, f"If given, weight has to be a Tensor of size V. Got: {ce_weight.shape}"
        assert torch.is_floating_point(ce_weight), (
            f"If given, weight has to be a Tensor of floating point dtype. Got: {ce_weight.dtype}"
        )
        total_sum_non_ignore_ce_weight = (
            torch.gather(ce_weight, dim=0, index=target.masked_select(target_mask)).sum().item()
        )
        ce_weight_sum = ce_weight.sum().item()
        if ce_weight.stride(-1) != 1:
            ce_weight = ce_weight.contiguous()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]  # chunk_size x H

        # when doing matmul, use the original precision
        logits_chunk = _input_chunk @ weight.t()  # chunk_size x V
        if bias is not None:
            logits_chunk = logits_chunk + bias

        target_chunk = target[start_idx:end_idx]  # chunk_size,

        n_rows = logits_chunk.shape[0]

        # Compute predicted probabilities for token scaling if needed
        if use_token_scaling:
            # Compute softmax probabilities for scaling
            # We need to compute this before the cross entropy kernel modifies logits_chunk
            logits_for_softmax = logits_chunk.detach().clone()  # Detach to avoid gradient flow
            if softcap is not None:
                logits_for_softmax = softcap * torch.tanh(logits_for_softmax / softcap)

            # Compute softmax to get predicted probabilities
            probs = torch.softmax(logits_for_softmax, dim=-1)

            # Get predicted probabilities for token scaling, handling ignored targets
            valid_target_mask = target_chunk != ignore_index
            valid_targets = target_chunk[valid_target_mask]

            if len(valid_targets) > 0:
                # Gather probabilities only for valid targets
                valid_probs = probs[valid_target_mask]
                pred_probs_valid = torch.gather(valid_probs, -1, valid_targets.unsqueeze(-1)).squeeze(-1)

                # Create full tensor with zeros for ignored targets
                pred_probs = torch.zeros_like(target_chunk, dtype=probs.dtype, device=probs.device)
                pred_probs[valid_target_mask] = pred_probs_valid
            else:
                # All targets are ignored
                pred_probs = torch.zeros_like(target_chunk, dtype=probs.dtype, device=probs.device)

            # Store the scaling factors
            scaling_factors = pred_probs.detach()  # Detach to ensure no gradient flow

        # unreduced loss
        loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size,
        z_loss_1d_slice = z_loss_1d[start_idx:end_idx] if return_z_loss else None
        token_accuracy_1d_slice = token_accuracy_1d[start_idx:end_idx] if return_token_accuracy else None

        # ensure _input and target are contiguous
        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        # Here we calculate the gradient of logits_chunk in place so we can save memory.
        liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),  # always 1
            weight_ptr=ce_weight,
            loss_ptr=loss_1d_slice,
            z_loss_ptr=z_loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-1),  # always 1
            token_accuracy_ptr=token_accuracy_1d_slice,
            token_accuracy_stride=token_accuracy_1d_slice.stride(-1)
            if return_token_accuracy
            else 0,  # always 1 if accuracy is enabled
            n_cols=V,
            n_non_ignore=total_n_non_ignore,
            sum_non_ignore_weight=total_sum_non_ignore_ce_weight,
            weight_sum=ce_weight_sum,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            RETURN_Z_LOSS=return_z_loss,
            RETURN_TOKEN_ACCURACY=return_token_accuracy,
            HAS_WEIGHT=True if ce_weight is not None else False,
            HAS_SOFTCAPPING=True if softcap is not None else False,
            HAS_GRADIENTS=input_requires_grad,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
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
        grad_logits_chunk = logits_chunk  # chunk_size x V

        # Apply token scaling to gradients if requested
        if use_token_scaling:
            # Expand scaling factors to match gradient dimensions
            scaling_factors_expanded = scaling_factors.unsqueeze(-1)  # chunk_size x 1
            grad_logits_chunk = grad_logits_chunk * scaling_factors_expanded

        if input_requires_grad:
            grad_input[start_idx:end_idx] = grad_logits_chunk @ weight

        if grad_weight is not None and input_requires_grad:
            grad_weight += torch.mm(grad_logits_chunk.t(), _input_chunk).float()

        if bias is not None and input_requires_grad:
            torch.add(
                input=grad_bias,
                other=grad_logits_chunk.sum(dim=0),
                out=grad_bias,
                alpha=1.0,
            )

    # Need extra calculations for backward if reduction=='none'. Not supporting reduction='none' now.
    # if reduction == "none":
    #     loss = loss_1d
    #     z_loss = z_loss_1d if return_z_loss else None

    if reduction == "none":
        # Return per-token losses
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
        token_accuracy = token_accuracy_1d if return_token_accuracy else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        # For accuracy, we compute the mean across all non-ignored tokens
        token_accuracy = torch.sum(token_accuracy_1d) / total_n_non_ignore if return_token_accuracy else None

    # Cast back to original dtype
    grad_weight = grad_weight.to(weight.dtype) if grad_weight is not None else None
    grad_bias = grad_bias.to(bias.dtype) if grad_bias is not None else None

    return loss, z_loss, token_accuracy, grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight, grad_bias):
    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
    if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
        # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        # handle grad_weight
        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_weight,
                grad_weight.stride(-2),
                grad_output,
                H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )

        if grad_bias is not None:
            V = grad_bias.shape[0]
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_bias,
                grad_bias.stride(-1),
                grad_output,
                1,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )
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
    ):
        """
        Fusing the last linear layer with cross-entropy loss
            Reference: https://github.com/mgmalek/efficient_cross_entropy

        Handle the forward and backward pass of the final linear layer via cross-entropy loss by avoiding
        the materialization of the large logits tensor. Since Cross Entropy Loss is the last layer, we can
        compute the gradient at the forward pass. By doing so, we don't have to store the _input and target
        for the backward pass.

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
        """

        loss, z_loss, token_accuracy, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
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
        )
        # downcast to dtype and store for backward
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if bias is not None else None,
        )
        ctx.return_z_loss = return_z_loss
        ctx.return_token_accuracy = return_token_accuracy
        return loss, z_loss, token_accuracy

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2, grad_output3):
        if ctx.return_z_loss:
            del grad_output2  # z_loss is only for logging
        if ctx.return_token_accuracy:
            del grad_output3  # token_accuracy is only for metrics
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
            None,  # use_token_scaling
            None,  # return_token_accuracy
        )
