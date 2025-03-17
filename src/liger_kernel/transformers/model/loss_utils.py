import torch.nn as nn

import liger_kernel.transformers.functional as F


def fixed_fused_linear_cross_entropy(
    hidden_states,
    lm_head_weight,
    target,
    num_items_in_batch: int = None,
    ignore_index: int = -100,
    **kwargs,
):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = F.liger_fused_linear_cross_entropy(
        hidden_states,
        lm_head_weight,
        target,
        reduction=reduction,
        ignore_index=ignore_index,
        **kwargs,
    )
    if reduction == "sum":
        loss = loss / num_items_in_batch

    return loss


def LigerForCausalLMLoss(
    hidden_states,
    lm_head_weight,
    labels,
    hidden_size: int,
    num_items_in_batch: int = None,
    ignore_index: int = -100,
    **kwargs,
):
    # Skip upcast since intermediate values for the loss are all fp32 in kernel
    labels = labels.to(hidden_states.device)
    # Shift so that token < n predict n
    labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    hidden_states = hidden_states.view(-1, hidden_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(hidden_states.device)
    loss = fixed_fused_linear_cross_entropy(
        hidden_states,
        lm_head_weight,
        shift_labels,
        num_items_in_batch,
        ignore_index,
        **kwargs,
    )
    return loss
