from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

import liger_kernel.transformers.functional as F

from liger_kernel.transformers.functional import CrossEntropyOutput


def unpack_cross_entropy_result(
    result,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if isinstance(result, CrossEntropyOutput):
        return result.loss, result.z_loss, result.token_accuracy

    if isinstance(result, tuple):
        loss = result[0]
        z_loss = result[1] if len(result) > 1 else None
        token_accuracy = result[2] if len(result) > 2 else None
        return loss, z_loss, token_accuracy

    return result, None, None


def fixed_fused_linear_cross_entropy(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    final_logit_softcapping: Optional[float] = None,
    accum_dtype: Optional[torch.dtype] = None,
    return_token_accuracy: bool = False,
    **kwargs,
):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    result = F.liger_fused_linear_cross_entropy(
        hidden_states,
        lm_head_weight,
        target,
        reduction=reduction,
        ignore_index=ignore_index,
        softcap=final_logit_softcapping,
        accum_dtype=accum_dtype,
        return_token_accuracy=return_token_accuracy,
        **kwargs,
    )

    loss, _, token_accuracy = unpack_cross_entropy_result(result)

    if reduction == "sum":
        loss = loss / num_items_in_batch

    if return_token_accuracy:
        return CrossEntropyOutput(loss=loss, token_accuracy=token_accuracy)

    return loss


def LigerForCausalLMLoss(
    hidden_states,
    lm_head_weight,
    labels,
    hidden_size: int,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    final_logit_softcapping: Optional[float] = None,
    return_token_accuracy: bool = False,
    **kwargs,
):
    # Skip upcast since intermediate values for the loss are all fp32 in kernel
    if shift_labels is None:
        # Shift so that token < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    hidden_states = hidden_states.view(-1, hidden_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(hidden_states.device)
    result = fixed_fused_linear_cross_entropy(
        hidden_states,
        lm_head_weight,
        shift_labels,
        num_items_in_batch,
        ignore_index,
        final_logit_softcapping,
        return_token_accuracy=return_token_accuracy,
        **kwargs,
    )
    return result
