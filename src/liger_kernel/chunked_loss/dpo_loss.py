from functools import partial

import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_preference import (
    LigerFusedLinearPreferenceBase,
)


def dpo_loss(chosen_logps, rejected_logps, beta=0.1):
    """
    Compute DPO loss (Direct Preference Optimization).
    Args:
        chosen_logps (torch.Tensor): Avg log probabilities of chosen tokens. Shape: (batch_size,).
        rejected_logps (torch.Tensor): Avg log probabilities of rejected tokens. Shape: (batch_size,).
        beta (float): Weight for the direct preference loss.
    """
    logits_diff = beta * (chosen_logps - rejected_logps)
    losses = -F.logsigmoid(logits_diff)
    return losses.sum()


def _compute_dpo_loss(
    input_chunk,
    weight,
    target_chunk,
    bias=None,
    full_target=None,
    ignore_index=-100,
    beta=0.1,
    compute_nll_loss=True,
):
    """
    Compute DPO loss for a chunk of input and target.
    Args:
        input_chunk (torch.Tensor): Chunk of input tensor. Shape: (2 * chunk_size, sequence_length, hidden_size).
        weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size).
        target_chunk (torch.Tensor): Chunk of target tensor. Shape: (2 * chunk_size, sequence_length).
        bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,).
        full_target (torch.Tensor): Full target tensor. Shape: (batch_size, sequence_length).
        ignore_index (int): Index to ignore for loss computation.
        beta (float): Weight for the direct preference loss.
    """

    len_chosen_chunk = target_chunk.shape[0] // 2

    logits_chunk = input_chunk @ weight.t()  # chunk_size x V
    if bias is not None:
        logits_chunk = logits_chunk + bias
    log_probs_chunk = F.log_softmax(
        logits_chunk.float(), dim=-1
    )  # Normalize the unnorm_logits

    # Compute NLL loss for chosen responses
    chosen_nll_loss = 0.0
    if compute_nll_loss:
        chosen_nll_loss = F.nll_loss(
            log_probs_chunk[:len_chosen_chunk].view(-1, log_probs_chunk.shape[-1]),
            target_chunk[:len_chosen_chunk].view(-1),
            reduction="sum",
            ignore_index=ignore_index,
        )
        chosen_nll_loss = (
            chosen_nll_loss
            / (full_target[: full_target.shape[0] // 2] != ignore_index).sum()
        )

    # Compute log probabilities for both chosen and rejected
    loss_mask = target_chunk != ignore_index
    label_chunk = torch.where(loss_mask, target_chunk, 0)

    per_token_logps = log_probs_chunk.gather(-1, label_chunk.unsqueeze(-1)).squeeze(-1)
    average_log_prob = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)

    chosen_logps = average_log_prob[:len_chosen_chunk]
    rejected_logps = average_log_prob[len_chosen_chunk:]

    # Compute DPO loss
    preference_loss = dpo_loss(chosen_logps, rejected_logps, beta=beta)
    preference_loss = preference_loss / (full_target.shape[0] // 2)

    # Total loss combines NLL and DPO loss
    loss = chosen_nll_loss + preference_loss
    return loss, (preference_loss, chosen_logps, rejected_logps)


class LigerFusedLinearDPOFunction(LigerFusedLinearPreferenceBase):
    @staticmethod
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ignore_index=-100,
        beta=0.1,
        compute_nll_loss=True,
        compiled=True,
    ):
        """
        Fused linear layer with DPO (Direct Preference Optimization) loss.
        Handles both the forward and backward pass of the final linear layer with DPO loss.
        Inspired from LigerFusedLinearCrossEntropyFunction (https://arxiv.org/abs/2410.10989) which fuses final linear layer and CE loss.
        """
        dpo_loss_fn = partial(
            _compute_dpo_loss,
            full_target=target,
            ignore_index=ignore_index,
            beta=beta,
            compute_nll_loss=compute_nll_loss,
        )
        return LigerFusedLinearPreferenceBase.forward(
            ctx, _input, weight, target, bias, loss_fn=dpo_loss_fn
        )

    @staticmethod
    def backward(ctx, grad_output):
        # Get gradients for _input, weight, bias, and target from the base class
        grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
        # Return these gradients, followed by None for the remaining inputs
        return *grads, None, None, None, None
