import torch.nn.functional as F
import torch
from liger_kernel.chunked_loss.fused_linear_preference import (
    LigerFusedLinearPreferenceBase,
)


class LigerFusedLinearDPOFunction(LigerFusedLinearPreferenceBase):

    @staticmethod
    def preference_loss_fn(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
        """
        Compute DPO loss (Direct Preference Optimization).
        Args:
            chosen_logps (torch.Tensor): Avg log probabilities of chosen tokens. Shape: (batch_size,).
            rejected_logps (torch.Tensor): Avg log probabilities of rejected tokens. Shape: (batch_size,).
            beta (float): Weight for the direct preference loss.
        """
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        logits_diff = beta * (chosen_logratios - rejected_logratios)
        losses = -F.logsigmoid(logits_diff)
        return losses.sum()

    @staticmethod
    def _compute_loss(
        input_chunk,
        weight,
        target_chunk,
        bias=None,
        preference_loss_fn=None,
        full_target=None,
        ignore_index=-100,
        alpha=1.0,
        beta=0.1,
        compute_nll_loss=True,
        ref_weight=None,
        ref_bias=None,
        **loss_kwargs,
    ):
        """
        Compute the total loss for a chunk of input and target, while using an alignment/preference loss function.
        Args:
            input_chunk (torch.Tensor): Chunk of input tensor. Shape: (2 * chunk_size, sequence_length, hidden_size).
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size).
            target_chunk (torch.Tensor): Chunk of target tensor. Shape: (2 * chunk_size, sequence_length).
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,).
            preference_loss_fn (callable): Loss function to compute the loss on a chunk of input/target.
            full_target (torch.Tensor): Full target tensor. Shape: (batch_size, sequence_length).
            ignore_index (int): Index to ignore for loss computation.
            alpha (float): Weight for the NLL loss.
            beta (float): Weight for the odds ratio loss.
            loss_kwargs (dict): Additional arguments for the loss function.
        """
        len_chosen_chunk = target_chunk.shape[0] // 2

        logits_chunk = input_chunk @ weight.t()  # chunk_size x V
        ref_logits_chunk = input_chunk @ ref_weight.t()  # chunk_size x V
        
        if bias is not None:
            logits_chunk = logits_chunk + bias
            ref_logits_chunk = ref_logits_chunk + ref_bias
        log_probs_chunk = F.log_softmax(logits_chunk.float(), dim=-1)
        ref_log_probs_chunk = F.log_softmax(ref_logits_chunk.float(), dim=-1)

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

        loss_mask = target_chunk != ignore_index
        label_chunk = torch.where(loss_mask, target_chunk, 0)

        per_token_logps = log_probs_chunk.gather(-1, label_chunk.unsqueeze(-1)).squeeze(
            -1
        )
        ref_per_token_logps = ref_log_probs_chunk.gather(-1, label_chunk.unsqueeze(-1)).squeeze(
            -1
        )
        average_log_prob = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        ref_average_log_prob = (ref_per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)

        chosen_logps = average_log_prob[:len_chosen_chunk]
        rejected_logps = average_log_prob[len_chosen_chunk:]
        ref_chosen_logps = ref_average_log_prob[:len_chosen_chunk]
        ref_rejected_logps = ref_average_log_prob[len_chosen_chunk:]

        alignment_loss = preference_loss_fn(
            chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=beta, **loss_kwargs
        )
        alignment_loss = alignment_loss / (full_target.shape[0] // 2)

        loss = alpha * chosen_nll_loss - alignment_loss
        return loss, (alignment_loss, chosen_logps, rejected_logps)

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
        """
        return LigerFusedLinearPreferenceBase.forward(
            ctx=ctx,
            _input=_input,
            weight=weight,
            target=target,
            bias=bias,
            chunk_fwd=LigerFusedLinearDPOFunction._compute_loss,
            loss_fn=LigerFusedLinearDPOFunction.preference_loss_fn,
            compute_nll_loss=compute_nll_loss,
            ignore_index=ignore_index,
            beta=beta,
            compiled=compiled,
        )

    @staticmethod
    def backward(ctx, grad_output):
        # Get gradients for _input, weight, bias, and target from the base class
        grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
        # Return these gradients, followed by None for the remaining inputs
        return *grads, None, None, None, None
