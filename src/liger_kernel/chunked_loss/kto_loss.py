import torch
import torch.nn.functional as F
from liger_kernel.chunked_loss.fused_linear_preference import (
    LigerFusedLinearPreferenceBase,
)


class LigerFusedLinearKTOFunction(LigerFusedLinearPreferenceBase):

    @staticmethod
    def preference_loss_fn(policy_chosen_logps,
                           policy_rejected_logps,
                           beta=0.1,
                           reference_logps=None,
                           labels=None):
        """
        Compute odds-ratio loss.
        Args:
            chosen_logps (torch.Tensor): Avg log probabilities of chosen tokens. Shape: (batch_size,).
            rejected_logps (torch.Tensor): Avg log probabilities of rejected tokens. Shape: (batch_size,).
            beta (float): Weight for the odds ratio loss.
        """
        desirable_weight = 1.0
        undesirable_weight = 1.0
        for i in range(reference_logps.shape[0]):
            chosen_idx = [i for i in range(reference_logps.shape[0]) if labels[i][0].item() == 1.]
            rejected_idx = [i for i in range(reference_logps.shape[0]) if labels[i][0].item() == 0.]
        reference_chosen_logps = reference_logps[chosen_idx, ...]
        reference_rejected_logps = reference_logps[rejected_idx, ...]

        if policy_chosen_logps.shape[0] != 0:
            chosen_rewards = (policy_chosen_logps.sum(-1) - reference_chosen_logps.sum(-1))
            chosen_losses = 1 - F.sigmoid(beta * (chosen_rewards - 0))
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([])
            chosen_rewards = torch.Tensor([])

        if policy_rejected_logps.shape[0] != 0:
            rejected_rewards = (policy_rejected_logps.sum(-1) - reference_rejected_logps.sum(-1))
            rejected_losses = 1 - F.sigmoid(beta * (0 - rejected_rewards))
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([])
            rejected_rewards = torch.Tensor([])
        losses = torch.cat(
            (desirable_weight * chosen_losses, undesirable_weight * rejected_losses),
            0)

        return losses.mean()

    @staticmethod
    def forward(
            ctx,
            _input,
            weight,
            target,
            bias=None,
            ignore_index=-100,
            beta=0.1,
            alpha=1.0,
            compute_nll_loss=False,
            compiled=True,
            reference_logps=None,
            labels=None
    ):
        """
        Fused linear layer with CPO (Odds-Ratio Preference Optimization) loss.
        Handles both the forward and backward pass of the final linear layer with CPO loss.
        Inspired from LigerFusedLinearCrossEntropyFunction (https://arxiv.org/abs/2410.10989) which fuses final linear layer and CE loss.
        """

        return LigerFusedLinearPreferenceBase.forward(
            ctx,
            _input,
            weight,
            target,
            bias,
            chunk_size=1,
            loss_fn=LigerFusedLinearKTOFunction.preference_loss_fn,
            compute_nll_loss=compute_nll_loss,
            ignore_index=ignore_index,
            alpha=alpha,
            beta=beta,
            compiled=compiled,
            reference_logps=reference_logps,
            labels=labels
        )

    @staticmethod
    def backward(ctx, grad_output):
        # Get gradients for _input, weight, bias, and target from the base class
        grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
        # Return these gradients, followed by None for the remaining inputs

        return *grads, None, None, None, None, None, None
