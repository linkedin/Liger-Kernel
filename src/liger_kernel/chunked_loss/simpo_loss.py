import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_preference import (
    LigerFusedLinearPreferenceBase,
)


class LigerFusedLinearSimPOFunction(LigerFusedLinearPreferenceBase):

    @staticmethod
    def preference_loss_fn(chosen_logps, rejected_logps, beta=0.1, gamma=0.5):
        """
        Compute odds-ratio loss.
        Args:
            chosen_logps (torch.Tensor): Avg log probabilities of chosen tokens. Shape: (batch_size,).
            rejected_logps (torch.Tensor): Avg log probabilities of rejected tokens. Shape: (batch_size,).
            beta (float): Weight for the odds ratio loss.
            gamma (float): The simpo gamma, margin term.
        """
        logits = beta * (chosen_logps - rejected_logps) - gamma
        loss = F.logsigmoid(logits).mean()
        return loss

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
        gamma=0.5,
    ):
        """
        Fused linear layer with SimPO (Simple Preference Optimization) loss. https://arxiv.org/pdf/2405.14734
        Handles both the forward and backward pass of the final linear layer with SimPO loss.
        Inspired from LigerFusedLinearCrossEntropyFunction (https://arxiv.org/abs/2410.10989) which fuses final linear layer and CE loss.
        """

        return LigerFusedLinearPreferenceBase.forward(
            ctx,
            _input,
            weight,
            target,
            bias,
            loss_fn=LigerFusedLinearSimPOFunction.preference_loss_fn,
            compute_nll_loss=compute_nll_loss,
            ignore_index=ignore_index,
            alpha=alpha,
            beta=beta,
            compiled=compiled,
            gamma=gamma,
        )

    @staticmethod
    def backward(ctx, grad_output):
        # Get gradients for _input, weight, bias, and target from the base class
        grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
        # Return these gradients, followed by None for the remaining inputs
        return *grads, None, None, None, None, None, None
