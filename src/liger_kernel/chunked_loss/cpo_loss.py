import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_preference import (
    LigerFusedLinearPreferenceBase,
)


class LigerFusedLinearCPOFunction(LigerFusedLinearPreferenceBase):

    @staticmethod
    def preference_loss_fn(chosen_logps, rejected_logps, beta=0.1):
        """
        Compute odds-ratio loss.
        Args:
            chosen_logps (torch.Tensor): Avg log probabilities of chosen tokens. Shape: (batch_size,).
            rejected_logps (torch.Tensor): Avg log probabilities of rejected tokens. Shape: (batch_size,).
            beta (float): Weight for the odds ratio loss.
        """
        logits = beta * (chosen_logps - rejected_logps)
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
        compute_nll_loss=True,
        compiled=True,
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
            loss_fn=LigerFusedLinearCPOFunction.preference_loss_fn,
            ignore_index=ignore_index,
            alpha=alpha,
            beta=beta,
            compute_nll_loss=compute_nll_loss,
            compiled=compiled,
        )

    @staticmethod
    def backward(ctx, grad_output):
        # Get gradients for _input, weight, bias, and target from the base class
        grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
        # Return these gradients, followed by None for the remaining inputs
        return *grads, None, None, None, None, None


class LigerFusedLinearCPOLoss(torch.nn.Module):
    """
    Fused linear layer with CPO loss.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        alpha: float = 1.0,
        compute_nll_loss: bool = True,
        compiled: bool = True,
    ):
        """
        Args:
            ignore_index (int): Index to ignore in the loss.
            beta (float): Weight for the odds ratio loss.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.beta = beta
        self.alpha = alpha
        self.compute_nll_loss = compute_nll_loss
        self.compiled = compiled

    def forward(self, lin_weight, _input, target, bias=None):
        return LigerFusedLinearCPOFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            self.ignore_index,
            self.beta,
            self.alpha,
            self.compute_nll_loss,
            self.compiled,
        )
