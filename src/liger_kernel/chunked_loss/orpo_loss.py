import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_preference import (
    LigerFusedLinearPreferenceBase,
)


class LigerFusedLinearORPOFunction(LigerFusedLinearPreferenceBase):

    @staticmethod
    def preference_loss_fn(chosen_logps, rejected_logps, beta=0.1):
        """
        Compute odds-ratio loss.
        Args:
            chosen_logps (torch.Tensor): Avg log probabilities of chosen tokens. Shape: (batch_size,).
            rejected_logps (torch.Tensor): Avg log probabilities of rejected tokens. Shape: (batch_size,).
            beta (float): Weight for the odds ratio loss.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps))
            - torch.log1p(-torch.exp(rejected_logps))
        )
        ratio = F.logsigmoid(log_odds)
        return beta * ratio.sum()

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
        auto_tune_chunk_size=False,
        chunk_size=1,
    ):
        """
        Fused linear layer with ORPO (Odds-Ratio Preference Optimization) loss.
        Handles both the forward and backward pass of the final linear layer with ORPO loss.
        Inspired from LigerFusedLinearCrossEntropyFunction (https://arxiv.org/abs/2410.10989) which fuses final linear layer and CE loss.
        """
        if hasattr(ctx, "chunk_size"):
            chunk_size = ctx.chunk_size
            auto_tune_chunk_size = False
        else:
            chunk_size = chunk_size

        loss, chunk_size = LigerFusedLinearPreferenceBase.forward(
            ctx=ctx,
            _input=_input,
            weight=weight,
            target=target,
            bias=bias,
            loss_fn=LigerFusedLinearORPOFunction.preference_loss_fn,
            compute_nll_loss=compute_nll_loss,
            ignore_index=ignore_index,
            beta=beta,
            compiled=compiled,
            chunk_size=chunk_size,
            auto_tune_chunk_size=auto_tune_chunk_size,
        )
        ctx.chunk_size = chunk_size
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # Get gradients for _input, weight, bias, and target from the base class
        grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
        # Return these gradients, followed by None for the remaining inputs
        return *grads, None, None, None, None, None
