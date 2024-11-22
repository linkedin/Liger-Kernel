import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_preference import (
    LigerFusedLinearPreferenceBase,
)


class LigerFusedLinearDPOFunction(LigerFusedLinearPreferenceBase):
    @staticmethod
    def preference_loss_fn(
        chosen_logps,
        rejected_logps,
        beta=0.1,
        ref_chosen_logps=None,
        ref_rejected_logps=None,
    ):
        """
        Compute DPO (Direct Preference Optimization) loss using policy and reference model probabilities.
        Args:
            chosen_logps (torch.Tensor): Policy model avg log probs of chosen tokens. Shape: (batch_size,).
            rejected_logps (torch.Tensor): Policy model avg log probs of rejected tokens. Shape: (batch_size,).
            beta (float): Temperature parameter for the DPO loss.
            ref_chosen_logps (torch.Tensor): Reference model avg log probs of chosen tokens. Shape: (batch_size,).
            ref_rejected_logps (torch.Tensor): Reference model avg log probs of rejected tokens. Shape: (batch_size,).
        """
        if ref_chosen_logps is None or ref_rejected_logps is None:
            raise ValueError("Reference model logits are required for DPO loss")

        chosen_advantages = chosen_logps - ref_chosen_logps
        rejected_advantages = rejected_logps - ref_rejected_logps

        logits = beta * (chosen_advantages - rejected_advantages)

        loss = -F.logsigmoid(logits).mean()
        return loss

    @staticmethod
    def forward(
        ctx,
        _input,
        weight,
        target,
        ref_chosen_logps,
        ref_rejected_logps,
        bias=None,
        ignore_index=-100,
        beta=0.1,
        compute_nll_loss=True,
        compiled=True,
    ):
        """
        Fused linear layer with DPO (Direct Preference Optimization) loss.
        Handles both the forward and backward pass of the final linear layer with DPO loss.

        Args:
            _input (torch.Tensor): Input tensor. Shape: (batch_size, seq_len, hidden_size).
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size).
            target (torch.Tensor): Target tensor. Shape: (batch_size, seq_len).
            ref_chosen_logps (torch.Tensor): Reference model log probs for chosen responses. Shape: (batch_size,).
            ref_rejected_logps (torch.Tensor): Reference model log probs for rejected responses. Shape: (batch_size,).
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,).
            ignore_index (int): Index to ignore in loss computation.
            beta (float): Temperature parameter for DPO loss.
            compute_nll_loss (bool): Whether to compute and add NLL loss.
            compiled (bool): Whether to use torch.compile for chunk accumulation.
        """
        # Create partial function with reference model logits
        partial_loss_fn = (
            lambda c, r, beta=beta: LigerFusedLinearDPOFunction.preference_loss_fn(
                c,
                r,
                beta=beta,
                ref_chosen_logps=ref_chosen_logps,
                ref_rejected_logps=ref_rejected_logps,
            )
        )

        return LigerFusedLinearPreferenceBase.forward(
            ctx=ctx,
            _input=_input,
            weight=weight,
            target=target,
            bias=bias,
            loss_fn=partial_loss_fn,
            compute_nll_loss=compute_nll_loss,
            ignore_index=ignore_index,
            beta=beta,
            compute_nll_loss=compute_nll_loss,
            compiled=compiled,
        )

    @staticmethod
    def backward(ctx, grad_output):
        # Get gradients from base class
        d_input, d_weight, d_target, d_bias = LigerFusedLinearPreferenceBase.backward(
            ctx, grad_output
        )[:4]
        # Return these gradients, followed by None for the remaining inputs
        return d_input, d_weight, d_target, None, None, d_bias, None, None, None


class LigerFusedLinearDPOLoss(torch.nn.Module):
    """
    Fused linear layer with DPO loss.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
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
        self.compute_nll_loss = compute_nll_loss
        self.compiled = compiled

    def forward(self, lin_weight, _input, target, bias=None):
        return LigerFusedLinearDPOFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            self.ignore_index,
            self.beta,
            self.compute_nll_loss,
            self.compiled,
        )
