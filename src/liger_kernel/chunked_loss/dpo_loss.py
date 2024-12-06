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
        full_target,
        ref_chosen_logps=None,
        ref_rejected_logps=None,
        beta=0.1,
    ):
        """
        Compute DPO loss (Direct Preference Optimization).
        Args:
            chosen_logps (torch.Tensor): Avg log probabilities of chosen tokens. Shape: (batch_size,).
            rejected_logps (torch.Tensor): Avg log probabilities of rejected tokens. Shape: (batch_size,).
            ref_chosen_logps (torch.Tensor, optional): Reference log probabilities of chosen tokens. Shape: (batch_size,).
            ref_rejected_logps (torch.Tensor, optional): Reference log probabilities of rejected tokens. Shape: (batch_size,).
            beta (float): Weight for the direct preference loss.
        """
        if ref_chosen_logps is None:
            ref_chosen_logps = torch.tensor(0.0, device=chosen_logps.device)
        if ref_rejected_logps is None:
            ref_rejected_logps = torch.tensor(0.0, device=rejected_logps.device)

        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps

        logits_diff = beta * (chosen_logratios - rejected_logratios)
        loss = -F.logsigmoid(logits_diff).sum() / (full_target.shape[0] // 2)
        return loss

    @staticmethod
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ref_weight=None,
        ref_bias=None,
        ignore_index=-100,
        beta=0.1,
        compute_nll_loss=True,
        compiled=True,
        use_ref_model=True,
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
            loss_fn=LigerFusedLinearDPOFunction.preference_loss_fn,
            ignore_index=ignore_index,
            beta=beta,
            compute_nll_loss=compute_nll_loss,
            compiled=compiled,
            use_ref_model=use_ref_model,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
        )

    @staticmethod
    def backward(ctx, *grad_output):
        # Get gradients for _input, weight, bias, and target from the base class
        grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
        # Return these gradients, followed by None for the remaining inputs
        return *grads, None, None, None, None, None, None, None


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
        use_ref_model: bool = False,
    ):
        """
        Args:
            ignore_index (int): Index to ignore in the loss.
            beta (float): Weight for the odds ratio loss.
            compute_nll_loss (bool): Whether to compute the NLL loss.
            compiled (bool): Whether to use the torch compiled kernel.
            use_ref_model (bool): Whether to use a reference model for the DPO loss.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.beta = beta
        self.compute_nll_loss = compute_nll_loss
        self.compiled = compiled
        self.use_ref_model = use_ref_model

    def forward(
        self, lin_weight, _input, target, bias=None, ref_weight=None, ref_bias=None
    ):
        return LigerFusedLinearDPOFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            ref_weight,
            ref_bias,
            self.ignore_index,
            self.beta,
            self.compute_nll_loss,
            self.compiled,
            self.use_ref_model,
        )
