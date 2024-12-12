import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_preference import (
    LigerFusedLinearPreferenceBase,
)


class LigerFusedLinearKTOFunction(LigerFusedLinearPreferenceBase):

    @staticmethod
    def preference_loss_fn(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps=None,
        ref_rejected_logps=None,
        beta=0.1,
    ):
        """
        Paper: https://arxiv.org/abs/2402.01306

        Formula:
        L_KTO = 1 - σ(β * (log[π(x)/π₀(x)] - KL(π||π₀)_y))

        Where:
        - σ: Sigmoid function
        - β: Temperature parameter
        - KL(π||π₀)_y is KL divergence estimated using the rejected response y

        Args:
            chosen_logps: Log probabilities of chosen tokens (batch_size,)
            rejected_logps: Log probabilities of rejected tokens (batch_size,)
            ref_chosen_logps: Reference log probs of chosen tokens (batch_size,)
            ref_rejected_logps: Reference log probs of rejected tokens (batch_size,)
            beta: Weight for the direct preference loss
        """
        if ref_chosen_logps is None:
            ref_chosen_logps = torch.tensor(0.0, device=chosen_logps.device)
        if ref_rejected_logps is None:
            ref_rejected_logps = torch.tensor(0.0, device=rejected_logps.device)

        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps

        kl = torch.zeros(1).to(chosen_logps.device)
        # chosen_KL = chosen_logratios.mean().clamp(min=0)
        # rejected_KL = rejected_logratios.mean().clamp(min=0)

        losses = torch.cat(
            (
                1 - F.sigmoid(beta * (chosen_logratios - kl)),
                1 - F.sigmoid(beta * (kl - rejected_logratios)),
            ),
            0,
        )

        chosen_rewards = beta * chosen_logratios.detach()
        rejected_rewards = beta * rejected_logratios.detach()

        return losses, chosen_rewards, rejected_rewards

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
        return LigerFusedLinearPreferenceBase.forward(
            ctx=ctx,
            _input=_input,
            weight=weight,
            target=target,
            bias=bias,
            loss_fn=LigerFusedLinearKTOFunction.preference_loss_fn,
            ignore_index=ignore_index,
            beta=beta,
            compute_nll_loss=compute_nll_loss,
            compiled=compiled,
        )

    @staticmethod
    def backward(ctx, *grad_output):
        grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
        return *grads, None, None, None, None, None, None, None, None


class LigerFusedLinearKTOLoss(torch.nn.Module):
    """
    Fused linear layer with Kahneman-Tversky Optimization (KTO) loss.
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
            ignore_index (int): Index to ignore in the loss calculation
            beta (float): Temperature parameter for the KTO loss
            compute_nll_loss (bool): Whether to compute the NLL loss alongside KTO
            compiled (bool): Whether to use compiled operations
            use_ref_model (bool): Whether to use a reference model for the DPO loss.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.beta = beta
        self.compute_nll_loss = compute_nll_loss
        self.compiled = compiled
        self.use_ref_model = use_ref_model

    def forward(
        self,
        lin_weight,
        _input,
        target,
        bias=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
    ):
        return LigerFusedLinearKTOFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            ref_input,
            ref_weight,
            ref_bias,
            self.ignore_index,
            self.beta,
            self.compute_nll_loss,
            self.compiled,
            self.use_ref_model,
        )
