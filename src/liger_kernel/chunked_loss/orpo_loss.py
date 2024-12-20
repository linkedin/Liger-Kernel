import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_preference import (
    LigerFusedLinearPreferenceBase,
)


class LigerFusedLinearORPOFunction(LigerFusedLinearPreferenceBase):

    @staticmethod
    def preference_loss_fn(chosen_logps_chunk, rejected_logps_chunk, full_target, beta=0.1):
        """
        Paper: https://arxiv.org/pdf/2403.07691

        Formula:
        Compute odds-ratio loss: L_OR = -log(σ(log(odds_θ(y_w|x) / odds_θ(y_l|x))))
        where odds_θ(y|x) = P_θ(y|x) / (1 - P_θ(y|x))

        Where:
        - P_θ(y|x): Policy (model) probability
        - y_w: Chosen sequence
        - y_l: Rejected sequence
        - σ: Sigmoid function
        - β: Weight for the odds ratio loss
        - odds_θ: Odds function for the policy

        Args:
            chosen_logps_chunk (torch.Tensor): Avg log probabilities of chosen tokens in the chunk. Shape: (batch_size,).
            rejected_logps_chunk (torch.Tensor): Avg log probabilities of rejected tokens in the chunk. Shape: (batch_size,).
            full_target (torch.Tensor): Non chunked full target tensor.
            beta (float): Weight for the odds ratio loss.
        """
        log_odds_chunk = (chosen_logps_chunk - rejected_logps_chunk) - (
            torch.log1p(-torch.exp(chosen_logps_chunk))
            - torch.log1p(-torch.exp(rejected_logps_chunk))
        )
        ratio_chunk = F.logsigmoid(log_odds_chunk)
        loss_chunk = beta * ratio_chunk.sum() / (full_target.shape[0] // 2)

        chosen_rewards_chunk = beta * chosen_logps_chunk
        rejected_rewards_chunk = beta * rejected_logps_chunk

        log_odds_ratio_chunk = torch.sum(ratio_chunk) / (full_target.shape[0] // 2)
        log_odds_chosen_chunk = torch.sum(log_odds_chunk) / (full_target.shape[0] // 2)

        return loss_chunk, chosen_rewards_chunk, rejected_rewards_chunk, log_odds_ratio_chunk, log_odds_chosen_chunk

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
            loss_fn=LigerFusedLinearORPOFunction.preference_loss_fn,
            ignore_index=ignore_index,
            beta=beta,
            compute_nll_loss=compute_nll_loss,
            compiled=compiled,
        )

    @staticmethod
    def backward(ctx, *grad_output):
        grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
        return *grads, None, None, None, None


class LigerFusedLinearORPOLoss(torch.nn.Module):
    """
    Fused linear layer with ORPO (Odds-Ratio Preference Optimization) loss.
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
        return LigerFusedLinearORPOFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            self.ignore_index,
            self.beta,
            self.compute_nll_loss,
            self.compiled,
        )
