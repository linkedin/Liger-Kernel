import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_preference import LigerFusedLinearPreferenceBase


class LigerFusedLinearSimPOFunction(LigerFusedLinearPreferenceBase):
    @staticmethod
    def preference_loss_fn(
        chosen_logps,
        rejected_logps,
        full_target,
        beta=0.1,
        gamma=0.5,
        label_smoothing=0.0,
    ):
        """
        Paper: https://arxiv.org/pdf/2405.14734

        Formula:
        L_SimPO(π_θ) = -E [log σ(β/|y_w| log π_θ(y_w|x) - β/|y_l| log π_θ(y_l|x) - γ)]

        Where:
        - π_θ(y|x): Policy (model) probability
        - y_w: Chosen sequence
        - y_l: Rejected sequence
        - |y_w|, |y_l|: Sequence lengths
        - σ: Sigmoid function
        - β: beta weight
        - γ: gemma margin term

        Args:
            chosen_logps (torch.Tensor): Avg log probabilities of chosen tokens. Shape: (batch_size,).
            rejected_logps (torch.Tensor): Avg log probabilities of rejected tokens. Shape: (batch_size,).
            full_target: Non chunked full target tensor
            beta (float): beta weight
            gamma (float): gemma margin term
            label_smoothing (float): Label smoothing factor, will reduce to Equation above when label_smoothing -> 0.
        """
        logits = beta * (chosen_logps - rejected_logps) - gamma
        loss = (-F.logsigmoid(logits) * (1 - label_smoothing) - F.logsigmoid(-logits) * label_smoothing).sum() / (
            full_target.shape[0] // 2
        )

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
        label_smoothing=0.0,
        compute_nll_loss=False,
        compiled=True,
        gamma=0.5,
    ):
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
            label_smoothing=label_smoothing,
            compiled=compiled,
            gamma=gamma,
        )

    @staticmethod
    def backward(ctx, *grad_output):
        grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
        return *grads, None, None, None, None, None, None, None


class LigerFusedLinearSimPOLoss(torch.nn.Module):
    """
    Fused linear layer with SimPO loss.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        alpha: float = 1.0,
        label_smoothing: float = 0.0,
        compute_nll_loss: bool = True,
        compiled: bool = True,
        gamma: float = 0.5,
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
        self.label_smoothing = label_smoothing
        self.compute_nll_loss = compute_nll_loss
        self.compiled = compiled
        self.gamma = gamma

    def forward(self, lin_weight, _input, target, bias=None):
        return LigerFusedLinearSimPOFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            self.ignore_index,
            self.beta,
            self.alpha,
            self.label_smoothing,
            self.compute_nll_loss,
            self.compiled,
            self.gamma,
        )
