import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_preference import LigerFusedLinearPreferenceBase


class LigerFusedLinearCPOFunction(LigerFusedLinearPreferenceBase):
    @staticmethod
    def preference_loss_fn(chosen_logps, rejected_logps, full_target, beta=0.1, label_smoothing=0.0):
        """
        Paper: https://arxiv.org/pdf/2401.08417

        Formula:
        L(π_θ; U) = -E_(x,y_w,y_l)~D[log σ(β log π_θ(y_w|x) - β log π_θ(y_l|x))]

        Where:
        - π_θ(y|x): Policy (model) probability
        - y_w: Chosen sequence
        - y_l: Rejected sequence
        - σ: Sigmoid function
        - β: Temperature parameter
        - E: Expected value over the dataset D
        - D: Dataset of preferences

        Args:
            chosen_logps (torch.Tensor): Avg log probabilities of chosen tokens. Shape: (batch_size,).
            rejected_logps (torch.Tensor): Avg log probabilities of rejected tokens. Shape: (batch_size,).
            full_target (torch.Tensor): Non chunked full target tensor
            beta (float): Weight for the CPO loss
            label_smoothing (float): Label smoothing factor, will reduce to Equation above when label_smoothing -> 0.
        """
        logits = beta * (chosen_logps - rejected_logps)
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
        compute_nll_loss=True,
        compiled=True,
    ):
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
            label_smoothing=label_smoothing,
            compute_nll_loss=compute_nll_loss,
            compiled=compiled,
        )

    @staticmethod
    def backward(ctx, *grad_output):
        grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
        return *grads, None, None, None, None, None, None


class LigerFusedLinearCPOLoss(torch.nn.Module):
    """
    Fused linear layer with CPO loss.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        alpha: float = 1.0,
        label_smoothing: float = 0.0,
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
        self.label_smoothing = label_smoothing
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
            self.label_smoothing,
            self.compute_nll_loss,
            self.compiled,
        )
