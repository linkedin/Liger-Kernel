import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_preference import LigerFusedLinearPreferenceBase


class LigerFusedLinearORPOFunction(LigerFusedLinearPreferenceBase):
    @staticmethod
    def preference_loss_fn(chosen_logps, rejected_logps, full_target, beta=0.1):
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
            chosen_logps (torch.Tensor): Avg log probabilities of chosen tokens. Shape: (batch_size,).
            rejected_logps (torch.Tensor): Avg log probabilities of rejected tokens. Shape: (batch_size,).
            full_target (torch.Tensor): Non chunked full target tensor
            beta (float): Weight for the odds ratio loss.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        ratio = F.logsigmoid(log_odds)
        loss = -beta * ratio.sum() / (full_target.shape[0] // 2)

        chosen_rewards = beta * chosen_logps
        rejected_rewards = beta * rejected_logps

        log_odds_ratio = torch.sum(ratio) / (full_target.shape[0] // 2)
        log_odds_chosen = torch.sum(log_odds) / (full_target.shape[0] // 2)

        return loss, chosen_rewards, rejected_rewards, log_odds_ratio, log_odds_chosen

    @classmethod
    def forward(
        cls,
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        ignore_index=-100,
        beta=0.1,
        compute_nll_loss=False,
        compiled=True,
        use_ref_model=True,
        chunk_size=1024,
    ):
        """
        Fused linear layer with ORPO loss.
        Args:
            _input (torch.Tensor): Input tensor. Shape: (batch_size * seq_len, hidden_size)
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size)
            target (torch.LongTensor): Target tensor. Shape: (batch_size * seq_len,)
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,)
            ref_input (torch.Tensor, optional): Reference model input tensor. Shape: (batch_size * seq_len, hidden_size)
            ref_weight (torch.Tensor, optional): Reference model weight tensor. Shape: (vocab_size, hidden_size)
            ref_bias (torch.Tensor, optional): Reference model bias tensor. Shape: (vocab_size,)
            ignore_index (int): Index to ignore in loss computation
            beta (float): Weight for the odds ratio loss
            compute_nll_loss (bool): Whether to compute the NLL loss
            compiled (bool): Whether to use torch compile
            use_ref_model (bool): Whether to use a reference model
            chunk_size (int): Size of chunks for processing. Default: `1024`.
        Returns:
            torch.Tensor: Computed loss
        """
        return super().forward(
            cls=cls,
            ctx=ctx,
            _input=_input,
            weight=weight,
            target=target,
            bias=bias,
            ignore_index=ignore_index,
            beta=beta,
            compute_nll_loss=compute_nll_loss,
            compiled=compiled,
            use_ref_model=use_ref_model,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            chunk_size=chunk_size,
        )

    @staticmethod
    def backward(ctx, *grad_output):
        grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
        return *grads, None, None, None, None, None, None, None, None, None


class LigerFusedLinearORPOLoss(torch.nn.Module):
    """
    Fused linear layer with ORPO (Odds-Ratio Preference Optimization) loss.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        compute_nll_loss: bool = False,
        compiled: bool = True,
        use_ref_model: bool = True,
        chunk_size: int = 1024,
    ):
        """
        Args:
            ignore_index (int): Index to ignore in the loss.
            beta (float): Weight for the odds ratio loss.
            compute_nll_loss (bool): Whether to compute the NLL loss.
            compiled (bool): Whether to use the torch compiled kernel.
            use_ref_model (bool): Whether to use a reference model for the ORPO loss.
            chunk_size (int): Size of chunks for processing. Default: `1024`.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.beta = beta
        self.compute_nll_loss = compute_nll_loss
        self.compiled = compiled
        self.use_ref_model = use_ref_model
        self.chunk_size = chunk_size

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
        return LigerFusedLinearORPOFunction.apply(
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
            self.chunk_size,
        )
