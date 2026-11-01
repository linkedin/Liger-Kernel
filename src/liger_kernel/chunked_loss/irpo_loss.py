import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_preference import LigerFusedLinearPreferenceBase


class LigerFusedLinearIRPOFunction(LigerFusedLinearPreferenceBase):
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
        Implements the preference (DPO) term of the Iterative Reasoning Preference
        Optimization (IRPO) loss function.
        Paper: "Iterative Reasoning Preference Optimization"
        https://arxiv.org/abs/2404.19733

        IRPO trains on preferred/rejected chain-of-thought pairs gathered by iterative
        rejection sampling and optimizes a DPO term plus a negative log-likelihood term
        on the chosen response (Equation 1 of the paper):

        L_IRPO = L_DPO + α * L_NLL

        The DPO term computed here is:

        L_DPO = -E[ log_sigmoid( β * (log[π_θ(y_w|x)/π_ref(y_w|x)] - log[π_θ(y_l|x)/π_ref(y_l|x)]) ) ]

        Where:
        - σ: Sigmoid function
        - β: Temperature parameter for the preference term (0.1 in the paper)
        - π_θ(y|x): Policy (current model) sequence probability, i.e. the SUM of the
          per-token log probabilities (the paper does not length-normalize this term)
        - π_ref(y|x): Reference policy. In the paper this is the previous iteration's
          model M_t; in practice whatever reference model the caller provides
        - y_w: Chosen (preferred) response
        - y_l: Rejected response

        The NLL term α * L_NLL is a length-normalized NLL on the chosen response; it is
        composed by the LigerFusedLinearPreferenceBase (compute_nll_loss=True) and is
        not part of this function.

        Args:
            chosen_logps: Log probabilities of chosen tokens (batch_size,)
            rejected_logps: Log probabilities of rejected tokens (batch_size,)
            full_target: Non chunked full target tensor
            ref_chosen_logps: Reference log probs of chosen tokens (batch_size,)
            ref_rejected_logps: Reference log probs of rejected tokens (batch_size,)
            beta: Weight for the preference term
        """
        if ref_chosen_logps is None:
            ref_chosen_logps = torch.tensor(0.0, device=chosen_logps.device)
        if ref_rejected_logps is None:
            ref_rejected_logps = torch.tensor(0.0, device=rejected_logps.device)

        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps

        chosen_rewards = beta * chosen_logratios
        rejected_rewards = beta * rejected_logratios

        logits_diff = beta * (chosen_logratios - rejected_logratios)
        n_pairs = full_target.shape[0] // 2

        loss = -F.logsigmoid(logits_diff).sum() / n_pairs

        return loss, chosen_rewards, rejected_rewards

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
        alpha=1.0,
        compiled=True,
        use_ref_model=True,
        average_log_prob=False,
        chunk_size=1,
    ):
        """
        Fused linear layer with IRPO loss (DPO term + NLL term on the chosen response).
        Args:
            _input (torch.Tensor): Input tensor. Shape: (batch_size, seq_len, hidden_size)
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size)
            target (torch.LongTensor): Target tensor. Shape: (batch_size, seq_len)
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,)
            ref_input (torch.Tensor, optional): Reference model input tensor. Shape: (batch_size, seq_len, hidden_size)
            ref_weight (torch.Tensor, optional): Reference model weight tensor. Shape: (vocab_size, hidden_size)
            ref_bias (torch.Tensor, optional): Reference model bias tensor. Shape: (vocab_size,)
            ignore_index (int): Index to ignore in loss computation
            beta (float): Temperature parameter for the preference term
            alpha (float): Weight for the NLL loss component
            compiled (bool): Whether to use torch compile
            use_ref_model (bool): Whether to use a reference model
            average_log_prob (bool): Whether to average the log probability per non-masked token
                (the IRPO paper uses summed log probabilities, i.e. False)
            chunk_size (int): Size of chunks for processing
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
            alpha=alpha,
            compute_nll_loss=True,
            compiled=compiled,
            use_ref_model=use_ref_model,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            average_log_prob=average_log_prob,
            chunk_size=chunk_size,
        )

    @staticmethod
    def backward(ctx, *grad_output):
        grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
        return *grads, None, None, None, None, None, None, None, None, None, None


class LigerFusedLinearIRPOLoss(torch.nn.Module):
    """
    Fused linear layer with Iterative Reasoning Preference Optimization (IRPO) loss.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        alpha: float = 1.0,
        compiled: bool = True,
        use_ref_model: bool = True,
        average_log_prob: bool = False,
        chunk_size: int = 1,
    ):
        """
        Args:
            ignore_index (int): Index to ignore in the loss.
            beta (float): Temperature parameter for the preference term.
            alpha (float): Weight for the NLL loss component.
            compiled (bool): Whether to use the torch compiled kernel.
            use_ref_model (bool): Whether to use a reference model for the preference term.
            average_log_prob (bool): Whether to average the log probability per non-masked token.
                The IRPO paper uses summed log probabilities, i.e. False.
            chunk_size (int): Size of chunks for processing.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.beta = beta
        self.alpha = alpha
        self.compiled = compiled
        self.use_ref_model = use_ref_model
        self.average_log_prob = average_log_prob
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
        return LigerFusedLinearIRPOFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            ref_input,
            ref_weight,
            ref_bias,
            self.ignore_index,
            self.beta,
            self.alpha,
            self.compiled,
            self.use_ref_model,
            self.average_log_prob,
            self.chunk_size,
        )
