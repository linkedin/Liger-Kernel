import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_preference import (
    LigerFusedLinearPreferenceBase,
)


class LigerFusedLinearDPOFunction(LigerFusedLinearPreferenceBase):

    @staticmethod
    def preference_loss_fn(
        chosen_logps_chunk,
        rejected_logps_chunk,
        full_target,
        ref_chosen_logps_chunk=None,
        ref_rejected_logps_chunk=None,
        beta=0.1,
    ):
        """
        Paper: https://arxiv.org/pdf/2305.18290

        Formula:
        L_DPO = -E[ log_sigmoid( β * (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x))) ) ]

        Where:
        - π(y|x): Policy (model) probability
        - π_ref(y|x): Reference model probability
        - y_w: Chosen sequence
        - y_l: Rejected sequence
        - β: Weight for the direct preference loss
        - E: Expected value over the dataset

        Args:
            chosen_logps_chunk: Log probabilities of chosen tokens in the chunk (batch_size,)
            rejected_logps_chunk: Log probabilities of rejected tokens in the chunk (batch_size,)
            full_target: Non chunked full target tensor
            ref_chosen_logps_chunk: Reference log probs of chosen tokens in the chunk (batch_size,)
            ref_rejected_logps_chunk: Reference log probs of rejected tokens in the chunk (batch_size,)
            beta: Weight for the direct preference loss
        """

        if ref_chosen_logps_chunk is None:
            ref_chosen_logps_chunk = torch.tensor(0.0, device=chosen_logps_chunk.device)
        if ref_rejected_logps_chunk is None:
            ref_rejected_logps_chunk = torch.tensor(0.0, device=rejected_logps_chunk.device)

        chosen_logratios_chunk = chosen_logps_chunk - ref_chosen_logps_chunk
        rejected_logratios_chunk = rejected_logps_chunk - ref_rejected_logps_chunk

        logits_diff_chunk = beta * (chosen_logratios_chunk - rejected_logratios_chunk)
        loss_chunk = -F.logsigmoid(logits_diff_chunk).sum() / (full_target.shape[0] // 2)
        return loss_chunk

    @staticmethod
    def forward(
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
    ):
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
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
        )

    @staticmethod
    def backward(ctx, *grad_output):
        grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
        return *grads, None, None, None, None, None, None, None, None


class LigerFusedLinearDPOLoss(torch.nn.Module):
    """
    Fused linear layer with DPO loss.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        compute_nll_loss: bool = False,
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
        self,
        lin_weight,
        _input,
        target,
        bias=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
    ):
        return LigerFusedLinearDPOFunction.apply(
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
