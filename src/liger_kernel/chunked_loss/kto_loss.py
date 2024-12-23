import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_unpaired_preference import LigerFusedLinearUnpairedPreferenceBase


class LigerFusedLinearKTOFunction(LigerFusedLinearUnpairedPreferenceBase):
    @staticmethod
    def preference_loss_fn(
        average_log_prob_chunk,
        preference_labels_chunk,
        full_target,
        ref_average_log_prob_chunk=None,
        beta=0.1,
        kl=None,
    ):
        """
        Implements the Kahneman-Tversky Optimization (KTO) loss function.
        Paper: "KTO: Model Alignment as Prospect Theory-Guided Optimization"
        https://arxiv.org/abs/2402.01306

        KTO loss is inspired by prospect theory (https://en.wikipedia.org/wiki/Prospect_theory)
        from behavioral economics, which models how humans make decisions under uncertainty.
        The loss function is asymmetric, treating gains and losses differently, similar to
        human decision-making patterns.

        Formula:
        When y is chosen:
        L_KTO = 1 - σ(β * (log[π(x)/π₀(x)] - KL(π||π₀)_y))
        When y is rejected:
        L_KTO = 1 - σ(β * (KL(π||π₀)_y - log[π(x)/π₀(x)]))

        Where:
        - σ: Sigmoid function
        - β: Temperature parameter controlling the strength of the preference signal
        - π(x): Policy (current model)
        - π₀(x): Reference policy (reference model)
        - KL(π||π₀)_y: KL divergence estimated using the rejected response y

        The loss encourages the model to:
        1. Assign higher probability to chosen responses
        2. Assign lower probability to rejected responses
        3. Maintain reasonable distance from the reference model

        Args:
            chosen_logps: Log probabilities of chosen tokens (batch_size,)
            rejected_logps: Log probabilities of rejected tokens (batch_size,)
            full_target: Non chunked full target tensor
            ref_chosen_logps: Reference log probs of chosen tokens (batch_size,)
            ref_rejected_logps: Reference log probs of rejected tokens (batch_size,)
            beta: Weight for the direct preference loss
            kl: KL divergence between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
        Returns:
            Tuple of (loss, chosen_rewards, rejected_rewards):
            - loss: The KTO loss value
            - chosen_rewards: Reward signals for chosen responses (detached)
            - rejected_rewards: Reward signals for rejected responses (detached)
        """
        logratios_chunk = average_log_prob_chunk - ref_average_log_prob_chunk
        multiplier_chunk = torch.where(preference_labels_chunk, 1, -1)
        if kl is not None:
            losses = 1 - F.sigmoid(beta * (logratios_chunk - kl) * multiplier_chunk)
        else:
            losses = 1 - F.sigmoid(beta * logratios_chunk * multiplier_chunk)

        return losses.sum() / (full_target.shape[0])

    @staticmethod
    def forward(
        ctx,
        _input,
        weight,
        target,
        preference_labels,
        bias=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        kl=None,
        ignore_index=-100,
        beta=0.1,
        compiled=True,
        use_ref_model=True,
    ):
        return LigerFusedLinearUnpairedPreferenceBase.forward(
            ctx=ctx,
            _input=_input,
            weight=weight,
            target=target,
            preference_labels=preference_labels,
            bias=bias,
            loss_fn=LigerFusedLinearKTOFunction.preference_loss_fn,
            ignore_index=ignore_index,
            beta=beta,
            compiled=compiled,
            use_ref_model=use_ref_model,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            kl=kl,
        )

    @staticmethod
    def backward(ctx, *grad_output):
        grads = LigerFusedLinearUnpairedPreferenceBase.backward(ctx, grad_output)[:5]
        return (
            *grads,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class LigerFusedLinearKTOLoss(torch.nn.Module):
    """
    Fused linear layer with Kahneman-Tversky Optimization (KTO) loss.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        compiled: bool = True,
        use_ref_model: bool = False,
    ):
        """
        Args:
            ignore_index (int): Index to ignore in the loss calculation
            beta (float): Temperature parameter for the KTO loss
            compiled (bool): Whether to use compiled operations
            use_ref_model (bool): Whether to use a reference model for the DPO loss.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.beta = beta
        self.compiled = compiled
        self.use_ref_model = use_ref_model

    def forward(
        self,
        _input,
        lin_weight,
        target,
        bias=None,
        preference_labels=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        kl=None,
    ):
        return LigerFusedLinearKTOFunction.apply(
            _input,
            lin_weight,
            target,
            preference_labels,
            bias,
            ref_input,
            ref_weight,
            ref_bias,
            kl,
            self.ignore_index,
            self.beta,
            self.compiled,
            self.use_ref_model,
        )
