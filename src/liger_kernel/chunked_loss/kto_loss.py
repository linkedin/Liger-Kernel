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
        full_target,
        ref_chosen_logps=None,
        ref_rejected_logps=None,
        beta=0.1,
        policy_KL_logps=None,
        ref_KL_logps=None,
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
            policy_KL_logps: KL divergence between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
            ref_KL_logps: KL divergence between the reference model and the policy model for the chosen responses. Shape: (batch_size,)

        Returns:
            Tuple of (loss, chosen_rewards, rejected_rewards):
            - loss: The KTO loss value
            - chosen_rewards: Reward signals for chosen responses (detached)
            - rejected_rewards: Reward signals for rejected responses (detached)
        """
        if ref_chosen_logps is None:
            ref_chosen_logps = torch.tensor(0.0, device=chosen_logps.device)
        if ref_rejected_logps is None:
            ref_rejected_logps = torch.tensor(0.0, device=rejected_logps.device)

        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps

        if policy_KL_logps is None:
            policy_KL_logps = torch.tensor(0.0, device=chosen_logps.device)
        if ref_KL_logps is None:
            ref_KL_logps = torch.tensor(0.0, device=chosen_logps.device)

        kl = policy_KL_logps - ref_KL_logps

        losses = torch.cat(
            (
                1 - F.sigmoid(beta * (chosen_logratios - kl)),
                1 - F.sigmoid(beta * (kl - rejected_logratios)),
            ),
            0,
        )

        chosen_rewards = beta * chosen_logratios.detach()
        rejected_rewards = beta * rejected_logratios.detach()

        return (
            # We don't divide by 2 because KTO Loss doesn't need pair-wise examples
            losses.sum() / (full_target.shape[0]),
            chosen_rewards,
            rejected_rewards,
        )

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
        ignore_index=-100,
        beta=0.1,
        compute_nll_loss=True,
        compiled=True,
        use_ref_model=True,
        policy_KL_logps=None,
        ref_KL_logps=None,
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
            use_ref_model=use_ref_model,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            unpaired=True,  # KTO loss functions use unpaired preference
            preference_labels=preference_labels,
            policy_KL_logps=policy_KL_logps,
            ref_KL_logps=ref_KL_logps,
        )

    @staticmethod
    def backward(ctx, *grad_output):
        grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
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
        compute_nll_loss: bool = True,
        compiled: bool = True,
        use_ref_model: bool = False,
        policy_KL_logps: torch.FloatTensor = None,
        ref_KL_logps: torch.FloatTensor = None,
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
        self.policy_KL_logps = policy_KL_logps
        self.ref_KL_logps = ref_KL_logps

    def forward(
        self,
        _input,
        lin_weight,
        target,
        preference_labels,
        bias=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
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
            self.ignore_index,
            self.beta,
            self.compute_nll_loss,
            self.compiled,
            self.use_ref_model,
            self.policy_KL_logps,
            self.ref_KL_logps,
        )
