import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_unpaired_preference import LigerFusedLinearUnpairedPreferenceBase


class LigerFusedLinearBCOFunction(LigerFusedLinearUnpairedPreferenceBase):
    @staticmethod
    def preference_loss_fn(
        log_prob_chunk,
        preference_labels_chunk,
        full_target,
        ref_log_prob_chunk=None,
        beta=0.1,
        delta=None,
    ):
        """
        Implements the Binary Classifier Optimization (BCO) loss function.
        Paper: "Binary Classifier Optimization for Large Language Model Alignment"
        https://arxiv.org/abs/2404.04656

        BCO trains the model as a binary classifier on unpaired thumbs-up / thumbs-down
        feedback, where the classifier logit acts as an implicit reward.

        Formula:
        The implicit reward is the log-ratio between the policy and the reference model:

        r_θ(x, y) = β * log[π_θ(y|x) / π_ref(y|x)]

        When y is chosen (liked):
        L_BCO = -log σ(r_θ(x, y) - δ)
        When y is rejected (disliked):
        L_BCO = -log σ(-(r_θ(x, y) - δ))

        Where:
        - σ: Sigmoid function
        - β: Temperature parameter controlling the strength of the preference signal
        - π_θ(y|x): Policy (current model)
        - π_ref(y|x): Reference policy (reference model). When no reference model is
          used, the log-ratio reduces to the policy log probability.
        - δ: Reward shift estimated as the running mean of the implicit rewards over
          previous batches (cf. Theorem 3 of the paper and the `RunningMoments` state
          used by TRL's BCOTrainer). It is an external training state that must be
          maintained by the caller and passed in per batch, analogous to the `kl`
          argument of the KTO loss.

        Note: the density-ratio weighting of the rejected term (Equation 11 of the
        paper, "underlying distribution matching") is not implemented.

        Args:
            log_prob_chunk: Log probabilities for the chunk (batch_size,)
            preference_labels_chunk: Preference labels for the chunk (batch_size,)
            full_target: Non chunked full target tensor
            ref_log_prob_chunk: Reference log probs for the chunk (batch_size,)
            beta: Weight for the BCO loss
            delta: Reward shift (running mean of implicit rewards). Float or 0-dim tensor.
        Returns:
            - loss: The BCO loss value
            - chosen_rewards_sum: Sum of implicit rewards over chosen responses
            - rejected_rewards_sum: Sum of implicit rewards over rejected responses
        """
        if ref_log_prob_chunk is not None:
            logratios_chunk = log_prob_chunk - ref_log_prob_chunk
        else:
            logratios_chunk = log_prob_chunk

        if delta is None:
            delta = 0.0

        rewards_chunk = beta * logratios_chunk
        margins_chunk = rewards_chunk - delta
        losses = torch.where(
            preference_labels_chunk,
            -F.logsigmoid(margins_chunk),
            -F.logsigmoid(-margins_chunk),
        )

        chosen_rewards_sum = (rewards_chunk * preference_labels_chunk).sum()
        rejected_rewards_sum = (rewards_chunk * (~preference_labels_chunk)).sum()

        return losses.sum() / (full_target.shape[0]), chosen_rewards_sum, rejected_rewards_sum

    @classmethod
    def forward(
        cls,
        ctx,
        _input,
        weight,
        target,
        preference_labels,
        bias=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        delta=None,
        ignore_index=-100,
        beta=0.1,
        compiled=True,
        use_ref_model=True,
        average_log_prob=False,
        chunk_size=1,
    ):
        """
        Fused linear layer with BCO loss.
        Args:
            _input (torch.Tensor): Input tensor. Shape: (batch_size * seq_len, hidden_size)
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size)
            target (torch.LongTensor): Target tensor. Shape: (batch_size * seq_len,)
            preference_labels (torch.Tensor): Preference labels tensor. Shape: (batch_size,)
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,)
            ref_input (torch.Tensor, optional): Reference model input tensor. Shape: (batch_size * seq_len, hidden_size)
            ref_weight (torch.Tensor, optional): Reference model weight tensor. Shape: (vocab_size, hidden_size)
            ref_bias (torch.Tensor, optional): Reference model bias tensor. Shape: (vocab_size,)
            delta (float | torch.Tensor, optional): Reward shift (running mean of implicit rewards). Shape: ()
            ignore_index (int): Index to ignore in loss computation
            beta (float): Temperature parameter for the BCO loss
            compiled (bool): Whether to use torch compile
            use_ref_model (bool): Whether to use a reference model
            average_log_prob (bool): Whether to average the log probability per non-masked token
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
            preference_labels=preference_labels,
            bias=bias,
            ignore_index=ignore_index,
            beta=beta,
            compiled=compiled,
            use_ref_model=use_ref_model,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            average_log_prob=average_log_prob,
            delta=delta,
            chunk_size=chunk_size,
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


class LigerFusedLinearBCOLoss(torch.nn.Module):
    """
    Fused linear layer with Binary Classifier Optimization (BCO) loss.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        compiled: bool = True,
        use_ref_model: bool = False,
        average_log_prob: bool = False,
        chunk_size: int = 1,
    ):
        """
        Args:
            ignore_index (int): Index to ignore in the loss calculation
            beta (float): Temperature parameter for the BCO loss
            compiled (bool): Whether to use compiled operations
            use_ref_model (bool): Whether to use a reference model for the BCO loss.
            average_log_prob (bool): Whether to average the log probability per non-masked token
            chunk_size (int): Size of chunks for processing
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.beta = beta
        self.compiled = compiled
        self.use_ref_model = use_ref_model
        self.average_log_prob = average_log_prob
        self.chunk_size = chunk_size

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
        delta=None,
    ):
        return LigerFusedLinearBCOFunction.apply(
            _input,
            lin_weight,
            target,
            preference_labels,
            bias,
            ref_input,
            ref_weight,
            ref_bias,
            delta,
            self.ignore_index,
            self.beta,
            self.compiled,
            self.use_ref_model,
            self.average_log_prob,
            self.chunk_size,
        )
