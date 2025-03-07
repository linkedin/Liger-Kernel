import torch

from liger_kernel.chunked_loss.fused_linear_rlhf import LigerFusedLinearRLHFBase


class LigerFusedLinearGRPOFunction(LigerFusedLinearRLHFBase):
    @staticmethod
    def rlhf_loss_fn(
        log_probs,
        attention_mask,
        rewards,
        ref_log_probs=None,
        beta=0.1,
        **kwargs,
    ):
        """GRPO Loss Function matching GRPOTrainer implementation."""
        # Get chosen token probabilities
        chosen_tokens = log_probs.argmax(dim=-1)  # (batch_size, seq_len)
        chosen_token_logprobs = log_probs.gather(dim=-1, index=chosen_tokens.unsqueeze(-1)).squeeze(
            -1
        )  # (batch_size, seq_len)

        # Get reference model probabilities
        if ref_log_probs is not None:
            with torch.no_grad():
                ref_token_logprobs = ref_log_probs.gather(dim=-1, index=chosen_tokens.unsqueeze(-1)).squeeze(-1)
        else:
            ref_token_logprobs = chosen_token_logprobs.detach()

        # Compute advantages per batch entry in a grouped fashion
        mean_grouped_rewards = rewards.mean()  # [batch_size,]
        std_grouped_rewards = rewards.std()  # [batch_size,]

        # Calculate advantages using the same epsilon as in GRPOTrainer
        eps = 1e-4
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + eps)

        # Compute policy gradient loss with importance sampling ratio
        ratio = torch.exp(chosen_token_logprobs - chosen_token_logprobs.detach())
        policy_loss = -ratio * advantages.unsqueeze(1)

        # Compute KL penalty
        kl_div = (
            torch.exp(ref_token_logprobs - chosen_token_logprobs) - (ref_token_logprobs - chosen_token_logprobs) - 1.0
        )

        # Combine losses
        per_token_loss = policy_loss + beta * kl_div

        # Apply masking and normalize
        masked_loss = per_token_loss * attention_mask
        seq_lengths = attention_mask.sum()
        seq_lengths = torch.clamp(seq_lengths, min=1.0)
        loss = masked_loss.sum() / seq_lengths

        # Calculate metrics
        metrics = (
            chosen_token_logprobs.mean(),  # mean log prob
            chosen_token_logprobs.std(),  # std log prob
            log_probs.mean(),  # mean all log probs
            ((kl_div * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)).mean(),  # mean KL div
        )

        return loss, metrics

    @classmethod
    def forward(
        cls,
        ctx,
        _input,
        weight,
        attention_mask,
        rewards,
        bias=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        beta=0.1,
        compiled=True,
        use_ref_model=True,
        num_generations=1,
        chunk_size=1,
    ):
        """
        Fused linear layer with GRPO loss.
        Args:
            _input (torch.Tensor): Input tensor. Shape: (batch_size * seq_len, hidden_size)
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size)
            attention_mask (torch.Tensor): Attention mask tensor. Shape: (batch_size, seq_len)
            rewards (torch.Tensor): Rewards tensor. Shape: (batch_size,)
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,)
            ref_input (torch.Tensor, optional): Reference model input tensor. Shape: (batch_size * seq_len, hidden_size)
            ref_weight (torch.Tensor, optional): Reference model weight tensor. Shape: (vocab_size, hidden_size)
            ref_bias (torch.Tensor, optional): Reference model bias tensor. Shape: (vocab_size,)
            beta (float): Weight for the KL penalty
            compiled (bool): Whether to use torch compile
            use_ref_model (bool): Whether to use a reference model
            num_generations (int): Number of generations per prompt
            chunk_size (int): Size of chunks for processing.
        Returns:
            torch.Tensor: Computed loss
        """
        return super().forward(
            cls=cls,
            ctx=ctx,
            _input=_input,
            weight=weight,
            attention_mask=attention_mask,
            rewards=rewards,
            bias=bias,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            beta=beta,
            compiled=compiled,
            use_ref_model=use_ref_model,
            num_generations=num_generations,
            chunk_size=chunk_size,
        )

    @staticmethod
    def backward(ctx, grad_output, *grad_metrics):
        """Backward pass for GRPO loss.

        Args:
            grad_output: Gradient of the loss (scalar)
            grad_metrics: Gradients of the metrics (not used in backward computation)
        """
        grads = LigerFusedLinearRLHFBase.backward(ctx, grad_output)
        return (
            *grads[:5],  # grad_input, grad_weight, grad_attention_mask, grad_rewards, grad_bias
            None,  # grad_ref_input
            None,  # grad_ref_weight
            None,  # grad_ref_bias
            None,  # grad_beta
            None,  # grad_compiled
            None,  # grad_use_ref_model
            None,  # grad_num_generations
            None,  # grad_chunk_size
        )


class LigerFusedLinearGRPOLoss(torch.nn.Module):
    """Fused linear layer with GRPO loss."""

    def __init__(
        self,
        beta: float = 0.1,
        compiled: bool = True,
        use_ref_model: bool = True,
        num_generations: int = 1,
        chunk_size: int = 1,
    ):
        """
        Args:
            beta (float): Weight for the KL penalty.
            compiled (bool): Whether to use torch compile.
            use_ref_model (bool): Whether to use a reference model.
            num_generations (int): Number of generations per prompt.
            chunk_size (int): Size of chunks for processing.
        """
        super().__init__()
        self.beta = beta
        self.compiled = compiled
        self.use_ref_model = use_ref_model
        self.num_generations = num_generations
        self.chunk_size = chunk_size

    def forward(
        self,
        _input,
        lin_weight,
        attention_mask,
        rewards,
        bias=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
    ):
        return LigerFusedLinearGRPOFunction.apply(
            _input,
            lin_weight,
            attention_mask,
            rewards,
            bias,
            ref_input,
            ref_weight,
            ref_bias,
            self.beta,
            self.compiled,
            self.use_ref_model,
            self.num_generations,
            self.chunk_size,
        )
