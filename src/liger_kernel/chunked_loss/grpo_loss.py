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

        # Compute advantages
        mean_grouped_rewards = rewards.mean()
        std_grouped_rewards = rewards.std()
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-8)

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
        seq_lengths = attention_mask.sum(dim=1, keepdim=True)
        seq_lengths = torch.clamp(seq_lengths, min=1.0)
        loss = (masked_loss.sum(dim=1) / seq_lengths.squeeze(-1)).mean()

        # Calculate metrics
        metrics = (
            chosen_token_logprobs.mean(),  # mean log prob
            chosen_token_logprobs.std(),  # std log prob
            log_probs.mean(),  # mean all log probs
            (kl_div * attention_mask).sum(1).mean() / attention_mask.sum(1).mean(),  # mean KL div
        )

        return loss, metrics

    @staticmethod
    def forward(
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
    ):
        return LigerFusedLinearRLHFBase.forward(
            ctx=ctx,
            _input=_input,
            weight=weight,
            attention_mask=attention_mask,
            loss_fn=LigerFusedLinearGRPOFunction.rlhf_loss_fn,
            rewards=rewards,
            bias=bias,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            beta=beta,
            compiled=compiled,
            use_ref_model=use_ref_model,
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
            *grads[:4],  # grad_input, grad_weight, grad_attention_mask, grad_rewards
            None,  # grad_bias
            None,  # grad_ref_input
            None,  # grad_ref_weight
            None,  # grad_ref_bias
            None,  # grad_beta
            None,  # grad_compiled
            None,  # grad_use_ref_model
        )


class LigerFusedLinearGRPOLoss(torch.nn.Module):
    """Fused linear layer with GRPO loss."""

    def __init__(
        self,
        beta: float = 0.1,
        compiled: bool = True,
        use_ref_model: bool = True,
    ):
        super().__init__()
        self.beta = beta
        self.compiled = compiled
        self.use_ref_model = use_ref_model

    def forward(
        self,
        lin_weight,
        _input,
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
        )
