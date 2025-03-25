import torch

from liger_kernel.chunked_loss.fused_linear_rlhf import LigerFusedLinearRLHFBase


class LigerFusedLinearGRPOFunction(LigerFusedLinearRLHFBase):
    @staticmethod
    def rlhf_loss_fn(
        log_probs,
        attention_mask,
        advantages,
        full_attention_mask,
        ref_log_probs=None,
        old_per_token_logps=None,
        epsilon_low=0.2,
        epsilon_high=0.2,
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

        # Compute policy gradient loss with importance sampling ratio
        old_per_token_logps = old_per_token_logps if old_per_token_logps is not None else chosen_token_logprobs.detach()
        coef_1 = torch.exp(chosen_token_logprobs - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if beta != 0.0:
            # Compute KL penalty
            kl_div = (
                torch.exp(ref_token_logprobs - chosen_token_logprobs) - (ref_token_logprobs - chosen_token_logprobs) - 1.0
            )
            # Combine losses
            per_token_loss = per_token_loss + beta * kl_div

        # Apply mask and compute average loss
        loss = (per_token_loss * attention_mask).sum() / torch.clamp(full_attention_mask.sum(), min=1.0)

        # Calculate metrics
        full_batch_size, seq_len = full_attention_mask.shape
        vocab_size = log_probs.shape[2]
        metrics = [
            chosen_token_logprobs.sum() / (full_batch_size * seq_len),  # mean log prob
            log_probs.sum() / (full_batch_size * seq_len * vocab_size),  # mean all log probs
        ]
        if beta != 0.0:
            metrics.append(
                ((kl_div * attention_mask).sum(dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1.0)).sum() / full_batch_size
            )
        return loss, metrics

    @classmethod
    def forward(
        cls,
        ctx,
        _input,
        weight,
        attention_mask,
        advantages,
        bias=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        old_per_token_logps=None,
        beta=0.1,
        epsilon_low=0.2,
        epsilon_high=0.2,
        temperature=1.0,
        compiled=True,
        use_ref_model=True,
        chunk_size=1,
    ):
        """
        Fused linear layer with GRPO loss.
        Args:
            _input (torch.Tensor): Input tensor. Shape: (batch_size * seq_len, hidden_size)
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size)
            attention_mask (torch.Tensor): Attention mask tensor. Shape: (batch_size, seq_len)
            advantages (torch.Tensor): Advantages tensor. Shape: (batch_size,)
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,)
            ref_input (torch.Tensor, optional): Reference model input tensor. Shape: (batch_size * seq_len, hidden_size)
            ref_weight (torch.Tensor, optional): Reference model weight tensor. Shape: (vocab_size, hidden_size)
            ref_bias (torch.Tensor, optional): Reference model bias tensor. Shape: (vocab_size,)
            beta (float): Weight for the KL penalty
            temperature (float): Temperature for the logits
            compiled (bool): Whether to use torch compile
            use_ref_model (bool): Whether to use a reference model
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
            advantages=advantages,
            bias=bias,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            old_per_token_logps=old_per_token_logps,
            beta=beta,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            temperature=temperature,
            compiled=compiled,
            use_ref_model=use_ref_model,
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
            *grads[:5],  # grad_input, grad_weight, grad_attention_mask, grad_advantages, grad_bias
            None,  # grad_ref_input
            None,  # grad_ref_weight
            None,  # grad_ref_bias
            None,  # grad_old_per_token_logps
            None,  # grad_beta
            None,  # grad_epsilon_low
            None,  # grad_epsilon_high
            None,  # grad_temperature
            None,  # grad_compiled
            None,  # grad_use_ref_model
            None,  # grad_chunk_size
        )


class LigerFusedLinearGRPOLoss(torch.nn.Module):
    """Fused linear layer with GRPO loss."""

    def __init__(
        self,
        beta: float = 0.1,
        compiled: bool = True,
        use_ref_model: bool = True,
        chunk_size: int = 1,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.2,
        temperature: float = 1.0,
    ):
        """
        Args:
            beta (float): Weight for the KL penalty.
            compiled (bool): Whether to use torch compile.
            use_ref_model (bool): Whether to use a reference model.
            chunk_size (int): Size of chunks for processing.
            epsilon_low (float): Lower bound for the importance sampling ratio.
            epsilon_high (float): Upper bound for the importance sampling ratio.
            temperature (float): Temperature for the logits.
        """
        super().__init__()
        self.beta = beta
        self.compiled = compiled
        self.use_ref_model = use_ref_model
        self.chunk_size = chunk_size
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.temperature = temperature

    def forward(
        self,
        _input,
        lin_weight,
        attention_mask,
        advantages,
        bias=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        old_per_token_logps=None,
    ):
        return LigerFusedLinearGRPOFunction.apply(
            _input,
            lin_weight,
            attention_mask,
            advantages,
            bias,
            ref_input,
            ref_weight,
            ref_bias,
            old_per_token_logps,
            self.epsilon_low,
            self.epsilon_high,
            self.beta,
            self.temperature,
            self.compiled,
            self.use_ref_model,
            self.chunk_size,
        )
