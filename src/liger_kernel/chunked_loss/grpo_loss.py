from typing import Optional

import torch

from liger_kernel.chunked_loss.fused_linear_ppo import LigerFusedLinearPPOBase


def k3_loss_fn(log_p, log_q):
    # computes k3 estimate of KL[q, p]
    # ref: http://joschu.net/blog/kl-approx.html
    return torch.exp(log_p - log_q) - (log_p - log_q) - 1.0


def clip_coef_fn(coef, epsilon_low, epsilon_high):
    return torch.clamp(coef, 1 - epsilon_low, 1 + epsilon_high)


class LigerFusedLinearGRPOFunction(LigerFusedLinearPPOBase):
    @staticmethod
    def ppo_loss_fn(
        per_token_logps,
        selected_token_ids,
        attention_mask,
        advantages,
        full_attention_mask,
        ref_per_token_logps=None,  # shape: [chunk_size, seq_len]
        old_per_token_logps=None,
        epsilon_low=0.2,
        epsilon_high=0.2,
        beta=0.04,
        loss_type="dapo",  # ["grpo", "bnpo", "dr_grpo", "dapo"]
        max_completion_length=None,  # Required for dr_grpo
        importance_sampling_level="token",  # ["token", "sequence"] - new parameter for GSPO
        num_items_in_batch=None,  # Total tokens in generation batch, required for dapo to match TRL
        importance_sampling_ratio=None,  # vLLM importance sampling correction ratio
        **kwargs,
    ):
        """GRPO Loss Function matching GRPOTrainer implementation.

        Args:
            per_token_logps: [B, T] log probabilities for selected tokens (already gathered)
            selected_token_ids: [B, T] selected token IDs (not used anymore, kept for compatibility)
            ...
        """
        # per_token_logps is now already gathered, no need for .gather() operation

        # Get reference model probabilities
        if ref_per_token_logps is None:
            ref_per_token_logps = per_token_logps.detach()

        # Compute policy gradient loss with importance sampling ratio
        old_per_token_logps = old_per_token_logps if old_per_token_logps is not None else per_token_logps.detach()
        log_ratio = per_token_logps - old_per_token_logps

        if importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * attention_mask).sum(-1) / attention_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)
        coef_1 = torch.exp(log_importance_weights)
        coef_2 = clip_coef_fn(coef_1, epsilon_low, epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Apply vLLM importance sampling correction if provided
        # This corrects for the mismatch between vLLM completion logprobs and recomputed training logprobs
        # IMPORTANT: This must be applied BEFORE adding KL penalty (to match TRL's implementation)
        if importance_sampling_ratio is not None:
            per_token_loss = per_token_loss * importance_sampling_ratio

        if beta != 0.0:
            # Compute KL penalty (approximates KL[per_token_logps, ref_per_token_logps])
            kl_div = k3_loss_fn(ref_per_token_logps, per_token_logps)
            # Combine losses
            per_token_loss = per_token_loss + beta * kl_div

        # Note: We normalize by the number of tokens in the batch (using full_attention_mask),
        # which is consistent with the DAPO loss implementation (https://arxiv.org/html/2503.14476v1)
        # and TRL GRPO implementation
        # (https://github.com/huggingface/trl/blob/e751a16df56e70190fb94bed4a2035eec3303777/trl/trainer/grpo_trainer.py#L966)
        if loss_type == "grpo":
            # Average per-sequence loss
            loss = (
                (per_token_loss * attention_mask).sum(-1) / torch.clamp(attention_mask.sum(-1), min=1.0)
            ).sum() / full_attention_mask.shape[0]
        elif loss_type == "bnpo":
            # Batch Normalized Per-token loss (original implementation)
            loss = (per_token_loss * attention_mask).sum() / torch.clamp(full_attention_mask.sum(), min=1.0)
        elif loss_type == "dr_grpo":
            # Dimension-Reduced GRPO (normalize by batch_size * max_completion_length)
            if max_completion_length is None:
                raise ValueError("max_completion_length must be provided for loss_type 'dr_grpo'")
            loss = (per_token_loss * attention_mask).sum() / (full_attention_mask.shape[0] * max_completion_length)
        elif loss_type == "dapo":
            # For dapo, use num_items_in_batch if provided (matches TRL's implementation)
            # TRL normalizes by num_items_in_batch / num_processes, where num_items_in_batch
            # is the total completion tokens across the entire generation batch.
            # If num_items_in_batch is not provided, fall back to attention_mask.sum()
            if num_items_in_batch is not None:
                loss_normalizer = torch.clamp(num_items_in_batch, min=1.0)
            else:
                loss_normalizer = torch.clamp(full_attention_mask.sum(), min=1.0)
            loss = (per_token_loss * attention_mask).sum() / loss_normalizer
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Calculate metrics
        metrics = []
        if beta != 0.0:
            metrics.append(((kl_div * attention_mask).sum() / torch.clamp(full_attention_mask.sum(), min=1.0)))

        # Adjust clipping metric calculation based on importance sampling level
        if importance_sampling_level == "token":
            is_clipped = ((coef_1 < 1 - epsilon_low) & (advantages.unsqueeze(1) < 0)) | (
                (coef_1 > 1 + epsilon_high) & (advantages.unsqueeze(1) > 0)
            )
        else:  # sequence level
            # For sequence level, coef_1 is shape (B, 1), advantages is shape (B,)
            is_clipped = ((coef_1.squeeze(-1) < 1 - epsilon_low) & (advantages < 0)) | (
                (coef_1.squeeze(-1) > 1 + epsilon_high) & (advantages > 0)
            )
            is_clipped = is_clipped.unsqueeze(1).expand_as(attention_mask)

        metrics.append((is_clipped * attention_mask).sum() / torch.clamp(full_attention_mask.sum(), min=1.0))
        return loss, metrics

    @classmethod
    def forward(
        cls,
        ctx,
        _input,
        weight,
        selected_token_ids,
        attention_mask,
        advantages,
        bias=None,
        ref_per_token_logps=None,
        old_per_token_logps=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        beta=0.04,
        epsilon_low=0.2,
        epsilon_high=0.2,
        loss_type="dapo",
        max_completion_length=None,
        importance_sampling_level="token",
        temperature=1.0,
        compiled=True,
        use_ref_model=True,
        chunk_size=1,
        num_items_in_batch=None,
        importance_sampling_ratio=None,
    ):
        """
        Fused linear layer with GRPO loss.
        Args:
            _input (torch.Tensor): Input tensor. Shape: (batch_size * seq_len, hidden_size)
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size)
            selected_token_ids (torch.Tensor): Selected token ids tensor. Shape: (batch_size, seq_len)
            attention_mask (torch.Tensor): Attention mask tensor. Shape: (batch_size, seq_len)
            advantages (torch.Tensor): Advantages tensor. Shape: (batch_size,)
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,)
            ref_per_token_logps:  Reference model log probs per token tensor. Shape:(batch_size, seq_len)
            ref_input (torch.Tensor, optional): Reference model input tensor. Shape: (batch_size * seq_len, hidden_size)
            ref_weight (torch.Tensor, optional): Reference model weight tensor. Shape: (vocab_size, hidden_size)
            ref_bias (torch.Tensor, optional): Reference model bias tensor. Shape: (vocab_size,)
            beta (float): Weight for the KL penalty
            loss_type (str): Type of loss calculation ("grpo", "bnpo", "dr_grpo", "dapo"). Defaults to "dapo".
            max_completion_length (int, optional): Maximum completion length, required for "dr_grpo". Defaults to None.
            importance_sampling_level (str): Level of importance sampling ("token" or "sequence"). Defaults to "token".
            temperature (float): Temperature for the logits
            compiled (bool): Whether to use torch compile
            use_ref_model (bool): Whether to use a reference model
            chunk_size (int): Size of chunks for processing.
            num_items_in_batch (float, optional): Total tokens in generation batch for dapo normalization (matches TRL).
            importance_sampling_ratio (torch.Tensor, optional): vLLM importance sampling correction ratio.
        Returns:
            torch.Tensor: Computed loss
        """
        return super().forward(
            cls=cls,
            ctx=ctx,
            _input=_input,
            weight=weight,
            selected_token_ids=selected_token_ids,
            attention_mask=attention_mask,
            advantages=advantages,
            bias=bias,
            ref_per_token_logps=ref_per_token_logps,
            old_per_token_logps=old_per_token_logps,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            beta=beta,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            loss_type=loss_type,
            max_completion_length=max_completion_length,
            temperature=temperature,
            compiled=compiled,
            use_ref_model=use_ref_model,
            chunk_size=chunk_size,
            importance_sampling_level=importance_sampling_level,
            num_items_in_batch=num_items_in_batch,
            importance_sampling_ratio=importance_sampling_ratio,
        )

    @staticmethod
    def backward(ctx, grad_output, *grad_metrics):
        """Backward pass for GRPO loss.

        Args:
            grad_output: Gradient of the loss (scalar)
            grad_metrics: Gradients of the metrics (not used in backward computation)
        """
        grads = LigerFusedLinearPPOBase.backward(ctx, grad_output)
        return (
            *grads[
                :6
            ],  # grad_input, grad_weight, grad_selected_token_ids, grad_attention_mask, grad_advantages, grad_bias
            None,  # grad_ref_per_token_logps
            None,  # grad_old_per_token_logps
            None,  # grad_ref_input
            None,  # grad_ref_weight
            None,  # grad_ref_bias
            None,  # grad_beta
            None,  # grad_epsilon_low
            None,  # grad_epsilon_high
            None,  # grad_loss_type (string, not differentiable)
            None,  # grad_max_completion_length (int, not differentiable)
            None,  # grad_importance_sampling_level (string, not differentiable)
            None,  # grad_temperature
            None,  # grad_compiled
            None,  # grad_use_ref_model
            None,  # grad_chunk_size
            None,  # grad_num_items_in_batch
            None,  # grad_importance_sampling_ratio
        )


class LigerFusedLinearGRPOLoss(torch.nn.Module):
    """Fused linear layer with GRPO loss."""

    def __init__(
        self,
        beta: float = 0.04,
        compiled: bool = True,
        use_ref_model: bool = True,
        chunk_size: int = 1,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.2,
        loss_type: str = "dapo",
        max_completion_length: Optional[int] = None,
        importance_sampling_level: str = "token",
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
            loss_type (str): Type of loss calculation ("grpo", "bnpo", "dr_grpo", "dapo"). Defaults to "dapo".
            max_completion_length (int, optional): Maximum completion length, required for "dr_grpo". Defaults to None.
            importance_sampling_level (str): Level of importance sampling ("token" or "sequence"). Defaults to "token".
            temperature (float): Temperature for the logits.
        """
        super().__init__()
        self.beta = beta
        self.compiled = compiled
        self.use_ref_model = use_ref_model
        self.chunk_size = chunk_size
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.loss_type = loss_type
        self.max_completion_length = max_completion_length
        self.importance_sampling_level = importance_sampling_level
        self.temperature = temperature

    def forward(
        self,
        _input,
        lin_weight,
        selected_token_ids,
        attention_mask,
        advantages,
        bias=None,
        ref_per_token_logps=None,
        old_per_token_logps=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        num_items_in_batch=None,
        importance_sampling_ratio=None,
    ):
        return LigerFusedLinearGRPOFunction.apply(
            _input,
            lin_weight,
            selected_token_ids,
            attention_mask,
            advantages,
            bias,
            ref_per_token_logps,
            old_per_token_logps,
            ref_input,
            ref_weight,
            ref_bias,
            self.beta,
            self.epsilon_low,
            self.epsilon_high,
            self.loss_type,
            self.max_completion_length,
            self.importance_sampling_level,
            self.temperature,
            self.compiled,
            self.use_ref_model,
            self.chunk_size,
            num_items_in_batch,
            importance_sampling_ratio,
        )
