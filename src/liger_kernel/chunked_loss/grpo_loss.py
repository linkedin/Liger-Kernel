import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_rlhf import LigerFusedLinearRLHFBase


class LigerFusedLinearGRPOFunction(LigerFusedLinearRLHFBase):
    @staticmethod
    def preference_loss_fn(
        logits,
        attention_mask,
        rewards,
        ref_logits=None,
        beta=0.1,
        **kwargs,
    ):
        """
        GRPO Loss Function as implemented in GRPOTrainer.

        Args:
            logits: Model logits (batch_size, seq_len, vocab_size)
            attention_mask: Attention mask (batch_size, seq_len)
            rewards: Rewards for each sequence (batch_size,)
            ref_logits: Reference model logits (batch_size, seq_len, vocab_size) or None
            beta: Weight for KL penalty
        """
        # Get log probabilities for policy
        log_probs = F.log_softmax(logits, dim=-1)

        # Get sequence-level log probs by taking max over vocab and summing over sequence
        policy_seq_logps = (log_probs.max(dim=-1).values * attention_mask).sum(dim=-1)

        # Get reference model log probabilities if provided
        if ref_logits is not None:
            with torch.no_grad():
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                ref_seq_logps = (ref_log_probs.max(dim=-1).values * attention_mask).sum(dim=-1)
        else:
            ref_seq_logps = policy_seq_logps.detach()

        # Compute advantages
        advantages = rewards - rewards.mean()
        if advantages.std() > 0:
            advantages = advantages / advantages.std()

        # Policy gradient loss
        policy_loss = -(advantages * policy_seq_logps)

        # KL penalty
        kl_div = policy_seq_logps - ref_seq_logps

        # Total loss
        loss = policy_loss + beta * kl_div

        return loss.mean()

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
        """Forward pass for GRPO loss."""
        # Save tensors needed for backward
        ctx.save_for_backward(_input, weight, attention_mask, bias)
        ctx.beta = beta
        ctx.rewards = rewards  # Save rewards for use in backward pass

        # Get policy logits
        batch_size, seq_len, hidden_size = _input.shape
        input_reshaped = _input.view(-1, hidden_size)
        policy_logits = (input_reshaped @ weight.t()).view(batch_size, seq_len, -1)
        if bias is not None:
            policy_logits = policy_logits + bias

        # Get reference logits if needed
        ref_logits = None
        if use_ref_model and ref_input is not None and ref_weight is not None:
            ref_input_reshaped = ref_input.view(-1, ref_input.size(-1))
            ref_logits = (ref_input_reshaped @ ref_weight.t()).view(batch_size, seq_len, -1)
            if ref_bias is not None:
                ref_logits = ref_logits + ref_bias

        # Get log probabilities
        log_probs = F.log_softmax(policy_logits, dim=-1)
        seq_log_probs = (log_probs.max(dim=-1).values * attention_mask).sum(dim=-1)

        # Get reference log probabilities
        if ref_logits is not None:
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_seq_logps = (ref_log_probs.max(dim=-1).values * attention_mask).sum(dim=-1)
        else:
            ref_seq_logps = seq_log_probs.detach()

        # Compute KL divergence
        kl_div = seq_log_probs - ref_seq_logps

        # Compute loss
        loss = LigerFusedLinearGRPOFunction.preference_loss_fn(
            logits=policy_logits,
            attention_mask=attention_mask,
            rewards=rewards,
            ref_logits=ref_logits,
            beta=beta,
        )

        # Return metrics matching the PyTorch implementation
        metrics = (
            seq_log_probs.mean(),  # policy log probs mean
            seq_log_probs.std(),  # policy log probs std
            policy_logits.mean(),  # policy logits mean
            kl_div.mean(),  # KL divergence mean
        )

        return loss, metrics

    @staticmethod
    def backward(ctx, grad_output, *grad_metrics):
        """Backward pass for GRPO loss.

        Args:
            grad_output: Gradient of the loss (scalar)
            grad_metrics: Gradients of the metrics (not used in backward computation)
        """
        _input, weight, attention_mask, bias = ctx.saved_tensors
        beta = ctx.beta  # Retrieve beta for KL scaling
        rewards = ctx.rewards  # Retrieve rewards for advantage computation

        # Initialize gradients
        grad_input = grad_weight = grad_bias = None

        # Compute gradients using autograd
        with torch.enable_grad():
            _input = _input.detach().requires_grad_()
            weight = weight.detach().requires_grad_()
            if bias is not None:
                bias = bias.detach().requires_grad_()

            # Forward pass
            batch_size, seq_len, hidden_size = _input.shape
            input_reshaped = _input.view(-1, hidden_size)
            logits = (input_reshaped @ weight.t()).view(batch_size, seq_len, -1)
            if bias is not None:
                logits = logits + bias

            # Compute log probabilities and sequence-level scores
            log_probs = F.log_softmax(logits, dim=-1)
            seq_log_probs = (log_probs.max(dim=-1).values * attention_mask).sum(dim=-1)

            # Compute advantages
            advantages = rewards - rewards.mean()
            if advantages.std() > 0:
                advantages = advantages / advantages.std()

            # Policy gradient loss with KL penalty
            policy_loss = -(advantages * seq_log_probs)
            kl_div = seq_log_probs - seq_log_probs.detach()  # KL divergence from current policy
            loss = (policy_loss + beta * kl_div).mean()  # Take mean to get scalar loss

            # Backward pass with scalar gradient
            loss.backward(grad_output)
            grad_input = _input.grad
            grad_weight = weight.grad
            grad_bias = bias.grad if bias is not None else None

        return (
            grad_input,
            grad_weight,
            None,  # grad_attention_mask
            None,  # grad_rewards
            grad_bias,
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
