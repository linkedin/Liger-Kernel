from abc import abstractmethod
from functools import partial

import torch
import torch.nn.functional as F


class LigerFusedLinearRLHFBase(torch.autograd.Function):
    @abstractmethod
    def preference_loss_fn(*args, **kwargs):
        """
        To be extended by subclasses.
        """
        raise NotImplementedError("Preference loss function must be implemented.")

    @staticmethod
    def forward(
        ctx,
        _input,
        weight,
        rewards,
        attention_mask,
        bias=None,
        loss_fn=None,
        chunk_size=1,
        beta=0.1,
        compiled=True,
        use_ref_model=False,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
    ):
        """
        Base class for fused linear layer with RLHF loss.
        Expects _input to contain the policy model inputs.

        Args:
            _input (torch.Tensor): Input tensor. Shape: (batch_size, seq_len, hidden_size).
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size).
            rewards (torch.Tensor): Rewards tensor. Shape: (batch_size,).
            attention_mask (torch.Tensor): Attention mask. Shape: (batch_size, seq_len).
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,).
            loss_fn (callable): Loss function to compute the loss on a chunk of input.
            chunk_size (int): Size of chunks to process.
            beta (float): Weight for KL penalty.
            compiled (bool): Whether to use torch compile for chunk accumulation.
            use_ref_model (bool): Whether to use a reference model.
            ref_input (torch.Tensor): Reference model input tensor.
            ref_weight (torch.Tensor): Reference model weight tensor.
            ref_bias (torch.Tensor, optional): Reference model bias tensor.
        """
        # TODO: Tune CHUNK_SIZE to fully utilize the GPU
        CHUNK_SIZE = chunk_size

        # Gradients to be accumulated
        grad_weight = torch.zeros_like(weight)
        grad_inputs = []
        grad_bias = torch.zeros_like(bias) if bias is not None else None

        # Loss to be accumulated
        loss_acc = torch.zeros((), device=_input.device)

        compute_loss = partial(
            LigerFusedLinearRLHFBase._compute_loss,
            preference_loss_fn=loss_fn,
            beta=beta,
            use_ref_model=use_ref_model,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            rewards=rewards,
        )

        def fused_fwd_bwd(input_chunk, attention_mask_chunk, ref_input_chunk):
            """
            Fused forward and backward pass for a chunk of input.
            """
            if bias is not None:
                return torch.func.grad_and_value(compute_loss, argnums=(0, 1, 4), has_aux=True)(
                    input_chunk,
                    weight,
                    attention_mask_chunk,
                    bias,
                    ref_input_chunk=ref_input_chunk,
                )
            else:
                return torch.func.grad_and_value(compute_loss, argnums=(0, 1), has_aux=True)(
                    input_chunk,
                    weight,
                    attention_mask_chunk,
                    ref_input_chunk=ref_input_chunk,
                )

        def accumulate_chunk(input_chunk, attention_mask_chunk, ref_input_chunk=None):
            if bias is not None:
                (chunk_grad_input, chunk_grad_weight, chunk_grad_bias), (chunk_loss, _) = fused_fwd_bwd(
                    input_chunk, attention_mask_chunk, ref_input_chunk
                )
                grad_bias.add_(chunk_grad_bias)
            else:
                (chunk_grad_input, chunk_grad_weight), (chunk_loss, _) = fused_fwd_bwd(
                    input_chunk, attention_mask_chunk, ref_input_chunk
                )

            # Accumulate gradients
            grad_weight.add_(chunk_grad_weight)
            grad_inputs.append(chunk_grad_input)

            # Accumulate loss
            loss_acc.add_(chunk_loss)

        if compiled:
            fused_fwd_bwd = torch.compile(fused_fwd_bwd)

        chunks = max(1, _input.shape[0] // CHUNK_SIZE)
        _input_chunks = torch.chunk(_input, chunks=chunks, dim=0)
        _attention_mask_chunks = torch.chunk(attention_mask, chunks=chunks, dim=0)

        if use_ref_model:
            _ref_input_chunks = torch.chunk(ref_input, chunks=chunks, dim=0)

        for input_chunk, attention_mask_chunk, ref_input_chunk in zip(
            _input_chunks,
            _attention_mask_chunks,
            (_ref_input_chunks if use_ref_model else [None] * len(_input_chunks)),
            strict=False,
        ):
            # Mark dynamic dimensions to prevent recompilation
            torch._dynamo.mark_dynamic(input_chunk, 1)
            torch._dynamo.mark_dynamic(attention_mask_chunk, 1)
            torch._dynamo.mark_dynamic(ref_input_chunk, 1) if use_ref_model else None

            # Accumulate loss and gradients
            accumulate_chunk(input_chunk, attention_mask_chunk, ref_input_chunk)

        # Combine gradients
        grad_input = torch.cat(grad_inputs, dim=0)

        ctx.save_for_backward(grad_input, grad_weight, grad_bias)
        return loss_acc, ()

    @staticmethod
    def backward(ctx, *grad_output):
        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        if torch.ne(grad_output[0][0], torch.tensor(1.0, device=grad_output[0][0].device)):
            grad_input = grad_input * grad_output[0][0]
            grad_weight = grad_weight * grad_output[0][0]
            grad_bias = grad_bias * grad_output[0][0] if grad_bias is not None else None

        return (
            grad_input,  # grad_input
            grad_weight,  # grad_weight
            None,  # grad_rewards
            None,  # grad_attention_mask
            grad_bias,  # grad_bias
            None,  # grad_loss_fn
            None,  # grad_chunk_size
            None,  # grad_beta
            None,  # grad_compiled
            None,  # grad_use_ref_model
            None,  # grad_ref_input
            None,  # grad_ref_weight
            None,  # grad_ref_bias
        )

    @staticmethod
    def _compute_loss(
        input_chunk,
        weight,
        rewards,
        attention_mask_chunk,
        bias=None,
        preference_loss_fn=None,
        beta=0.1,
        use_ref_model=False,
        ref_input_chunk=None,
        ref_weight=None,
        ref_bias=None,
    ):
        """
        Compute the total loss for a chunk of input, using an RLHF loss function.
        Args:
            input_chunk: Policy model hidden states (batch_size, seq_len, hidden_size)
            weight: Linear layer weights (vocab_size, hidden_size)
            attention_mask_chunk: Attention mask (batch_size, seq_len)
            bias: Optional linear layer bias (vocab_size,)
            preference_loss_fn: Loss function (e.g. GRPO loss)
            beta: KL penalty weight
            rewards: Rewards for advantage computation
            use_ref_model: Whether to use reference model
            ref_input_chunk: Reference model hidden states
            ref_weight: Reference model weights
            ref_bias: Reference model bias
        """
        # Get policy logits and log probs
        batch_size, seq_len, hidden_size = input_chunk.shape
        input_reshaped = input_chunk.view(-1, hidden_size)
        logits = (input_reshaped @ weight.t()).view(batch_size, seq_len, -1)
        if bias is not None:
            logits = logits + bias
        log_probs = F.log_softmax(logits, dim=-1)

        # Get sequence-level log probs by taking max over vocab
        seq_log_probs = log_probs.max(dim=-1).values

        # Get reference model log probs if needed
        ref_seq_log_probs = None
        if use_ref_model and ref_input_chunk is not None and ref_weight is not None:
            with torch.no_grad():
                ref_input_reshaped = ref_input_chunk.view(-1, ref_input_chunk.size(-1))
                ref_logits = (ref_input_reshaped @ ref_weight.t()).view(batch_size, seq_len, -1)
                if ref_bias is not None:
                    ref_logits = ref_logits + ref_bias
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                ref_seq_log_probs = ref_log_probs.max(dim=-1).values

        # Compute KL divergence if using reference model
        kl_div = None
        if use_ref_model and ref_seq_log_probs is not None:
            kl_div = seq_log_probs - ref_seq_log_probs

        # Compute loss using the provided loss function
        loss = preference_loss_fn(
            seq_log_probs=seq_log_probs,
            ref_seq_log_probs=ref_seq_log_probs,
            attention_mask=attention_mask_chunk,
            rewards=rewards,
            beta=beta,
        )

        # Return metrics for logging
        metrics = (
            seq_log_probs.mean(),  # policy log probs mean
            seq_log_probs.std(),  # policy log probs std
            logits.mean(),  # policy logits mean
            kl_div.mean() if kl_div is not None else torch.tensor(0.0, device=loss.device),  # KL divergence mean
        )

        return loss, metrics
