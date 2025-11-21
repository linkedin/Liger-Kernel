from abc import abstractmethod
from functools import partial

import torch
import torch._dynamo.config
import torch.nn.functional as F


class LigerFusedLinearPPOBase(torch.autograd.Function):
    @abstractmethod
    def ppo_loss_fn(*args, **kwargs):
        """
        To be extended by subclasses.
        """
        raise NotImplementedError("PPO loss function must be implemented.")

    @staticmethod
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
        epsilon_low=0.2,
        epsilon_high=0.2,
        beta=0.04,
        loss_type="dapo",
        max_completion_length=None,
        importance_sampling_level="token",
        temperature=1.0,
        compiled=True,
        use_ref_model=False,
        chunk_size=1,
    ):
        # TODO: check torch compile matmul
        """Chunked forward pass for PPO loss computation.

        Args:
            cls: The class
            ctx: Context for backward
            _input: Input tensor
            weight: Weight tensor
            selected_token_ids: Selected token ids tensor
            attention_mask: Attention mask tensor
            advantages: Advantages tensor
            bias: Bias tensor
            ref_per_token_logps: Reference model log probs per token tensor
            old_per_token_logps: Old per token log probabilities tensor
            ref_input: Reference model input tensor
            ref_weight: Reference model weight tensor
            ref_bias: Reference model bias tensor
            epsilon_low: Lower bound for clipping the importance sampling ratio
            epsilon_high: Upper bound for clipping the importance sampling ratio
            beta: Weight for the KL penalty
            loss_type: Type of loss calculation ("grpo", "bnpo", "dr_grpo", "dapo")
            max_completion_length: Maximum completion length required for "dr_grpo"
            temperature: Temperature for the logits
            compiled: Whether to use torch compile
            use_ref_model: Whether to use a reference model
            chunk_size: Size of chunks for processing in other loss modules
        """
        if use_ref_model:
            assert ref_per_token_logps is not None or ref_input is not None, (
                "If use_ref_model is True, ref_per_token_logps or ref_input must be provided"
            )
            if ref_per_token_logps is not None and ref_input is not None:
                raise Warning("Both ref_per_token_logps and ref_input are provided. Using ref_per_token_logps.")
        if loss_type == "dr_grpo":
            assert max_completion_length is not None, "max_completion_length must be provided for loss_type 'dr_grpo'"
        # Initialize accumulators
        loss_acc = torch.zeros((), device=_input.device, dtype=torch.float32)
        grad_weight = torch.zeros_like(weight)  # [V, H]
        grad_inputs = []
        grad_bias = torch.zeros_like(bias) if bias is not None else None  # [V]
        aggregated_metrics = []

        # Create a partial function with fixed arguments
        compute_loss = partial(
            LigerFusedLinearPPOBase._compute_chunk_loss,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            full_attention_mask=attention_mask,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            beta=beta,
            loss_type=loss_type,
            max_completion_length=max_completion_length,
            importance_sampling_level=importance_sampling_level,
            temperature=temperature,
            use_ref_model=use_ref_model,
            ppo_loss_fn=cls.ppo_loss_fn,
        )

        def fused_fwd_bwd(
            input_chunk,
            selected_token_ids_chunk,
            attention_mask_chunk,
            advantages_chunk,
            ref_per_token_logps_chunk,
            old_per_token_logps_chunk,
            ref_input_chunk,
        ):
            """Fused forward and backward for a chunk."""
            argnums = (0, 1, 5) if bias is not None else (0, 1)
            return torch.func.grad_and_value(compute_loss, argnums=argnums, has_aux=True)(
                input_chunk,  # arg 0
                weight,  # arg 1
                selected_token_ids_chunk,  # arg 2
                attention_mask_chunk,  # arg 3
                advantages_chunk,  # arg 4
                bias,  # arg 5
                ref_per_token_logps_chunk=ref_per_token_logps_chunk,  # arg 6
                old_per_token_logps_chunk=old_per_token_logps_chunk,  # arg 7
                ref_input_chunk=ref_input_chunk,  # arg 8
            )

        def accumulate_chunk(
            input_chunk,
            selected_token_ids_chunk,
            attention_mask_chunk,
            advantages_chunk,
            ref_per_token_logps_chunk=None,
            old_per_token_logps_chunk=None,
            ref_input_chunk=None,
        ):
            (chunk_grad_input, chunk_grad_weight, *chunk_grad_bias), (chunk_loss, chunk_metrics) = fused_fwd_bwd(
                input_chunk,
                selected_token_ids_chunk,
                attention_mask_chunk,
                advantages_chunk,
                ref_per_token_logps_chunk,
                old_per_token_logps_chunk,
                ref_input_chunk,
            )
            if bias is not None:
                grad_bias.add_(chunk_grad_bias[0])

            # Accumulate gradients and loss
            grad_weight.add_(chunk_grad_weight)
            grad_inputs.append(chunk_grad_input)
            loss_acc.add_(chunk_loss)
            # Initialize storage for metrics on first chunk
            if len(aggregated_metrics) == 0:
                for metric in chunk_metrics:
                    if metric.ndim == 0:
                        aggregated_metrics.append(torch.zeros((), device=metric.device))
                    else:
                        aggregated_metrics.append([])

            # Accumulate metrics
            for i, metric in enumerate(chunk_metrics):
                if metric.ndim == 0:
                    aggregated_metrics[i].add_(metric)
                else:
                    aggregated_metrics[i].append(metric)

        if compiled:
            # TODO: Figure out what is better to compile here
            # accumulate_chunk = torch.compile(accumulate_chunk)
            fused_fwd_bwd = torch.compile(fused_fwd_bwd)

        # Process input in chunks based on chunk_size
        chunks = max(1, _input.shape[0] // chunk_size)
        _input_chunks = torch.chunk(_input, chunks=chunks, dim=0)
        _selected_token_ids_chunks = torch.chunk(selected_token_ids, chunks=chunks, dim=0)
        _attention_mask_chunks = torch.chunk(attention_mask, chunks=chunks, dim=0)
        _advantages_chunks = torch.chunk(advantages, chunks=chunks, dim=0)
        _ref_per_token_logps_chunks = (
            torch.chunk(ref_per_token_logps, chunks=chunks, dim=0)
            if use_ref_model and ref_per_token_logps is not None
            else [None] * chunks
        )
        _old_per_token_logps_chunks = (
            torch.chunk(old_per_token_logps, chunks=chunks, dim=0)
            if old_per_token_logps is not None
            else [None] * chunks
        )
        # if ref_log_probs is not none, then we don't need ref_input to calculate the log probs
        _ref_input_chunks = (
            torch.chunk(ref_input, chunks=chunks, dim=0)
            if use_ref_model and ref_per_token_logps is None
            else [None] * chunks
        )

        for (
            input_chunk,
            selected_token_ids_chunk,
            attention_mask_chunk,
            advantages_chunk,
            ref_per_token_logps_chunk,
            old_per_token_logps_chunk,
            ref_input_chunk,
        ) in zip(
            _input_chunks,
            _selected_token_ids_chunks,
            _attention_mask_chunks,
            _advantages_chunks,
            _ref_per_token_logps_chunks,
            _old_per_token_logps_chunks,
            _ref_input_chunks,
        ):
            # Mark dynamic dimensions
            torch._dynamo.mark_dynamic(input_chunk, 1)
            torch._dynamo.mark_dynamic(selected_token_ids_chunk, 1)
            torch._dynamo.mark_dynamic(attention_mask_chunk, 1)
            if ref_per_token_logps_chunk is not None:
                torch._dynamo.mark_dynamic(ref_per_token_logps_chunk, 1)
            if ref_input_chunk is not None:
                torch._dynamo.mark_dynamic(ref_input_chunk, 1)
            if old_per_token_logps_chunk is not None:
                torch._dynamo.mark_dynamic(old_per_token_logps_chunk, 1)

            accumulate_chunk(
                input_chunk,
                selected_token_ids_chunk,
                attention_mask_chunk,
                advantages_chunk,
                ref_per_token_logps_chunk,
                old_per_token_logps_chunk,
                ref_input_chunk,
            )

        # Combine gradients
        grad_input = torch.cat(grad_inputs, dim=0)

        # Save for backward
        ctx.save_for_backward(grad_input, grad_weight, grad_bias)

        # Finalize metrics
        final_metrics = []
        for metric in aggregated_metrics:
            if isinstance(metric, list):
                final_metrics.append(torch.cat(metric, dim=0))
            else:
                final_metrics.append(metric)

        return loss_acc, tuple(final_metrics)

    @staticmethod
    def _compute_dapo_normalizer(attention_mask):
        """Global active tokens averaged per process."""
        normalizer = attention_mask.to(torch.float32).sum()
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            import torch.distributed as dist

            normalizer = normalizer.clone()
            dist.all_reduce(normalizer, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()

        normalizer = normalizer / world_size
        return torch.clamp(normalizer, min=1.0)

    @staticmethod
    def _compute_chunk_loss(
        input_chunk,
        weight,
        selected_token_ids_chunk,
        attention_mask_chunk,
        advantages_chunk,
        bias=None,
        ref_per_token_logps_chunk=None,
        old_per_token_logps_chunk=None,
        ref_input_chunk=None,
        ref_weight=None,
        ref_bias=None,
        full_attention_mask=None,
        epsilon_low=0.2,
        epsilon_high=0.2,
        beta=0.04,
        loss_type="dapo",
        max_completion_length=None,
        importance_sampling_level="token",
        temperature=1.0,
        use_ref_model=False,
        ppo_loss_fn=None,
    ):
        """Compute loss for a single chunk."""
        # Get policy log probabilities using chunk_forward
        log_probs, _ = LigerFusedLinearPPOBase.chunk_forward(input_chunk, weight, bias=bias, temperature=temperature)

        # Get reference log probabilities if needed
        ref_log_probs = None
        if use_ref_model and ref_per_token_logps_chunk is None:
            with torch.no_grad():
                ref_log_probs, _ = LigerFusedLinearPPOBase.chunk_forward(
                    ref_input_chunk, ref_weight, bias=ref_bias, temperature=temperature
                )

        # Compute chunk loss and metrics using the provided loss function
        chunk_loss, chunk_metrics = ppo_loss_fn(
            log_probs=log_probs,
            selected_token_ids=selected_token_ids_chunk,
            attention_mask=attention_mask_chunk,
            advantages=advantages_chunk,
            full_attention_mask=full_attention_mask,
            ref_per_token_logps=ref_per_token_logps_chunk.float() if ref_per_token_logps_chunk is not None else None,
            old_per_token_logps=old_per_token_logps_chunk.float() if old_per_token_logps_chunk is not None else None,
            ref_log_probs=ref_log_probs,  # used when ref_per_token_logps is None
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            beta=beta,
            loss_type=loss_type,
            max_completion_length=max_completion_length,
            importance_sampling_level=importance_sampling_level,
        )

        return chunk_loss, chunk_metrics

    @staticmethod
    def chunk_forward(input_chunk, weight, bias=None, temperature=1.0):
        """Forward pass computation for a single chunk without explicit reshaping."""
        # Directly compute logits via batched matrix multiplication: [B, T, H] @ [H, V] -> [B, T, V]
        logits = torch.matmul(input_chunk, weight.t())
        if bias is not None:
            logits = logits + bias  # Broadcasts bias to [B, T, V]
        if temperature != 1.0:
            logits = logits / temperature

        # Compute log probabilities using softmax over the last dimension
        log_probs = F.log_softmax(logits.float(), dim=-1)

        return log_probs, logits

    @staticmethod
    def backward(ctx, grad_output, *grad_metrics):
        """Backward pass for PPO loss."""
        grad_input, grad_weight, grad_bias = ctx.saved_tensors

        if grad_output != 1.0:
            grad_input = grad_input * grad_output
            grad_weight = grad_weight * grad_output
            if grad_bias is not None:
                grad_bias = grad_bias * grad_output

        return (
            grad_input,
            grad_weight,
            None,  # grad_selected_token_ids
            None,  # grad_attention_mask
            None,  # grad_advantages
            grad_bias,
            None,  # grad_ref_per_token_logps
            None,  # grad_old_per_token_logps
            None,  # grad_ref_input
            None,  # grad_ref_weight
            None,  # grad_ref_bias
            None,  # grad_epsilon_low
            None,  # grad_epsilon_high
            None,  # grad_beta
            None,  # grad_loss_type
            None,  # grad_max_completion_length
            None,  # grad_importance_sampling_level
            None,  # grad_temperature
            None,  # grad_compiled
            None,  # grad_use_ref_model
            None,  # grad_chunk_size
        )
