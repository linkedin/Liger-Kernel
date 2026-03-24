from abc import abstractmethod
from functools import partial

import torch
import torch._dynamo.config

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_SELECTIVE_LOGPROB_VOCAB_CHUNK_SIZE = 2048
_SELECTIVE_LOGPROB_EXACT_VOCAB_THRESHOLD = 4096


def _next_power_of_two(x):
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def _maybe_mark_dynamic_dim1(tensor):
    if tensor is not None:
        torch._dynamo.maybe_mark_dynamic(tensor, 1)


if _TRITON_AVAILABLE:

    @triton.jit
    def _selective_logprob_kernel(
        hidden_ptr,
        weight_ptr,
        bias_ptr,
        targets_ptr,
        logprobs_ptr,
        log_z_ptr,
        n_rows,
        hidden_size,
        vocab_size,
        stride_hidden_n,
        stride_hidden_h,
        stride_weight_v,
        stride_weight_h,
        inv_t,
        HAS_BIAS: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= n_rows:
            return

        hidden_row_ptr = hidden_ptr + pid * stride_hidden_n
        target_idx = tl.load(targets_ptr + pid)
        hidden_offsets_base = tl.arange(0, BLOCK_H)

        max_old = -float("inf")
        sum_exp = 0.0
        target_logit = 0.0

        for vocab_start in tl.range(0, vocab_size, BLOCK_V):
            vocab_offsets = vocab_start + tl.arange(0, BLOCK_V)
            vocab_mask = vocab_offsets < vocab_size
            logits = tl.zeros((BLOCK_V,), dtype=tl.float32)

            for hidden_start in tl.range(0, hidden_size, BLOCK_H):
                hidden_offsets = hidden_start + hidden_offsets_base
                hidden_mask = hidden_offsets < hidden_size
                hidden = tl.load(hidden_row_ptr + hidden_offsets * stride_hidden_h, mask=hidden_mask, other=0.0)
                weight_ptrs = (
                    weight_ptr + vocab_offsets[:, None] * stride_weight_v + hidden_offsets[None, :] * stride_weight_h
                )
                weight = tl.load(weight_ptrs, mask=vocab_mask[:, None] & hidden_mask[None, :], other=0.0)
                logits += tl.sum(weight.to(tl.float32) * hidden[None, :].to(tl.float32), axis=1)

            if HAS_BIAS:
                bias = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=0.0)
                logits += bias.to(tl.float32)

            logits *= inv_t
            masked_logits = tl.where(vocab_mask, logits, -float("inf"))

            chunk_max = tl.max(masked_logits, axis=0)
            max_new = tl.maximum(max_old, chunk_max)
            rescale = tl.exp(max_old - max_new)
            chunk_exp = tl.exp(masked_logits - max_new)

            sum_exp = sum_exp * rescale + tl.sum(chunk_exp, axis=0)
            target_logit += tl.sum(tl.where((vocab_offsets == target_idx) & vocab_mask, logits, 0.0), axis=0)
            max_old = max_new

        log_z = max_old + tl.log(sum_exp)
        tl.store(log_z_ptr + pid, log_z)
        tl.store(logprobs_ptr + pid, target_logit - log_z)


def _selective_logprob_forward_torch(hidden, weight, targets, bias=None, temperature=1.0, vocab_chunk_size=2048):
    device = hidden.device
    n_rows, _ = hidden.shape
    vocab_size, _ = weight.shape
    inv_t = 1.0 / temperature

    max_old = torch.full((n_rows,), float("-inf"), device=device, dtype=torch.float32)
    sum_exp = torch.zeros((n_rows,), device=device, dtype=torch.float32)
    target_logit = torch.zeros((n_rows,), device=device, dtype=torch.float32)

    mm_buf = torch.empty((n_rows, vocab_chunk_size), device=device, dtype=hidden.dtype)
    logits_buf = torch.empty((n_rows, vocab_chunk_size), device=device, dtype=torch.float32)
    row_idx = torch.arange(n_rows, device=device)

    for start in range(0, vocab_size, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab_size)
        chunk_width = end - start
        weight_chunk = weight[start:end].to(hidden.dtype)
        logits_chunk = logits_buf[:, :chunk_width]
        torch.mm(hidden, weight_chunk.t(), out=mm_buf[:, :chunk_width])
        logits_chunk.copy_(mm_buf[:, :chunk_width])
        if bias is not None:
            logits_chunk.add_(bias[start:end].to(torch.float32))
        logits_chunk.mul_(inv_t)

        chunk_max = logits_chunk.amax(dim=-1)
        max_new = torch.maximum(max_old, chunk_max)
        rescale = torch.exp(max_old - max_new)
        chunk_exp = torch.exp(logits_chunk - max_new.unsqueeze(-1))

        sum_exp = sum_exp * rescale + chunk_exp.sum(dim=-1)
        max_old = max_new

        in_chunk = (targets >= start) & (targets < end)
        local_idx = torch.clamp(targets - start, 0, end - start - 1)
        target_logit += logits_chunk[row_idx, local_idx] * in_chunk

    log_z = max_old + torch.log(sum_exp)
    return target_logit - log_z, log_z


def _selective_logprob_forward_autograd(hidden, weight, targets, bias=None, temperature=1.0, vocab_chunk_size=2048):
    n_rows, _ = hidden.shape
    vocab_size, _ = weight.shape
    inv_t = 1.0 / temperature

    if vocab_size <= _SELECTIVE_LOGPROB_EXACT_VOCAB_THRESHOLD:
        logits = hidden @ weight.to(hidden.dtype).t()
        logits = logits.float()
        if bias is not None:
            logits = logits + bias.to(torch.float32)
        logits = logits * inv_t
        return torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    max_old = torch.full((n_rows,), float("-inf"), device=hidden.device, dtype=torch.float32)
    sum_exp = torch.zeros((n_rows,), device=hidden.device, dtype=torch.float32)
    target_logit = torch.zeros((n_rows,), device=hidden.device, dtype=torch.float32)
    row_idx = torch.arange(n_rows, device=hidden.device)

    for start in range(0, vocab_size, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab_size)
        logits_chunk = hidden @ weight[start:end].to(hidden.dtype).t()
        logits_chunk = logits_chunk.float()
        if bias is not None:
            logits_chunk = logits_chunk + bias[start:end].to(torch.float32)
        logits_chunk = logits_chunk * inv_t

        chunk_max = logits_chunk.amax(dim=-1)
        max_new = torch.maximum(max_old, chunk_max)
        rescale = torch.exp(max_old - max_new)
        chunk_exp = torch.exp(logits_chunk - max_new.unsqueeze(-1))

        sum_exp = sum_exp * rescale + chunk_exp.sum(dim=-1)
        max_old = max_new

        in_chunk = (targets >= start) & (targets < end)
        local_idx = torch.clamp(targets - start, 0, end - start - 1)
        target_logit = target_logit + logits_chunk[row_idx, local_idx] * in_chunk

    log_z = max_old + torch.log(sum_exp)
    return target_logit - log_z


def _selective_logprob_forward_triton(hidden, weight, targets, bias=None, temperature=1.0, vocab_chunk_size=2048):
    n_rows, hidden_size = hidden.shape
    block_h = min(128, _next_power_of_two(hidden_size))
    block_v = min(vocab_chunk_size, 128)
    num_warps = 4 if block_v <= 64 else 8

    logprobs = torch.empty((n_rows,), device=hidden.device, dtype=torch.float32)
    log_z = torch.empty((n_rows,), device=hidden.device, dtype=torch.float32)

    _selective_logprob_kernel[(n_rows,)](
        hidden,
        weight,
        bias if bias is not None else hidden,
        targets,
        logprobs,
        log_z,
        n_rows,
        hidden_size,
        weight.shape[0],
        hidden.stride(0),
        hidden.stride(1),
        weight.stride(0),
        weight.stride(1),
        1.0 / temperature,
        HAS_BIAS=bias is not None,
        BLOCK_H=block_h,
        BLOCK_V=block_v,
        num_warps=num_warps,
    )
    return logprobs, log_z


def _selective_logprob_forward(hidden, weight, targets, bias=None, temperature=1.0, vocab_chunk_size=2048):
    return _selective_logprob_forward_torch(hidden, weight, targets, bias, temperature, vocab_chunk_size)


class _ChunkedSelectiveLogProbFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, bias, temperature, vocab_chunk_size):
        logprobs, log_z = _selective_logprob_forward(hidden, weight, targets, bias, temperature, vocab_chunk_size)
        if bias is None:
            bias = hidden.new_empty((0,))
        ctx.save_for_backward(hidden, weight, targets, bias, log_z)
        ctx.has_bias = bias.numel() > 0
        ctx.temperature = temperature
        ctx.vocab_chunk_size = vocab_chunk_size
        return logprobs

    @staticmethod
    def backward(ctx, grad_logprobs):
        hidden, weight, targets, bias, log_z = ctx.saved_tensors
        grad_hidden, grad_weight, grad_bias = _selective_logprob_backward(
            hidden=hidden,
            weight=weight,
            targets=targets,
            bias=bias if ctx.has_bias else None,
            log_z=log_z,
            grad_logprobs=grad_logprobs,
            temperature=ctx.temperature,
            vocab_chunk_size=ctx.vocab_chunk_size,
        )
        return (
            grad_hidden.to(hidden.dtype),
            grad_weight.to(weight.dtype),
            None,
            grad_bias.to(bias.dtype) if ctx.has_bias else None,
            None,
            None,
        )


def _selective_logprob_backward(hidden, weight, targets, bias, log_z, grad_logprobs, temperature, vocab_chunk_size):
    inv_t = 1.0 / temperature
    n_rows, _ = hidden.shape
    vocab_size = weight.shape[0]
    has_bias = bias is not None
    hidden_fp32 = hidden.float()

    grad_hidden = torch.zeros(hidden.shape, device=hidden.device, dtype=torch.float32)
    grad_weight = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)
    grad_bias = torch.zeros((vocab_size,), device=weight.device, dtype=torch.float32) if has_bias else None

    logits_buf = torch.empty((n_rows, vocab_chunk_size), device=hidden.device, dtype=torch.float32)

    grad_logprobs = grad_logprobs.to(torch.float32)
    row_idx = torch.arange(n_rows, device=hidden.device)

    for start in range(0, vocab_size, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab_size)
        chunk_width = end - start
        weight_chunk = weight[start:end].float()
        logits_chunk = logits_buf[:, :chunk_width]
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        if hidden.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.mm(hidden_fp32, weight_chunk.t(), out=logits_chunk)
        finally:
            if hidden.is_cuda:
                torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if has_bias:
            logits_chunk.add_(bias[start:end].to(torch.float32))
        logits_chunk.mul_(inv_t)

        probs = torch.exp(logits_chunk - log_z.unsqueeze(-1))
        grad_logits = (-grad_logprobs).unsqueeze(-1) * probs

        in_chunk = (targets >= start) & (targets < end)
        local_idx = torch.clamp(targets - start, 0, end - start - 1)
        grad_logits[row_idx, local_idx] += grad_logprobs * in_chunk
        grad_logits.mul_(inv_t)

        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        if hidden.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
        try:
            grad_hidden.add_(grad_logits @ weight_chunk)
            grad_weight[start:end].add_(grad_logits.t() @ hidden_fp32)
        finally:
            if hidden.is_cuda:
                torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if has_bias:
            grad_bias[start:end].add_(grad_logits.sum(dim=0))

    return grad_hidden, grad_weight, grad_bias


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
        sapo_temperature_pos=1.0,
        sapo_temperature_neg=1.05,
        vllm_is_ratio=None,
        delta=None,
        use_bias_correction_kl=False,
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
            loss_type: Type of loss calculation ("grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo")
            max_completion_length: Maximum completion length required for "dr_grpo"
            importance_sampling_level: Level of importance sampling ("token" or "sequence")
            temperature: Temperature for the logits
            compiled: Whether to use torch compile
            use_ref_model: Whether to use a reference model
            chunk_size: Size of chunks for processing in other loss modules
            sapo_temperature_pos: Temperature for positive advantages in SAPO
            sapo_temperature_neg: Temperature for negative advantages in SAPO
            vllm_is_ratio: vLLM importance sampling ratio tensor (batch_size, seq_len) or (batch_size, 1) or None.
                Used to correct for distribution mismatch when using vLLM for generation.
        """
        if use_ref_model:
            assert ref_per_token_logps is not None or ref_input is not None, (
                "If use_ref_model is True, ref_per_token_logps or ref_input must be provided"
            )
            if ref_per_token_logps is not None and ref_input is not None:
                raise Warning("Both ref_per_token_logps and ref_input are provided. Using ref_per_token_logps.")
        if loss_type == "dr_grpo":
            assert max_completion_length is not None, "max_completion_length must be provided for loss_type 'dr_grpo'"
        if vllm_is_ratio is not None:
            B, T = attention_mask.shape
            assert vllm_is_ratio.dim() in (1, 2), (
                f"vllm_is_ratio must be 1D (B,) or 2D (B, T) / (B, 1), got {vllm_is_ratio.dim()}D"
            )
            if vllm_is_ratio.dim() == 2:
                assert vllm_is_ratio.shape[0] == B and vllm_is_ratio.shape[1] in (1, T), (
                    f"vllm_is_ratio shape must be ({B}, 1) or ({B}, {T}), got {tuple(vllm_is_ratio.shape)}"
                )
            else:
                assert vllm_is_ratio.shape[0] == B, (
                    f"vllm_is_ratio shape must be ({B},), got {tuple(vllm_is_ratio.shape)}"
                )
                vllm_is_ratio = vllm_is_ratio.unsqueeze(-1)  # (B,) -> (B, 1) for broadcasting
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
            sapo_temperature_pos=sapo_temperature_pos,
            sapo_temperature_neg=sapo_temperature_neg,
            delta=delta,
            use_bias_correction_kl=use_bias_correction_kl,
        )
        use_compiled_compute_loss = compiled and weight.shape[0] > _SELECTIVE_LOGPROB_EXACT_VOCAB_THRESHOLD
        compiled_compute_loss = torch.compile(compute_loss) if use_compiled_compute_loss else compute_loss

        def fused_fwd_bwd(
            input_chunk,
            selected_token_ids_chunk,
            attention_mask_chunk,
            advantages_chunk,
            ref_per_token_logps_chunk,
            old_per_token_logps_chunk,
            ref_input_chunk,
            vllm_is_ratio_chunk,
        ):
            """Fused forward and backward for a chunk."""
            with torch.enable_grad():
                input_chunk = input_chunk.detach().requires_grad_(True)
                weight_local = weight.detach().requires_grad_(True)
                bias_local = bias.detach().requires_grad_(True) if bias is not None else None
                chunk_loss, chunk_metrics = compiled_compute_loss(
                    input_chunk,  # arg 0
                    weight_local,  # arg 1
                    selected_token_ids_chunk,  # arg 2
                    attention_mask_chunk,  # arg 3
                    advantages_chunk,  # arg 4
                    bias_local,  # arg 5
                    ref_per_token_logps_chunk=ref_per_token_logps_chunk,  # arg 6
                    old_per_token_logps_chunk=old_per_token_logps_chunk,  # arg 7
                    ref_input_chunk=ref_input_chunk,  # arg 8
                    vllm_is_ratio_chunk=vllm_is_ratio_chunk,  # arg 9
                )
                grad_targets = [input_chunk, weight_local]
                if bias_local is not None:
                    grad_targets.append(bias_local)
                grads = torch.autograd.grad(chunk_loss, grad_targets)
            return grads, (chunk_loss.detach(), tuple(metric.detach() for metric in chunk_metrics))

        def accumulate_chunk(
            input_chunk,
            selected_token_ids_chunk,
            attention_mask_chunk,
            advantages_chunk,
            ref_per_token_logps_chunk=None,
            old_per_token_logps_chunk=None,
            ref_input_chunk=None,
            vllm_is_ratio_chunk=None,
        ):
            (chunk_grad_input, chunk_grad_weight, *chunk_grad_bias), (chunk_loss, chunk_metrics) = fused_fwd_bwd(
                input_chunk,
                selected_token_ids_chunk,
                attention_mask_chunk,
                advantages_chunk,
                ref_per_token_logps_chunk,
                old_per_token_logps_chunk,
                ref_input_chunk,
                vllm_is_ratio_chunk,
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

        # Process input in chunks based on chunk_size
        if weight.shape[0] <= _SELECTIVE_LOGPROB_EXACT_VOCAB_THRESHOLD:
            chunks = 1
        else:
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
        # If ref_per_token_logps is provided, we don't need ref_input to calculate the reference log probs.
        _ref_input_chunks = (
            torch.chunk(ref_input, chunks=chunks, dim=0)
            if use_ref_model and ref_per_token_logps is None
            else [None] * chunks
        )
        _vllm_is_ratio_chunks = (
            torch.chunk(vllm_is_ratio, chunks=chunks, dim=0) if vllm_is_ratio is not None else [None] * chunks
        )

        for (
            input_chunk,
            selected_token_ids_chunk,
            attention_mask_chunk,
            advantages_chunk,
            ref_per_token_logps_chunk,
            old_per_token_logps_chunk,
            ref_input_chunk,
            vllm_is_ratio_chunk,
        ) in zip(
            _input_chunks,
            _selected_token_ids_chunks,
            _attention_mask_chunks,
            _advantages_chunks,
            _ref_per_token_logps_chunks,
            _old_per_token_logps_chunks,
            _ref_input_chunks,
            _vllm_is_ratio_chunks,
        ):
            # Allow torch.compile to generalize sequence length without forcing it to be dynamic.
            _maybe_mark_dynamic_dim1(input_chunk)
            _maybe_mark_dynamic_dim1(selected_token_ids_chunk)
            _maybe_mark_dynamic_dim1(attention_mask_chunk)
            _maybe_mark_dynamic_dim1(ref_per_token_logps_chunk)
            _maybe_mark_dynamic_dim1(ref_input_chunk)
            _maybe_mark_dynamic_dim1(old_per_token_logps_chunk)
            _maybe_mark_dynamic_dim1(vllm_is_ratio_chunk)

            accumulate_chunk(
                input_chunk,
                selected_token_ids_chunk,
                attention_mask_chunk,
                advantages_chunk,
                ref_per_token_logps_chunk,
                old_per_token_logps_chunk,
                ref_input_chunk,
                vllm_is_ratio_chunk,
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
        vllm_is_ratio_chunk=None,
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
        sapo_temperature_pos=1.0,
        sapo_temperature_neg=1.05,
        delta=None,
        use_bias_correction_kl=False,
    ):
        """Compute loss for a single chunk."""
        per_token_logps = LigerFusedLinearPPOBase.chunk_forward(
            input_chunk,
            weight,
            selected_token_ids_chunk,
            bias=bias,
            temperature=temperature,
        )

        if use_ref_model and ref_per_token_logps_chunk is None:
            with torch.no_grad():
                if ref_weight.shape[0] <= _SELECTIVE_LOGPROB_EXACT_VOCAB_THRESHOLD:
                    ref_logits = ref_input_chunk @ ref_weight.t()
                    if ref_bias is not None:
                        ref_logits = ref_logits + ref_bias.float()
                    ref_logits = ref_logits.float()
                    if temperature != 1.0:
                        ref_logits = ref_logits / temperature
                    ref_per_token_logps_chunk = torch.log_softmax(ref_logits, dim=-1).gather(
                        -1, selected_token_ids_chunk.unsqueeze(-1)
                    ).squeeze(-1)
                else:
                    ref_per_token_logps_chunk = LigerFusedLinearPPOBase.chunk_forward(
                        ref_input_chunk,
                        ref_weight,
                        selected_token_ids_chunk,
                        bias=ref_bias,
                        temperature=temperature,
                    )

        # Compute chunk loss and metrics using the provided loss function
        chunk_loss, chunk_metrics = ppo_loss_fn(
            per_token_logps=per_token_logps,
            attention_mask=attention_mask_chunk,
            advantages=advantages_chunk,
            full_attention_mask=full_attention_mask,
            ref_per_token_logps=ref_per_token_logps_chunk.float() if ref_per_token_logps_chunk is not None else None,
            old_per_token_logps=old_per_token_logps_chunk.float() if old_per_token_logps_chunk is not None else None,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            beta=beta,
            loss_type=loss_type,
            max_completion_length=max_completion_length,
            importance_sampling_level=importance_sampling_level,
            sapo_temperature_pos=sapo_temperature_pos,
            sapo_temperature_neg=sapo_temperature_neg,
            vllm_is_ratio=vllm_is_ratio_chunk,
            delta=delta,
            use_bias_correction_kl=use_bias_correction_kl,
        )

        return chunk_loss, chunk_metrics

    @staticmethod
    def chunk_forward(input_chunk, weight, selected_token_ids, bias=None, temperature=1.0):
        """Compute selected-token log probabilities without materializing full vocab logits."""
        if weight.shape[0] <= _SELECTIVE_LOGPROB_EXACT_VOCAB_THRESHOLD:
            logits = input_chunk @ weight.t()
            if bias is not None:
                logits = logits + bias
            logits = logits.float()
            if temperature != 1.0:
                logits = logits / temperature
            return torch.log_softmax(logits, dim=-1).gather(-1, selected_token_ids.unsqueeze(-1)).squeeze(-1)

        batch_size, seq_len, hidden_size = input_chunk.shape
        hidden = input_chunk.reshape(batch_size * seq_len, hidden_size).contiguous()
        targets = selected_token_ids.reshape(batch_size * seq_len).contiguous()
        per_token_logps = _selective_logprob_forward_autograd(
            hidden, weight, targets, bias, temperature, _SELECTIVE_LOGPROB_VOCAB_CHUNK_SIZE
        )
        return per_token_logps.reshape(batch_size, seq_len)

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
            None,  # grad_sapo_temperature_pos
            None,  # grad_sapo_temperature_neg
            None,  # grad_vllm_is_ratio
            None,  # grad_delta
            None,  # grad_use_bias_correction_kl
        )
