from functools import partial

import torch

from torch.nn import functional as F


class ChunkMatmul(torch.autograd.Function):
    buf: dict[str, torch.Tensor | int] = {"count": 0}

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if weight.requires_grad:
            ctx.save_for_backward(x, weight)
            ChunkMatmul.buf["count"] += 1
        return x.matmul(weight.T)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        grad_x = grad_out.matmul(weight)
        grad_out_flat = grad_out.view(-1, grad_out.shape[-1])
        x_flat = x.view(-1, x.shape[-1])
        if "grad" not in ChunkMatmul.buf:
            ChunkMatmul.buf["grad"] = torch.zeros_like(weight)
        ChunkMatmul.buf["grad"].addmm_(grad_out_flat.T, x_flat)
        ChunkMatmul.buf["count"] -= 1
        if ChunkMatmul.buf["count"] == 0:
            grad_w = ChunkMatmul.buf["grad"]
            del ChunkMatmul.buf["grad"]
        else:
            grad_w = None
        return grad_x, grad_w


class LigerChunkedLinearPreferenceBase(torch.nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        compute_nll_loss: bool = False,
        compiled: bool = True,
        average_log_prob=True,
        chunk_size: int = 1024,
    ):
        """
        Args:
            ignore_index (int): Index to ignore in the loss.
            beta (float): Weight for the odds ratio loss.
            compute_nll_loss (bool): Whether to compute the NLL loss.
            compiled (bool): Whether to use the torch compiled kernel.
            average_log_prob (bool): Whether to average the log probability per non-masked token.
            chunk_size (int): Size of chunks for processing.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.beta = beta
        self.compute_nll_loss = compute_nll_loss
        self.compiled = compiled
        self.average_log_prob = average_log_prob
        self.chunk_size = chunk_size

    def forward(
        self,
        _input,
        weight,
        target,
        preference_loss_fn,
        bias=None,
        alpha=1.0,
        nll_target=None,
        use_ref_model=False,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        **loss_kwargs,
    ):
        """
        Base class for chunked linear layer with preference loss.
        Expects _input to be stacked with chosen and rejected inputs on the batch dimension.

        The mental model is:

        forward()
        ├── Loop over chunks
            └── compute_logp()
                └── chunk_forward()  # Compute logits and log probs

        Args:
            _input (torch.Tensor): Input tensor. Shape: (batch_size, seq_len, hidden_size).
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size).
            target (torch.Tensor): Target tensor. Shape: (batch_size, seq_len).
            preference_loss_fn (callable): Preference loss function to compute the preference loss on a chunk of input/target.
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,).
            alpha (float): Weight for the NLL loss.
            nll_target (torch.Tensor, optional): Target tensor for NLL loss. Shape: (batch_size, seq_len). If not provided the target is used.
            use_ref_model (bool): Whether to use a reference model for the alignment loss.
            ref_weight (torch.Tensor): Reference weight tensor. Shape: (vocab_size, hidden_size).
            ref_bias (torch.Tensor, optional): Reference bias tensor. Shape: (vocab_size,).
            loss_kwargs (dict): Other possible arguments that a loss function might need
        """
        beta = self.beta
        compiled = self.compiled
        ignore_index = self.ignore_index
        average_log_prob = self.average_log_prob
        # TODO: Tune CHUNK_SIZE to fully utilize the GPU
        CHUNK_SIZE = self.chunk_size

        # NLL Loss to be accumulated
        nll_loss_acc = torch.zeros((), device=_input.device)

        # Metrics to be recorded
        policy_chosen_logps = []
        policy_rejected_logps = []
        policy_chosen_logits_mean = torch.zeros((), device=_input.device)
        policy_rejected_logits_mean = torch.zeros((), device=_input.device)
        reference_chosen_logps = []
        reference_rejected_logps = []

        compute_logp = partial(
            self._compute_logp,
            full_target=target,
            use_ref_model=use_ref_model,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            full_nll_target=nll_target,
        )

        def accumulate_chunk(input_chunk, target_chunk, ref_input_chunk=None, chosen_nll_target_chunk=None):
            if bias is not None:
                outputs = compute_logp(
                    input_chunk,
                    weight,
                    target_chunk,
                    bias,
                    ref_input_chunk=ref_input_chunk,
                    chosen_nll_target_chunk=chosen_nll_target_chunk,
                )
            else:
                outputs = compute_logp(
                    input_chunk,
                    weight,
                    target_chunk,
                    ref_input_chunk=ref_input_chunk,
                    chosen_nll_target_chunk=chosen_nll_target_chunk,
                )
            (
                chunk_nll_loss,
                (
                    chunk_chosen_logps,
                    chunk_rejected_logps,
                    chunk_chosen_logits_mean,
                    chunk_rejected_logits_mean,
                    chunk_ref_chosen_logps,
                    chunk_ref_rejected_logps,
                ),
            ) = outputs

            # Accumulate NLL loss
            nll_loss_acc.add_(chunk_nll_loss)

            # Accumulate metrics
            policy_chosen_logps.append(chunk_chosen_logps)
            policy_rejected_logps.append(chunk_rejected_logps)
            policy_chosen_logits_mean.add_(chunk_chosen_logits_mean)
            policy_rejected_logits_mean.add_(chunk_rejected_logits_mean)
            reference_chosen_logps.append(chunk_ref_chosen_logps)
            reference_rejected_logps.append(chunk_ref_rejected_logps)

        if compiled:
            compute_logp = torch.compile(compute_logp)

        hidden_size = _input.shape[-1]
        len_chosen = target.shape[0] // 2
        chunks = max(1, _input.shape[1] // CHUNK_SIZE)
        _chosen_input = _input[:len_chosen]
        _chosen_target = target[:len_chosen]
        _rejected_input = _input[len_chosen:]
        _rejected_target = target[len_chosen:]
        _chosen_input_chunks = torch.chunk(_chosen_input.view(-1, hidden_size), chunks=chunks, dim=0)
        _chosen_target_chunks = torch.chunk(_chosen_target.view(-1), chunks=chunks, dim=0)
        _rejected_input_chunks = torch.chunk(_rejected_input.view(-1, hidden_size), chunks=chunks, dim=0)
        _rejected_target_chunks = torch.chunk(_rejected_target.view(-1), chunks=chunks, dim=0)

        if nll_target is not None:
            _chosen_nll_target = nll_target[:len_chosen].view(-1)
            _chosen_nll_target_chunks = torch.chunk(_chosen_nll_target, chunks=chunks, dim=0)

        if use_ref_model:
            _ref_chosen_input = ref_input[:len_chosen].view(-1, hidden_size)
            _ref_rejected_input = ref_input[len_chosen:].view(-1, hidden_size)
            _ref_chosen_input_chunks = torch.chunk(_ref_chosen_input, chunks=chunks, dim=0)
            _ref_rejected_input_chunks = torch.chunk(_ref_rejected_input, chunks=chunks, dim=0)

        for (
            chosen_input_chunk,
            rejected_input_chunk,
            chosen_target_chunk,
            rejected_target_chunk,
            ref_chosen_input_chunk,
            ref_rejected_input_chunk,
            chosen_nll_target_chunk,
        ) in zip(
            _chosen_input_chunks,
            _rejected_input_chunks,
            _chosen_target_chunks,
            _rejected_target_chunks,
            (_ref_chosen_input_chunks if use_ref_model else [None] * len(_chosen_input_chunks)),
            (_ref_rejected_input_chunks if use_ref_model else [None] * len(_rejected_input_chunks)),
            (_chosen_nll_target_chunks if nll_target is not None else [None] * len(_chosen_input_chunks)),
            strict=False,
        ):
            input_chunk = torch.stack([chosen_input_chunk, rejected_input_chunk])
            ref_input_chunk = (
                torch.stack([ref_chosen_input_chunk, ref_rejected_input_chunk]) if use_ref_model else None
            )
            target_chunk = torch.stack([chosen_target_chunk, rejected_target_chunk])

            # mark input_chunk, target_chunk, and target dimension 1 as dynamic to prevent torch.compile recompilation
            torch._dynamo.mark_dynamic(input_chunk, 1)
            torch._dynamo.mark_dynamic(target_chunk, 1)
            torch._dynamo.mark_dynamic(target, 1)
            torch._dynamo.mark_dynamic(ref_input_chunk, 1) if use_ref_model else None
            torch._dynamo.mark_dynamic(chosen_nll_target_chunk, 1) if nll_target is not None else None

            # accumulate loss, gradients, and metrics
            accumulate_chunk(input_chunk, target_chunk, ref_input_chunk, chosen_nll_target_chunk)

        policy_chosen_logps = torch.cat(policy_chosen_logps, dim=0).view(len_chosen, -1).sum(-1)
        policy_rejected_logps = torch.cat(policy_rejected_logps, dim=0).view(len_chosen, -1).sum(-1)
        reference_chosen_logps = torch.cat(reference_chosen_logps, dim=0).view(len_chosen, -1).sum(-1)
        reference_rejected_logps = torch.cat(reference_rejected_logps, dim=0).view(len_chosen, -1).sum(-1)
        if average_log_prob:
            _chosen_loss_mask = _chosen_target != ignore_index
            _rejected_loss_mask = _rejected_target != ignore_index
            policy_chosen_logps = policy_chosen_logps / _chosen_loss_mask.sum(-1)
            policy_rejected_logps = policy_rejected_logps / _rejected_loss_mask.sum(-1)
            reference_chosen_logps = reference_chosen_logps / _chosen_loss_mask.sum(-1)
            reference_rejected_logps = reference_rejected_logps / _rejected_loss_mask.sum(-1)

        preference_loss_outputs = preference_loss_fn(
            policy_chosen_logps, policy_rejected_logps, target, reference_chosen_logps, reference_rejected_logps, beta
        )
        if isinstance(preference_loss_outputs, tuple):
            preference_loss, *aux_outputs = preference_loss_outputs
        else:
            preference_loss, aux_outputs = preference_loss_outputs, []

        loss = alpha * nll_loss_acc + preference_loss

        return_vars = (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits_mean,
            policy_rejected_logits_mean,
            nll_loss_acc,
        )
        return loss, (*return_vars, *aux_outputs)

    def chunk_forward(
        self,
        input_chunk,
        weight,
        target_chunk,
        bias=None,
        compute_nll_loss=True,
        chosen_nll_target_chunk=None,
    ):
        ignore_index = self.ignore_index

        logits_chunk = ChunkMatmul.apply(input_chunk, weight)
        if bias is not None:
            logits_chunk = logits_chunk + bias
        log_probs_chunk = F.log_softmax(logits_chunk, dim=-1, dtype=torch.float32)

        chosen_nll_loss = 0.0
        if compute_nll_loss:
            nll_labels = (
                chosen_nll_target_chunk if chosen_nll_target_chunk is not None else target_chunk[0]
            )
            chosen_nll_loss = F.nll_loss(
                log_probs_chunk[0],
                nll_labels.view(-1),
                reduction="sum",
                ignore_index=ignore_index,
            )

        loss_mask = target_chunk != ignore_index
        label_chunk = torch.where(loss_mask, target_chunk, 0)

        per_token_logps = log_probs_chunk.gather(-1, label_chunk.unsqueeze(-1)).squeeze(-1)
        log_prob = (per_token_logps * loss_mask)

        chosen_logps = log_prob[0]
        rejected_logps = log_prob[1]

        chosen_logits = logits_chunk[0]
        rejected_logits = logits_chunk[1]

        return (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            chosen_nll_loss,
        )

    def _compute_logp(
        self,
        input_chunk,
        weight,
        target_chunk,
        bias=None,
        full_target=None,
        use_ref_model=False,
        ref_input_chunk=None,
        ref_weight=None,
        ref_bias=None,
        full_nll_target=None,
        chosen_nll_target_chunk=None,
    ):
        """
        Compute the total loss for a chunk of input and target, while using an alignment/preference loss function.
        Args:
            input_chunk (torch.Tensor): Chunk of input tensor. Shape: (2, chunk_size, hidden_size).
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size).
            target_chunk (torch.Tensor): Chunk of target tensor. Shape: (2, chunk_size).
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,).
            full_target (torch.Tensor): Full target tensor. Shape: (batch_size, sequence_length).
            use_ref_model (bool): Whether to use a reference model for the alignment loss.
            ref_weight (torch.Tensor): Reference weight tensor. Shape: (vocab_size, hidden_size).
            ref_bias (torch.Tensor, optional): Reference bias tensor. Shape: (vocab_size,).
            full_nll_target (torch.Tensor, optional): Full target tensor for NLL loss. Shape: (batch_size, sequence_length).
            chosen_nll_target_chunk (torch.Tensor, optional): Target tensor for NLL loss. Shape: (chunk_size,) If not provided the target_chunk is used.
        """
        ignore_index = self.ignore_index
        compute_nll_loss = self.compute_nll_loss
        (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            chosen_nll_loss,
        ) = self.chunk_forward(
            input_chunk,
            weight,
            target_chunk,
            bias=bias,
            compute_nll_loss=compute_nll_loss,
            chosen_nll_target_chunk=chosen_nll_target_chunk,
        )
        if full_nll_target is not None:
            chosen_nll_loss = chosen_nll_loss / (full_nll_target[: full_nll_target.shape[0] // 2] != ignore_index).sum()
        else:
            chosen_nll_loss = chosen_nll_loss / (full_target[: full_target.shape[0] // 2] != ignore_index).sum()

        chosen_logits_mean = chosen_logits.sum() / (full_target.shape[0] // 2 * input_chunk.shape[1] * weight.shape[0])
        rejected_logits_mean = rejected_logits.sum() / (
            full_target.shape[0] // 2 * input_chunk.shape[1] * weight.shape[0]
        )

        if use_ref_model:
            with torch.no_grad():
                (
                    ref_chosen_logps,
                    ref_rejected_logps,
                    _,
                    _,
                    _,
                ) = self.chunk_forward(
                    ref_input_chunk,
                    ref_weight,
                    target_chunk,
                    ref_bias,
                    compute_nll_loss=False,  # We don't need NLL loss for the reference model
                    chosen_nll_target_chunk=None,
                )
        else:
            ref_chosen_logps = None
            ref_rejected_logps = None

        return_vars = (
            chosen_logps,
            rejected_logps,
            chosen_logits_mean,
            rejected_logits_mean,
            ref_chosen_logps,
            ref_rejected_logps,
        )
        return chosen_nll_loss, return_vars
