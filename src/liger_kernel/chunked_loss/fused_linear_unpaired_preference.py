from abc import abstractmethod
from functools import partial

import torch
from torch.nn import functional as F


class LigerFusedLinearUnpairedPreferenceBase(torch.autograd.Function):

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
        target,
        bias=None,
        loss_fn=None,
        chunk_size=1,
        ignore_index=-100,
        alpha=1.0,
        beta=0.1,
        compiled=True,
        preference_labels=None,
        use_ref_model=False,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        **loss_kwargs,
    ):
        """
        Base class for fused linear layer with preference loss.
        Expects _input to be stacked with chosen and rejected inputs on the batch dimension.

        The mental model is:

        forward()
        ├── Loop over chunks
            └── compute_loss()
                ├── chunk_forward()  # Compute logits and log probs
                └── prefer_loss()    # Calculate preference loss

        Args:
            _input (torch.Tensor): Input tensor. Shape: (batch_size, seq_len, hidden_size).
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size).
            target (torch.Tensor): Target tensor. Shape: (batch_size, seq_len).
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,).
            loss_fn (callable): Loss function to compute the loss on a chunk of input/target.
            chunk_size (int): Size of a chunk (# of batches of stacked chosen and rejected inputs).
            ignore_index (int): Index to ignore for loss computation.
            beta (float): Weight for the preference loss.
            compiled (bool): Whether to use torch compile for chunk accumulation.
            use_ref_model (bool): Whether to use a reference model for the alignment loss.
            preference_labels (torch.Tensor): Boolean tensor indicating chosen (True) vs rejected (False) examples.
                Shape: (batch_size,).
            ref_weight (torch.Tensor): Reference weight tensor. Shape: (vocab_size, hidden_size).
            ref_bias (torch.Tensor, optional): Reference bias tensor. Shape: (vocab_size,).
            loss_kwargs (dict): Other possible arguments that a loss function might need
        """
        # TODO: Tune CHUNK_SIZE to fully utilize the GPU
        CHUNK_SIZE = chunk_size

        # Gradients to be accumulated
        grad_inputs = []
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias) if bias is not None else None

        # Loss to be accumulated
        loss_acc = torch.zeros((), device=_input.device)

        # Metrics to be recorded
        policy_chosen_logps = []
        policy_rejected_logps = []
        policy_chosen_logits_mean = torch.zeros((), device=_input.device)
        policy_rejected_logits_mean = torch.zeros((), device=_input.device)
        policy_nll_loss = torch.zeros((), device=_input.device)
        aggregated_aux_outputs = []  # aggregated aux outputs from all chunks

        compute_loss = partial(
            LigerFusedLinearUnpairedPreferenceBase._compute_loss,
            preference_loss_fn=loss_fn,
            ignore_index=ignore_index,
            alpha=alpha,
            beta=beta,
            full_target=target,
            use_ref_model=use_ref_model,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            preference_labels=preference_labels,
            **loss_kwargs,
        )

        def fused_fwd_bwd(
            input_chunk, target_chunk, ref_input_chunk, preference_labels_chunk
        ):
            """
            Fused forward and backward pass for a chunk of input and target.
            """
            if bias is not None:
                return torch.func.grad_and_value(
                    compute_loss, argnums=(0, 1, 3), has_aux=True
                )(
                    input_chunk,
                    weight,
                    target_chunk,
                    bias,
                    ref_input_chunk=ref_input_chunk,
                    preference_labels=preference_labels_chunk,
                )
            else:
                return torch.func.grad_and_value(
                    compute_loss, argnums=(0, 1), has_aux=True
                )(
                    input_chunk,
                    weight,
                    target_chunk,
                    ref_input_chunk=ref_input_chunk,
                    preference_labels=preference_labels_chunk,
                )

        def accumulate_chunk(
            input_chunk,
            target_chunk,
            ref_input_chunk=None,
            preference_labels_chunk=None,
        ):
            if bias is not None:
                (chunk_grad_input, chunk_grad_weight, chunk_grad_bias), (
                    chunk_loss,
                    (
                        chunk_chosen_logps,
                        chunk_rejected_logps,
                        chunk_chosen_logits_mean,
                        chunk_rejected_logits_mean,
                        chunk_policy_nll_loss,
                        *aux_outputs,
                    ),
                ) = fused_fwd_bwd(
                    input_chunk, target_chunk, ref_input_chunk, preference_labels_chunk
                )
                grad_bias.add_(chunk_grad_bias)  # accumulate bias gradient
            else:
                (chunk_grad_input, chunk_grad_weight), (
                    chunk_loss,
                    (
                        chunk_chosen_logps,
                        chunk_rejected_logps,
                        chunk_chosen_logits_mean,
                        chunk_rejected_logits_mean,
                        chunk_policy_nll_loss,
                        *aux_outputs,
                    ),
                ) = fused_fwd_bwd(
                    input_chunk, target_chunk, ref_input_chunk, preference_labels_chunk
                )

            # Accumulate gradients
            grad_weight.add_(chunk_grad_weight)
            grad_inputs.append(chunk_grad_input)

            # Accumulate loss
            loss_acc.add_(chunk_loss)

            # Accumulate metrics
            policy_chosen_logps.append(chunk_chosen_logps)
            policy_rejected_logps.append(chunk_rejected_logps)
            policy_chosen_logits_mean.add_(chunk_chosen_logits_mean)
            policy_rejected_logits_mean.add_(chunk_rejected_logits_mean)
            policy_nll_loss.add_(chunk_policy_nll_loss)

            # aux_outputs
            # Initialize storage for aux_outputs
            if len(aggregated_aux_outputs) == 0:
                for aux in aux_outputs:
                    if aux.ndim == 0:
                        aggregated_aux_outputs.append(
                            torch.zeros((), device=aux.device)
                        )
                    else:
                        aggregated_aux_outputs.append([])

            # Process each aux_output
            for i, aux in enumerate(aux_outputs):
                if aux.ndim == 0:
                    aggregated_aux_outputs[i].add_(aux)
                else:
                    aggregated_aux_outputs[i].append(aux)

        if compiled:
            fused_fwd_bwd = torch.compile(fused_fwd_bwd)

        # When not paired, use labels to separate chosen and rejected
        assert (
            preference_labels is not None
        ), "preference_labels must be provided for unpaired preference loss"

        # TODO: Update the 2 * CHUNK_SIZE to CHUNK_SIZE
        chunks = max(1, _input.shape[0] // (2 * CHUNK_SIZE))
        _input_chunks = torch.chunk(_input, chunks=chunks, dim=0)
        _target_chunks = torch.chunk(target, chunks=chunks, dim=0)
        _preference_labels_chunks = torch.chunk(preference_labels, chunks=chunks, dim=0)

        if use_ref_model:
            _ref_input_chunks = torch.chunk(ref_input, chunks=chunks, dim=0)

        for (
            input_chunk,
            target_chunk,
            ref_input_chunk,
            preference_labels_chunk,
        ) in zip(
            _input_chunks,
            _target_chunks,
            (_ref_input_chunks if use_ref_model else [None] * len(_input_chunks)),
            _preference_labels_chunks,
        ):
            input_chunk = input_chunk
            ref_input_chunk = ref_input_chunk if use_ref_model else None
            target_chunk = target_chunk

            # mark input_chunk, target_chunk, and target dimension 1 as dynamic to prevent torch.compile recompilation
            torch._dynamo.mark_dynamic(input_chunk, 1)
            torch._dynamo.mark_dynamic(target_chunk, 1)
            torch._dynamo.mark_dynamic(target, 1)
            torch._dynamo.mark_dynamic(ref_input_chunk, 1) if use_ref_model else None
            torch._dynamo.mark_dynamic(preference_labels_chunk, 1)

            # accumulate loss, gradients, and metrics
            accumulate_chunk(
                input_chunk, target_chunk, ref_input_chunk, preference_labels_chunk
            )

        policy_chosen_logps = torch.cat(policy_chosen_logps, dim=0)
        policy_rejected_logps = torch.cat(policy_rejected_logps, dim=0)

        # Aggregate aux outputs lists into tensors
        for i, aux in enumerate(aggregated_aux_outputs):
            if isinstance(aux, list):
                aggregated_aux_outputs[i] = torch.cat(aux, dim=0)

        ctx.save_for_backward(
            torch.cat(grad_inputs, dim=0),
            grad_weight,
            grad_bias,
        )
        return_vars = (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits_mean,
            policy_rejected_logits_mean,
            policy_nll_loss,
        )
        return loss_acc, (*return_vars, *aggregated_aux_outputs)

    @staticmethod
    def backward(ctx, *grad_output):
        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        if torch.ne(
            grad_output[0][0], torch.tensor(1.0, device=grad_output[0][0].device)
        ):
            grad_input = grad_input * grad_output[0][0]
            grad_weight = grad_weight * grad_output[0][0]
            grad_bias = grad_bias * grad_output[0][0] if grad_bias is not None else None

        return grad_input, grad_weight, None, grad_bias, None, None, None

    @staticmethod
    def chunk_forward(
        input_chunk,
        weight,
        target_chunk,
        bias=None,
        ignore_index=-100,
        preference_labels=None,
    ):
        # Data is not ordered, so we need to use preference_labels to separate chosen and rejected
        logits_chunk = input_chunk @ weight.t()
        if bias is not None:
            logits_chunk = logits_chunk + bias
        log_probs_chunk = F.log_softmax(logits_chunk.float(), dim=-1)

        loss_mask = target_chunk != ignore_index
        label_chunk = torch.where(loss_mask, target_chunk, 0)

        per_token_logps = log_probs_chunk.gather(-1, label_chunk.unsqueeze(-1)).squeeze(
            -1
        )
        average_log_prob = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)

        chosen_logps = average_log_prob[preference_labels == 1]
        rejected_logps = average_log_prob[preference_labels == 0]

        chosen_logits = logits_chunk[preference_labels == 1]
        rejected_logits = logits_chunk[preference_labels == 0]
        policy_nll_loss = torch.tensor(0.0, device=chosen_logps.device)

        return (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            policy_nll_loss,
        )

    @staticmethod
    def _compute_loss(
        input_chunk,
        weight,
        target_chunk,
        bias=None,
        preference_loss_fn=None,
        full_target=None,
        ignore_index=-100,
        alpha=1.0,
        beta=0.1,
        use_ref_model=False,
        ref_input_chunk=None,
        ref_weight=None,
        ref_bias=None,
        preference_labels=None,
        **loss_kwargs,
    ):
        """
        Compute the total loss for a chunk of input and target, while using an alignment/preference loss function.
        Args:
            preference_loss_fn (callable): Loss function to compute the loss on a chunk of input/target.
            input_chunk (torch.Tensor): Chunk of input tensor. Shape: (2 * chunk_size, sequence_length, hidden_size).
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size).
            target_chunk (torch.Tensor): Chunk of target tensor. Shape: (2 * chunk_size, sequence_length).
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,).
            full_target (torch.Tensor): Full target tensor. Shape: (batch_size, sequence_length).
            ignore_index (int): Index to ignore for loss computation.
            beta (float): Weight for the preference loss.
            use_ref_model (bool): Whether to use a reference model for the alignment loss.
            ref_weight (torch.Tensor): Reference weight tensor. Shape: (vocab_size, hidden_size).
            ref_bias (torch.Tensor, optional): Reference bias tensor. Shape: (vocab_size,).
            loss_kwargs (dict): Additional arguments for the loss function.
        """
        (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            policy_nll_loss,
        ) = LigerFusedLinearUnpairedPreferenceBase.chunk_forward(
            input_chunk,
            weight,
            target_chunk,
            bias=bias,
            ignore_index=ignore_index,
            preference_labels=preference_labels,
        )

        num_chosen = (preference_labels == 1).sum()
        num_rejected = (preference_labels == 0).sum()

        chosen_logits_mean = (
            chosen_logits.sum() / (num_chosen * input_chunk.shape[1] * weight.shape[0])
            if num_chosen > 0
            else torch.tensor(0.0, device=chosen_logits.device)
        )
        rejected_logits_mean = (
            rejected_logits.sum()
            / (num_rejected * input_chunk.shape[1] * weight.shape[0])
            if num_rejected > 0
            else torch.tensor(0.0, device=rejected_logits.device)
        )

        if use_ref_model:
            with torch.no_grad():
                (
                    ref_chosen_logps,
                    ref_rejected_logps,
                    ref_chosen_logits,
                    ref_rejected_logits,
                    policy_nll_loss,
                ) = LigerFusedLinearUnpairedPreferenceBase.chunk_forward(
                    ref_input_chunk,
                    ref_weight,
                    target_chunk,
                    ref_bias,
                    ignore_index=ignore_index,
                    preference_labels=preference_labels,
                )
            loss_kwargs["ref_chosen_logps"] = ref_chosen_logps
            loss_kwargs["ref_rejected_logps"] = ref_rejected_logps

        preference_loss_outputs = preference_loss_fn(
            chosen_logps, rejected_logps, full_target, beta=beta, **loss_kwargs
        )
        if isinstance(preference_loss_outputs, tuple):
            preference_loss, *aux_outputs = preference_loss_outputs
        else:
            preference_loss, aux_outputs = preference_loss_outputs, []

        loss = 0 - preference_loss
        return_vars = (
            chosen_logps,
            rejected_logps,
            chosen_logits_mean,
            rejected_logits_mean,
            policy_nll_loss,
        )
        return loss, (*return_vars, *aux_outputs)
