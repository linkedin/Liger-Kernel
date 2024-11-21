from abc import abstractmethod
from functools import partial

import torch
from torch.nn import functional as F


class LigerFusedLinearPreferenceBase(torch.autograd.Function):

    @abstractmethod
    def preference_loss_fn(chosen_logps, rejected_logps, beta=0.1):
        """
        Compute preference loss.
        Args:
            chosen_logps (torch.Tensor): Avg log probabilities of chosen tokens. Shape: (batch_size,).
            rejected_logps (torch.Tensor): Avg log probabilities of rejected tokens. Shape: (batch_size,).
            beta (float): Weight for the odds ratio loss.
        """
        raise NotImplementedError("Preference loss function must be implemented.")

    @staticmethod
    def chunk_forward(
        input_chunk,
        weight,
        target_chunk,
        bias=None,
        ignore_index=-100,
        compute_nll_loss=True,
    ):
        len_chosen_chunk = target_chunk.shape[0] // 2
        logits_chunk = input_chunk @ weight.t()
        if bias is not None:
            logits_chunk = logits_chunk + bias
        log_probs_chunk = F.log_softmax(logits_chunk.float(), dim=-1)

        chosen_nll_loss = 0.0
        if compute_nll_loss:
            chosen_nll_loss = F.nll_loss(
                log_probs_chunk[:len_chosen_chunk].view(-1, log_probs_chunk.shape[-1]),
                target_chunk[:len_chosen_chunk].view(-1),
                reduction="sum",
                ignore_index=ignore_index,
            )

        loss_mask = target_chunk != ignore_index
        label_chunk = torch.where(loss_mask, target_chunk, 0)

        per_token_logps = log_probs_chunk.gather(-1, label_chunk.unsqueeze(-1)).squeeze(
            -1
        )
        average_log_prob = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)

        chosen_logps = average_log_prob[:len_chosen_chunk]
        rejected_logps = average_log_prob[len_chosen_chunk:]
        return chosen_logps, rejected_logps, chosen_nll_loss

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
        compute_nll_loss=True,
        compiled=True,
        use_ref_model=False,
        ref_weight=None,
        ref_bias=None,
        **loss_kwargs,
    ):
        """
        Base class for fused linear layer with preference loss.
        Expects _input to be stacked with chosen and rejected inputs on the batch dimension.

        Args:
            _input (torch.Tensor): Input tensor. Shape: (batch_size, seq_len, hidden_size).
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size).
            target (torch.Tensor): Target tensor. Shape: (batch_size, seq_len).
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,).
            loss_fn (callable): Loss function to compute the loss on a chunk of input/target.
            chunk_size (int): Size of a chunk (# of batches of stacked chosen and rejected inputs).
            compute_nll_loss (bool): Whether to compute NLL loss.
            ignore_index (int): Index to ignore for loss computation.
            alpha (float): Weight for the NLL loss.
            beta (float): Weight for the odds ratio loss.
            compute_nll_loss (bool): Whether to compute NLL loss.
            compiled (bool): Whether to use torch compile for chunk accumulation.
            use_ref_model (bool): Whether to use a reference model for the alignment loss.
            ref_weight (torch.Tensor): Reference weight tensor. Shape: (vocab_size, hidden_size).
            ref_bias (torch.Tensor, optional): Reference bias tensor. Shape: (vocab_size,).
            loss_kwargs (dict): Other possible arguments that a loss function might need
        """
        # TODO: Tune CHUNK_SIZE to fully utilize the GPU
        CHUNK_SIZE = chunk_size

        grad_weight = torch.zeros_like(weight)
        grad_chosen_inputs = []
        grad_rejected_inputs = []
        grad_bias = torch.zeros_like(bias) if bias is not None else None
        loss_acc = torch.zeros((), device=_input.device)

        loss_func_to_call = partial(
            LigerFusedLinearPreferenceBase._compute_loss,
            preference_loss_fn=loss_fn,
            ignore_index=ignore_index,
            alpha=alpha,
            beta=beta,
            compute_nll_loss=compute_nll_loss,
            full_target=target,
            use_ref_model=use_ref_model,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            **loss_kwargs,
        )

        def accumulate_chunk(input_chunk, target_chunk):
            if bias is not None:
                (chunk_grad_input, chunk_grad_weight, chunk_grad_bias), (
                    chunk_loss,
                    (chunk_or_loss, chunk_chosen_logps, chunk_rejected_logps),
                ) = torch.func.grad_and_value(
                    loss_func_to_call, argnums=(0, 1, 3), has_aux=True
                )(
                    input_chunk, weight, target_chunk, bias
                )
                grad_bias.add_(chunk_grad_bias)
            else:
                (chunk_grad_input, chunk_grad_weight), (
                    chunk_loss,
                    (chunk_or_loss, chunk_chosen_logps, chunk_rejected_logps),
                ) = torch.func.grad_and_value(
                    loss_func_to_call, argnums=(0, 1), has_aux=True
                )(
                    input_chunk, weight, target_chunk
                )
            grad_weight.add_(chunk_grad_weight)
            loss_acc.add_(chunk_loss)
            return chunk_grad_input

        if compiled:
            accumulate_chunk = torch.compile(accumulate_chunk)

        len_chosen = target.shape[0] // 2
        chunks = max(1, _input.shape[0] // (2 * CHUNK_SIZE))
        _chosen_input_chunks = torch.chunk(_input[:len_chosen], chunks=chunks, dim=0)
        _chosen_target_chunks = torch.chunk(target[:len_chosen], chunks=chunks, dim=0)
        _rejected_input_chunks = torch.chunk(_input[len_chosen:], chunks=chunks, dim=0)
        _rejected_target_chunks = torch.chunk(target[len_chosen:], chunks=chunks, dim=0)

        for (
            chosen_input_chunk,
            rejected_input_chunk,
            chosen_target_chunk,
            rejected_target_chunk,
        ) in zip(
            _chosen_input_chunks,
            _rejected_input_chunks,
            _chosen_target_chunks,
            _rejected_target_chunks,
        ):
            input_chunk = torch.cat([chosen_input_chunk, rejected_input_chunk], dim=0)
            target_chunk = torch.cat(
                [chosen_target_chunk, rejected_target_chunk], dim=0
            )

            grad_input = accumulate_chunk(input_chunk, target_chunk)

            grad_chosen_inputs.append(grad_input[: chosen_target_chunk.shape[0]])
            grad_rejected_inputs.append(grad_input[chosen_target_chunk.shape[0] :])

        # combine grad_chosen_inputs and grad_rejected_inputs
        grad_inputs = grad_chosen_inputs + grad_rejected_inputs

        ctx.save_for_backward(
            torch.cat(grad_inputs, dim=0),
            grad_weight,
            grad_bias,
        )
        return loss_acc

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        if torch.ne(grad_output, torch.tensor(1.0, device=grad_output.device)):
            grad_input = grad_input * grad_output
            grad_weight = grad_weight * grad_output
            grad_bias = grad_bias * grad_output if grad_bias is not None else None

        return grad_input, grad_weight, None, grad_bias, None, None, None

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
        compute_nll_loss=True,
        use_ref_model=False,
        ref_weight=None,
        ref_bias=None,
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
            alpha (float): Weight for the NLL loss.
            beta (float): Weight for the odds ratio loss.
            compute_nll_loss (bool): Whether to compute NLL loss.
            use_ref_model (bool): Whether to use a reference model for the alignment loss.
            ref_weight (torch.Tensor): Reference weight tensor. Shape: (vocab_size, hidden_size).
            ref_bias (torch.Tensor, optional): Reference bias tensor. Shape: (vocab_size,).
            loss_kwargs (dict): Additional arguments for the loss function.
        """
        chosen_logps, rejected_logps, chosen_nll_loss = (
            LigerFusedLinearPreferenceBase.chunk_forward(
                input_chunk,
                weight,
                target_chunk,
                bias=bias,
                ignore_index=ignore_index,
                compute_nll_loss=compute_nll_loss,
            )
        )
        chosen_nll_loss = (
            chosen_nll_loss
            / (full_target[: full_target.shape[0] // 2] != ignore_index).sum()
        )

        if use_ref_model:
            with torch.no_grad():
                ref_chosen_logps, ref_rejected_logps, _ = (
                    LigerFusedLinearPreferenceBase.chunk_forward(
                        input_chunk,
                        ref_weight,
                        target_chunk,
                        ref_bias,
                        ignore_index=ignore_index,
                        compute_nll_loss=False,
                    )
                )
            loss_kwargs["ref_chosen_logps"] = ref_chosen_logps
            loss_kwargs["ref_rejected_logps"] = ref_rejected_logps

        alignment_loss = preference_loss_fn(
            chosen_logps, rejected_logps, beta=beta, **loss_kwargs
        )
        alignment_loss = alignment_loss / (full_target.shape[0] // 2)

        loss = alpha * chosen_nll_loss - alignment_loss
        return loss, (alignment_loss, chosen_logps, rejected_logps)
