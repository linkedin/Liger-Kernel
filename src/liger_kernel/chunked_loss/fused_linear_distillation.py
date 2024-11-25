from abc import abstractmethod
from functools import partial

import torch
from torch.nn import functional as F


class LigerFusedLinearDistillationBase(torch.autograd.Function):

    @abstractmethod
    def distillation_loss_fn(teacher_logits, student_logits, beta=0.5):
        """
        Compute preference loss.
        Args:
            teacher_logits (torch.Tensor): Logits from the teacher model
            student_logits (torch.Tensor): Logits from the student model
            beta (float): Weight for the loss
        Returns:
            torch.Tensor: Computed loss
        """
        raise NotImplementedError("Preference loss function must be implemented.")

    @staticmethod
    def chunk_forward(
        student_input_chunk,
        student_weight,
        teacher_input_chunk,
        teacher_weight,
        target_chunk,
        bias=None,
        ignore_index=-100,
        compute_ce_loss=True,
    ):
        # Compute logits for student and teacher
        student_logits_chunk = student_input_chunk @ student_weight.t()
        if bias is not None:
            student_logits_chunk += bias

        # Compute cross-entropy loss
        chosen_nll_loss = 0.0
        if compute_ce_loss:
            chosen_nll_loss = F.cross_entropy(
                student_logits_chunk.view(-1, student_logits_chunk.shape[-1]),
                target_chunk.view(-1),
                reduction='sum',
                ignore_index=ignore_index,
            )

        # Create loss mask
        loss_mask = target_chunk != ignore_index
        label_chunk = torch.where(loss_mask, target_chunk, 0)

        # Compute average logits for student
        student_per_token_logits = student_logits_chunk.gather(-1, label_chunk.unsqueeze(-1)).squeeze(-1)
        student_average_logits = (student_per_token_logits * loss_mask).sum(-1) / loss_mask.sum(-1)

        # Compute logits for teacher
        teacher_logits_chunk = teacher_input_chunk @ teacher_weight.t()
        if bias is not None:
            teacher_logits_chunk += bias

        # Compute average logits for teacher
        teacher_per_token_logits = teacher_logits_chunk.gather(-1, label_chunk.unsqueeze(-1)).squeeze(-1)
        teacher_average_logits = (teacher_per_token_logits * loss_mask).sum(-1) / loss_mask.sum(-1)

        return student_average_logits, teacher_average_logits, chosen_nll_loss

    @staticmethod
    def forward(
        ctx,
        student_input,
        student_weight,
        teacher_input,
        teacher_weight,
        target,
        bias=None,
        loss_fn=None,
        chunk_size=1,
        ignore_index=-100,
        beta=0.1,
        compute_ce_loss=True,
        compiled=True,
        **loss_kwargs,
    ):
        # Input validation
        if loss_fn is None:
            raise ValueError("A loss function must be provided")

        CHUNK_SIZE = chunk_size

        # Initialize gradient accumulators
        grad_weight = torch.zeros_like(student_weight)
        grad_bias = torch.zeros_like(bias) if bias is not None else None
        loss_acc = torch.zeros((), device=student_input.device)

        # Partial function for loss computation
        loss_func_to_call = partial(
            LigerFusedLinearDistillationBase._compute_loss,
            distillation_loss_fn=loss_fn,
            ignore_index=ignore_index,
            beta=beta,
            compute_nll_loss=compute_ce_loss,
            full_target=target,
            bias=bias,
            **loss_kwargs,
        )

        def accumulate_chunk(student_input_chunk, teacher_input_chunk, target_chunk):
            if bias is not None:
                (chunk_grad_input, chunk_grad_weight, chunk_grad_bias), (
                    chunk_loss,
                    (chunk_or_loss, chunk_chosen_logps, chunk_rejected_logps),
                ) = torch.func.grad_and_value(
                    loss_func_to_call, argnums=(0, 1, 3), has_aux=True
                )(
                    student_input_chunk, student_weight, teacher_input_chunk, teacher_weight, target_chunk, bias
                )
                grad_bias.add_(chunk_grad_bias)
            else:
                (chunk_grad_input, chunk_grad_weight), (
                    chunk_loss,
                    (chunk_or_loss, chunk_chosen_logps, chunk_rejected_logps),
                ) = torch.func.grad_and_value(
                    loss_func_to_call, argnums=(0, 1), has_aux=True
                )(
                    student_input_chunk, student_weight, teacher_input_chunk, teacher_weight, target_chunk
                )
            grad_weight.add_(chunk_grad_weight)
            loss_acc.add_(chunk_loss)
            return chunk_grad_input


        # Optional compilation
        if compiled:
            accumulate_chunk = torch.compile(accumulate_chunk)

        # Chunk processing
        chunks = max(1, student_input.shape[0] // (2 * CHUNK_SIZE))
        student_input_chunks = torch.chunk(student_input, chunks=chunks, dim=0)
        teacher_input_chunks = torch.chunk(teacher_input, chunks=chunks, dim=0)
        _target_chunks = torch.chunk(target, chunks=chunks, dim=0)

        grad_inputs = []
        for student_input_chunk, teacher_input_chunk, target_chunk in zip(
            student_input_chunks, teacher_input_chunks, _target_chunks
        ):
            grad_input = accumulate_chunk(student_input_chunk, teacher_input_chunk, target_chunk)
            grad_inputs.append(grad_input[: target_chunk.shape[0]])

        # Save tensors for backward pass
        ctx.save_for_backward(
            torch.cat(grad_inputs, dim=0),
            grad_weight,
            grad_bias,
        )
        return loss_acc

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        
        # Scale gradients by grad_output
        grad_input = grad_input * grad_output
        grad_weight = grad_weight * grad_output
        grad_bias = grad_bias * grad_output if grad_bias is not None else None

        return grad_input, grad_weight, None, None, None, grad_bias, None, None, None, None

    @staticmethod
    def _compute_loss(
        student_input_chunk,
        student_weight,
        teacher_input_chunk,
        teacher_weight,
        target_chunk,
        bias=None,
        distillation_loss_fn=None,
        full_target=None,
        ignore_index=-100,
        beta=0.1,
        compute_nll_loss=True,
        **loss_kwargs,
    ):
        # Compute forward pass with chunk_forward
        student_logits, teacher_logits, chosen_nll_loss = (
            LigerFusedLinearDistillationBase.chunk_forward(
                student_input_chunk,
                student_weight,
                teacher_input_chunk,
                teacher_weight,
                target_chunk,
                bias=bias,
                ignore_index=ignore_index,
                compute_ce_loss=compute_nll_loss,
            )
        )

        # Normalize NLL loss
        chosen_nll_loss = (
            chosen_nll_loss / (full_target != ignore_index).sum()
        )

        # Compute distillation loss
        distillation_loss = distillation_loss_fn(
            teacher_logits, student_logits, beta=beta, **loss_kwargs
        )
        distillation_loss = distillation_loss / (full_target.shape[0])

        # Combine losses
        loss = beta * chosen_nll_loss + (1 - beta) * distillation_loss
        return loss, (distillation_loss, student_logits, teacher_logits)