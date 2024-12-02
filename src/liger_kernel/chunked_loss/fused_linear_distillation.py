from abc import abstractmethod
from functools import partial

import torch
from torch.nn import functional as F


class LigerFusedLinearDistillationBase(torch.autograd.Function):

    @abstractmethod
    def distillation_loss_fn(student_logits, teacher_logits, temperature):
        """
        Compute distillation loss.
        Args:
            student_logits (torch.Tensor): Raw logits of student tokens. Shape: (batch_size * seq_len, vocab_size).
            teacher_logits (torch.Tensor): Raw logits of teacher tokens. Shape: (batch_size * seq_len, vocab_size).
        """
        raise NotImplementedError("Distillation loss function must be implemented.")

    @staticmethod
    def chunk_forward(
        student_input_chunk,
        student_weight,
        teacher_input_chunk,
        teacher_weight,
        target_chunk,
        student_bias=None,
        teacher_bias=None,
        ignore_index=-100,
        compute_ce_loss=True,
    ):
        # Student
        student_per_token_logits_chunk = student_input_chunk @ student_weight.t()
        if student_bias is not None:
            student_per_token_logits_chunk += student_bias
        student_per_token_log_probs_chunk = F.log_softmax(
            student_per_token_logits_chunk.float(), dim=-1
        )

        # Teacher
        teacher_per_token_logits_chunk = teacher_input_chunk @ teacher_weight.t()
        if teacher_bias is not None:
            teacher_per_token_logits_chunk += teacher_bias

        # The hard/task loss
        ce_loss = 0.0
        if compute_ce_loss:
            ce_loss = F.cross_entropy(
                student_per_token_log_probs_chunk.view(
                    -1, student_per_token_log_probs_chunk.shape[-1]
                ),
                target_chunk.view(-1),
                reduction="sum",
                ignore_index=ignore_index,
            )

        return student_per_token_logits_chunk, teacher_per_token_logits_chunk, ce_loss

    @staticmethod
    def forward(
        ctx,
        student_input,
        student_weight,
        teacher_input,
        teacher_weight,
        target,
        student_bias=None,
        teacher_bias=None,
        loss_fn=None,
        chunk_size=1,
        ignore_index=-100,
        beta=0.5,
        compute_ce_loss=True,
        temperature=1.0,
        compiled=True,
        **loss_kwargs,
    ):
        """
        Base class for fused linear layer with distillation loss.
        Only need to compute gradients for student model.

        Args:
            student_input (torch.Tensor): Student input tensor. Shape: (batch_size * seq_len, hidden_size).
            student_weight (torch.Tensor): Student weight tensor. Shape: (vocab_size, hidden_size).
            teacher_input (torch.Tensor): Teacher input tensor. Shape: (batch_size * seq_len, hidden_size).
            teacher_weight (torch.Tensor): Teacher weight tensor. Shape: (vocab_size, hidden_size).
            target (torch.Tensor): Target truth label tensor. Shape: (batch_size * seq_len).
            student_bias (torch.Tensor, optional): Student bias tensor. Shape: (vocab_size,).
            teacher_bias (torch.Tensor, optional): Teacher bias tensor. Shape: (vocab_size,).
            loss_fn (callable): Loss function to compute the loss on a chunk of input/target.
            chunk_size (int): Size of a chunk (# of batches of stacked chosen and rejected inputs).
            compute_ce_loss (bool): Whether to compute CE loss.
            ignore_index (int): Index to ignore for loss computation.
            beta (float): Weight between soft and hard loss.
            compiled (bool): Whether to use torch compile for chunk accumulation.
            loss_kwargs (dict): Other possible arguments that a loss function might need
        """
        CHUNK_SIZE = chunk_size

        grad_weight = torch.zeros_like(student_weight)
        grad_inputs = []
        grad_bias = torch.zeros_like(student_bias) if student_bias is not None else None
        loss_acc = torch.zeros((), device=student_input.device)

        loss_func_to_call = partial(
            LigerFusedLinearDistillationBase._compute_loss,
            distillation_loss_fn=loss_fn,
            ignore_index=ignore_index,
            beta=beta,
            compute_ce_loss=compute_ce_loss,
            temperature=temperature,
            full_target=target,
            **loss_kwargs,
        )

        def accumulate_chunk(student_input_chunk, teacher_input_chunk, target_chunk):
            if student_bias is not None:
                (chunk_grad_input, chunk_grad_weight, chunk_grad_bias), (
                    chunk_loss,
                    (
                        chunk_soft_loss,
                        chunk_hard_loss,
                        chunk_student_logits,
                        chunk_teacher_logits,
                    ),
                ) = torch.func.grad_and_value(
                    loss_func_to_call, argnums=(0, 1, 5), has_aux=True
                )(
                    student_input_chunk,
                    student_weight,
                    teacher_input_chunk,
                    teacher_weight,
                    target_chunk,
                    student_bias,
                    teacher_bias,
                )
                grad_bias.add_(chunk_grad_bias)
            else:
                (chunk_grad_input, chunk_grad_weight), (
                    chunk_loss,
                    (
                        chunk_soft_loss,
                        chunk_hard_loss,
                        chunk_student_logits,
                        chunk_teacher_logits,
                    ),
                ) = torch.func.grad_and_value(
                    loss_func_to_call, argnums=(0, 1), has_aux=True
                )(
                    student_input_chunk,
                    student_weight,
                    teacher_input_chunk,
                    teacher_weight,
                    target_chunk,
                    student_bias,
                    teacher_bias,
                )
            grad_weight.add_(chunk_grad_weight)
            loss_acc.add_(chunk_loss)
            return chunk_grad_input

        if compiled:
            accumulate_chunk = torch.compile(accumulate_chunk)

        num_chunks = max(1, student_input.shape[0] // CHUNK_SIZE)
        _student_input_chunks = torch.chunk(student_input, chunks=num_chunks, dim=0)
        _teacher_input_chunks = torch.chunk(teacher_input, chunks=num_chunks, dim=0)
        _target_chunks = torch.chunk(target, chunks=num_chunks, dim=0)

        for student_input_chunk, teacher_input_chunk, target_chunk in zip(
            _student_input_chunks, _teacher_input_chunks, _target_chunks
        ):
            grad_input = accumulate_chunk(
                student_input_chunk, teacher_input_chunk, target_chunk
            )
            grad_inputs.append(grad_input)

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
        student_input_chunk,
        student_weight,
        teacher_input_chunk,
        teacher_weight,
        target_chunk,
        student_bias=None,
        teacher_bias=None,
        distillation_loss_fn=None,
        full_target=None,
        ignore_index=-100,
        temperature=1.0,
        beta=0.5,
        compute_ce_loss=True,
        **loss_kwargs,
    ):
        """
        Compute the total loss for a chunk of input and target, while using an knowleedge distillation loss function.
        Args:
            distillation_loss_fn (callable): Loss function to compute the loss on a chunk of input/target.
            student_input_chunk (torch.Tensor): Chunk of input tensor. Shape: (chunk_size, hidden_size).
            student_weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size).
            teacher_input_chunk (torch.Tensor): Chunk of input tensor. Shape: (chunk_size, hidden_size).
            teacher_weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size).
            target_chunk (torch.Tensor): Chunk of target tensor. Shape: (chunk_size,).
            student_bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,).
            teacher_bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,).
            full_target (torch.Tensor): Full target tensor. Shape: (chunk_size,).
            ignore_index (int): Index to ignore for loss computation.
            beta (float): Weight between soft and hard loss.
            compute_ce_loss (bool): Whether to compute CE loss.
            loss_kwargs (dict): Additional arguments for the loss function.
        """
        student_logits, teacher_logits, hard_loss = (
            LigerFusedLinearDistillationBase.chunk_forward(
                student_input_chunk,
                student_weight,
                teacher_input_chunk,
                teacher_weight,
                target_chunk,
                student_bias=student_bias,
                teacher_bias=teacher_bias,
                ignore_index=ignore_index,
                compute_ce_loss=compute_ce_loss,
            )
        )

        hard_loss = hard_loss / (full_target != ignore_index).sum()

        soft_loss = distillation_loss_fn(student_logits, teacher_logits, temperature)
        soft_loss = soft_loss / (full_target != ignore_index).sum()

        loss = beta * hard_loss + (1 - beta) * soft_loss
        return loss, (soft_loss, hard_loss, student_logits, teacher_logits)
