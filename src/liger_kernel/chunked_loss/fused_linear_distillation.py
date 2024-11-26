from abc import abstractmethod
from functools import partial
import torch
from torch.nn import functional as F


class LigerFusedLinearDistillationBase(torch.autograd.Function):
    @abstractmethod
    def distill_loss_fn(student_logits, teacher_logits, temperature=1.0):
        """
        Compute distillation loss.
        Args:
            student_logits (torch.Tensor): Logits from the student model. Shape: (batch_size, seq_len, vocab_size).
            teacher_logits (torch.Tensor): Logits from the teacher model. Shape: (batch_size, seq_len, vocab_size).
            temperature (float): Temperature for softening probability distributions.
        """
        raise NotImplementedError("Distillation loss function must be implemented.")

    @staticmethod
    def chunk_forward(
        student_chunk,
        teacher_chunk,
        student_weight,
        teacher_weight,
        target_chunk,
        bias=None,
        ignore_index=-100,
        compute_ce_loss=True,
    ):
        # Project both student and teacher inputs to logits
        student_logits_chunk = student_chunk @ student_weight.t()
        teacher_logits_chunk = teacher_chunk @ teacher_weight.t()

        if bias is not None:
            student_logits_chunk += bias
            teacher_logits_chunk += bias

        ce_loss = 0.0
        if compute_ce_loss:
            ce_loss = F.cross_entropy(
                student_logits_chunk.view(-1, student_logits_chunk.shape[-1]),
                target_chunk.view(-1),
                reduction="sum",
                ignore_index=ignore_index,
            )

        return student_logits_chunk, teacher_logits_chunk, ce_loss

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
        beta=1.0,
        temperature=1.0,
        compute_ce_loss=True,
        compiled=True,
        **loss_kwargs,
    ):
        """
        Base class for fused linear layer with distillation loss, supporting differing input sizes.
        Args:
            student_input (torch.Tensor): Input tensor from the student model. Shape: (batch_size, seq_len, student_hidden_size).
            teacher_input (torch.Tensor): Input tensor from the teacher model. Shape: (batch_size, seq_len, teacher_hidden_size).
            student_weight (torch.Tensor): Weight tensor for the student model. Shape: (vocab_size, student_hidden_size).
            teacher_weight (torch.Tensor): Weight tensor for the teacher model. Shape: (vocab_size, teacher_hidden_size).
            target (torch.Tensor): Target tensor. Shape: (batch_size, seq_len).
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,).
            loss_fn (callable): Loss function to compute the distillation loss.
            chunk_size (int): Size of a chunk (# of batches).
            compute_ce_loss (bool): Whether to compute cross-entropy loss.
            ignore_index (int): Index to ignore for loss computation.
            beta (float): Weight for the cross-entropy loss.
            temperature (float): Temperature for softening probability distributions.
            compiled (bool): Whether to use torch compile for chunk accumulation.
            loss_kwargs (dict): Other possible arguments that a loss function might need.
        """
        CHUNK_SIZE = chunk_size
        grad_student_weight = torch.zeros_like(student_weight)
        grad_teacher_weight = torch.zeros_like(teacher_weight)
        grad_inputs = []
        grad_bias = torch.zeros_like(bias) if bias is not None else None
        loss_acc = torch.zeros((), device=student_input.device)

        loss_func_to_call = partial(
            LigerFusedLinearDistillationBase._compute_loss,
            distill_loss_fn=loss_fn,
            ignore_index=ignore_index,
            beta=beta,
            temperature=temperature,
            compute_ce_loss=compute_ce_loss,
            full_target=target,
            **loss_kwargs,
        )

        def accumulate_chunk(
            student_chunk, teacher_chunk, target_chunk
        ):
            if bias is not None:
                (chunk_grad_input, chunk_grad_student_weight, chunk_grad_teacher_weight, chunk_grad_bias), (
                    chunk_loss,
                    (chunk_distill_loss, chunk_student_logits),
                ) = torch.func.grad_and_value(
                    loss_func_to_call, argnums=(0, 1, 2, 3), has_aux=True
                )(
                    student_chunk,
                    teacher_chunk,
                    student_weight,
                    teacher_weight,
                    target_chunk,
                    bias,
                )
                grad_bias.add_(chunk_grad_bias)
            else:
                (chunk_grad_input, chunk_grad_student_weight, chunk_grad_teacher_weight), (
                    chunk_loss,
                    (chunk_distill_loss, chunk_student_logits),
                ) = torch.func.grad_and_value(
                    loss_func_to_call, argnums=(0, 1, 2), has_aux=True
                )(
                    student_chunk, teacher_chunk, student_weight, teacher_weight, target_chunk, None
                )
            grad_student_weight.add_(chunk_grad_student_weight)
            grad_teacher_weight.add_(chunk_grad_teacher_weight)
            loss_acc.add_(chunk_loss)
            return chunk_grad_input

        if compiled:
            accumulate_chunk = torch.compile(accumulate_chunk)

        chunks = max(1, student_input.shape[0] // CHUNK_SIZE)
        student_chunks = torch.chunk(student_input, chunks=chunks, dim=0)
        teacher_chunks = torch.chunk(teacher_input, chunks=chunks, dim=0)
        target_chunks = torch.chunk(target, chunks=chunks, dim=0)

        for student_chunk, teacher_chunk, target_chunk in zip(
            student_chunks, teacher_chunks, target_chunks
        ):
            grad_input = accumulate_chunk(student_chunk, teacher_chunk, target_chunk)
            grad_inputs.append(grad_input)

        ctx.save_for_backward(
            torch.cat(grad_inputs, dim=0),
            grad_student_weight,
            grad_teacher_weight,
            grad_bias,
        )
        return loss_acc

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_student_weight, grad_teacher_weight, grad_bias = ctx.saved_tensors
        if torch.ne(grad_output, torch.tensor(1.0, device=grad_output.device)):
            grad_input = grad_input * grad_output
            grad_student_weight = grad_student_weight * grad_output
            grad_teacher_weight = grad_teacher_weight * grad_output
            grad_bias = grad_bias * grad_output if grad_bias is not None else None
        return grad_input, grad_student_weight, grad_teacher_weight, None, grad_bias, None, None, None

    @staticmethod
    def _compute_loss(
        student_chunk,
        teacher_chunk,
        student_weight,
        teacher_weight,
        target_chunk,
        bias,
        distill_loss_fn=None,
        full_target=None,
        ignore_index=-100,
        beta=1.0,
        temperature=1.0,
        compute_ce_loss=True,
        **loss_kwargs,
    ):
        """
        Compute the total loss for a chunk of input and target, while using a distillation loss function.
        """
        student_logits, teacher_logits, ce_loss = LigerFusedLinearDistillationBase.chunk_forward(
            student_chunk,
            teacher_chunk,
            student_weight,
            teacher_weight,
            target_chunk,
            bias=bias,
            ignore_index=ignore_index,
            compute_ce_loss=compute_ce_loss,
        )

        if compute_ce_loss:
            ce_loss = ce_loss / (full_target != ignore_index).sum()
        else:
            ce_loss = 0.0

        distill_loss = distill_loss_fn(
            student_logits, teacher_logits, temperature=temperature, **loss_kwargs
        )
        distill_loss = distill_loss / student_chunk.shape[0]  # Normalize by chunk size

        total_loss = beta * ce_loss + (1 - beta) * distill_loss

        return total_loss, (distill_loss, student_logits)
