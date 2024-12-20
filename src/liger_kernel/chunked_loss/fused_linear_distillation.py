from abc import abstractmethod
from functools import partial

import torch


class LigerFusedLinearDistillationBase(torch.autograd.Function):

    @abstractmethod
    def distillation_loss_fn(
        student_logits_chunk, teacher_logits_chunk, target_chunk, full_target, **kwargs
    ):
        """
        Compute distillation loss.
        Args:
            student_logits_chunk (torch.Tensor): Chunk of student logits tensor. Shape: (chunk_size, vocab_size).
            teacher_logits_chunk (torch.Tensor): Chunk of teacher logits tensor. Shape: (chunk_size, vocab_size).
            target_chunk (torch.Tensor): Chunk of target tensor. Shape: (chunk_size,).
            full_target (torch.Tensor): Full target tensor. Shape: (chunk_size,).
            kwargs: Additional arguments for the loss function.
        """
        raise NotImplementedError("Distillation loss function must be implemented.")

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
        chunk_size=1024,
        compiled=True,
        **loss_kwargs,
    ):
        """
        Base class for fused linear layer with distillation loss.
        Only need to compute gradients for student model.

        Args:
            student_input (torch.Tensor): Student input tensor. Shape: (batch_size * seq_len, student_hidden_size).
            student_weight (torch.Tensor): Student weight tensor. Shape: (vocab_size, student_hidden_size).
            teacher_input (torch.Tensor): Teacher input tensor. Shape: (batch_size * seq_len, teacher_hidden_size).
            teacher_weight (torch.Tensor): Teacher weight tensor. Shape: (vocab_size, teacher_hidden_size).
            target (torch.Tensor): Target truth label tensor. Shape: (batch_size * seq_len).
            student_bias (torch.Tensor, optional): Student bias tensor. Shape: (vocab_size,).
            teacher_bias (torch.Tensor, optional): Teacher bias tensor. Shape: (vocab_size,).
            loss_fn (callable): Loss function to compute the loss on a chunk of input/target.
            chunk_size (int): Size of a chunk.
            compiled (bool): Whether to use torch compile for chunk accumulation.
            loss_kwargs (dict): Other possible arguments that a loss function might need
        """
        CHUNK_SIZE = chunk_size
        grad_weight = torch.zeros_like(student_weight)
        grad_inputs = []
        grad_bias = torch.zeros_like(student_bias) if student_bias is not None else None
        loss_acc = torch.zeros((), device=student_input.device)

        compute_loss = partial(
            LigerFusedLinearDistillationBase._compute_loss,
            distillation_loss_fn=loss_fn,
            full_target=target,
            **loss_kwargs,
        )

        def fused_fwd_bwd(student_input_chunk, teacher_input_chunk, target_chunk):
            """
            Fused forward and backward pass for a chunk of student input, teacher input and target.
            """
            argnums = (0, 1, 5) if student_bias is not None else (0, 1)
            return torch.func.grad_and_value(
                compute_loss, argnums=argnums, has_aux=True
            )(
                student_input_chunk,
                student_weight,
                teacher_input_chunk,
                teacher_weight,
                target_chunk,
                student_bias,
                teacher_bias,
            )

        def accumulate_chunk(student_input_chunk, teacher_input_chunk, target_chunk):
            (chunk_grad_input, chunk_grad_weight, *chunk_grad_bias), (
                chunk_loss,
                (
                    chunk_student_logits,
                    chunk_teacher_logits,
                ),
            ) = fused_fwd_bwd(student_input_chunk, teacher_input_chunk, target_chunk)

            if student_bias is not None:
                grad_bias.add_(chunk_grad_bias)

            grad_weight.add_(chunk_grad_weight)
            loss_acc.add_(chunk_loss)
            grad_inputs.append(chunk_grad_input)

        if compiled:
            fused_fwd_bwd = torch.compile(fused_fwd_bwd)

        num_chunks = max(1, student_input.shape[0] // CHUNK_SIZE)
        _student_input_chunks = torch.chunk(student_input, chunks=num_chunks, dim=0)
        _teacher_input_chunks = torch.chunk(teacher_input, chunks=num_chunks, dim=0)
        _target_chunks = torch.chunk(target, chunks=num_chunks, dim=0)

        for student_input_chunk, teacher_input_chunk, target_chunk in zip(
            _student_input_chunks, _teacher_input_chunks, _target_chunks
        ):
            accumulate_chunk(student_input_chunk, teacher_input_chunk, target_chunk)

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

        return grad_input, grad_weight, None, grad_bias

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
        **loss_kwargs,
    ):
        """
        Compute the total loss for a chunk of input and target, while using an knowleedge distillation loss function.
        Args:
            student_input_chunk (torch.Tensor): Chunk of input tensor. Shape: (chunk_size, student_hidden_size).
            student_weight (torch.Tensor): Weight tensor. Shape: (vocab_size, student_hidden_size).
            teacher_input_chunk (torch.Tensor): Chunk of input tensor. Shape: (chunk_size, teacher_hidden_size).
            teacher_weight (torch.Tensor): Weight tensor. Shape: (vocab_size, teacher_hidden_size).
            target_chunk (torch.Tensor): Chunk of target tensor. Shape: (chunk_size,).
            student_bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,).
            teacher_bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,).
            distillation_loss_fn (callable): Loss function to compute the loss on a chunk of input/target.
            full_target (torch.Tensor): Full target tensor. Shape: (chunk_size,).
            loss_kwargs (dict): Additional arguments for the loss function.
        """
        # Student
        student_logits_chunk = student_input_chunk @ student_weight.t()
        if student_bias is not None:
            student_logits_chunk += student_bias

        # Teacher
        with torch.no_grad():
            teacher_logits_chunk = teacher_input_chunk @ teacher_weight.t()
            if teacher_bias is not None:
                teacher_logits_chunk += teacher_bias

        loss_chunk = distillation_loss_fn(
            student_logits_chunk,
            teacher_logits_chunk,
            target_chunk,
            full_target,
            **loss_kwargs,
        )

        return loss_chunk, (student_logits_chunk, teacher_logits_chunk)
