from abc import abstractmethod
from functools import partial

import torch
from torch.nn import functional as F


class LigerFusedLinearDistillationBase(torch.autograd.Function):

    @abstractmethod
    def distillation_loss_fn(student_logps, teacher_logps):
        """
        Compute preference loss.
        Args:
            student_logps (torch.Tensor): Avg log probabilities of student inputs. Shape: (batch_size, hidden_size,).
            teacher_logps (torch.Tensor): Avg log probabilities of teacher inputs. Shape: (batch_size, hidden_size,).
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
        ## Student
        student_logits_chunk = student_input_chunk @ student_weight.t()
        if student_bias is not None:
            student_logits_chunk = student_logits_chunk + student_bias
        student_log_probs_chunk = F.log_softmax(student_logits_chunk.float(), dim=-1)

        ce_loss = 0.0
        if compute_ce_loss:
            ce_loss = F.cross_entropy(
                student_log_probs_chunk.view(-1, student_log_probs_chunk.shape[-1]),
                target_chunk.view(-1),
                reduction="sum",
                ignore_index=ignore_index,
            )

        loss_mask = target_chunk != ignore_index
        label_chunk = torch.where(loss_mask, target_chunk, 0)

        student_per_token_logps = student_log_probs_chunk.gather(-1, label_chunk.unsqueeze(-1)).squeeze(
            -1
        )
        student_average_log_prob = (student_per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)

        student_logps = student_average_log_prob

        ## Teacher
        teacher_logits_chunk = teacher_input_chunk @ teacher_weight.t()
        if teacher_bias is not None:
            teacher_logits_chunk = teacher_logits_chunk + teacher_bias
        teacher_log_probs_chunk = F.log_softmax(teacher_logits_chunk.float(), dim=-1)

        # ce_loss = 0.0
        # if compute_ce_loss:
        #     ce_loss = F.cross_entropy(
        #         teacher_log_probs_chunk.view(-1, teacher_log_probs_chunk.shape[-1]),
        #         target_chunk.view(-1),
        #         reduction="sum",
        #         ignore_index=ignore_index,
        #     )

        # loss_mask = target_chunk != ignore_index
        # label_chunk = torch.where(loss_mask, target_chunk, 0)

        teacher_per_token_logps = teacher_log_probs_chunk.gather(-1, label_chunk.unsqueeze(-1)).squeeze(
            -1
        )
        teacher_average_log_prob = (teacher_per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)

        teacher_logps = teacher_average_log_prob

        return student_logps, teacher_logps, ce_loss

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
        beta=0.5,
        compute_ce_loss=True,
        compiled=True,
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

        grad_weight = torch.zeros_like(student_weight)
        grad_inputs = []
        grad_bias = torch.zeros_like(bias) if bias is not None else None
        loss_acc = torch.zeros((), device=student_input.device)

        loss_func_to_call = partial(
            LigerFusedLinearDistillationBase._compute_loss,
            distillation_loss_fn=loss_fn,
            ignore_index=ignore_index,
            beta=beta,
            compute_ce_loss=compute_ce_loss,
            full_target=target,
            **loss_kwargs,
        )

        def accumulate_chunk(student_input_chunk, teacher_input_chunk, target_chunk):
            if bias is not None:
                (chunk_grad_input, chunk_grad_weight, chunk_grad_bias), (
                    chunk_loss,
                    (chunk_distillation_loss, chunk_ce_loss, chunk_student_logps, chunk_teacher_logps),
                ) = torch.func.grad_and_value(
                    loss_func_to_call, argnums=(0, 1, 5), has_aux=True
                )(
                    student_input_chunk, student_weight, teacher_input_chunk, teacher_weight, target_chunk, bias
                )
                grad_bias.add_(chunk_grad_bias)
            else:
                (chunk_grad_input, chunk_grad_weight), (
                    chunk_loss,
                    (chunk_distillation_loss, chunk_ce_loss, chunk_student_logps, chunk_teacher_logps),
                ) = torch.func.grad_and_value(
                    loss_func_to_call, argnums=(0, 1), has_aux=True
                )(
                    student_input_chunk, student_weight, teacher_input_chunk, teacher_weight, target_chunk
                )
            grad_weight.add_(chunk_grad_weight)
            loss_acc.add_(chunk_loss)
            return chunk_grad_input

        if compiled:
            accumulate_chunk = torch.compile(accumulate_chunk)

        # len_chosen = target.shape[0] // 2
        student_chunks = max(1, student_input.shape[0] // (2 * CHUNK_SIZE))
        teacher_chunks = max(1, teacher_input.shape[0] // (2 * CHUNK_SIZE))
        target_chunks = max(1, target.shape[0] // (2 * CHUNK_SIZE))


        _student_input_chunks = torch.chunk(student_input, chunks=student_chunks, dim=0)
        _teacher_input_chunks = torch.chunk(teacher_input, chunks=teacher_chunks, dim=0)
        _target_chunks = torch.chunk(target, chunks=target_chunks, dim=0)


        for (
            student_input_chunk,
            teacher_input_chunk,
            target_chunk
        ) in zip(
            _student_input_chunks,
            _teacher_input_chunks,
            _target_chunks
        ):
            # input_chunk = torch.cat([chosen_input_chunk, rejected_input_chunk], dim=0)
            # target_chunk = torch.cat(
            #     [chosen_target_chunk, rejected_target_chunk], dim=0
            # )

            grad_input = accumulate_chunk(student_input_chunk, teacher_input_chunk, target_chunk)

            grad_inputs.append(grad_input)

        # combine grad_chosen_inputs and grad_rejected_inputs
        # grad_inputs = grad_chosen_inputs + grad_rejected_inputs

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
        bias=None,
        distillation_loss_fn=None,
        full_target=None,
        ignore_index=-100,
        beta=0.5,
        compute_ce_loss=True,
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
        student_logps, teacher_logps, ce_loss = (
            LigerFusedLinearDistillationBase.chunk_forward(
                student_input_chunk,
                student_weight,
                teacher_input_chunk,
                teacher_weight,
                target_chunk,
                student_bias=bias, # TODO: 
                teacher_bias=bias,
                ignore_index=ignore_index,
                compute_ce_loss=compute_ce_loss,
            )
        )
        ce_loss = ce_loss / (full_target != ignore_index).sum()
        

        distillation_loss = distillation_loss_fn(
            student_logps, teacher_logps
        )
        distillation_loss = distillation_loss / (full_target.shape[0])

        loss = beta * ce_loss + (1-beta) * distillation_loss
        return loss, (distillation_loss, ce_loss, student_logps, teacher_logps)
