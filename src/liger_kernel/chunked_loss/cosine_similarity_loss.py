from typing import Tuple
from typing import Union

import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_distillation import LigerFusedLinearDistillationBase


class LigerFusedLinearCosineSimilarityFunction(LigerFusedLinearDistillationBase):
    @staticmethod
    def distillation_loss_fn(
        student_logits,
        teacher_logits,
        target=None,
        ignore_index=None,
        beta=1.0,
    ):
        """
        Compute Cosine loss (Cosine Similarity Loss).
        Args:
            student_logits (torch.Tensor): Logits of student tokens. Shape: (batch_size * seq_len,).
            teacher_logits (torch.Tensor): Logits of teacher tokens. Shape: (batch_size * seq_len,).
            beta: Coefficient beta of generalized Cosine Similarity in the interval [0, 1]. Default: `1.0` (float): .
        Returns:
            torch.Tensor: cosine similarity loss
        """
        student_norm = F.normalize(student_logits, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_logits, p=2, dim=-1)

        cosine_sim = F.cosine_similarity(student_norm, teacher_norm, dim=-1)
        loss = beta * (1 - cosine_sim)
        return loss.sum()

    @classmethod
    def forward(
        cls,
        ctx,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        true_labels: torch.LongTensor,
        student_bias: torch.Tensor,
        teacher_bias: torch.Tensor,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
        compiled: bool = True,
        chunk_size: int = 1024,
        return_soft_hard_loss: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return super().forward(
            cls=cls,
            ctx=ctx,
            student_input=student_input,
            student_weight=student_weight,
            teacher_input=teacher_input,
            teacher_weight=teacher_weight,
            target=true_labels,
            student_bias=student_bias,
            teacher_bias=teacher_bias,
            chunk_size=chunk_size,
            weight_hard_loss=weight_hard_loss,
            weight_soft_loss=weight_soft_loss,
            beta=beta,
            ignore_index=ignore_index,
            temperature=temperature,
            compiled=compiled,
            return_soft_hard_loss=return_soft_hard_loss,
        )

    @staticmethod
    def backward(ctx, grad_output, *args):
        grads = LigerFusedLinearDistillationBase.backward(ctx, grad_output, *args)[:6]

        return (
            *grads,
            None,  # teacher_bias
            None,  # weight_hard_loss
            None,  # weight_soft_loss
            None,  # beta
            None,  # ignore_index
            None,  # temperature
            None,  # compiled
            None,  # chunk_size
            None,  # return_soft_hard_loss
        )


class LigerFusedLinearCosineSimilarityLoss(torch.nn.Module):
    def __init__(
        self,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
        compiled: bool = True,
        chunk_size: int = 1024,
        return_soft_hard_loss: bool = False,
    ):
        super().__init__()
        assert temperature != 0, "Temperature cannot be 0."
        self.weight_hard_loss = weight_hard_loss
        self.weight_soft_loss = weight_soft_loss
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.compiled = compiled
        self.beta = beta
        self.chunk_size = chunk_size
        self.return_soft_hard_loss = return_soft_hard_loss

    def forward(
        self,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        true_labels: torch.LongTensor,
        student_bias: torch.Tensor = None,
        teacher_bias: torch.Tensor = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return LigerFusedLinearCosineSimilarityFunction.apply(
            student_input,
            student_weight,
            teacher_input,
            teacher_weight,
            true_labels,
            student_bias,
            teacher_bias,
            self.weight_hard_loss,
            self.weight_soft_loss,
            self.beta,
            self.ignore_index,
            self.temperature,
            self.compiled,
            self.chunk_size,
            self.return_soft_hard_loss,
        )
