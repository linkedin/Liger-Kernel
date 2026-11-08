import math

from typing import Tuple
from typing import Union

import torch

from liger_kernel.chunked_loss.fused_linear_distillation import LigerFusedLinearDistillationBase


def _log_normalizer(logits, max_logits):
    # log sum exp(logits - max), keeping the mass below the max when it is far under 1 (arXiv 2610.31642, Eq. 5)
    shifted = logits - max_logits
    at_max = shifted == 0
    n = at_max.sum(-1, keepdim=True)
    others = torch.where(at_max, 0.0, shifted.exp()).sum(-1, keepdim=True)
    return torch.where(n == 1, torch.log1p(others), torch.log(n + others))


def _logit_gap(zs, zt, s_max, t_max):
    # (zs - s_max) - (zt - t_max), carrying the exact rounding error of both subtractions so it keeps a small
    # student-teacher difference as well as independent shifts of either side
    delta = zs - zt
    delta_err = torch.where(zs.abs() >= zt.abs(), (zs - delta) - zt, zs - (delta + zt))
    shift = s_max - t_max
    shift_err = torch.where(s_max.abs() >= t_max.abs(), (s_max - shift) - t_max, s_max - (shift + t_max))
    return (delta - shift) + (delta_err - shift_err)


def _log_mixture_ratio(log_ratio, w):
    # log(a / m) for m = w * b + (1 - w) * a, from log_ratio = log(a / b)
    return torch.where(
        log_ratio.abs() <= 0.5,
        -torch.log1p(w * torch.expm1(-log_ratio.clamp(-0.5, 0.5))),
        -torch.logaddexp(torch.full_like(log_ratio, math.log1p(-w)), math.log(w) - log_ratio),
    )


def _jsd_and_grad(student_logits, teacher_logits, beta):
    """
    Per-row generalized JSD and its gradient w.r.t. the student logits. Both come from the logit gap log(q / p)
    instead of rounded probabilities (arXiv 2610.31642, Eq. 9), so the gradient stays accurate as the student
    approaches the teacher and is exactly 0 when they match.
    """
    zs, zt = student_logits.float(), teacher_logits.float()
    s_max = zs.max(-1, keepdim=True).values
    t_max = zt.max(-1, keepdim=True).values
    s_log_z, t_log_z = _log_normalizer(zs, s_max), _log_normalizer(zt, t_max)
    q = (zs - s_max - s_log_z).exp()
    p = (zt - t_max - t_log_z).exp()

    # log(q / p) = gap - (mean_gap + forward_kl). Near the teacher, forward_kl = log1p(sum p expm1(gap - mean_gap))
    # keeps its small part; elsewhere the two normalizers are far enough apart to subtract.
    gap = _logit_gap(zs, zt, s_max, t_max)
    mean_gap = (p * gap).sum(-1, keepdim=True)
    d = gap - mean_gap
    near = ((d.abs() <= 0.5) | (p == 0)).all(-1, keepdim=True)
    forward_kl = torch.where(
        near, torch.log1p((p * torch.expm1(d.clamp(max=0.5))).sum(-1, keepdim=True)), s_log_z - t_log_z - mean_gap
    )
    log_q_p = d - forward_kl

    if beta == 0:
        # As in the paper's soft_target_cross_entropy (Eq. 9): u = q - p through expm1 of log(q / p), plain q - p where
        # |log(q / p)| >= 1 so p rounding to 0 can't give 0 * inf, then projected so the gradient sums to 0.
        u = torch.where(log_q_p.abs() < 1, p * torch.expm1(log_q_p), q - p)
        return forward_kl.squeeze(-1), u - q * u.sum(-1, keepdim=True)
    if beta == 1:
        h = log_q_p
        loss = (q * h).sum(-1)
    else:
        h = _log_mixture_ratio(log_q_p, beta)
        loss = (1 - beta) * (q * h).sum(-1) + beta * (p * _log_mixture_ratio(-log_q_p, 1 - beta)).sum(-1)
    # The gradient is q * (h - E_q[h]); shifting h by its value at the student's argmax keeps E_q[h] from cancelling at
    # a dominant token (as in unslothai/unsloth-zoo#2079).
    h = h - h.gather(-1, zs.argmax(-1, keepdim=True))
    grad = q * (h - (q * h).sum(-1, keepdim=True))
    return loss, grad if beta == 1 else (1 - beta) * grad


class LigerFusedLinearJSDFunction(LigerFusedLinearDistillationBase):
    @staticmethod
    def distillation_loss_fn(student_logits, teacher_logits, beta=0.5, target=None, ignore_index=-100):
        """
        Compute JSD loss (Jensen-Shannon Divergence Loss).
        Args:
            student_logits (torch.Tensor): Logits of student tokens. Shape: (batch_size * seq_len,).
            teacher_logits (torch.Tensor): Logits of teacher tokens. Shape: (batch_size * seq_len,).
            beta (float): Coefficient beta of generalized JSD in the interval [0, 1]. Default: `0.5`.
            target (torch.Tensor): Target labels for masking. Shape: (chunk_size,).
            ignore_index (int): Index to ignore in loss computation.
        Returns:
            torch.Tensor: Jensen-Shannon Divergence loss
        Note:
            - Uses reduction="none" to preserve per-token losses for masking
            - KL divergence requires summing over vocab dimension (not mean)
            - Masking excludes padding/prompt tokens from loss computation
        """
        loss, grad = _jsd_and_grad(student_logits.detach(), teacher_logits, beta)
        # The surrogate term is 0 in value and has gradient `grad`, as in the paper's reference implementation.
        jsd_loss = loss + (grad * (student_logits - student_logits.detach())).sum(-1)

        # Apply ignore_index mask
        if target is not None:
            mask = target != ignore_index
            jsd_loss = jsd_loss.masked_fill(~mask, 0.0)

        return jsd_loss.sum()

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
    ):
        """
        Fused linear layer with JSD distillation loss.
        Args:
            student_input (torch.Tensor): Student input tensor. Shape: (batch_size * seq_len, hidden_size_student)
            student_weight (torch.Tensor): Student weight tensor. Shape: (vocab_size, hidden_size_student)
            teacher_input (torch.Tensor): Teacher input tensor. Shape: (batch_size * seq_len, hidden_size_teacher)
            teacher_weight (torch.Tensor): Teacher weight tensor. Shape: (vocab_size, hidden_size_teacher)
            true_labels (torch.LongTensor): Target tensor. Shape: (batch_size * seq_len,)
            weight_hard_loss (float): Weight for hard loss.
            weight_soft_loss (float): Weight for soft loss.
            beta (float): Coefficient beta of generalized JSD in the interval [0, 1]. Default: `0.5`.
            ignore_index (int): Index to ignore in loss computation
            temperature (float): Temperature for softening/sharpening distributions
            compiled (bool): Whether to use torch compile
            chunk_size (int): Size of chunks for processing.
            return_soft_hard_loss (bool): Whether to return soft and hard losses separately. Default: False.
        Returns:
            torch.Tensor: Computed loss, or tuple (loss, soft_loss, hard_loss) if return_soft_hard_loss=True
        """
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


class LigerFusedLinearJSDLoss(torch.nn.Module):
    """
    Fused linear layer with JSD distillation loss.
    """

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
        """
        Args:
            weight_hard_loss (float): Weight for hard loss.
            weight_soft_loss (float): Weight for soft loss.
            ignore_index (int): Index to ignore in the loss
            temperature (float): Temperature for softening distributions
            compiled (bool): Whether to use torch compile
            beta (float): Coefficient beta of generalized JSD in the interval [0, 1]. Default: `0.5`.
            chunk_size (int): Size of chunks for processing.
            return_soft_hard_loss (bool): Whether to return soft and hard losses separately. Default: False.
        """
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
        """
        Compute the JSD distillation loss.

        Args:
            student_input (torch.Tensor): Student input tensor
            student_weight (torch.Tensor): Student weight tensor
            teacher_input (torch.Tensor): Teacher input tensor
            teacher_weight (torch.Tensor): Teacher weight tensor
            true_labels (torch.LongTensor): Target labels tensor

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                If return_soft_hard_loss is False: Computed combined loss
                If return_soft_hard_loss is True: Tuple of (combined_loss, soft_loss, hard_loss)
        """
        return LigerFusedLinearJSDFunction.apply(
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
