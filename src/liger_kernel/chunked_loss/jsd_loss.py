import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_distillation import LigerFusedLinearDistillationBase


class LigerFusedLinearJSDFunction(LigerFusedLinearDistillationBase):
    @staticmethod
    def distillation_loss_fn(student_logits, teacher_logits, beta=0.5):
        """
        Compute JSD loss (Jensen-Shannon Divergence Loss).
        Args:
            student_logits (torch.Tensor): Logits of student tokens. Shape: (batch_size * seq_len,).
            teacher_logits (torch.Tensor): Logits of teacher tokens. Shape: (batch_size * seq_len,).
            beta (float): Coefficient beta of generalized JSD in the interval [0, 1]. Default: `0.5`.
        Returns:
            torch.Tensor: Jensen-Shannon Divergence loss
        """
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd_loss = F.kl_div(student_log_probs, teacher_log_probs, reduction="sum", log_target=True)
        elif beta == 1:
            jsd_loss = F.kl_div(teacher_log_probs, student_log_probs, reduction="sum", log_target=True)
        else:
            # Compute probabilities (only required for mean calculation)
            mean_probs = (1 - beta) * student_log_probs.exp() + beta * teacher_log_probs.exp()
            log_mean_probs = mean_probs.log()

            student_kl = F.kl_div(log_mean_probs, student_log_probs, reduction="sum", log_target=True)
            teacher_kl = F.kl_div(log_mean_probs, teacher_log_probs, reduction="sum", log_target=True)

            # JSD is the weighted average of the KL divergences
            jsd_loss = beta * teacher_kl + (1 - beta) * student_kl
        return jsd_loss

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
        Returns:
            torch.Tensor: Computed loss
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
        )

    @staticmethod
    def backward(ctx, grad_output):
        grads = LigerFusedLinearDistillationBase.backward(ctx, grad_output)[:6]

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

    def forward(
        self,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        true_labels: torch.LongTensor,
        student_bias: torch.Tensor,
        teacher_bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the JSD distillation loss.

        Args:
            student_input (torch.Tensor): Student input tensor
            student_weight (torch.Tensor): Student weight tensor
            teacher_input (torch.Tensor): Teacher input tensor
            teacher_weight (torch.Tensor): Teacher weight tensor
            true_labels (torch.LongTensor): Target labels tensor

        Returns:
            torch.Tensor: Computed loss
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
        )
