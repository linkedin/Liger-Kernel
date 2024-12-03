import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_distillation import LigerFusedLinearDistillationBase


class LigerFusedLinearJSDFunction(LigerFusedLinearDistillationBase):
    @staticmethod
    def distillation_loss_fn(student_logits, teacher_logits, temperature):
        """
        Compute JSD loss (Jensen-Shannon Divergence Loss).
        Args:
            student_logits (torch.Tensor): Raw logits of student tokens. Shape: (batch_size,).
            teacher_logits (torch.Tensor): Raw logits of teacher tokens. Shape: (batch_size,).
            temperature (float): Temperature for softening probability distributions
        Returns:
            torch.Tensor: Jensen-Shannon Divergence loss
        """
        # Scale logits by temperature
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature
        # Convert to probabilities
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        log_mean_probs = torch.log((student_probs + teacher_probs) / 2)

        student_kl = F.kl_div(
            log_mean_probs,
            torch.log(student_probs),
            reduction="batchmean",
            log_target=True,
        )
        teacher_kl = F.kl_div(
            log_mean_probs,
            torch.log(teacher_probs),
            reduction="batchmean",
            log_target=True,
        )

        # JSD is the average of the KL divergences
        jsd_loss = (student_kl + teacher_kl) / 2
        return jsd_loss

    @staticmethod
    def forward(
        ctx,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        true_labels: torch.LongTensor,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
        compiled: bool = True,
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
            ignore_index (int): Index to ignore in loss computation
            temperature (float): Temperature for softening distributions
            compiled (bool): Whether to use torch compile
        Returns:
            torch.Tensor: Computed loss
        """
        return LigerFusedLinearDistillationBase.forward(
            ctx=ctx,
            student_input=student_input,
            student_weight=student_weight,
            teacher_input=teacher_input,
            teacher_weight=teacher_weight,
            target=true_labels,
            loss_fn=LigerFusedLinearJSDFunction.distillation_loss_fn,
            weight_hard_loss=weight_hard_loss,
            weight_soft_loss=weight_soft_loss,
            ignore_index=ignore_index,
            temperature=temperature,
            compiled=compiled,
        )

    @staticmethod
    def backward(ctx, grad_output):
        grads = LigerFusedLinearDistillationBase.backward(ctx, grad_output)[:4]

        return (*grads, None, None, None, None, None, None)


class LigerFusedLinearJSDLoss(torch.nn.Module):
    """
    Fused linear layer with JSD distillation loss.
    """

    def __init__(
        self,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
        compiled: bool = True,
    ):
        """
        Args:
            weight_hard_loss (float): Weight for hard loss.
            weight_soft_loss (float): Weight for soft loss.
            ignore_index (int): Index to ignore in the loss
            temperature (float): Temperature for softening distributions
            compiled (bool): Whether to use torch compile
        """
        super().__init__()
        assert temperature != 0, "Temperature cannot be 0."
        self.weight_hard_loss = weight_hard_loss
        self.weight_soft_loss = weight_soft_loss
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.compiled = compiled

    def forward(
        self,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        true_labels: torch.LongTensor,
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
            self.weight_hard_loss,
            self.weight_soft_loss,
            self.ignore_index,
            self.temperature,
            self.compiled,
        )
