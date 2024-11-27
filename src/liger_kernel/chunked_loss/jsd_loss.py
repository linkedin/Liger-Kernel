import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_distillation import (
    LigerFusedLinearDistillationBase,
)


class LigerFusedLinearJSDFunction(LigerFusedLinearDistillationBase):
    @staticmethod
    def distillation_loss_fn(student_logps, teacher_logps, temperature):
        """
        Compute Jensen-Shannon Divergence loss between student and teacher distributions.
        Args:
            student_logps (torch.Tensor): Log probabilities from student model (Raw logits after log_softmax)
            teacher_logps (torch.Tensor): Log probabilities from teacher model (Raw logits after log_softmax)
            temperature (float): Temperature for softening probability distributions
        Returns:
            torch.Tensor: Jensen-Shannon Divergence loss
        """
        # TODO: should incorporate with (high) temperature scaling on raw logits

        # For instance,
        # Scale logits by temperature
        # student_logits = student_logits / temperature
        # teacher_logits = teacher_logits / temperature
        # Convert to probabilities
        # student_probs = F.softmax(student_logits, dim=-1)
        # teacher_probs = F.softmax(teacher_logits, dim=-1)

        log_mean_probs = torch.log((torch.exp(student_logps) + torch.exp(teacher_logps)) / 2)

        student_kl = F.kl_div(
            log_mean_probs, 
            student_logps, 
            reduction="batchmean", 
            log_target=True
        )
        
        teacher_kl = F.kl_div(
            log_mean_probs, 
            teacher_logps, 
            reduction="batchmean", 
            log_target=True
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
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
        compiled: bool = True,
    ):
        """
        Fused linear layer with JSD distillation loss.
        Args:
            student_input (torch.Tensor): Student input tensor. Shape: (BT, H_s)
            student_weight (torch.Tensor): Student weight tensor. Shape: (V, H_s)
            teacher_input (torch.Tensor): Teacher input tensor. Shape: (BT, H_t)
            teacher_weight (torch.Tensor): Teacher weight tensor. Shape: (V, H_t)
            true_labels (torch.LongTensor): Target tensor. Shape: (BT,)
            beta (float): Weight for distillation loss
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
            beta=beta,
            ignore_index=ignore_index,
            temperature=temperature,
            compiled=compiled,
        )

    @staticmethod
    def backward(ctx, grad_output):
        grads = LigerFusedLinearDistillationBase.backward(ctx, grad_output)[:4]

        return (*grads, None, None, None, None, None)


class LigerFusedLinearJSDLoss(torch.nn.Module):
    """
    Fused linear layer with JSD distillation loss.
    """

    def __init__(
        self,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
        compiled: bool = False,
    ):
        """
        Args:
            beta (float): Weight for distillation loss
            ignore_index (int): Index to ignore in the loss
            temperature (float): Temperature for softening distributions
            compiled (bool): Whether to use torch compile
        """
        super().__init__()
        assert temperature != 0, "Temperature cannot be 0."
        self.beta = beta
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
            self.beta,
            self.ignore_index,
            self.temperature,
            self.compiled,
        )
