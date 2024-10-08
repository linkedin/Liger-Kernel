import torch.nn as nn

from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction


class LigerFusedLinearJSD(nn.Module):
    def __init__(self, jsd_beta=0.5, temperature=1.0):
        super().__init__()
        self.jsd_beta = jsd_beta
        self.temperature = temperature

    def forward(
        self,
        student_input,
        student_weight,
        teacher_input,
        teacher_weight,
    ):
        return LigerFusedLinearJSDFunction.apply(
            student_input,
            student_weight,
            teacher_input,
            teacher_weight,
            self.jsd_beta,
            self.temperature,
        )
