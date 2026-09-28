import torch.nn as nn

from liger_kernel.ops import LigerFusedCETVDFunction


class LigerFusedCETVDLoss(nn.Module):
    """Per-token cross-entropy and total variation distance against a teacher.

    Returns both terms unreduced so the caller owns masking, per-token
    weighting, and normalization.
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, student_logits, teacher_logits, target):
        return LigerFusedCETVDFunction.apply(student_logits, teacher_logits, target, self.ignore_index)
