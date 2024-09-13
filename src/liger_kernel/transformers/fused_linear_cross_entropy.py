import torch.nn as nn

from liger_kernel.ops.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyFunction,
)


class LigerFusedLinearCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index=-100,
        label_smoothing=0.0,
        reduction="mean",
        lse_square_scale=0.0,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.lse_square_scale = lse_square_scale
        assert (self.label_smoothing >= 0) and (
            self.label_smoothing <= 1
        ), f"label_smoothing must be between 0.0 and 1.0. Got: {self.label_smoothing}"

    def forward(self, lin_weight, _input, target, bias=None):
        return LigerFusedLinearCrossEntropyFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            self.ignore_index,
            self.lse_square_scale,
            self.label_smoothing,
            self.reduction,
        )
