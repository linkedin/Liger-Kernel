import torch.nn as nn

from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction


class LigerCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        return_z_loss=False,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.lse_square_scale = lse_square_scale
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.return_z_loss = return_z_loss

        assert (self.label_smoothing >= 0) and (
            self.label_smoothing <= 1
        ), f"label_smoothing must be between 0.0 and 1.0. Got: {self.label_smoothing}"
        assert self.reduction in {
            "mean",
            "sum",
            "none",
        }, f"reduction must be one of 'mean', 'sum', or 'none'. Got: {self.reduction}"

    def forward(self, _input, target):
        loss, z_loss = LigerCrossEntropyFunction.apply(
            _input,
            target,
            self.ignore_index,
            self.lse_square_scale,
            self.label_smoothing,
            self.reduction,
            self.return_z_loss,
        )
        if not self.return_z_loss:
            return loss
        return loss, z_loss
