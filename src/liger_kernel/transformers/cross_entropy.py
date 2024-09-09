import torch.nn as nn

from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction


class LigerCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index=-100,
        label_smoothing=0.0,
        z_loss_scale=0.0,
        return_z_loss=False,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.z_loss_scale = z_loss_scale
        self.return_z_loss = return_z_loss
        assert (self.label_smoothing >= 0) and (
            self.label_smoothing <= 1
        ), f"label_smoothing must be between 0.0 and 1.0. Got: {self.label_smoothing}"

    def forward(self, _input, target):
        loss, z_loss = LigerCrossEntropyFunction.apply(
            _input,
            target,
            self.ignore_index,
            self.label_smoothing,
            self.z_loss_scale,
            self.return_z_loss,
        )
        if not self.return_z_loss:
            return loss
        return loss, z_loss
