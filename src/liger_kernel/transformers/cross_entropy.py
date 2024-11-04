import torch

from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction


class LigerCrossEntropyLoss(torch.nn.Module):
    def __init__(self, ignore_index=-100, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        assert (label_smoothing >= 0) and (
            label_smoothing <= 1
        ), f"label_smoothing must be between 0.0 and 1.0. Got: {self.label_smoothing}"
        assert reduction in {
            "mean",
            "sum",
            "none",
        }, f"reduction must be one of 'mean', 'sum', or 'none'. Got: {self.reduction}"
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, _input, target, inplace):
        loss, _ = LigerCrossEntropyFunction.apply(
            _input,
            target,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
            inplace=inplace,
        )
        return loss
