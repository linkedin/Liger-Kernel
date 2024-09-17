from torch.nn import CrossEntropyLoss

from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction


class LigerCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super(LigerCrossEntropyLoss, self).__init__(*args, **kwargs)
        assert (self.label_smoothing >= 0) and (
            self.label_smoothing <= 1
        ), f"label_smoothing must be between 0.0 and 1.0. Got: {self.label_smoothing}"
        assert self.reduction in {
            "mean",
            "sum",
            "none",
        }, f"reduction must be one of 'mean', 'sum', or 'none'. Got: {self.reduction}"

    def forward(self, _input, target):
        return LigerCrossEntropyFunction.apply(
            _input, target, self.ignore_index, self.label_smoothing, self.reduction
        )
