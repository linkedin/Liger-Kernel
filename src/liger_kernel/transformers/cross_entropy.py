from typing import Optional

import torch

from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction


class LigerCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
    ):
        super().__init__()
        assert (label_smoothing >= 0) and (
            self.label_smoothing <= 1
        ), f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}"
        assert reduction in {
            "mean",
            "sum",
            "none",
        }, f"reduction must be one of 'mean', 'sum', or 'none'. Got: {reduction}"
        assert (
            softcap > 0 or softcap is None
        ), f"softcap must greater than 0.0 or None. Got: {softcap}"
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.softcap = softcap

    def forward(self, _input: torch.Tensor, target: torch.Tensor):
        return LigerCrossEntropyFunction.apply(
            _input,
            target,
            self.ignore_index,
            self.label_smoothing,
            self.reduction,
            self.softcap,
        )
