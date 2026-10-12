from typing import Optional

import torch

from liger_kernel.ops import liger_cce


class LigerCCELoss(torch.nn.Module):
    """Cut Cross Entropy loss for a linear language-model head."""

    def __init__(
        self,
        ignore_index: int = -100,
        logit_scale: float = 1.0,
        softcap: Optional[float] = None,
        reduction: str = "mean",
        return_metrics: bool = False,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.logit_scale = logit_scale
        self.softcap = softcap
        self.reduction = reduction
        self.return_metrics = return_metrics

    def forward(self, lin_weight, _input, target, bias=None):
        return liger_cce(
            _input,
            lin_weight,
            target,
            bias=bias,
            ignore_index=self.ignore_index,
            logit_scale=self.logit_scale,
            softcap=self.softcap,
            reduction=self.reduction,
            return_metrics=self.return_metrics,
        )

    def extra_repr(self) -> str:
        return (
            f"ignore_index={self.ignore_index}, logit_scale={self.logit_scale}, "
            f"softcap={self.softcap}, reduction={self.reduction!r}, return_metrics={self.return_metrics}"
        )
