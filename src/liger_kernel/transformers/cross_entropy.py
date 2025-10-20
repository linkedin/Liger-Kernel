from typing import Optional

import torch

from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.transformers.functional import CrossEntropyOutput


class LigerCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        weight: Optional[torch.FloatTensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        return_token_accuracy: bool = False,
    ):
        super().__init__()
        assert (label_smoothing >= 0) and (label_smoothing <= 1), (
            f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}"
        )
        assert reduction in {
            "mean",
            "sum",
            "none",
        }, f"reduction must be one of 'mean', 'sum', or 'none'. Got: {reduction}"
        assert softcap is None or softcap > 0, f"softcap must greater than 0.0 or None. Got: {softcap}"
        self.weight = weight
        self.ignore_index = ignore_index
        self.lse_square_scale = lse_square_scale
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.softcap = softcap
        self.return_z_loss = return_z_loss
        self.return_token_accuracy = return_token_accuracy

    def forward(self, _input: torch.Tensor, target: torch.Tensor):
        loss, z_loss, token_accuracy = LigerCrossEntropyFunction.apply(
            _input,
            target,
            self.weight,
            self.ignore_index,
            self.lse_square_scale,
            self.label_smoothing,
            self.reduction,
            self.softcap,
            self.return_z_loss,
            self.return_token_accuracy,
        )
        if not self.return_z_loss and not self.return_token_accuracy:
            return loss

        return CrossEntropyOutput(
            loss=loss,
            z_loss=z_loss if self.return_z_loss else None,
            token_accuracy=token_accuracy if self.return_token_accuracy else None,
        )
