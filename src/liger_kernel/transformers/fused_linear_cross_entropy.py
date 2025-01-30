from typing import Optional

import torch

from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction


class LigerFusedLinearCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        ce_weight: Optional[torch.FloatTensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
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
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.lse_square_scale = lse_square_scale
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.softcap = softcap
        self.return_z_loss = return_z_loss

    def forward(self, lin_weight, _input, target, bias=None):
        loss, z_loss = LigerFusedLinearCrossEntropyFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            self.ce_weight,
            self.ignore_index,
            self.lse_square_scale,
            self.label_smoothing,
            self.reduction,
            self.softcap,
            self.return_z_loss,
        )
        if not self.return_z_loss:
            return loss
        return loss, z_loss
