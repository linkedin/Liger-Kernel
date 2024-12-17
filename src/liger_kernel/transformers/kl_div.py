import torch.nn as nn

from liger_kernel.ops.kl_div import LigerKLDivLossFunction


class LigerKLDIVLoss(nn.KLDivLoss):
    def __init__(self, eps: float = 1e-10, *args, **kwargs):
        super(LigerKLDIVLoss, self).__init__(*args, **kwargs)
        self.eps = eps

    def forward(self, y_pred, y_true):
        return LigerKLDivLossFunction.apply(y_pred, y_true, self.reduction, self.log_target, self.eps)
