import torch.nn as nn

from liger_kernel.ops.tvd import LigerTVDLossFunction


class LigerTVDLoss(nn.Module):
    def __init__(self, reduction="batchmean", ignore_index: int = -100):
        super(LigerTVDLoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, p, q, shift_labels=None):
        return LigerTVDLossFunction.apply(p, q, shift_labels, self.reduction, self.ignore_index)
