import torch.nn as nn

from liger_kernel.ops.tvd import LigerTVDLossFunction


class LigerTVDLoss(nn.Module):
    def __init__(self, reduction="batchmean"):
        super(LigerTVDLoss, self).__init__()
        self.reduction = reduction

    def forward(self, p, q):
        return LigerTVDLossFunction.apply(p, q, self.reduction)
