import torch
import torch.nn as nn

from liger_kernel.ops.dyt import LigerDyTFunction


class LigerDyT(nn.Module):
    def __init__(self, C, init_alpha):
        super().__init__()
        self.C = C
        self.init_alpha = init_alpha
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(C))
        self.beta = nn.Parameter(torch.zeros(C))

    def forward(self, x):
        return LigerDyTFunction.apply(x, self.alpha, self.gamma, self.beta)

    def extra_repr(self):
        return f"{self.C}, init_alpha={self.init_alpha}"
