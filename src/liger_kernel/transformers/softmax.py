import torch
import torch.nn as nn

from liger_kernel.ops import LigerSoftmaxFunction


class LigerSoftmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return LigerSoftmaxFunction.apply(x, self.dim)
