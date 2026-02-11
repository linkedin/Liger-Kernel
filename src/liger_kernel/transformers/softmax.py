import torch
import torch.nn as nn

from liger_kernel.ops import LigerSoftmaxFunction


class LigerSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return LigerSoftmaxFunction.apply(x)
