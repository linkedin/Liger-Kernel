import torch.nn as nn

from liger_kernel.ops import LigerReLUSquaredFunction


class LigerReLUSquared(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return LigerReLUSquaredFunction.apply(x)[0]
