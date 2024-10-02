import torch.nn as nn

from liger_kernel.ops.jsd import LigerJSDFunction


class LigerJSD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p, q):
        return LigerJSDFunction.apply(p, q)
