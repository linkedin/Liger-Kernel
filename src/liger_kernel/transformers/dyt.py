import torch
import torch.nn as nn

from liger_kernel.ops import LigerDyTFunction


class LigerDyT(nn.Module):
    def __init__(self, hidden_size, beta=True, init_alpha=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.init_alpha = init_alpha
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = None
        if beta:
            self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        return LigerDyTFunction.apply(x, self.alpha, self.gamma, self.beta)

    def extra_repr(self):
        return f"{self.hidden_size}, init_alpha={self.init_alpha}, beta={self.beta}"
