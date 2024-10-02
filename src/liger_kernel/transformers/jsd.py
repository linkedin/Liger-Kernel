import torch.nn as nn

from liger_kernel.ops.jsd import LigerJSDFunction


class LigerJSD(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        assert (
            beta > 0 and beta < 1
        ), f"beta must be greater than 0 and less than 1. Got: {beta}"
        self.beta = beta

    def forward(self, log_q, log_p):
        return LigerJSDFunction.apply(log_q, log_p, self.beta)
