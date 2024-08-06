import torch
import torch.nn as nn

from liger_kernel.ops.rms_norm import LigerRMSNormFunction


class LigerRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return LigerRMSNormFunction.apply(
            hidden_states, self.weight, self.variance_epsilon
        )
