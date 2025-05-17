import torch
import torch.nn as nn

from liger_kernel.ops.sparsemax import LigerSparsemaxFunction


class LigerSparsemax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LigerSparsemaxFunction.apply(x, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
