import torch
import torch.nn as nn

from liger_kernel.ops.poly_norm import LigerPolyNormFunction


class LigerPolyNorm(nn.Module):
    """
    PolyNorm layer wrapper for Liger kernel.

    PolyNorm formula:
        y = w₀·norm(x³) + w₁·norm(x²) + w₂·norm(x) + b
        where norm(u) = u / sqrt(mean(u²) + ε)

    Reference:
        https://github.com/BryceZhuo/PolyCom/

    Args:
        eps: epsilon for numerical stability (default: 1e-6)
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.3, 0.4, 0.3]))
        self.bias = nn.Parameter(torch.tensor(0.1))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return LigerPolyNormFunction.apply(
            hidden_states,
            self.weight,
            self.bias,
            self.variance_epsilon,
        )

    def extra_repr(self):
        return f"weight_shape={tuple(self.weight.shape)}, eps={self.variance_epsilon}"
