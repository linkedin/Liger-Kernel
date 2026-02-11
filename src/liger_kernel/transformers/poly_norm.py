import torch
import torch.nn as nn

from liger_kernel.ops import LigerPolyNormFunction


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
        in_place: whether to in-place modify grad_output in backward to save memory (default: False).
                  Set to True to save memory if grad_output is not needed elsewhere.
    """

    def __init__(self, eps=1e-6, in_place=True):
        super().__init__()
        # Align with PolyCom reference: initialize weights to (1/3, 1/3, 1/3) and bias to 1.0
        self.weight = nn.Parameter(torch.full((3,), 1.0 / 3.0))
        self.bias = nn.Parameter(torch.tensor(1.0))
        self.variance_epsilon = eps
        self.in_place = in_place

    def forward(self, hidden_states):
        return LigerPolyNormFunction.apply(
            hidden_states,
            self.weight,
            self.bias,
            self.variance_epsilon,
            self.in_place,
        )

    def extra_repr(self):
        return f"weight_shape={tuple(self.weight.shape)}, eps={self.variance_epsilon}, in_place={self.in_place}"
