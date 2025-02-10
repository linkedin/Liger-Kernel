import torch
import torch.nn as nn

from liger_kernel.ops.batch_norm import LigerBatchNormFunction


class LigerBatchNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, bias=True, init_fn="ones"):
        """
        Initialize the LigerBatchNorm class.

        Arguments:
            hidden_size (int): The size of the input features (i.e., the C dimension).
            eps (float): Small constant to prevent division by zero.
            bias (bool): Whether to use the bias term.
            init_fn (str): Initialization method for the weight, either "ones" or "zeros".
        """
        super().__init__()

        # Ensure init_fn parameter is valid
        assert init_fn in ["ones", "zeros"], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"

        self.hidden_size = hidden_size
        self.eps = eps

        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size) if not bias else torch.randn(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        Forward pass.

        Arguments:
            hidden_states (torch.Tensor): The input tensor, shape (N, C), where N is the batch size and C is the feature dimension.

        Returns:
            torch.Tensor: The normalized output tensor.
        """
        return LigerBatchNormFunction.apply(hidden_states, self.weight, self.bias, self.variance_epsilon)

    def extra_repr(self):
        """
        Returns additional information about the class, typically used to print more details when displaying the model.
        """
        return f"{self.hidden_size}, eps={self.eps}"
