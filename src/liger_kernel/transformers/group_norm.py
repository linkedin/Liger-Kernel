import torch
import torch.nn as nn

from liger_kernel.ops.group_norm import LigerGroupNormFunction


class LigerGroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups, eps=1e-6, bias=False, init_fn="ones"):
        """
        A Group Normalization layer.
        Args:
            num_channels (int): Number of channels in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
            eps (float, optional): A value added to the denominator for numerical stability. Default: 1e-6.
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``.
            init_fn (str, optional): Initialization function for the learnable parameters. Default: "ones".
        """
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"

        assert (
            num_channels % num_groups == 0
        ), f"Number of channels {num_channels} must be divisible by num_groups {num_groups}"
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels) if init_fn == "ones" else torch.zeros(num_channels))
        self.bias = nn.Parameter(torch.randn(num_channels) if bias else torch.zeros(num_channels))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # hidden_states: (batch_size, num_channels, *)
        assert hidden_states.dim() >= 3, f"Input must have atleast 3 dimensions, got {hidden_states.dim()}"
        assert (
            hidden_states.size(1) == self.num_channels
        ), f"Input tensor must have {self.num_channels} channels, got {hidden_states.size(1)}"
        return LigerGroupNormFunction.apply(
            hidden_states,
            self.weight,
            self.bias,
            self.num_channels,
            self.num_groups,
            self.variance_epsilon,
        )

    def extra_repr(self):
        return f"{self.hidden_size}, num_channels={self.num_channels}, num_groups={self.num_groups}, eps={self.eps}"
