import math

import torch
import torch.nn as nn

from torch.nn.modules.utils import _pair

from liger_kernel.ops.multi_token_attention import LigerMultiTokenAttentionFunction


class LigerMultiTokenAttention(nn.Module):
    """
    Multi-Token Attention:
        out = mask_{0}(conv2d(softmax(mask_{-\inf}(scores))))

    Reference: https://arxiv.org/pdf/2504.00927
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias: bool = True,
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        kH, kW = _pair(kernel_size)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kH, kW))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return LigerMultiTokenAttentionFunction.apply(
            scores,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
