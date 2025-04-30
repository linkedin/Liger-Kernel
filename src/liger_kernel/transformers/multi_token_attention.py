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
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        sparse: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.sparse = sparse

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return LigerMultiTokenAttentionFunction.apply(
            scores,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.sparse,
        )
