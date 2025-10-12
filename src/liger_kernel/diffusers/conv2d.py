import torch
import torch.nn as nn
from liger_kernel.ops.conv2d import TritonConv2dFunction

class TritonConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        return TritonConv2dFunction.apply(x, self.weight, self.bias, self.padding, self.stride)