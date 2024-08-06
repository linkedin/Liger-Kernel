import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from liger_kernel.ops.linear_fused_cross_entropy import (
    LigerLinearFusedCrossEntropyFunction,
)


class LigerLinearFusedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, in_features: int, num_classes: int, ignore_index: int = -100):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.linear = torch.nn.Parameter(
            torch.empty(self.num_classes, self.in_features)
        )
        torch.nn.init.kaiming_uniform_(self.linear, a=np.sqrt(5))

    def forward(self, _input, target):
        return LigerLinearFusedCrossEntropyFunction.apply(
            _input, self.linear, target, self.ignore_index
        )


class LigerStatelessLCE(CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super(LigerStatelessLCE, self).__init__(*args, **kwargs)

    def forward(self, lin_weight, _input, target):
        return LigerLinearFusedCrossEntropyFunction.apply(
            _input, lin_weight, target, self.ignore_index
        )
