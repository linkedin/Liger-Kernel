from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from torch.nn import CrossEntropyLoss


class LigerCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super(LigerCrossEntropyLoss, self).__init__(*args, **kwargs)

    def forward(self, _input, target):
        return LigerCrossEntropyFunction.apply(_input, target, self.ignore_index)
