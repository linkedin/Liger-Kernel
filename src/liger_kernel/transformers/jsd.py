import torch.nn as nn

from liger_kernel.ops.jsd import LigerJSDFunction


class LigerJSD(nn.Module):
    r"""The generalized Jensen-Shannon Divergence.
    .. math::
    JSD(\beta)(P || Q)
        = \beta * KLDiv(P || (\beta * P + (1 - \beta) * Q)) + (1 - \beta) * KLDiv(Q || (\beta * P + (1 - \beta) * Q))
    .. note::
    As all the other losses in PyTorch, this function expects the first argument,
    :attr:`log_q`, to be the predictions, the output of the student model in log-space,
    and the second, :attr:`log_p`, to be the observations, the output of the teacher model in log-space.
    This differs from the standard mathematical notation :math:`JSD(P || Q)` where
    :math:`P` denotes the teacher model and :math:`Q` denotes the student model.

    Args:
        beta (float): coefficient beta of generalized JSD in the open interval (0, 1). Default: `0.5`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: a scalar.

    Examples:
    ```python
    >>> jsd = LigerJSD(beta=0.1)
    >>> # input should be a distribution in the log space
    >>> input = torch.randn(3, 5, requires_grad=True).log_softmax(dim=-1)
    >>> target = torch.randn(3, 5, requires_grad=True).log_softmax(dim=-1)
    >>> output = jsd(input, target)
    ```
    """

    def __init__(self, beta=0.5):
        super().__init__()
        assert (
            beta > 0 and beta < 1
        ), f"beta must be greater than 0 and less than 1. Got: {beta}"
        self.beta = beta

    def forward(self, log_q, log_p):
        return LigerJSDFunction.apply(log_q, log_p, self.beta)
