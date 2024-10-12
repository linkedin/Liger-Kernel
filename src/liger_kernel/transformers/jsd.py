from typing import Optional

import torch

from liger_kernel.ops.jsd import LigerJSDFunction


class LigerJSD(torch.nn.Module):
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
        ignore_index (int): The index to ignore in the target. Default: `-100`

    Shape:
        - Input: :math:`(BT, V)`, where B is batch size, T is sequence length, V is vocab size.
        - Target: :math:`(BT, V)`, same shape as the input.
        - Label: :math:`(BT,)`
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

    def __init__(self, beta: float = 0.5, ignore_index: int = -100):
        super().__init__()
        assert (
            beta > 0 and beta < 1
        ), f"beta must be greater than 0 and less than 1. Got: {beta}"
        self.beta = beta
        self.ignore_index = ignore_index

    def forward(
        self,
        log_q: torch.tensor,
        log_p: torch.tensor,
        label: Optional[torch.tensor] = None,
    ):
        return LigerJSDFunction.apply(log_q, log_p, label, self.beta, self.ignore_index)
