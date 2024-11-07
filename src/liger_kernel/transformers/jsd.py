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
        - shift_labels (Optional): :math:`(BT,)`
        - Output: a scalar.

    Examples:
    ```python
    >>> (B, T, V) = (2, 2, 5)
    >>> jsd = LigerJSD(beta=0.1)
    >>> # input should be a distribution in the log space
    >>> input = torch.randn(B * T, V, requires_grad=True).log_softmax(dim=-1)
    >>> target = torch.randn(B * T, V).log_softmax(dim=-1)
    >>> output = jsd(input, target)
    >>>
    >>> # Example with labels for supervised fine-tuning (SFT) context
    >>> # Assume logits and corresponding labels are given
    >>> student_logits = torch.randn(B * T, V, requires_grad=True).log_softmax(dim=-1)
    >>> teacher_logits = torch.randn(B * T, V).log_softmax(dim=-1)
    >>> labels = torch.randint(0, V, (B * T,), torch.long)
    >>> # Shift so that tokens < n predict n
    >>> shift_student_logits = student_logits[..., :-1, :].contiguous()
    >>> shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
    >>> shift_labels = labels[..., 1:].contiguous()
    >>> # Flatten tokens
    >>> shift_student_logits = shift_student_logits.view(-1, V)
    >>> shift_teacher_logits = shift_teacher_logits.view(-1, V)
    >>> shift_labels = shift_labels.view(-1)
    >>> # Calculate loss
    >>> loss_fct = LigerJSD(beta=0.1)
    >>> loss = loss_fct(shift_studetn_logits, shift_teacher_logits, shift_labels)

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
        log_q: torch.Tensor,
        log_p: torch.Tensor,
        shift_labels: Optional[torch.LongTensor] = None,
    ):
        return LigerJSDFunction.apply(
            log_q, log_p, shift_labels, self.beta, self.ignore_index
        )
