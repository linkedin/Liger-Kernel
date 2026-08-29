from typing import Optional

import torch

from liger_kernel.ops import LigerFusedLinearKLDivFunction


class LigerFusedLinearKLDivLoss(torch.nn.Module):
    r"""Fusing the last linear layer with KL divergence against an explicit target distribution.

    Handle the forward and backward pass of the final linear layer via KL by avoiding
    the materialization of the large logits tensor. Unlike :class:`LigerFusedLinearJSD`,
    the teacher distribution is given directly as a probability tensor, so no teacher
    projection is fused: this is the fused-linear counterpart of
    :class:`liger_kernel.transformers.kl_div.LigerKLDIVLoss` (and ``torch.nn.KLDivLoss``).

    Useful when the teacher distribution is available as data, e.g. cached soft labels
    from offline distillation datasets.

    Args:
        reduction (str): reduction to be applied over the per-row KL losses: `"batchmean"`, `"mean"` or `"sum"` (same semantics as torch.nn.KLDivLoss). `"none"` is not supported. Default: `"batchmean"`
        ignore_index (int): The index to ignore in the target. Default: `-100`
        temperature (float): temperature applied to the student logits before softmax. Default: `1.0`
        eps (float): small value clamping the target before log so entries with q = 0 contribute 0 (0 * log 0 = 0). Default: `1e-10`
        accum_dtype (torch.dtype): the dtype of intermediate result buffers for weight gradient accumulations. Default: `None`

    Shape:
        - student_input: :math:`(BT, H)`, where B is batch size, T is sequence length, H is hidden dimension.
        - student_weight: :math:`(V, H)`, where V is vocab size.
        - target: :math:`(BT, V)`, target probabilities.
        - shift_labels: :math:`(BT,)`
        - Output: a scalar.

    Examples:
    ```python
    >>> (B, T, H, V) = (2, 2, 3, 10)
    >>> fused_kl = LigerFusedLinearKLDivLoss(temperature=2.0)
    >>> # generate inputs and weights
    >>> student_input = torch.rand(B * T, H, device="cuda", requires_grad=True)
    >>> student_lin = torch.nn.Linear(H, V, bias=False, device="cuda")
    >>> # target distribution given directly, e.g. cached teacher probabilities
    >>> target = torch.rand(B * T, V, device="cuda").softmax(dim=-1)
    >>> output = fused_kl(student_input, student_lin.weight, target)
    >>> output.backward()
    ```
    """

    def __init__(
        self,
        reduction: Optional[str] = "batchmean",
        ignore_index: int = -100,
        temperature: float = 1.0,
        eps: float = 1e-10,
        accum_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        assert temperature != 0, "temperature cannot be 0."
        self.reduction = reduction
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.eps = eps
        self.accum_dtype = accum_dtype

    def forward(
        self,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        target: torch.Tensor,
        shift_labels: Optional[torch.LongTensor] = None,
    ):
        return LigerFusedLinearKLDivFunction.apply(
            student_input,
            student_weight,
            target,
            shift_labels,
            self.reduction,
            self.ignore_index,
            self.temperature,
            self.eps,
            self.accum_dtype,
        )

    def extra_repr(self):
        return (
            f"reduction={self.reduction}, ignore_index={self.ignore_index}, "
            f"temperature={self.temperature}, eps={self.eps}"
        )
