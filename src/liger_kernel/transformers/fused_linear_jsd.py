import torch.nn as nn

from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction


class LigerFusedLinearJSD(nn.Module):
    r"""Fusing the last linear layer with generalized JSD

    Handle the forward and backward pass of the final linear layer via JSD by avoiding
    the materialization of the large logits tensor.

    Args:
        jsd_beta (float): coefficient beta of generalized JSD in the open interval (0, 1). Default: `0.5`
        temperature (float): temperature in softmax function to control the output probability distribution. Default: `1.0`

    Shape:
        - student_input: :math:`(BT, H)`, where B is batch size, T is sequence length, H is hidden dimension.
        - student_weight: :math:`(V, H)`, where V is vocab size.
        - teacher_input: :math:`(BT, H')`, where H' is hidden dimension of the teacher model.
        - teacher_weight: :math:`(V, H')`, where hidden size H and H' can be different.
        - Output: a scalar.

    Examples:
    ```python
    >>> (B, T, H, V) = (2, 2, 3, 5)
    >>> fused_jsd = LigerFusedLinearJSD(jsd_beta=0.1, temperature=2.0)
    >>> # generate inputs and weights
    >>> student_input = torch.rand(B * T, H, device="cuda", requires_grad=True)
    >>> student_lin = torch.nn.Linear(H, V, bias=False, device="cuda")
    >>> # teacher input doesn't require grad, hidden_dim can be different from student's
    >>> teacher_input = torch.rand(B * T, H * 2, device="cuda")
    >>> teacher_lin = torch.nn.Linear(H * 2, V, bias=False, device="cuda")
    >>> output = fused_jsd(student_input, student_lin.weight, teacher_input, teacher_lin.weight)
    >>> output.backward()
    ```
    """

    def __init__(self, jsd_beta=0.5, temperature=1.0):
        super().__init__()
        assert (
            jsd_beta > 0 and jsd_beta < 1
        ), f"beta must be greater than 0 and less than 1. Got: {jsd_beta}"
        assert temperature != 0, "temperature cannot be 0."
        self.jsd_beta = jsd_beta
        self.temperature = temperature

    def forward(
        self,
        student_input,
        student_weight,
        teacher_input,
        teacher_weight,
    ):
        return LigerFusedLinearJSDFunction.apply(
            student_input,
            student_weight,
            teacher_input,
            teacher_weight,
            self.jsd_beta,
            self.temperature,
        )
