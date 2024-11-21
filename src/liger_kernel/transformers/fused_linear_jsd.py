from typing import Optional

import torch

from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction


class LigerFusedLinearJSD(torch.nn.Module):
    r"""Fusing the last linear layer with generalized JSD

    Handle the forward and backward pass of the final linear layer via JSD by avoiding
    the materialization of the large logits tensor.

    Args:
        jsd_beta (float): coefficient beta of generalized JSD in the interval [0, 1]. It implements forward/reverse KL when beta equals 0 and 1 respectively. Default: `0.5`
        ignore_index (int): The index to ignore in the target. Default: `-100`
        temperature (float): temperature in softmax function to control the output probability distribution. Default: `1.0`

    Shape:
        - student_input: :math:`(BT, H)`, where B is batch size, T is sequence length, H is hidden dimension.
        - student_weight: :math:`(V, H)`, where V is vocab size.
        - teacher_input: :math:`(BT, H')`, where H' is hidden dimension of the teacher model.
        - teacher_weight: :math:`(V, H')`, where hidden size H and H' can be different.
        - shift_labels: :math:`(BT,)`
        - Output: a scalar.

    Examples:
    ```python
    >>> (B, T, H_s, H_t, V) = (2, 2, 3, 5, 10)
    >>> fused_jsd = LigerFusedLinearJSD(jsd_beta=0.1, temperature=2.0)
    >>> # generate inputs and weights
    >>> student_input = torch.rand(B * T, H_s, device="cuda", requires_grad=True)
    >>> student_lin = torch.nn.Linear(H_s, V, bias=False, device="cuda")
    >>> # teacher input doesn't require grad, hidden_dim can be different from student's
    >>> teacher_input = torch.rand(B * T, H_t, device="cuda")
    >>> teacher_lin = torch.nn.Linear(H_t, V, bias=False, device="cuda")
    >>> output = fused_jsd(student_input, student_lin.weight, teacher_input, teacher_lin.weight)
    >>> output.backward()
    >>>
    >>> # Example with labels for supervised fine-tuning (SFT) context:
    >>>
    >>> # Assume hidden_states, lm_heads and corresponding labels are given
    >>> student_lm_head = torch.nn.Linear(H_s, V, bias=False)
    >>> student_hidden_states = torch.randn(B * T, H_s, requires_grad=True).log_softmax(dim=-1)
    >>> teacher_lm_head = torch.nn.Linear(H_t, V, bias=False)
    >>> teacher_hidden_states = torch.randn(B * T, H_t).log_softmax(dim=-1)
    >>> labels = torch.randint(0, V, (B * T,), torch.long)
    >>>
    >>> # Shift so that tokens < n predict n
    >>> shift_student_hidden_states = student_hidden_states[..., :-1, :].contiguous()
    >>> shift_teacher_hidden_states = teacher_hidden_states[..., :-1, :].contiguous()
    >>> shift_labels = labels[..., 1:].contiguous()
    >>>
    >>> # Flatten tokens
    >>> shift_student_hidden_states = shift_student_hidden_states.view(-1, V)
    >>> shift_teacher_hidden_states = shift_teacher_hidden_states.view(-1, V)
    >>> shift_labels = shift_labels.view(-1)
    >>>
    >>> # Calculate loss
    >>> loss_fct = LigerJSD(beta=0.1)
    >>> loss = loss_fct(
    >>>     shift_studetn_hidden_states,
    >>>     student_lm_head.weight,
    >>>     shift_teacher_hidden_states,
    >>>     teacher_lm_head.weight,
    >>>     shift_labels
    >>> )
    ```
    """

    def __init__(self, jsd_beta=0.5, ignore_index=-100, temperature=1.0):
        super().__init__()
        assert temperature != 0, "temperature cannot be 0."
        self.jsd_beta = jsd_beta
        self.temperature = temperature
        self.ignore_index = ignore_index

    def forward(
        self,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        shift_labels: Optional[torch.LongTensor],
    ):
        return LigerFusedLinearJSDFunction.apply(
            student_input,
            student_weight,
            teacher_input,
            teacher_weight,
            shift_labels,
            self.jsd_beta,
            self.ignore_index,
            self.temperature,
        )
