from test.transformers.test_jsd import JSD as TorchJSD
from test.utils import assert_verbose_allclose, set_seed

import pytest
import torch

from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction
from liger_kernel.transformers.functional import liger_fused_linear_jsd
from liger_kernel.transformers.fused_linear_jsd import LigerFusedLinearJSD

set_seed(42)


class TorchLMHeadCE(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based jsd loss.

    :param H: hidden size
    :param V: vocab size
    """

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        temperature: float = 1.0,
        beta: float = 0.5,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(in_features=H, out_features=V, dtype=dtype)
        self.teacher_lin = torch.nn.Linear(in_features=H, out_features=V, dtype=dtype)
        self.jsd = TorchJSD()
        self.temperature = temperature
        self.beta = beta

    def forward(self, student_input, teacher_input):
        student_logits = self.student_lin(student_input)
        teacher_logits = self.teacher_lin_lin(teacher_input)
        student_prob = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_prob = torch.log_softmax(teacher_logits / self.temperature, dim=-1)

        return self.jsd(student_prob, teacher_prob)


class LigerLMHeadCE(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, dtype=dtype)
        self.ce_loss = LigerFusedLinearJSD()

    def forward(self, x, y):
        return self.ce_loss(self.lin.weight, x, y, self.lin.bias)


#############################################################################
# Test the correctness of the fused linear cross entropy loss
#############################################################################


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        # (2, 4, 512, 512),  # The test does not work on some CI GPUs. Issue #160
        (8, 2048, 4096, 32000),  # llama2, mistral
        # Comment out to speed up testing
        # (4, 2048, 4096, 128256),  # llama3 8B
        # (4, 1024, 8192, 128256),  # llama3 70B
        (4, 423, 8192, 32000),  # random shape
    ],
)
@pytest.mark.parametrize(
    "reduction, scalar, dtype, atol, rtol",
    [
        ("mean", 1.0, torch.bfloat16, 5e-3, 5e-2),
        ("mean", 1.0, torch.float32, 1e-5, 5e-4),
        ("sum", 1.0, torch.bfloat16, 5e-0, 5e1),
        ("sum", 1.0, torch.float32, 1e-3, 5e-2),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("label_smoothing", [0, 0.1])
def test_correctness(
    B, T, H, V, scalar, dtype, bias, label_smoothing, reduction, atol, rtol
):
    device = "cuda"
    torch_lm_head_ce = TorchLMHeadCE(
        H=H,
        V=V,
        bias=bias,
        label_smoothing=label_smoothing,
        reduction=reduction,
        dtype=dtype,
    ).to(device)
    liger_lm_head_ce = LigerLMHeadCE(
        H=H,
        V=V,
        bias=bias,
        label_smoothing=label_smoothing,
        reduction=reduction,
        dtype=dtype,
    ).to(device)

    # init the linear in all CEs with the same weights
    torch_lm_head_ce.lin.weight.data = liger_lm_head_ce.lin.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )

    if bias:
        torch_lm_head_ce.lin.bias.data = liger_lm_head_ce.lin.bias.data = torch.rand(
            V, device=device, dtype=dtype
        )

    _tensor = torch.randn(B * T, H, device=device, dtype=dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    output1 = torch_lm_head_ce(_input1, target)
    output2 = liger_lm_head_ce(_input2, target)

    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

    output1.backward()
    output2.backward()

    assert_verbose_allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)

    assert_verbose_allclose(
        torch_lm_head_ce.lin.weight.grad,
        liger_lm_head_ce.lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )

    if bias:
        assert_verbose_allclose(
            torch_lm_head_ce.lin.bias.grad,
            liger_lm_head_ce.lin.bias.grad,
            atol=atol,
            rtol=rtol,
        )