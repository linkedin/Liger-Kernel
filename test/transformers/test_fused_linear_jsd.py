from test.transformers.test_jsd import JSD as TorchJSD
from test.utils import set_seed

import pytest
import torch

from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction
from liger_kernel.transformers.functional import liger_fused_linear_jsd
from liger_kernel.transformers.fused_linear_jsd import LigerFusedLinearJSD

set_seed(42)


class TorchLMHeadJSD(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based jsd loss.

    :param H: hidden size
    :param V: vocab size
    :param temperature: softmax temperature
    :param beta: jsd beta
    """

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
        temperature: float = 1.0,
        beta: float = 0.5,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(
            in_features=H // 2, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.teacher_lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.jsd = TorchJSD(beta, dtype=dtype)
        self.temperature = temperature

    def forward(self, student_input, teacher_input):
        student_logits = self.student_lin(student_input)
        teacher_logits = self.teacher_lin(teacher_input)
        student_prob = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_prob = torch.log_softmax(teacher_logits / self.temperature, dim=-1)

        return self.jsd(student_prob, teacher_prob)


class LigerLMHeadJSD(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
        temperature: float = 1.0,
        beta: float = 0.5,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(
            in_features=H // 2, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.teacher_lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.fused_jsd = LigerFusedLinearJSD(beta, temperature)

    def forward(self, student_input, teacher_input):
        return self.fused_jsd(
            student_input,
            self.student_lin.weight,
            teacher_input,
            self.teacher_lin.weight,
        )


#############################################################################
# Test the correctness of the fused linear JSD
#############################################################################


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (2, 4, 2048, 3200),
        (2, 2048, 4096, 32000),  # llama2, mistral
        # Comment out to speed up testing
        # (4, 2048, 4096, 128256),  # llama3 8B
        # (4, 1024, 8192, 128256),  # llama3 70B
        (4, 423, 8192, 32000),  # random shape
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (0.5, torch.bfloat16, 5e-3, 5e-2),  # setting scalar=0.5 to prevent overflow
        (0.5, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize(
    "temperature, beta",
    [
        (1.0, 0.5),
        (2.0, 0.1),
    ],
)
def test_correctness(B, T, H, V, scalar, dtype, beta, temperature, atol, rtol):
    device = "cuda"
    torch_lm_head_jsd = TorchLMHeadJSD(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        beta=beta,
    ).to(device)
    liger_lm_head_jsd = LigerLMHeadJSD(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        beta=beta,
    ).to(device)

    # init the linear in all FusedLinearJSDs with the same weights
    torch_lm_head_jsd.student_lin.weight.data = (
        liger_lm_head_jsd.student_lin.weight.data
    ) = torch.rand(V, H // 2, device=device, dtype=dtype)
    torch_lm_head_jsd.teacher_lin.weight.data = (
        liger_lm_head_jsd.teacher_lin.weight.data
    ) = torch.rand(V, H, device=device, dtype=dtype)

    _tensor = torch.rand(B * T, H // 2, device=device, dtype=dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype) * scalar

    output1 = torch_lm_head_jsd(_input1, teacher_input)
    output2 = liger_lm_head_jsd(_input2, teacher_input)

    assert torch.allclose(output1, output2, atol=atol, rtol=rtol)

    output1.backward()
    output2.backward()

    assert torch.allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)

    assert torch.allclose(
        torch_lm_head_jsd.student_lin.weight.grad,
        liger_lm_head_jsd.student_lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (2, 4, 2048, 3200),
        (2, 2048, 4096, 32000),  # llama2, mistral
        # Comment out to speed up testing
        # (4, 2048, 4096, 128256),  # llama3 8B
        # (4, 1024, 8192, 128256),  # llama3 70B
        (4, 423, 8192, 32000),  # random shape
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (0.5, torch.bfloat16, 5e-3, 5e-2),
        (0.5, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize("temperature, beta", [(1.0, 0.5), (2.0, 0.1)])
def test_correctness_functional(
    B, T, H, V, scalar, dtype, beta, temperature, atol, rtol
):
    device = "cuda"

    # init the linear in all FusedLinearJSDs with the same weights
    _weight = torch.rand(V, H // 2, device=device, dtype=dtype)
    _weight1 = _weight.detach().clone().requires_grad_(True)
    _weight2 = _weight.detach().clone().requires_grad_(True)
    teacher_weight = torch.rand(V, H, device=device, dtype=dtype)

    _tensor = torch.rand(B * T, H // 2, device=device, dtype=dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)
    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype) * scalar

    output1 = liger_fused_linear_jsd(
        _input1, _weight1, teacher_input, teacher_weight, beta, temperature
    )
    output2 = LigerFusedLinearJSDFunction.apply(
        _input2, _weight2, teacher_input, teacher_weight, beta, temperature
    )

    assert torch.allclose(output1, output2, atol=atol, rtol=rtol)

    output1.backward()
    output2.backward()

    assert torch.allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)

    assert torch.allclose(_weight1.grad, _weight2.grad, atol=atol, rtol=rtol)
