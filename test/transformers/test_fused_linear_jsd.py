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
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(
            in_features=H // 2, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.teacher_lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.jsd = TorchJSD(beta=beta, ignore_index=ignore_index, dtype=dtype)
        self.temperature = temperature

    def forward(self, student_input, teacher_input, label=None):
        student_logits = self.student_lin(student_input)
        teacher_logits = self.teacher_lin(teacher_input)
        student_prob = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_prob = torch.log_softmax(teacher_logits / self.temperature, dim=-1)

        return self.jsd(student_prob, teacher_prob, label)


class LigerLMHeadJSD(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(
            in_features=H // 2, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.teacher_lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.fused_jsd = LigerFusedLinearJSD(
            jsd_beta=beta, ignore_index=ignore_index, temperature=temperature
        )

    def forward(self, student_input, teacher_input, label=None):
        return self.fused_jsd(
            student_input,
            self.student_lin.weight,
            teacher_input,
            self.teacher_lin.weight,
            label,
        )


#############################################################################
# Test the correctness of the fused linear JSD
#############################################################################


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (2, 2, 512, 1600),
        (2, 4, 1024, 1600),
        # Comment out to speed up testing
        # (4, 2048, 4096, 128256),  # llama3 8B
        # (4, 1024, 8192, 128256),  # llama3 70B
        (4, 423, 167, 1423),  # random shape
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-3, 5e-2),
        (1.0, torch.float32, 1e-5, 5e-4),
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

    with torch.autograd.detect_anomaly():
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
        (1.0, torch.bfloat16, 5e-3, 5e-2),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize(
    "temperature, beta, ignore_index",
    [
        (1.0, 0.5, 2),
        (2.0, 0.1, 42),
    ],
)
def test_correctness_with_ignore_index(
    B, T, H, V, scalar, dtype, beta, ignore_index, temperature, atol, rtol
):
    device = "cuda"
    torch_lm_head_jsd = TorchLMHeadJSD(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        ignore_index=ignore_index,
        beta=beta,
    ).to(device)
    liger_lm_head_jsd = LigerLMHeadJSD(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        ignore_index=ignore_index,
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

    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[
        :num_elements_to_assign
    ]  # Randomly select indices
    label[indices_to_assign] = ignore_index

    output1 = torch_lm_head_jsd(_input1, teacher_input, label)
    output2 = liger_lm_head_jsd(_input2, teacher_input, label)

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
@pytest.mark.parametrize(
    "temperature, beta, ignore_index", [(1.0, 0.5, -100), (2.0, 0.1, 42)]
)
def test_correctness_functional(
    B, T, H, V, scalar, dtype, beta, ignore_index, temperature, atol, rtol
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

    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[
        :num_elements_to_assign
    ]  # Randomly select indices
    label[indices_to_assign] = ignore_index

    output1 = liger_fused_linear_jsd(
        _input1,
        _weight1,
        teacher_input,
        teacher_weight,
        label,
        beta,
        ignore_index,
        temperature,
    )
    output2 = LigerFusedLinearJSDFunction.apply(
        _input2,
        _weight2,
        teacher_input,
        teacher_weight,
        label,
        beta,
        ignore_index,
        temperature,
    )

    assert torch.allclose(output1, output2, atol=atol, rtol=rtol)

    output1.backward()
    output2.backward()

    assert torch.allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)

    assert torch.allclose(_weight1.grad, _weight2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (2, 4, 2048, 3200),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-3, 5e-2),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize(
    "temperature, beta, ignore_index",
    [
        (1.0, 0.5, 2),
        (2.0, 0.1, 42),
    ],
)
def test_correctness_all_ignored(
    B, T, H, V, scalar, dtype, beta, ignore_index, temperature, atol, rtol
):
    device = "cuda"
    torch_lm_head_jsd = TorchLMHeadJSD(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        ignore_index=ignore_index,
        beta=beta,
    ).to(device)
    liger_lm_head_jsd = LigerLMHeadJSD(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        ignore_index=ignore_index,
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

    label = torch.full((B * T,), ignore_index, device=device, dtype=torch.long)

    output1 = torch_lm_head_jsd(_input1, teacher_input, label)
    output2 = liger_lm_head_jsd(_input2, teacher_input, label)

    assert torch.allclose(output1, output2, atol=atol, rtol=rtol)
    assert torch.allclose(output2, torch.zeros_like(output2), atol=atol, rtol=rtol)

    output2.backward()

    assert torch.allclose(
        torch.zeros_like(_input2.grad), _input2.grad, atol=atol, rtol=rtol
    )
