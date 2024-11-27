
from test.utils import HFDistillationLoss, assert_verbose_allclose, set_seed

import pytest
import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss
from liger_kernel.chunked_loss.jsd_loss import LigerFusedLinearJSDFunction
from liger_kernel.chunked_loss.functional import liger_fused_linear_jsd
from liger_kernel.utils import infer_device

device = infer_device()

# set random seed globally
set_seed()


class HFJSDLoss(HFDistillationLoss):
    """
    Naive implementation of a distillation loss using Jensen-Shannon Divergence (JSD).
    """

    def __init__(
        self,
        temperature: float = 1.0,
        ignore_index: int = -100,
        beta: float = 0.5
    ):
        super().__init__(ignore_index=ignore_index, beta=beta)
        self.temperature = temperature

    def distillation_loss(
        self,
        student_logps: torch.FloatTensor,
        teacher_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Compute Jensen-Shannon Divergence loss between student and teacher distributions.
        Args:
            student_logps (torch.Tensor): Log probabilities from student model (Raw logits after log_softmax)
            teacher_logps (torch.Tensor): Log probabilities from teacher model (Raw logits after log_softmax)
            temperature (float): Temperature for softening probability distributions
        Returns:
            torch.Tensor: Jensen-Shannon Divergence loss
        """
        # TODO: should incorporate with (high) temperature scaling on raw logits

        # For instance,
        # Scale logits by temperature
        # student_logits = student_logits / self.temperature
        # teacher_logits = teacher_logits / self.temperature
        # Convert to probabilities
        # student_probs = F.softmax(student_logits, dim=-1)
        # teacher_probs = F.softmax(teacher_logits, dim=-1)

        mean_probs = (torch.exp(student_logps) + torch.exp(teacher_logps)) / 2

        student_kl = F.kl_div(
            student_logps,
            mean_probs,
            reduction='batchmean',
            log_target=False,
        )

        teacher_kl = F.kl_div(
            teacher_logps,
            mean_probs,
            reduction='batchmean',
            log_target=False,
        )

        # JSD is the average of the KL divergences
        jsd_loss = (student_kl + teacher_kl) / 2
        return jsd_loss


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
        # Create smaller student weight
        self.student_lin = torch.nn.Linear(
            in_features=H // 2, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.teacher_lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.jsd = HFJSDLoss(ignore_index=ignore_index, beta=beta).get_batch_loss_metrics
        self.temperature = temperature

    def forward(self, student_input, teacher_input, target):

        jsd_loss = self.jsd(
            student_input,
            self.student_lin.weight,
            teacher_input,
            self.teacher_lin.weight,
            target,
        )
        return jsd_loss


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
        # Create smaller student weight
        self.student_lin = torch.nn.Linear(
            in_features=H // 2, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.teacher_lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.chunked_jsd = LigerFusedLinearJSDLoss(
            beta=beta, ignore_index=ignore_index
        )
        self.temperature = temperature

    def forward(self, student_input, teacher_input, target):
        return self.chunked_jsd(
            student_input,
            self.student_lin.weight,
            teacher_input,
            self.teacher_lin.weight,
            target,
        )


#############################################################################
# Test the correctness of the fused linear JSD
#############################################################################


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (8, 128, 1024, 4096),
        (3, 47, 31, 123),  # random shape
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
        (1.0, 0.0),
        (1.0, 1.0),
    ],
)
def test_correctness(B, T, H, V, scalar, dtype, beta, temperature, atol, rtol):
    torch_lm_head_jsd = TorchLMHeadJSD(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        beta=beta,
    )
    liger_lm_head_jsd = LigerLMHeadJSD(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        beta=beta,
    )

    torch_lm_head_jsd.student_lin.weight.data = (
        liger_lm_head_jsd.student_lin.weight.data
    ) = torch.rand(V, H // 2, device=device, dtype=dtype)
    torch_lm_head_jsd.teacher_lin.weight.data = (
        liger_lm_head_jsd.teacher_lin.weight.data
    ) = torch.rand(V, H, device=device, dtype=dtype)

    _tensor = torch.rand(B * T, H // 2, device=device, dtype=dtype) * scalar
    student_input1 = _tensor.detach().clone().requires_grad_(True)
    student_input2 = _tensor.detach().clone().requires_grad_(True)

    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype) * scalar

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    with torch.autograd.detect_anomaly():
        output1 = torch_lm_head_jsd(student_input1, teacher_input, target)
        output2 = liger_lm_head_jsd(student_input2, teacher_input, target)

        assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

    output1.backward()
    output2.backward()

    assert_verbose_allclose(student_input1.grad, student_input2.grad, atol=atol, rtol=rtol)

    assert_verbose_allclose(
        torch_lm_head_jsd.student_lin.weight.grad,
        liger_lm_head_jsd.student_lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (8, 128, 1024, 4096),
        (3, 47, 31, 123),  # random shape
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
        (1.0, 0.0, 2),
        (2.0, 0.1, 42),
        (1.0, 1.0, 2),
    ],
)
def test_correctness_with_ignore_index(
    B, T, H, V, scalar, dtype, beta, ignore_index, temperature, atol, rtol
):
    torch_lm_head_jsd = TorchLMHeadJSD(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        ignore_index=ignore_index,
        beta=beta,
    )
    liger_lm_head_jsd = LigerLMHeadJSD(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        ignore_index=ignore_index,
        beta=beta,
    )

    torch_lm_head_jsd.student_lin.weight.data = (
        liger_lm_head_jsd.student_lin.weight.data
    ) = torch.rand(V, H // 2, device=device, dtype=dtype)
    torch_lm_head_jsd.teacher_lin.weight.data = (
        liger_lm_head_jsd.teacher_lin.weight.data
    ) = torch.rand(V, H, device=device, dtype=dtype)

    _tensor = torch.rand(B * T, H // 2, device=device, dtype=dtype) * scalar
    student_input1 = _tensor.detach().clone().requires_grad_(True)
    student_input2 = _tensor.detach().clone().requires_grad_(True)

    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype) * scalar

    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()
    indices_to_assign = torch.randperm(B * T)[
        :num_elements_to_assign
    ]
    label[indices_to_assign] = ignore_index

    output1 = torch_lm_head_jsd(student_input1, teacher_input, label)
    output2 = liger_lm_head_jsd(student_input2, teacher_input, label)

    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

    output1.backward()
    output2.backward()

    assert_verbose_allclose(student_input1.grad, student_input2.grad, atol=atol, rtol=rtol)

    assert_verbose_allclose(
        torch_lm_head_jsd.student_lin.weight.grad,
        liger_lm_head_jsd.student_lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (2, 2, 8, 8),
        (9, 7, 41, 41),
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
    # init the linear in all FusedLinearJSDs with the same weights
    _weight = torch.rand(V, H // 2, device=device, dtype=dtype)
    student_weight1 = _weight.detach().clone().requires_grad_(True)
    student_weight2 = _weight.detach().clone().requires_grad_(True)
    teacher_weight = torch.rand(V, H, device=device, dtype=dtype)

    _tensor = torch.rand(B * T, H // 2, device=device, dtype=dtype) * scalar
    student_input1 = _tensor.detach().clone().requires_grad_(True)
    student_input2 = _tensor.detach().clone().requires_grad_(True)
    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype) * scalar

    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()
    indices_to_assign = torch.randperm(B * T)[
        :num_elements_to_assign
    ]
    label[indices_to_assign] = ignore_index

    output1 = liger_fused_linear_jsd(
        student_input1,
        student_weight1,
        teacher_input,
        teacher_weight,
        label,
        beta,
        ignore_index,
        temperature,
    )
    output2 = LigerFusedLinearJSDFunction.apply(
        student_input2,
        student_weight2,
        teacher_input,
        teacher_weight,
        label,
        beta,
        ignore_index,
        temperature,
    )

    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

    output1.backward()
    output2.backward()

    assert_verbose_allclose(student_input1.grad, student_input2.grad, atol=atol, rtol=rtol)

    assert_verbose_allclose(student_weight1.grad, student_weight2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (8, 128, 1024, 4096),
        (3, 47, 31, 123),  # random shape
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
    torch_lm_head_jsd = TorchLMHeadJSD(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        ignore_index=ignore_index,
        beta=beta,
    )
    liger_lm_head_jsd = LigerLMHeadJSD(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        ignore_index=ignore_index,
        beta=beta,
    )

    torch_lm_head_jsd.student_lin.weight.data = (
        liger_lm_head_jsd.student_lin.weight.data
    ) = torch.rand(V, H // 2, device=device, dtype=dtype)
    torch_lm_head_jsd.teacher_lin.weight.data = (
        liger_lm_head_jsd.teacher_lin.weight.data
    ) = torch.rand(V, H, device=device, dtype=dtype)

    _tensor = torch.rand(B * T, H // 2, device=device, dtype=dtype) * scalar
    student_input1 = _tensor.detach().clone().requires_grad_(True)
    student_input2 = _tensor.detach().clone().requires_grad_(True)

    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype) * scalar

    label = torch.full((B * T,), ignore_index, device=device, dtype=torch.long)

    output1 = torch_lm_head_jsd(student_input1, teacher_input, label)
    output2 = liger_lm_head_jsd(student_input2, teacher_input, label)

    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)
    assert_verbose_allclose(output2, torch.zeros_like(output2), atol=atol, rtol=rtol)

    output2.backward()

    assert_verbose_allclose(
        torch.zeros_like(student_input2.grad), student_input2.grad, atol=atol, rtol=rtol
    )


@pytest.mark.parametrize(
    "autocast_dtype, atol, rtol",
    [
        (torch.bfloat16, 5e-3, 5e-2),
        (torch.float16, 5e-3, 5e-2),
    ],
)
def test_amp(autocast_dtype, atol, rtol):
    B = 2
    T = 4
    H = 2048
    V = 3200
    scalar = 1.0
    ignore_index = -100
    temperature = 1.0
    beta = 0.5
    dtype = torch.float32
    torch_lm_head_jsd = TorchLMHeadJSD(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        ignore_index=ignore_index,
        beta=beta,
    )
    liger_lm_head_jsd = LigerLMHeadJSD(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        ignore_index=ignore_index,
        beta=beta,
    )

    torch_lm_head_jsd.student_lin.weight.data = (
        liger_lm_head_jsd.student_lin.weight.data
    ) = torch.rand(V, H // 2, device=device, dtype=dtype)
    torch_lm_head_jsd.teacher_lin.weight.data = (
        liger_lm_head_jsd.teacher_lin.weight.data
    ) = torch.rand(V, H, device=device, dtype=dtype)

    _tensor = torch.rand(B * T, H // 2, device=device, dtype=autocast_dtype) * scalar
    student_input1 = _tensor.detach().clone().requires_grad_(True)
    student_input2 = _tensor.detach().clone().requires_grad_(True)

    teacher_input = torch.rand(B * T, H, device=device, dtype=autocast_dtype) * scalar

    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()
    indices_to_assign = torch.randperm(B * T)[
        :num_elements_to_assign
    ]
    label[indices_to_assign] = ignore_index

    with torch.autocast(device_type=device, dtype=autocast_dtype):
        output1 = torch_lm_head_jsd(student_input1, teacher_input, label)
        output2 = liger_lm_head_jsd(student_input2, teacher_input, label)

        assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

        output1.backward()
        output2.backward()

    assert_verbose_allclose(student_input1.grad, student_input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(
        torch_lm_head_jsd.student_lin.weight.grad,
        liger_lm_head_jsd.student_lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )
