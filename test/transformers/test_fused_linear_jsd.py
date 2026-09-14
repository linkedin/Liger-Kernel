import pytest
import torch

from packaging.version import Version
from test.transformers.test_jsd import JSD as TorchJSD
from test.utils import assert_verbose_allclose
from test.utils import set_seed

from liger_kernel.ops import LigerFusedLinearJSDFunction
from liger_kernel.transformers.functional import liger_fused_linear_jsd
from liger_kernel.transformers.fused_linear_jsd import LigerFusedLinearJSD
from liger_kernel.utils import infer_device

device = infer_device()

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
        self.student_lin = torch.nn.Linear(in_features=H // 2, out_features=V, bias=False, dtype=dtype, device=device)
        self.teacher_lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype, device=device)
        self.jsd = TorchJSD(beta=beta, ignore_index=ignore_index, dtype=dtype)
        self.temperature = temperature

    def forward(self, student_input, teacher_input, label=None):
        student_logits = self.student_lin(student_input).to(torch.float32)
        teacher_logits = self.teacher_lin(teacher_input).to(torch.float32)
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
        self.student_lin = torch.nn.Linear(in_features=H // 2, out_features=V, bias=False, dtype=dtype, device=device)
        self.teacher_lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype, device=device)
        self.fused_jsd = LigerFusedLinearJSD(jsd_beta=beta, ignore_index=ignore_index, temperature=temperature)

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
        (8, 128, 1024, 4096),
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
        (1.0, 0.0),  # FKL
        (1.0, 1.0),  # RKL
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
    torch_lm_head_jsd.student_lin.weight.data = liger_lm_head_jsd.student_lin.weight.data = torch.rand(
        V, H // 2, device=device, dtype=dtype
    )
    torch_lm_head_jsd.teacher_lin.weight.data = liger_lm_head_jsd.teacher_lin.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )

    _tensor = torch.rand(B * T, H // 2, device=device, dtype=dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype) * scalar

    with torch.autograd.detect_anomaly():
        output1 = torch_lm_head_jsd(_input1, teacher_input)
        output2 = liger_lm_head_jsd(_input2, teacher_input)

        assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

    output1.backward()
    output2.backward()

    assert_verbose_allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)

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
    "temperature, beta, ignore_index",
    [
        (1.0, 0.5, 2),
        (1.0, 0.0, 2),
        (2.0, 0.1, 42),
        (1.0, 1.0, 2),
    ],
)
def test_correctness_with_ignore_index(B, T, H, V, scalar, dtype, beta, ignore_index, temperature, atol, rtol):
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
    torch_lm_head_jsd.student_lin.weight.data = liger_lm_head_jsd.student_lin.weight.data = torch.rand(
        V, H // 2, device=device, dtype=dtype
    )
    torch_lm_head_jsd.teacher_lin.weight.data = liger_lm_head_jsd.teacher_lin.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )

    _tensor = torch.rand(B * T, H // 2, device=device, dtype=dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype) * scalar

    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]  # Randomly select indices
    label[indices_to_assign] = ignore_index

    output1 = torch_lm_head_jsd(_input1, teacher_input, label)
    output2 = liger_lm_head_jsd(_input2, teacher_input, label)

    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

    output1.backward()
    output2.backward()

    assert_verbose_allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)

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
        # weird shapes
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
@pytest.mark.parametrize("temperature, beta, ignore_index", [(1.0, 0.5, -100), (2.0, 0.1, 42)])
@pytest.mark.parametrize("accum_dtype", [None, torch.float32])
def test_correctness_functional(B, T, H, V, scalar, dtype, beta, ignore_index, temperature, accum_dtype, atol, rtol):
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
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]  # Randomly select indices
    label[indices_to_assign] = ignore_index

    output1 = liger_fused_linear_jsd(
        student_input=_input1,
        student_weight=_weight1,
        teacher_input=teacher_input,
        teacher_weight=teacher_weight,
        shift_labels=label,
        jsd_beta=beta,
        ignore_index=ignore_index,
        temperature=temperature,
        accum_dtype=accum_dtype,
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
        accum_dtype,
        None,
        None,
    )

    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

    output1.backward()
    output2.backward()

    assert_verbose_allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)

    assert_verbose_allclose(_weight1.grad, _weight2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (8, 128, 1024, 4096),
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
    "temperature, beta, ignore_index",
    [
        (1.0, 0.5, 2),
        (2.0, 0.1, 42),
    ],
)
def test_correctness_all_ignored(B, T, H, V, scalar, dtype, beta, ignore_index, temperature, atol, rtol):
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
    torch_lm_head_jsd.student_lin.weight.data = liger_lm_head_jsd.student_lin.weight.data = torch.rand(
        V, H // 2, device=device, dtype=dtype
    )
    torch_lm_head_jsd.teacher_lin.weight.data = liger_lm_head_jsd.teacher_lin.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )

    _tensor = torch.rand(B * T, H // 2, device=device, dtype=dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype) * scalar

    label = torch.full((B * T,), ignore_index, device=device, dtype=torch.long)

    output1 = torch_lm_head_jsd(_input1, teacher_input, label)
    output2 = liger_lm_head_jsd(_input2, teacher_input, label)

    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)
    assert_verbose_allclose(output2, torch.zeros_like(output2), atol=atol, rtol=rtol)

    output2.backward()

    assert_verbose_allclose(torch.zeros_like(_input2.grad), _input2.grad, atol=atol, rtol=rtol)


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
    torch_lm_head_jsd.student_lin.weight.data = liger_lm_head_jsd.student_lin.weight.data = torch.rand(
        V, H // 2, device=device, dtype=dtype
    )
    torch_lm_head_jsd.teacher_lin.weight.data = liger_lm_head_jsd.teacher_lin.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )

    _tensor = torch.rand(B * T, H // 2, device=device, dtype=autocast_dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    teacher_input = torch.rand(B * T, H, device=device, dtype=autocast_dtype) * scalar

    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]  # Randomly select indices
    label[indices_to_assign] = ignore_index

    with torch.autocast(device_type=device, dtype=autocast_dtype):
        output1 = torch_lm_head_jsd(_input1, teacher_input, label)
        output2 = liger_lm_head_jsd(_input2, teacher_input, label)

        assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

        output1.backward()
        output2.backward()

    assert_verbose_allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(
        torch_lm_head_jsd.student_lin.weight.grad,
        liger_lm_head_jsd.student_lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )


def _float32_projection_reference(student_input, student_weight, teacher_input, teacher_weight, beta, temperature):
    """Reference whose projection is *not* rounded to the input dtype.

    ``TorchLMHeadJSD`` above runs its ``nn.Linear`` in the input dtype and only then casts to
    float32, so it rounds the logits exactly the way the kernel used to. That makes it unable to
    detect logit rounding. This oracle casts the operands first, which is exact for bfloat16 and
    float16 operands, so the only difference from the kernel is where the rounding happens.
    """
    student_logits = student_input.float() @ student_weight.float().t()
    teacher_logits = teacher_input.float() @ teacher_weight.float().t()
    student_prob = torch.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = torch.log_softmax(teacher_logits / temperature, dim=-1)
    return TorchJSD(beta=beta, dtype=torch.float32)(student_prob, teacher_prob)


# Logit spread chosen per beta so the reference gradient stays well inside the storage dtype's
# range. The generalized-JSD mixture sits near a stationary point at beta=0.5: a wider spread there
# drives |grad_input| to ~4e-6, below float16's smallest normal (6.1e-5), so output quantization
# rather than the projection would dominate the error. A wider spread at beta=0.5 also makes the
# mixture ``lerp(exp(log_q), exp(log_p), beta)`` underflow to zero and the loss go NaN -- in the
# TorchJSD reference above as much as in the kernel, so it is not something this test can pin on
# the projection.
_PROJECTION_LOGIT_STD = {0.0: 30.0, 0.5: 10.0, 1.0: 30.0}


@pytest.mark.skipif(
    not torch.cuda.is_available() or Version(torch.__version__.split("+")[0]) < Version("2.8.0"),
    reason="the float32 projection needs torch>=2.8 out_dtype support on CUDA sm_80+",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_logits_projection_is_not_rounded_to_input_dtype(dtype, beta):
    """Test that the fused projection keeps float32 logits instead of rounding them first.

    Regression test for the projection matmul running in the input dtype: its output was rounded to
    8 (bfloat16) or 11 (float16) mantissa bits before the float32 cast could preserve anything. The
    JSD scalar hides this because it is dominated by large terms, but ``dx`` is a difference of
    nearly-equal quantities, so the rounding cancels catastrophically. Before the fix the gradients
    were off by 1.3-23%; after it they land within 0.7%, which is where a float32 projection sits.

    The existing ``test_correctness`` cases cannot catch this: they draw both operands from
    ``torch.rand``, which yields all-positive correlated inputs whose logits are tightly clustered.
    Scaling a ``randn`` weight to a realistic logit spread is what exposes it.
    """
    if dtype is torch.float16 and beta == 0.5:
        pytest.skip(
            "at beta=0.5 the reference gradient peaks at ~4e-6, below float16's smallest normal, so "
            "float16 quantization of the output buffer dominates and the projection cannot be isolated"
        )

    BT, H, V = 256, 1024, 32000
    temperature = 1.0
    logit_std = _PROJECTION_LOGIT_STD[beta]
    torch.manual_seed(0)

    student_input = torch.randn(BT, H, device=device, dtype=dtype)
    teacher_input = torch.randn(BT, H, device=device, dtype=dtype)
    student_weight = (torch.randn(V, H, device=device) * (logit_std / H**0.5)).to(dtype)
    teacher_weight = (torch.randn(V, H, device=device) * (logit_std / H**0.5)).to(dtype)

    liger_input = student_input.detach().clone().requires_grad_(True)
    liger_weight = student_weight.detach().clone().requires_grad_(True)
    LigerFusedLinearJSD(jsd_beta=beta, temperature=temperature)(
        liger_input, liger_weight, teacher_input, teacher_weight, None
    ).backward()

    ref_input = student_input.detach().clone().requires_grad_(True)
    ref_weight = student_weight.detach().clone().requires_grad_(True)
    _float32_projection_reference(ref_input, ref_weight, teacher_input, teacher_weight, beta, temperature).backward()

    for name, actual, expected in (
        ("grad_input", liger_input.grad, ref_input.grad),
        ("grad_weight", liger_weight.grad, ref_weight.grad),
    ):
        assert torch.isfinite(actual).all(), f"{name} has non-finite values"
        relative_error = (
            (actual.float() - expected.float()).abs().max() / expected.float().abs().max().clamp_min(1e-12)
        ).item()
        assert relative_error < 0.01, (
            f"{name} deviates {relative_error:.2%} from the float32-projection reference, "
            f"which means the projection rounded its logits to {dtype}"
        )
