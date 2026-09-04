import pytest
import torch

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
        (1, 4, 8, 2048),  # forces chunk_size == 1, i.e. one kernel launch per token
        (2, 64, 128, 4096),
    ],
)
@pytest.mark.parametrize(
    "temperature, beta, ignore_index",
    [
        (1.0, 0.0, -100),
        (1.0, 0.5, -100),
        (2.0, 0.1, 42),
        (1.0, 1.0, 2),
    ],
)
@torch.no_grad()
def test_correctness_with_torch_compile(B, T, H, V, beta, ignore_index, temperature):
    """`torch.compile` must not change the loss when some labels are `ignore_index`.

    Regression test for silent numerical corruption: the per-chunk loss buffer used to be a
    `loss_1d[start_idx:end_idx]` view with a non-zero storage offset. `torch.compile` functionalizes
    the kernel's mutation of it by cloning `storage_offset + numel` elements out of a buffer
    Inductor may size to the view alone, i.e. an out-of-bounds read. That garbage was normally
    overwritten by the kernel, but on an `ignore_index` row `_jsd_kernel` returns early without
    writing `loss_ptr`, so the garbage survived into the reduced loss.

    Forward-only: `fused_linear_jsd_backward` branches on a tensor value, so the backward pass
    cannot be captured under `torch.compile` in one graph.

    Uses plain `torch.compile` rather than `fullgraph=True`: `fused_linear_jsd_forward` now reaches
    `_jsd_kernel` through `liger_kernel.backends.dispatch`, which is decorated
    `@torch.compiler.disable`, so requesting a full graph raises
    `Unsupported: Skip calling torch.compiler.disable()d function`. That graph break also keeps the
    Triton kernel out of Inductor on the default path, so this asserts the eager/compiled contract
    rather than actively reproducing the original over-read; the kernel-side zeroing in
    `_jsd_kernel` is what keeps that contract true if the kernel is ever lowered again.
    """
    student_weight = torch.rand(V, H // 2, device=device, dtype=torch.float32)
    teacher_weight = torch.rand(V, H, device=device, dtype=torch.float32)
    student_input = torch.rand(B * T, H // 2, device=device, dtype=torch.float32)
    teacher_input = torch.rand(B * T, H, device=device, dtype=torch.float32)

    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)
    # at least one ignored label, but not all of them
    label[torch.randperm(B * T)[: max(1, B * T // 4)]] = ignore_index

    def call(student_input, teacher_input, label):
        return liger_fused_linear_jsd(
            student_input=student_input,
            student_weight=student_weight,
            teacher_input=teacher_input,
            teacher_weight=teacher_weight,
            shift_labels=label,
            jsd_beta=beta,
            ignore_index=ignore_index,
            temperature=temperature,
        )

    expected = call(student_input, teacher_input, label)

    torch._dynamo.reset()
    actual = torch.compile(call)(student_input, teacher_input, label)

    # Identical kernels on identical inputs: the only permitted difference is reduction ordering.
    assert_verbose_allclose(actual, expected, atol=1e-5, rtol=1e-5)


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
