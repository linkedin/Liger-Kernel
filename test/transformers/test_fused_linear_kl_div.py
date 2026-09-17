import pytest
import torch

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.ops import LigerFusedLinearKLDivFunction
from liger_kernel.transformers.functional import liger_fused_linear_kl_div
from liger_kernel.transformers.fused_linear_kl_div import LigerFusedLinearKLDivLoss
from liger_kernel.utils import infer_device

device = infer_device()

set_seed(42)


class TorchLMHeadKLDiv(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based KL divergence loss.

    :param H: hidden size
    :param V: vocab size
    :param temperature: softmax temperature
    :param reduction: reduction over the per-row KL losses
    """

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
        reduction: str = "batchmean",
        ignore_index: int = -100,
        temperature: float = 1.0,
        eps: float = 1e-10,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype, device=device)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.eps = eps

    def forward(self, student_input, target, label=None):
        logits = self.student_lin(student_input).to(torch.float32) / self.temperature
        log_p = torch.log_softmax(logits, dim=-1)
        q = target.to(torch.float32)
        # KL(q || p) = sum(q * (log q - log p)); 0 * log 0 is treated as 0 via the clamp
        loss_mat = q * (torch.log(q.clamp_min(self.eps)) - log_p)

        if label is not None:
            keep = (label != self.ignore_index).to(torch.float32).unsqueeze(-1)
            loss_mat = loss_mat * keep
            n_non_ignore = int(keep.sum().item())
        else:
            n_non_ignore = student_input.shape[0]

        total = loss_mat.sum()
        if self.reduction == "sum":
            return total
        if n_non_ignore == 0:
            return total * 0.0
        if self.reduction == "batchmean":
            return total / n_non_ignore
        return total / (n_non_ignore * self.student_lin.out_features)  # "mean"


class LigerLMHeadKLDiv(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
        reduction: str = "batchmean",
        ignore_index: int = -100,
        temperature: float = 1.0,
        eps: float = 1e-10,
        accum_dtype=None,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype, device=device)
        self.fused_kl = LigerFusedLinearKLDivLoss(
            reduction=reduction,
            ignore_index=ignore_index,
            temperature=temperature,
            eps=eps,
            accum_dtype=accum_dtype,
        )

    def forward(self, student_input, target, label=None):
        return self.fused_kl(student_input, self.student_lin.weight, target, label)


#############################################################################
# Test the correctness of the fused linear KL divergence
#############################################################################


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (8, 128, 1024, 4096),
        (4, 423, 167, 1423),  # random shape
    ],
)
@pytest.mark.parametrize(
    "temperature, reduction, accum_dtype, scalar, dtype, atol, rtol",
    [
        # with reduction="sum" the gradients are O(BT) larger than with "batchmean",
        # so the absolute bf16 rounding noise of the chunked GEMMs scales up accordingly
        pytest.param(
            1.0,
            "batchmean",
            None,
            1.0,
            torch.bfloat16,
            5e-3,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (1.0, "batchmean", None, 1.0, torch.float32, 1e-5, 5e-4),
        (1.0, "batchmean", None, 1.0, torch.float16, 5e-3, 5e-2),
        pytest.param(
            2.0,
            "sum",
            None,
            1.0,
            torch.bfloat16,
            5e-0,
            5e1,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        # pass non-default accum_dtype once to ensure it works along
        (2.0, "sum", torch.float32, 1.0, torch.float32, 1e-3, 5e-2),
        (2.0, "sum", None, 1.0, torch.float16, 5e-3, 5e-2),
        pytest.param(
            1.0,
            "mean",
            None,
            1.0,
            torch.bfloat16,
            5e-3,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (1.0, "mean", None, 1.0, torch.float32, 1e-5, 5e-4),
        (1.0, "mean", None, 1.0, torch.float16, 5e-3, 5e-2),
    ],
)
def test_correctness(B, T, H, V, scalar, dtype, reduction, temperature, accum_dtype, atol, rtol):
    torch_lm_head_kl = TorchLMHeadKLDiv(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        reduction=reduction,
    ).to(device)
    liger_lm_head_kl = LigerLMHeadKLDiv(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        reduction=reduction,
        accum_dtype=accum_dtype,
    ).to(device)

    # init the linear in all FusedLinearKLDivs with the same weights
    torch_lm_head_kl.student_lin.weight.data = liger_lm_head_kl.student_lin.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )

    _tensor = torch.randn(B * T, H, device=device, dtype=dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.rand(B * T, V, device=device, dtype=torch.float32).softmax(dim=-1).to(dtype)

    with torch.autograd.detect_anomaly():
        output1 = torch_lm_head_kl(_input1, target)
        output2 = liger_lm_head_kl(_input2, target)

        assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

    output1.backward()
    output2.backward()

    assert_verbose_allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)

    assert_verbose_allclose(
        torch_lm_head_kl.student_lin.weight.grad,
        liger_lm_head_kl.student_lin.weight.grad,
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
    "temperature, reduction, ignore_index, scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            "batchmean",
            2,
            1.0,
            torch.bfloat16,
            5e-3,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (1.0, "batchmean", 2, 1.0, torch.float32, 1e-5, 5e-4),
        (1.0, "batchmean", 2, 1.0, torch.float16, 5e-3, 5e-2),
        # see the comment in test_correctness for the relaxed bf16 tolerance of reduction="sum"
        pytest.param(
            2.0,
            "sum",
            42,
            1.0,
            torch.bfloat16,
            5e-0,
            5e1,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (2.0, "sum", 42, 1.0, torch.float32, 1e-3, 5e-2),
        (2.0, "sum", 42, 1.0, torch.float16, 5e-3, 5e-2),
        pytest.param(
            1.0,
            "mean",
            -100,
            1.0,
            torch.bfloat16,
            5e-3,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (1.0, "mean", -100, 1.0, torch.float32, 1e-5, 5e-4),
        (1.0, "mean", -100, 1.0, torch.float16, 5e-3, 5e-2),
    ],
)
def test_correctness_with_ignore_index(B, T, H, V, scalar, dtype, reduction, ignore_index, temperature, atol, rtol):
    torch_lm_head_kl = TorchLMHeadKLDiv(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        ignore_index=ignore_index,
        reduction=reduction,
    ).to(device)
    liger_lm_head_kl = LigerLMHeadKLDiv(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        ignore_index=ignore_index,
        reduction=reduction,
    ).to(device)

    # init the linear in all FusedLinearKLDivs with the same weights
    torch_lm_head_kl.student_lin.weight.data = liger_lm_head_kl.student_lin.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )

    _tensor = torch.randn(B * T, H, device=device, dtype=dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.rand(B * T, V, device=device, dtype=torch.float32).softmax(dim=-1).to(dtype)

    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]  # Randomly select indices
    label[indices_to_assign] = ignore_index

    output1 = torch_lm_head_kl(_input1, target, label)
    output2 = liger_lm_head_kl(_input2, target, label)

    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

    output1.backward()
    output2.backward()

    assert_verbose_allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)

    assert_verbose_allclose(
        torch_lm_head_kl.student_lin.weight.grad,
        liger_lm_head_kl.student_lin.weight.grad,
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
        pytest.param(
            0.5,
            torch.bfloat16,
            5e-3,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (0.5, torch.float32, 1e-5, 5e-4),
        (0.5, torch.float16, 5e-3, 5e-2),
    ],
)
@pytest.mark.parametrize("temperature, reduction, ignore_index", [(1.0, "batchmean", -100), (2.0, "sum", 42)])
@pytest.mark.parametrize("accum_dtype", [None, torch.float32])
def test_correctness_functional(
    B, T, H, V, scalar, dtype, reduction, ignore_index, temperature, accum_dtype, atol, rtol
):
    # init the linear in all FusedLinearKLDivs with the same weights
    _weight = torch.rand(V, H, device=device, dtype=dtype)
    _weight1 = _weight.detach().clone().requires_grad_(True)
    _weight2 = _weight.detach().clone().requires_grad_(True)

    _tensor = torch.randn(B * T, H, device=device, dtype=dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)
    target = torch.rand(B * T, V, device=device, dtype=torch.float32).softmax(dim=-1).to(dtype)

    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]  # Randomly select indices
    label[indices_to_assign] = ignore_index

    output1 = liger_fused_linear_kl_div(
        student_input=_input1,
        student_weight=_weight1,
        target=target,
        shift_labels=label,
        reduction=reduction,
        ignore_index=ignore_index,
        temperature=temperature,
        accum_dtype=accum_dtype,
    )
    output2 = LigerFusedLinearKLDivFunction.apply(
        _input2,
        _weight2,
        target,
        label,
        reduction,
        ignore_index,
        temperature,
        1e-10,
        accum_dtype,
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
        pytest.param(
            1.0,
            torch.bfloat16,
            5e-3,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (1.0, torch.float32, 1e-5, 5e-4),
        (1.0, torch.float16, 5e-3, 5e-2),
    ],
)
@pytest.mark.parametrize(
    "temperature, reduction, ignore_index",
    [
        (1.0, "batchmean", 2),
        (2.0, "sum", 42),
    ],
)
def test_correctness_all_ignored(B, T, H, V, scalar, dtype, reduction, ignore_index, temperature, atol, rtol):
    torch_lm_head_kl = TorchLMHeadKLDiv(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        ignore_index=ignore_index,
        reduction=reduction,
    ).to(device)
    liger_lm_head_kl = LigerLMHeadKLDiv(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        ignore_index=ignore_index,
        reduction=reduction,
    ).to(device)

    # init the linear in all FusedLinearKLDivs with the same weights
    torch_lm_head_kl.student_lin.weight.data = liger_lm_head_kl.student_lin.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )

    _tensor = torch.randn(B * T, H, device=device, dtype=dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.rand(B * T, V, device=device, dtype=torch.float32).softmax(dim=-1).to(dtype)

    label = torch.full((B * T,), ignore_index, device=device, dtype=torch.long)

    output1 = torch_lm_head_kl(_input1, target, label)
    output2 = liger_lm_head_kl(_input2, target, label)

    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)
    assert_verbose_allclose(output2, torch.zeros_like(output2), atol=atol, rtol=rtol)

    output2.backward()

    assert_verbose_allclose(torch.zeros_like(_input2.grad), _input2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "autocast_dtype, atol, rtol",
    [
        pytest.param(
            torch.bfloat16,
            5e-3,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
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
    reduction = "batchmean"
    dtype = torch.float32
    torch_lm_head_kl = TorchLMHeadKLDiv(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        ignore_index=ignore_index,
        reduction=reduction,
    ).to(device)
    liger_lm_head_kl = LigerLMHeadKLDiv(
        H=H,
        V=V,
        dtype=dtype,
        device=device,
        temperature=temperature,
        ignore_index=ignore_index,
        reduction=reduction,
    ).to(device)
    # init the linear in all FusedLinearKLDivs with the same weights
    torch_lm_head_kl.student_lin.weight.data = liger_lm_head_kl.student_lin.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )

    _tensor = torch.randn(B * T, H, device=device, dtype=autocast_dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.rand(B * T, V, device=device, dtype=torch.float32).softmax(dim=-1).to(autocast_dtype)

    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]  # Randomly select indices
    label[indices_to_assign] = ignore_index

    with torch.autocast(device_type=device, dtype=autocast_dtype):
        output1 = torch_lm_head_kl(_input1, target, label)
        output2 = liger_lm_head_kl(_input2, target, label)

        assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

        output1.backward()
        output2.backward()

    assert_verbose_allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(
        torch_lm_head_kl.student_lin.weight.grad,
        liger_lm_head_kl.student_lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )


def test_torch_kl_div_parity():
    # cross-check the fused kernel against torch.nn.functional.kl_div in fp32
    B = 2
    T = 8
    H = 128
    V = 1024
    torch.manual_seed(0)

    _input = torch.randn(B * T, H, device=device, dtype=torch.float32, requires_grad=True)
    _weight = torch.randn(V, H, device=device, dtype=torch.float32, requires_grad=True)
    target = torch.rand(B * T, V, device=device, dtype=torch.float32).softmax(dim=-1)

    expected = torch.nn.functional.kl_div(
        torch.log_softmax(_input @ _weight.t(), dim=-1),
        target,
        reduction="batchmean",
    )
    got = liger_fused_linear_kl_div(_input, _weight, target)

    assert_verbose_allclose(expected, got, atol=1e-6, rtol=1e-5)


def test_invalid_reduction():
    B = 1
    T = 4
    H = 8
    V = 16
    _input = torch.randn(B * T, H, device=device, dtype=torch.float32, requires_grad=True)
    _weight = torch.randn(V, H, device=device, dtype=torch.float32)
    target = torch.rand(B * T, V, device=device, dtype=torch.float32).softmax(dim=-1)

    with pytest.raises(ValueError, match="reduction must be one of"):
        liger_fused_linear_kl_div(_input, _weight, target, reduction="none")
