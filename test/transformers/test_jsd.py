from test.utils import assert_verbose_allclose, set_seed, supports_bfloat16
from typing import Optional

import pytest
import torch
from torch.nn import KLDivLoss

from liger_kernel.transformers.functional import liger_jsd
from liger_kernel.transformers.jsd import LigerJSD, LigerJSDFunction

set_seed(42)


class JSD(torch.nn.Module):
    def __init__(
        self,
        beta: float = 0.5,
        ignore_index: int = -100,
        dtype: torch.dtype = torch.float,
    ):
        super(JSD, self).__init__()
        self.kl = KLDivLoss(reduction="none", log_target=True)
        self.beta = beta
        self.ignore_index = ignore_index
        self.dtype = dtype

    def forward(
        self,
        log_q: torch.Tensor,  # input
        log_p: torch.Tensor,  # target
        label: Optional[torch.Tensor] = None,
    ):
        log_p, log_q = log_p.to(torch.float), log_q.to(torch.float)
        log_p, log_q = log_p.view(-1, log_p.size(-1)), log_q.view(-1, log_q.size(-1))
        m = torch.lerp(torch.exp(log_q), torch.exp(log_p), self.beta)
        loss = self.beta * self.kl(torch.log(m), log_p).sum(dim=-1) + (
            1 - self.beta
        ) * self.kl(torch.log(m), log_q).sum(dim=-1)

        if label is not None:
            loss = torch.where(label != self.ignore_index, loss, 0.0)
            n_non_ignore = (label != self.ignore_index).sum().item()
            if n_non_ignore == 0:
                loss = 0.0
            else:
                loss = loss.sum() / n_non_ignore
        else:
            loss = loss.sum() / log_q.shape()
        return loss.to(self.dtype)


_SHAPE_PARAMS = (
    "B, T, V",
    [
        (2, 4096, 32000),  # llama2, mistral
        (2, 4096, 32000),  # llama2, mistral
        # weird shape
        (41, 401, 1271),
        pytest.param(
            1,
            4096,
            128256,
            marks=pytest.mark.skipif(
                torch.cuda.get_device_properties(0).total_memory
                < 36 * 1000 * 1000 * 1000,
                reason="This test requires a GPU with at least 36GB of memory",
            ),
        ),
        (3, 423, 32000),
    ],
)

_DTYPE_PARAMS = (
    "dtype, atol, rtol",
    [
        pytest.param(
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        (torch.float32, 1e-8, 1e-6),
        (torch.float16, 1e-3, 1e-3),
    ],
)


def _test_correctness_once(
    target_jsd,
    B,
    T,
    V,
    dtype,
    atol,
    rtol,
    is_last_layer=True,
    device="cuda",
):
    torch_jsd = JSD(dtype=dtype)

    input = torch.randn(
        B * T, V, device=device, dtype=dtype, requires_grad=True
    ).log_softmax(dim=-1)

    x1 = input.detach().clone().requires_grad_(True)
    x2 = input.detach().clone().requires_grad_(True)
    x3 = input.detach().clone().requires_grad_(True)

    with torch.no_grad():
        target = torch.randn(B * T, V, dtype=dtype, device=device).log_softmax(dim=-1)

    output = torch_jsd(x1, target)
    output2 = target_jsd(x2, target)
    assert torch.allclose(output, output2, atol=atol, rtol=rtol)
    # symmetry
    output3 = target_jsd(target, x3)
    assert torch.allclose(output3, output2, atol=atol, rtol=rtol)
    if (
        not is_last_layer
    ):  # if the loss is the last layer, grad_output is 1.0 and mul op is skipped, testing for that reason
        output = output * 2.0
        output2 = output2 * 2.0

    output.backward()
    output2.backward()
    assert_verbose_allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


def _test_correctness_with_beta_once(
    target_jsd,
    beta,
    B,
    T,
    V,
    dtype,
    atol,
    rtol,
    is_last_layer=True,
    device="cuda",
):
    torch_jsd = JSD(beta=beta, dtype=dtype)

    input = torch.randn(
        B * T, V, device=device, dtype=dtype, requires_grad=True
    ).log_softmax(dim=-1)

    x1 = input.detach().clone().requires_grad_(True)
    x2 = input.detach().clone().requires_grad_(True)

    with torch.no_grad():
        target = torch.randn(B * T, V, dtype=dtype, device=device).log_softmax(dim=-1)

    output = torch_jsd(x1, target)
    output2 = target_jsd(x2, target)
    assert_verbose_allclose(output, output2, atol=atol, rtol=rtol)
    if (
        not is_last_layer
    ):  # if the loss is the last layer, grad_output is 1.0 and mul op is skipped, testing for that reason
        output = output * 2.0
        output2 = output2 * 2.0

    output.backward()
    output2.backward()
    assert_verbose_allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


def _test_correctness_with_ignore_index_once(
    target_jsd,
    ignore_index,
    B,
    T,
    V,
    dtype,
    atol,
    rtol,
    device="cuda",
):
    torch_jsd = JSD(ignore_index=ignore_index, dtype=dtype)

    input = torch.randn(
        B * T, V, device=device, dtype=dtype, requires_grad=True
    ).log_softmax(dim=-1)

    x1 = input.detach().clone().requires_grad_(True)
    x2 = input.detach().clone().requires_grad_(True)

    with torch.no_grad():
        target = torch.randn(B * T, V, dtype=dtype, device=device).log_softmax(dim=-1)

    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[
        :num_elements_to_assign
    ]  # Randomly select indices
    label[indices_to_assign] = ignore_index

    output = torch_jsd(x1, target, label)
    output2 = target_jsd(x2, target, label)
    assert_verbose_allclose(output, output2, atol=atol, rtol=rtol)

    output.backward()
    output2.backward()
    assert_verbose_allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


def _test_correctness_functional(
    B, T, V, beta, ignore_index, is_last_layer, dtype, atol, rtol, device="cuda"
):
    input = torch.randn(
        B * T, V, device=device, dtype=dtype, requires_grad=True
    ).log_softmax(dim=-1)

    x1 = input.detach().clone().requires_grad_(True)
    x2 = input.detach().clone().requires_grad_(True)

    with torch.no_grad():
        target = torch.randn(B * T, V, dtype=dtype, device=device).log_softmax(dim=-1)

    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[
        :num_elements_to_assign
    ]  # Randomly select indices
    label[indices_to_assign] = ignore_index

    output = LigerJSDFunction.apply(x1, target, label, beta, ignore_index)
    output2 = liger_jsd(x2, target, label, beta, ignore_index)
    assert torch.allclose(output, output2, atol=atol, rtol=rtol)
    if (
        not is_last_layer
    ):  # if the loss is the last layer, grad_output is 1.0 and mul op is skipped, testing for that reason
        output = output * 2.0
        output2 = output2 * 2.0
    output.backward()
    output2.backward()
    assert_verbose_allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize(*_DTYPE_PARAMS)
def test_correctness(B, T, V, dtype, atol, rtol):
    liger_jsd = LigerJSD()
    _test_correctness_once(liger_jsd, B, T, V, dtype, atol, rtol)


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize(*_DTYPE_PARAMS)
def test_correctness_not_last(B, T, V, dtype, atol, rtol):
    liger_jsd = LigerJSD()

    _test_correctness_once(liger_jsd, B, T, V, dtype, atol, rtol, is_last_layer=False)


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize(*_DTYPE_PARAMS)
@pytest.mark.parametrize("beta", [0.1, 0.5, 0.9])
def test_correctness_with_beta(B, T, V, beta, dtype, atol, rtol):
    liger_jsd = LigerJSD(beta=beta)
    _test_correctness_with_beta_once(liger_jsd, beta, B, T, V, dtype, atol, rtol)


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize(*_DTYPE_PARAMS)
@pytest.mark.parametrize("ignore_index", [2, 42])
def test_correctness_with_ignore_index(B, T, V, ignore_index, dtype, atol, rtol):
    liger_jsd = LigerJSD(ignore_index=ignore_index)
    _test_correctness_with_ignore_index_once(
        liger_jsd, ignore_index, B, T, V, dtype, atol, rtol
    )


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize(*_DTYPE_PARAMS)
@pytest.mark.parametrize(
    "beta, is_last_layer",
    [
        (0.5, False),
        (0.1, True),
    ],
)
def test_correctness_functional(B, T, V, beta, is_last_layer, dtype, atol, rtol):
    _test_correctness_functional(B, T, V, beta, is_last_layer, dtype, atol, rtol)
