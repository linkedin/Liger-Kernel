import pytest
from liger_kernel.transformers.kl_div import LigerKLDIVLoss
from torch.nn import KLDivLoss
from test.utils import supports_bfloat16, assert_verbose_allclose
import torch


def _test_correctness_once(target_kldiv, B, T, V, dtype, atol, rtol):
    torch.manual_seed(0)
    torch_kldiv = KLDivLoss(reduction="batchmean")

    input = torch.randn(
        B * T, V, device="cuda", dtype=dtype, requires_grad=True
    ).log_softmax(dim=-1)

    x1 = input.detach().clone().requires_grad_(True)
    x2 = input.detach().clone().requires_grad_(True)

    with torch.no_grad():
        target = torch.randn(B * T, V, device="cuda").softmax(dim=-1)

    output = torch_kldiv(x1, target)
    output2 = target_kldiv(x2, target)
    assert_verbose_allclose(output, output2, atol=atol, rtol=rtol)

    output.backward()
    output2.backward()
    assert_verbose_allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "B, T, V",
    [
        (1, 4096, 32000),
        (1, 4096, 128256),
        (3, 423, 32000),
    ],
)
@pytest.mark.parametrize(
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
        pytest.param(
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        pytest.param(
            torch.bfloat16,
            1e-7,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        (torch.float32, 1e-8, 1e-6),
        (torch.float32, 1e-8, 1e-6),
        (torch.float32, 1e-8, 1e-6),
    ],
)
@pytest.mark.skipif(
    # TODO: Check what is the peak memory usage to determine the condition
    torch.cuda.get_device_properties(0).total_memory < 16 * 1000 * 1000 * 1000,
    reason="Needs 16GB+ GPU memory.",
)
def test_correctness(B, T, V, dtype, atol, rtol):
    liger_kldiv = LigerKLDIVLoss(reduction="batchmean")
    _test_correctness_once(liger_kldiv, B, T, V, dtype, atol, rtol)
