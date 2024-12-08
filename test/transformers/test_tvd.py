from test.utils import supports_bfloat16

import pytest
import torch

from liger_kernel.transformers.tvd import LigerTVDLoss


class TorchTVDLoss(torch.nn.Module):
    def __init__(self, reduction="batchmean"):
        super(TorchTVDLoss, self).__init__()
        self.reduction = reduction

    def forward(self, p, q):

        tvd = torch.abs(p - q) / 2.0

        if self.reduction == "mean":
            return torch.sum(tvd) / (p.size(0) * p.size(1))
        elif self.reduction == "sum":
            return torch.sum(tvd)
        elif self.reduction == "none":
            return tvd
        elif self.reduction == "batchmean":
            return torch.sum(tvd) / p.size(0)
        else:
            raise ValueError("Invalid reduction type.")


_SHAPE_PARAMS = (
    "B, T, V",
    [
        (1, 4096, 32000),
        (32, 4096, 1024),
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
            1e-6,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        (torch.float32, 1e-8, 1e-6),
        (torch.float16, 1e-3, 1e-3),
    ],
)


def _test_correctness_once(
    target_tvd,
    torch_tvd,
    B,
    T,
    V,
    dtype,
    atol,
    rtol,
    reduction,
    is_last_layer=True,
    device="cuda",
):
    torch.manual_seed(0)
    input = torch.randn(B * T, V, device=device, dtype=dtype, requires_grad=True)

    x1 = input.detach().clone().requires_grad_(True)
    x2 = input.detach().clone().requires_grad_(True)

    with torch.no_grad():
        target = torch.randn(B * T, V, device=device).softmax(dim=-1)

    output = target_tvd(x1, target)
    output2 = torch_tvd(x2, target)

    assert torch.allclose(output, output2, atol=atol, rtol=rtol)

    if not is_last_layer:
        output = output * 2.0
        output2 = output2 * 2.0

    if reduction == "none":
        return

    output.backward()
    output2.backward()
    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize("reduction", ["batchmean", "sum", "mean", "none"])
@pytest.mark.parametrize(*_DTYPE_PARAMS)
def test_correctness(B, T, V, reduction, dtype, atol, rtol):
    liger_tvd = LigerTVDLoss(reduction=reduction)
    torch_tvd = TorchTVDLoss(reduction=reduction)
    _test_correctness_once(liger_tvd, torch_tvd, B, T, V, dtype, atol, rtol, reduction)


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize("reduction", ["batchmean", "sum", "mean", "none"])
@pytest.mark.parametrize(*_DTYPE_PARAMS)
def test_correctness_not_last(B, T, V, reduction, dtype, atol, rtol):
    liger_tvd = LigerTVDLoss(reduction=reduction)
    torch_tvd = TorchTVDLoss(reduction=reduction)
    _test_correctness_once(
        liger_tvd,
        torch_tvd,
        B,
        T,
        V,
        dtype,
        atol,
        rtol,
        reduction,
        is_last_layer=False,
    )
