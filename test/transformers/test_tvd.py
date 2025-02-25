import pytest
import torch

from test.utils import supports_bfloat16

from liger_kernel.transformers.tvd import LigerTVDLoss
from liger_kernel.utils import infer_device


class TorchTVDLoss(torch.nn.Module):
    def __init__(self, reduction="batchmean", ignore_index: int = -100):
        super(TorchTVDLoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, p, q, label=None):
        tvd = torch.abs(p - q) / 2.0
        n_non_ignore = p.size(0)
        if label is not None:
            tvd = torch.where(label.unsqueeze(1) != self.ignore_index, tvd, torch.zeros_like(tvd))
            n_non_ignore = (label != self.ignore_index).sum().item()
            if n_non_ignore == 0:
                return torch.tensor(0.0).to(tvd.device)

        if self.reduction == "mean":
            return torch.sum(tvd) / (n_non_ignore * p.size(1))
        elif self.reduction == "sum":
            return torch.sum(tvd)
        elif self.reduction == "none":
            return tvd
        elif self.reduction == "batchmean":
            return torch.sum(tvd) / n_non_ignore
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
                hasattr(torch, infer_device())
                and getattr(torch, infer_device()).is_available()
                and getattr(torch, infer_device()).get_device_properties(0).total_memory < 36e9,
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
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (torch.float32, 1e-8, 1e-6),
        # (torch.float16, 1e-1, 1e-2), # turn off because of numerical instability of torch.float16
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
    device=infer_device(),
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


def _test_correctness_with_ignore_index_once(
    target_tvd,
    torch_tvd,
    ignore_index,
    B,
    T,
    V,
    dtype,
    atol,
    rtol,
    reduction,
    device=infer_device(),
):
    input = torch.randn(B * T, V, device=device, dtype=dtype, requires_grad=True)

    x1 = input.detach().clone().requires_grad_(True)
    x2 = input.detach().clone().requires_grad_(True)

    with torch.no_grad():
        target = torch.randn(B * T, V, device=device).softmax(dim=-1)

    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    label[indices_to_assign] = ignore_index

    output = torch_tvd(x1, target, label)
    output2 = target_tvd(x2, target, label)

    assert torch.allclose(output, output2, atol=atol, rtol=rtol)

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


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize("reduction", ["batchmean", "sum", "mean", "none"])
@pytest.mark.parametrize(*_DTYPE_PARAMS)
@pytest.mark.parametrize("ignore_index", [-100, 0, 1])
def test_correctness_with_ignore_index(B, T, V, reduction, dtype, atol, rtol, ignore_index):
    liger_tvd = LigerTVDLoss(reduction=reduction, ignore_index=ignore_index)
    torch_tvd = TorchTVDLoss(reduction=reduction, ignore_index=ignore_index)
    _test_correctness_with_ignore_index_once(liger_tvd, torch_tvd, ignore_index, B, T, V, dtype, atol, rtol, reduction)
