import pytest
import torch

from test.utils import assert_verbose_allclose
from test.utils import set_seed

from liger_kernel.transformers.functional import liger_sparsemax
from liger_kernel.transformers.sparsemax import LigerSparsemax
from liger_kernel.utils import infer_device

device = infer_device()


def torch_sparsemax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    input_dims = input_tensor.dim()
    if dim < 0:
        dim = input_dims + dim
    input_sorted, _ = torch.sort(input_tensor, dim=dim, descending=True)
    cumsum_input = torch.cumsum(input_sorted, dim=dim)
    input_size = input_tensor.size(dim)
    range_tensor = torch.arange(1, input_size + 1, device=input_tensor.device, dtype=input_tensor.dtype)
    shape = [1] * input_dims
    shape[dim] = input_size
    range_tensor = range_tensor.view(shape)
    k_bound = 1 + range_tensor * input_sorted
    support = k_bound > cumsum_input
    k = support.sum(dim=dim, keepdim=True).clamp(min=1)
    support_sum = (input_sorted * support).sum(dim=dim, keepdim=True)
    tau = (support_sum - 1) / k
    return torch.clamp(input_tensor - tau, min=0)


@pytest.mark.parametrize(
    "batch_size, seq_len, features",
    [
        (2, 128, 512),
        (5, 123, 123),
    ],
)
@pytest.mark.parametrize("dim", [-1, 1])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [(torch.float32, 1e-5, 1e-5)],
)
def test_liger_sparsemax_correctness(batch_size, seq_len, features, dim, dtype, atol, rtol):
    set_seed(0)
    shape = (batch_size, seq_len, features)
    if dim >= len(shape) or dim < -len(shape):
        pytest.skip("invalid dim")
    if shape[dim if dim >= 0 else len(shape) + dim] <= 1:
        pytest.skip("trivial dim")

    x = torch.randn(*shape, dtype=dtype, device=device)
    lx = x.clone().requires_grad_(True)
    tx = x.clone().requires_grad_(True)

    model = LigerSparsemax(dim=dim).to(device)
    out_l = model(lx)
    out_t = torch_sparsemax(tx, dim=dim)
    assert_verbose_allclose(out_l, out_t, atol=atol, rtol=rtol)

    sum_l = out_l.sum(dim=dim)
    sum_t = out_t.sum(dim=dim)
    assert_verbose_allclose(sum_l, torch.ones_like(sum_l), atol=atol * 10, rtol=rtol * 10)
    assert_verbose_allclose(sum_t, torch.ones_like(sum_t), atol=atol * 10, rtol=rtol * 10)

    g = torch.randn_like(x)
    out_l.backward(g)
    out_t.backward(g)
    assert_verbose_allclose(lx.grad, tx.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "batch_size, seq_len, features",
    [
        (2, 128, 512),
        (5, 123, 123),
    ],
)
@pytest.mark.parametrize("dim", [-1, 1])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
    ],
)
def test_liger_sparsemax_functional_correctness(batch_size, seq_len, features, dim, dtype, atol, rtol):
    set_seed(0)
    shape = (batch_size, seq_len, features)
    if dim >= len(shape) or dim < -len(shape):
        pytest.skip("invalid dim")
    if shape[dim if dim >= 0 else len(shape) + dim] <= 1:
        pytest.skip("trivial dim")

    x = torch.randn(*shape, dtype=dtype, device=device)
    lx = x.clone().requires_grad_(True)
    tx = x.clone().requires_grad_(True)

    out_l = liger_sparsemax(lx, dim=dim)
    out_t = torch_sparsemax(tx, dim=dim)
    assert_verbose_allclose(out_l, out_t, atol=atol, rtol=rtol)

    sum_l = out_l.sum(dim=dim)
    sum_t = out_t.sum(dim=dim)
    assert_verbose_allclose(sum_l, torch.ones_like(sum_l), atol=atol * 10, rtol=rtol * 10)
    assert_verbose_allclose(sum_t, torch.ones_like(sum_t), atol=atol * 10, rtol=rtol * 10)

    g = torch.randn_like(x)
    out_l.backward(g)
    out_t.backward(g)
    assert_verbose_allclose(lx.grad, tx.grad, atol=atol, rtol=rtol)
