import pytest
import torch

from test.utils import assert_verbose_allclose
from test.utils import set_seed

from liger_kernel.ops.sparsemax import _sparsemax_backward_kernel
from liger_kernel.ops.sparsemax import _sparsemax_forward_kernel
from liger_kernel.ops.utils import calculate_settings
from liger_kernel.transformers.functional import liger_sparsemax
from liger_kernel.transformers.sparsemax import LigerSparsemax
from liger_kernel.utils import infer_device

device = infer_device()

_SPARSEMAX_OVERFLOW_ROWS = 2**16 + 16
_SPARSEMAX_OVERFLOW_COLS = 2**15
# forward peak: bf16 input + fp32 sorted input + bf16 output
_SPARSEMAX_MIN_FREE_BYTES = (2 + 4 + 2) * _SPARSEMAX_OVERFLOW_ROWS * _SPARSEMAX_OVERFLOW_COLS


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


def test_sparsemax_int32_row_offset_wraps():
    """The first overflowing row offset wraps in int32."""
    row = torch.tensor(2**16, dtype=torch.int32)
    row_stride = torch.tensor(_SPARSEMAX_OVERFLOW_COLS, dtype=torch.int32)

    assert (row * row_stride).item() == -(2**31)
    assert row.to(torch.int64).mul(row_stride).item() == 2**31


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < _SPARSEMAX_MIN_FREE_BYTES,
    reason="requires 17.2 GB of free CUDA memory for rows beyond the int32 offset range",
)
def test_sparsemax_large_row_offset():
    """Forward and backward kernels must address rows whose offset exceeds 2^31 elements.

    The kernels are launched directly: at this size the torch.sort in the autograd
    path would need ~26 GB more memory for the sorted values and int64 indices.
    """
    n_rows, n_cols = _SPARSEMAX_OVERFLOW_ROWS, _SPARSEMAX_OVERFLOW_COLS
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    last = torch.randn(n_cols, dtype=torch.bfloat16, device="cuda")
    x = torch.zeros(n_rows, n_cols, dtype=torch.bfloat16, device="cuda")
    x[-1] = last
    x_sorted = torch.zeros(n_rows, n_cols, dtype=torch.float32, device="cuda")
    x_sorted[-1] = torch.sort(last.float(), descending=True).values
    out = torch.empty_like(x)
    _sparsemax_forward_kernel[(n_rows,)](
        x,
        x.stride(0),
        x_sorted,
        x_sorted.stride(0),
        out,
        out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    del x, x_sorted

    x_ref = last.float().requires_grad_(True)
    out_ref = torch_sparsemax(x_ref)
    assert_verbose_allclose(out[-1].float(), out_ref, atol=5e-2, rtol=5e-2)

    grad_out = torch.zeros_like(out)
    grad_out[-1] = torch.randn(n_cols, dtype=torch.bfloat16, device="cuda")
    grad_in = torch.empty_like(out)
    _sparsemax_backward_kernel[(n_rows,)](
        out,
        grad_out,
        grad_in,
        out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    out_ref.backward(grad_out[-1].float())
    assert_verbose_allclose(grad_in[-1].float(), x_ref.grad, atol=5e-2, rtol=5e-2)
