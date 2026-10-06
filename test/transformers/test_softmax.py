import pytest
import torch

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.transformers.functional import liger_softmax
from liger_kernel.transformers.softmax import LigerSoftmax
from liger_kernel.utils import infer_device

device = infer_device()
set_seed()

_SOFTMAX_OVERFLOW_ROWS = 2**16 + 16
_SOFTMAX_OVERFLOW_COLS = 2**15
# x, y, grad_output and x.grad in bf16
_SOFTMAX_MIN_FREE_BYTES = 4 * _SOFTMAX_OVERFLOW_ROWS * _SOFTMAX_OVERFLOW_COLS * 2


@pytest.mark.parametrize(
    "shape",
    [
        (2, 8),
        (4, 16),
        (1, 1023),  # Large single row single-block dispatch
        (3, 7, 256),  # 3D input
        (1, 4096),  # test multi-block dispatch
        (1, 2, 4096),  # test multi-block dispatch on 3D input
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            5e-2,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(),
                reason="bfloat16 not supported on this device",
            ),
        ),
    ],
)
def test_liger_softmax(shape, dtype, atol, rtol):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype, device=device)
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)

    torch_softmax = torch.nn.Softmax(dim=-1)
    ref_out = torch_softmax(x1)
    liger_softmax = LigerSoftmax().to(device).to(dtype)
    liger_out = liger_softmax(x2)

    assert_verbose_allclose(ref_out, liger_out, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(ref_out)
    ref_out.backward(grad_output, retain_graph=True)
    liger_out.backward(grad_output, retain_graph=True)

    assert_verbose_allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 8),
        (4, 16),
        (1, 1023),
        (3, 7, 256),
        (1, 4096),
        (1, 2, 4096),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            5e-2,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(),
                reason="bfloat16 not supported on this device",
            ),
        ),
    ],
)
def test_liger_softmax_functional(shape, dtype, atol, rtol):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype, device=device)
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)

    ref_out = torch.nn.functional.softmax(x1, dim=-1)
    liger_out = liger_softmax(x2)

    assert_verbose_allclose(ref_out, liger_out, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(ref_out)
    ref_out.backward(grad_output, retain_graph=True)
    liger_out.backward(grad_output, retain_graph=True)

    assert_verbose_allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


def test_softmax_int32_row_offset_wraps():
    """The first overflowing row offset wraps in int32."""
    row_id = torch.tensor(2**16, dtype=torch.int32)
    row_stride = torch.tensor(_SOFTMAX_OVERFLOW_COLS, dtype=torch.int32)

    assert (row_id * row_stride).item() == -(2**31)
    assert row_id.to(torch.int64).mul(row_stride).item() == 2**31


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < _SOFTMAX_MIN_FREE_BYTES,
    reason="requires 17.2 GB of free CUDA memory for rows beyond the int32 offset range",
)
def test_softmax_large_row_offset():
    """Forward and backward must address rows whose offset exceeds 2^31 elements."""
    x = torch.zeros(_SOFTMAX_OVERFLOW_ROWS, _SOFTMAX_OVERFLOW_COLS, dtype=torch.bfloat16, device="cuda")
    x[-1] = torch.randn(_SOFTMAX_OVERFLOW_COLS, dtype=torch.bfloat16, device="cuda")
    x.requires_grad_(True)

    y = liger_softmax(x)

    grad_output = torch.zeros_like(y)
    grad_output[-1] = torch.randn(_SOFTMAX_OVERFLOW_COLS, dtype=torch.bfloat16, device="cuda")
    y.backward(grad_output)

    x_ref = x[-1].detach().float().requires_grad_(True)
    y_ref = torch.softmax(x_ref, dim=-1)
    y_ref.backward(grad_output[-1].float())

    assert_verbose_allclose(y[-1].float(), y_ref, atol=5e-2, rtol=5e-2)
    assert_verbose_allclose(x.grad[-1].float(), x_ref.grad, atol=5e-2, rtol=5e-2)
