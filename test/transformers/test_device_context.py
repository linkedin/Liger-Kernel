import pytest
import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import device_context
from liger_kernel.utils import infer_device

device = infer_device()


@triton.jit
def _scale_rows_kernel(x_ptr, y_ptr, stride, scale, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride + cols, mask=mask)
    tl.store(y_ptr + row * stride + cols, x * scale, mask=mask)


def _scale_rows(x, scale):
    n_cols = x.shape[-1]
    x_2d = x.contiguous().view(-1, n_cols)
    y = torch.empty_like(x_2d)
    # Launched like the Liger ops: the grid and the row stride are read from the tensors
    # inside device_context. In backward these are symbolic under torch.compile.
    with device_context(x.device):
        _scale_rows_kernel[(x_2d.shape[0],)](
            x_2d,
            y,
            y.stride(-2),
            scale,
            n_cols=n_cols,
            BLOCK_SIZE=triton.next_power_of_2(n_cols),
        )
    return y.view(x.shape)


class _ScaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return _scale_rows(x, scale)

    @staticmethod
    def backward(ctx, grad_output):
        return _scale_rows(grad_output, ctx.scale), None


def _fn(x):
    return _ScaleFunction.apply(x, 3.0)


def _run(fn, x, grad_output):
    x = x.detach().clone().requires_grad_(True)
    out = fn(x)
    out.backward(grad_output)
    return out.detach(), x.grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_device_context_under_torch_compile(dtype):
    torch._dynamo.reset()
    x = torch.randn(4, 64, 512, device=device, dtype=dtype)
    grad_output = torch.randn_like(x)

    eager_out, eager_grad = _run(_fn, x, grad_output)
    compiled_out, compiled_grad = _run(torch.compile(_fn, fullgraph=True), x, grad_output)

    torch.testing.assert_close(compiled_out, eager_out, atol=0, rtol=0)
    torch.testing.assert_close(compiled_grad, eager_grad, atol=0, rtol=0)


@pytest.mark.skipif(device != "cuda" or torch.cuda.device_count() < 2, reason="needs two CUDA devices")
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_device_context_on_non_current_device(compiled):
    torch._dynamo.reset()
    x = torch.randn(4, 64, 512, device="cuda:1")
    grad_output = torch.randn_like(x)

    with torch.cuda.device(1):
        expected_out, expected_grad = _run(_fn, x, grad_output)
    with torch.cuda.device(0):
        out, grad = _run(torch.compile(_fn, fullgraph=True) if compiled else _fn, x, grad_output)

    assert out.device == torch.device("cuda:1")
    torch.testing.assert_close(out, expected_out, atol=0, rtol=0)
    torch.testing.assert_close(grad, expected_grad, atol=0, rtol=0)
