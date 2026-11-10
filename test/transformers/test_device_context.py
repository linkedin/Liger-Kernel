import pytest
import torch

from liger_kernel.transformers.functional import liger_geglu
from liger_kernel.transformers.functional import liger_rms_norm
from liger_kernel.transformers.functional import liger_swiglu
from liger_kernel.utils import infer_device

device = infer_device()


def _gated_inputs(dtype, device=device):
    a = torch.randn(4, 64, 512, device=device, dtype=dtype)
    return a, torch.randn_like(a)


def _rms_norm_inputs(dtype, device=device):
    return torch.randn(4, 64, 512, device=device, dtype=dtype), torch.randn(512, device=device, dtype=dtype)


OPS = {
    "swiglu": (liger_swiglu, _gated_inputs),
    "geglu": (liger_geglu, _gated_inputs),
    "rms_norm": (lambda x, w: liger_rms_norm(x, w, 1e-6, in_place=False), _rms_norm_inputs),
}


def _run(fn, inputs):
    # Fresh leaves per call: the SwiGLU/GeGLU backward writes its gradients into the saved inputs.
    leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
    out = fn(*leaves)
    out.backward(torch.ones_like(out))
    return out.detach(), [t.grad for t in leaves]


@pytest.mark.parametrize("name", list(OPS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_torch_compile_matches_eager(name, dtype):
    torch._dynamo.reset()
    fn, make_inputs = OPS[name]
    inputs = make_inputs(dtype)

    eager_out, eager_grads = _run(fn, inputs)
    compiled_out, compiled_grads = _run(torch.compile(fn), inputs)

    torch.testing.assert_close(compiled_out, eager_out)
    for compiled_grad, eager_grad in zip(compiled_grads, eager_grads):
        torch.testing.assert_close(compiled_grad, eager_grad)


@pytest.mark.skipif(device != "cuda" or torch.cuda.device_count() < 2, reason="needs two CUDA devices")
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize("name", list(OPS))
def test_launch_on_non_current_device(name, compiled):
    torch._dynamo.reset()
    fn, make_inputs = OPS[name]
    inputs = make_inputs(torch.float32, device="cuda:1")

    with torch.cuda.device(1):
        expected_out, expected_grads = _run(fn, inputs)
    with torch.cuda.device(0):
        out, grads = _run(torch.compile(fn) if compiled else fn, inputs)

    assert out.device == torch.device("cuda:1")
    torch.testing.assert_close(out, expected_out)
    for grad, expected_grad in zip(grads, expected_grads):
        torch.testing.assert_close(grad, expected_grad)
