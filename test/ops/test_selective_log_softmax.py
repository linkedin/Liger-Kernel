import pytest
import torch
import torch.nn.functional as F

from test.utils import assert_verbose_allclose

from liger_kernel.ops.selective_log_softmax import liger_selective_log_softmax


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(2, 7, 31), (3, 5, 8193)])
def test_selective_log_softmax_matches_fp32_reference(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA or ROCm GPU")

    logits = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    target = torch.randint(0, shape[-1], shape[:-1], device="cuda")
    reference_logits = logits.detach().clone().requires_grad_(True)

    actual = liger_selective_log_softmax(logits, target)
    expected = F.log_softmax(reference_logits.float(), dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    atol = 2e-3 if dtype != torch.float32 else 1e-5
    rtol = 2e-3 if dtype != torch.float32 else 1e-5
    assert actual.dtype == torch.float32
    assert_verbose_allclose(actual, expected, atol=atol, rtol=rtol)
    assert_verbose_allclose(logits.grad, reference_logits.grad, atol=atol, rtol=rtol)
