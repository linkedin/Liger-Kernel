import importlib.util

import pytest
import torch


def _has_cuda_tile():
    try:
        return importlib.util.find_spec("cuda.tile") is not None
    except ModuleNotFoundError:
        return False


_CUDA_TILE_AVAILABLE = _has_cuda_tile()
if _CUDA_TILE_AVAILABLE:
    from liger_kernel.ops.cutile.ops.rms_norm import LigerRMSNormFunction as CuTileRMSNormFunction
else:
    CuTileRMSNormFunction = None

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile RMSNorm requires CUDA"),
    pytest.mark.skipif(not _CUDA_TILE_AVAILABLE, reason="cuda-tile is not installed"),
]

_TOLERANCES = {
    torch.bfloat16: (1e-1, 3e-2),
    torch.float32: (3e-4, 2e-5),
}


def _run_cutile(x, weight, grad_output, casting_mode, in_place):
    x_local = x.clone().detach().requires_grad_(True)
    weight_local = None if weight is None else weight.clone().detach().requires_grad_(True)
    output = CuTileRMSNormFunction.apply(x_local, weight_local, 1e-6, 0.0, casting_mode, in_place, None)
    output.backward(grad_output.clone())
    weight_grad = None if weight_local is None else weight_local.grad
    return output, x_local.grad, weight_grad


def _run_reference(x, weight, grad_output, casting_mode):
    x_local = x.clone().detach().requires_grad_(True)
    weight_local = None if weight is None else weight.clone().detach().requires_grad_(True)
    x_float = x_local.float()
    reciprocal_rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + 1e-6)
    normalized = x_float * reciprocal_rms
    if casting_mode == "llama":
        normalized = normalized.to(x_local.dtype)
    if weight_local is not None:
        normalized = normalized.float() * weight_local.float()
    output = normalized.to(x_local.dtype)
    output.backward(grad_output.clone())
    weight_grad = None if weight_local is None else weight_local.grad
    return output, x_local.grad, weight_grad


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hidden_size", [1000, 8192])
@pytest.mark.parametrize("elementwise_affine", [True, False])
def test_cutile_rms_norm_parity(dtype, hidden_size, elementwise_affine):
    torch.manual_seed(42)
    x = torch.randn(257, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype) if elementwise_affine else None
    grad_output = torch.randn_like(x)

    cutile = _run_cutile(x, weight, grad_output, "llama", False)
    reference = _run_reference(x, weight, grad_output, "llama")
    atol, rtol = _TOLERANCES[dtype]
    for actual, expected in zip(cutile, reference):
        if actual is None:
            assert expected is None
        else:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("casting_mode", ["gemma", "none"])
def test_cutile_rms_norm_casting_modes(casting_mode):
    torch.manual_seed(7)
    x = torch.randn(129, 1024, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn_like(x)

    cutile = _run_cutile(x, weight, grad_output, casting_mode, False)
    reference = _run_reference(x, weight, grad_output, casting_mode)
    for actual, expected in zip(cutile, reference):
        torch.testing.assert_close(actual, expected, atol=2e-1, rtol=3e-2)


def test_cutile_rms_norm_in_place_backward():
    torch.manual_seed(11)
    x = torch.randn(131, 1024, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn_like(x)

    cutile = _run_cutile(x, weight, grad_output, "llama", True)
    reference = _run_reference(x, weight, grad_output, "llama")
    for actual, expected in zip(cutile, reference):
        torch.testing.assert_close(actual, expected, atol=6e-2, rtol=3e-2)
