"""Correctness tests for the CuteDSL SwiGLU kernels.

These tests require the optional ``nvidia-cutlass-dsl`` package and an NVIDIA
GPU; they are skipped otherwise. Numerics are checked both against a pure
PyTorch reference and against the existing Triton kernel
(``LigerSiLUMulFunction``).
"""

import importlib.util

import pytest
import torch

from test.utils import supports_bfloat16

from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.utils import infer_device

device = infer_device()

cutedsl_available = importlib.util.find_spec("cutlass") is not None and torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not cutedsl_available,
    reason="nvidia-cutlass-dsl + CUDA GPU required for CuteDSL SwiGLU",
)

if cutedsl_available:
    from liger_kernel.ops.swiglu_cutedsl import LigerSiLUMulCuteDSLFunction
    from liger_kernel.ops.swiglu_cutedsl import swiglu_forward as cutedsl_forward


def _tol(dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-4
    # bf16/fp16: the fast exp2-based sigmoid differs from Triton's sigmoid by up
    # to ~1 ULP on a small fraction of elements.
    return 1e-2, 1e-2


def _torch_silu_mul_ref(a, b, gate_multiplier=1.0, down_multiplier=1.0):
    """Pure-PyTorch reference for silu(a * gate_mult) * b * down_mult."""
    return torch.nn.functional.silu(a * gate_multiplier) * b * down_multiplier


@pytest.mark.parametrize(
    "shape",
    [
        (4096, 11008),  # llama, tile-aligned
        (2, 256, 512),
        (6, 42, 431),  # non-tile-aligned (exercises predicated kernel)
        (1, 1, 7),  # tiny, predicated
        (3, 1023),  # odd number of columns
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        torch.float16,
        torch.float32,
    ],
)
@pytest.mark.parametrize("gate_mult, down_mult", [(1.0, 1.0), (1.5, 0.75)])
def test_cutedsl_matches_triton(shape, dtype, gate_mult, down_mult):
    torch.manual_seed(0)
    atol, rtol = _tol(dtype)
    a = torch.randn(*shape, device=device, dtype=dtype)
    b = torch.randn(*shape, device=device, dtype=dtype)

    a_ref = a.clone().requires_grad_(True)
    b_ref = b.clone().requires_grad_(True)
    a_cut = a.clone().requires_grad_(True)
    b_cut = b.clone().requires_grad_(True)

    c_ref = LigerSiLUMulFunction.apply(a_ref, b_ref, gate_mult, down_mult)
    c_cut = LigerSiLUMulCuteDSLFunction.apply(a_cut, b_cut, gate_mult, down_mult)

    torch.testing.assert_close(c_cut.float(), c_ref.float(), atol=atol, rtol=rtol)

    grad = torch.randn_like(c_ref)
    c_ref.backward(grad)
    c_cut.backward(grad.clone())

    torch.testing.assert_close(a_cut.grad.float(), a_ref.grad.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(b_cut.grad.float(), b_ref.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 8, 1024),
        (9, 7, 41),
    ],
)
@pytest.mark.parametrize(
    "gate_mult, down_mult",
    [
        (1.0, 1.0),
        (0.7, 1.3),
        (1.5, 0.5),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_cutedsl_matches_pytorch(shape, gate_mult, down_mult, dtype):
    """Independent ground-truth check vs pure PyTorch (catches shared CuteDSL/Triton bugs)."""
    torch.manual_seed(0)
    atol, rtol = _tol(dtype)
    a = torch.randn(*shape, device=device, dtype=dtype)
    b = torch.randn(*shape, device=device, dtype=dtype)

    a_ref = a.clone().requires_grad_(True)
    b_ref = b.clone().requires_grad_(True)
    a_cut = a.clone().requires_grad_(True)
    b_cut = b.clone().requires_grad_(True)

    y_ref = _torch_silu_mul_ref(a_ref, b_ref, gate_mult, down_mult)
    y_cut = LigerSiLUMulCuteDSLFunction.apply(a_cut, b_cut, gate_mult, down_mult)

    torch.testing.assert_close(y_cut.float(), y_ref.float(), atol=atol, rtol=rtol)

    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    y_cut.backward(grad.clone())

    torch.testing.assert_close(a_cut.grad.float(), a_ref.grad.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(b_cut.grad.float(), b_ref.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cutedsl_forward_values(dtype):
    """Exercise the lower-level functional ``swiglu_forward`` directly."""
    if dtype == torch.bfloat16 and not supports_bfloat16():
        pytest.skip("bfloat16 not supported on this GPU")
    atol, rtol = _tol(dtype)
    a = torch.randn(128, 512, device=device, dtype=dtype)
    b = torch.randn(128, 512, device=device, dtype=dtype)

    _, _, c = cutedsl_forward(a.clone(), b.clone(), 1.0)
    ref = (torch.nn.functional.silu(a.float()) * b.float()).to(dtype)
    torch.testing.assert_close(c.float(), ref.float(), atol=atol, rtol=rtol)


def test_cutedsl_default_multipliers_backward_compat():
    """Calling without multipliers must equal calling with (1.0, 1.0)."""
    a = torch.randn(4, 16, 32, device=device, dtype=torch.float32)
    b = torch.randn(4, 16, 32, device=device, dtype=torch.float32)

    a1 = a.clone().requires_grad_(True)
    b1 = b.clone().requires_grad_(True)
    a2 = a.clone().requires_grad_(True)
    b2 = b.clone().requires_grad_(True)

    y_default = LigerSiLUMulCuteDSLFunction.apply(a1, b1)
    y_explicit = LigerSiLUMulCuteDSLFunction.apply(a2, b2, 1.0, 1.0)

    torch.testing.assert_close(y_default, y_explicit)

    grad = torch.randn_like(y_default)
    y_default.backward(grad.clone())
    y_explicit.backward(grad.clone())

    torch.testing.assert_close(a1.grad, a2.grad)
    torch.testing.assert_close(b1.grad, b2.grad)


@pytest.mark.parametrize(
    "shape",
    [
        (128 * 4,),  # 1-D, tile-aligned for fp32 (vec=4, tile=512)
        (123,),  # 1-D, unaligned -> predicated
        (3, 5, 7),  # 3-D, unaligned
        (2, 4, 8, 16),  # 4-D, exercises shape-flatten path
    ],
)
def test_cutedsl_shape_flexibility(shape):
    """Forward + backward work for arbitrary >=1-D shapes."""
    dtype = torch.float32
    atol, rtol = _tol(dtype)
    a = torch.randn(*shape, device=device, dtype=dtype)
    b = torch.randn(*shape, device=device, dtype=dtype)

    a_ref = a.clone().requires_grad_(True)
    b_ref = b.clone().requires_grad_(True)
    a_cut = a.clone().requires_grad_(True)
    b_cut = b.clone().requires_grad_(True)

    y_ref = _torch_silu_mul_ref(a_ref, b_ref)
    y_cut = LigerSiLUMulCuteDSLFunction.apply(a_cut, b_cut)

    assert y_cut.shape == y_ref.shape
    torch.testing.assert_close(y_cut, y_ref, atol=atol, rtol=rtol)

    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    y_cut.backward(grad.clone())

    torch.testing.assert_close(a_cut.grad, a_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(b_cut.grad, b_ref.grad, atol=atol, rtol=rtol)
