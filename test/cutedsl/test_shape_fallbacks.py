"""Explicit Triton-fallback coverage, not native CuTe DSL correctness.

Odd widths hit vector-alignment guards. Separate aligned widths exercise the
size/training guards without launching an unsupported native kernel. All cases
assert the fallback reason before checking the forwarded results and gradients.
"""

import pytest
import torch

import liger_kernel.functional  # noqa: F401

from liger_kernel.backends import available_impls
from liger_kernel.backends import dispatch
from test.cutedsl.test_gap_ops import _cutedsl_available
from test.cutedsl.test_gap_ops import _dispatch_cutedsl_detecting_fallback


def _requires_cutedsl(op_name):
    if not torch.cuda.is_available():
        pytest.skip("CuTe DSL shape-fallback tests require CUDA")
    if not _cutedsl_available():
        pytest.skip("CuTe DSL shape-fallback tests require cutlass.cute and sm_90+")
    assert "nvidia-cutedsl" in available_impls(op_name), f"CuTe DSL {op_name} failed registration on a capable host"


def _dispatch_fallback(op, *args, reason):
    output, fell_back = _dispatch_cutedsl_detecting_fallback(op, *args, fallback_reason=reason)
    assert fell_back, f"{op}: expected a Triton fallback"
    return output


def _assert_gradient_pairs(*pairs):
    for actual, expected in pairs:
        for tensor in (actual, expected):
            assert tensor.grad is not None, "missing gradient"
            assert tensor.grad.shape == tensor.shape and tensor.grad.dtype == tensor.dtype
            assert torch.isfinite(tensor.grad).all()
            assert torch.count_nonzero(tensor.grad) > 0, "all-zero gradient"
        torch.testing.assert_close(actual.grad, expected.grad)


@pytest.mark.parametrize(
    "dtype,width",
    [
        (torch.bfloat16, 769),
        (torch.float16, 769),
        (torch.float32, 769),
        (torch.bfloat16, 50257),
        (torch.bfloat16, 32776),
    ],
)
def test_softmax_cutedsl_shape_fallback_matches_triton(dtype, width):
    _requires_cutedsl("softmax")
    x = torch.randn(2, width, device="cuda", dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    reason = "not divisible by vector width" if width % (16 // x.element_size()) else "exceeds CuTe DSL limit"
    actual = _dispatch_fallback("softmax", x, reason=reason)
    expected = dispatch("softmax", x_ref, impl="nvidia-triton")
    torch.testing.assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad.clone())
    expected.backward(grad.clone())
    _assert_gradient_pairs((x, x_ref))


@pytest.mark.parametrize("dtype,width", [(torch.bfloat16, 128257), (torch.bfloat16, 32776), (torch.float32, 28676)])
def test_kl_div_cutedsl_large_vocab_fallback_matches_triton(dtype, width):
    _requires_cutedsl("kl_div")
    y_pred = torch.log_softmax(
        torch.randn(2, width, device="cuda", dtype=dtype),
        dim=-1,
    ).requires_grad_(True)
    y_true = torch.softmax(torch.randn_like(y_pred), dim=-1)
    y_pred_ref = y_pred.detach().clone().requires_grad_(True)
    y_true_ref = y_true.detach().clone()

    actual = _dispatch_fallback(
        "kl_div", y_pred, y_true, "batchmean", False, 1e-10, reason="exceeds CuTe DSL fwd limit"
    )
    expected = dispatch("kl_div", y_pred_ref, y_true_ref, "batchmean", False, 1e-10, impl="nvidia-triton")
    torch.testing.assert_close(actual, expected)

    actual.backward()
    expected.backward()
    _assert_gradient_pairs((y_pred, y_pred_ref))


@pytest.mark.parametrize("width", [769, 32769, 32776])
def test_rms_norm_cutedsl_shape_fallback_matches_triton(width):
    _requires_cutedsl("rms_norm")
    x = torch.randn(2, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)

    reason = "not divisible by vector width" if width % 8 else "exceeds CuTe DSL limit"
    actual = _dispatch_fallback("rms_norm", x, weight, 1e-6, 0.0, "llama", False, None, reason=reason)
    expected = dispatch(
        "rms_norm",
        x_ref,
        weight_ref,
        1e-6,
        0.0,
        "llama",
        False,
        None,
        impl="nvidia-triton",
    )
    torch.testing.assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad.clone())
    expected.backward(grad.clone())
    _assert_gradient_pairs((x, x_ref), (weight, weight_ref))


@pytest.mark.parametrize("width", [769, 32769, 8200])
def test_layer_norm_cutedsl_shape_fallback_matches_triton(width):
    _requires_cutedsl("layer_norm")
    x = torch.randn(2, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)

    reason = "not divisible by vector width" if width % 8 else "exceeds CuTe DSL limit"
    actual = _dispatch_fallback("layer_norm", x, weight, bias, 1e-6, reason=reason)
    expected = dispatch("layer_norm", x_ref, weight_ref, bias_ref, 1e-6, impl="nvidia-triton")
    torch.testing.assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad.clone())
    expected.backward(grad.clone())
    _assert_gradient_pairs((x, x_ref), (weight, weight_ref), (bias, bias_ref))


@pytest.mark.parametrize("width", [769, 32769, 32776])
def test_fused_add_rms_norm_cutedsl_shape_fallback_matches_triton(width):
    _requires_cutedsl("fused_add_rms_norm")
    x = torch.randn(2, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn_like(x, requires_grad=True)
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    residual_ref = residual.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)

    reason = "not divisible by vector width" if width % 8 else "exceeds B200 fwd+bwd crossover"
    actual = _dispatch_fallback(
        "fused_add_rms_norm",
        x,
        residual,
        weight,
        1e-6,
        0.0,
        "llama",
        False,
        reason=reason,
    )
    expected = dispatch(
        "fused_add_rms_norm",
        x_ref,
        residual_ref,
        weight_ref,
        1e-6,
        0.0,
        "llama",
        False,
        impl="nvidia-triton",
    )
    assert len(actual) == len(expected) == 2
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])

    grad_y = torch.randn_like(actual[0])
    grad_residual = torch.randn_like(actual[1])
    torch.autograd.backward(actual, (grad_y.clone(), grad_residual.clone()))
    torch.autograd.backward(expected, (grad_y.clone(), grad_residual.clone()))
    _assert_gradient_pairs((x, x_ref), (residual, residual_ref), (weight, weight_ref))
