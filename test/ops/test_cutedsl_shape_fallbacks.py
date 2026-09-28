"""Shape-fallback coverage for CuTe DSL backends."""

import pytest
import torch

import liger_kernel.functional  # noqa: F401

from liger_kernel.backends import available_impls
from liger_kernel.backends import dispatch


def _requires_cutedsl(op_name):
    if "nvidia-cutedsl" not in available_impls(op_name):
        pytest.skip(f"CuTe DSL {op_name} is unavailable")


@pytest.mark.parametrize(
    "dtype,width",
    [
        (torch.bfloat16, 769),
        (torch.float16, 769),
        (torch.float32, 769),
        (torch.bfloat16, 50257),
    ],
)
def test_softmax_cutedsl_shape_fallback_matches_triton(dtype, width):
    _requires_cutedsl("softmax")
    x = torch.randn(2, width, device="cuda", dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    actual = dispatch("softmax", x, impl="nvidia-cutedsl")
    expected = dispatch("softmax", x_ref, impl="nvidia-triton")
    torch.testing.assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(x.grad, x_ref.grad)


def test_kl_div_cutedsl_large_vocab_fallback_matches_triton():
    _requires_cutedsl("kl_div")
    width = 128257
    y_pred = torch.log_softmax(
        torch.randn(2, width, device="cuda", dtype=torch.bfloat16),
        dim=-1,
    ).requires_grad_(True)
    y_true = torch.softmax(torch.randn_like(y_pred), dim=-1)
    y_pred_ref = y_pred.detach().clone().requires_grad_(True)

    actual = dispatch("kl_div", y_pred, y_true, "batchmean", False, 1e-10, impl="nvidia-cutedsl")
    expected = dispatch("kl_div", y_pred_ref, y_true, "batchmean", False, 1e-10, impl="nvidia-triton")
    torch.testing.assert_close(actual, expected)

    actual.backward()
    expected.backward()
    torch.testing.assert_close(y_pred.grad, y_pred_ref.grad)


@pytest.mark.parametrize("width", [769, 32769])
def test_rms_norm_cutedsl_shape_fallback_matches_triton(width):
    _requires_cutedsl("rms_norm")
    x = torch.randn(2, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)

    actual = dispatch("rms_norm", x, weight, 1e-6, 0.0, "llama", False, None, impl="nvidia-cutedsl")
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
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(weight.grad, weight_ref.grad)


@pytest.mark.parametrize("width", [769, 32769])
def test_layer_norm_cutedsl_shape_fallback_matches_triton(width):
    _requires_cutedsl("layer_norm")
    x = torch.randn(2, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)

    actual = dispatch("layer_norm", x, weight, bias, 1e-6, impl="nvidia-cutedsl")
    expected = dispatch("layer_norm", x_ref, weight_ref, bias_ref, 1e-6, impl="nvidia-triton")
    torch.testing.assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(weight.grad, weight_ref.grad)
    torch.testing.assert_close(bias.grad, bias_ref.grad)


@pytest.mark.parametrize("width", [769, 32769])
def test_fused_add_rms_norm_cutedsl_shape_fallback_matches_triton(width):
    _requires_cutedsl("fused_add_rms_norm")
    x = torch.randn(2, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn_like(x, requires_grad=True)
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    residual_ref = residual.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)

    actual = dispatch(
        "fused_add_rms_norm",
        x,
        residual,
        weight,
        1e-6,
        0.0,
        "llama",
        False,
        impl="nvidia-cutedsl",
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
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])

    grad_y = torch.randn_like(actual[0])
    grad_residual = torch.randn_like(actual[1])
    torch.autograd.backward(actual, (grad_y, grad_residual))
    torch.autograd.backward(expected, (grad_y, grad_residual))
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(residual.grad, residual_ref.grad)
    torch.testing.assert_close(weight.grad, weight_ref.grad)
