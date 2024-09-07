import pytest
import torch

from liger_kernel.transformers.conv2d import LigerConv2d


@pytest.mark.parametrize(
    "N, C, H, W, K, R, S, pad_h, pad_w, U, V, dila_h, dila_w",
    [
        (1, 64, 56, 56, 64, 1, 1, 0, 0, 1, 1, 1, 1),
        (1, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1),
        (1, 256, 14, 14, 256, 5, 5, 2, 2, 2, 2, 1, 1),
        (1, 512, 7, 7, 512, 7, 7, 3, 3, 1, 1, 2, 2),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 2e-2, 2e-2),
    ],
)
def test_conv2d_forward(
    N, C, H, W, K, R, S, pad_h, pad_w, U, V, dila_h, dila_w, dtype, atol, rtol
):
    torch.manual_seed(0)

    x = torch.randn(N, C, H, W, device="cuda", dtype=dtype)
    w = torch.randn(K, C, R, S, device="cuda", dtype=dtype)
    conv2d = (
        torch.nn.Conv2d(
            C,
            K,
            (R, S),
            stride=(U, V),
            padding=(pad_h, pad_w),
            dilation=(dila_h, dila_w),
            bias=False,
        )
        .cuda()
        .to(dtype)
    )
    conv2d.weight.data = w

    y_torch = conv2d(x)

    liger_conv2d = (
        LigerConv2d(
            C,
            K,
            (R, S),
            stride=(U, V),
            padding=(pad_h, pad_w),
            dilation=(dila_h, dila_w),
            bias=False,
        )
        .cuda()
        .to(dtype)
    )
    liger_conv2d.weight.data = w

    y_liger = liger_conv2d(x)

    assert torch.allclose(y_torch, y_liger, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "N, C, H, W, K, R, S, pad_h, pad_w, U, V, dila_h, dila_w",
    [
        (1, 64, 56, 56, 64, 1, 1, 0, 0, 1, 1, 1, 1),
        (1, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1),
        (1, 256, 14, 14, 256, 5, 5, 2, 2, 2, 2, 1, 1),
        (1, 512, 7, 7, 512, 7, 7, 3, 3, 1, 1, 2, 2),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 2e-2, 2e-2),
    ],
)
def test_conv2d_backward(
    N, C, H, W, K, R, S, pad_h, pad_w, U, V, dila_h, dila_w, dtype, atol, rtol
):
    torch.manual_seed(0)

    x = torch.randn(N, C, H, W, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(K, C, R, S, device="cuda", dtype=dtype, requires_grad=True)
    conv2d = (
        torch.nn.Conv2d(
            C,
            K,
            (R, S),
            stride=(U, V),
            padding=(pad_h, pad_w),
            dilation=(dila_h, dila_w),
            bias=False,
        )
        .cuda()
        .to(dtype)
    )
    conv2d.weight.data = w.clone()

    y_torch = conv2d(x)
    grad_output = torch.randn_like(y_torch)
    y_torch.backward(grad_output)

    dx_torch = x.grad.clone()
    dw_torch = conv2d.weight.grad.clone()

    x.grad = None
    w.grad = None

    liger_conv2d = (
        LigerConv2d(
            C,
            K,
            (R, S),
            stride=(U, V),
            padding=(pad_h, pad_w),
            dilation=(dila_h, dila_w),
            bias=False,
        )
        .cuda()
        .to(dtype)
    )
    liger_conv2d.weight.data = w.clone()

    y_liger = liger_conv2d(x)
    y_liger.backward(grad_output)

    dx_liger = x.grad
    dw_liger = liger_conv2d.weight.grad

    assert torch.allclose(dx_torch, dx_liger, atol=atol, rtol=rtol)
    assert torch.allclose(dw_torch, dw_liger, atol=atol, rtol=rtol)
