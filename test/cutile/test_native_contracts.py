"""Focused regressions for native cuTile shape and backward contracts."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cuda.tile", reason="cuda.tile is not installed")

from liger_kernel.ops.cutile.ops.group_norm import LigerGroupNormFunction  # noqa: E402
from liger_kernel.ops.cutile.ops.multi_token_attention import LigerMultiTokenAttentionFunction  # noqa: E402
from liger_kernel.ops.cutile.ops.sparsemax import LigerSparsemaxFunction  # noqa: E402
from liger_kernel.ops.cutile.ops.tiled_mlp import apply_tiled_mlp  # noqa: E402
from test.cutile.test_cutile_backends_parity import _cutile_supported  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="native cuTile regressions require CUDA")


@pytest.fixture(autouse=True)
def require_native_cutile_stack():
    supported, reason = _cutile_supported()
    if not supported:
        pytest.skip(reason)


@pytest.mark.parametrize("shape", [(2, 6, 15), (2, 6, 3, 5), (2, 6, 2, 3, 5)], ids=["1d", "2d", "3d"])
def test_group_norm_spatial_rank(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", requires_grad=True)
    weight = torch.randn(6, device="cuda", requires_grad=True)
    bias = torch.randn(6, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    wr = weight.detach().clone().requires_grad_(True)
    br = bias.detach().clone().requires_grad_(True)
    expected = F.group_norm(xr, 3, wr, br, 1e-5)
    actual = LigerGroupNormFunction.apply(x, weight, bias, 6, 3, 1e-5)
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad)
    for result, reference in ((actual, expected), (x.grad, xr.grad), (weight.grad, wr.grad), (bias.grad, br.grad)):
        torch.testing.assert_close(result, reference, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("kernel_size", [1, 3], ids=["pointwise", "spatial"])
def test_attention_unpadded_convolution_backward(kernel_size, monkeypatch):
    # Compare full-precision convolution and GEMM, not TF32 against IEEE FP32.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    torch.manual_seed(0)
    scores = torch.randn(2, 2, 8, 8, device="cuda", requires_grad=True)
    weight = torch.randn(3, 2, kernel_size, kernel_size, device="cuda", requires_grad=True)
    bias = torch.randn(3, device="cuda", requires_grad=True)
    sr = scores.detach().clone().requires_grad_(True)
    wr = weight.detach().clone().requires_grad_(True)
    br = bias.detach().clone().requires_grad_(True)
    upper = torch.ones(8, 8, device="cuda", dtype=torch.bool).triu(1)
    probs = sr.masked_fill(upper, -torch.inf).softmax(-1)
    conv = F.conv2d(probs, wr, br)
    out_width = conv.shape[-1]
    out_upper = torch.ones(out_width, out_width, device="cuda", dtype=torch.bool).triu(1)
    expected = conv.masked_fill(out_upper, 0.0)
    actual = LigerMultiTokenAttentionFunction.apply(scores, weight, bias)
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad)
    for result, reference in ((actual, expected), (scores.grad, sr.grad), (weight.grad, wr.grad), (bias.grad, br.grad)):
        torch.testing.assert_close(result, reference, atol=1e-4, rtol=1e-4)


def test_sparsemax_singleton_support_gradient():
    x = torch.tensor([[2.0, 0.0, -1.0]], device="cuda", requires_grad=True)
    actual = LigerSparsemaxFunction.apply(x, -1)
    torch.testing.assert_close(actual, torch.tensor([[1.0, 0.0, 0.0]], device="cuda"), atol=0.0, rtol=0.0)
    actual.backward(torch.tensor([[8.0, 2.0, -3.0]], device="cuda"))
    # A singleton support is locally constant, so its Jacobian is exactly zero.
    torch.testing.assert_close(x.grad, torch.zeros_like(x), atol=0.0, rtol=0.0)


def test_tiled_mlp_non_contiguous_forward_backward():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7, device="cuda").transpose(0, 1).detach().requires_grad_(True)
    grad = torch.randn(2, 3, 7, device="cuda").transpose(0, 1)
    assert not x.is_contiguous() and not grad.is_contiguous()
    xr = x.detach().clone().requires_grad_(True)
    layer = torch.nn.Linear(7, 7, device="cuda")
    reference_layer = torch.nn.Linear(7, 7, device="cuda")
    reference_layer.load_state_dict(layer.state_dict())
    expected = reference_layer(xr)
    actual = apply_tiled_mlp(lambda module, shard: module(shard), layer, x, num_shards=2)
    expected.backward(grad)
    actual.backward(grad)
    for result, reference in (
        (actual, expected),
        (x.grad, xr.grad),
        (layer.weight.grad, reference_layer.weight.grad),
        (layer.bias.grad, reference_layer.bias.grad),
    ):
        torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)
