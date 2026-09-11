"""cuTile cross-entropy regressions against softcapped PyTorch logits."""

import pytest
import torch

pytest.importorskip("cuda.tile", reason="cuda.tile is not installed")

from liger_kernel.ops.cutile.ops.cross_entropy import LigerCrossEntropyFunction  # noqa: E402
from test.cutile.test_cutile_backends_parity import _cutile_impl  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile cross-entropy requires CUDA")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("vocab_size", [3, 4097], ids=["v3", "v4097"])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("entrypoint", ["native", "dispatcher"])
def test_softcap_partial_vocabulary(dtype, vocab_size, reduction, entrypoint):
    impl = _cutile_impl("cross_entropy")
    softcap = 1.0
    logits = torch.linspace(-2.0, 2.0, vocab_size, device="cuda", dtype=dtype).repeat(4, 1)
    logits[0].zero_()
    logits[2].neg_()
    target = torch.tensor([0, vocab_size - 1, vocab_size // 2, -100], device="cuda")
    reference_input = logits.float().detach().requires_grad_(True)
    reference_loss = torch.nn.functional.cross_entropy(
        softcap * torch.tanh(reference_input / softcap), target, reduction=reduction
    )
    upstream = (
        torch.tensor([0.5, 1.5, -0.75, 2.0], device="cuda", dtype=dtype)
        if reduction == "none"
        else torch.tensor(1.5, device="cuda", dtype=dtype)
    )
    reference_loss.backward(upstream.float())

    logits.requires_grad_(True)
    if entrypoint == "native":
        loss, _, _, _ = LigerCrossEntropyFunction.apply(logits, target, None, -100, 0.0, 0.0, reduction, softcap)
    else:
        from liger_kernel import functional

        loss, _, _, _ = functional.cross_entropy(logits, target, reduction=reduction, softcap=softcap, impl=impl)
    loss.backward(upstream)

    rtol = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    atol_grad = 1e-7 if dtype == torch.float32 else 1e-6
    torch.testing.assert_close(loss.float(), reference_loss.detach(), atol=0.0, rtol=rtol)
    torch.testing.assert_close(logits.grad.float(), reference_input.grad, atol=atol_grad, rtol=rtol)


@pytest.mark.parametrize("ignore_index", [-100, 5])
@pytest.mark.parametrize("weighted", [False, True])
def test_validation_statistics_preserve_ignore_and_weighting(ignore_index, weighted):
    _cutile_impl("cross_entropy")
    logits = torch.tensor([[0.5, -0.5, 1.0], [1.0, 2.0, 0.0], [-1.0, 0.0, 1.0]], device="cuda")
    target = torch.tensor([0, ignore_index, 2], device="cuda")
    weight = torch.tensor([0.25, 1.0, 2.0], device="cuda") if weighted else None
    reference_input = logits.detach().clone().requires_grad_(True)
    expected = torch.nn.functional.cross_entropy(reference_input, target, weight=weight, ignore_index=ignore_index)
    expected.backward()
    logits.requires_grad_(True)
    actual = LigerCrossEntropyFunction.apply(logits, target, weight, ignore_index, 0.0, 0.0, "mean", None, False)[0]
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(logits.grad, reference_input.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("ignore_index", [-100, 5])
def test_validation_statistics_preserve_all_ignored(ignore_index):
    _cutile_impl("cross_entropy")
    logits = torch.randn(3, 3, device="cuda", requires_grad=True)
    target = torch.full((3,), ignore_index, device="cuda", dtype=torch.int64)
    loss = LigerCrossEntropyFunction.apply(logits, target, None, ignore_index, 0.0, 0.0, "mean", None, False)[0]
    loss.backward()
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=0, rtol=0)
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=0, rtol=0)


@pytest.mark.parametrize("invalid_target", [-1, 3])
def test_validation_statistics_reject_invalid_targets(invalid_target):
    _cutile_impl("cross_entropy")
    logits = torch.randn(3, 3, device="cuda", requires_grad=True)
    target = torch.tensor([0, invalid_target, 2], device="cuda")
    with pytest.raises(AssertionError, match="out of bounds"):
        LigerCrossEntropyFunction.apply(logits, target, None, -100, 0.0, 0.0, "mean", None, False)
