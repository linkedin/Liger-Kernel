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
