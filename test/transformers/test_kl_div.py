import pytest
import torch

from test.utils import supports_bfloat16
from torch.nn import KLDivLoss

from liger_kernel.transformers.kl_div import LigerKLDIVLoss
from liger_kernel.utils import infer_device

device = infer_device()

_SHAPE_PARAMS = (
    "B, T, V",
    [
        (1, 4096, 32000),
        # weird shape
        (41, 401, 1271),
    ],
)

_DTYPE_PARAMS = (
    "dtype, atol, rtol",
    [
        pytest.param(
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (torch.float32, 1e-8, 1e-6),
        (torch.float16, 1e-3, 1e-3),
    ],
)


def _test_correctness_once(
    target_kldiv,
    B,
    T,
    V,
    dtype,
    atol,
    rtol,
    reduction,
    log_target,
    is_last_layer=True,
    device=device,
):
    torch.manual_seed(0)
    torch_kldiv = KLDivLoss(reduction=reduction, log_target=log_target)

    input = torch.randn(B * T, V, device=device, dtype=dtype, requires_grad=True).log_softmax(dim=-1)

    x1 = input.detach().clone().requires_grad_(True)
    x2 = input.detach().clone().requires_grad_(True)

    with torch.no_grad():
        target = torch.randn(B * T, V, device=device).softmax(dim=-1)

    output = torch_kldiv(x1, target)
    output2 = target_kldiv(x2, target)
    assert torch.allclose(output, output2, atol=atol, rtol=rtol)

    if (
        not is_last_layer
    ):  # if the loss is the last layer, grad_output is 1.0 and mul op is skipped, testing for that reason
        output = output * 2.0
        output2 = output2 * 2.0

    if reduction == "none":
        return

    output.backward()
    output2.backward()
    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize("log_target", [True, False])
@pytest.mark.parametrize("reduction", ["batchmean", "sum", "mean", "none"])
@pytest.mark.parametrize(*_DTYPE_PARAMS)
def test_correctness(B, T, V, log_target, reduction, dtype, atol, rtol):
    liger_kldiv = LigerKLDIVLoss(reduction=reduction, log_target=log_target)
    _test_correctness_once(liger_kldiv, B, T, V, dtype, atol, rtol, reduction, log_target)


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize("log_target", [True, False])
@pytest.mark.parametrize("reduction", ["batchmean", "sum", "mean", "none"])
@pytest.mark.parametrize(*_DTYPE_PARAMS)
def test_correctness_not_last(B, T, V, log_target, reduction, dtype, atol, rtol):
    liger_kldiv = LigerKLDIVLoss(reduction=reduction, log_target=log_target)
    _test_correctness_once(
        liger_kldiv,
        B,
        T,
        V,
        dtype,
        atol,
        rtol,
        reduction,
        log_target,
        is_last_layer=False,
    )


@pytest.mark.parametrize("log_target", [True, False])
@pytest.mark.parametrize("reduction", ["batchmean", "sum", "mean", "none"])
def test_torch_compile_matches_eager(log_target, reduction):
    # inductor re-emits the kernel source; a signature default that names a module global must
    # still resolve there, and eps must stay fp32 (inductor passes Python floats as fp64). The
    # unreduced loss matches eager bit for bit; a reduced loss may differ in the last bits
    # because inductor fuses the wrapper's sum/division
    torch._dynamo.reset()
    torch.manual_seed(0)
    liger_kldiv = LigerKLDIVLoss(reduction=reduction, log_target=log_target)
    compiled = torch.compile(liger_kldiv)
    input = torch.randn(64, 1271, device=device, dtype=torch.float32).log_softmax(dim=-1)
    with torch.no_grad():
        target = torch.randn(64, 1271, device=device).softmax(dim=-1)
        if log_target:
            target = target.log()
    x1 = input.detach().clone().requires_grad_(True)
    x2 = input.detach().clone().requires_grad_(True)
    out1 = liger_kldiv(x1, target)
    out2 = compiled(x2, target)
    if reduction == "none":
        assert torch.equal(out1, out2)
        return
    torch.testing.assert_close(out2, out1)
    out1.backward()
    out2.backward()
    torch.testing.assert_close(x2.grad, x1.grad)
