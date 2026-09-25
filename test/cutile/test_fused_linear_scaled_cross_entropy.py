"""cuTile-specific tests for the fused linear scaled cross-entropy op.

These were split out of ``test/transformers/test_fused_linear_scaled_cross_entropy.py``
(which keeps the backend-agnostic frontend/dispatcher tests). Every test here is gated
on ``LIGER_KERNEL_IMPL=cutile`` and CUDA, so the suite is a no-op unless the cuTile
backend is actively selected. Run it via::

    LIGER_KERNEL_IMPL=cutile pytest test/cutile/test_fused_linear_scaled_cross_entropy.py -v

The cuTile results are compared against the pure-torch ``_FusedLinearPPOFallbackFunction``
loaded directly from the frontend module (via ``_FRONTEND``), so the reference does not
depend on which backend is active.
"""

import importlib.util
import os

from pathlib import Path

import pytest
import torch

_FRONTEND_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "liger_kernel" / "ops" / "fused_linear_scaled_cross_entropy.py"
)
_SPEC = importlib.util.spec_from_file_location("_fused_linear_scaled_cross_entropy_frontend", _FRONTEND_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_FRONTEND = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FRONTEND)

_CUTILE_ENABLED = os.environ.get("LIGER_KERNEL_IMPL", "").strip().lower() == "cutile"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile requires CUDA")
@pytest.mark.skipif(not _CUTILE_ENABLED, reason="requires LIGER_KERNEL_IMPL=cutile")
@pytest.mark.parametrize(
    ("shape", "dtype", "atol", "rtol"),
    [
        ((17, 31, 257), torch.float32, 6e-5, 6e-5),
        ((1025, 64, 131072), torch.bfloat16, 8e-2, 5e-2),
    ],
    ids=["fp32-ragged", "bf16-multichunk"],
)
def test_cutile_matches_verl_fallback_forward_and_combined_backward(shape, dtype, atol, rtol):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported")

    from liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy import LigerFusedLinearScaledCrossEntropyFunction

    torch.manual_seed(42)
    token_count, hidden_size, vocab_size = shape
    temperature = 0.8
    target = torch.randint(vocab_size, (token_count,), device="cuda", dtype=torch.int64)
    target[::11] = -100
    grad_nll = torch.randn(token_count, device="cuda", dtype=torch.float32)
    grad_entropy = torch.randn(token_count, device="cuda", dtype=dtype)

    actual_input = (torch.randn(token_count, hidden_size, device="cuda", dtype=dtype) * 0.25).requires_grad_(True)
    actual_weight = (
        torch.randn(vocab_size, hidden_size, device="cuda", dtype=dtype) / hidden_size**0.5
    ).requires_grad_(True)
    actual_nll, actual_entropy = LigerFusedLinearScaledCrossEntropyFunction.apply(
        actual_input,
        actual_weight,
        target,
        temperature,
        -100,
        3,
        True,
    )
    torch.autograd.backward((actual_nll, actual_entropy), (grad_nll, grad_entropy))

    expected_input = actual_input.detach().clone().requires_grad_(True)
    expected_weight = actual_weight.detach().clone().requires_grad_(True)
    expected_nll, expected_entropy = _FRONTEND._FusedLinearPPOFallbackFunction.apply(
        expected_input,
        expected_weight,
        target,
        temperature,
        -100,
        True,
    )
    torch.autograd.backward((expected_nll, expected_entropy), (grad_nll, grad_entropy))

    torch.testing.assert_close(actual_nll, expected_nll, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_entropy, expected_entropy, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_input.grad, expected_input.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_weight.grad, expected_weight.grad, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile requires CUDA")
@pytest.mark.skipif(not _CUTILE_ENABLED, reason="requires LIGER_KERNEL_IMPL=cutile")
@pytest.mark.parametrize("output", ["nll", "entropy"])
def test_cutile_supports_single_output_backward(output):
    from liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy import LigerFusedLinearScaledCrossEntropyFunction

    torch.manual_seed(43)
    token_count, hidden_size, vocab_size = 19, 33, 259
    target = torch.randint(vocab_size, (token_count,), device="cuda", dtype=torch.int64)
    target[::5] = -100
    grad_output = torch.randn(token_count, device="cuda")

    actual_input = (torch.randn(token_count, hidden_size, device="cuda") * 0.25).requires_grad_(True)
    actual_weight = (torch.randn(vocab_size, hidden_size, device="cuda") / hidden_size**0.5).requires_grad_(True)
    actual_nll, actual_entropy = LigerFusedLinearScaledCrossEntropyFunction.apply(
        actual_input,
        actual_weight,
        target,
        1.2,
        -100,
        2,
        True,
    )
    (actual_nll if output == "nll" else actual_entropy).backward(grad_output)

    expected_input = actual_input.detach().clone().requires_grad_(True)
    expected_weight = actual_weight.detach().clone().requires_grad_(True)
    expected_nll, expected_entropy = _FRONTEND._FusedLinearPPOFallbackFunction.apply(
        expected_input,
        expected_weight,
        target,
        1.2,
        -100,
        True,
    )
    (expected_nll if output == "nll" else expected_entropy).backward(grad_output)

    torch.testing.assert_close(actual_input.grad, expected_input.grad, atol=6e-5, rtol=6e-5)
    torch.testing.assert_close(actual_weight.grad, expected_weight.grad, atol=6e-5, rtol=6e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile requires CUDA")
@pytest.mark.skipif(not _CUTILE_ENABLED, reason="requires LIGER_KERNEL_IMPL=cutile")
@pytest.mark.parametrize(
    ("token_count", "vocab_size", "element_size", "expected"),
    [
        (4096, 131072, 2, 1024),
        (1025, 131072, 2, 513),
        (2500, 100000, 2, 1250),
        (5000, 131072, 2, 1024),
    ],
)
def test_cutile_token_chunk_size_balances_small_tails(token_count, vocab_size, element_size, expected):
    from liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy import _calculate_token_chunk_size

    assert _calculate_token_chunk_size(token_count, vocab_size, element_size) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile requires CUDA")
@pytest.mark.skipif(not _CUTILE_ENABLED, reason="requires LIGER_KERNEL_IMPL=cutile")
@pytest.mark.parametrize(
    ("architecture", "token_count", "vocab_size", "element_size", "expected_mb"),
    [
        ("hopper", 4096, 131072, 2, 256),
        ("blackwell", 4096, 131072, 2, 512),
        ("blackwell_ultra", 4096, 131072, 2, 512),
        ("blackwell", 2048, 131072, 2, 256),
        ("blackwell", 4096, 65536, 2, 256),
        ("blackwell", 4096, 131072, 4, 256),
    ],
)
def test_cutile_selects_workspace_by_architecture(
    monkeypatch,
    architecture,
    token_count,
    vocab_size,
    element_size,
    expected_mb,
):
    import liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy as scaled_ce

    monkeypatch.delenv(scaled_ce._WORKSPACE_MB_ENV, raising=False)
    monkeypatch.setattr(scaled_ce, "infer_device_arch", lambda _: architecture)

    actual = scaled_ce._select_logits_workspace_bytes(0, token_count, vocab_size, element_size)
    assert actual == expected_mb * scaled_ce._MIB


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile requires CUDA")
@pytest.mark.skipif(not _CUTILE_ENABLED, reason="requires LIGER_KERNEL_IMPL=cutile")
def test_cutile_workspace_override(monkeypatch):
    import liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy as scaled_ce

    monkeypatch.setenv(scaled_ce._WORKSPACE_MB_ENV, "1024")
    workspace_bytes = scaled_ce._select_logits_workspace_bytes(0, 4096, 131072, 2)

    assert workspace_bytes == 1024 * scaled_ce._MIB
    assert scaled_ce._calculate_token_chunk_size(4096, 131072, 2, workspace_bytes) == 4096


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile requires CUDA")
@pytest.mark.skipif(not _CUTILE_ENABLED, reason="requires LIGER_KERNEL_IMPL=cutile")
@pytest.mark.parametrize("value", ["invalid", "0", "-1"])
def test_cutile_rejects_invalid_workspace_override(monkeypatch, value):
    import liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy as scaled_ce

    monkeypatch.setenv(scaled_ce._WORKSPACE_MB_ENV, value)
    with pytest.raises(ValueError, match=scaled_ce._WORKSPACE_MB_ENV):
        scaled_ce._select_logits_workspace_bytes(0, 4096, 131072, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile requires CUDA")
@pytest.mark.skipif(not _CUTILE_ENABLED, reason="requires LIGER_KERNEL_IMPL=cutile")
def test_cutile_rejects_out_of_range_target():
    from liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy import LigerFusedLinearScaledCrossEntropyFunction

    _input = torch.randn(2, 8, device="cuda")
    weight = torch.randn(16, 8, device="cuda")
    target = torch.tensor([0, 16], device="cuda", dtype=torch.int64)

    with pytest.raises(ValueError, match=r"outside \[0, 16\)"):
        LigerFusedLinearScaledCrossEntropyFunction.apply(_input, weight, target)
