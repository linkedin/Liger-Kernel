"""Verify LIGER_KERNEL_IMPL=cutile routes ops and transformers to cuTile implementations."""

import importlib
import os

from unittest.mock import patch

import pytest
import torch

CUTILE_PREFIX = "liger_kernel.ops.cutile."

# Transformer modules bind liger_kernel.ops.* at import time (geglu != GELUMul name).
TRANSFORMER_MODULES = {
    "LigerDyTFunction": "liger_kernel.transformers.dyt",
    "LigerFusedAddRMSNormFunction": "liger_kernel.transformers.fused_add_rms_norm",
    "LigerFusedLinearCrossEntropyFunction": "liger_kernel.transformers.fused_linear_cross_entropy",
    "LigerFusedLinearJSDFunction": "liger_kernel.transformers.fused_linear_jsd",
    "LigerGELUMulFunction": "liger_kernel.transformers.geglu",
    "GrpoLossFunction": "liger_kernel.transformers.grpo_loss",
    "LigerJSDFunction": "liger_kernel.transformers.jsd",
    "LigerLayerNormFunction": "liger_kernel.transformers.layer_norm",
    "LigerPolyNormFunction": "liger_kernel.transformers.poly_norm",
    "LigerRMSNormFunction": "liger_kernel.transformers.rms_norm",
    "LigerSiLUMulFunction": "liger_kernel.transformers.swiglu",
    "LigerSoftmaxFunction": "liger_kernel.transformers.softmax",
}

DISPATCH_OPS = (
    "cross_entropy",
    "fused_add_rms_norm",
    "geglu",
    "kl_div",
    "rope",
    "swiglu",
)

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="cuTile backend requires CUDA",
    ),
    pytest.mark.skipif(
        os.environ.get("LIGER_KERNEL_IMPL", "").strip().lower() != "cutile",
        reason="cuTile backend selection test requires LIGER_KERNEL_IMPL=cutile",
    ),
]


def test_liger_kernel_impl_cutile_routes():
    import liger_kernel.ops as ops
    import liger_kernel.ops.cutile.ops as cutile_ops

    failures = []

    for name in cutile_ops.__all__:
        if not name.endswith("Function"):
            continue
        cls = getattr(ops, name)
        if not cls.__module__.startswith(CUTILE_PREFIX):
            failures.append(f"liger_kernel.ops.{name} -> {cls.__module__}")

    for name, module_path in TRANSFORMER_MODULES.items():
        cls = getattr(importlib.import_module(module_path), name)
        if not cls.__module__.startswith(CUTILE_PREFIX):
            failures.append(f"{module_path}.{name} -> {cls.__module__}")

    assert not failures, "expected cuTile routing:\n" + "\n".join(failures)


def test_functional_dispatcher_registers_cutile_ops():
    from liger_kernel.backends.registry import get_registered

    missing = [op_name for op_name in DISPATCH_OPS if get_registered(op_name, "nvidia-cutile") is None]
    assert not missing, f"missing cuTile dispatcher adapters: {missing}"
    for op_name in DISPATCH_OPS:
        impl = get_registered(op_name, "nvidia-cutile")
        assert impl.modes == ("default",)
        assert impl.default_mode == "default"
        assert impl.capability.min_cc == (10, 0)
        assert impl.preference_rank == 80


@pytest.mark.parametrize("op_name", DISPATCH_OPS)
def test_cutile_adapters_reject_unknown_modes(op_name):
    from liger_kernel.backends.registry import get_registered

    impl = get_registered(op_name, "nvidia-cutile")
    assert impl is not None
    x = torch.empty(2, 4)
    args = {
        "cross_entropy": (x, torch.empty(2, dtype=torch.int64)),
        "fused_add_rms_norm": (x, x, torch.empty(4)),
        "geglu": (x, x),
        "kl_div": (x, x),
        "rope": (x, x, x, x),
        "swiglu": (x, x),
    }
    with pytest.raises(ValueError, match="has only mode='default'"):
        impl.call(*args[op_name], mode="unsupported")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="cuTile cross_entropy/rope require Blackwell (sm_100+); skip where they are not registered.",
)
def test_cutile_dispatch_transformers_execute(monkeypatch):
    from liger_kernel.ops.cutile.ops.cross_entropy import LigerCrossEntropyFunction
    from liger_kernel.ops.cutile.ops.rope import LigerRopeFunction
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from test.cutile.test_rope import _rotate_reference

    monkeypatch.setenv("LIGER_KERNEL_IMPL_CROSS_ENTROPY", "nvidia-cutile")
    monkeypatch.setenv("LIGER_KERNEL_IMPL_ROPE", "nvidia-cutile")
    logits = torch.randn(8, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    logits_ref = logits.detach().clone().requires_grad_(True)
    target = torch.randint(0, 64, (8,), device="cuda")
    with patch.object(LigerCrossEntropyFunction, "apply", wraps=LigerCrossEntropyFunction.apply) as native_ce:
        loss = LigerCrossEntropyLoss()(logits, target)
        native_ce.assert_called_once()
    expected_loss = torch.nn.functional.cross_entropy(logits_ref, target)
    loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(loss, expected_loss, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(logits.grad, logits_ref.grad, atol=1e-6, rtol=1e-5)

    q = torch.randn(1, 2, 4, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(1, 2, 4, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    angles = torch.randn(1, 4, 4, device="cuda", dtype=torch.float32)
    cos = angles.cos().repeat(1, 1, 2)
    sin = angles.sin().repeat(1, 1, 2)
    with patch.object(LigerRopeFunction, "apply", wraps=LigerRopeFunction.apply) as native_rope:
        q_out, k_out = liger_rotary_pos_emb(q, k, cos, sin)
        native_rope.assert_called_once()
    q_expected, k_expected = _rotate_reference(q_ref, cos, sin), _rotate_reference(k_ref, cos, sin)
    (q_out.sum() + k_out.sum()).backward()
    (q_expected.sum() + k_expected.sum()).backward()
    torch.testing.assert_close(q_out, q_expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k_out, k_expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=1e-5, rtol=1e-5)
