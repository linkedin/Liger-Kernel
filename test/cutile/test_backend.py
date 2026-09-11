"""Verify LIGER_KERNEL_IMPL=cutile routes ops and transformers to cuTile implementations."""

import importlib
import os

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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="cuTile cross_entropy/rope require Blackwell (sm_100+); skip where they are not registered.",
)
def test_cutile_dispatch_transformers_execute():
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.rope import liger_rotary_pos_emb

    logits = torch.randn(8, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, 64, (8,), device="cuda")
    loss = LigerCrossEntropyLoss()(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()

    q = torch.randn(1, 2, 4, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(1, 2, 4, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    cos = torch.ones(1, 4, 8, device="cuda", dtype=torch.float32)
    sin = torch.zeros_like(cos)
    q_out, k_out = liger_rotary_pos_emb(q, k, cos, sin)
    (q_out.sum() + k_out.sum()).backward()
    assert torch.isfinite(q_out).all()
    assert torch.isfinite(k_out).all()
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
