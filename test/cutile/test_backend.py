"""Verify LIGER_KERNEL_IMPL=cutile routes ops and transformers to cuTile implementations."""

import importlib
import os

import pytest
import torch

CUTILE_PREFIX = "liger_kernel.ops.cutile."
SHARED_TILED_MLP_EXPORTS = ("LigerTiledMLPFunction", "apply_tiled_mlp")

# Transformer modules bind liger_kernel.ops.* at import time (geglu != GELUMul name).
TRANSFORMER_MODULES = {
    "LigerCrossEntropyFunction": "liger_kernel.transformers.cross_entropy",
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
    "apply_tiled_mlp": "liger_kernel.transformers.tiled_mlp",
}

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

    # Backend-independent exports such as tiled MLP retain their original module.
    # Routing must preserve the backend's exact objects, including helper functions.
    for name in cutile_ops.__all__:
        actual = getattr(ops, name)
        expected = getattr(cutile_ops, name)
        if actual is not expected:
            failures.append(f"liger_kernel.ops.{name} -> {actual!r} (expected {expected!r})")
        if name not in SHARED_TILED_MLP_EXPORTS and not expected.__module__.startswith(CUTILE_PREFIX):
            failures.append(f"cuTile export {name} -> {expected.__module__}")

    for name, module_path in TRANSFORMER_MODULES.items():
        actual = getattr(importlib.import_module(module_path), name)
        expected = getattr(cutile_ops, name)
        if actual is not expected:
            failures.append(f"{module_path}.{name} -> {actual!r} (expected {expected!r})")

    assert not failures, "expected cuTile routing:\n" + "\n".join(failures)


@pytest.mark.parametrize("name", SHARED_TILED_MLP_EXPORTS)
def test_cutile_tiled_mlp_uses_shared_implementation(name):
    import liger_kernel.ops.cutile.ops as cutile_ops
    import liger_kernel.ops.tiled_mlp as tiled_mlp

    assert getattr(cutile_ops, name) is getattr(tiled_mlp, name)
