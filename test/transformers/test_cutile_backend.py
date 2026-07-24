"""Verify LIGER_KERNEL_IMPL=cutile routes ops and transformers to cuTile implementations."""

import importlib
import os

import pytest
import torch

CUTILE_PREFIX = "liger_kernel.ops.cutile."

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
