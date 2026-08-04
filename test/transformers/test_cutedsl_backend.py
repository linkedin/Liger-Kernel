"""Verify LIGER_KERNEL_IMPL=cutedsl routes ops and transformers to CuTe DSL implementations.

Run with the backend selected, e.g.::

    LIGER_KERNEL_IMPL=cutedsl pytest test/transformers/test_cutedsl_backend.py -v

The suite is skipped unless CUDA is present *and* the env var is set, so it never
runs (or fails) in an environment where the CuTe DSL backend isn't active.
"""

import importlib
import os

import pytest
import torch

CUTEDSL_PREFIX = "liger_kernel.ops.cutedsl."

# Transformer modules bind liger_kernel.ops.* symbols at import time. Each entry maps
# the exported Function name to the transformer module expected to re-bind it.
TRANSFORMER_MODULES = {
    "LigerCrossEntropyFunction": "liger_kernel.transformers.cross_entropy",
    "LigerRMSNormFunction": "liger_kernel.transformers.rms_norm",
}

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="cutedsl backend requires CUDA",
    ),
    pytest.mark.skipif(
        os.environ.get("LIGER_KERNEL_IMPL", "").strip().lower() != "cutedsl",
        reason="cutedsl backend selection test requires LIGER_KERNEL_IMPL=cutedsl",
    ),
]


def test_liger_kernel_impl_cutedsl_routes():
    import liger_kernel.ops as ops
    import liger_kernel.ops.cutedsl.ops as cutedsl_ops

    failures = []

    # Every Function the cutedsl backend exports (RMSNorm + cross entropy) must be the
    # one exposed by liger_kernel.ops after backend replacement.
    for name in cutedsl_ops.__all__:
        if not name.endswith("Function"):
            continue
        cls = getattr(ops, name)
        if not cls.__module__.startswith(CUTEDSL_PREFIX):
            failures.append(f"liger_kernel.ops.{name} -> {cls.__module__}")

    for name, module_path in TRANSFORMER_MODULES.items():
        cls = getattr(importlib.import_module(module_path), name)
        if not cls.__module__.startswith(CUTEDSL_PREFIX):
            failures.append(f"{module_path}.{name} -> {cls.__module__}")

    assert not failures, "expected cutedsl routing:\n" + "\n".join(failures)


def test_cutedsl_rms_norm_is_selected():
    """Focused check that RMSNorm specifically resolves to the CuTe DSL kernel."""
    import liger_kernel.ops as ops

    assert ops.LigerRMSNormFunction.__module__.startswith(CUTEDSL_PREFIX)
    assert ops.rms_norm_forward.__module__.startswith(CUTEDSL_PREFIX)
    assert ops.rms_norm_backward.__module__.startswith(CUTEDSL_PREFIX)
