"""Tests for apply_liger_kernel_to_megatron's patch mechanism.

Megatron-LM is not a test dependency. We inject stub modules into
``sys.modules`` so the patch function can run entirely on CPU without a real
megatron-core install. Tests verify:

- the patch replaces ``fused_vocab_parallel_cross_entropy`` on the stub module
- ``reduction != "none"`` is rejected
- TP>1 at patch time raises RuntimeError
- missing megatron-core raises a helpful ImportError
- missing symbol path raises a helpful ImportError
- the constructed LigerCrossEntropyLoss receives the user-supplied kwargs
"""

import sys
import types

from unittest.mock import patch

import pytest


def _install_fake_megatron(tp_size: int = 1, with_fused_symbol: bool = True):
    """Install stub megatron modules into sys.modules; return the fused module."""
    megatron = types.ModuleType("megatron")
    megatron_core = types.ModuleType("megatron.core")
    fusions = types.ModuleType("megatron.core.fusions")
    fused_ce = types.ModuleType("megatron.core.fusions.fused_cross_entropy")
    parallel_state = types.ModuleType("megatron.core.parallel_state")

    if with_fused_symbol:

        def original_fused_vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group=None):
            raise AssertionError("original megatron kernel called — patch failed")

        fused_ce.fused_vocab_parallel_cross_entropy = original_fused_vocab_parallel_cross_entropy

    parallel_state.get_tensor_model_parallel_world_size = lambda: tp_size

    sys.modules["megatron"] = megatron
    sys.modules["megatron.core"] = megatron_core
    sys.modules["megatron.core.fusions"] = fusions
    sys.modules["megatron.core.fusions.fused_cross_entropy"] = fused_ce
    sys.modules["megatron.core.parallel_state"] = parallel_state

    megatron.core = megatron_core
    megatron_core.fusions = fusions
    megatron_core.parallel_state = parallel_state
    fusions.fused_cross_entropy = fused_ce

    return fused_ce


def _uninstall_fake_megatron():
    for mod in [
        "megatron.core.parallel_state",
        "megatron.core.fusions.fused_cross_entropy",
        "megatron.core.fusions",
        "megatron.core",
        "megatron",
    ]:
        sys.modules.pop(mod, None)


@pytest.fixture
def fake_megatron():
    fused_ce = _install_fake_megatron(tp_size=1)
    try:
        yield fused_ce
    finally:
        _uninstall_fake_megatron()


def test_patch_replaces_fused_symbol(fake_megatron):
    from liger_kernel.megatron import apply_liger_kernel_to_megatron

    original = fake_megatron.fused_vocab_parallel_cross_entropy
    apply_liger_kernel_to_megatron()
    patched = fake_megatron.fused_vocab_parallel_cross_entropy

    assert patched is not original
    assert patched.__name__ == "liger_fused_vocab_parallel_cross_entropy"


def test_patch_rejects_non_none_reduction(fake_megatron):
    from liger_kernel.megatron import apply_liger_kernel_to_megatron

    with pytest.raises(AssertionError, match="reduction must be 'none'"):
        apply_liger_kernel_to_megatron(reduction="mean")


def test_patch_raises_on_tp_greater_than_one():
    _install_fake_megatron(tp_size=2)
    try:
        from liger_kernel.megatron import apply_liger_kernel_to_megatron

        with pytest.raises(RuntimeError, match="tensor_model_parallel_size=1"):
            apply_liger_kernel_to_megatron()
    finally:
        _uninstall_fake_megatron()


def test_patch_defers_tp_check_when_parallel_state_not_initialized():
    """If get_tensor_model_parallel_world_size() raises, patch should still succeed."""
    fused_ce = _install_fake_megatron(tp_size=1)

    def raising_tp_size():
        raise AssertionError("parallel_state not initialized")

    sys.modules["megatron.core.parallel_state"].get_tensor_model_parallel_world_size = raising_tp_size

    try:
        from liger_kernel.megatron import apply_liger_kernel_to_megatron

        apply_liger_kernel_to_megatron()
        assert fused_ce.fused_vocab_parallel_cross_entropy.__name__ == "liger_fused_vocab_parallel_cross_entropy"
    finally:
        _uninstall_fake_megatron()


def test_patch_raises_when_megatron_not_installed():
    _uninstall_fake_megatron()
    # Block imports of any "megatron*" module to simulate absent install.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "megatron" or name.startswith("megatron."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=blocking_import):
        from liger_kernel.megatron import apply_liger_kernel_to_megatron

        with pytest.raises(ImportError, match="requires megatron-core"):
            apply_liger_kernel_to_megatron()


def test_patch_raises_when_fused_symbol_missing():
    _install_fake_megatron(tp_size=1, with_fused_symbol=False)
    try:
        from liger_kernel.megatron import apply_liger_kernel_to_megatron

        with pytest.raises(ImportError, match="symbol path may have changed"):
            apply_liger_kernel_to_megatron()
    finally:
        _uninstall_fake_megatron()


def test_patch_forwards_ignore_index_and_label_smoothing(fake_megatron):
    from liger_kernel.megatron import cross_entropy as mod

    captured = {}

    class FakeLoss:
        def __init__(self, ignore_index, label_smoothing, reduction):
            captured["ignore_index"] = ignore_index
            captured["label_smoothing"] = label_smoothing
            captured["reduction"] = reduction

        def __call__(self, _input, target):
            raise AssertionError("not expected to be called in this test")

    with patch.object(mod, "LigerCrossEntropyLoss", FakeLoss):
        mod.apply_liger_kernel_to_megatron(ignore_index=42, label_smoothing=0.25)

    assert captured == {"ignore_index": 42, "label_smoothing": 0.25, "reduction": "none"}
