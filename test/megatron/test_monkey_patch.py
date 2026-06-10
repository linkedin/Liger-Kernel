"""Tests for ``apply_liger_kernel_to_megatron``'s cross-entropy patch mechanism.

Megatron-LM is not a test dependency. We inject stub modules into ``sys.modules`` so the
patch helpers can run entirely on CPU without a real megatron-core install. Tests verify:

- the patch replaces both the fused symbol
  (``megatron.core.fusions.fused_cross_entropy.fused_vocab_parallel_cross_entropy``)
  AND the unfused symbol
  (``megatron.core.tensor_parallel.cross_entropy.vocab_parallel_cross_entropy``)
- TP>1 at patch time raises ``RuntimeError``
- missing megatron-core / missing symbol path raise helpful ``ImportError``\\s
- the patch constructs ``LigerMegatronCrossEntropy`` with class defaults (matches Megatron
  native behavior — no CE-specific kwargs on the public ``apply_liger_kernel_to_megatron``
  API)
- the unfused wrapper honors a runtime ``label_smoothing`` override (Megatron's unfused
  signature is ``(logits, target, label_smoothing=0.0, tp_group=None)``)
"""

import sys
import types

from unittest.mock import patch

import pytest


def _install_fake_megatron(
    tp_size: int = 1,
    with_fused_symbol: bool = True,
    with_unfused_symbol: bool = True,
):
    """Install stub megatron modules into ``sys.modules``.

    Returns a tuple ``(fused_ce_module, unfused_ce_module)`` so tests can inspect what
    the patch helpers wrote onto them.
    """
    megatron = types.ModuleType("megatron")
    megatron_core = types.ModuleType("megatron.core")
    fusions = types.ModuleType("megatron.core.fusions")
    fused_ce = types.ModuleType("megatron.core.fusions.fused_cross_entropy")
    tensor_parallel = types.ModuleType("megatron.core.tensor_parallel")
    unfused_ce = types.ModuleType("megatron.core.tensor_parallel.cross_entropy")
    parallel_state = types.ModuleType("megatron.core.parallel_state")

    if with_fused_symbol:

        def original_fused_vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group=None):
            raise AssertionError("original megatron fused kernel called — patch failed")

        fused_ce.fused_vocab_parallel_cross_entropy = original_fused_vocab_parallel_cross_entropy

    if with_unfused_symbol:

        def original_vocab_parallel_cross_entropy(
            vocab_parallel_logits, target, label_smoothing=0.0, tp_group=None,
        ):
            raise AssertionError("original megatron unfused kernel called — patch failed")

        unfused_ce.vocab_parallel_cross_entropy = original_vocab_parallel_cross_entropy

    parallel_state.get_tensor_model_parallel_world_size = lambda: tp_size

    sys.modules["megatron"] = megatron
    sys.modules["megatron.core"] = megatron_core
    sys.modules["megatron.core.fusions"] = fusions
    sys.modules["megatron.core.fusions.fused_cross_entropy"] = fused_ce
    sys.modules["megatron.core.tensor_parallel"] = tensor_parallel
    sys.modules["megatron.core.tensor_parallel.cross_entropy"] = unfused_ce
    sys.modules["megatron.core.parallel_state"] = parallel_state

    megatron.core = megatron_core
    megatron_core.fusions = fusions
    megatron_core.tensor_parallel = tensor_parallel
    megatron_core.parallel_state = parallel_state
    fusions.fused_cross_entropy = fused_ce
    tensor_parallel.cross_entropy = unfused_ce

    return fused_ce, unfused_ce


def _uninstall_fake_megatron():
    for mod in [
        "megatron.core.parallel_state",
        "megatron.core.fusions.fused_cross_entropy",
        "megatron.core.fusions",
        "megatron.core.tensor_parallel.cross_entropy",
        "megatron.core.tensor_parallel",
        "megatron.core",
        "megatron",
    ]:
        sys.modules.pop(mod, None)


@pytest.fixture
def fake_megatron():
    fused_ce, unfused_ce = _install_fake_megatron(tp_size=1)
    try:
        yield fused_ce, unfused_ce
    finally:
        _uninstall_fake_megatron()


# ---------------------------------------------------------------------------
# Both symbols get replaced.
# ---------------------------------------------------------------------------


def test_patch_replaces_fused_symbol(fake_megatron):
    fused_ce, _ = fake_megatron
    from liger_kernel.megatron import apply_liger_kernel_to_megatron

    original = fused_ce.fused_vocab_parallel_cross_entropy
    apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)

    assert fused_ce.fused_vocab_parallel_cross_entropy is not original
    assert fused_ce.fused_vocab_parallel_cross_entropy.__name__ == "liger_fused_vocab_parallel_cross_entropy"


def test_patch_replaces_unfused_symbol(fake_megatron):
    _, unfused_ce = fake_megatron
    from liger_kernel.megatron import apply_liger_kernel_to_megatron

    original = unfused_ce.vocab_parallel_cross_entropy
    apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)

    assert unfused_ce.vocab_parallel_cross_entropy is not original
    assert unfused_ce.vocab_parallel_cross_entropy.__name__ == "liger_vocab_parallel_cross_entropy"


def test_patch_replaces_both_fused_and_unfused_symbols_in_one_call(fake_megatron):
    """A single ``cross_entropy=True`` call must replace both Megatron CE paths."""
    fused_ce, unfused_ce = fake_megatron
    from liger_kernel.megatron import apply_liger_kernel_to_megatron

    apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)

    assert fused_ce.fused_vocab_parallel_cross_entropy.__name__ == "liger_fused_vocab_parallel_cross_entropy"
    assert unfused_ce.vocab_parallel_cross_entropy.__name__ == "liger_vocab_parallel_cross_entropy"


def test_patch_with_cross_entropy_false_leaves_ce_symbols_untouched(fake_megatron):
    """Default ``cross_entropy=False`` must not touch the CE symbols even if the call runs."""
    fused_ce, unfused_ce = fake_megatron
    from liger_kernel.megatron import apply_liger_kernel_to_megatron

    fused_before = fused_ce.fused_vocab_parallel_cross_entropy
    unfused_before = unfused_ce.vocab_parallel_cross_entropy

    apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=False)

    assert fused_ce.fused_vocab_parallel_cross_entropy is fused_before
    assert unfused_ce.vocab_parallel_cross_entropy is unfused_before


def test_patch_is_idempotent_for_both_symbols(fake_megatron):
    """Calling ``apply_liger_kernel_to_megatron(cross_entropy=True)`` twice must not stack
    wrappers — the sentinel attribute guards against double-patching."""
    fused_ce, unfused_ce = fake_megatron
    from liger_kernel.megatron import apply_liger_kernel_to_megatron

    apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)
    fused_first = fused_ce.fused_vocab_parallel_cross_entropy
    unfused_first = unfused_ce.vocab_parallel_cross_entropy

    apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)
    # Same identity → no stacked wrapping.
    assert fused_ce.fused_vocab_parallel_cross_entropy is fused_first
    assert unfused_ce.vocab_parallel_cross_entropy is unfused_first
    # __wrapped__ still references the original Megatron symbol, not the first Liger wrapper.
    assert fused_first.__wrapped__.__name__ == "original_fused_vocab_parallel_cross_entropy"
    assert unfused_first.__wrapped__.__name__ == "original_vocab_parallel_cross_entropy"


def test_patch_fused_wrapper_passes_tp_group_through(fake_megatron):
    """The fused wrapper closure must forward ``tp_group`` to the underlying class.

    We swap the CE class for a recording fake so the call doesn't need CUDA — just
    confirms ``tp_group`` reaches the class's ``__call__``."""
    import torch

    fused_ce, _ = fake_megatron
    from liger_kernel.megatron import apply_liger_kernel_to_megatron
    from liger_kernel.megatron import cross_entropy as ce_mod

    captured = {}

    class _FakeCE:
        def __init__(self, ignore_index=-100, label_smoothing=0.0, reduction="none"):
            pass

        def __call__(self, logits, target, tp_group=None):
            captured["tp_group"] = tp_group
            captured["shape"] = tuple(logits.shape)
            return torch.zeros(logits.shape[:2])

    class _FakeGroup:
        def size(self):
            return 1

    with patch.object(ce_mod, "LigerMegatronCrossEntropy", _FakeCE):
        apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)
        logits = torch.zeros(2, 1, 4)
        target = torch.zeros(2, 1, dtype=torch.long)
        group = _FakeGroup()
        fused_ce.fused_vocab_parallel_cross_entropy(logits, target, group)

    assert captured["tp_group"] is group
    assert captured["shape"] == (2, 1, 4)


# ---------------------------------------------------------------------------
# TP-1 guard.
# ---------------------------------------------------------------------------


def test_patch_raises_on_tp_greater_than_one():
    _install_fake_megatron(tp_size=2)
    try:
        from liger_kernel.megatron import apply_liger_kernel_to_megatron

        with pytest.raises(RuntimeError, match="tensor_model_parallel_size=1"):
            apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)
    finally:
        _uninstall_fake_megatron()


def test_patch_defers_tp_check_when_parallel_state_not_initialized():
    """If get_tensor_model_parallel_world_size() raises, patch should still succeed."""
    fused_ce, unfused_ce = _install_fake_megatron(tp_size=1)

    def raising_tp_size():
        raise AssertionError("parallel_state not initialized")

    sys.modules["megatron.core.parallel_state"].get_tensor_model_parallel_world_size = raising_tp_size

    try:
        from liger_kernel.megatron import apply_liger_kernel_to_megatron

        apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)
        assert fused_ce.fused_vocab_parallel_cross_entropy.__name__ == "liger_fused_vocab_parallel_cross_entropy"
        assert unfused_ce.vocab_parallel_cross_entropy.__name__ == "liger_vocab_parallel_cross_entropy"
    finally:
        _uninstall_fake_megatron()


# ---------------------------------------------------------------------------
# Missing-megatron / missing-symbol errors.
# ---------------------------------------------------------------------------


def test_patch_raises_when_megatron_not_installed():
    _uninstall_fake_megatron()
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "megatron" or name.startswith("megatron."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=blocking_import):
        from liger_kernel.megatron import apply_liger_kernel_to_megatron

        with pytest.raises(ImportError, match="requires megatron-core"):
            apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)


def test_patch_raises_when_fused_symbol_missing():
    _install_fake_megatron(tp_size=1, with_fused_symbol=False)
    try:
        from liger_kernel.megatron import apply_liger_kernel_to_megatron

        with pytest.raises(ImportError, match="symbol path may have changed"):
            apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)
    finally:
        _uninstall_fake_megatron()


def test_patch_raises_when_unfused_symbol_missing():
    """Symmetric to the fused-missing case; the unfused module exists but its symbol doesn't."""
    _install_fake_megatron(tp_size=1, with_unfused_symbol=False)
    try:
        from liger_kernel.megatron import apply_liger_kernel_to_megatron

        with pytest.raises(ImportError, match="symbol path may have changed"):
            apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)
    finally:
        _uninstall_fake_megatron()


# ---------------------------------------------------------------------------
# Class-default construction + runtime label_smoothing override on the unfused path.
# ---------------------------------------------------------------------------


def test_patch_constructs_ce_with_class_defaults(fake_megatron):
    """The public ``apply_liger_kernel_to_megatron`` API exposes no CE-specific kwargs;
    the patch must therefore construct ``LigerMegatronCrossEntropy`` with class defaults.

    This intentionally matches Megatron's native fused-CE behavior (no ignore_index, no
    label_smoothing). Callers needing custom config use ``LigerMegatronCrossEntropy``
    directly (Mode 2)."""
    from liger_kernel.megatron import apply_liger_kernel_to_megatron
    from liger_kernel.megatron import cross_entropy as ce_mod

    captured = []
    real_ctor = ce_mod.LigerMegatronCrossEntropy.__init__

    def recording_init(self, ignore_index=-100, label_smoothing=0.0, reduction="none"):
        captured.append({
            "ignore_index": ignore_index,
            "label_smoothing": label_smoothing,
            "reduction": reduction,
        })
        real_ctor(self, ignore_index=ignore_index, label_smoothing=label_smoothing, reduction=reduction)

    with patch.object(ce_mod.LigerMegatronCrossEntropy, "__init__", recording_init):
        apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)

    # Fused wrapper builds 1 instance; unfused wrapper builds 1 default instance.
    assert len(captured) >= 2
    for entry in captured:
        assert entry == {"ignore_index": -100, "label_smoothing": 0.0, "reduction": "none"}


def test_unfused_wrapper_honors_runtime_label_smoothing(fake_megatron):
    """The unfused signature takes ``label_smoothing`` as a runtime arg; the wrapper must honor it.

    When the caller passes a non-default value, the wrapper constructs a fresh
    ``LigerMegatronCrossEntropy`` with that value rather than reusing the patch-time default.

    We verify this by replacing the class with a recording fake **before** calling
    ``apply_liger_kernel_to_megatron`` — the patch helper does a fresh
    ``from … import LigerMegatronCrossEntropy`` so the closure captures the fake.
    """
    import torch

    _, unfused_ce = fake_megatron
    from liger_kernel.megatron import apply_liger_kernel_to_megatron
    from liger_kernel.megatron import cross_entropy as ce_mod

    constructed = []

    class _FakeCE:
        def __init__(self, ignore_index=-100, label_smoothing=0.0, reduction="none"):
            constructed.append(label_smoothing)
            self.ignore_index = ignore_index
            self.label_smoothing = label_smoothing
            self.reduction = reduction

        def __call__(self, logits, target, tp_group=None):
            # Skip Liger kernel — just return a CPU-friendly tensor in the right shape.
            return torch.zeros(logits.shape[:2])

    with patch.object(ce_mod, "LigerMegatronCrossEntropy", _FakeCE):
        apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)

        # Reset the recorder to focus on calls triggered by the next line.
        constructed.clear()
        logits = torch.zeros(2, 1, 4)
        target = torch.zeros(2, 1, dtype=torch.long)
        unfused_ce.vocab_parallel_cross_entropy(logits, target, label_smoothing=0.3)

    assert constructed == [0.3], (
        f"unfused wrapper should construct one fresh instance with the runtime override; "
        f"got: {constructed}"
    )


def test_unfused_wrapper_uses_default_when_caller_does_not_pass_label_smoothing(fake_megatron):
    """When the caller doesn't pass ``label_smoothing``, the wrapper reuses the patch-time
    ``default_ce`` instance — no fresh allocation per call."""
    import torch

    _, unfused_ce = fake_megatron
    from liger_kernel.megatron import apply_liger_kernel_to_megatron
    from liger_kernel.megatron import cross_entropy as ce_mod

    constructed = []

    class _FakeCE:
        def __init__(self, ignore_index=-100, label_smoothing=0.0, reduction="none"):
            constructed.append(label_smoothing)

        def __call__(self, logits, target, tp_group=None):
            return torch.zeros(logits.shape[:2])

    with patch.object(ce_mod, "LigerMegatronCrossEntropy", _FakeCE):
        apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)
        constructed.clear()
        logits = torch.zeros(2, 1, 4)
        target = torch.zeros(2, 1, dtype=torch.long)
        # No label_smoothing arg — wrapper reuses the default_ce instance.
        unfused_ce.vocab_parallel_cross_entropy(logits, target)
        # Second positional call also without label_smoothing — still no new construction.
        unfused_ce.vocab_parallel_cross_entropy(logits, target)

    assert constructed == [], (
        f"default-path calls must reuse default_ce — no fresh instances; got: {constructed}"
    )


def test_unfused_wrapper_honors_explicit_zero_label_smoothing(fake_megatron):
    """Explicit ``label_smoothing=0.0`` at call time must be honored verbatim, not silently
    replaced by the patch-time default.

    This guards against the bug where the wrapper used ``if label_smoothing == 0.0:`` to
    detect "caller passed nothing" — that conflated "caller didn't pass" with "caller
    explicitly asked for 0.0" and corrupted loss math for Megatron callers that pass 0.0
    positionally."""
    import torch

    _, unfused_ce = fake_megatron
    from liger_kernel.megatron import apply_liger_kernel_to_megatron
    from liger_kernel.megatron import cross_entropy as ce_mod

    constructed = []

    class _FakeCE:
        def __init__(self, ignore_index=-100, label_smoothing=0.0, reduction="none"):
            constructed.append(label_smoothing)

        def __call__(self, logits, target, tp_group=None):
            return torch.zeros(logits.shape[:2])

    with patch.object(ce_mod, "LigerMegatronCrossEntropy", _FakeCE):
        apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)
        constructed.clear()
        logits = torch.zeros(2, 1, 4)
        target = torch.zeros(2, 1, dtype=torch.long)

        # Explicit positional 0.0 — must construct a fresh instance with 0.0.
        unfused_ce.vocab_parallel_cross_entropy(logits, target, 0.0)

    assert constructed == [0.0], (
        f"explicit label_smoothing=0.0 at call time must be honored verbatim; "
        f"got: {constructed}"
    )
