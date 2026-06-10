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


# ---------------------------------------------------------------------------
# Public-API surface checks (mirrors transformers-side ``test_import_from_root``
# and ``test_apply_liger_kernel_only_passes_valid_kwargs`` patterns).
# ---------------------------------------------------------------------------


def test_import_from_root():
    """All public Megatron symbols must be reachable from ``liger_kernel.megatron``.

    Mirrors the import-smoke pattern from ``test/transformers/test_monkey_patch.py``: catches
    accidental __init__.py removals so the docs' import snippets keep working."""
    try:
        from liger_kernel.megatron import LigerMegatronCrossEntropy  # noqa: F401
        from liger_kernel.megatron import LigerMegatronRMSNorm  # noqa: F401
        from liger_kernel.megatron import apply_liger_kernel_to_megatron  # noqa: F401
    except Exception:
        pytest.fail("Importing public Megatron symbols from liger_kernel.megatron failed.")


def test_public_apply_function_has_no_ce_specific_kwargs():
    """The framework-level patch entry point intentionally hides CE-specific knobs
    (ignore_index, label_smoothing, reduction). Catch accidental re-introduction —
    Mode-2 callers use ``LigerMegatronCrossEntropy`` directly for that config surface."""
    import inspect

    from liger_kernel.megatron import apply_liger_kernel_to_megatron

    sig = inspect.signature(apply_liger_kernel_to_megatron)
    leaked = {"ignore_index", "label_smoothing", "reduction"} & set(sig.parameters)
    assert not leaked, (
        f"apply_liger_kernel_to_megatron has re-grown CE-specific kwargs: {sorted(leaked)}. "
        f"Those belong on LigerMegatronCrossEntropy, not on the framework patch entry point."
    )


# ---------------------------------------------------------------------------
# End-to-end integration through the patched symbols.
#
# Earlier tests verify symbol identity + stub plumbing; the suite was missing
# the "patch + call with real tensors + check the numbers" coverage. These
# tests install the fake megatron, apply the patch, then invoke the resulting
# wrapper with live torch tensors and compare against ``F.cross_entropy``.
# That's the only way to catch wrapper-math bugs that pass the identity tests.
# ---------------------------------------------------------------------------


import torch  # noqa: E402  (deferred so the no-torch import-smoke tests above are unaffected)
import torch.nn.functional as F  # noqa: E402

from liger_kernel.utils import infer_device  # noqa: E402
from test.utils import assert_verbose_allclose  # noqa: E402

_device = infer_device()


def _ref_loss_sbv(logits_sbv: torch.Tensor, target_sb: torch.Tensor,
                  ignore_index: int = -100, label_smoothing: float = 0.0) -> torch.Tensor:
    """Reference CE for [s, b, v] logits / [s, b] target, returning [s, b]."""
    s, b, v = logits_sbv.shape
    loss_flat = F.cross_entropy(
        logits_sbv.reshape(-1, v).float(),
        target_sb.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    return loss_flat.reshape(s, b)


def test_patched_fused_symbol_computes_correct_loss(fake_megatron):
    """End-to-end: install stub megatron, patch, invoke the resulting fused symbol with real
    [s, b, v] logits, verify the loss matches ``F.cross_entropy``. Closes the gap between
    "patch wired correctly" (existing tests) and "patched function does the right math"."""
    fused_ce, _ = fake_megatron
    from liger_kernel.megatron import apply_liger_kernel_to_megatron

    apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)

    s, b, v = 16, 2, 1024
    torch.manual_seed(0)
    logits = torch.randn(s, b, v, device=_device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=_device, dtype=torch.long)

    ref = _ref_loss_sbv(logits.clone(), target)
    # Call through the patched symbol. tp_group=None is what Megatron's
    # LanguageModule passes when TP is uninitialized.
    got = fused_ce.fused_vocab_parallel_cross_entropy(logits.clone(), target, None)

    assert got.shape == (s, b)
    assert_verbose_allclose(got.float(), ref.float(), atol=1e-6, rtol=1e-5)


def test_patched_unfused_symbol_computes_correct_loss(fake_megatron):
    """Same as the fused case, but through the unfused symbol — verifies both wrappers
    are exercised and exercises the no-label_smoothing default branch (caller doesn't pass)."""
    _, unfused_ce = fake_megatron
    from liger_kernel.megatron import apply_liger_kernel_to_megatron

    apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)

    s, b, v = 8, 4, 512
    torch.manual_seed(1)
    logits = torch.randn(s, b, v, device=_device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=_device, dtype=torch.long)

    ref = _ref_loss_sbv(logits.clone(), target)
    # Native unfused signature: (logits, target, label_smoothing=0.0, tp_group=None).
    # Pass only positional args the caller normally would.
    got = unfused_ce.vocab_parallel_cross_entropy(logits.clone(), target)

    assert got.shape == (s, b)
    assert_verbose_allclose(got.float(), ref.float(), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_patched_unfused_symbol_runtime_label_smoothing_matches_pytorch(fake_megatron, label_smoothing):
    """The unfused wrapper's main feature beyond the fused path is honoring a runtime
    label_smoothing arg. Verify the resulting loss actually matches
    ``F.cross_entropy(..., label_smoothing=...)``, not just that a fresh CE instance is built."""
    _, unfused_ce = fake_megatron
    from liger_kernel.megatron import apply_liger_kernel_to_megatron

    apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)

    s, b, v = 8, 2, 256
    torch.manual_seed(2)
    logits = torch.randn(s, b, v, device=_device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=_device, dtype=torch.long)

    ref = _ref_loss_sbv(logits.clone(), target, label_smoothing=label_smoothing)
    got = unfused_ce.vocab_parallel_cross_entropy(
        logits.clone(), target, label_smoothing=label_smoothing,
    )
    assert_verbose_allclose(got.float(), ref.float(), atol=1e-5, rtol=1e-4)


def test_patched_fused_symbol_preserves_gradients(fake_megatron):
    """Backward through the patched fused symbol: gradient shape + parity vs.
    PyTorch's reference. Liger writes the gradient back into the input buffer,
    so verifying ``.grad`` after backward exercises both the reshape contract
    and the in-place write."""
    fused_ce, _ = fake_megatron
    from liger_kernel.megatron import apply_liger_kernel_to_megatron

    apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)

    s, b, v = 8, 2, 256
    torch.manual_seed(3)
    base = torch.randn(s, b, v, device=_device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=_device, dtype=torch.long)

    h_ref = base.detach().clone().requires_grad_(True)
    h_got = base.detach().clone().requires_grad_(True)
    ref = _ref_loss_sbv(h_ref, target)
    got = fused_ce.fused_vocab_parallel_cross_entropy(h_got, target, None)

    ref.sum().backward()
    got.sum().backward()

    assert h_got.grad is not None
    assert h_got.grad.shape == h_got.shape
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=1e-6, rtol=1e-5)


def test_patched_unfused_symbol_preserves_gradients(fake_megatron):
    """Symmetric to the fused-gradient test; ensures the closure in
    ``_patch_vocab_parallel_cross_entropy`` doesn't break autograd."""
    _, unfused_ce = fake_megatron
    from liger_kernel.megatron import apply_liger_kernel_to_megatron

    apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)

    s, b, v = 8, 2, 256
    torch.manual_seed(4)
    base = torch.randn(s, b, v, device=_device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=_device, dtype=torch.long)

    h_ref = base.detach().clone().requires_grad_(True)
    h_got = base.detach().clone().requires_grad_(True)
    ref = _ref_loss_sbv(h_ref, target)
    got = unfused_ce.vocab_parallel_cross_entropy(h_got, target)

    ref.sum().backward()
    got.sum().backward()
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=1e-6, rtol=1e-5)


def test_patched_fused_symbol_default_ignore_index_minus_100(fake_megatron):
    """Patch-time defaults: targets containing -100 should be treated as ignored — Liger's
    kernel zeros those loss positions, matching ``F.cross_entropy(ignore_index=-100)``.

    Native Megatron's fused CE has no ignore_index concept and would silently produce
    garbage on -100; this is one place where Liger is strictly better than the symbol
    it replaces, and the test pins that behavior."""
    fused_ce, _ = fake_megatron
    from liger_kernel.megatron import apply_liger_kernel_to_megatron

    apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=True)

    s, b, v = 8, 2, 128
    torch.manual_seed(5)
    logits = torch.randn(s, b, v, device=_device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=_device, dtype=torch.long)
    # Plant some -100 sentinel positions.
    flat = target.view(-1)
    flat[: flat.numel() // 4] = -100

    ref = _ref_loss_sbv(logits.clone(), target, ignore_index=-100)
    got = fused_ce.fused_vocab_parallel_cross_entropy(logits.clone(), target, None)

    # Per-token loss at masked positions should be exactly 0.
    mask = (target != -100).float()
    assert torch.all(got * (1 - mask) == 0)
    assert_verbose_allclose(got.float(), ref.float(), atol=1e-6, rtol=1e-5)


def test_rms_norm_only_patch_does_not_touch_ce_symbols(fake_megatron):
    """Symmetric to ``test_patch_with_cross_entropy_false_leaves_ce_symbols_untouched``,
    but for the opposite split. With ``rms_norm=True, cross_entropy=False`` (RMSNorm
    helpers require real megatron and will ImportError on the stub — that's fine, we
    only need to confirm the CE symbols are not pre-emptively touched before the RMSNorm
    helpers run). Documenting this protects against future apply_… reorderings that would
    silently couple the two."""
    fused_ce, unfused_ce = fake_megatron
    from liger_kernel.megatron import apply_liger_kernel_to_megatron

    fused_before = fused_ce.fused_vocab_parallel_cross_entropy
    unfused_before = unfused_ce.vocab_parallel_cross_entropy

    # RMSNorm helpers do their own megatron import; on the stub they'll raise. Catch
    # any exception so the assertion at the end runs regardless — we only care that the
    # CE symbols weren't touched.
    try:
        apply_liger_kernel_to_megatron(rms_norm=True, cross_entropy=False)
    except Exception:
        pass

    assert fused_ce.fused_vocab_parallel_cross_entropy is fused_before
    assert unfused_ce.vocab_parallel_cross_entropy is unfused_before
