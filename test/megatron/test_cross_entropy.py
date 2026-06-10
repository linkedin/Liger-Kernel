"""Correctness tests for ``LigerMegatronCrossEntropy``.

The class is the public Mode-2 API; the monkey-patch wrappers in
``monkey_patch.py`` are thin closures around an instance of this class. Tests
target the class directly — that's the single source of truth for the CE math.

Mirrors ``test/megatron/test_rms_norm.py``'s parametrize style for the
fp32/bf16 sweep so the visual symmetry across the two megatron-side files is
preserved.
"""

import pytest
import torch
import torch.nn.functional as F

from liger_kernel.megatron import LigerMegatronCrossEntropy
from liger_kernel.utils import infer_device
from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

device = infer_device()
set_seed(42)


def _reference_loss(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    label_smoothing: float,
) -> torch.Tensor:
    s, b, v = vocab_parallel_logits.shape
    loss_flat = F.cross_entropy(
        vocab_parallel_logits.reshape(-1, v).float(),
        target.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    return loss_flat.reshape(s, b)


# ---------------------------------------------------------------------------
# Forward correctness vs. F.cross_entropy reference.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "s, b, v",
    [
        (8, 2, 128),
        (16, 4, 4096),
        (32, 1, 32000),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-7, 1e-6),
        pytest.param(
            torch.bfloat16,
            1e-2,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported"),
        ),
    ],
)
def test_class_matches_pytorch_cross_entropy(s, b, v, dtype, atol, rtol):
    ce = LigerMegatronCrossEntropy()

    logits = torch.randn(s, b, v, device=device, dtype=dtype) * 0.5
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    ref = _reference_loss(logits, target, ignore_index=-100, label_smoothing=0.0)
    got = ce(logits, target)

    assert got.shape == (s, b)
    assert_verbose_allclose(got.float(), ref.float(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Configuration plumbing — wrapper-specific contracts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ignore_index", [-100, 0])
def test_class_respects_ignore_index(ignore_index):
    s, b, v = 16, 2, 1024
    ce = LigerMegatronCrossEntropy(ignore_index=ignore_index)

    logits = torch.randn(s, b, v, device=device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)
    target.view(-1)[: (s * b) // 4] = ignore_index

    ref = _reference_loss(logits, target, ignore_index=ignore_index, label_smoothing=0.0)
    got = ce(logits, target)
    assert_verbose_allclose(got.float(), ref.float(), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_class_respects_label_smoothing(label_smoothing):
    s, b, v = 8, 2, 512
    ce = LigerMegatronCrossEntropy(label_smoothing=label_smoothing)

    logits = torch.randn(s, b, v, device=device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    ref = _reference_loss(logits, target, ignore_index=-100, label_smoothing=label_smoothing)
    got = ce(logits, target)
    assert_verbose_allclose(got.float(), ref.float(), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("bad_reduction", ["mean", "sum", "MEAN", "garbage"])
def test_class_rejects_non_none_reduction(bad_reduction):
    """Megatron's contract is per-token loss; mean/sum break the [s, b] return shape."""
    with pytest.raises(ValueError, match="reduction must be 'none'"):
        LigerMegatronCrossEntropy(reduction=bad_reduction)


def test_class_rejects_non_3d_logits():
    """The class explicitly guards against HuggingFace-shape [b, s, v] callers etc."""
    ce = LigerMegatronCrossEntropy()
    bad = torch.randn(8, 16, device=device)               # 2-D
    target = torch.randint(0, 16, (8,), device=device, dtype=torch.long)
    with pytest.raises(ValueError, match="3-D"):
        ce(bad, target)

    too_many = torch.randn(2, 2, 4, 16, device=device)    # 4-D
    target2 = torch.randint(0, 16, (2, 2, 4), device=device, dtype=torch.long)
    with pytest.raises(ValueError, match="3-D"):
        ce(too_many, target2)


# ---------------------------------------------------------------------------
# TP guard — the only safety net the class itself enforces.
# ---------------------------------------------------------------------------


def test_class_raises_on_tp_group_size_greater_than_one():
    ce = LigerMegatronCrossEntropy()
    logits = torch.randn(4, 1, 32, device=device)
    target = torch.randint(0, 32, (4, 1), device=device, dtype=torch.long)

    class _FakeGroup:
        def size(self):
            return 2

    with pytest.raises(RuntimeError, match="tensor_model_parallel_size=1"):
        ce(logits, target, tp_group=_FakeGroup())


def test_class_accepts_single_rank_tp_group():
    ce = LigerMegatronCrossEntropy()
    logits = torch.randn(4, 1, 32, device=device)
    target = torch.randint(0, 32, (4, 1), device=device, dtype=torch.long)

    class _FakeGroup:
        def size(self):
            return 1

    out = ce(logits, target, tp_group=_FakeGroup())
    assert out.shape == (4, 1)


# ---------------------------------------------------------------------------
# Gradient sanity — Liger's CE writes gradients in place; verify the class
# preserves them through Megatron's [s, b, v] reshape contract.
# ---------------------------------------------------------------------------


def test_class_preserves_gradients():
    s, b, v = 8, 2, 256
    ce = LigerMegatronCrossEntropy()

    logits = torch.randn(s, b, v, device=device, dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    loss = ce(logits, target).sum()
    loss.backward()

    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def test_class_extra_repr():
    ce = LigerMegatronCrossEntropy(ignore_index=42, label_smoothing=0.07)
    rep = ce.extra_repr()
    assert "ignore_index=42" in rep
    assert "label_smoothing=0.07" in rep
    assert "reduction='none'" in rep
