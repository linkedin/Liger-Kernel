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


# ---------------------------------------------------------------------------
# Beefier sweeps adapted from test/transformers/test_cross_entropy.py.
#
# LigerMegatronCrossEntropy is a [s, b, v] -> [s, b] reshape around Liger's
# CE op (reduction='none'). Numerical behavior should match the kernel itself,
# so the same parametrization patterns the kernel suite uses are the right
# coverage shape here — just adapted to the 3-D contract.
# ---------------------------------------------------------------------------


def _assign_ignore_index(target: torch.Tensor, ignore_index: int, frac: float = 0.25) -> None:
    """In-place: replace ~frac of target positions with ignore_index.

    Matches the transformers-side helpers that randomize the masked-out indices
    so the test isn't degenerate on a particular row layout.
    """
    flat = target.view(-1)
    n = max(1, int(flat.numel() * frac))
    idx = torch.randperm(flat.numel(), device=flat.device)[:n]
    flat[idx] = ignore_index


@pytest.mark.parametrize(
    "s, b, v",
    [
        (16, 1, 4096),
        (32, 2, 32000),  # llama-ish vocab
        (5, 3, 123),     # weird shape
    ],
)
@pytest.mark.parametrize("scalar", [0.5, 1.0, 5.0])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-7, 1e-6),
        pytest.param(
            torch.bfloat16,
            1e-2,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported"),
        ),
    ],
)
def test_class_correctness_scalar_sweep(s, b, v, scalar, dtype, atol, rtol):
    """Vary input magnitude — guards against numerical drift at large logit scales
    (mirrors the ``scalar`` parametrize in ``test/transformers/test_cross_entropy.py``)."""
    ce = LigerMegatronCrossEntropy()

    base = torch.randn(s, b, v, device=device, dtype=dtype) * scalar
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    # Backward parity: feed the same starting tensor through both paths.
    h_ref = base.detach().clone().requires_grad_(True)
    h_got = base.detach().clone().requires_grad_(True)

    ref = _reference_loss(h_ref, target, ignore_index=-100, label_smoothing=0.0)
    got = ce(h_got, target)

    assert got.shape == (s, b)
    assert_verbose_allclose(got.float(), ref.float(), atol=atol, rtol=rtol)

    ref.sum().backward()
    got.sum().backward()
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "s, b, v, ignore_index",
    [
        (16, 1, 4096, -100),  # standard hf sentinel
        (32, 2, 32000, 2),    # positive id (valid vocab slot used as ignore)
        (5, 3, 123, -123),    # weird negative
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-7, 1e-6),
        pytest.param(
            torch.bfloat16,
            1e-2,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported"),
        ),
    ],
)
def test_class_correctness_with_ignore_index_sweep(s, b, v, ignore_index, dtype, atol, rtol):
    """Broader ignore_index sweep including positive/negative sentinels and forward+backward
    correctness vs. PyTorch's reference. Mirrors transformers-side ``test_correctness_with_ignore_index``."""
    ce = LigerMegatronCrossEntropy(ignore_index=ignore_index)

    base = torch.randn(s, b, v, device=device, dtype=dtype)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)
    _assign_ignore_index(target, ignore_index, frac=0.3)

    h_ref = base.detach().clone().requires_grad_(True)
    h_got = base.detach().clone().requires_grad_(True)

    ref = _reference_loss(h_ref, target, ignore_index=ignore_index, label_smoothing=0.0)
    got = ce(h_got, target)
    assert_verbose_allclose(got.float(), ref.float(), atol=atol, rtol=rtol)

    ref.sum().backward()
    got.sum().backward()
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "s, b, v, ignore_index, label_smoothing",
    [
        (16, 1, 4096, 1, 0.1),
        (32, 2, 32000, -100, 0.2),
        (5, 3, 123, -300, 0.05),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-6, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-2,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported"),
        ),
    ],
)
def test_class_correctness_with_label_smoothing_and_ignore_index(
    s, b, v, ignore_index, label_smoothing, dtype, atol, rtol,
):
    """Combined ignore_index × label_smoothing sweep — the two are independent in Liger's CE
    kernel but mixing them historically surfaced bugs in the smoothing math. Mirrors
    ``test_correctness_with_label_smoothing_with_ignore_index_once`` from the kernel suite."""
    ce = LigerMegatronCrossEntropy(ignore_index=ignore_index, label_smoothing=label_smoothing)

    base = torch.randn(s, b, v, device=device, dtype=dtype)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)
    _assign_ignore_index(target, ignore_index, frac=0.25)

    h_ref = base.detach().clone().requires_grad_(True)
    h_got = base.detach().clone().requires_grad_(True)

    ref = _reference_loss(h_ref, target, ignore_index=ignore_index, label_smoothing=label_smoothing)
    got = ce(h_got, target)
    assert_verbose_allclose(got.float(), ref.float(), atol=atol, rtol=rtol)

    ref.sum().backward()
    got.sum().backward()
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "s, b, v",
    [
        (16, 1, 4096),
        (5, 3, 123),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-6, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-2,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported"),
        ),
    ],
)
def test_class_correctness_not_last_layer(s, b, v, dtype, atol, rtol):
    """Loss is multiplied by a downstream factor before ``.backward(grad_output)`` — verifies
    that Liger's in-place gradient write through the wrapper survives non-trivial chained
    autograd (i.e. CE isn't the last op in the graph). Mirrors transformers-side
    ``test_correctness_not_last_layer``."""
    ce = LigerMegatronCrossEntropy()

    base = torch.randn(s, b, v, device=device, dtype=dtype)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    h_ref = base.detach().clone().requires_grad_(True)
    h_got = base.detach().clone().requires_grad_(True)

    ref = _reference_loss(h_ref, target, ignore_index=-100, label_smoothing=0.0)
    got = ce(h_got, target)
    assert_verbose_allclose(got.float(), ref.float(), atol=atol, rtol=rtol)

    # Chain: loss = ref * 3 then backward with arbitrary grad_output.
    loss_ref = ref * 3.0
    loss_got = got * 3.0
    grad_out = torch.rand_like(ref)
    loss_ref.backward(gradient=grad_out)
    loss_got.backward(gradient=grad_out)
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("ignore_index", [-100, 2])
def test_class_rejects_out_of_bounds_target(ignore_index):
    """Liger's CE kernel asserts target ∈ [0, V); a stray out-of-bounds target should
    raise rather than silently produce garbage. Mirrors transformers-side
    ``test_correctness_with_out_of_bounds_target_once``."""
    s, b, v = 8, 2, 64
    ce = LigerMegatronCrossEntropy(ignore_index=ignore_index)

    logits = torch.randn(s, b, v, device=device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)
    # Plant a couple of out-of-bounds values; ignore_index is permitted but the
    # >=V poisoned slots are not.
    flat = target.view(-1)
    poison = torch.randperm(flat.numel(), device=flat.device)[:2]
    flat[poison] = v + 5  # >= V; the kernel-level assert should fire.

    with pytest.raises(AssertionError, match="out of bounds"):
        ce(logits, target)


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-7, 1e-6),
        pytest.param(
            torch.bfloat16,
            1e-2,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported"),
        ),
    ],
)
def test_class_correctness_forward_only(dtype, atol, rtol):
    """Forward-only path (under ``torch.no_grad()``) — verifies the wrapper still returns the
    right loss when autograd is disabled, AND that a subsequent ``.backward()`` raises the
    expected "does not require grad" error. Mirrors transformers-side ``test_correctness_with_forward_only``."""
    s, b, v = 16, 2, 1024
    ce = LigerMegatronCrossEntropy()

    logits_input = torch.randn(s, b, v, device=device, dtype=dtype)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    with torch.no_grad():
        # Clone the input separately for each path because Liger writes gradient
        # state in-place; sharing a buffer would corrupt the reference.
        ref = _reference_loss(logits_input.clone(), target, ignore_index=-100, label_smoothing=0.0)
        got = ce(logits_input.clone(), target)
        assert_verbose_allclose(got.float(), ref.float(), atol=atol, rtol=rtol)

    # Attempting backward on a forward-only output should raise.
    with pytest.raises(RuntimeError, match="does not require grad"):
        got.sum().backward()
