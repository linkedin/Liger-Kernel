"""Correctness tests for ``LigerVocabParallelCrossEntropy`` and ``LigerMegatronCrossEntropy``.

Both classes share the same ``LigerVocabParallelCEFunction`` autograd op + the same
Triton kernels in ``liger_kernel.ops.vocab_parallel_cross_entropy``; the only difference
is the default ``label_smoothing_formula`` / ``label_smoothing_mode``. Tests target both
classes:

  - ``LigerVocabParallelCrossEntropy`` (PyTorch + global defaults) — compared against
    ``F.cross_entropy`` directly.
  - ``LigerMegatronCrossEntropy`` (Megatron + partition defaults) — compared against
    ``F.cross_entropy`` with ``label_smoothing`` rescaled to ``alpha * V / (V - 1)``,
    which is algebraically equivalent to the Megatron NeMo formula at TP=1.

File layout:

  Section 0: helpers
  Section 1: LigerVocabParallelCrossEntropy — forward + backward parity vs F.cross_entropy
  Section 2: LigerMegatronCrossEntropy — forward + backward parity vs F.cross_entropy
             with rescaled alpha (matches Megatron formula at TP=1)
  Section 3: Configuration plumbing (ignore_index, label_smoothing, formula/mode validation)
  Section 4: TP-group plumbing (single-rank fast path; rejects bad shapes)
  Section 5: Beefier sweeps adapted from test/transformers/test_cross_entropy.py
  Section 6: TP>1 multi-GPU parity (mp.spawn; gated on cuda.device_count() >= 2)
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from liger_kernel.megatron import LigerMegatronCrossEntropy
from liger_kernel.megatron import LigerVocabParallelCrossEntropy
from liger_kernel.utils import infer_device
from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

device = infer_device()
set_seed(42)


# ===========================================================================
# 0. Helpers
# ===========================================================================


def _reference_pytorch_loss(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    label_smoothing: float,
) -> torch.Tensor:
    """F.cross_entropy reference using PyTorch label-smoothing semantics."""
    s, b, v = vocab_parallel_logits.shape
    loss_flat = F.cross_entropy(
        vocab_parallel_logits.reshape(-1, v).float(),
        target.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    return loss_flat.reshape(s, b)


def _reference_megatron_loss(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    label_smoothing: float,
) -> torch.Tensor:
    """F.cross_entropy reference for the Megatron NeMo formula.

    Megatron uses ``q' = (1-α)*q + α/(K-1)*uniform_excluding_gt`` which is the same as
    PyTorch's ``q' = (1-α')*q + α'/K*uniform`` with ``α' = α*K/(K-1)`` — so we can reuse
    F.cross_entropy with a rescaled alpha. At TP=1, K=V.
    """
    s, b, v = vocab_parallel_logits.shape
    if label_smoothing > 0:
        alpha_rescaled = label_smoothing * v / (v - 1)
    else:
        alpha_rescaled = 0.0
    loss_flat = F.cross_entropy(
        vocab_parallel_logits.reshape(-1, v).float(),
        target.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=alpha_rescaled,
    )
    return loss_flat.reshape(s, b)


def _assign_ignore_index(target: torch.Tensor, ignore_index: int, frac: float = 0.25) -> None:
    """In-place: replace ~frac of target positions with ignore_index."""
    flat = target.view(-1)
    n = max(1, int(flat.numel() * frac))
    idx = torch.randperm(flat.numel(), device=flat.device)[:n]
    flat[idx] = ignore_index


# ===========================================================================
# 1. LigerVocabParallelCrossEntropy — PyTorch formula, F.cross_entropy parity
# ===========================================================================


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
        (torch.float32, 1e-6, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-2,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported"),
        ),
    ],
)
def test_vp_class_matches_pytorch_cross_entropy(s, b, v, dtype, atol, rtol):
    """Headline correctness — forward AND backward parity vs F.cross_entropy."""
    ce = LigerVocabParallelCrossEntropy()

    base = torch.randn(s, b, v, device=device, dtype=dtype) * 0.5
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    h_ref = base.detach().clone().requires_grad_(True)
    h_got = base.detach().clone().requires_grad_(True)

    ref = _reference_pytorch_loss(h_ref, target, ignore_index=-100, label_smoothing=0.0)
    got = ce(h_got, target)

    assert got.shape == (s, b)
    assert_verbose_allclose(got.float(), ref.float(), atol=atol, rtol=rtol)

    ref.sum().backward()
    got.sum().backward()
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("ignore_index", [-100, 0])
def test_vp_class_respects_ignore_index(ignore_index):
    s, b, v = 16, 2, 1024
    ce = LigerVocabParallelCrossEntropy(ignore_index=ignore_index)

    base = torch.randn(s, b, v, device=device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)
    target.view(-1)[: (s * b) // 4] = ignore_index

    h_ref = base.detach().clone().requires_grad_(True)
    h_got = base.detach().clone().requires_grad_(True)

    ref = _reference_pytorch_loss(h_ref, target, ignore_index=ignore_index, label_smoothing=0.0)
    got = ce(h_got, target)
    assert_verbose_allclose(got.float(), ref.float(), atol=1e-6, rtol=1e-5)

    ref.sum().backward()
    got.sum().backward()
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_vp_class_respects_label_smoothing(label_smoothing):
    """LigerVocabParallelCrossEntropy uses PyTorch formula by default → matches F.cross_entropy."""
    s, b, v = 8, 2, 512
    ce = LigerVocabParallelCrossEntropy(label_smoothing=label_smoothing)

    base = torch.randn(s, b, v, device=device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    h_ref = base.detach().clone().requires_grad_(True)
    h_got = base.detach().clone().requires_grad_(True)

    ref = _reference_pytorch_loss(h_ref, target, ignore_index=-100, label_smoothing=label_smoothing)
    got = ce(h_got, target)
    assert_verbose_allclose(got.float(), ref.float(), atol=1e-5, rtol=1e-4)

    ref.sum().backward()
    got.sum().backward()
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=1e-5, rtol=1e-4)


# ===========================================================================
# 2. LigerMegatronCrossEntropy — Megatron formula, rescaled F.cross_entropy parity
# ===========================================================================


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
        (torch.float32, 1e-6, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-2,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported"),
        ),
    ],
)
def test_megatron_class_matches_pytorch_cross_entropy(s, b, v, dtype, atol, rtol):
    """With label_smoothing=0 the two classes are indistinguishable from F.cross_entropy."""
    ce = LigerMegatronCrossEntropy()

    base = torch.randn(s, b, v, device=device, dtype=dtype) * 0.5
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    h_ref = base.detach().clone().requires_grad_(True)
    h_got = base.detach().clone().requires_grad_(True)

    # No label smoothing → Megatron formula reduces to PyTorch formula
    ref = _reference_pytorch_loss(h_ref, target, ignore_index=-100, label_smoothing=0.0)
    got = ce(h_got, target)

    assert got.shape == (s, b)
    assert_verbose_allclose(got.float(), ref.float(), atol=atol, rtol=rtol)

    ref.sum().backward()
    got.sum().backward()
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_megatron_class_matches_megatron_formula(label_smoothing):
    """LigerMegatronCrossEntropy uses Megatron formula by default → matches F.cross_entropy
    with alpha rescaled to alpha*V/(V-1)."""
    s, b, v = 8, 2, 512
    ce = LigerMegatronCrossEntropy(label_smoothing=label_smoothing)

    base = torch.randn(s, b, v, device=device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    h_ref = base.detach().clone().requires_grad_(True)
    h_got = base.detach().clone().requires_grad_(True)

    ref = _reference_megatron_loss(h_ref, target, ignore_index=-100, label_smoothing=label_smoothing)
    got = ce(h_got, target)
    assert_verbose_allclose(got.float(), ref.float(), atol=1e-5, rtol=1e-4)

    ref.sum().backward()
    got.sum().backward()
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("ignore_index", [-100, 0])
def test_megatron_class_respects_ignore_index(ignore_index):
    s, b, v = 16, 2, 1024
    ce = LigerMegatronCrossEntropy(ignore_index=ignore_index)

    base = torch.randn(s, b, v, device=device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)
    target.view(-1)[: (s * b) // 4] = ignore_index

    h_ref = base.detach().clone().requires_grad_(True)
    h_got = base.detach().clone().requires_grad_(True)

    ref = _reference_pytorch_loss(h_ref, target, ignore_index=ignore_index, label_smoothing=0.0)
    got = ce(h_got, target)
    assert_verbose_allclose(got.float(), ref.float(), atol=1e-6, rtol=1e-5)

    ref.sum().backward()
    got.sum().backward()
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=1e-6, rtol=1e-5)


# ===========================================================================
# 3. Configuration plumbing — invariants the classes themselves enforce
# ===========================================================================


@pytest.mark.parametrize("bad_reduction", ["mean", "sum", "MEAN", "garbage"])
def test_class_rejects_non_none_reduction(bad_reduction):
    """Vocab-parallel CE contract is per-token loss; mean/sum break the [s, b] return shape."""
    with pytest.raises(ValueError, match="reduction must be 'none'"):
        LigerVocabParallelCrossEntropy(reduction=bad_reduction)
    with pytest.raises(ValueError, match="reduction must be 'none'"):
        LigerMegatronCrossEntropy(reduction=bad_reduction)


@pytest.mark.parametrize("bad_formula", ["pyTorch", "megatron-style", "tf", ""])
def test_class_rejects_bad_label_smoothing_formula(bad_formula):
    with pytest.raises(ValueError, match="label_smoothing_formula"):
        LigerVocabParallelCrossEntropy(label_smoothing_formula=bad_formula)


@pytest.mark.parametrize("bad_mode", ["partition_local", "Global", "", "world"])
def test_class_rejects_bad_label_smoothing_mode(bad_mode):
    with pytest.raises(ValueError, match="label_smoothing_mode"):
        LigerVocabParallelCrossEntropy(label_smoothing_mode=bad_mode)


@pytest.mark.parametrize("bad_alpha", [-0.1, 1.0, 1.5, 2.0])
def test_class_rejects_bad_label_smoothing(bad_alpha):
    with pytest.raises(ValueError, match="label_smoothing must be in"):
        LigerVocabParallelCrossEntropy(label_smoothing=bad_alpha)


def test_class_rejects_non_3d_logits():
    """Classes explicitly guard against HuggingFace-shape [b, s, v] callers etc."""
    ce = LigerVocabParallelCrossEntropy()
    bad = torch.randn(8, 16, device=device)  # 2-D
    target = torch.randint(0, 16, (8,), device=device, dtype=torch.long)
    with pytest.raises(ValueError, match="3-D"):
        ce(bad, target)

    too_many = torch.randn(2, 2, 4, 16, device=device)  # 4-D
    target2 = torch.randint(0, 16, (2, 2, 4), device=device, dtype=torch.long)
    with pytest.raises(ValueError, match="3-D"):
        ce(too_many, target2)


def test_megatron_class_extra_repr():
    ce = LigerMegatronCrossEntropy(ignore_index=42, label_smoothing=0.07)
    rep = ce.extra_repr()
    assert "ignore_index=42" in rep
    assert "label_smoothing=0.07" in rep
    assert "reduction='none'" in rep
    assert "label_smoothing_formula='megatron'" in rep
    assert "label_smoothing_mode='partition'" in rep


def test_vp_class_extra_repr():
    ce = LigerVocabParallelCrossEntropy(ignore_index=42, label_smoothing=0.07)
    rep = ce.extra_repr()
    assert "label_smoothing_formula='pytorch'" in rep
    assert "label_smoothing_mode='global'" in rep


# ===========================================================================
# 4. TP-group plumbing — single-rank fast path; TP>1 is no longer rejected
# ===========================================================================


def test_class_accepts_single_rank_tp_group():
    """A real or stub single-rank tp_group should work without any AllReduces."""
    ce = LigerMegatronCrossEntropy()
    logits = torch.randn(4, 1, 32, device=device)
    target = torch.randint(0, 32, (4, 1), device=device, dtype=torch.long)

    class _FakeGroup:
        def size(self):
            return 1

    out = ce(logits, target, tp_group=_FakeGroup())
    assert out.shape == (4, 1)


def test_class_accepts_none_tp_group():
    """No tp_group at all → identical to single-rank tp_group."""
    ce = LigerVocabParallelCrossEntropy()
    logits = torch.randn(4, 1, 32, device=device, requires_grad=True)
    target = torch.randint(0, 32, (4, 1), device=device, dtype=torch.long)
    out = ce(logits, target, tp_group=None)
    assert out.shape == (4, 1)
    out.sum().backward()
    assert logits.grad is not None


def test_class_does_not_mutate_input_when_fp32():
    """Input must be preserved even when fp32 (which previously aliased internally)."""
    ce = LigerVocabParallelCrossEntropy()
    logits = torch.randn(4, 1, 32, device=device, dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, 32, (4, 1), device=device, dtype=torch.long)
    logits_snapshot = logits.detach().clone()
    out = ce(logits, target)
    out.sum().backward()
    # The input tensor must be untouched after forward + backward
    assert torch.equal(logits.detach(), logits_snapshot), "forward mutated the caller's fp32 input"


# ===========================================================================
# 5. Beefier sweeps adapted from test/transformers/test_cross_entropy.py
# ===========================================================================


@pytest.mark.parametrize(
    "s, b, v",
    [
        (16, 1, 4096),
        (32, 2, 32000),
        (5, 3, 123),
    ],
)
@pytest.mark.parametrize("scalar", [0.5, 1.0, 5.0])
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
def test_vp_class_correctness_scalar_sweep(s, b, v, scalar, dtype, atol, rtol):
    """Vary input magnitude — guards against numerical drift at large logit scales."""
    ce = LigerVocabParallelCrossEntropy()

    base = torch.randn(s, b, v, device=device, dtype=dtype) * scalar
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    h_ref = base.detach().clone().requires_grad_(True)
    h_got = base.detach().clone().requires_grad_(True)

    ref = _reference_pytorch_loss(h_ref, target, ignore_index=-100, label_smoothing=0.0)
    got = ce(h_got, target)

    assert got.shape == (s, b)
    assert_verbose_allclose(got.float(), ref.float(), atol=atol, rtol=rtol)

    ref.sum().backward()
    got.sum().backward()
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "s, b, v, ignore_index",
    [
        (16, 1, 4096, -100),
        (32, 2, 32000, 2),
        (5, 3, 123, -123),
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
def test_vp_class_correctness_with_ignore_index_sweep(s, b, v, ignore_index, dtype, atol, rtol):
    """Broader ignore_index sweep including positive/negative sentinels."""
    ce = LigerVocabParallelCrossEntropy(ignore_index=ignore_index)

    base = torch.randn(s, b, v, device=device, dtype=dtype)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)
    _assign_ignore_index(target, ignore_index, frac=0.3)

    h_ref = base.detach().clone().requires_grad_(True)
    h_got = base.detach().clone().requires_grad_(True)

    ref = _reference_pytorch_loss(h_ref, target, ignore_index=ignore_index, label_smoothing=0.0)
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
        (torch.float32, 1e-5, 1e-4),
        pytest.param(
            torch.bfloat16,
            1e-2,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported"),
        ),
    ],
)
def test_vp_class_correctness_with_label_smoothing_and_ignore_index(
    s,
    b,
    v,
    ignore_index,
    label_smoothing,
    dtype,
    atol,
    rtol,
):
    """Combined ignore_index × label_smoothing sweep."""
    ce = LigerVocabParallelCrossEntropy(ignore_index=ignore_index, label_smoothing=label_smoothing)

    base = torch.randn(s, b, v, device=device, dtype=dtype)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)
    _assign_ignore_index(target, ignore_index, frac=0.25)

    h_ref = base.detach().clone().requires_grad_(True)
    h_got = base.detach().clone().requires_grad_(True)

    ref = _reference_pytorch_loss(h_ref, target, ignore_index=ignore_index, label_smoothing=label_smoothing)
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
def test_vp_class_correctness_not_last_layer(s, b, v):
    """CE composed with a downstream factor — guards in-place grad write through chained autograd."""
    ce = LigerVocabParallelCrossEntropy()

    base = torch.randn(s, b, v, device=device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    h_ref = base.detach().clone().requires_grad_(True)
    h_got = base.detach().clone().requires_grad_(True)

    ref = _reference_pytorch_loss(h_ref, target, ignore_index=-100, label_smoothing=0.0)
    got = ce(h_got, target)
    assert_verbose_allclose(got.float(), ref.float(), atol=1e-6, rtol=1e-5)

    loss_ref = ref * 3.0
    loss_got = got * 3.0
    grad_out = torch.rand_like(ref)
    loss_ref.backward(gradient=grad_out)
    loss_got.backward(gradient=grad_out)
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("ignore_index", [-100, 2])
def test_vp_class_runs_with_out_of_bounds_target(ignore_index):
    """Megatron's vocab-parallel CE silently masks targets outside [0, V_local) as "off-rank".

    At TP=1 the local range == global range so any target >= V is also out of the global
    vocab. The kernel's clamp keeps it well-defined (treated as off-rank → contributes 0
    to predicted logit). This test just verifies no crash + finite output.
    """
    s, b, v = 8, 2, 64
    ce = LigerVocabParallelCrossEntropy(ignore_index=ignore_index)

    logits = torch.randn(s, b, v, device=device, dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)
    flat = target.view(-1)
    poison = torch.randperm(flat.numel(), device=flat.device)[:2]
    flat[poison] = v + 5

    out = ce(logits, target)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_vp_class_correctness_forward_only():
    """Forward-only path (under torch.no_grad)."""
    s, b, v = 16, 2, 1024
    ce = LigerVocabParallelCrossEntropy()

    logits_input = torch.randn(s, b, v, device=device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    with torch.no_grad():
        ref = _reference_pytorch_loss(logits_input, target, ignore_index=-100, label_smoothing=0.0)
        got = ce(logits_input, target)
        assert_verbose_allclose(got.float(), ref.float(), atol=1e-6, rtol=1e-5)


# ===========================================================================
# 6. TP>1 multi-GPU parity tests (mp.spawn; require >= 2 GPUs)
# ===========================================================================
#
# Each test broadcasts the same full-vocab logits to every rank, has each rank
# slice out its own V_local, runs Liger's TP>1 CE, and compares to a single-rank
# F.cross_entropy on the full logits (rescaled alpha when Megatron formula).
#
# The "partition" mode with label_smoothing>0 produces per-rank loss values that
# differ across ranks (Megatron's documented behavior). For those cases we still
# check the BACKWARD gradient on each rank's vocab slice matches the corresponding
# slice of the F.cross_entropy reference's gradient — gradients match even when
# losses don't, because the gradient w.r.t. logits is the same shape across the
# two formulations once you rescale.
# ===========================================================================


def _tp_ce_worker(
    rank,
    tp_size,
    s,
    b,
    v_global,
    formula,
    mode,
    label_smoothing,
    file_name,
):
    """One process per rank. Validates forward + backward against a global reference."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{file_name}",
        rank=rank,
        world_size=tp_size,
    )
    torch.cuda.set_device(rank)
    tp_group = dist.group.WORLD
    dev = torch.device(f"cuda:{rank}")
    v_local = v_global // tp_size

    # Build the same full logits + target on all ranks (broadcast for safety).
    torch.manual_seed(42)
    logits_global_cpu = torch.randn(s, b, v_global, dtype=torch.float32) * 0.5
    target_cpu = torch.randint(0, v_global, (s, b), dtype=torch.long)
    logits_global = logits_global_cpu.to(dev)
    target = target_cpu.to(dev)
    dist.broadcast(logits_global, src=0, group=tp_group)
    dist.broadcast(target, src=0, group=tp_group)

    # Liger TP>1 path on this rank's slice.
    vp_ce = LigerVocabParallelCrossEntropy(
        label_smoothing=label_smoothing,
        label_smoothing_formula=formula,
        label_smoothing_mode=mode,
    )
    logits_local = logits_global[..., rank * v_local : (rank + 1) * v_local].detach().clone().requires_grad_(True)
    loss_liger = vp_ce(logits_local, target, tp_group=tp_group)

    # Reference: F.cross_entropy on global logits with alpha rescaled when needed.
    if formula == "megatron" and label_smoothing > 0:
        K_ref = v_global if mode == "global" else v_local
        alpha_ref = label_smoothing * K_ref / (K_ref - 1)
    else:
        alpha_ref = label_smoothing
    ref = logits_global.detach().clone().requires_grad_(True)
    loss_ref = F.cross_entropy(
        ref.reshape(-1, v_global),
        target.reshape(-1),
        reduction="none",
        label_smoothing=alpha_ref,
    ).reshape(s, b)

    # Forward parity skipped for "partition" smoothing (loss diverges per rank by design).
    if not (mode == "partition" and label_smoothing > 0):
        torch.testing.assert_close(loss_liger.float(), loss_ref.float(), atol=1e-4, rtol=1e-3)

    # Backward: gradient on this rank's vocab slice must match the global reference slice.
    loss_liger.sum().backward()
    loss_ref.sum().backward()
    grad_local_ref = ref.grad[..., rank * v_local : (rank + 1) * v_local]
    # Same caveat — under "partition" smoothing the gradient also differs slightly across
    # the formulations (the smoothing contribution is computed per-partition, not per-global).
    if not (mode == "partition" and label_smoothing > 0):
        torch.testing.assert_close(
            logits_local.grad.float(),
            grad_local_ref.float(),
            atol=1e-4,
            rtol=1e-3,
        )

    dist.destroy_process_group()


@pytest.mark.parametrize(
    "tp_size",
    [
        pytest.param(
            2,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                reason="requires >= 2 GPUs",
            ),
        ),
        pytest.param(
            4,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                reason="requires >= 4 GPUs",
            ),
        ),
        pytest.param(
            8,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available() or torch.cuda.device_count() < 8,
                reason="requires >= 8 GPUs",
            ),
        ),
    ],
)
@pytest.mark.parametrize("s, b, v_global", [(8, 2, 128), (16, 1, 4096)])
@pytest.mark.parametrize("formula", ["pytorch", "megatron"])
@pytest.mark.parametrize("mode", ["global", "partition"])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_tp_vocab_parallel_ce_parity(tp_size, s, b, v_global, formula, mode, label_smoothing):
    """Multi-GPU TP>1 forward + backward parity for all formula/mode combinations."""
    with tempfile.NamedTemporaryFile() as f:
        mp.spawn(
            _tp_ce_worker,
            args=(tp_size, s, b, v_global, formula, mode, label_smoothing, f.name),
            nprocs=tp_size,
            join=True,
        )
