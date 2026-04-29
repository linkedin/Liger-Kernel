"""Correctness tests for the Liger Megatron cross-entropy wrapper.

These tests exercise ``_build_wrapper`` directly without importing
megatron-core — the wrapper is the [s, b, v] -> [s, b] reshape shim around
``LigerCrossEntropyLoss`` and is meaningful to test on its own.

The wrapper calls the underlying Triton kernel, so these tests require a
Liger-supported accelerator (same as ``test/transformers/test_cross_entropy.py``).
"""

import pytest
import torch
import torch.nn.functional as F

from liger_kernel.megatron.cross_entropy import _build_wrapper
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.utils import infer_device
from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

device = infer_device()
set_seed(42)


def _make_wrapper(ignore_index: int = -100, label_smoothing: float = 0.0):
    loss_fn = LigerCrossEntropyLoss(
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    return _build_wrapper(loss_fn)


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
def test_wrapper_matches_pytorch_cross_entropy(s, b, v, dtype, atol, rtol):
    wrapper = _make_wrapper()

    logits = torch.randn(s, b, v, device=device, dtype=dtype) * 0.5
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    ref = _reference_loss(logits, target, ignore_index=-100, label_smoothing=0.0)
    got = wrapper(logits, target)

    assert got.shape == (s, b)
    assert_verbose_allclose(got.float(), ref.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("ignore_index", [-100, 0])
def test_wrapper_respects_ignore_index(ignore_index):
    s, b, v = 16, 2, 1024
    wrapper = _make_wrapper(ignore_index=ignore_index)

    logits = torch.randn(s, b, v, device=device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)
    target.view(-1)[: (s * b) // 4] = ignore_index

    ref = _reference_loss(logits, target, ignore_index=ignore_index, label_smoothing=0.0)
    got = wrapper(logits, target)
    assert_verbose_allclose(got.float(), ref.float(), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_wrapper_respects_label_smoothing(label_smoothing):
    s, b, v = 8, 2, 512
    wrapper = _make_wrapper(label_smoothing=label_smoothing)

    logits = torch.randn(s, b, v, device=device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    ref = _reference_loss(logits, target, ignore_index=-100, label_smoothing=label_smoothing)
    got = wrapper(logits, target)
    assert_verbose_allclose(got.float(), ref.float(), atol=1e-5, rtol=1e-4)


def test_wrapper_rejects_unknown_kwargs():
    wrapper = _make_wrapper()
    logits = torch.randn(4, 1, 32, device=device)
    target = torch.randint(0, 32, (4, 1), device=device, dtype=torch.long)
    with pytest.raises(TypeError):
        wrapper(logits, target, unknown_arg=123)


def test_wrapper_rejects_multi_rank_tp_group():
    wrapper = _make_wrapper()
    logits = torch.randn(4, 1, 32, device=device)
    target = torch.randint(0, 32, (4, 1), device=device, dtype=torch.long)

    class _FakeGroup:
        def size(self):
            return 2

    with pytest.raises(RuntimeError, match="tensor_model_parallel_size=1"):
        wrapper(logits, target, tp_group=_FakeGroup())


def test_wrapper_accepts_single_rank_tp_group():
    wrapper = _make_wrapper()
    logits = torch.randn(4, 1, 32, device=device)
    target = torch.randint(0, 32, (4, 1), device=device, dtype=torch.long)

    class _FakeGroup:
        def size(self):
            return 1

    out = wrapper(logits, target, tp_group=_FakeGroup())
    assert out.shape == (4, 1)


def test_wrapper_preserves_gradients():
    s, b, v = 8, 2, 256
    wrapper = _make_wrapper()

    logits = torch.randn(s, b, v, device=device, dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    loss = wrapper(logits, target).sum()
    loss.backward()

    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
