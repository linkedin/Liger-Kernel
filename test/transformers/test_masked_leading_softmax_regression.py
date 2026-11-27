"""Regression tests for online-softmax kernels with an all-`-inf` leading tile.

These kernels rescale the running sum by ``exp(old_max - new_max)``. When the
first streamed tile of logits is entirely ``-inf``, both maxima are ``-inf`` and
that term becomes ``-inf - -inf = NaN``, contaminating the result. Each test
masks exactly the first tile (tile sizes differ per kernel) and compares against
the PyTorch reference.
"""

import pytest
import torch
import torch.nn.functional as F

from liger_kernel.ops.fused_ce_tvd import LigerFusedCETVDFunction
from liger_kernel.ops.grpo_loss import fused_selective_log_softmax
from liger_kernel.transformers.functional import liger_cross_entropy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to exercise the Triton kernels")

# Each kernel caps its streamed tile differently (MAX_FUSED_SIZE), so the vocab
# must exceed the cap to get a second tile. Mask the first cap entries.
CE_TILE = 32768  # cross_entropy.py: 65536 // 2 on CUDA
TVD_TILE = 16384  # fused_ce_tvd.py: 65536 // 4


def _masked_logits(batch, rows, vocab, masked_cols):
    logits = torch.randn(batch, rows, vocab, device="cuda", dtype=torch.float32)
    logits[..., :masked_cols] = float("-inf")
    return logits


def test_cross_entropy_with_masked_leading_tile():
    torch.manual_seed(0)
    B, T, V = 2, 3, CE_TILE * 2
    # liger_cross_entropy expects a 2-D (BT, V) input, not (B, T, V).
    logits = torch.randn(B * T, V, device="cuda", dtype=torch.float32)
    logits[:, :CE_TILE] = float("-inf")
    target = torch.randint(CE_TILE, V, (B * T,), device="cuda")

    actual = liger_cross_entropy(logits, target, reduction="none")
    expected = F.cross_entropy(logits, target, reduction="none")

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_fused_ce_tvd_with_masked_leading_tile():
    torch.manual_seed(0)
    BT, V = 4, TVD_TILE * 2

    # Case 1: the student's first tile is masked.
    student = _masked_logits(1, BT, V, TVD_TILE).squeeze(0)
    teacher = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(TVD_TILE, V, (BT,), device="cuda")

    ce, tvd = LigerFusedCETVDFunction.apply(student, teacher, target)
    p = torch.softmax(student.float(), dim=-1)
    q = torch.softmax(teacher.float(), dim=-1)
    ce_ref = F.cross_entropy(student.float(), target, reduction="none")
    tvd_ref = 0.5 * (p - q).abs().sum(dim=-1)

    assert torch.isfinite(ce).all() and torch.isfinite(tvd).all()
    torch.testing.assert_close(ce, ce_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(tvd, tvd_ref, atol=1e-3, rtol=1e-3)

    # Case 2: the teacher's first tile is masked instead.
    student = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    teacher = _masked_logits(1, BT, V, TVD_TILE).squeeze(0)

    ce, tvd = LigerFusedCETVDFunction.apply(student, teacher, target)
    p = torch.softmax(student.float(), dim=-1)
    q = torch.softmax(teacher.float(), dim=-1)
    ce_ref = F.cross_entropy(student.float(), target, reduction="none")
    tvd_ref = 0.5 * (p - q).abs().sum(dim=-1)

    assert torch.isfinite(ce).all() and torch.isfinite(tvd).all()
    torch.testing.assert_close(ce, ce_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(tvd, tvd_ref, atol=1e-3, rtol=1e-3)


def test_selective_log_softmax_with_masked_leading_tile():
    torch.manual_seed(0)
    B, L, N = 2, 3, 4096
    logits = _masked_logits(B, L + 1, N, 2048)  # fused_selective_log_softmax uses BLOCK_N=2048
    input_ids = torch.randint(2048, N, (B, L), device="cuda")

    actual = fused_selective_log_softmax(logits, input_ids, temperature=0.9)
    expected = torch.log_softmax(logits[:, :-1] / 0.9, dim=-1).gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
