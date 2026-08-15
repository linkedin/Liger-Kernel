import pytest
import torch
import torch.nn.functional as F

from test.utils import supports_bfloat16

from liger_kernel.ops.fused_ce_tvd import LigerFusedCETVDFunction
from liger_kernel.utils import infer_device

device = infer_device()


def torch_ce_tvd(student_logits, teacher_logits, target, ignore_index=-100):
    """Eager reference: per-token CE and TVD, both computed in float32."""
    student = student_logits.float()
    teacher = teacher_logits.float()

    p = torch.softmax(student, dim=-1)
    q = torch.softmax(teacher, dim=-1)

    ce = F.cross_entropy(student, target, reduction="none", ignore_index=ignore_index)
    tvd = 0.5 * (p - q).abs().sum(dim=-1)

    ignored = target == ignore_index
    ce = torch.where(ignored, torch.zeros_like(ce), ce)
    tvd = torch.where(ignored, torch.zeros_like(tvd), tvd)
    return ce, tvd


_SHAPE_PARAMS = (
    "B, T, V",
    [
        (2, 8, 128),
        (1, 16, 4096),
        (2, 4, 32000),
        # Vocabulary wider than one tile, exercising the streaming loops.
        (1, 8, 131072),
        # Non power-of-two vocab, exercising the tail mask.
        (2, 4, 30001),
    ],
)

_DTYPE_PARAMS = (
    "dtype, atol, rtol",
    [
        pytest.param(
            torch.bfloat16,
            1e-2,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this device"),
        ),
        (torch.float32, 1e-5, 1e-5),
    ],
)


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize(*_DTYPE_PARAMS)
def test_forward_matches_eager(B, T, V, dtype, atol, rtol):
    torch.manual_seed(0)
    BT = B * T

    student = torch.randn(BT, V, device=device, dtype=dtype)
    teacher = torch.randn(BT, V, device=device, dtype=dtype)
    target = torch.randint(0, V, (BT,), device=device, dtype=torch.long)

    ce_ref, tvd_ref = torch_ce_tvd(student, teacher, target)
    ce, tvd = LigerFusedCETVDFunction.apply(student, teacher, target)

    assert torch.allclose(ce, ce_ref, atol=atol, rtol=rtol)
    assert torch.allclose(tvd, tvd_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize(*_DTYPE_PARAMS)
def test_backward_matches_eager(B, T, V, dtype, atol, rtol):
    torch.manual_seed(0)
    BT = B * T

    student = torch.randn(BT, V, device=device, dtype=dtype)
    teacher = torch.randn(BT, V, device=device, dtype=dtype)
    target = torch.randint(0, V, (BT,), device=device, dtype=torch.long)

    # Asymmetric upstream gradients so the two terms cannot mask each other.
    grad_ce = torch.randn(BT, device=device, dtype=torch.float32)
    grad_tvd = torch.randn(BT, device=device, dtype=torch.float32)

    student_ref = student.clone().requires_grad_(True)
    ce_ref, tvd_ref = torch_ce_tvd(student_ref, teacher, target)
    (ce_ref * grad_ce + tvd_ref * grad_tvd).sum().backward()

    student_liger = student.clone().requires_grad_(True)
    ce, tvd = LigerFusedCETVDFunction.apply(student_liger, teacher, target)
    (ce * grad_ce + tvd * grad_tvd).sum().backward()

    assert torch.allclose(student_liger.grad, student_ref.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(*_DTYPE_PARAMS)
def test_ignore_index(dtype, atol, rtol):
    torch.manual_seed(0)
    BT, V = 8, 4096
    ignore_index = -100

    student = torch.randn(BT, V, device=device, dtype=dtype)
    teacher = torch.randn(BT, V, device=device, dtype=dtype)
    target = torch.randint(0, V, (BT,), device=device, dtype=torch.long)
    target[::2] = ignore_index

    ce_ref, tvd_ref = torch_ce_tvd(student, teacher, target, ignore_index)

    student_liger = student.clone().requires_grad_(True)
    ce, tvd = LigerFusedCETVDFunction.apply(student_liger, teacher, target, ignore_index)

    assert torch.allclose(ce, ce_ref, atol=atol, rtol=rtol)
    assert torch.allclose(tvd, tvd_ref, atol=atol, rtol=rtol)

    (ce.sum() + tvd.sum()).backward()
    # Ignored rows must contribute no gradient at all.
    assert torch.equal(student_liger.grad[::2], torch.zeros_like(student_liger.grad[::2]))
    assert not torch.equal(student_liger.grad[1::2], torch.zeros_like(student_liger.grad[1::2]))


def test_teacher_receives_no_gradient():
    torch.manual_seed(0)
    BT, V = 4, 512

    student = torch.randn(BT, V, device=device, dtype=torch.float32, requires_grad=True)
    teacher = torch.randn(BT, V, device=device, dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, V, (BT,), device=device, dtype=torch.long)

    ce, tvd = LigerFusedCETVDFunction.apply(student, teacher, target)
    (ce.sum() + tvd.sum()).backward()

    assert student.grad is not None
    assert teacher.grad is None


def test_identical_distributions_give_zero_tvd():
    """Self-distillation: every slot ties, so both the distance and its gradient vanish."""
    torch.manual_seed(0)
    BT, V = 4, 1024

    logits = torch.randn(BT, V, device=device, dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, V, (BT,), device=device, dtype=torch.long)

    _, tvd = LigerFusedCETVDFunction.apply(logits, logits.detach().clone(), target)

    assert torch.allclose(tvd, torch.zeros_like(tvd), atol=1e-6, rtol=0)

    tvd.sum().backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_gradient_at_probability_ties():
    """Exact ``p == q`` ties need the three-way sign, not a ``p > q`` two-way split.

    A two-way split folds ties into the ``p < q`` branch, which shifts every
    tied slot's contribution to ``sigma`` and corrupts the whole row. Ties are
    reachable in practice: self-distillation ties everything, and a teacher that
    permutes the student's logits leaves the log-sum-exp untouched so every
    unpermuted slot ties exactly while still carrying probability mass.
    """
    torch.manual_seed(0)
    BT, V = 4, 256

    student = torch.randn(BT, V, device=device, dtype=torch.float32)
    teacher = student.clone()
    teacher[:, [0, 1]] = teacher[:, [1, 0]]
    target = torch.randint(0, V, (BT,), device=device, dtype=torch.long)

    assert int((torch.softmax(student, -1) == torch.softmax(teacher, -1)).sum()) > 0, "no ties constructed"

    student_ref = student.clone().requires_grad_(True)
    _, tvd_ref = torch_ce_tvd(student_ref, teacher, target)
    tvd_ref.sum().backward()

    student_liger = student.clone().requires_grad_(True)
    _, tvd = LigerFusedCETVDFunction.apply(student_liger, teacher, target)
    tvd.sum().backward()

    torch.testing.assert_close(student_liger.grad, student_ref.grad, atol=1e-6, rtol=1e-5)


def test_shape_validation():
    BT, V = 4, 128
    student = torch.randn(BT, V, device=device)
    target = torch.randint(0, V, (BT,), device=device, dtype=torch.long)

    with pytest.raises(ValueError, match="same shape"):
        LigerFusedCETVDFunction.apply(student, torch.randn(BT, V * 2, device=device), target)

    with pytest.raises(ValueError, match="must be 2D"):
        LigerFusedCETVDFunction.apply(student.view(1, BT, V), torch.randn(1, BT, V, device=device), target)

    with pytest.raises(ValueError, match="target must have shape"):
        LigerFusedCETVDFunction.apply(student, torch.randn(BT, V, device=device), target[:-1])
