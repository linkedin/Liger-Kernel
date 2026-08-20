"""Fused cross-entropy + total-variation-distance over a vocabulary axis.

Distillation objectives that mix a hard-label term with a distributional
distance term normally pay for two full ``(BT, V)`` float32 softmax temporaries
plus a third buffer for the saved gradient. This op computes both terms in a
single streaming pass over the vocabulary and keeps only ``O(BT)`` state for the
backward pass, recomputing the two distributions from the logits that the caller
already holds.

Both terms are returned *unreduced*, one value per token, so the caller keeps
ownership of masking, per-token weighting, and normalization. That matters when
the denominator is not simply the token count -- for example when it is summed
across a data-parallel group before the division happens.
"""

from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import device_context
from liger_kernel.ops.utils import ensure_contiguous

# Vocabularies are streamed in tiles, so this bounds the tile rather than the
# row. Matching ops/tvd.py keeps register pressure comparable.
MAX_FUSED_SIZE = 65536 // 4


def get_num_warps(BLOCK_SIZE: int) -> int:
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    return num_warps


@triton.jit
def _fused_ce_tvd_forward_kernel(
    student_ptr,
    student_stride,
    teacher_ptr,
    teacher_stride,
    target_ptr,
    ce_ptr,
    tvd_ptr,
    student_lse_ptr,
    teacher_lse_ptr,
    sigma_ptr,
    n_cols,
    ignore_index: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per token row.

    Writes the per-token CE and TVD alongside the three scalars the backward
    pass needs to rebuild both distributions: the two log-sum-exps and
    ``sigma = sum_v p_v * sign(p_v - q_v)``, the softmax-Jacobian correction.
    """
    pid = tl.program_id(0).to(tl.int64)
    student_ptr += pid * student_stride
    teacher_ptr += pid * teacher_stride

    target = tl.load(target_ptr + pid)
    if target == ignore_index:
        tl.store(ce_ptr + pid, 0.0)
        tl.store(tvd_ptr + pid, 0.0)
        # Sentinel log-sum-exps are never consumed: backward returns early on
        # the same predicate.
        tl.store(student_lse_ptr + pid, 0.0)
        tl.store(teacher_lse_ptr + pid, 0.0)
        tl.store(sigma_ptr + pid, 0.0)
        return

    base_offsets = tl.arange(0, BLOCK_SIZE)

    # Pass 1: online max / sum-exp for both rows at once.
    student_max = -float("inf")
    student_sumexp = 0.0
    teacher_max = -float("inf")
    teacher_sumexp = 0.0

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols

        student = tl.load(student_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        teacher = tl.load(teacher_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)

        student_block_max = tl.max(student, axis=0)
        new_student_max = tl.maximum(student_max, student_block_max)
        student_sumexp = student_sumexp * tl.exp(student_max - new_student_max) + tl.sum(
            tl.exp(student - new_student_max), axis=0
        )
        student_max = new_student_max

        teacher_block_max = tl.max(teacher, axis=0)
        new_teacher_max = tl.maximum(teacher_max, teacher_block_max)
        teacher_sumexp = teacher_sumexp * tl.exp(teacher_max - new_teacher_max) + tl.sum(
            tl.exp(teacher - new_teacher_max), axis=0
        )
        teacher_max = new_teacher_max

    student_lse = student_max + tl.log(student_sumexp)
    teacher_lse = teacher_max + tl.log(teacher_sumexp)

    # Pass 2: total variation and the Jacobian correction scalar.
    abs_diff_sum = 0.0
    sigma = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols

        student = tl.load(student_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        teacher = tl.load(teacher_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)

        # Masked lanes carry -inf, so both probabilities vanish and contribute
        # nothing to either accumulator.
        p = tl.exp(student - student_lse)
        q = tl.exp(teacher - teacher_lse)

        diff = p - q
        # sign() with a zero subgradient at ties, matching torch.abs backward so
        # the eager reference and this kernel agree exactly.
        sign = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))

        abs_diff_sum += tl.sum(tl.abs(diff), axis=0)
        sigma += tl.sum(p * sign, axis=0)

    student_at_target = tl.load(student_ptr + target).to(tl.float32)

    tl.store(ce_ptr + pid, student_lse - student_at_target)
    tl.store(tvd_ptr + pid, 0.5 * abs_diff_sum)
    tl.store(student_lse_ptr + pid, student_lse)
    tl.store(teacher_lse_ptr + pid, teacher_lse)
    tl.store(sigma_ptr + pid, sigma)


@triton.jit
def _fused_ce_tvd_backward_kernel(
    student_ptr,
    student_stride,
    teacher_ptr,
    teacher_stride,
    target_ptr,
    grad_ce_ptr,
    grad_tvd_ptr,
    student_lse_ptr,
    teacher_lse_ptr,
    sigma_ptr,
    grad_student_ptr,
    grad_student_stride,
    n_cols,
    ignore_index: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Rebuild both distributions and emit the student-logit gradient.

    With ``p = softmax(student)`` and ``s_v = sign(p_v - q_v)``::

        d(ce)/d(student_v)  = p_v - 1[v == target]
        d(tvd)/d(student_v) = 0.5 * p_v * (s_v - sum_u p_u * s_u)

    The second line is the softmax Jacobian applied to ``0.5 * s``; the summed
    term is the ``sigma`` scalar the forward pass already reduced.
    """
    pid = tl.program_id(0).to(tl.int64)
    student_ptr += pid * student_stride
    teacher_ptr += pid * teacher_stride
    grad_student_ptr += pid * grad_student_stride

    base_offsets = tl.arange(0, BLOCK_SIZE)

    target = tl.load(target_ptr + pid)
    if target == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + base_offsets
            tl.store(grad_student_ptr + offsets, 0.0, mask=offsets < n_cols)
        return

    grad_ce = tl.load(grad_ce_ptr + pid).to(tl.float32)
    grad_tvd = tl.load(grad_tvd_ptr + pid).to(tl.float32)
    student_lse = tl.load(student_lse_ptr + pid)
    teacher_lse = tl.load(teacher_lse_ptr + pid)
    sigma = tl.load(sigma_ptr + pid)

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols

        student = tl.load(student_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        teacher = tl.load(teacher_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)

        p = tl.exp(student - student_lse)
        q = tl.exp(teacher - teacher_lse)

        diff = p - q
        sign = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))

        grad = grad_ce * p + grad_tvd * 0.5 * p * (sign - sigma)
        grad = grad - tl.where(offsets == target, grad_ce, 0.0)

        tl.store(grad_student_ptr + offsets, grad, mask=mask)


def fused_ce_tvd_forward(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
):
    BT, V = student_logits.shape

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    num_warps = get_num_warps(BLOCK_SIZE)

    ce = torch.empty(BT, device=student_logits.device, dtype=torch.float32)
    tvd = torch.empty(BT, device=student_logits.device, dtype=torch.float32)
    student_lse = torch.empty(BT, device=student_logits.device, dtype=torch.float32)
    teacher_lse = torch.empty(BT, device=student_logits.device, dtype=torch.float32)
    sigma = torch.empty(BT, device=student_logits.device, dtype=torch.float32)

    with device_context(student_logits.device):
        _fused_ce_tvd_forward_kernel[(BT,)](
            student_logits,
            student_logits.stride(0),
            teacher_logits,
            teacher_logits.stride(0),
            target,
            ce,
            tvd,
            student_lse,
            teacher_lse,
            sigma,
            V,
            ignore_index=ignore_index,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return ce, tvd, student_lse, teacher_lse, sigma


def fused_ce_tvd_backward(
    grad_ce: torch.Tensor,
    grad_tvd: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    student_lse: torch.Tensor,
    teacher_lse: torch.Tensor,
    sigma: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    BT, V = student_logits.shape

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    num_warps = get_num_warps(BLOCK_SIZE)

    grad_student = torch.empty_like(student_logits)

    with device_context(student_logits.device):
        _fused_ce_tvd_backward_kernel[(BT,)](
            student_logits,
            student_logits.stride(0),
            teacher_logits,
            teacher_logits.stride(0),
            target,
            grad_ce,
            grad_tvd,
            student_lse,
            teacher_lse,
            sigma,
            grad_student,
            grad_student.stride(0),
            V,
            ignore_index=ignore_index,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return grad_student


class LigerFusedCETVDFunction(torch.autograd.Function):
    """Per-token cross-entropy and total variation distance in one pass."""

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            student_logits: ``(BT, V)`` logits of the model being trained.
            teacher_logits: ``(BT, V)`` reference logits. Treated as a constant;
                no gradient flows back through this argument.
            target: ``(BT,)`` hard labels for the cross-entropy term.
            ignore_index: rows whose target equals this value produce zero for
                both terms and a zero gradient.

        Returns:
            ``(ce, tvd)``, both ``(BT,)`` float32 and both unreduced. ``tvd`` is
            ``0.5 * sum_v |p_v - q_v|``, matching :mod:`liger_kernel.ops.tvd`.
        """
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"student_logits and teacher_logits must have the same shape. "
                f"Got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}."
            )
        if student_logits.ndim != 2:
            raise ValueError(f"student_logits must be 2D (BT, V). Got {student_logits.ndim}D.")
        if target.shape != (student_logits.shape[0],):
            raise ValueError(f"target must have shape ({student_logits.shape[0]},). Got {tuple(target.shape)}.")

        ce, tvd, student_lse, teacher_lse, sigma = fused_ce_tvd_forward(
            student_logits, teacher_logits, target, ignore_index
        )

        # Only O(BT) state is retained; the distributions are rebuilt in the
        # backward pass from logits the caller already keeps alive.
        ctx.save_for_backward(student_logits, teacher_logits, target, student_lse, teacher_lse, sigma)
        ctx.ignore_index = ignore_index

        return ce, tvd

    @staticmethod
    @ensure_contiguous
    def backward(
        ctx,
        grad_ce: torch.Tensor,
        grad_tvd: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], None, None, None]:
        student_logits, teacher_logits, target, student_lse, teacher_lse, sigma = ctx.saved_tensors

        grad_student = fused_ce_tvd_backward(
            grad_ce,
            grad_tvd,
            student_logits,
            teacher_logits,
            target,
            student_lse,
            teacher_lse,
            sigma,
            ctx.ignore_index,
        )

        return grad_student, None, None, None
