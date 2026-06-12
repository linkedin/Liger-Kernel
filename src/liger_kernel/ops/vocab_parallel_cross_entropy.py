"""Vocab-parallel cross-entropy Triton kernels for Megatron-style TP>=1.

Two-kernel design — required because the standard CE kernel cannot straddle the
Python-level AllReduce that vocab-parallel CE needs:

    Kernel 1 (forward stats): shifted_logits -> sum_exp_local, predicted_logit_local
                              also writes exp(shifted_logits) in-place
    Python:                   AllReduce(SUM) x2 (no-op at TP=1)
                              loss = log(sum_exp_global) - predicted_global
                              optional label-smoothing adjustment
    Kernel 2 (backward grad): exp_logits / sum_exp_global -> softmax, write grad in-place

At TP=1 the same kernels run with no AllReduces (single-rank group is skipped at the
Python wrapper level), so this is a single code path for any TP size.

Label smoothing — both PyTorch's alpha/K and Megatron's NeMo (alpha*V/(V-1))
formulas share the structural form

    loss = (1 - alpha_eff) * H(q, p) + alpha_eff * H(u, p)

so the kernel takes only ``alpha_eff`` and ``eps_eff = alpha_eff / K`` as inputs.
Choosing the formula and the averaging scope (V_local for "partition" mode,
V_global for "global" mode) happens at the Python wrapper.
"""

from __future__ import annotations

import triton
import triton.language as tl

from liger_kernel.ops.utils import is_hip
from liger_kernel.utils import infer_device

# Same heuristic as the standard CE kernel — 32K BLOCK_SIZE keeps register
# spills low on Hopper / Ampere; XPU / NPU use smaller blocks.
if infer_device() == "xpu":
    MAX_FUSED_SIZE = 4096
elif infer_device() == "npu":
    MAX_FUSED_SIZE = 2048
else:
    MAX_FUSED_SIZE = 65536 // 2


@triton.jit
def liger_vocab_parallel_ce_forward_kernel(
    X_ptr,  # [BT, V_local] shifted logits (in-place: receives exp(shifted))
    X_stride,
    Y_ptr,  # [BT] masked_target — local indices into V_local (0 if off-rank)
    target_mask_ptr,  # [BT] int8: 1 if target NOT on this rank
    ignore_mask_ptr,  # [BT] int8: 1 if target == ignore_index
    pred_ptr,  # [BT] OUT: predicted_logit_local (0 if off-rank / ignored)
    sum_exp_ptr,  # [BT] OUT: sum_exp_local
    n_cols,  # V_local
    BLOCK_SIZE: tl.constexpr,
):
    """Per-token forward stats. One program per row.

    Computes ``predicted_logit_local`` (X[masked_target] if target is owned by this
    rank and not ignored, else 0) and ``sum_exp_local = sum(exp(shifted_logits))``,
    and writes ``exp(shifted_logits)`` back to ``X_ptr`` in-place so the backward
    kernel can reuse the buffer without recomputing exp.

    Shifted logits are <= 0 so exp() is in (0, 1] — no overflow possible.
    """
    program_id = tl.program_id(0).to(tl.int64)
    X_ptr += program_id * X_stride

    y = tl.load(Y_ptr + program_id)
    target_off_rank = tl.load(target_mask_ptr + program_id)
    is_ignored = tl.load(ignore_mask_ptr + program_id)

    # Predicted logit: only meaningful on the rank that owns the target, and only
    # when the position is not an ignored label. Off-rank/ignored positions store
    # 0 here; the AllReduce(SUM) gathers the one nonzero contribution.
    if (is_ignored != 0) or (target_off_rank != 0):
        pred_local = 0.0
    else:
        pred_local = tl.load(X_ptr + y).cast(tl.float32)

    # Single pass over V_local: accumulate sum_exp and write exp in-place.
    sum_exp = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        col_mask = X_offsets < n_cols
        X_block = tl.load(X_ptr + X_offsets, mask=col_mask, other=0.0).cast(tl.float32)
        exp_block = tl.exp(X_block)
        exp_block = tl.where(col_mask, exp_block, 0.0)
        sum_exp += tl.sum(exp_block)
        tl.store(X_ptr + X_offsets, exp_block, mask=col_mask)

    tl.store(pred_ptr + program_id, pred_local)
    tl.store(sum_exp_ptr + program_id, sum_exp)


@triton.jit
def liger_vocab_parallel_ce_backward_kernel(
    EXP_ptr,  # [BT, V_local] exp(shifted) from kernel 1; receives grad in-place
    EXP_stride,
    sum_exp_ptr,  # [BT] sum_exp_global (post-AllReduce on TP>1, kernel 1 output on TP=1)
    Y_ptr,  # [BT] masked_target
    target_mask_ptr,  # [BT] int8
    ignore_mask_ptr,  # [BT] int8
    grad_out_ptr,  # [BT] upstream gradient
    n_cols,  # V_local
    alpha_eff,  # scalar (already rescaled for chosen formula)
    eps_eff,  # scalar = alpha_eff / K (K = V_local or V_global per mode)
    HAS_LABEL_SMOOTHING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-token backward grad. One program per row.

    Reads exp(shifted) from EXP_ptr, normalizes by ``sum_exp_global`` to get the
    local slice of the global softmax, subtracts the one-hot target on the owning
    rank, applies optional label-smoothing offset, scales by ``grad_output``, and
    writes the result back to EXP_ptr (now the gradient buffer).
    """
    program_id = tl.program_id(0).to(tl.int64)
    EXP_ptr += program_id * EXP_stride

    is_ignored = tl.load(ignore_mask_ptr + program_id)
    if is_ignored != 0:
        # Ignored target: zero the whole row. Loss was 0 in forward so gradient
        # w.r.t. logits must also be 0 regardless of upstream grad_output.
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(EXP_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    y = tl.load(Y_ptr + program_id)
    target_off_rank = tl.load(target_mask_ptr + program_id)
    sum_exp_global = tl.load(sum_exp_ptr + program_id)
    grad_out = tl.load(grad_out_ptr + program_id).cast(tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        col_mask = X_offsets < n_cols
        exp_block = tl.load(EXP_ptr + X_offsets, mask=col_mask, other=0.0).cast(tl.float32)
        grad_block = exp_block / sum_exp_global  # = softmax over global vocab (local slice)

        if HAS_LABEL_SMOOTHING:
            grad_block = grad_block - eps_eff

        # Target adjustment only on the rank that owns the target.
        if target_off_rank == 0:
            is_target = X_offsets == y
            grad_block = tl.where(is_target, grad_block - (1.0 - alpha_eff), grad_block)

        grad_block = grad_block * grad_out
        tl.store(EXP_ptr + X_offsets, grad_block, mask=col_mask)


def get_num_warps(block_size: int) -> int:
    """Match the standard CE kernel's warp heuristic."""
    return 32 if not is_hip() else 16


def select_block_size(n_cols: int) -> int:
    return min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
