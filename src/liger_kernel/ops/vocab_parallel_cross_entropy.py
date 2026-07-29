"""Vocab-parallel cross-entropy Triton kernels for Megatron-style TP>=1.

Two-kernel design — required because the standard CE kernel cannot straddle the
Python-level AllReduce that vocab-parallel CE needs:

    Kernel 1 (forward stats): raw logits + global max -> exp(shifted) buffer
                              + sum_exp_local + predicted_logit_local
    Python:                   AllReduce(SUM) x2 (no-op at TP=1)
                              loss = log(sum_exp_global) - predicted_global
                              optional label-smoothing adjustment
    Kernel 2 (backward grad): exp(shifted) / sum_exp_global -> softmax, write grad in-place

At TP=1 the same kernels run with no AllReduces (single-rank group is skipped at the
Python wrapper level), so this is a single code path for any TP size.

The kernels do everything that depends only on per-token state — shift by global max,
target on/off-rank determination, ignore_index handling — internally, so the Python
wrapper does the minimum required for AllReduce orchestration. This is what makes the
small-V case competitive: at V=4K the kernel itself is ~50 µs, so each PyTorch op we
move into Triton saves ~10-30 µs of Python dispatch.

Label smoothing — both PyTorch's alpha/K and Megatron's NeMo (alpha*V/(V-1))
formulas share the structural form

    loss = (1 - alpha_eff) * H(q, p) + alpha_eff * H(u, p)

so the kernel takes only ``alpha_eff`` and ``eps_eff = alpha_eff / K`` as inputs.
Choosing the formula and the averaging scope (V_local for "partition" mode,
V_global for "global" mode) happens at the Python wrapper.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from liger_kernel.ops.utils import is_hip
from liger_kernel.utils import infer_device

_FORMULAS = ("pytorch", "megatron")
_MODES = ("global", "partition")

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
    X_ptr,  # [BT, V_local] raw logits (any dtype) — READ-ONLY, never mutated
    X_stride,
    EXP_ptr,  # [BT, V_local] OUT: exp(logits - logits_max), fp32 buffer
    EXP_stride,
    logits_max_ptr,  # [BT] fp32 per-token global max (post-AllReduce(MAX))
    Y_ptr,  # [BT] int64 GLOBAL target indices in [0, V_global)
    pred_ptr,  # [BT] OUT: predicted_logit_local (= X[masked_target] - max,
    # 0 if off-rank / ignored)
    sum_exp_ptr,  # [BT] OUT: sum_exp_local
    vocab_start,  # int — first global vocab index owned by this rank
    n_cols,  # V_local
    ignore_index,  # int — target value that means "skip this position"
    BLOCK_SIZE: tl.constexpr,
):
    """Per-token forward stats. One program per row.

    All per-token bookkeeping (off-rank check, ignore-index check, shift by global
    max) happens here so the Python wrapper doesn't have to allocate a target_mask
    tensor or a shifted-logits buffer.
    """
    program_id = tl.program_id(0).to(tl.int64)
    X_ptr += program_id * X_stride
    EXP_ptr += program_id * EXP_stride

    y_global = tl.load(Y_ptr + program_id)
    m = tl.load(logits_max_ptr + program_id).cast(tl.float32)

    is_ignored = y_global == ignore_index
    target_off_rank = (y_global < vocab_start) | (y_global >= vocab_start + n_cols)
    y_local = y_global - vocab_start  # only used when on-rank

    # Predicted logit — shifted by the global max. Off-rank / ignored positions
    # store 0; the AllReduce(SUM) gathers the one nonzero contribution.
    if is_ignored or target_off_rank:
        pred_local = 0.0
    else:
        pred_local = tl.load(X_ptr + y_local).cast(tl.float32) - m

    # Single pass over V_local: shift inside, exp, accumulate sum_exp,
    # store exp(shifted) to the fp32 output buffer. Shifted logits are <= 0 so
    # exp() is in (0, 1] — no overflow possible.
    sum_exp = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        col_mask = X_offsets < n_cols
        X_block = tl.load(X_ptr + X_offsets, mask=col_mask, other=0.0).cast(tl.float32)
        exp_block = tl.exp(X_block - m)
        exp_block = tl.where(col_mask, exp_block, 0.0)
        sum_exp += tl.sum(exp_block)
        tl.store(EXP_ptr + X_offsets, exp_block, mask=col_mask)

    tl.store(pred_ptr + program_id, pred_local)
    tl.store(sum_exp_ptr + program_id, sum_exp)


@triton.jit
def liger_vocab_parallel_ce_backward_kernel(
    EXP_ptr,  # [BT, V_local] exp(shifted) from kernel 1; receives grad in-place
    EXP_stride,
    sum_exp_ptr,  # [BT] fp32 sum_exp_global (post-AllReduce on TP>1)
    Y_ptr,  # [BT] int64 GLOBAL target indices
    grad_out_ptr,  # [BT] upstream gradient
    vocab_start,  # int — first global vocab index owned by this rank
    n_cols,  # V_local
    ignore_index,  # int
    alpha_eff,  # scalar (already rescaled for chosen formula)
    eps_eff,  # scalar = alpha_eff / K (K = V_local or V_global per mode)
    HAS_LABEL_SMOOTHING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-token backward grad. One program per row.

    Recomputes off-rank / ignore-index from the raw global target — the wrapper
    doesn't pass intermediate mask tensors, so the kernel does it once here from
    the cheaply-loaded scalar.
    """
    program_id = tl.program_id(0).to(tl.int64)
    EXP_ptr += program_id * EXP_stride

    y_global = tl.load(Y_ptr + program_id)
    if y_global == ignore_index:
        # Ignored: gradient w.r.t. logits is 0 regardless of upstream grad_output
        # (because loss was 0 in forward).
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(EXP_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    target_off_rank = (y_global < vocab_start) | (y_global >= vocab_start + n_cols)
    y_local = y_global - vocab_start
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
        if not target_off_rank:
            is_target = X_offsets == y_local
            grad_block = tl.where(is_target, grad_block - (1.0 - alpha_eff), grad_block)

        grad_block = grad_block * grad_out
        tl.store(EXP_ptr + X_offsets, grad_block, mask=col_mask)


def _get_num_warps(block_size: int) -> int:
    """Match the standard CE kernel's warp heuristic."""
    return 32 if not is_hip() else 16


def _select_block_size(n_cols: int) -> int:
    return min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))


def _is_tp_distributed(tp_group) -> bool:
    """True iff ``tp_group`` is a real multi-rank ProcessGroup."""
    return tp_group is not None and hasattr(tp_group, "size") and tp_group.size() > 1


class LigerVocabParallelCEFunction(torch.autograd.Function):
    """Autograd op for vocab-parallel cross-entropy. Backs both
    :class:`liger_kernel.transformers.LigerVocabParallelCrossEntropy` and
    :class:`liger_kernel.megatron.LigerMegatronCrossEntropy`.

    Forward layout:
      1. Per-token global max (cheap reduction on input dtype, cast result to fp32)
      2. AllReduce(MAX) (no-op at TP=1)
      3. Kernel 1: shift internally, write exp(shifted) to a fp32 output buffer,
         compute sum_exp_local and predicted_logit_local in one pass
      4. AllReduce(SUM) × 2
      5. Loss = log(sum_exp_global) - predicted_global, then optional label smoothing
      6. Save the exp buffer for backward

    Backward layout:
      1. Kernel 2 reads exp(shifted), normalizes by sum_exp_global, subtracts the
         one-hot target on the owning rank, scales by grad_output, writes grad in-place
         to the buffer it consumed. No AllReduce needed.

    All AllReduces are skipped when ``tp_group`` is None or has size 1, so the same
    code path serves TP=1 and TP>1.
    """

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        tp_group,
        ignore_index: int,
        label_smoothing: float,
        label_smoothing_formula: str,
        label_smoothing_mode: str,
    ) -> torch.Tensor:
        # Module wrappers already check this, but the Function is reachable directly
        # via ``liger_kernel.ops`` so guard here too to avoid an unhelpful
        # "not enough values to unpack" error from the shape destructuring below.
        if vocab_parallel_logits.dim() != 3:
            raise ValueError(
                f"vocab_parallel_logits must be 3-D ([seq, batch, vocab]); "
                f"got shape {tuple(vocab_parallel_logits.shape)}."
            )
        s, b, v_local = vocab_parallel_logits.shape
        BT = s * b

        tp_distributed = _is_tp_distributed(tp_group)
        if tp_distributed:
            rank = dist.get_rank(tp_group)
            world = dist.get_world_size(tp_group)
        else:
            rank, world = 0, 1
        vocab_start = rank * v_local

        # --- 1. Per-token global max (cheap; small output) ---
        # amax on the input dtype, cast the [S, B] result to fp32 for the kernel.
        # No full-buffer fp32 conversion.
        x = vocab_parallel_logits.contiguous()
        logits_max = x.amax(dim=-1).float()
        if tp_distributed:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        # --- 2. Flatten target once ---
        flat_target = target.reshape(-1).contiguous().to(torch.int64)

        # --- 3. Kernel 1: shift, exp, sum_exp, predicted_local ---
        # exp_buf is fp32 for precision — matches Megatron's vocab-parallel CE,
        # which also keeps the saved softmax in fp32. Moving to bf16 (or input
        # dtype) would save BT*V_local*2 bytes per call but introduces ~1 %
        # relative error on low-probability softmax tails; gated on a separate
        # convergence study.
        exp_buf = torch.empty(BT, v_local, device=x.device, dtype=torch.float32)
        predicted_local = torch.empty(BT, device=x.device, dtype=torch.float32)
        sum_exp_local = torch.empty(BT, device=x.device, dtype=torch.float32)

        block_size = _select_block_size(v_local)
        num_warps = _get_num_warps(block_size)

        x_2d = x.view(BT, v_local)
        liger_vocab_parallel_ce_forward_kernel[(BT,)](
            X_ptr=x_2d,
            X_stride=x_2d.stride(0),
            EXP_ptr=exp_buf,
            EXP_stride=exp_buf.stride(0),
            logits_max_ptr=logits_max,
            Y_ptr=flat_target,
            pred_ptr=predicted_local,
            sum_exp_ptr=sum_exp_local,
            vocab_start=vocab_start,
            n_cols=v_local,
            ignore_index=ignore_index,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )

        # --- 4. AllReduces (no-ops at TP=1) ---
        if tp_distributed:
            dist.all_reduce(predicted_local, op=dist.ReduceOp.SUM, group=tp_group)
            dist.all_reduce(sum_exp_local, op=dist.ReduceOp.SUM, group=tp_group)

        # --- 5. Loss ---
        loss = torch.log(sum_exp_local) - predicted_local

        # --- 6. Label smoothing (Python; chooses K, formula, averaging scope) ---
        if label_smoothing > 0:
            v_for_K = v_local * world if label_smoothing_mode == "global" else v_local
            if label_smoothing_formula == "megatron":
                alpha_eff = label_smoothing * v_for_K / (v_for_K - 1)
            else:  # "pytorch"
                alpha_eff = label_smoothing
            eps_eff = alpha_eff / v_for_K

            # log_softmax_i = log(exp_i / sum_exp_global) = log(exp_i) - log(sum_exp_global)
            log_probs_local = torch.log(exp_buf) - torch.log(sum_exp_local).unsqueeze(-1)
            if label_smoothing_mode == "global":
                sum_log_probs = log_probs_local.sum(dim=-1)
                if tp_distributed:
                    dist.all_reduce(sum_log_probs, op=dist.ReduceOp.SUM, group=tp_group)
                mean_log_probs = sum_log_probs / v_for_K
            else:  # "partition"
                mean_log_probs = log_probs_local.mean(dim=-1)

            loss = (1.0 - alpha_eff) * loss + alpha_eff * (-mean_log_probs)
        else:
            alpha_eff = 0.0
            eps_eff = 0.0

        # Zero out ignored positions (one Python op, can't easily move into the kernel
        # since the kernel writes element-wise to exp_buf and we need a scalar loss).
        if ignore_index is not None:
            ignore_mask = flat_target == ignore_index
            loss = torch.where(ignore_mask, torch.zeros_like(loss), loss)

        # --- 7. Save for backward ---
        ctx.save_for_backward(exp_buf, sum_exp_local, flat_target)
        ctx.alpha_eff = float(alpha_eff)
        ctx.eps_eff = float(eps_eff)
        ctx.has_smoothing = bool(label_smoothing > 0)
        ctx.vocab_start = vocab_start
        ctx.v_local = v_local
        ctx.ignore_index = ignore_index
        ctx.orig_dtype = vocab_parallel_logits.dtype
        ctx.orig_shape = (s, b)
        ctx.block_size = block_size

        return loss.reshape(s, b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        exp_buf, sum_exp_global, flat_target = ctx.saved_tensors

        grad_out_flat = grad_output.contiguous().reshape(-1).to(torch.float32)
        BT = exp_buf.size(0)
        num_warps = _get_num_warps(ctx.block_size)

        liger_vocab_parallel_ce_backward_kernel[(BT,)](
            EXP_ptr=exp_buf,
            EXP_stride=exp_buf.stride(0),
            sum_exp_ptr=sum_exp_global,
            Y_ptr=flat_target,
            grad_out_ptr=grad_out_flat,
            vocab_start=ctx.vocab_start,
            n_cols=ctx.v_local,
            ignore_index=ctx.ignore_index,
            alpha_eff=ctx.alpha_eff,
            eps_eff=ctx.eps_eff,
            HAS_LABEL_SMOOTHING=ctx.has_smoothing,
            BLOCK_SIZE=ctx.block_size,
            num_warps=num_warps,
        )

        s, b = ctx.orig_shape
        grad_input = exp_buf.to(ctx.orig_dtype).view(s, b, ctx.v_local)
        # 7 inputs to forward → 7 returns from backward (only the logits tensor has grad)
        return (grad_input, None, None, None, None, None, None)


def validate_formula_and_mode(label_smoothing_formula: str, label_smoothing_mode: str) -> None:
    """Shared validation used by the Module wrappers in transformers/ and megatron/."""
    if label_smoothing_formula not in _FORMULAS:
        raise ValueError(f"label_smoothing_formula must be one of {_FORMULAS}, got {label_smoothing_formula!r}.")
    if label_smoothing_mode not in _MODES:
        raise ValueError(f"label_smoothing_mode must be one of {_MODES}, got {label_smoothing_mode!r}.")
