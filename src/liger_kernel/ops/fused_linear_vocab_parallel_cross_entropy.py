"""Fused linear + vocab-parallel cross-entropy for Megatron-style TP>=1.

Merges the chunking-over-BT trick from :mod:`liger_kernel.ops.fused_linear_cross_entropy`
with the AllReduce orchestration from :mod:`liger_kernel.ops.vocab_parallel_cross_entropy`.
The result avoids ever materializing the per-rank ``[BT, V_local]`` logits tensor — the
dominant activation in long-context LLM training.

Forward (two-pass over BT chunks):

  Pass 1: for each chunk, ``logits_chunk = x_chunk @ W_local.T`` (+ bias),
          ``max_local[chunk] = logits_chunk.amax(-1)``. Logits tile dies.
  AllReduce(MAX) on ``[BT]``.
  Pass 2: for each chunk, recompute ``logits_chunk``, run Triton stats kernel
          to emit ``(sum_exp_local, pred_local, sum_x_minus_m_local)`` per row.
  AllReduce(SUM) on ``[BT, 2]`` (or ``[BT, 3]`` if mode='global' label smoothing).
  ``loss = log(sum_exp_global) - pred_global``; apply optional label smoothing.

Backward (recompute, no per-chunk softmax saved):

  For each chunk, recompute ``logits_chunk``, run Triton grad kernel to produce
  ``G [chunk, V_local] fp32`` (softmax − one_hot(target_on_rank) − eps_smoothing,
  scaled by ``grad_out``), then ``grad_x[chunk] += G @ W_local`` and
  ``grad_W += G.T @ x_chunk``. ``grad_b += G.sum(0)`` if bias is given.
  AllReduce(SUM) on ``grad_x`` (no-op at TP=1) is handled by the wrapping
  SP-gather op; for non-SP this is a single AllReduce.

At TP=1 the same kernels run with no AllReduces (single-rank group is skipped at
the Python wrapper level) — single code path serves any TP size.

Memory math (per rank, Llama3-70B TP=8 V_local=16K H=8K, BT=128K, chunk=2K):

  Baseline VP-CE saves [BT, V_local] bf16 + [BT, V_local] fp32 exp_buf = 6·BT·V_local
    = 12.0 GiB
  VP-FLCE fwd transient (per chunk): [chunk, V_local] bf16 = 64 MiB
  VP-FLCE bwd transient (per chunk): [chunk, V_local] bf16 + [chunk, V_local] fp32 G
    + [V_local, H] dW (bf16 or fp32) + [chunk, H] grad_x slice = ~200 MiB - 800 MiB
    depending on accum_dtype

  Net saving at long context: ~11 GiB per rank. See the design document at
  VP_FLCE_DESIGN_CONVERSATION.md for the full table.

Throughput cost: +2 matmul launches per step over the unfused baseline
(baseline = 3 matmuls; VP-FLCE = 5 matmuls). At Llama3-70B BT=8K this is ~+4.4 ms
on H100 (~0.05% of step); at BT=128K it's ~+70 ms (~0.5-1.4% of step). Acceptable
given the activation savings unlock larger micro-batch.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from liger_kernel.ops.utils import is_hip
from liger_kernel.ops.vocab_parallel_cross_entropy import _is_tp_distributed
from liger_kernel.ops.vocab_parallel_cross_entropy import validate_formula_and_mode
from liger_kernel.utils import infer_device

# Same BLOCK_SIZE heuristic as VP-CE and FLCE.
if infer_device() == "xpu":
    MAX_FUSED_SIZE = 4096
elif infer_device() == "npu":
    MAX_FUSED_SIZE = 2048
else:
    MAX_FUSED_SIZE = 65536 // 2

# Cap on per-chunk transient memory. The FLCE chunk-size heuristic targets
# a per-chunk logits tile size ~= activation tile size, which for VP-FLCE leaves
# huge transient tiles (e.g. 384 MiB per chunk at Llama3-70B shapes). This cap
# trades a few extra chunks for bounded transient memory.
_MEM_BUDGET_BYTES = 128 * 1024 * 1024  # 128 MiB
_BYTES_PER_ELEM_TRANSIENT = 6  # 2 bf16 logits + 4 fp32 grad-logits


@triton.jit
def liger_fused_linear_vpce_fwd_stats_kernel(
    LOGITS_ptr,  # [chunk, V_local] (input dtype, typically bf16) — READ-ONLY
    LOGITS_stride,
    max_ptr,  # [chunk] fp32 — global max post-AllReduce(MAX)
    Y_ptr,  # [chunk] int64 — GLOBAL target indices
    sum_exp_out_ptr,  # [chunk] fp32 OUT — sum_v exp(logit_v - max_global)
    pred_out_ptr,  # [chunk] fp32 OUT — predicted_logit - max_global (0 if off-rank/ignored)
    sum_x_minus_m_out_ptr,  # [chunk] fp32 OUT — sum_v (logit_v - max_global) (label smoothing)
    vocab_start,  # int — first global vocab index owned by this rank
    n_cols,  # V_local
    ignore_index,  # int
    HAS_LABEL_SMOOTHING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-token forward stats. One program per row of the chunk."""
    program_id = tl.program_id(0).to(tl.int64)
    LOGITS_ptr += program_id * LOGITS_stride

    y_global = tl.load(Y_ptr + program_id)
    m = tl.load(max_ptr + program_id).cast(tl.float32)

    is_ignored = y_global == ignore_index
    target_off_rank = (y_global < vocab_start) | (y_global >= vocab_start + n_cols)
    y_local = y_global - vocab_start

    # Predicted logit shifted by global max. Off-rank / ignored positions
    # contribute 0; the AllReduce(SUM) gathers the one nonzero contribution.
    if is_ignored or target_off_rank:
        pred_local = 0.0
    else:
        pred_local = tl.load(LOGITS_ptr + y_local).cast(tl.float32) - m

    # Single pass over V_local: shift inside, exp, accumulate sum_exp.
    # Shifted logits are <= 0 so exp() is in (0, 1] — no overflow.
    sum_exp = 0.0
    sum_x_minus_m = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        col_mask = X_offsets < n_cols
        X_block = tl.load(LOGITS_ptr + X_offsets, mask=col_mask, other=0.0).cast(tl.float32)
        shifted = X_block - m
        exp_block = tl.exp(shifted)
        exp_block = tl.where(col_mask, exp_block, 0.0)
        sum_exp += tl.sum(exp_block)
        if HAS_LABEL_SMOOTHING:
            sum_x_minus_m += tl.sum(tl.where(col_mask, shifted, 0.0))

    tl.store(pred_out_ptr + program_id, pred_local)
    tl.store(sum_exp_out_ptr + program_id, sum_exp)
    if HAS_LABEL_SMOOTHING:
        tl.store(sum_x_minus_m_out_ptr + program_id, sum_x_minus_m)


@triton.jit
def liger_fused_linear_vpce_bwd_grad_kernel(
    LOGITS_ptr,  # [chunk, V_local] recomputed logits (input dtype) — overwritten with grad
    LOGITS_stride,
    max_ptr,  # [chunk] fp32 — global max
    sum_exp_ptr,  # [chunk] fp32 — global sum_exp
    Y_ptr,  # [chunk] int64
    grad_out_ptr,  # [chunk] fp32 — upstream gradient (already includes loss-mask zeros)
    vocab_start,
    n_cols,
    ignore_index,
    alpha_eff,  # scalar (already rescaled by formula)
    eps_eff,  # scalar = alpha_eff / K
    HAS_LABEL_SMOOTHING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-token grad kernel — writes grads in-place over the logits tile (same
    dtype as input). Matches FLCE's pattern; avoids a separate fp32 grad buffer.

    Grad math: ``G = (softmax - eps_eff - one_hot(target_on_rank)·(1 - alpha_eff)) · grad_out``.
    Ignored rows produce all-zero grad.
    """
    program_id = tl.program_id(0).to(tl.int64)
    LOGITS_ptr += program_id * LOGITS_stride

    y_global = tl.load(Y_ptr + program_id)
    if y_global == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(LOGITS_ptr + offsets, 0.0, mask=offsets < n_cols)
        return

    target_off_rank = (y_global < vocab_start) | (y_global >= vocab_start + n_cols)
    y_local = y_global - vocab_start
    m = tl.load(max_ptr + program_id).cast(tl.float32)
    sum_exp_global = tl.load(sum_exp_ptr + program_id)
    grad_out = tl.load(grad_out_ptr + program_id).cast(tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        col_mask = offsets < n_cols
        X_block = tl.load(LOGITS_ptr + offsets, mask=col_mask, other=0.0).cast(tl.float32)
        softmax = tl.exp(X_block - m) / sum_exp_global

        if HAS_LABEL_SMOOTHING:
            softmax = softmax - eps_eff

        if not target_off_rank:
            is_target = offsets == y_local
            softmax = tl.where(is_target, softmax - (1.0 - alpha_eff), softmax)

        grad_block = softmax * grad_out
        grad_block = tl.where(col_mask, grad_block, 0.0)
        tl.store(LOGITS_ptr + offsets, grad_block, mask=col_mask)


def _get_num_warps(block_size: int) -> int:
    return 32 if not is_hip() else 16


def _select_block_size(n_cols: int) -> int:
    return min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))


def _select_chunk_size(BT: int, v_local: int, h: int) -> int:
    """Chunk-size heuristic.

    Starts from FLCE's ``next_pow2(BT / cdiv(V, H))`` (which targets per-chunk
    logits tile ≈ activation tile), then caps it so the per-chunk transient
    stays under ``_MEM_BUDGET_BYTES``. The cap matters at high V_local where
    the unbounded FLCE heuristic leaves huge tiles.
    """
    inc_factor = max(triton.cdiv(v_local, h), 1)
    chunk = triton.next_power_of_2(max(triton.cdiv(BT, inc_factor), 1))
    max_by_mem = max(_MEM_BUDGET_BYTES // (v_local * _BYTES_PER_ELEM_TRANSIENT), 64)
    chunk = min(chunk, triton.next_power_of_2(max_by_mem))
    chunk = min(chunk, BT)
    return max(chunk, 1)


class LigerFusedLinearVPCEFunction(torch.autograd.Function):
    """Autograd op for fused linear + vocab-parallel cross-entropy.

    Inputs:
      hidden_states: [S, B, H] (any dtype)
      weight:        [V_local, H] (any dtype, typically bf16)
      target:        [S, B] int64, GLOBAL vocab indices in [0, V_global)
      bias:          [V_local] or None
      tp_group:      torch.distributed.ProcessGroup or None
      ignore_index:  int
      label_smoothing:         float in [0, 1)
      label_smoothing_formula: "pytorch" or "megatron"
      label_smoothing_mode:    "global" or "partition"
      accum_dtype:   dtype for grad_W / grad_b accumulator (None -> weight.dtype).
                     ``torch.float32`` recommended for Megatron's
                     ``gradient_accumulation_fusion`` interop.

    Returns: per-token loss tensor shape [S, B].

    Backward returns grads w.r.t. ``hidden_states``, ``weight``, ``bias``;
    ``None`` for ``weight`` / ``bias`` when ``weight.main_grad`` is present
    (Megatron's fp32 grad accumulator; we add into it in-place and let the
    bucket reducer skip ``.grad`` accumulation).
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias,
        tp_group,
        ignore_index: int,
        label_smoothing: float,
        label_smoothing_formula: str,
        label_smoothing_mode: str,
        accum_dtype,
    ) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must be 3-D ([seq, batch, hidden]); got shape {tuple(hidden_states.shape)}."
            )
        if weight.dim() != 2:
            raise ValueError(f"weight must be 2-D ([vocab_local, hidden]); got shape {tuple(weight.shape)}.")
        s, b, h = hidden_states.shape
        BT = s * b
        v_local = weight.shape[0]
        if weight.shape[1] != h:
            raise ValueError(f"weight hidden dim mismatch: hidden_states H={h} vs weight H={weight.shape[1]}.")

        tp_distributed = _is_tp_distributed(tp_group)
        if tp_distributed:
            rank = dist.get_rank(tp_group)
            world = dist.get_world_size(tp_group)
        else:
            rank, world = 0, 1
        vocab_start = rank * v_local

        # Flatten input. Use contiguous storage so chunk slices are contiguous.
        x = hidden_states.reshape(BT, h).contiguous()
        flat_target = target.reshape(-1).contiguous().to(torch.int64)

        chunk_size = _select_chunk_size(BT, v_local, h)
        num_chunks = triton.cdiv(BT, chunk_size)

        # Per-token stats live for the duration of the forward.
        max_local = torch.empty(BT, device=x.device, dtype=torch.float32)
        sum_exp_local = torch.empty(BT, device=x.device, dtype=torch.float32)
        pred_local = torch.empty(BT, device=x.device, dtype=torch.float32)
        # Allocate sum_x_minus_m only when label smoothing is on. The kernel
        # writes to it only when HAS_LABEL_SMOOTHING is true (constexpr-gated).
        has_smoothing = label_smoothing > 0
        sum_x_minus_m_local = torch.empty(BT, device=x.device, dtype=torch.float32) if has_smoothing else None

        block_size = _select_block_size(v_local)
        num_warps = _get_num_warps(block_size)

        # === Pass 1: per-row max ===
        for chunk_id in range(num_chunks):
            s_idx = chunk_id * chunk_size
            e_idx = min((chunk_id + 1) * chunk_size, BT)
            x_chunk = x[s_idx:e_idx]
            logits_chunk = x_chunk @ weight.t()
            if bias is not None:
                logits_chunk = logits_chunk + bias
            max_local[s_idx:e_idx] = logits_chunk.amax(dim=-1).float()
            del logits_chunk

        if tp_distributed:
            dist.all_reduce(max_local, op=dist.ReduceOp.MAX, group=tp_group)

        # === Pass 2: per-row sum_exp, pred (and sum_x_minus_m if smoothing) ===
        # ``sum_x_minus_m_local`` may be None when smoothing=0. The kernel
        # only writes via this ptr when HAS_LABEL_SMOOTHING is True, so we
        # pass a 1-element dummy in that case (Triton requires a valid ptr).
        sum_x_arg = sum_x_minus_m_local if has_smoothing else pred_local
        for chunk_id in range(num_chunks):
            s_idx = chunk_id * chunk_size
            e_idx = min((chunk_id + 1) * chunk_size, BT)
            n = e_idx - s_idx
            x_chunk = x[s_idx:e_idx]
            logits_chunk = x_chunk @ weight.t()
            if bias is not None:
                logits_chunk = logits_chunk + bias
            logits_chunk = logits_chunk.contiguous()

            liger_fused_linear_vpce_fwd_stats_kernel[(n,)](
                LOGITS_ptr=logits_chunk,
                LOGITS_stride=logits_chunk.stride(0),
                max_ptr=max_local[s_idx:e_idx],
                Y_ptr=flat_target[s_idx:e_idx],
                sum_exp_out_ptr=sum_exp_local[s_idx:e_idx],
                pred_out_ptr=pred_local[s_idx:e_idx],
                sum_x_minus_m_out_ptr=sum_x_arg[s_idx:e_idx] if has_smoothing else sum_x_arg,
                vocab_start=vocab_start,
                n_cols=v_local,
                ignore_index=ignore_index,
                HAS_LABEL_SMOOTHING=has_smoothing,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
            del logits_chunk

        # === AllReduce stats ===
        has_global_smoothing = has_smoothing and label_smoothing_mode == "global"
        if tp_distributed:
            if has_global_smoothing:
                stats = torch.stack([sum_exp_local, pred_local, sum_x_minus_m_local], dim=-1)
            else:
                stats = torch.stack([sum_exp_local, pred_local], dim=-1)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=tp_group)
            if has_global_smoothing:
                sum_exp_global, pred_global, sum_x_minus_m_global = stats.unbind(dim=-1)
                sum_exp_global = sum_exp_global.contiguous()
                pred_global = pred_global.contiguous()
                sum_x_minus_m_global = sum_x_minus_m_global.contiguous()
            else:
                sum_exp_global, pred_global = stats.unbind(dim=-1)
                sum_exp_global = sum_exp_global.contiguous()
                pred_global = pred_global.contiguous()
                sum_x_minus_m_global = sum_x_minus_m_local
        else:
            sum_exp_global = sum_exp_local
            pred_global = pred_local
            sum_x_minus_m_global = sum_x_minus_m_local

        # === Compute per-token loss ===
        # The global max shift cancels: loss = log(Σ exp(x-m)) - (x_y - m) = log(Σ exp x) - x_y.
        loss = torch.log(sum_exp_global) - pred_global

        # === Label smoothing ===
        if has_smoothing:
            v_for_K = v_local * world if label_smoothing_mode == "global" else v_local
            if label_smoothing_formula == "megatron":
                alpha_eff = label_smoothing * v_for_K / (v_for_K - 1)
            else:  # "pytorch"
                alpha_eff = label_smoothing
            eps_eff = alpha_eff / v_for_K

            # log_softmax_v = (x_v - m_global) - log(sum_exp_global)
            # mean over K: sum(x_v - m) / K - log(sum_exp)
            if label_smoothing_mode == "global":
                mean_log_softmax = sum_x_minus_m_global / v_for_K - torch.log(sum_exp_global)
            else:  # partition
                mean_log_softmax = sum_x_minus_m_local / v_for_K - torch.log(sum_exp_global)
            loss = (1.0 - alpha_eff) * loss + alpha_eff * (-mean_log_softmax)
        else:
            alpha_eff = 0.0
            eps_eff = 0.0

        # Mask out ignored positions.
        if ignore_index is not None:
            loss = torch.where(flat_target == ignore_index, torch.zeros_like(loss), loss)

        # === Save for backward ===
        ctx.save_for_backward(x, weight, bias, flat_target, max_local, sum_exp_global)
        ctx.alpha_eff = float(alpha_eff)
        ctx.eps_eff = float(eps_eff)
        ctx.has_smoothing = bool(has_smoothing)
        ctx.vocab_start = vocab_start
        ctx.v_local = v_local
        ctx.ignore_index = ignore_index
        ctx.chunk_size = chunk_size
        ctx.num_chunks = num_chunks
        ctx.BT = BT
        ctx.orig_shape = (s, b, h)
        ctx.block_size = block_size
        ctx.accum_dtype = accum_dtype

        return loss.reshape(s, b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, bias, flat_target, max_global, sum_exp_global = ctx.saved_tensors
        BT, H = x.shape
        V_local = weight.shape[0]

        grad_out_flat = grad_output.contiguous().reshape(-1).to(torch.float32)

        weight_needs_grad = weight.requires_grad
        bias_needs_grad = bias is not None and bias.requires_grad

        grad_x = torch.empty_like(x)
        accum_dtype = ctx.accum_dtype if ctx.accum_dtype is not None else weight.dtype
        grad_W = torch.zeros(V_local, H, device=x.device, dtype=accum_dtype) if weight_needs_grad else None
        grad_b = torch.zeros(V_local, device=x.device, dtype=accum_dtype) if bias_needs_grad else None

        block_size = ctx.block_size
        num_warps = _get_num_warps(block_size)

        for chunk_id in range(ctx.num_chunks):
            s_idx = chunk_id * ctx.chunk_size
            e_idx = min((chunk_id + 1) * ctx.chunk_size, BT)

            x_chunk = x[s_idx:e_idx]
            # Recompute logits tile (the design tradeoff: +1 matmul over baseline,
            # avoid materializing [BT, V_local] for backward).
            logits_chunk = x_chunk @ weight.t()
            if bias is not None:
                logits_chunk = logits_chunk + bias
            logits_chunk = logits_chunk.contiguous()

            # Kernel writes grad in-place over logits_chunk (matches FLCE).
            liger_fused_linear_vpce_bwd_grad_kernel[(e_idx - s_idx,)](
                LOGITS_ptr=logits_chunk,
                LOGITS_stride=logits_chunk.stride(0),
                max_ptr=max_global[s_idx:e_idx],
                sum_exp_ptr=sum_exp_global[s_idx:e_idx],
                Y_ptr=flat_target[s_idx:e_idx],
                grad_out_ptr=grad_out_flat[s_idx:e_idx],
                vocab_start=ctx.vocab_start,
                n_cols=V_local,
                ignore_index=ctx.ignore_index,
                alpha_eff=ctx.alpha_eff,
                eps_eff=ctx.eps_eff,
                HAS_LABEL_SMOOTHING=ctx.has_smoothing,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
            grad_logits_chunk = logits_chunk  # alias for clarity

            # grad_x slice: grad_logits [n, V_local] @ W [V_local, H] -> [n, H].
            grad_x[s_idx:e_idx] = (grad_logits_chunk @ weight).to(grad_x.dtype)

            if grad_W is not None:
                if accum_dtype == torch.float32:
                    grad_W.addmm_(grad_logits_chunk.t().float(), x_chunk.float())
                else:
                    grad_W.add_((grad_logits_chunk.t() @ x_chunk).to(accum_dtype))

            if grad_b is not None:
                grad_b.add_(grad_logits_chunk.sum(dim=0).to(grad_b.dtype))

        # main_grad routing for Megatron's gradient_accumulation_fusion.
        # If the weight has a fp32 main_grad accumulator (set up by Megatron's
        # DistributedDataParallel wrappers), add into it in-place and return
        # None to autograd so the bucket reducer doesn't double-add.
        weight_has_main_grad = grad_W is not None and hasattr(weight, "main_grad") and weight.main_grad is not None
        bias_has_main_grad = grad_b is not None and hasattr(bias, "main_grad") and bias.main_grad is not None

        if weight_has_main_grad:
            weight.main_grad.add_(grad_W.to(weight.main_grad.dtype))
            # Mirror Megatron's marker so the bucket reducer skips this param.
            weight.grad_added_to_main_grad = True
            grad_W_return = None
        else:
            grad_W_return = grad_W.to(weight.dtype) if grad_W is not None else None

        if bias_has_main_grad:
            bias.main_grad.add_(grad_b.to(bias.main_grad.dtype))
            bias.grad_added_to_main_grad = True
            grad_b_return = None
        else:
            grad_b_return = grad_b.to(bias.dtype) if grad_b is not None else None

        s, b, h = ctx.orig_shape
        grad_x_3d = grad_x.view(s, b, h)

        # 10 inputs to forward -> 10 returns from backward.
        # (hidden_states, weight, target, bias, tp_group, ignore_index,
        #  label_smoothing, formula, mode, accum_dtype)
        return (
            grad_x_3d,
            grad_W_return,
            None,
            grad_b_return,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# Re-export validate helper so wrappers in transformers/ and megatron/ can import
# from a single module without re-importing from vocab_parallel_cross_entropy.
__all__ = ["LigerFusedLinearVPCEFunction", "validate_formula_and_mode"]
