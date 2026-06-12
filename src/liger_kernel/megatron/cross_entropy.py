"""Megatron-Core compatible cross-entropy backed by Liger's Triton kernel.

Two public Module classes, both backed by the same ``LigerVocabParallelCEFunction``
autograd op and the same Triton kernels in ``liger_kernel.ops.vocab_parallel_cross_entropy``:

  - ``LigerVocabParallelCrossEntropy`` — general-purpose vocab-parallel CE with
    defaults matching ``F.cross_entropy`` semantics (``formula="pytorch"``,
    ``mode="global"``).

  - ``LigerMegatronCrossEntropy`` — same kernel, defaults matching Megatron-LM's
    ``vocab_parallel_cross_entropy`` bit-for-bit (``formula="megatron"``,
    ``mode="partition"``). One-line replacement for Megatron's CE for users coming
    from ``megatron.core.tensor_parallel.cross_entropy``.

Supports all tensor-parallel sizes (TP=1 and TP>1). At TP=1 the same code path
runs with no AllReduces.

Label smoothing — two formulas:

  - ``"pytorch"``: ``q' = (1 - alpha) * q + (alpha / K) * uniform``. Same as
    ``torch.nn.functional.cross_entropy(..., label_smoothing=alpha)``.
  - ``"megatron"``: ``q' = (1 - alpha) * q + (alpha / (K - 1)) * uniform_excluding_gt``.
    Same as Megatron's NeMo-derived formula.

The structural form is identical (``loss = (1 - alpha_eff) * H(q, p) + alpha_eff * H(u, p)``)
so the kernel only needs ``alpha_eff = alpha`` (PyTorch) or ``alpha_eff = alpha * K / (K - 1)``
(Megatron). The Python wrapper rescales and forwards.

Averaging scope at TP>1 (irrelevant at TP=1 since V_local == V_global):

  - ``"global"``: ``H(u, p)`` averaged over V_global. Adds one AllReduce on
    ``sum(log_softmax_local, dim=-1)``. Exact result — matches a single-rank
    reference computed on the full logits.
  - ``"partition"``: ``H(u, p)`` averaged over V_local on each rank. No extra
    AllReduce. Loss tensor differs per rank under this mode — this mirrors
    Megatron's existing behavior verbatim.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from liger_kernel.ops.vocab_parallel_cross_entropy import get_num_warps
from liger_kernel.ops.vocab_parallel_cross_entropy import liger_vocab_parallel_ce_backward_kernel
from liger_kernel.ops.vocab_parallel_cross_entropy import liger_vocab_parallel_ce_forward_kernel
from liger_kernel.ops.vocab_parallel_cross_entropy import select_block_size

_FORMULAS = ("pytorch", "megatron")
_MODES = ("global", "partition")


def _is_tp_distributed(tp_group) -> bool:
    """True iff ``tp_group`` is a real multi-rank ProcessGroup."""
    return tp_group is not None and hasattr(tp_group, "size") and tp_group.size() > 1


class LigerVocabParallelCEFunction(torch.autograd.Function):
    """Autograd op for vocab-parallel cross-entropy. Shared by both Module classes.

    Forward layout:
      1. fp32 working buffer (cloned from input — never modify the caller's tensor)
      2. AllReduce(MAX) on per-token max  → shift logits by global max
      3. Kernel 1: per-token sum_exp_local + predicted_logit_local; writes exp(shifted)
         in-place into the working buffer for backward reuse
      4. AllReduce(SUM) x2 to aggregate predicted and sum_exp across ranks
      5. Loss = log(sum_exp_global) - predicted_global, then optional label smoothing
      6. Save the in-place buffer (now exp(shifted)) for backward

    Backward layout:
      1. Kernel 2 reads exp(shifted), normalizes by sum_exp_global, subtracts the
         one-hot target on the owning rank, scales by grad_output, writes grad
         in-place. No AllReduce needed.
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
        s, b, v_local = vocab_parallel_logits.shape

        # Cloned fp32 working buffer — we mutate it in-place across kernels, so it
        # MUST be independent of the caller's tensor. .to(copy=True) forces a copy
        # even if the input was already fp32.
        logits = vocab_parallel_logits.detach().to(dtype=torch.float32, copy=True).contiguous()

        tp_distributed = _is_tp_distributed(tp_group)
        if tp_distributed:
            rank = dist.get_rank(tp_group)
            world = dist.get_world_size(tp_group)
        else:
            rank, world = 0, 1

        # --- 1. Global max + shift ---
        logits_max = logits.amax(dim=-1)  # [S, B]
        if tp_distributed:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
        logits.sub_(logits_max.unsqueeze(-1))  # in-place on the cloned buffer

        # --- 2. Vocab range and masks ---
        v_local_int = int(v_local)
        vocab_start = rank * v_local_int
        flat_target = target.reshape(-1).contiguous().to(torch.int64)
        target_off_rank_bool = (flat_target < vocab_start) | (flat_target >= vocab_start + v_local_int)
        masked_target = (flat_target - vocab_start).clamp(0, v_local_int - 1).to(torch.int64)
        ignore_mask_bool = flat_target == ignore_index

        # int8 because Triton's per-scalar tl.load is more robust with int8 than bool
        target_mask_i8 = target_off_rank_bool.to(torch.int8)
        ignore_mask_i8 = ignore_mask_bool.to(torch.int8)

        # --- 3. Kernel 1: forward stats + in-place exp ---
        logits_2d = logits.view(-1, v_local_int)
        BT = s * b
        predicted_local = torch.zeros(BT, device=logits.device, dtype=torch.float32)
        sum_exp_local = torch.zeros(BT, device=logits.device, dtype=torch.float32)
        block_size = select_block_size(v_local_int)
        num_warps = get_num_warps(block_size)

        liger_vocab_parallel_ce_forward_kernel[(BT,)](
            X_ptr=logits_2d,
            X_stride=logits_2d.stride(0),
            Y_ptr=masked_target,
            target_mask_ptr=target_mask_i8,
            ignore_mask_ptr=ignore_mask_i8,
            pred_ptr=predicted_local,
            sum_exp_ptr=sum_exp_local,
            n_cols=v_local_int,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )

        # --- 4. AllReduces (no-ops at TP=1) ---
        if tp_distributed:
            dist.all_reduce(predicted_local, op=dist.ReduceOp.SUM, group=tp_group)
            dist.all_reduce(sum_exp_local, op=dist.ReduceOp.SUM, group=tp_group)

        # --- 5. Loss ---
        # log(sum_exp_global) - predicted_global. Ignored positions get loss=0
        # but we still computed exp+sum_exp normally so log() is safe (sum_exp > 0).
        loss = torch.log(sum_exp_local) - predicted_local

        # --- 6. Label smoothing (Python; chooses K, formula, averaging scope) ---
        if label_smoothing > 0:
            v_for_K = v_local_int * world if label_smoothing_mode == "global" else v_local_int
            if label_smoothing_formula == "megatron":
                alpha_eff = label_smoothing * v_for_K / (v_for_K - 1)
            else:  # "pytorch"
                alpha_eff = label_smoothing
            eps_eff = alpha_eff / v_for_K

            # logits_2d now holds exp(shifted) from Kernel 1. log_softmax_i =
            # log(exp_i / sum_exp_global) = log(exp_i) - log(sum_exp_global).
            log_probs_local = torch.log(logits_2d) - torch.log(sum_exp_local).unsqueeze(-1)

            if label_smoothing_mode == "global":
                sum_log_probs = log_probs_local.sum(dim=-1)
                if tp_distributed:
                    dist.all_reduce(sum_log_probs, op=dist.ReduceOp.SUM, group=tp_group)
                mean_log_probs = sum_log_probs / v_for_K
            else:  # "partition"
                mean_log_probs = log_probs_local.mean(dim=-1)

            # H(u, p) = -mean_log_probs
            loss = (1.0 - alpha_eff) * loss + alpha_eff * (-mean_log_probs)
        else:
            alpha_eff = 0.0
            eps_eff = 0.0

        # Mask ignored positions AFTER all loss computation
        loss = torch.where(ignore_mask_bool, torch.zeros_like(loss), loss)

        # --- 7. Save for backward (logits_2d == exp(shifted) buffer) ---
        ctx.save_for_backward(logits_2d, sum_exp_local, target_mask_i8, masked_target, ignore_mask_i8)
        ctx.alpha_eff = float(alpha_eff)
        ctx.eps_eff = float(eps_eff)
        ctx.has_smoothing = bool(label_smoothing > 0)
        ctx.v_local = v_local_int
        ctx.orig_dtype = vocab_parallel_logits.dtype
        ctx.orig_shape = (s, b)
        ctx.block_size = block_size

        return loss.reshape(s, b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        exp_buf, sum_exp_global, target_mask_i8, masked_target, ignore_mask_i8 = ctx.saved_tensors

        grad_out_flat = grad_output.contiguous().reshape(-1).to(torch.float32)
        BT = exp_buf.size(0)
        num_warps = get_num_warps(ctx.block_size)

        liger_vocab_parallel_ce_backward_kernel[(BT,)](
            EXP_ptr=exp_buf,
            EXP_stride=exp_buf.stride(0),
            sum_exp_ptr=sum_exp_global,
            Y_ptr=masked_target,
            target_mask_ptr=target_mask_i8,
            ignore_mask_ptr=ignore_mask_i8,
            grad_out_ptr=grad_out_flat,
            n_cols=ctx.v_local,
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


class LigerVocabParallelCrossEntropy(nn.Module):
    """General-purpose vocab-parallel cross-entropy. PyTorch/F.cross_entropy semantics.

    See module docstring for full details on the ``label_smoothing_formula`` and
    ``label_smoothing_mode`` arguments.

    Args:
        ignore_index: Target value to ignore (no loss, no gradient).
        label_smoothing: Smoothing strength in [0, 1).
        reduction: Must be ``"none"`` — vocab-parallel CE returns per-token loss
            shaped ``[seq, batch]``; reduction happens downstream.
        label_smoothing_formula: ``"pytorch"`` (default) or ``"megatron"``. Selects
            which label-smoothing definition the math follows. With
            ``label_smoothing=0`` both are identical.
        label_smoothing_mode: ``"global"`` (default) or ``"partition"``. Selects
            whether the ``H(u, p)`` averaging covers V_global (one extra
            AllReduce, exact at TP>1) or V_local on each rank (no extra
            AllReduce, per-rank loss may differ). Irrelevant at TP=1.

    Shape:
        Input: ``vocab_parallel_logits`` shape ``[S, B, V_local]``; ``target``
            shape ``[S, B]``. ``target`` values are GLOBAL vocab indices in
            ``[0, V_global)``.
        Output: per-token loss shape ``[S, B]``.

    Forward:
        ``ce(logits, target, tp_group=None)``. Pass ``tp_group`` (a
        ``torch.distributed.ProcessGroup``) when running with TP>1; pass ``None``
        for TP=1 / single-GPU. Passing a single-rank group is also valid.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "none",
        label_smoothing_formula: str = "pytorch",
        label_smoothing_mode: str = "global",
    ):
        super().__init__()
        if reduction != "none":
            raise ValueError(f"Vocab-parallel CE returns per-token loss; reduction must be 'none', got {reduction!r}.")
        if label_smoothing_formula not in _FORMULAS:
            raise ValueError(f"label_smoothing_formula must be one of {_FORMULAS}, got {label_smoothing_formula!r}.")
        if label_smoothing_mode not in _MODES:
            raise ValueError(f"label_smoothing_mode must be one of {_MODES}, got {label_smoothing_mode!r}.")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}.")
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.label_smoothing_formula = label_smoothing_formula
        self.label_smoothing_mode = label_smoothing_mode

    def forward(
        self,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        tp_group=None,
    ) -> torch.Tensor:
        if vocab_parallel_logits.dim() != 3:
            raise ValueError(
                f"vocab_parallel_logits must be 3-D ([seq, batch, vocab]); "
                f"got shape {tuple(vocab_parallel_logits.shape)}. (HuggingFace's "
                f"[batch, seq, vocab] callers must transpose before calling.)"
            )
        return LigerVocabParallelCEFunction.apply(
            vocab_parallel_logits,
            target,
            tp_group,
            self.ignore_index,
            self.label_smoothing,
            self.label_smoothing_formula,
            self.label_smoothing_mode,
        )

    def extra_repr(self) -> str:
        return (
            f"ignore_index={self.ignore_index}, label_smoothing={self.label_smoothing}, "
            f"reduction={self.reduction!r}, label_smoothing_formula={self.label_smoothing_formula!r}, "
            f"label_smoothing_mode={self.label_smoothing_mode!r}"
        )


class LigerMegatronCrossEntropy(LigerVocabParallelCrossEntropy):
    """``nn.Module`` drop-in for Megatron's vocab-parallel cross-entropy.

    Same kernels + same autograd op as :class:`LigerVocabParallelCrossEntropy`,
    but defaults match Megatron-LM's ``vocab_parallel_cross_entropy`` bit-for-bit:
    ``label_smoothing_formula="megatron"``, ``label_smoothing_mode="partition"``.

    With ``label_smoothing=0`` (the common case) the two classes are
    indistinguishable. With ``label_smoothing > 0``, this class reproduces
    Megatron's NeMo smoothing formula and Megatron's V_local averaging choice —
    so users monkey-patching from Megatron see zero numerical change.

    Conforms to ``megatron.core.fusions.fused_cross_entropy.fused_vocab_parallel_cross_entropy``'s
    signature, ``(vocab_parallel_logits, target, tp_group=None)``.

    Args mirror :class:`LigerVocabParallelCrossEntropy`; only the defaults
    differ.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "none",
        label_smoothing_formula: str = "megatron",
        label_smoothing_mode: str = "partition",
    ):
        super().__init__(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
            label_smoothing_formula=label_smoothing_formula,
            label_smoothing_mode=label_smoothing_mode,
        )
