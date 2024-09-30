import triton
import triton.language as tl
from triton import Config

from typing import List, Any, Dict
from src.liger_kernel.ops.flash_attention.forward.compute_row_blocks import (
    compute_row_block,
)
from src.liger_kernel.ops.flash_attention.utils import load_fn

# TODO: exit causal blocks early
# TODO: can we initialize accO to empty instead of 0?

MIN_B = 32


def early_config_prune_fwd_kernel(
    configs: List[Config],
    named_args: Dict[str, Any],
    **kwargs,
) -> List[Config]:
    # Remove the configs where BLOCK_ > seqlen_
    kept_configs = []
    for cfg in configs:
        block_m_too_large = cfg.kwargs["BLOCK_M"] > named_args["seqlen_q"]
        block_n_too_large = cfg.kwargs["BLOCK_N"] > named_args["seqlen_k"]
        if block_m_too_large or block_n_too_large:
            pass
        else:
            kept_configs.append(cfg)
    # If no config is left, go for the minimal config
    if kept_configs:
        return kept_configs
    return [Config({"BLOCK_M": MIN_B, "BLOCK_N": MIN_B}, num_warps=4, num_stages=1)]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": MIN_B, "BLOCK_N": MIN_B}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=4, num_stages=1),
    ],
    key=[
        "CACHE_KEY_SEQLEN_Q",
        "CACHE_KEY_SEQLEN_K",
        "DTYPE",
        "VARLEN",
        "USE_DROPOUT",
        "IS_CAUSAL",
        "BIAS_ON",
        "BLOCK_HEADDIM",
    ],
    prune_configs_by={"early_config_prune": early_config_prune_fwd_kernel},
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    Lse,
    Bias,
    softmax_scale,
    dropout_p,
    dropout_seed,
    stride_qb,
    stride_qh,
    stride_qm,  # Q stride for the batch, head and sequence axis (sequence subscript is m for rows)
    stride_kb,
    stride_kh,
    stride_kn,  # Same for K (sequence subscript is n for cols)
    stride_vb,
    stride_vh,
    stride_vn,  # Same for V (sequence subscript is n for cols)
    stride_ob,
    stride_oh,
    stride_om,  # Same for O (sequence subscript is m for rows)
    stride_bb,
    stride_bh,
    stride_bm,
    nheads_q,
    head_ratio,
    seqlen_q,
    cum_seqlens_q,
    seqlen_k,
    max_seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    DTYPE,
    VARLEN: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_ON: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Locate kernel inside the grid
    i_start_m = tl.program_id(0)  # current block in the Q matrix
    off_head_and_batch = tl.program_id(1)
    off_head_q = off_head_and_batch % nheads_q
    off_head_kv = off_head_q // head_ratio
    off_batch = off_head_and_batch // nheads_q

    # Infer actual sequence length of Q and the offset to the last sequence
    if VARLEN:
        cu_seq_start_q = tl.load(cum_seqlens_q + off_batch)
        actual_seqlen_q = tl.load(cum_seqlens_q + off_batch + 1) - cu_seq_start_q
        if i_start_m * BLOCK_M >= actual_seqlen_q:
            return
        actual_seqlen_k = (
            actual_seqlen_q  # TODO: support packed + varlen? rn, check is done outside
        )
        cu_seq_start_k = cu_seq_start_q
        off_batch = 0
    else:
        actual_seqlen_q = seqlen_q
        actual_seqlen_k = seqlen_k
        cu_seq_start_q = 0
        cu_seq_start_k = 0

    softmax_scale = softmax_scale * 1.44269504089
    # Initialize offsets
    offs_m = i_start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # When in VARLEN mode, since we dimension the grid to be large enough for all sequences, the
    # current sequence might have less rows than the current row (detemined through the grid).

    fully_masked_lines = actual_seqlen_q - actual_seqlen_k if IS_CAUSAL else 0
    if fully_masked_lines >= (i_start_m + 1) * BLOCK_M:
        return

    # Initialize pointers to Q, K, V
    offseted_Q = (
        Q + off_batch * stride_qb + off_head_q * stride_qh + cu_seq_start_q * stride_qm
    )
    q_ptrs = offseted_Q + (offs_m[:, None] * stride_qm + offs_d[None, :])
    offseted_K = (
        K + off_batch * stride_kb + off_head_kv * stride_kh + cu_seq_start_k * stride_kn
    )
    k_ptrs = offseted_K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    offseted_V = (
        V + off_batch * stride_vb + off_head_kv * stride_vh + cu_seq_start_k * stride_vn
    )
    v_ptrs = offseted_V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    # ...and maybe bias
    if BIAS_ON:
        offseted_Bias = (
            Bias
            + off_batch * stride_bb
            + off_head_kv * stride_bh
            + cu_seq_start_q * stride_bm
        )
        bias_ptrs = offseted_Bias + (offs_m[:, None] * stride_bm + offs_n[None, :])
    else:
        bias_ptrs = None
    # ...and maybe dropout
    if USE_DROPOUT:
        dropout_off = actual_seqlen_k * (
            cu_seq_start_q + actual_seqlen_q * (off_head_q + nheads_q * off_batch)
        )
        dropout_offs = dropout_off + offs_m[:, None] * actual_seqlen_k + offs_n[None, :]
    else:
        dropout_offs = None

    # Initialize pointers to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # Load Q, which will stay in SRAM for the whole loop
    pad_rows = (not EVEN_M) or (
        VARLEN and (i_start_m * BLOCK_M > actual_seqlen_q)
    )  # this works while other bools fail. Why?
    q = load_fn(
        q_ptrs,
        offs_m,
        offs_d,
        PAD_AXIS_0=pad_rows,
        PAD_AXIS_1=PADDED_HEADS,
        LIM_AXIS_0=actual_seqlen_q,
        LIM_AXIS_1=headdim,
    )

    # Compute last visited column of KV which
    if IS_CAUSAL:
        end_n = min(
            actual_seqlen_k - actual_seqlen_q + (i_start_m + 1) * BLOCK_M,
            actual_seqlen_k,
        )
        # For a seqlen_q >> seqlen_k, there migh be entire block skipped
        if end_n < 0:
            return
    else:
        end_n = actual_seqlen_k

    # first_masked_block = min(start_m * BLOCK_M + 1 + actual_seqlen_k - actual_seqlen_q, end_n) if IS_CAUSAL else end_n
    uneven_n = actual_seqlen_k % BLOCK_N != 0
    attention_padding = VARLEN & uneven_n
    if IS_CAUSAL:
        first_masked_col = i_start_m * BLOCK_M + 1 + actual_seqlen_k - actual_seqlen_q
    elif attention_padding:
        first_masked_col = actual_seqlen_k
    else:
        first_masked_col = end_n
    nb_full_blocks = first_masked_col // BLOCK_N

    next_start_n = 0
    if nb_full_blocks > 0:
        for _ in range(0, nb_full_blocks):
            m_i, lse_i, acc_o = compute_row_block(
                q,
                m_i,
                lse_i,
                k_ptrs,
                v_ptrs,
                bias_ptrs,
                acc_o,
                offs_m,
                offs_n,
                offs_d,
                softmax_scale,
                dropout_p,
                dropout_seed,
                dropout_offs,
                stride_kn,
                stride_vn,
                next_start_n,
                actual_seqlen_q,
                actual_seqlen_k,
                headdim,
                USE_DROPOUT=USE_DROPOUT,
                IS_CAUSAL=IS_CAUSAL,
                BIAS_ON=BIAS_ON,
                MASKED=False,
                PADDED_COLS=False,
                PADDED_HEADS=PADDED_HEADS,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
            next_start_n += BLOCK_N

    if next_start_n < end_n:
        for I_start_n in range(next_start_n, end_n, BLOCK_N):
            pad_cols = (not EVEN_N) or VARLEN  # TODO: refine varlen side
            m_i, lse_i, acc_o = compute_row_block(
                q,
                m_i,
                lse_i,
                k_ptrs,
                v_ptrs,
                bias_ptrs,
                acc_o,
                offs_m,
                offs_n,
                offs_d,
                softmax_scale,
                dropout_p,
                dropout_seed,
                dropout_offs,
                stride_kn,
                stride_vn,
                I_start_n,
                actual_seqlen_q,
                actual_seqlen_k,
                headdim,
                USE_DROPOUT=USE_DROPOUT,
                IS_CAUSAL=IS_CAUSAL,
                BIAS_ON=BIAS_ON,
                MASKED=True,
                PADDED_COLS=pad_cols,
                PADDED_HEADS=PADDED_HEADS,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

    # Final scaling of the output accumulator
    if USE_DROPOUT:
        o_scale = tl.exp2((m_i - lse_i) - tl.log2(1 - dropout_p))
    else:
        o_scale = tl.exp2(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]

    # For seqlen_q >> seqlen_k, there might be entire lines masked, so we account for that
    if fully_masked_lines > i_start_m * BLOCK_M:
        acc_o = tl.where(offs_m[:, None] < fully_masked_lines, 0, acc_o)

    # rematerialize offsets to save registers (?)
    i_start_m = tl.program_id(0)
    offs_m = i_start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Write back l and m
    # Q + off_batch * stride_qb + off_head * stride_qh + cu_seq_start_q * stride_qm
    lse_ptrs = Lse + off_head_and_batch * max_seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # Initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_batch * stride_ob
        + off_head_q * stride_oh
        + cu_seq_start_q * stride_om
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )

    # Store O (same mechanism as Q) BUG: here, the store instruction seems to fail when one of the two bools is false
    if True:
        tl.store(
            out_ptrs,
            acc_o,
            mask=(offs_m[:, None] < actual_seqlen_q) & (offs_d[None, :] < headdim),
        )
    elif pad_rows:
        tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < actual_seqlen_q)
    elif PADDED_HEADS:  # nothing is padded
        tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:  # only heads are padded
        tl.store(out_ptrs, acc_o)
