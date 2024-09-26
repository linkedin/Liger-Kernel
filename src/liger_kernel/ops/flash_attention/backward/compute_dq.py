import triton
import triton.language as tl

from src.liger_kernel.ops.flash_attention.utils import load_fn


@triton.jit
def _compute_single_block_dq(
    I_start_n,
    q,
    dq,
    do,
    lse_i,
    delta_i,
    offs_m,
    offs_n,
    offs_d,
    k_ptrs,
    v_ptrs,
    bias_ptrs,
    dropout_offs,
    softmax_scale,
    dropout_p,
    dropout_seed,
    stride_kn,
    stride_vn,
    actual_seqlen_q,
    actual_seqlen_k,
    headdim,
    MASKED: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_ON: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    PAD_COLS: tl.constexpr,
    HEADS_PADDED: tl.constexpr,
):
    # Relocate pointers and offsets
    k_ptrs = k_ptrs + I_start_n * stride_kn
    v_ptrs = v_ptrs + I_start_n * stride_vn
    offs_n_curr = I_start_n + offs_n
    if BIAS_ON:
        bias_ptrs += I_start_n
    if USE_DROPOUT:
        dropout_offs += I_start_n

    # Load K, V and LSE now to reduce pipeline stall
    k = load_fn(k_ptrs, offs_n_curr, offs_d, PAD_COLS, HEADS_PADDED, actual_seqlen_k, headdim)
    v = load_fn(v_ptrs, offs_n_curr, offs_d, PAD_COLS, HEADS_PADDED, actual_seqlen_k, headdim)
    if BIAS_ON:
        bias = load_fn(bias_ptrs, offs_m, offs_n_curr, True, PAD_COLS, actual_seqlen_q, actual_seqlen_k)  # TODO: pad rows

    # Recompute P_ij = softmax(qk, dim=-1).T
    qk = tl.dot(q, tl.trans(k))
    if BIAS_ON:
        qk += bias / softmax_scale  # TODO: check if this is optimal

    offs_n_causal = (offs_n_curr - actual_seqlen_k + actual_seqlen_q)

    # Attention and causal mask
    if MASKED:
        if PAD_COLS:
            if IS_CAUSAL:
                qk = tl.where(tl.minimum(actual_seqlen_q - 1, offs_m)[:, None] >= offs_n_causal[None, :], qk, float("-inf"))
            else:
                qk = tl.where(actual_seqlen_q - 1 >= offs_n_causal[None, :], qk, float("-inf"))
        elif IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n_causal[None, :], qk, float("-inf"))
    tl.debug_barrier()

    p = tl.exp2(qk * (softmax_scale * 1.44269504089) - lse_i[:, None])
    dp = tl.dot(do, tl.trans(v))

    ds = (p * (dp - delta_i[:, None]) * softmax_scale).to(q.dtype)

    # compute dq
    dq += tl.dot(ds, k)

    return dq


@triton.jit
def _compute_row_blocks_dq(
    I_start_m,
    Q,
    K,
    V,
    Bias,
    Dropout,
    DO,
    DQ,
    LSE,
    D,
    softmax_scale,
    dropout_p,
    dropout_seed,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_bm,
    stride_dom,
    stride_dqm,
    actual_seqlen_q,
    actual_seqlen_k,
    headdim,
    VARLEN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_ON: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    PAD_ROWS: tl.constexpr,
    HEADS_PADDED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    # This fuction goes through a row, so it always starts at i = 0 but the end can vary because of causality
    if IS_CAUSAL:
        I_end_n = min(actual_seqlen_k - actual_seqlen_q + I_start_m + BLOCK_M, actual_seqlen_k)
        # For a seqlen_q >> seqlen_k, there migh be entire block skipped
        if I_end_n < 0:
            return
    else:
        I_end_n = actual_seqlen_k
    # Compute the number of fully masked lines
    fully_masked_lines = actual_seqlen_q - actual_seqlen_k if IS_CAUSAL else 0
    # Exit if the block is fully masked or the current row is greater than the actual sequence length
    if (I_start_m >= actual_seqlen_q) or (fully_masked_lines >= I_start_m + BLOCK_M):
        return

    # Initialize offsets
    offs_m = tl.arange(0, BLOCK_M) + I_start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize value-related pointer (not stats-related)
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_d[None, :])
    do_ptrs = DO + (offs_m[:, None] * stride_dom + offs_d[None, :])
    if BIAS_ON:
        bias_ptrs = Bias + (offs_m[:, None] * stride_bm + offs_n[None, :])
    else:
        bias_ptrs = None
    if USE_DROPOUT:
        dropout_offs = Dropout + (offs_m[:, None] * stride_bm + offs_n[None, :])
    else:
        dropout_offs = None

    # Initialize the dq accumulator
    dq = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # Load Q, DO, LSE and D, which will stay in SRAM for the row-wise loop
    q = load_fn(
        q_ptrs, offs_m, offs_d,
        PAD_AXIS_0=PAD_ROWS, PAD_AXIS_1=HEADS_PADDED,
        LIM_AXIS_0=actual_seqlen_q, LIM_AXIS_1=headdim,
    )
    do = load_fn(
        do_ptrs, offs_m, offs_d,
        PAD_AXIS_0=PAD_ROWS, PAD_AXIS_1=HEADS_PADDED,
        LIM_AXIS_0=actual_seqlen_q, LIM_AXIS_1=headdim,
    )
    lse_i = tl.load(LSE + offs_m)  # since lse is padded to max_seqlen_q, should be good
    delta_i = tl.load(D + offs_m)  # same as LSE for now

    # Infer the number of full and partially masked blocks
    uneven_n = (actual_seqlen_k % BLOCK_N != 0)
    attention_padding = VARLEN & uneven_n
    if IS_CAUSAL:
        first_masked_col = I_start_m + 1 + actual_seqlen_k - actual_seqlen_q
    elif attention_padding:
        first_masked_col = actual_seqlen_k
    else:
        first_masked_col = I_end_n
    nb_full_blocks = first_masked_col // BLOCK_N

    # Loop over rows to compute dk and dv
    I_next_start_n = 0
    if nb_full_blocks > 0:
        for _ in range(0, nb_full_blocks):
            I_next_start_n = tl.multiple_of(I_next_start_n, BLOCK_N)
            dq = _compute_single_block_dq(
                I_next_start_n,
                q,
                dq,
                do,
                lse_i,
                delta_i,
                offs_m,
                offs_n,
                offs_d,
                k_ptrs,
                v_ptrs,
                bias_ptrs,
                dropout_offs,
                softmax_scale,
                dropout_p,
                dropout_seed,
                stride_kn,
                stride_vn,
                actual_seqlen_q,
                actual_seqlen_k,
                headdim,
                IS_CAUSAL=IS_CAUSAL,
                BIAS_ON=BIAS_ON,
                USE_DROPOUT=USE_DROPOUT,
                MASKED=False,
                PAD_COLS=False,
                HEADS_PADDED=HEADS_PADDED,
            )
            I_next_start_n += BLOCK_N

    if I_next_start_n < I_end_n:
        for I_start_n in range(I_next_start_n, I_end_n, BLOCK_N):
            pad_cols = (not EVEN_N) or (VARLEN and (I_start_n + BLOCK_N > actual_seqlen_k))
            dq = _compute_single_block_dq(
                I_start_n,
                q,
                dq,
                do,
                lse_i,
                delta_i,
                offs_m,
                offs_n,
                offs_d,
                k_ptrs,
                v_ptrs,
                bias_ptrs,
                dropout_offs,
                softmax_scale,
                dropout_p,
                dropout_seed,
                stride_kn,
                stride_vn,
                actual_seqlen_q,
                actual_seqlen_k,
                headdim,
                IS_CAUSAL=IS_CAUSAL,
                BIAS_ON=BIAS_ON,
                USE_DROPOUT=USE_DROPOUT,
                MASKED=True,
                PAD_COLS=pad_cols,
                HEADS_PADDED=HEADS_PADDED,
            )

    # Account for fully masked lines
    if fully_masked_lines > 0:
        dq = tl.where(offs_m[:, None] < fully_masked_lines, 0, dq)

    # Store dq
    if HEADS_PADDED:
        if PAD_ROWS:
            tl.store(dq_ptrs, dq, mask=(offs_m[:, None] < actual_seqlen_q) & (offs_d[None, :] < headdim))
        else:
            tl.store(dq_ptrs, dq, mask=offs_d[None, :] < headdim)
    else:
        if PAD_ROWS:
            tl.store(dq_ptrs, dq, mask=offs_m[:, None] < actual_seqlen_q)
        else:
            tl.store(dq_ptrs, dq)
