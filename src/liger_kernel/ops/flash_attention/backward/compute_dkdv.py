import triton
import triton.language as tl

from src.liger_kernel.ops.flash_attention.utils import load_fn


@triton.jit
def _compute_single_block_dkdv(
    I_start_m,
    k,
    v,
    dk,
    dv,
    LSE,
    D,
    offs_m,
    offs_n,
    offs_d,
    q_ptrs,
    bias_ptrs,
    dropout_offs,
    do_ptrs,
    softmax_scale,
    dropout_p,
    dropout_seed,
    stride_qm,
    stride_bm,
    stride_dom,
    actual_seqlen_q,
    actual_seqlen_k,
    fully_masked_lines,
    headdim,
    MASKED: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_ON: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    PAD_ROWS: tl.constexpr,
    PAD_COLS: tl.constexpr,
    HEADS_PADDED: tl.constexpr,
):
    # Relocate pointers
    q_ptrs = q_ptrs + I_start_m * stride_qm
    do_ptrs = do_ptrs + I_start_m * stride_dom
    if BIAS_ON:
        bias_ptrs = bias_ptrs + I_start_m * stride_bm
    if USE_DROPOUT:
        dropout_offs += I_start_m * actual_seqlen_k

    # Update row variables
    offs_m_curr = I_start_m + offs_m

    # Load Q and LSE now to reduce pipeline stall
    # BUG: if one is true and the ther not, q is filled with wrong values
    q = load_fn(
        q_ptrs,
        offs_m_curr,
        offs_d,
        PAD_ROWS or HEADS_PADDED,
        PAD_ROWS or HEADS_PADDED,
        actual_seqlen_q,
        headdim,
    )
    lse_i = tl.load(
        LSE + offs_m_curr
    )  # since lsm is padded to max_seqlen_q, should be good
    if BIAS_ON:
        bias = load_fn(
            bias_ptrs,
            offs_m_curr,
            offs_n,
            PAD_ROWS or HEADS_PADDED,
            PAD_ROWS or HEADS_PADDED,
            actual_seqlen_q,
            actual_seqlen_k,
        )

    # Recompute P_ij = softmax(qk, dim=-1).T
    qk = tl.dot(q, tl.trans(k))
    if BIAS_ON:
        qk += bias / softmax_scale  # TODO: check if this is optimal

    # Attention and causal mask
    offs_n_causal = offs_n - actual_seqlen_k + actual_seqlen_q
    if MASKED:
        if PAD_COLS:
            if IS_CAUSAL:
                qk = tl.where(
                    tl.minimum(actual_seqlen_q - 1, offs_m_curr)[:, None]
                    >= offs_n_causal[None, :],
                    qk,
                    float("-inf"),
                )
            else:
                qk = tl.where(
                    actual_seqlen_q - 1 >= offs_n_causal[None, :], qk, float("-inf")
                )
        elif IS_CAUSAL:
            qk = tl.where(
                offs_m_curr[:, None] >= offs_n_causal[None, :], qk, float("-inf")
            )
    tl.debug_barrier()

    p = tl.exp2(qk * (softmax_scale * 1.44269504089) - lse_i[:, None])

    # Account for fully masked lines
    if MASKED:
        if fully_masked_lines > 0:
            p = tl.where(offs_m_curr[:, None] < fully_masked_lines, 0, p)

    # Load the gradient of O
    do = load_fn(
        do_ptrs, offs_m_curr, offs_d, PAD_ROWS, HEADS_PADDED, actual_seqlen_q, headdim
    )

    # Compute the gradient of V
    dv += tl.dot(tl.trans(p).to(do.dtype), do)

    # Compute auxiliary gradients
    dp = tl.dot(do, tl.trans(v))

    # Compute the gradient of the scores. Placing the substraction before the matmul apparently speeds up the process
    Di = tl.load(D + offs_m_curr)
    # Converting ds to q.dtype here reduces register pressure and makes it much faster for BLOCK_HEADDIM=128
    ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
    # compute dk = dot(ds.T, q)
    dk += tl.dot(tl.trans(ds), q)

    return dk, dv


@triton.jit
def _compute_column_blocks_dkdv(
    I_start_n,
    Q,
    K,
    V,
    Bias,
    Dropout,
    DO,
    DK,
    DV,
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
    stride_dkn,
    stride_dvn,
    actual_seqlen_q,
    actual_seqlen_k,
    headdim,
    IS_CAUSAL: tl.constexpr,
    BIAS_ON: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    PAD_COLS: tl.constexpr,
    HEADS_PADDED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    # This fuction goes through a column, so it always ends at m = actual_seqlen_q but can start early due to causality
    I_begin_m = (
        max(I_start_n + actual_seqlen_q - actual_seqlen_k, 0) if IS_CAUSAL else 0
    )
    I_begin_m = (I_begin_m // BLOCK_M) * BLOCK_M
    I_end_m = actual_seqlen_q

    fully_masked_lines = (actual_seqlen_q - actual_seqlen_k) if IS_CAUSAL else 0
    # Since we are in a grid dimensionned to fit max_seqlen_q, some blocks may exist early
    if (I_begin_m >= actual_seqlen_q) or (I_start_n >= actual_seqlen_k):
        return

    # Initialize offsets
    offs_n = I_start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize states pointer
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    do_ptrs = DO + (offs_m[:, None] * stride_dom + offs_d[None, :])
    # ...and maybe bias
    if BIAS_ON:
        bias_ptrs = Bias + (offs_m[:, None] * stride_bm + offs_n[None, :])
    else:
        bias_ptrs = None
    # ...and maybe dropout
    if USE_DROPOUT:
        dropout_offs = Dropout + offs_m[:, None] * actual_seqlen_k + offs_n[None, :]
    else:
        dropout_offs = None

    # Initialize dv and dk
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

    # Load K and V, which will stay in SRAM for the row-wise loop
    k = load_fn(
        k_ptrs,
        offs_n,
        offs_d,
        PAD_AXIS_0=PAD_COLS,
        PAD_AXIS_1=HEADS_PADDED,
        LIM_AXIS_0=actual_seqlen_k,
        LIM_AXIS_1=headdim,
    )
    v = load_fn(
        v_ptrs,
        offs_n,
        offs_d,
        PAD_AXIS_0=PAD_COLS,
        PAD_AXIS_1=HEADS_PADDED,
        LIM_AXIS_0=actual_seqlen_k,
        LIM_AXIS_1=headdim,
    )

    # Loop over rows to compute dk and dv
    first_full_row = max(0, I_start_n + BLOCK_N - 1 + actual_seqlen_q - actual_seqlen_k)
    first_full_block = BLOCK_M * (
        (min(first_full_row, actual_seqlen_q) + BLOCK_M - 1) // BLOCK_M
    )
    num_masked_blocks = (first_full_block - I_begin_m) // BLOCK_M if IS_CAUSAL else 0
    I_next_start_m = I_begin_m

    # Partially masked blocks
    if num_masked_blocks > 0:
        for _ in range(0, num_masked_blocks):
            dk, dv = _compute_single_block_dkdv(
                I_next_start_m,
                k,
                v,
                dk,
                dv,
                LSE,
                D,
                offs_m,
                offs_n,
                offs_d,
                q_ptrs,
                bias_ptrs,
                dropout_offs,
                do_ptrs,
                softmax_scale,
                dropout_p,
                dropout_seed,
                stride_qm,
                stride_bm,
                stride_dom,
                actual_seqlen_q,
                actual_seqlen_k,
                fully_masked_lines,
                headdim,
                MASKED=True,
                IS_CAUSAL=IS_CAUSAL,
                BIAS_ON=BIAS_ON,
                USE_DROPOUT=USE_DROPOUT,
                PAD_ROWS=True,  # TODO: fix this
                PAD_COLS=PAD_COLS,
                HEADS_PADDED=HEADS_PADDED,
            )
            I_next_start_m += BLOCK_M

    # Full blocks
    if I_next_start_m < I_end_m:
        for I_start_m in range(I_next_start_m, I_end_m, BLOCK_M):
            dk, dv = _compute_single_block_dkdv(
                I_start_m,
                k,
                v,
                dk,
                dv,
                LSE,
                D,
                offs_m,
                offs_n,
                offs_d,
                q_ptrs,
                bias_ptrs,
                dropout_offs,
                do_ptrs,
                softmax_scale,
                dropout_p,
                dropout_seed,
                stride_qm,
                stride_bm,
                stride_dom,
                actual_seqlen_q,
                actual_seqlen_k,
                fully_masked_lines,
                headdim,
                MASKED=False,
                IS_CAUSAL=IS_CAUSAL,
                BIAS_ON=BIAS_ON,
                USE_DROPOUT=USE_DROPOUT,
                PAD_ROWS=True,  # TODO: fix this
                PAD_COLS=PAD_COLS,
                HEADS_PADDED=HEADS_PADDED,
            )

    # Store dk and dv
    if HEADS_PADDED:
        if PAD_COLS:
            tl.store(
                dk_ptrs,
                dk,
                mask=(offs_n[:, None] < actual_seqlen_k) & (offs_d[None, :] < headdim),
            )
            tl.store(
                dv_ptrs,
                dv,
                mask=(offs_n[:, None] < actual_seqlen_k) & (offs_d[None, :] < headdim),
            )
        else:
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
    else:
        if PAD_COLS:
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < actual_seqlen_k)
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < actual_seqlen_k)
        else:
            tl.store(dk_ptrs, dk)
            tl.store(dv_ptrs, dv)
