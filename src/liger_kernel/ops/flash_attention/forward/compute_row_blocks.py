import triton
import triton.language as tl

from src.liger_kernel.ops.flash_attention.utils import load_fn


@triton.jit
def compute_row_block(
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
    USE_DROPOUT: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_ON: tl.constexpr,
    MASKED: tl.constexpr,
    PADDED_COLS: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    I_start_n = tl.multiple_of(I_start_n, BLOCK_N)

    # Load K (same mechanism as for Q, only check cols instead of rows)
    offset_k_ptrs = k_ptrs + I_start_n * stride_kn
    k = load_fn(
        offset_k_ptrs,
        I_start_n + offs_n, offs_d,
        PAD_AXIS_0=PADDED_COLS, PAD_AXIS_1=PADDED_HEADS,
        LIM_AXIS_0=actual_seqlen_k, LIM_AXIS_1=headdim,
    )
    if BIAS_ON:
        bias = load_fn(
            bias_ptrs + I_start_n,
            offs_m, I_start_n + offs_n,
            PAD_AXIS_0=True, PAD_AXIS_1=PADDED_COLS,  # check
            LIM_AXIS_0=actual_seqlen_q, LIM_AXIS_1=actual_seqlen_k,
        )

    # Compute QK
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))

    # Apply attention masking and/or account for padding of the keys
    if PADDED_COLS:  # TODO: check impact on speed when conditionned by MASKED (always true?)
        qk += tl.where((I_start_n + offs_n)[None, :] < actual_seqlen_k, 0, float("-inf"))
    # Apply causal mask
    if MASKED and IS_CAUSAL:
        causal_mask = offs_m[:, None] >= (I_start_n + offs_n - actual_seqlen_k + actual_seqlen_q)[None, :]
        qk += tl.where(causal_mask, 0, float("-inf"))

    if BIAS_ON:
        qk += bias * (1.44269504089 / softmax_scale)  # TODO: check if this is optimal

    m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
    P_ij = tl.exp2(qk * softmax_scale - m_ij[:, None])
    l_ij = tl.sum(P_ij, 1)

    # Dropout
    if USE_DROPOUT:
        dropout_offs = dropout_offs + I_start_n
        dropout_mask = (tl.rand(dropout_seed, dropout_offs) > dropout_p)  # TODO: replace this w/ randint for better perfs
        P_ij = tl.where(dropout_mask, P_ij, 0.0)

    # Scale the output accumulator
    acc_o_scale = tl.exp2(m_i - m_ij)
    acc_o = acc_o * acc_o_scale[:, None]

    # Load V (same mechanism as K)
    offset_v_ptrs = v_ptrs + I_start_n * stride_vn
    v = load_fn(
        offset_v_ptrs,
        I_start_n + offs_n, offs_d,
        PAD_AXIS_0=PADDED_COLS, PAD_AXIS_1=PADDED_HEADS,
        LIM_AXIS_0=actual_seqlen_k, LIM_AXIS_1=headdim,
    )

    # Update the output accumulator
    P_ij = P_ij.to(v.dtype)
    acc_o += tl.dot(P_ij, v)

    # Update the statistics
    m_i = m_ij
    l_i_new = tl.exp2(lse_i - m_ij) + l_ij
    lse_i = m_ij + tl.log2(l_i_new)

    return m_i, lse_i, acc_o
