# Triton kernels for fused MoE on Ascend.

import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Tile sizes (fixed — Ascend launch grid must stay below ~32K programs)
# ---------------------------------------------------------------------------

ASCEND_GEMM_BLOCK_N = 192
ASCEND_GEMM_BLOCK_K = 64

ASCEND_MAX_GRID_PROGRAMS = 32768
ASCEND_BWD_BLOCK_N = ASCEND_GEMM_BLOCK_N

ASCEND_TOKEN_GATHER_BLOCK_H = 128
ASCEND_TOKEN_GATHER_BLOCK_K = 8

ASCEND_DW_BLOCK_M = 128
ASCEND_DW_BLOCK_N = 240
ASCEND_DW_BLOCK_K = 32


# ---------------------------------------------------------------------------
# Routing metadata: histogram → exclusive prefix → per-row rank → tile bounds
# A wide masked cumsum does not assign unique ranks.
# ---------------------------------------------------------------------------


@triton.jit
def _moe_router_histogram_kernel(
    topk_ptr,  # (TK,) int32, flat (token, k)
    count_ptr,  # (E,) int32, zeroed by the host
    TK,
    E,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for i in tl.range(pid, TK, nprog):
        expert = tl.load(topk_ptr + i)
        if (expert >= 0) & (expert < E):
            tl.atomic_add(count_ptr + expert, 1)


@triton.jit
def _moe_router_prefix_sum_kernel(
    count_ptr,  # (E,) int32
    start_ptr,  # (E+1,) int32 exclusive prefix of counts
    tile_ptr,  # (E+1,) int32 exclusive prefix of ceil(count / BLOCK_M_TOKEN)
    E,
    BLOCK_M_TOKEN: tl.constexpr,
):
    running = 0
    tiles = 0
    for e in tl.range(0, E):
        count = tl.load(count_ptr + e)
        tl.store(start_ptr + e, running)
        tl.store(tile_ptr + e, tiles)
        running += count
        tiles += (count + BLOCK_M_TOKEN - 1) // BLOCK_M_TOKEN
    tl.store(start_ptr + E, running)
    tl.store(tile_ptr + E, tiles)


@triton.jit
def _moe_router_scatter_kernel(
    topk_ptr,
    cursor_ptr,  # (E,) int32, zeroed; atomic_add returns the previous rank
    start_ptr,  # (E+1,) int32
    s_scatter_ptr,  # sorted_pos → flat (t, k)
    s_reverse_ptr,  # flat (t, k) → sorted_pos
    x_gather_ptr,  # sorted_pos → token index
    TK,
    K,
    E,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for i in tl.range(pid, TK, nprog):
        expert = tl.load(topk_ptr + i)
        if (expert >= 0) & (expert < E):
            rank = tl.atomic_add(cursor_ptr + expert, 1).to(tl.int32)
            slot = (tl.load(start_ptr + expert) + rank).to(tl.int32)
            token = (i // K).to(tl.int32)
            tl.store(s_reverse_ptr + i, slot)
            tl.store(s_scatter_ptr + slot, i.to(tl.int32))
            tl.store(x_gather_ptr + slot, token)


@triton.jit
def _moe_router_tile_kernel(
    start_ptr,  # (E+1,) int32
    tile_ptr,  # (E+1,) int32
    tile_row_ptr,  # (num_m_tiles,) int32
    tile_expert_ptr,  # (num_m_tiles,) int32
    BLOCK_M_TOKEN: tl.constexpr,
):
    expert = tl.program_id(0)
    row0 = tl.load(start_ptr + expert)
    tile0 = tl.load(tile_ptr + expert)
    n_tiles = tl.load(tile_ptr + expert + 1) - tile0
    for tile in tl.range(0, n_tiles):
        tl.store(tile_row_ptr + tile0 + tile, row0 + tile * BLOCK_M_TOKEN)
        tl.store(tile_expert_ptr + tile0 + tile, expert)


# ---------------------------------------------------------------------------
# Forward GEMM: up-proj + SwiGLU, down-proj
# ---------------------------------------------------------------------------


@triton.jit
def _fused_up_proj_swiglu_kernel(
    x_ptr,  # (T, H)
    gate_up_proj_ptr,  # (E, 2*I, H)
    x_gather_idx_ptr,  # (TK,) int32
    expert_start_ptr,  # (E+1,) int32
    tile_row_start_ptr,  # (num_m_tiles,) int32 — row_start per M-tile
    tile_expert_ptr,  # (num_m_tiles,) int32 — expert index per M-tile
    pre_act_ptr,  # (TK, 2*I)  pre-SwiGLU activations [saved for backward]
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_x_T,
    stride_x_H: tl.constexpr,
    stride_w_E,
    stride_w_N,
    stride_w_K: tl.constexpr,
    stride_pre_TK,
    stride_pre_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grid: (num_m_tiles,). One CTA per M-tile; N-tiles iterated in-kernel.

    Cube-only: writes pre-SwiGLU [gate, up]. SwiGLU is a separate vector kernel
    because triton-ascend 3.2.2 ConvertLinalgRToBinary cannot lower mix
    cube+vector (gather + dual tl.dot + silu + GM stores) to static UB shapes.
    """
    pid_m = tl.program_id(0)

    row_start = tl.load(tile_row_start_ptr + pid_m)
    expert_idx = tl.load(tile_expert_ptr + pid_m)
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)

    m_offs = tl.arange(0, BLOCK_M)
    k_offs = tl.arange(0, BLOCK_K)

    row_offs = row_start + m_offs
    row_mask = row_offs < expert_end
    token_idx = tl.load(x_gather_idx_ptr + row_offs, mask=row_mask, other=0)

    for n_start in tl.range(0, I_dim, BLOCK_N):
        n_offs = tl.arange(0, BLOCK_N)
        n_idx = n_start + n_offs
        n_mask = n_idx < I_dim

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in tl.range(0, H_dim, BLOCK_K):
            k_idx = k + k_offs
            k_mask = k_idx < H_dim

            x_ptrs = x_ptr + token_idx[:, None] * stride_x_T + k_idx[None, :] * stride_x_H
            x_tile = tl.load(
                x_ptrs,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            w_mask = n_mask[:, None] & k_mask[None, :]
            w_gate_ptrs = (
                gate_up_proj_ptr + expert_idx * stride_w_E + n_idx[:, None] * stride_w_N + k_idx[None, :] * stride_w_K
            )
            w_gate = tl.load(w_gate_ptrs, mask=w_mask, other=0.0)
            acc_gate += tl.dot(x_tile, tl.trans(w_gate))

            w_up_ptrs = w_gate_ptrs + I_dim * stride_w_N
            w_up = tl.load(w_up_ptrs, mask=w_mask, other=0.0)
            acc_up += tl.dot(x_tile, tl.trans(w_up))

        out_mask = row_mask[:, None] & n_mask[None, :]

        pre_gate_ptrs = pre_act_ptr + row_offs[:, None] * stride_pre_TK + n_idx[None, :] * stride_pre_N
        pre_up_ptrs = pre_gate_ptrs + I_dim * stride_pre_N
        tl.store(pre_gate_ptrs, acc_gate.to(pre_act_ptr.dtype.element_ty), mask=out_mask)
        tl.store(pre_up_ptrs, acc_up.to(pre_act_ptr.dtype.element_ty), mask=out_mask)


@triton.jit
def _swiglu_from_pre_act_kernel(
    pre_act_ptr,  # (TK, 2*I)
    post_act_ptr,  # (TK, I)
    TK,
    I_dim: tl.constexpr,
    stride_pre_TK,
    stride_pre_N: tl.constexpr,
    stride_post_TK,
    stride_post_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Vector epilogue: post = silu(gate) * up. Grid-stride over TK rows."""
    pid = tl.program_id(0)
    n_prog = tl.num_programs(0)
    n_offs = tl.arange(0, BLOCK_N)

    for tk in tl.range(pid, TK, n_prog):
        for n_start in tl.range(0, I_dim, BLOCK_N):
            n_idx = n_start + n_offs
            n_mask = n_idx < I_dim
            gate = tl.load(
                pre_act_ptr + tk * stride_pre_TK + n_idx * stride_pre_N,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                pre_act_ptr + tk * stride_pre_TK + (n_idx + I_dim) * stride_pre_N,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            out = gate * tl.sigmoid(gate) * up
            tl.store(
                post_act_ptr + tk * stride_post_TK + n_idx * stride_post_N,
                out.to(post_act_ptr.dtype.element_ty),
                mask=n_mask,
            )


@triton.jit
def _fused_down_proj_kernel(
    post_act_ptr,  # (TK, I)
    down_proj_ptr,  # (E, H, I)
    expert_start_ptr,  # (E+1,) int32
    tile_row_start_ptr,  # (num_m_tiles,) int32
    tile_expert_ptr,  # (num_m_tiles,) int32
    Y_ptr,  # (TK, H)
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_post_TK,
    stride_post_I: tl.constexpr,
    stride_w_E,
    stride_w_H,
    stride_w_I: tl.constexpr,
    stride_Y_TK,
    stride_Y_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grid: (num_m_tiles,). One CTA per M-tile; N-tiles iterated in-kernel.

    Ascend: 1D grid avoids exceeding Triton's <65536 program limit at large T.
    """
    pid_m = tl.program_id(0)

    row_start = tl.load(tile_row_start_ptr + pid_m)
    expert_idx = tl.load(tile_expert_ptr + pid_m)
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)

    m_offs = tl.arange(0, BLOCK_M)
    k_offs = tl.arange(0, BLOCK_K)

    row_offs = row_start + m_offs
    row_mask = row_offs < expert_end

    for n_start in tl.range(0, H_dim, BLOCK_N):
        n_offs = tl.arange(0, BLOCK_N)
        n_idx = n_start + n_offs
        n_mask = n_idx < H_dim

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, I_dim, BLOCK_K):
            k_idx = k + k_offs
            k_mask = k_idx < I_dim

            a_ptrs = post_act_ptr + row_offs[:, None] * stride_post_TK + k_idx[None, :] * stride_post_I
            a_tile = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

            w_ptrs = down_proj_ptr + expert_idx * stride_w_E + n_idx[:, None] * stride_w_H + k_idx[None, :] * stride_w_I
            w_tile = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

            acc += tl.dot(a_tile, tl.trans(w_tile))

        Y_ptrs = Y_ptr + row_offs[:, None] * stride_Y_TK + n_idx[None, :] * stride_Y_H
        tl.store(Y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=row_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Token aggregation
# ---------------------------------------------------------------------------


@triton.jit
def _token_gather_weighted_sum_kernel(
    Y_ptr,  # (TK, H)
    w_ptr,  # (TK,) routing weights; unused when w_is_None=True
    s_rev_ptr,  # (TK,) int32 s_reverse_scatter_idx: flat(t,k) → sorted position
    out_ptr,  # (T, H)
    H_dim: tl.constexpr,
    K_dim: tl.constexpr,
    stride_Y_TK,
    stride_Y_H: tl.constexpr,
    stride_out_T,
    stride_out_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    w_is_None: tl.constexpr,  # True → unweighted gather-sum (backward dx)
):
    """One CTA per token. Gathers K expert outputs and reduces along K.

    Ascend: avoid uint32 index tensors; use a dedicated path when K=1.
    """
    t = tl.program_id(0)
    IS_K1: tl.constexpr = K_dim == 1
    TILE_H: tl.constexpr = BLOCK_H + BLOCK_H // 2

    for h_tile in tl.static_range(triton.cdiv(H_dim, TILE_H)):
        h_idx = h_tile * TILE_H + tl.arange(0, TILE_H)
        h_mask = h_idx < H_dim

        if IS_K1:
            perm_idx = tl.load(s_rev_ptr + t)
            y_ptrs = Y_ptr + perm_idx * stride_Y_TK + h_idx * stride_Y_H
            y_vals = tl.load(y_ptrs, mask=h_mask, other=0.0).to(tl.float32)
            if w_is_None:
                acc = y_vals
            else:
                w_val = tl.load(w_ptr + t).to(tl.float32)
                acc = y_vals * w_val
        else:
            acc = tl.zeros([TILE_H], dtype=tl.float32)
            for k_tile in tl.range(triton.cdiv(K_dim, BLOCK_K)):
                k_offs = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
                k_mask = k_offs < K_dim

                flat_idx = t * K_dim + k_offs
                perm_idx = tl.load(s_rev_ptr + flat_idx, mask=k_mask, other=0)

                y_ptrs = Y_ptr + perm_idx[:, None] * stride_Y_TK + h_idx[None, :] * stride_Y_H
                y_vals = tl.load(y_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)

                if w_is_None:
                    w_vals = k_mask.to(tl.float32)
                else:
                    w_vals = tl.load(w_ptr + flat_idx, mask=k_mask, other=0.0).to(tl.float32)
                acc += tl.sum(y_vals * w_vals[:, None], axis=0)

        out_ptrs = out_ptr + t * stride_out_T + h_idx * stride_out_H
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=h_mask)


# ---------------------------------------------------------------------------
# Backward: down-proj, weight grads, input grads
# ---------------------------------------------------------------------------


@triton.jit
def _moe_bwd_down_proj_kernel(
    dO_ptr,  # (T, H)   — ∂L/∂O, upstream gradient
    x_gather_idx_ptr,  # (TK,)    — σ_x: sorted_pos → original token index
    s_scatter_idx_ptr,  # (TK,)    — σ_s: sorted_pos → flat (t,k) index
    topk_weights_ptr,  # (TK,)    — s_k: routing weights in flat (t,k) order
    down_proj_ptr,  # (E, H, I) — W2
    pre_act_ptr,  # (TK, 2I) — z = [gate, up] saved from forward
    expert_start_ptr,  # (E+1,)   int32
    tile_row_start_ptr,  # (num_m_tiles,) int32
    tile_expert_ptr,  # (num_m_tiles,) int32
    d_pre_act_ptr,  # (TK, 2I) — output: ∂L/∂z = [dgate, dup]
    weighted_act_ptr,  # (TK, I)  — output: s_k * y1 (for dW2 kernel)
    dS_ptr,  # (TK,)    — output: ∂L/∂s_k, indexed by flat (t,k)
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_dO_T,
    stride_dO_H: tl.constexpr,
    stride_w_E,
    stride_w_H,
    stride_w_I: tl.constexpr,
    stride_pre_TK,
    stride_pre_N: tl.constexpr,
    stride_d_pre_TK,
    stride_d_pre_N: tl.constexpr,
    stride_wact_TK,
    stride_wact_I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    pid_base,
):
    """Grid: (num_row_programs, ceil(I/BLOCK_N)). One live row per program.

    Recomputes dA' = dO @ W2^T, applies SwiGLU backward, and writes d_pre_act,
    weighted_act, and dS. Caller chunks the M dimension when the grid overflows.
    """
    pid_m = tl.program_id(0) + pid_base
    pid_n = tl.program_id(1)

    tile_id = pid_m // BLOCK_M
    row_off = pid_m % BLOCK_M
    row_start = tl.load(tile_row_start_ptr + tile_id) + row_off
    expert_idx = tl.load(tile_expert_ptr + tile_id)
    n_start = pid_n * BLOCK_N
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)

    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    row_offs = row_start + m_offs
    row_mask = (m_offs == 0) & (row_offs < expert_end)
    n_idx = n_start + n_offs
    n_mask = n_idx < I_dim
    out_mask = row_mask[:, None] & n_mask[None, :]

    # Hoist per-row routing metadata (constant across H K-loop).
    token_idx = tl.load(x_gather_idx_ptr + row_offs, mask=row_mask, other=0)
    flat_tk_idx = tl.load(s_scatter_idx_ptr + row_offs, mask=row_mask, other=0)
    weights = tl.load(topk_weights_ptr + flat_tk_idx, mask=row_mask, other=0.0).to(tl.float32)

    # K-loop: accumulate dA' = dO @ W2^T (unscaled; scale once after loop).
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, H_dim, BLOCK_K):
        k_idx = k + k_offs
        k_mask = k_idx < H_dim

        dO_ptrs = dO_ptr + token_idx[:, None] * stride_dO_T + k_idx[None, :] * stride_dO_H
        dO_tile = tl.load(dO_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        w_ptrs = down_proj_ptr + expert_idx * stride_w_E + k_idx[:, None] * stride_w_H + n_idx[None, :] * stride_w_I
        w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc += tl.dot(dO_tile, w_tile)

    # Epilogue: recompute y1 = silu(gate) * up from saved pre_act.
    # These loads need fp32 for the sigmoid/silu computation.
    gate_ptrs = pre_act_ptr + row_offs[:, None] * stride_pre_TK + n_idx[None, :] * stride_pre_N
    up_ptrs = gate_ptrs + I_dim * stride_pre_N
    gate = tl.load(gate_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    sig_gate = tl.sigmoid(gate)
    silu_gate = gate * sig_gate
    y1 = silu_gate * up  # (BLOCK_M, BLOCK_N)

    # Write weighted_act = s_k * y1 for dW2.
    wact_ptrs = weighted_act_ptr + row_offs[:, None] * stride_wact_TK + n_idx[None, :] * stride_wact_I
    tl.store(wact_ptrs, (weights[:, None] * y1).to(weighted_act_ptr.dtype.element_ty), mask=out_mask)

    # dS: ∂L/∂s_k = sum_I((dO @ W2^T) * y1) — accumulate across all N-tiles.
    # IMPORTANT: use atomic_add, not store — the grid has ceil(I/BLOCK_N) N-tiles per
    # M-tile, each contributing a partial sum over its I-chunk.  tl.store would
    # overwrite previous tiles, leaving only the last chunk's contribution.
    dS_partial = tl.sum(acc * y1, axis=1)
    tl.atomic_add(dS_ptr + flat_tk_idx, dS_partial, mask=row_mask)

    # Scale once: dA' = s_k * (dO @ W2^T)
    acc = acc * weights[:, None]

    # SwiGLU backward: dgate = d_silu(gate) * up * dA', dup = silu(gate) * dA'.
    dgate = acc * (silu_gate * (1.0 - sig_gate) + sig_gate) * up
    dup = acc * silu_gate
    dgate_ptrs = d_pre_act_ptr + row_offs[:, None] * stride_d_pre_TK + n_idx[None, :] * stride_d_pre_N
    dup_ptrs = dgate_ptrs + I_dim * stride_d_pre_N
    tl.store(dgate_ptrs, dgate.to(d_pre_act_ptr.dtype.element_ty), mask=out_mask)
    tl.store(dup_ptrs, dup.to(d_pre_act_ptr.dtype.element_ty), mask=out_mask)


@triton.jit
def _moe_bwd_dW2_kernel(
    weighted_act_ptr,  # (TK, I) — s_k * y1 from backward down-proj kernel
    dout_ptr,  # (T, H)  — upstream gradient (gathered by x_gather_idx)
    x_gather_idx_ptr,  # (TK,)   — sorted_pos → original token index
    expert_start_ptr,  # (E+1,)  int32
    dW2_ptr,  # (E, H, I) — output
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_wact_TK,
    stride_wact_I: tl.constexpr,
    stride_dout_T,
    stride_dout_H: tl.constexpr,
    stride_dW2_E,
    stride_dW2_H,
    stride_dW2_I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """dW2[e, h, i] += sum_t weighted_act[t, i] * dout[token(t), h]. Early exit for empty experts."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    N_M_TILES: tl.constexpr = (I_dim + BLOCK_M - 1) // BLOCK_M
    expert_idx = pid0 // N_M_TILES
    m_tile = pid0 % N_M_TILES

    expert_start = tl.load(expert_start_ptr + expert_idx)
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)
    M_e = expert_end - expert_start
    if M_e == 0:
        return

    m_start = m_tile * BLOCK_M
    n_start = pid1 * BLOCK_N

    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    i_idx = m_start + m_offs
    h_idx = n_start + n_offs
    i_mask = i_idx < I_dim
    h_mask = h_idx < H_dim

    # Accumulate in (H_blk, I_blk) so the first axis matches h_idx.
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k in tl.range(0, M_e, BLOCK_K):
        k_idx = k + k_offs
        k_mask = k_idx < M_e
        row_offs = expert_start + k_idx

        wact_ptrs = weighted_act_ptr + row_offs[:, None] * stride_wact_TK + i_idx[None, :] * stride_wact_I
        wact_tile = tl.load(wact_ptrs, mask=k_mask[:, None] & i_mask[None, :], other=0.0)

        token_idx = tl.load(x_gather_idx_ptr + row_offs, mask=k_mask, other=0)
        dout_ptrs = dout_ptr + token_idx[:, None] * stride_dout_T + h_idx[None, :] * stride_dout_H
        dout_tile = tl.load(dout_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0)

        # dW2[h, i] = sum_t dout_g[t, h] * wa[t, i]  <=>  (H,T) @ (T,I)
        # Keep weighted_act as (T_blk, I_blk) so only one transpose is needed.
        acc += tl.dot(tl.trans(dout_tile), wact_tile)

    # acc layout is (H_blk, I_blk) — match dW2[e, h, i] with h on the first broadcast axis.
    dW2_ptrs = dW2_ptr + expert_idx * stride_dW2_E + h_idx[:, None] * stride_dW2_H + i_idx[None, :] * stride_dW2_I
    tl.store(dW2_ptrs, acc.to(dW2_ptr.dtype.element_ty), mask=h_mask[:, None] & i_mask[None, :])


@triton.jit
def _moe_bwd_dX_expanded_kernel(
    d_pre_act_ptr,  # (TK, 2*I)
    gate_up_proj_ptr,  # (E, 2*I, H) — W1
    expert_start_ptr,  # (E+1,) int32
    tile_row_start_ptr,  # (num_m_tiles,) int32
    tile_expert_ptr,  # (num_m_tiles,) int32
    dx_expanded_ptr,  # (TK, H) — output: clean write, indexed by sorted_pos
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_d_pre_TK,
    stride_d_pre_N: tl.constexpr,
    stride_w_E,
    stride_w_N,
    stride_w_K: tl.constexpr,
    stride_dxe_TK,
    stride_dxe_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    pid_base,
):
    """Grid: (num_row_programs, ceil(H/BLOCK_N)). One live row per program.

    dx_expanded[sorted_pos] = d_gate @ W1_gate^T + d_up @ W1_up^T.
    """
    pid_m = tl.program_id(0) + pid_base
    pid_n = tl.program_id(1)

    tile_id = pid_m // BLOCK_M
    row_off = pid_m % BLOCK_M
    row_start = tl.load(tile_row_start_ptr + tile_id) + row_off
    expert_idx = tl.load(tile_expert_ptr + tile_id)
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)

    m_offs = tl.arange(0, BLOCK_M)
    k_offs = tl.arange(0, BLOCK_K)
    n_offs = tl.arange(0, BLOCK_N)

    row_offs = row_start + m_offs
    row_mask = (m_offs == 0) & (row_offs < expert_end)

    h_idx = pid_n * BLOCK_N + n_offs
    h_mask = h_idx < H_dim

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(0, I_dim, BLOCK_K):
        k_idx = k + k_offs
        k_mask = k_idx < I_dim

        d_gate_ptrs = d_pre_act_ptr + row_offs[:, None] * stride_d_pre_TK + k_idx[None, :] * stride_d_pre_N
        d_gate = tl.load(d_gate_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        w_gate_ptrs = (
            gate_up_proj_ptr + expert_idx * stride_w_E + k_idx[:, None] * stride_w_N + h_idx[None, :] * stride_w_K
        )
        w_gate = tl.load(w_gate_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0)
        acc += tl.dot(d_gate, w_gate)

        d_up_ptrs = d_pre_act_ptr + row_offs[:, None] * stride_d_pre_TK + (I_dim + k_idx)[None, :] * stride_d_pre_N
        d_up = tl.load(d_up_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        w_up_ptrs = (
            gate_up_proj_ptr
            + expert_idx * stride_w_E
            + (I_dim + k_idx)[:, None] * stride_w_N
            + h_idx[None, :] * stride_w_K
        )
        w_up = tl.load(w_up_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0)
        acc += tl.dot(d_up, w_up)

    dxe_ptrs = dx_expanded_ptr + row_offs[:, None] * stride_dxe_TK + h_idx[None, :] * stride_dxe_H
    tl.store(dxe_ptrs, acc.to(dx_expanded_ptr.dtype.element_ty), mask=row_mask[:, None] & h_mask[None, :])


@triton.jit
def _moe_bwd_dW1_kernel(
    x_ptr,  # (T, H)
    d_pre_act_ptr,  # (TK, 2*I)
    x_gather_idx_ptr,  # (TK,) int32
    expert_start_ptr,  # (E+1,) int32
    dW1_ptr,  # (E, 2*I, H) — output
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_x_T,
    stride_x_H: tl.constexpr,
    stride_d_pre_TK,
    stride_d_pre_N: tl.constexpr,
    stride_dW1_E,
    stride_dW1_N,
    stride_dW1_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """dW1[e, n, h] += sum_t X[token(t), h] * d_pre_act[t, n], n in [0, 2I). Early exit for empty experts."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    N_M_TILES: tl.constexpr = (H_dim + BLOCK_M - 1) // BLOCK_M
    expert_idx = pid0 // N_M_TILES
    m_tile = pid0 % N_M_TILES

    expert_start = tl.load(expert_start_ptr + expert_idx)
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)
    M_e = expert_end - expert_start
    if M_e == 0:
        return

    m_start = m_tile * BLOCK_M
    n_start = pid1 * BLOCK_N

    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    h_idx = m_start + m_offs
    n_idx = n_start + n_offs
    h_mask = h_idx < H_dim
    n_mask = n_idx < 2 * I_dim

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k in tl.range(0, M_e, BLOCK_K):
        k_idx = k + k_offs
        k_mask = k_idx < M_e
        row_offs = expert_start + k_idx

        token_idx = tl.load(x_gather_idx_ptr + row_offs, mask=k_mask, other=0)
        x_ptrs = x_ptr + token_idx[:, None] * stride_x_T + h_idx[None, :] * stride_x_H
        x_tile = tl.load(x_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0)

        d_pre_ptrs = d_pre_act_ptr + row_offs[:, None] * stride_d_pre_TK + n_idx[None, :] * stride_d_pre_N
        d_pre_tile = tl.load(d_pre_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(tl.trans(d_pre_tile), x_tile)

    dW1_ptrs = dW1_ptr + expert_idx * stride_dW1_E + n_idx[:, None] * stride_dW1_N + h_idx[None, :] * stride_dW1_H
    tl.store(dW1_ptrs, acc.to(dW1_ptr.dtype.element_ty), mask=n_mask[:, None] & h_mask[None, :])
