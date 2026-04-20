# Triton kernels for fused MoE expert computation.
#
# Routing metadata kernels (Kernels 1-3) are adapted from:
#   SonicMoE (https://github.com/linkedin/sonic-moe)
#   Copyright 2025 Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# Grouped GEMM kernels and backward kernels are new Triton implementations
# inspired by the SonicMoE paper (arXiv:2512.14080), ported to portable Triton
# (no Hopper-specific WGMMA/TMA) for general GPU support.

import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Routing metadata overview
#
# Three kernels produce permutation arrays for grouped GEMM:
#
#   K1 — Histogram: count (token,k) assignments per expert per tile.
#   K2 — Prefix sums: convert tile counts to exclusive prefix sums;
#         compute expert_start_idx (token offsets) and tile offsets.
#   K3 — Scatter: sort by expert, assign globally sorted positions,
#         write x_gather_idx / s_scatter_idx / s_reverse_scatter_idx
#         and tile metadata (tile_row_start, tile_expert).
#
# GEMM kernels consume:
#   x_gather_idx          (TK,)   sorted_pos → original token index
#   s_scatter_idx         (TK,)   sorted_pos → flat (t,k) index
#   s_reverse_scatter_idx (TK,)   flat (t,k) → sorted_pos
#   expert_start_idx      (E+1,)  exclusive cumsum of tokens per expert
#   tile_row_start        (M,)    absolute row_start in sorted space per M-tile
#   tile_expert           (M,)    expert index per M-tile
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tiled histogram of expert token counts
# Adapted from sonic-moe _compute_col_partial_sum_kernel
# ---------------------------------------------------------------------------


@triton.jit
def _moe_router_histogram_kernel(
    topk_indices_ptr,  # (T, K) int32
    partial_sum_ptr,  # (E, n_tiles) int32 — output; partial_sum[e, tile] = count
    T,
    E: tl.constexpr,
    n_tiles,
    TOKENS_PER_TILE: tl.constexpr,
    K_POW2: tl.constexpr,
    K: tl.constexpr,
    E_POW2: tl.constexpr,
):
    """Count how many of this tile's (token, k) assignments route to each expert.

    Grid: (n_tiles,).  Each CTA owns one contiguous slice of TOKENS_PER_TILE
    tokens and atomically increments partial_sum[expert_id, tile_id] for every
    (token, k) pair it sees.

    partial_sum is stored row-major with shape (E, n_tiles) so that K2 can
    read each expert's column (partial_sum[e, :]) with a stride-1 access.

    Ascend note: avoid 2-D tensor + tl.reshape — BiShengHIR fails on
    memref.collapse_shape.  Use 1-D tl.arange + elementwise indexing (no reshape).
    """
    tile_id = tl.program_id(0)

    # Zero this tile's column before counting — partial_sum is not pre-cleared.
    e_offs = tl.arange(0, E_POW2)
    tl.store(
        partial_sum_ptr + e_offs * n_tiles + tile_id,
        tl.zeros([E_POW2], tl.int32),
        mask=e_offs < E,
    )

    FLAT: tl.constexpr = TOKENS_PER_TILE * K_POW2
    offs = tl.arange(0, FLAT)
    ti = offs // K_POW2
    ki = offs % K_POW2
    tok = tile_id * TOKENS_PER_TILE + ti
    m = (tok < T) & (ki < K)
    sk = tl.minimum(ki, K - 1)
    eid = tl.load(topk_indices_ptr + tok * K + sk, mask=m, other=0)
    tl.atomic_add(partial_sum_ptr + eid * n_tiles + tile_id, 1, mask=m)


# ---------------------------------------------------------------------------
# Per-expert tile prefix sums + global token/tile offsets
# Adapted from sonic-moe _bitmatrix_metadata_compute_stage1
# ---------------------------------------------------------------------------


@triton.jit
def _moe_router_prefix_sum_kernel(
    expert_freq_ptr,  # (E,) int32 — total tokens assigned to each expert
    expert_freq_offs_ptr,  # (E+1,) int32 — output: exclusive cumsum of expert_frequency
    expert_tile_offset_ptr,  # (E+1,) int32 — output: exclusive cumsum of ceil(freq/BLOCK_M_TOKEN)
    E: tl.constexpr,
    partial_sum_ptr,  # (E, n_tiles) int32 — in-place: raw tile counts → tile prefix sums
    n_tiles,
    TK,  # T * K, written as sentinel into expert_freq_offs[E]
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M_TOKEN: tl.constexpr,
):
    """Convert histogram counts into prefix sums; compute token and tile offsets.

    Grid: (E+2,).  Three disjoint roles, all running concurrently:

    PIDs 0..E-1  — Per-expert tile prefix scan
        Each CTA converts its expert's row of partial_sum from raw tile
        counts into exclusive prefix sums across tiles.  After K3 reads
        partial_sum[e, tile_id], it knows how many of expert e's tokens
        appeared in earlier tiles, which it adds to within_expert_rank to
        get the global sorted position.

    PID E  — Global token and M-tile offset computation
        Sequentially scans all expert frequencies in blocks of BLOCK_N to
        build two exclusive cumsums in a single pass:
          expert_start_idx[e]    = sum of expert_frequency[0..e-1]
          expert_tile_offset[e]  = sum of ceil(freq[0..e-1] / BLOCK_M_TOKEN)
        Also writes the sentinel expert_tile_offset[E] = total M-tiles.

    PID E+1  — Token sentinel
        Writes expert_start_idx[E] = TK.
    """
    pid = tl.program_id(0)
    if pid < E:
        # Per-expert tile prefix scan: transform partial_sum[pid, :] from
        # raw counts to exclusive prefix sums for conflict-free output positions.
        expert_partial_sum_ptr = partial_sum_ptr + pid * n_tiles
        curr_sum = 0
        for start in range(0, n_tiles, BLOCK_M):
            offs = start + tl.arange(0, BLOCK_M)
            tile_counts = tl.load(expert_partial_sum_ptr + offs, mask=offs < n_tiles, other=0)
            excl_cumsum = tl.cumsum(tile_counts, 0) - tile_counts + curr_sum
            curr_sum += tl.sum(tile_counts, 0)
            tl.store(expert_partial_sum_ptr + offs, excl_cumsum, mask=offs < n_tiles)
    elif pid == E:
        # Global token and M-tile offsets (single sequential CTA).
        # Both expert_start_idx and expert_tile_offset are exclusive prefix sums
        # accumulated together to avoid a second pass.
        curr_freq_sum = 0
        curr_tile_sum = 0
        for start in tl.static_range(0, E, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            expert_freq = tl.load(expert_freq_ptr + offs, mask=offs < E, other=0)

            excl_freq = tl.cumsum(expert_freq, 0) - expert_freq + curr_freq_sum
            curr_freq_sum += tl.sum(expert_freq, 0)
            tl.store(expert_freq_offs_ptr + offs, excl_freq, mask=offs < E)

            # Number of BLOCK_M_TOKEN-sized M-tiles needed for each expert.
            expert_m_tiles = (expert_freq + BLOCK_M_TOKEN - 1) // BLOCK_M_TOKEN
            excl_tile = tl.cumsum(expert_m_tiles, 0) - expert_m_tiles + curr_tile_sum
            curr_tile_sum += tl.sum(expert_m_tiles, 0)
            tl.store(expert_tile_offset_ptr + offs, excl_tile, mask=offs < E)

        # Write total M-tile count as the sentinel.
        tl.store(expert_tile_offset_ptr + E, curr_tile_sum)
    elif pid == E + 1:
        # Token sentinel: expert_start_idx[E] = TK.
        tl.store(expert_freq_offs_ptr + E, TK)


# ---------------------------------------------------------------------------
# Sort assignments by expert, compute output positions, emit tile metadata
# Adapted from sonic-moe _bitmatrix_metadata_compute_stage2
# ---------------------------------------------------------------------------


@triton.jit
def _moe_router_scatter_kernel(
    s_scatter_idx_ptr,  # (TK,) int32 — output: sorted_pos → flat (t,k) index
    s_reverse_scatter_idx_ptr,  # (TK,) int32 — output: flat (t,k) → sorted_pos
    x_gather_idx_ptr,  # (TK,) int32 — output: sorted_pos → token index t
    tile_row_start_ptr,  # (num_m_tiles,) int32 — output: absolute row_start per M-tile
    tile_expert_ptr,  # (num_m_tiles,) int32 — output: expert index per M-tile
    topk_indices_ptr,  # (T, K) int32
    T,
    TK,
    partial_sum_ptr,  # (E, n_tiles) int32 — tile prefix sums from K2 (read-only here)
    n_tiles,
    expert_offs_ptr,  # (E,) int32 — expert_start_idx[0:E] from K2
    expert_tile_offset_ptr,  # (E,) int32 — expert_tile_offset[0:E] from K2
    rank_scratch_ptr,  # (n_tiles, E) int32 — must be zeroed before launch
    E: tl.constexpr,
    K_POW2: tl.constexpr,
    K: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    BLOCK_M_TOKEN: tl.constexpr,
):
    """Assign every (token, k) pair its globally-sorted output position.

    Grid: (n_tiles,).  Ascend cannot compile tl.sort / tl.associative_scan here
    (BiShengHIR issues).  Instead: walk slots in fixed (token, k) order within
    the tile and assign within-tile expert ranks via tl.atomic_add on a scratch
    row — equivalent to a stable sort by (expert_id, original_index).

    rank_scratch[tile_id, e] counts how many earlier slots in this tile already
    claimed an index for expert e; the returned old counter is the within-tile
    rank r used with partial_sum[e, tile_id].

    Ascend: avoid masked tl.atomic_add on invalid/padded slots. Some builds can
    ignore the mask and execute the atomic using a bogus expert id (e.g. -1),
    corrupting scratch and producing out-of-range / duplicate s_rev.

    Instead, we still walk the full padded BLOCK_SIZE in a fixed order (so the
    valid entries match the original stable sort order), but for invalid slots
    we:
      - redirect expert index to 0 (in-bounds), and
      - add a delta of 0 (no-op) in the atomic.
    This keeps the loop fully compile-time while eliminating out-of-bounds
    atomics and mask-dependent behavior.
    """
    BLOCK_SIZE: tl.constexpr = TOKENS_PER_BLOCK * K_POW2
    IS_POW2_K: tl.constexpr = K == K_POW2
    tl.static_assert(BLOCK_SIZE <= 32768)

    pid_m = tl.program_id(0)
    base = pid_m * BLOCK_SIZE

    for i in tl.range(0, BLOCK_SIZE):
        if IS_POW2_K:
            gi = base + i
            valid = gi < TK
            expert_i = tl.load(topk_indices_ptr + gi, mask=valid, other=0).to(tl.int32)
            entry_idx = gi.to(tl.int32)
            token_i = (gi // K).to(tl.int32)
        else:
            token_i_local = i // K_POW2
            k_slot = i % K_POW2
            token_i = (pid_m * TOKENS_PER_BLOCK + token_i_local).to(tl.int32)
            valid = (token_i < T) & (k_slot < K)
            sk = tl.minimum(k_slot, K - 1)
            expert_i = tl.load(topk_indices_ptr + token_i * K + sk, mask=valid, other=0).to(tl.int32)
            entry_idx = (token_i * K + sk).to(tl.int32)

        # For invalid slots: atomic add 0 to an in-bounds expert (0), and mask all writes.
        expert_safe = tl.where(valid, expert_i, 0)
        delta = valid.to(tl.int32)

        r = tl.atomic_add(rank_scratch_ptr + pid_m * E + expert_safe, delta).to(tl.int32)

        within_expert = tl.load(partial_sum_ptr + pid_m + expert_safe * n_tiles, mask=valid, other=0) + r
        expert_start = tl.load(expert_offs_ptr + expert_safe, mask=valid, other=0).to(tl.int32)
        s_rev = expert_start + within_expert

        is_tile_start = (within_expert % BLOCK_M_TOKEN) == 0
        t_within = within_expert // BLOCK_M_TOKEN
        tile_base = tl.load(expert_tile_offset_ptr + expert_safe, mask=valid & is_tile_start, other=0).to(tl.int32)
        flat_tile_idx = tile_base + t_within
        tl.store(tile_row_start_ptr + flat_tile_idx, s_rev, mask=valid & is_tile_start)
        tl.store(tile_expert_ptr + flat_tile_idx, expert_safe, mask=valid & is_tile_start)

        tl.store(s_reverse_scatter_idx_ptr + entry_idx, s_rev, mask=valid)
        tl.store(s_scatter_idx_ptr + s_rev, entry_idx, mask=valid)
        tl.store(x_gather_idx_ptr + s_rev, token_i, mask=valid)


# ---------------------------------------------------------------------------
# Shared autotune config for all GEMM kernels
# ---------------------------------------------------------------------------


def _get_gemm_autotune_configs():
    # Ascend: each autotune candidate is a full BiSheng compile; a large grid makes
    # tests and first-run unbearable.  One stable config is enough for correctness.
    # Keep BLOCK_N/BLOCK_K moderate — large tiles overflow UB in bwd kernels.
    return [
        # NOTE: Triton requires total grid programs < 65536. Keep a robust
        # compile-safe config for Ascend.
        triton.Config({"BLOCK_N": 192, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    ]


# ---------------------------------------------------------------------------
# Forward — fused gather + grouped GEMM + SwiGLU
# 2D grid: (num_m_tiles, ceil(I/BLOCK_N))
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_gemm_autotune_configs(),
    key=["H_dim", "I_dim"],
)
@triton.jit
def _fused_up_proj_swiglu_kernel(
    x_ptr,  # (T, H)
    gate_up_proj_ptr,  # (E, 2*I, H)
    x_gather_idx_ptr,  # (TK,) int32
    expert_start_ptr,  # (E+1,) int32
    tile_row_start_ptr,  # (num_m_tiles,) int32 — row_start per M-tile
    tile_expert_ptr,  # (num_m_tiles,) int32 — expert index per M-tile
    pre_act_ptr,  # (TK, 2*I)  pre-SwiGLU activations [saved for backward]
    post_act_ptr,  # (TK, I)    post-SwiGLU activations
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_x_T,
    stride_x_H: tl.constexpr,
    stride_w_E,
    stride_w_N,
    stride_w_K: tl.constexpr,
    stride_pre_TK,
    stride_pre_N: tl.constexpr,
    stride_post_TK,
    stride_post_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grid: (num_m_tiles, ceil(I/BLOCK_N)).
    pid_m selects M-tile via tile_row_start/tile_expert; pid_n selects N-tile."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = tl.load(tile_row_start_ptr + pid_m)
    expert_idx = tl.load(tile_expert_ptr + pid_m)
    n_start = pid_n * BLOCK_N
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)

    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    row_offs = row_start + m_offs
    row_mask = row_offs < expert_end

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    n_idx = n_start + n_offs
    n_mask = n_idx < I_dim
    token_idx = tl.load(x_gather_idx_ptr + row_offs, mask=row_mask, other=0)
    for k in tl.range(0, H_dim, BLOCK_K):
        k_idx = k + k_offs
        k_mask = k_idx < H_dim

        x_ptrs = x_ptr + token_idx[:, None] * stride_x_T + k_idx[None, :] * stride_x_H
        # Keep bf16 for dot operands → tensor cores. acc stays fp32 for precision.
        x_tile = tl.load(
            x_ptrs,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",  # token rows not reused; free L2 for weights
        )

        w_mask = n_mask[:, None] & k_mask[None, :]
        w_gate_ptrs = (
            gate_up_proj_ptr + expert_idx * stride_w_E + n_idx[:, None] * stride_w_N + k_idx[None, :] * stride_w_K
        )
        w_gate = tl.load(
            w_gate_ptrs,
            mask=w_mask,
            other=0.0,
        )
        acc_gate += tl.dot(x_tile, tl.trans(w_gate))

        w_up_ptrs = w_gate_ptrs + I_dim * stride_w_N
        w_up = tl.load(
            w_up_ptrs,
            mask=w_mask,
            other=0.0,
        )

        acc_up += tl.dot(x_tile, tl.trans(w_up))

    out_mask = row_mask[:, None] & n_mask[None, :]

    pre_gate_ptrs = pre_act_ptr + row_offs[:, None] * stride_pre_TK + n_idx[None, :] * stride_pre_N
    pre_up_ptrs = pre_gate_ptrs + I_dim * stride_pre_N
    tl.store(pre_gate_ptrs, acc_gate.to(pre_act_ptr.dtype.element_ty), mask=out_mask)
    tl.store(pre_up_ptrs, acc_up.to(pre_act_ptr.dtype.element_ty), mask=out_mask)

    sig_gate = tl.sigmoid(acc_gate)
    silu_gate = acc_gate * sig_gate
    a_out = silu_gate * acc_up

    post_ptrs = post_act_ptr + row_offs[:, None] * stride_post_TK + n_idx[None, :] * stride_post_N
    tl.store(post_ptrs, a_out.to(post_act_ptr.dtype.element_ty), mask=out_mask)


# ---------------------------------------------------------------------------
# Forward — grouped GEMM down-projection
# 2D grid: (num_m_tiles, ceil(H/BLOCK_N))
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_gemm_autotune_configs(),
    key=["H_dim", "I_dim"],
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
    """Grid: (num_m_tiles,).
    Each CTA processes one M-tile and iterates over N-tiles in-kernel.

    Ascend: Triton enforces total grid programs < 65536. With large MoE shapes,
    a 2D grid (num_m_tiles, ceil(H/BLOCK_N)) can hit that limit exactly.
    Collapsing to 1D avoids the limit without changing math.
    """
    pid_m = tl.program_id(0)

    row_start = tl.load(tile_row_start_ptr + pid_m)
    expert_idx = tl.load(tile_expert_ptr + pid_m)
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)

    m_offs = tl.arange(0, BLOCK_M)
    k_offs = tl.arange(0, BLOCK_K)

    row_offs = row_start + m_offs
    row_mask = row_offs < expert_end
    # Preload A tile once per K-slice; reused across N-tiles.
    for n_start in tl.range(0, H_dim, BLOCK_N):
        n_offs = tl.arange(0, BLOCK_N)
        n_idx = n_start + n_offs
        n_mask = n_idx < H_dim

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, I_dim, BLOCK_K):
            k_idx = k + k_offs
            k_mask = k_idx < I_dim

            a_ptrs = post_act_ptr + row_offs[:, None] * stride_post_TK + k_idx[None, :] * stride_post_I
            # Keep bf16 for dot operands → tensor cores. acc stays fp32.
            a_tile = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

            w_ptrs = (
                down_proj_ptr
                + expert_idx * stride_w_E
                + n_idx[:, None] * stride_w_H
                + k_idx[None, :] * stride_w_I
            )
            w_tile = tl.load(
                w_ptrs,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            acc += tl.dot(a_tile, tl.trans(w_tile))

        Y_ptrs = Y_ptr + row_offs[:, None] * stride_Y_TK + n_idx[None, :] * stride_Y_H
        tl.store(Y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=row_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Forward — token gather + weighted sum
# Adapted from sonic-moe token_gather_sum_kernel
# ---------------------------------------------------------------------------


def _get_token_gather_autotune_configs():
    return [
        triton.Config({"BLOCK_H": 128, "BLOCK_K": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_H": 256, "BLOCK_K": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_H": 256, "BLOCK_K": 16}, num_warps=4, num_stages=4),
    ]


@triton.autotune(
    configs=_get_token_gather_autotune_configs(),
    key=["H_dim", "K_dim", "w_is_None"],
)
@triton.jit
def _token_gather_weighted_sum_kernel(
    Y_ptr,  # (TK, H)
    w_ptr,  # (TK,) routing weights, or None when w_is_None=True
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
    w_is_None: tl.constexpr,  # True → unweighted gather-sum (used for dx backward)
):
    """One CTA per token. Gathers K expert outputs, reduces with routing weights
    (forward) or without weights (backward dx via _token_broadcast_backward).

    Ascend: avoid uint32 index tensors — BiShengHIR rejects uint32→i64 casts on pointers.
    """
    t = tl.program_id(0)

    for h_tile in tl.static_range(triton.cdiv(H_dim, BLOCK_H)):
        h_idx = h_tile * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = h_idx < H_dim
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        for k_tile in tl.range(triton.cdiv(K_dim, BLOCK_K)):
            k_offs = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K_dim

            flat_idx = t * K_dim + k_offs
            perm_idx = tl.load(s_rev_ptr + flat_idx, mask=k_mask, other=0)

            y_ptrs = Y_ptr + perm_idx[:, None] * stride_Y_TK + h_idx[None, :] * stride_Y_H
            y_vals = tl.load(y_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)

            if w_is_None:
                acc += tl.sum(y_vals, axis=0)
            else:
                w_vals = tl.load(w_ptr + flat_idx, mask=k_mask, other=0.0).to(tl.float32)
                acc += tl.sum(y_vals * w_vals[:, None], axis=0)

        out_ptrs = out_ptr + t * stride_out_T + h_idx * stride_out_H
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=h_mask)


# ---------------------------------------------------------------------------
# Backward — fused down-proj backward + SwiGLU backward
# 2D grid: (num_m_tiles, ceil(I/BLOCK_N))
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_gemm_autotune_configs(),
    key=["H_dim", "I_dim"],
    reset_to_zero=["dS_ptr"],  # autotune runs multiple configs; atomic_add accumulates, so reset between runs
)
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
):
    """Grid: (num_m_tiles, ceil(I/BLOCK_N)).
    Accumulates dA' = dO @ W2^T (dO stays in registers), recomputes y1 from
    pre_act, applies SwiGLU backward, writes d_pre_act, weighted_act, and dS."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = tl.load(tile_row_start_ptr + pid_m)
    expert_idx = tl.load(tile_expert_ptr + pid_m)
    n_start = pid_n * BLOCK_N
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)

    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    row_offs = row_start + m_offs
    row_mask = row_offs < expert_end
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


# ---------------------------------------------------------------------------
# Backward — dW2 = weighted_act^T @ dout_gathered (per expert, weight grad)
# Lambda grid: (E * ceil(I/BLOCK_M), ceil(H/BLOCK_N))
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        # Keep total grid programs < 65536 for large E/H/I.
        # Larger BLOCK_M/BLOCK_K improve arithmetic intensity on long-sequence
        # backward workloads and cut K-loop overhead.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 240, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    ],
    key=["H_dim", "I_dim"],
    reset_to_zero=["dW2_ptr"],
)
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
    """dW2[e, h, i] += sum_t weighted_act[t, i] * dout[token(t), h] for tokens in e.
    Grid: (E * ceil(I/BLOCK_M), ceil(H/BLOCK_N)). Early exit for empty experts."""
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


# ---------------------------------------------------------------------------
# Backward — dx_expanded = d_pre_act @ W1^T (grouped GEMM, no atomics)
# 2D grid: (num_m_tiles, ceil(H/BLOCK_N))
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_gemm_autotune_configs(),
    key=["H_dim", "I_dim"],
)
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
):
    """Grid: (num_m_tiles,).
    dx_expanded[sorted_pos] = d_gate @ W1_gate^T + d_up @ W1_up^T.
    No atomics — rows are unique per CTA in sorted space.

    Ascend: collapse the 2D grid to avoid hitting Triton's <65536 grid limit
    on large (num_m_tiles, ceil(H/BLOCK_N)) products.
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
        h_idx = n_start + n_offs
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


# ---------------------------------------------------------------------------
# Backward — dW1 = Gathered_X^T @ d_pre_act (per expert, weight grad)
# Lambda grid: (E * ceil(H/BLOCK_M), ceil(2I/BLOCK_N))
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        # Keep total grid programs < 65536 for large E/H/I.
        # Match dW2 tiling for better large-shape throughput.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 240, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    ],
    key=["H_dim", "I_dim"],
    reset_to_zero=["dW1_ptr"],
)
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
    """dW1[e, n, h] += sum_t X[token(t), h] * d_pre_act[t, n], where n in [0, 2I).
    Grid: (E * ceil(H/BLOCK_M), ceil(2I/BLOCK_N)). Early exit for empty experts."""
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
