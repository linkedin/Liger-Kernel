"""
Fused MoE expert computation via Triton grouped GEMM.

Forward: routing metadata (3 kernels, sync-free) → fused gather+GEMM+SwiGLU →
down-proj → token aggregation.
Backward: memory-efficient — recomputes dA' = dO@W2^T instead of caching Y (TK×H)
and accumulates router-score gradients in fp32.

Runtime properties:
- No host↔device sync anywhere on the hot path (the m-tile count is upper-bounded
  host-side and GEMM CTAs early-exit past the device-side actual count), so the op
  is CUDA-graph-capturable.
- Tile sizes adapt to tokens-per-expert; grouped GEMMs autotune per
  (H, I, BLOCK_M, TMA) and the dW kernels per (H, I, tokens-per-expert bucket).
- Expert-weight loads use TMA descriptors on Hopper+ when shapes are 16B-aligned.
- Blackwell datacenter parts (sm100/sm103) tune over an extended config space
  (wide-N/deep-stage tiles enabled by TMEM accumulators); other archs are unchanged.
- Inference (no input requires grad) skips saving/storing pre-activations.

Env flags:
- LIGER_FUSED_MOE_AUTOTUNE=0: pin one config per kernel (skip tuning; see #1246).
- LIGER_FUSED_MOE_MEMORY_EFFICIENT=1: backward writes SwiGLU gradients in place
  over the saved pre-activations (saves TK*2I*itemsize bytes) and drops the
  (TK, I) weighted_act buffer (dW2 recomputes s_k*silu(gate)*up on the fly;
  saves TK*I*itemsize bytes). Combined ≈ −1.2 GB peak at T=32768/I=768/bf16 for a
  ~10-15% slower backward. In this mode a second backward over the same graph
  (retain_graph) raises a version-counter error — by design.
"""

import torch
import triton

import liger_kernel.ops.fused_moe_kernels as _kernels_mod

from liger_kernel.ops.fused_moe_kernels import _fused_down_proj_kernel
from liger_kernel.ops.fused_moe_kernels import _fused_up_proj_swiglu_kernel
from liger_kernel.ops.fused_moe_kernels import _moe_bwd_down_proj_kernel
from liger_kernel.ops.fused_moe_kernels import _moe_bwd_dW1_kernel
from liger_kernel.ops.fused_moe_kernels import _moe_bwd_dW2_kernel
from liger_kernel.ops.fused_moe_kernels import _moe_bwd_dX_expanded_kernel
from liger_kernel.ops.fused_moe_kernels import _moe_router_histogram_kernel
from liger_kernel.ops.fused_moe_kernels import _moe_router_prefix_sum_kernel
from liger_kernel.ops.fused_moe_kernels import _moe_router_scatter_kernel
from liger_kernel.ops.fused_moe_kernels import _token_gather_weighted_sum_kernel
from liger_kernel.ops.utils import ensure_contiguous

# LIGER_FUSED_MOE_MEMORY_EFFICIENT=1 → backward trades a little speed and
# retain_graph support for TK*(2I + I) bytes of peak memory (see module docstring).
# Must be set before importing liger_kernel (the kernels module reads it at import
# to configure the autotuner's restore_value for the in-place alias).
_MEMORY_EFFICIENT = _kernels_mod._MEMORY_EFFICIENT


# Device-side TMA descriptors (tl.make_tensor_descriptor) need a global-memory
# scratch allocator. Triton stores it in a ContextVar, which does NOT propagate to
# the autograd engine's backward thread — so it is (re)registered at the top of
# both forward() and backward() instead of only at import time.
def _tma_alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _ensure_tma_allocator():
    triton.set_allocator(_tma_alloc_fn)


# Fallback M-tile size (used when callers pass no explicit block_m_token).
# The autograd function picks it adaptively per call via _pick_block_m_token.
BLOCK_M_TOKEN = 64


def _tma_eligibility(t, H: int, intermediate_dim: int, E: int):
    """TMA needs sm90+, 16-byte-aligned rows, and int32-addressable row counts.
    W1 view is (E*2I, H) (row stride H), W2 view is (E*H, I) (row stride I)."""
    if t.device.type != "cuda" or torch.cuda.get_device_capability(t.device)[0] < 9:
        return False, False
    itemsize = t.element_size()
    w1_ok = (H * itemsize) % 16 == 0 and E * 2 * intermediate_dim < 2**31
    w2_ok = (intermediate_dim * itemsize) % 16 == 0 and E * H < 2**31
    return w1_ok, w2_ok


def _pick_block_m_token(TK: int, E: int) -> int:
    """Match the M-tile to the expected expert segment length: large tiles at high
    occupancy amortize weight re-reads; small tiles at low occupancy avoid running
    mostly-padded MMAs (e.g. T=128,K=8,E=128 → 8 tokens/expert → 87% padding at 64)."""
    avg = max(1, TK // max(1, E))
    b = triton.next_power_of_2(avg)
    return max(16, min(128, b))


# ---------------------------------------------------------------------------
# Routing metadata
# ---------------------------------------------------------------------------


def compute_routing_metadata(topk_indices: torch.Tensor, E: int, block_m_token: int = BLOCK_M_TOKEN):
    """Compute token→expert routing permutation metadata via 3 Triton kernels.

    Fully sync-free: tile metadata is allocated at a host-computable upper bound
    (TK//block_m_token + min(E, TK)); the actual m-tile count stays on device in
    expert_tile_offset[E] and the GEMM kernels early-exit CTAs past it. No CPU
    loop, no .item(), so the whole path is CUDA-graph-capturable.

    Args:
        topk_indices:  (T, K) int32 — pre-computed top-k expert indices per token
        E:             number of experts
        block_m_token: BLOCK_M for token-dimension tiling (default BLOCK_M_TOKEN)

    Returns:
        expert_token_count:     (E,)              int32
        expert_start_idx:       (E+1,)            int32
        x_gather_idx:           (TK,)             int32
        s_scatter_idx:          (TK,)             int32
        s_reverse_scatter_idx:  (TK,)             int32
        tile_row_start:         (num_m_tiles_max,) int32 — absolute row_start per M-tile
        tile_expert:            (num_m_tiles_max,) int32 — expert index per M-tile
        expert_tile_offset:     (E+1,)            int32 — cumsum of per-expert tile
                                counts; [E] holds the actual total m-tile count
    """
    T, K = topk_indices.shape
    TK = T * K
    device = topk_indices.device
    E_POW2 = triton.next_power_of_2(E)
    K_POW2 = triton.next_power_of_2(K)
    TOKENS_PER_BLOCK = max(1, 1024 // K_POW2)
    n_tiles = triton.cdiv(T, TOKENS_PER_BLOCK)

    # Kernel 1: tiled histogram → tile_expert_counts (E, n_tiles)
    tile_expert_counts = torch.empty(E, n_tiles, dtype=torch.int32, device=device)
    _moe_router_histogram_kernel[(n_tiles,)](
        topk_indices,
        tile_expert_counts,
        T,
        E=E,
        n_tiles=n_tiles,
        TOKENS_PER_TILE=TOKENS_PER_BLOCK,
        K_POW2=K_POW2,
        K=K,
        E_POW2=E_POW2,
    )

    expert_token_count = tile_expert_counts.sum(dim=1, dtype=torch.int32)  # (E,)

    # Kernel 2: prefix sums + expert offsets + tile offsets (all in one pass)
    expert_start_idx = torch.empty(E + 1, dtype=torch.int32, device=device)
    expert_tile_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    _moe_router_prefix_sum_kernel[(E + 2,)](
        expert_token_count,
        expert_start_idx,
        expert_tile_offset,
        E=E,
        partial_sum_ptr=tile_expert_counts,
        n_tiles=n_tiles,
        TK=TK,
        BLOCK_M=128,
        BLOCK_N=E_POW2,
        BLOCK_M_TOKEN=block_m_token,
    )

    # No host sync: allocate tile metadata at the worst-case bound and let GEMM
    # CTAs past the actual count (expert_tile_offset[E], read on device) exit early.
    # Bound: sum_e ceil(f_e / B) <= floor(TK / B) + #nonempty_experts <= TK//B + min(E, TK).
    num_m_tiles_max = TK // block_m_token + min(E, TK)

    tile_row_start = torch.empty(num_m_tiles_max, dtype=torch.int32, device=device)
    tile_expert = torch.empty(num_m_tiles_max, dtype=torch.int32, device=device)

    # Kernel 3: sort by expert + scatter permutation arrays + tile metadata
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    if TK > 0:
        _moe_router_scatter_kernel[(n_tiles,)](
            s_scatter_idx,
            s_reverse_scatter_idx,
            x_gather_idx,
            tile_row_start,
            tile_expert,
            topk_indices,
            T,
            tile_expert_counts,  # non-contiguous (E, n_tiles) view
            n_tiles,
            expert_start_idx[:E],  # E entries (without TK sentinel)
            expert_tile_offset[:E],  # E entries of cumulative tile counts
            K_POW2=K_POW2,
            K=K,
            TOKENS_PER_BLOCK=TOKENS_PER_BLOCK,
            BLOCK_M_TOKEN=block_m_token,
        )

    return (
        expert_token_count,
        expert_start_idx,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        tile_row_start,
        tile_expert,
        expert_tile_offset,
    )


def _token_aggregation(Y, topk_weights_flat, s_reverse_scatter_idx, T, K, H):
    """Weighted gather-sum: out[t] = sum_k w[t,k] * Y[s_rev[t*K+k]]."""
    out = torch.empty(T, H, dtype=Y.dtype, device=Y.device)
    _token_gather_weighted_sum_kernel[lambda meta: (T, triton.cdiv(H, meta["BLOCK_H"]))](
        Y,
        topk_weights_flat,
        s_reverse_scatter_idx,
        out,
        H_dim=H,
        K_dim=K,
        stride_Y_TK=Y.stride(0),
        stride_Y_H=Y.stride(1),
        stride_out_T=out.stride(0),
        stride_out_H=out.stride(1),
        w_is_None=False,
    )
    return out


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------


class LigerFusedMoEFunction(torch.autograd.Function):
    """Fused grouped GEMM MoE forward + memory-efficient backward.

    Forward: routing metadata → fused gather+GEMM+SwiGLU → down-proj → token aggregation.
    Backward: avoids caching Y (TK×H) by recomputing dA' = dO@W2^T.

    Troubleshooting:
        If Triton's autotune ``do_bench`` loop OOMs (each config holds its own
        working set — see issue #1246), set ``LIGER_FUSED_MOE_AUTOTUNE=0`` before
        importing liger_kernel to pin each kernel to a single config and skip the
        benchmark loop. Temporary escape hatch until triton's autotuner handles
        such errors itself.

        Set ``LIGER_FUSED_MOE_MEMORY_EFFICIENT=1`` to shave another ~TK*3I*itemsize
        bytes off backward peak memory (in-place SwiGLU backward + weighted_act
        recompute) at ~10-15% slower backward; retain_graph re-backward then raises.
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, x, gate_up_proj, down_proj, top_k_index, top_k_weights):
        """
        Args:
            x:             (T, H)      input tokens
            gate_up_proj:  (E, 2*intermediate_dim, H) gate+up projection weights
            down_proj:     (E, H, intermediate_dim)   down projection weights
            top_k_index:   (T, K) int32 — pre-computed routing indices
            top_k_weights: (T, K) float — pre-computed routing scores
        Returns:
            output: (T, H)
        """
        T, K = top_k_index.shape
        E = gate_up_proj.shape[0]
        H = x.shape[1]
        intermediate_dim = gate_up_proj.shape[1] // 2
        TK = T * K

        block_m_token = _pick_block_m_token(TK, E)
        use_tma_w1, use_tma_w2 = _tma_eligibility(x, H, intermediate_dim, E)
        if use_tma_w1 or use_tma_w2:
            _ensure_tma_allocator()
        # Inference: no input needs grad → skip saving/storing pre_act (TK×2I bytes).
        needs_grad = (
            x.requires_grad or gate_up_proj.requires_grad or down_proj.requires_grad or (top_k_weights.requires_grad)
        )

        with torch.no_grad():
            (
                _,
                expert_start_idx,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
                tile_row_start,
                tile_expert,
                expert_tile_offset,
            ) = compute_routing_metadata(top_k_index, E, block_m_token)

        num_m_tiles = tile_row_start.shape[0]  # upper bound; actual count lives on device
        total_tiles_dev = expert_tile_offset[E:]

        post_act = torch.empty(TK, intermediate_dim, dtype=x.dtype, device=x.device)
        # pre_act only exists in training; in inference the kernel skips the store
        # entirely (post_act doubles as a dummy pointer that is never written).
        pre_act = torch.empty(TK, 2 * intermediate_dim, dtype=x.dtype, device=x.device) if needs_grad else post_act

        if num_m_tiles > 0:
            _fused_up_proj_swiglu_kernel[lambda meta: (num_m_tiles * triton.cdiv(intermediate_dim, meta["BLOCK_N"]),)](
                x,
                gate_up_proj,
                x_gather_idx,
                expert_start_idx,
                tile_row_start,
                tile_expert,
                total_tiles_dev,
                pre_act,
                post_act,
                w_rows=E * 2 * intermediate_dim,
                H_dim=H,
                I_dim=intermediate_dim,
                stride_x_T=x.stride(0),
                stride_x_H=x.stride(1),
                stride_w_E=gate_up_proj.stride(0),
                stride_w_N=gate_up_proj.stride(1),
                stride_w_K=gate_up_proj.stride(2),
                stride_pre_TK=pre_act.stride(0),
                stride_pre_N=pre_act.stride(1),
                stride_post_TK=post_act.stride(0),
                stride_post_N=post_act.stride(1),
                BLOCK_M=block_m_token,
                USE_TMA=use_tma_w1,
                STORE_PREACT=needs_grad,
            )

        Y = torch.empty(TK, H, dtype=x.dtype, device=x.device)

        if num_m_tiles > 0:
            _fused_down_proj_kernel[lambda meta: (num_m_tiles * triton.cdiv(H, meta["BLOCK_N"]),)](
                post_act,
                down_proj,
                expert_start_idx,
                tile_row_start,
                tile_expert,
                total_tiles_dev,
                Y,
                w_rows=E * H,
                H_dim=H,
                I_dim=intermediate_dim,
                stride_post_TK=post_act.stride(0),
                stride_post_I=post_act.stride(1),
                stride_w_E=down_proj.stride(0),
                stride_w_H=down_proj.stride(1),
                stride_w_I=down_proj.stride(2),
                stride_Y_TK=Y.stride(0),
                stride_Y_H=Y.stride(1),
                BLOCK_M=block_m_token,
                USE_TMA=use_tma_w2,
            )

        topk_weights_flat = top_k_weights.flatten().contiguous()
        out = _token_aggregation(Y, topk_weights_flat, s_reverse_scatter_idx, T, K, H)

        if needs_grad:
            ctx.save_for_backward(
                x,
                gate_up_proj,
                down_proj,
                pre_act,
                topk_weights_flat,
                expert_start_idx,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
                tile_row_start,
                tile_expert,
                total_tiles_dev,
            )
        ctx.T = T
        ctx.K = K
        ctx.E = E
        ctx.H = H
        ctx.intermediate_dim = intermediate_dim
        ctx.TK = TK
        ctx.num_m_tiles = num_m_tiles
        ctx.block_m_token = block_m_token
        ctx.mark_non_differentiable(top_k_index)
        ctx.set_materialize_grads(False)

        return out

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dO):
        if dO is None:
            return None, None, None, None, None

        (
            x,
            gate_up_proj,
            down_proj,
            pre_act,
            topk_weights_flat,
            expert_start_idx,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            tile_row_start,
            tile_expert,
            total_tiles_dev,
        ) = ctx.saved_tensors

        T = ctx.T
        K = ctx.K
        E = ctx.E
        H = ctx.H
        intermediate_dim = ctx.intermediate_dim
        TK = ctx.TK
        num_m_tiles = ctx.num_m_tiles
        block_m_token = ctx.block_m_token
        use_tma_w1, use_tma_w2 = _tma_eligibility(dO, H, intermediate_dim, E)
        if use_tma_w1 or use_tma_w2:
            _ensure_tma_allocator()

        mem_eff = _MEMORY_EFFICIENT

        # Tokens-per-expert bucket for the dW autotune keys: the best dW tile is
        # regime-dependent (output-write-bound at small TK/E, K-loop-bound at
        # large TK/E), but H_dim/I_dim alone can't see the difference — without
        # this key the config tuned at the first-seen T is reused for every T.
        # Clamped: below 16 / beyond 4096 tokens-per-expert the optimum stops
        # moving, so extreme sizes share the edge buckets instead of retuning.
        tpe_bucket = max(16, min(4096, triton.next_power_of_2(max(1, TK // max(1, E)))))

        # ---- dW2 = (s_k * y1)^T @ dO_gathered (memory-efficient order) ------
        # In memory-efficient mode s_k*y1 is recomputed from pre_act inside dW2, so
        # dW2 MUST run before the bwd-down-proj kernel overwrites pre_act in place.
        # empty (not zeros): the kernel writes every element, storing 0 for empty experts.
        ddown_proj = torch.empty_like(down_proj)
        if mem_eff:
            # No weighted_act buffer, and d_pre_act aliases pre_act (in-place):
            # each (row, n) element is consumed and produced by the same CTA, so
            # the alias is race-free and saves a (TK, 2I) buffer. Costs support
            # for a second backward over the same graph (version check raises).
            weighted_act = pre_act[:0]  # dummy, never read
            d_pre_act = pre_act
            _moe_bwd_dW2_kernel[
                lambda meta: (E * triton.cdiv(intermediate_dim, meta["BLOCK_M"]) * triton.cdiv(H, meta["BLOCK_N"]),)
            ](
                weighted_act,
                pre_act,
                s_scatter_idx,
                topk_weights_flat,
                dO,
                x_gather_idx,
                expert_start_idx,
                ddown_proj,
                H_dim=H,
                I_dim=intermediate_dim,
                stride_wact_TK=0,
                stride_wact_I=1,
                stride_pre_TK=pre_act.stride(0),
                stride_pre_N=pre_act.stride(1),
                stride_dout_T=dO.stride(0),
                stride_dout_H=dO.stride(1),
                stride_dW2_E=ddown_proj.stride(0),
                stride_dW2_H=ddown_proj.stride(1),
                stride_dW2_I=ddown_proj.stride(2),
                RECOMPUTE_WACT=True,
                TPE_BUCKET=tpe_bucket,
            )
        else:
            weighted_act = torch.empty(TK, intermediate_dim, dtype=dO.dtype, device=dO.device)
            d_pre_act = torch.empty(TK, 2 * intermediate_dim, dtype=dO.dtype, device=dO.device)

        # ---- dA' = dO @ W2^T, SwiGLU backward → d_pre_act, dS ---------------
        # fp32 dS: atomic_add accumulates ceil(I/BLOCK_N) partials per element;
        # low-precision atomics would round every partial.
        dS = torch.zeros(TK, dtype=torch.float32, device=dO.device)

        if num_m_tiles > 0:
            _moe_bwd_down_proj_kernel[lambda meta: (num_m_tiles * triton.cdiv(intermediate_dim, meta["BLOCK_N"]),)](
                dO,
                x_gather_idx,
                s_scatter_idx,
                topk_weights_flat,
                down_proj,
                pre_act,
                expert_start_idx,
                tile_row_start,
                tile_expert,
                total_tiles_dev,
                d_pre_act,
                weighted_act,
                dS,
                w_rows=E * H,
                H_dim=H,
                I_dim=intermediate_dim,
                stride_dO_T=dO.stride(0),
                stride_dO_H=dO.stride(1),
                stride_w_E=down_proj.stride(0),
                stride_w_H=down_proj.stride(1),
                stride_w_I=down_proj.stride(2),
                stride_pre_TK=pre_act.stride(0),
                stride_pre_N=pre_act.stride(1),
                stride_d_pre_TK=d_pre_act.stride(0),
                stride_d_pre_N=d_pre_act.stride(1),
                stride_wact_TK=weighted_act.stride(0) if not mem_eff else 0,
                stride_wact_I=weighted_act.stride(1) if not mem_eff else 1,
                BLOCK_M=block_m_token,
                USE_TMA=use_tma_w2,
                WRITE_WACT=not mem_eff,
            )

        if mem_eff:
            # pre_act now holds d_pre_act; bump its autograd version so a second
            # backward through the same graph errors out instead of silently
            # producing garbage.
            torch.autograd.graph.increment_version(pre_act)
        else:
            _moe_bwd_dW2_kernel[
                lambda meta: (E * triton.cdiv(intermediate_dim, meta["BLOCK_M"]) * triton.cdiv(H, meta["BLOCK_N"]),)
            ](
                weighted_act,
                pre_act,
                s_scatter_idx,
                topk_weights_flat,
                dO,
                x_gather_idx,
                expert_start_idx,
                ddown_proj,
                H_dim=H,
                I_dim=intermediate_dim,
                stride_wact_TK=weighted_act.stride(0),
                stride_wact_I=weighted_act.stride(1),
                stride_pre_TK=pre_act.stride(0),
                stride_pre_N=pre_act.stride(1),
                stride_dout_T=dO.stride(0),
                stride_dout_H=dO.stride(1),
                stride_dW2_E=ddown_proj.stride(0),
                stride_dW2_H=ddown_proj.stride(1),
                stride_dW2_I=ddown_proj.stride(2),
                RECOMPUTE_WACT=False,
                TPE_BUCKET=tpe_bucket,
            )

        # dx_expanded = d_pre_act @ W1^T
        dx_expanded = torch.empty(TK, H, dtype=dO.dtype, device=dO.device)

        if num_m_tiles > 0:
            _moe_bwd_dX_expanded_kernel[lambda meta: (num_m_tiles * triton.cdiv(H, meta["BLOCK_N"]),)](
                d_pre_act,
                gate_up_proj,
                expert_start_idx,
                tile_row_start,
                tile_expert,
                total_tiles_dev,
                dx_expanded,
                w_rows=E * 2 * intermediate_dim,
                H_dim=H,
                I_dim=intermediate_dim,
                stride_d_pre_TK=d_pre_act.stride(0),
                stride_d_pre_N=d_pre_act.stride(1),
                stride_w_E=gate_up_proj.stride(0),
                stride_w_N=gate_up_proj.stride(1),
                stride_w_K=gate_up_proj.stride(2),
                stride_dxe_TK=dx_expanded.stride(0),
                stride_dxe_H=dx_expanded.stride(1),
                BLOCK_M=block_m_token,
                USE_TMA=use_tma_w1,
            )

        # dx = unweighted gather-sum of dx_expanded
        # empty (not zeros): the gather-sum kernel stores every (t, h) element.
        dx = torch.empty(T, H, dtype=dO.dtype, device=dO.device)
        if TK > 0:
            _token_gather_weighted_sum_kernel[lambda meta: (T, triton.cdiv(H, meta["BLOCK_H"]))](
                dx_expanded,
                dS,  # dummy w_ptr — never loaded when w_is_None=True
                s_reverse_scatter_idx,
                dx,
                H_dim=H,
                K_dim=K,
                stride_Y_TK=dx_expanded.stride(0),
                stride_Y_H=dx_expanded.stride(1),
                stride_out_T=dx.stride(0),
                stride_out_H=dx.stride(1),
                w_is_None=True,
            )

        # dW1 = X_gathered^T @ d_pre_act
        # empty (not zeros): the kernel writes every element, storing 0 for empty experts.
        dgate_up_proj = torch.empty_like(gate_up_proj)
        _moe_bwd_dW1_kernel[
            lambda meta: (E * triton.cdiv(H, meta["BLOCK_M"]) * triton.cdiv(2 * intermediate_dim, meta["BLOCK_N"]),)
        ](
            x,
            d_pre_act,
            x_gather_idx,
            expert_start_idx,
            dgate_up_proj,
            H_dim=H,
            I_dim=intermediate_dim,
            stride_x_T=x.stride(0),
            stride_x_H=x.stride(1),
            stride_d_pre_TK=d_pre_act.stride(0),
            stride_d_pre_N=d_pre_act.stride(1),
            stride_dW1_E=dgate_up_proj.stride(0),
            stride_dW1_N=dgate_up_proj.stride(1),
            stride_dW1_H=dgate_up_proj.stride(2),
            TPE_BUCKET=tpe_bucket,
        )

        return dx, dgate_up_proj, ddown_proj, None, dS.to(topk_weights_flat.dtype).view(T, K)
