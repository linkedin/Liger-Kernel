"""
Fused MoE expert computation using Triton grouped GEMM.

Implements a memory-efficient forward+backward for Mixture-of-Experts layers,
inspired by SonicMoE (arXiv:2512.14080).

Key properties:
- Routing metadata via 3 Triton kernels (port from SonicMoE)
- GPU tile metadata (expert_for_tile, tile_expert) computed in Kernel 3 — no CPU loop
- Fused gather + grouped GEMM + SwiGLU in forward
- Memory-efficient backward: dS = <dA', A> avoids caching Y (TK×H bytes)
- General GPU support: pure Triton, no Hopper-specific WGMMA/TMA
"""

import torch
import triton

from liger_kernel.ops.fused_moe_kernels import (
    _fused_down_proj_kernel,
    _fused_up_proj_swiglu_kernel,
    _moe_bwd_down_proj_kernel,
    _moe_bwd_dW1_kernel,
    _moe_bwd_dW2_kernel,
    _moe_bwd_dX_expanded_kernel,
    _moe_router_histogram_kernel,
    _moe_router_prefix_sum_kernel,
    _moe_router_scatter_kernel,
    _token_gather_weighted_sum_kernel,
)
from liger_kernel.ops.utils import ensure_contiguous

# Fixed token-dimension tile size for M (not autotuned).
# Decouples tile metadata computation from GEMM autotune.
BLOCK_M_TOKEN = 16


# ---------------------------------------------------------------------------
# Routing metadata
# ---------------------------------------------------------------------------


def compute_routing_metadata(topk_indices: torch.Tensor, E: int, block_m_token: int = BLOCK_M_TOKEN):
    """Compute token→expert routing permutation metadata via 3 Triton kernels.

    Also computes GPU tile metadata (expert_for_tile, tile_expert) inside
    Kernel 3 — no CPU loop, one .item() sync for num_m_tiles allocation.

    Args:
        topk_indices:  (T, K) int32 — pre-computed top-k expert indices per token
        E:             number of experts
        block_m_token: BLOCK_M for token-dimension tiling (default BLOCK_M_TOKEN)

    Returns:
        expert_frequency:       (E,)            int32
        expert_freq_offset:     (E+1,)          int32
        x_gather_idx:           (TK,)           int32
        s_scatter_idx:          (TK,)           int32
        s_reverse_scatter_idx:  (TK,)           int32
        expert_for_tile:        (num_m_tiles,)  int32 — absolute row_start per M-tile
        tile_expert:            (num_m_tiles,)  int32 — expert index per M-tile
    """
    T, K = topk_indices.shape
    TK = T * K
    device = topk_indices.device
    E_POW2 = triton.next_power_of_2(E)
    K_POW2 = triton.next_power_of_2(K)
    TOKENS_PER_BLOCK = max(1, 1024 // K_POW2)
    n_tiles = triton.cdiv(T, TOKENS_PER_BLOCK)

    # Kernel 1: tiled histogram → partial_sum (E, n_tiles), stored column-per-tile
    col_partial_sum_trans = torch.empty(E, n_tiles, dtype=torch.int32, device=device)
    _moe_router_histogram_kernel[(n_tiles,)](
        topk_indices,
        col_partial_sum_trans,
        T,
        E=E,
        n_tiles=n_tiles,
        TOKENS_PER_TILE=TOKENS_PER_BLOCK,
        K_POW2=K_POW2,
        K=K,
        E_POW2=E_POW2,
    )

    expert_frequency = col_partial_sum_trans.sum(dim=1, dtype=torch.int32)  # (E,)

    # Kernel 2: prefix sums + expert offsets + tile offsets (all in one pass)
    expert_freq_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    expert_tile_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    _moe_router_prefix_sum_kernel[(E + 2,)](
        expert_frequency,
        expert_freq_offset,
        expert_tile_offset,
        E=E,
        partial_sum_ptr=col_partial_sum_trans,
        n_tiles=n_tiles,
        TK=TK,
        BLOCK_M=128,
        BLOCK_N=E_POW2,
        BLOCK_M_TOKEN=block_m_token,
    )

    # One sync to get num_m_tiles for buffer allocation and GEMM grid.
    num_m_tiles = int(expert_tile_offset[-1].item())

    expert_for_tile = torch.empty(num_m_tiles, dtype=torch.int32, device=device)
    tile_expert = torch.empty(num_m_tiles, dtype=torch.int32, device=device)

    # Kernel 3: sort by expert + scatter permutation arrays + tile metadata
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    if TK > 0:
        _moe_router_scatter_kernel[(n_tiles,)](
            s_scatter_idx,
            s_reverse_scatter_idx,
            x_gather_idx,
            expert_for_tile,
            tile_expert,
            topk_indices,
            T,
            col_partial_sum_trans,  # non-contiguous (E, n_tiles) view
            n_tiles,
            expert_freq_offset[:E],  # E entries (without TK sentinel)
            expert_tile_offset[:E],  # E entries of cumulative tile counts
            K_POW2=K_POW2,
            K=K,
            TOKENS_PER_BLOCK=TOKENS_PER_BLOCK,
            BLOCK_M_TOKEN=block_m_token,
        )

    return (
        expert_frequency,
        expert_freq_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        expert_for_tile,
        tile_expert,
    )


def _token_aggregation(Y, topk_weights_flat, s_reverse_scatter_idx, T, K, H):
    """Weighted gather-sum: O[t] = sum_k w[t,k] * Y[s_rev[t*K+k]]."""
    O = torch.empty(T, H, dtype=Y.dtype, device=Y.device)
    _token_gather_weighted_sum_kernel[(T,)](
        Y,
        topk_weights_flat,
        s_reverse_scatter_idx,
        O,
        H_dim=H,
        K_dim=K,
        stride_Y_TK=Y.stride(0),
        stride_Y_H=Y.stride(1),
        stride_out_T=O.stride(0),
        stride_out_H=O.stride(1),
        w_is_None=False,
    )
    return O


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------


class LigerFusedMoEFunction(torch.autograd.Function):
    """Fused grouped GEMM MoE forward + memory-efficient backward.

    Forward: routing metadata → fused gather+GEMM+SwiGLU → down-proj → token aggregation
    Backward: avoids caching Y (TK×H) by recomputing dA' = dO@W2^T in backward
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, x, gate_up_proj, down_proj, top_k_index, top_k_weights):
        """
        Args:
            x:             (T, H)      input tokens
            gate_up_proj:  (E, 2*I, H) gate+up projection weights
            down_proj:     (E, H, I)   down projection weights
            top_k_index:   (T, K) int32 — pre-computed routing indices
            top_k_weights: (T, K) float — pre-computed routing scores
        Returns:
            output: (T, H)
        """
        T, K = top_k_index.shape
        E = gate_up_proj.shape[0]
        H = x.shape[1]
        I = gate_up_proj.shape[1] // 2
        TK = T * K

        with torch.no_grad():
            (
                _,
                expert_freq_offset,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
                expert_for_tile,
                tile_expert,
            ) = compute_routing_metadata(top_k_index, E)

        num_m_tiles = expert_for_tile.shape[0]

        H_pre = torch.empty(TK, 2 * I, dtype=x.dtype, device=x.device)
        A_post = torch.empty(TK, I, dtype=x.dtype, device=x.device)

        if num_m_tiles > 0:
            _fused_up_proj_swiglu_kernel[
                lambda meta: (num_m_tiles, triton.cdiv(I, meta["BLOCK_N"]))
            ](
                x,
                gate_up_proj,
                x_gather_idx,
                expert_freq_offset,
                expert_for_tile,
                tile_expert,
                H_pre,
                A_post,
                H_dim=H,
                I_dim=I,
                stride_x_T=x.stride(0),
                stride_x_H=x.stride(1),
                stride_w_E=gate_up_proj.stride(0),
                stride_w_N=gate_up_proj.stride(1),
                stride_w_K=gate_up_proj.stride(2),
                stride_Hpre_TK=H_pre.stride(0),
                stride_Hpre_N=H_pre.stride(1),
                stride_A_TK=A_post.stride(0),
                stride_A_N=A_post.stride(1),
                BLOCK_M=BLOCK_M_TOKEN,
            )

        Y = torch.empty(TK, H, dtype=x.dtype, device=x.device)

        if num_m_tiles > 0:
            _fused_down_proj_kernel[
                lambda meta: (num_m_tiles, triton.cdiv(H, meta["BLOCK_N"]))
            ](
                A_post,
                down_proj,
                expert_freq_offset,
                expert_for_tile,
                tile_expert,
                Y,
                H_dim=H,
                I_dim=I,
                stride_A_TK=A_post.stride(0),
                stride_A_I=A_post.stride(1),
                stride_w_E=down_proj.stride(0),
                stride_w_H=down_proj.stride(1),
                stride_w_I=down_proj.stride(2),
                stride_Y_TK=Y.stride(0),
                stride_Y_H=Y.stride(1),
                BLOCK_M=BLOCK_M_TOKEN,
            )

        topk_weights_flat = top_k_weights.flatten().contiguous()
        O = _token_aggregation(Y, topk_weights_flat, s_reverse_scatter_idx, T, K, H)

        ctx.save_for_backward(
            x,
            gate_up_proj,
            down_proj,
            H_pre,
            topk_weights_flat,
            expert_freq_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            expert_for_tile,
            tile_expert,
        )
        ctx.T = T
        ctx.K = K
        ctx.E = E
        ctx.H = H
        ctx.I = I
        ctx.TK = TK
        ctx.num_m_tiles = num_m_tiles
        ctx.mark_non_differentiable(top_k_index)
        ctx.set_materialize_grads(False)

        return O

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dO):
        if dO is None:
            return None, None, None, None, None

        (
            x,
            gate_up_proj,
            down_proj,
            H_pre,
            topk_weights_flat,
            expert_freq_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            expert_for_tile,
            tile_expert,
        ) = ctx.saved_tensors

        T = ctx.T
        K = ctx.K
        E = ctx.E
        H = ctx.H
        I = ctx.I
        TK = ctx.TK
        num_m_tiles = ctx.num_m_tiles

        # Step B1: Fused down-proj backward (port of SonicMoE _down_projection_backward_act).
        # Computes dA' = dY @ W2^T (dY stays in registers, never materialized),
        # recomputes y1 = silu(gate)*up from H_pre, writes dH_pre + y1s_scaled + dS.
        dH_pre = torch.empty(TK, 2 * I, dtype=dO.dtype, device=dO.device)
        y1s_scaled = torch.empty(TK, I, dtype=dO.dtype, device=dO.device)
        # dS is written directly in flat (t,k) order by the kernel (no post-scatter needed).
        dS = torch.empty(TK, dtype=dO.dtype, device=dO.device)

        if num_m_tiles > 0:
            _moe_bwd_down_proj_kernel[
                lambda meta: (num_m_tiles, triton.cdiv(I, meta["BLOCK_N"]))
            ](
                dO,
                x_gather_idx,
                s_scatter_idx,
                topk_weights_flat,
                down_proj,
                H_pre,
                expert_freq_offset,
                expert_for_tile,
                tile_expert,
                dH_pre,
                y1s_scaled,
                dS,
                H_dim=H,
                I_dim=I,
                stride_dO_T=dO.stride(0),
                stride_dO_H=dO.stride(1),
                stride_w_E=down_proj.stride(0),
                stride_w_H=down_proj.stride(1),
                stride_w_I=down_proj.stride(2),
                stride_Hpre_TK=H_pre.stride(0),
                stride_Hpre_N=H_pre.stride(1),
                stride_dHpre_TK=dH_pre.stride(0),
                stride_dHpre_N=dH_pre.stride(1),
                stride_y1s_TK=y1s_scaled.stride(0),
                stride_y1s_I=y1s_scaled.stride(1),
                BLOCK_M=BLOCK_M_TOKEN,
            )

        # Step B2: dW2 = y1s_scaled^T @ dout_gathered (port of _down_projection_backward_weight).
        # Gathers dout (T,H) by x_gather_idx — no dY (TK,H) buffer needed.
        ddown_proj = torch.zeros_like(down_proj)
        _moe_bwd_dW2_kernel[
            lambda meta: (
                E * triton.cdiv(I, meta["BLOCK_M"]),
                triton.cdiv(H, meta["BLOCK_N"]),
            )
        ](
            y1s_scaled,
            dO,
            x_gather_idx,
            expert_freq_offset,
            ddown_proj,
            H_dim=H,
            I_dim=I,
            stride_y1s_TK=y1s_scaled.stride(0),
            stride_y1s_I=y1s_scaled.stride(1),
            stride_dout_T=dO.stride(0),
            stride_dout_H=dO.stride(1),
            stride_dW2_E=ddown_proj.stride(0),
            stride_dW2_H=ddown_proj.stride(1),
            stride_dW2_I=ddown_proj.stride(2),
        )

        # Step B3: dx_expanded = dH_pre @ W1^T (port of _up_projection_backward_act).
        # Clean write to (TK, H) — no atomic scatter.
        dx_expanded = torch.empty(TK, H, dtype=dO.dtype, device=dO.device)

        if num_m_tiles > 0:
            _moe_bwd_dX_expanded_kernel[
                lambda meta: (num_m_tiles, triton.cdiv(H, meta["BLOCK_N"]))
            ](
                dH_pre,
                gate_up_proj,
                expert_freq_offset,
                expert_for_tile,
                tile_expert,
                dx_expanded,
                H_dim=H,
                I_dim=I,
                stride_dH_TK=dH_pre.stride(0),
                stride_dH_N=dH_pre.stride(1),
                stride_w_E=gate_up_proj.stride(0),
                stride_w_N=gate_up_proj.stride(1),
                stride_w_K=gate_up_proj.stride(2),
                stride_dxe_TK=dx_expanded.stride(0),
                stride_dxe_H=dx_expanded.stride(1),
                BLOCK_M=BLOCK_M_TOKEN,
            )

        # Step B4: dx = reduce(dx_expanded) (port of _token_broadcast_backward).
        # Unweighted gather-sum: dx[t] = sum_k dx_expanded[s_rev[t*K+k]].
        dx = torch.zeros(T, H, dtype=dO.dtype, device=dO.device)
        if TK > 0:
            _token_gather_weighted_sum_kernel[(T,)](
                dx_expanded,
                dS,              # dummy w_ptr — never loaded when w_is_None=True
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

        # Step B5: dW1 = Gathered_X^T @ dH_pre (per expert, lambda grid).
        dgate_up_proj = torch.zeros_like(gate_up_proj)
        _moe_bwd_dW1_kernel[
            lambda meta: (
                E * triton.cdiv(H, meta["BLOCK_M"]),
                triton.cdiv(2 * I, meta["BLOCK_N"]),
            )
        ](
            x,
            dH_pre,
            x_gather_idx,
            expert_freq_offset,
            dgate_up_proj,
            H_dim=H,
            I_dim=I,
            stride_x_T=x.stride(0),
            stride_x_H=x.stride(1),
            stride_dH_TK=dH_pre.stride(0),
            stride_dH_N=dH_pre.stride(1),
            stride_dW1_E=dgate_up_proj.stride(0),
            stride_dW1_N=dgate_up_proj.stride(1),
            stride_dW1_H=dgate_up_proj.stride(2),
        )

        return dx, dgate_up_proj, ddown_proj, None, dS.view(T, K)
