"""
Fused MoE expert computation via Triton grouped GEMM (Ascend backend).

Forward: routing metadata (3 kernels) → fused gather+GEMM+SwiGLU → down-proj → token aggregation
Backward: memory-efficient — recomputes dA' = dO@W2^T to avoid caching Y (TK×H bytes)
"""

import torch
import triton

from liger_kernel.ops.utils import ensure_contiguous

from .fused_moe_kernels import ASCEND_BWD_BLOCK_N
from .fused_moe_kernels import ASCEND_DW_BLOCK_K
from .fused_moe_kernels import ASCEND_DW_BLOCK_M
from .fused_moe_kernels import ASCEND_DW_BLOCK_N
from .fused_moe_kernels import ASCEND_GEMM_BLOCK_K
from .fused_moe_kernels import ASCEND_GEMM_BLOCK_N
from .fused_moe_kernels import ASCEND_MAX_GRID_PROGRAMS
from .fused_moe_kernels import ASCEND_TOKEN_GATHER_BLOCK_H
from .fused_moe_kernels import ASCEND_TOKEN_GATHER_BLOCK_K
from .fused_moe_kernels import _fused_down_proj_kernel
from .fused_moe_kernels import _fused_up_proj_swiglu_kernel
from .fused_moe_kernels import _moe_bwd_down_proj_kernel
from .fused_moe_kernels import _moe_bwd_dW1_kernel
from .fused_moe_kernels import _moe_bwd_dW2_kernel
from .fused_moe_kernels import _moe_bwd_dX_expanded_kernel
from .fused_moe_kernels import _moe_router_histogram_kernel
from .fused_moe_kernels import _moe_router_prefix_sum_kernel
from .fused_moe_kernels import _moe_router_scatter_kernel
from .fused_moe_kernels import _token_gather_weighted_sum_kernel

# Token-dimension tile size for M. Fixed (not autotuned) because tile_row_start,
# tile_expert, and the GEMM grid dim-0 depend on this value.
BLOCK_M_TOKEN = 32


# ---------------------------------------------------------------------------
# Routing metadata
# ---------------------------------------------------------------------------


def compute_routing_metadata(topk_indices: torch.Tensor, E: int, block_m_token: int = BLOCK_M_TOKEN):
    """Compute token→expert routing permutation metadata via 3 Triton kernels.

    Also computes GPU tile metadata (tile_row_start, tile_expert) inside
    kernel 3 — no CPU loop, one .item() sync for num_m_tiles allocation.

    Args:
        topk_indices:  (T, K) int32 — pre-computed top-k expert indices per token
        E:             number of experts
        block_m_token: BLOCK_M for token-dimension tiling (default BLOCK_M_TOKEN)

    Returns:
        expert_token_count:     (E,)            int32
        expert_start_idx:       (E+1,)          int32
        x_gather_idx:           (TK,)           int32
        s_scatter_idx:          (TK,)           int32
        s_reverse_scatter_idx:  (TK,)           int32
        tile_row_start:         (num_m_tiles,)  int32 — absolute row_start per M-tile
        tile_expert:            (num_m_tiles,)  int32 — expert index per M-tile
    """
    T, K = topk_indices.shape
    TK = T * K
    device = topk_indices.device
    E_POW2 = triton.next_power_of_2(E)
    K_POW2 = triton.next_power_of_2(K)
    TOKENS_PER_BLOCK = max(1, 1024 // K_POW2)
    n_tiles = triton.cdiv(T, TOKENS_PER_BLOCK)

    # Kernel 1: per-expert counts per token tile
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

    # Kernel 2: expert offsets, tile offsets, and tile-level prefix sums
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

    num_m_tiles = int(expert_tile_offset[-1].item())

    tile_row_start = torch.empty(num_m_tiles, dtype=torch.int32, device=device)
    tile_expert = torch.empty(num_m_tiles, dtype=torch.int32, device=device)
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    # Kernel 3: scatter into expert-sorted order and emit GEMM tile metadata
    if TK > 0:
        _moe_router_scatter_kernel[(n_tiles,)](
            s_scatter_idx,
            s_reverse_scatter_idx,
            x_gather_idx,
            tile_row_start,
            tile_expert,
            topk_indices,
            T,
            TK,
            tile_expert_counts,
            n_tiles,
            expert_start_idx[:E],
            expert_tile_offset[:E],
            E=E,
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
    )


# ---------------------------------------------------------------------------
# Token aggregation
# ---------------------------------------------------------------------------


def _token_gather_block_k(K: int) -> int:
    """Ascend ttir_to_linalg rejects BLOCK_K > K when K=1."""
    return min(ASCEND_TOKEN_GATHER_BLOCK_K, K)


def _launch_token_gather_kernel(src, weights, s_reverse_scatter_idx, out, T, K, H, *, weighted: bool):
    """Gather K routed rows per token and reduce along K (weighted or sum)."""
    _token_gather_weighted_sum_kernel[(T,)](
        src,
        weights,
        s_reverse_scatter_idx,
        out,
        H_dim=H,
        K_dim=K,
        stride_Y_TK=src.stride(0),
        stride_Y_H=src.stride(1),
        stride_out_T=out.stride(0),
        stride_out_H=out.stride(1),
        w_is_None=not weighted,
        BLOCK_H=ASCEND_TOKEN_GATHER_BLOCK_H,
        BLOCK_K=_token_gather_block_k(K),
    )


def _token_aggregation(Y, topk_weights_flat, s_reverse_scatter_idx, T, K, H):
    """Weighted gather-sum: out[t] = sum_k w[t,k] * Y[s_rev[t*K+k]]."""
    out = torch.empty(T, H, dtype=Y.dtype, device=Y.device)
    _launch_token_gather_kernel(Y, topk_weights_flat, s_reverse_scatter_idx, out, T, K, H, weighted=True)
    return out


def _token_scatter_sum(src, s_reverse_scatter_idx, T, K, H):
    """Unweighted gather-sum for backward dx: out[t] = sum_k src[s_rev[t*K+k]]."""
    out = torch.zeros(T, H, dtype=src.dtype, device=src.device)
    if T * K > 0:
        _launch_token_gather_kernel(src, src, s_reverse_scatter_idx, out, T, K, H, weighted=False)
    return out


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

        with torch.no_grad():
            (
                _,
                expert_start_idx,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
                tile_row_start,
                tile_expert,
            ) = compute_routing_metadata(top_k_index, E)

        num_m_tiles = tile_row_start.shape[0]
        pre_act = torch.empty(TK, 2 * intermediate_dim, dtype=x.dtype, device=x.device)
        post_act = torch.empty(TK, intermediate_dim, dtype=x.dtype, device=x.device)

        if num_m_tiles > 0:
            _fused_up_proj_swiglu_kernel[(num_m_tiles,)](
                x,
                gate_up_proj,
                x_gather_idx,
                expert_start_idx,
                tile_row_start,
                tile_expert,
                pre_act,
                post_act,
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
                BLOCK_M=BLOCK_M_TOKEN,
                BLOCK_N=ASCEND_GEMM_BLOCK_N,
                BLOCK_K=ASCEND_GEMM_BLOCK_K,
            )

        Y = torch.empty(TK, H, dtype=x.dtype, device=x.device)
        if num_m_tiles > 0:
            _fused_down_proj_kernel[(num_m_tiles,)](
                post_act,
                down_proj,
                expert_start_idx,
                tile_row_start,
                tile_expert,
                Y,
                H_dim=H,
                I_dim=intermediate_dim,
                stride_post_TK=post_act.stride(0),
                stride_post_I=post_act.stride(1),
                stride_w_E=down_proj.stride(0),
                stride_w_H=down_proj.stride(1),
                stride_w_I=down_proj.stride(2),
                stride_Y_TK=Y.stride(0),
                stride_Y_H=Y.stride(1),
                BLOCK_M=BLOCK_M_TOKEN,
                BLOCK_N=ASCEND_GEMM_BLOCK_N,
                BLOCK_K=ASCEND_GEMM_BLOCK_K,
            )

        topk_weights_flat = top_k_weights.flatten().contiguous()
        out = _token_aggregation(Y, topk_weights_flat, s_reverse_scatter_idx, T, K, H)

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
        )
        ctx.T = T
        ctx.K = K
        ctx.H = H
        ctx.intermediate_dim = intermediate_dim
        ctx.TK = TK
        ctx.num_m_tiles = num_m_tiles
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
        ) = ctx.saved_tensors

        T = ctx.T
        K = ctx.K
        H = ctx.H
        intermediate_dim = ctx.intermediate_dim
        num_m_tiles = ctx.num_m_tiles

        TK = ctx.TK
        d_pre_act = torch.empty(TK, 2 * intermediate_dim, dtype=dO.dtype, device=dO.device)
        weighted_act = torch.empty(TK, intermediate_dim, dtype=dO.dtype, device=dO.device)
        dS = torch.zeros(TK, dtype=dO.dtype, device=dO.device)  # atomic_add accumulates across N-tiles

        if num_m_tiles > 0:
            n_i_tiles = triton.cdiv(intermediate_dim, ASCEND_BWD_BLOCK_N)
            max_m_per_launch = max(1, ASCEND_MAX_GRID_PROGRAMS // n_i_tiles)
            for m_off in range(0, num_m_tiles, max_m_per_launch):
                m_count = min(max_m_per_launch, num_m_tiles - m_off)
                _moe_bwd_down_proj_kernel[(m_count, n_i_tiles)](
                    dO,
                    x_gather_idx,
                    s_scatter_idx,
                    topk_weights_flat,
                    down_proj,
                    pre_act,
                    expert_start_idx,
                    tile_row_start[m_off : m_off + m_count],
                    tile_expert[m_off : m_off + m_count],
                    d_pre_act,
                    weighted_act,
                    dS,
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
                    stride_wact_TK=weighted_act.stride(0),
                    stride_wact_I=weighted_act.stride(1),
                    BLOCK_M=BLOCK_M_TOKEN,
                    BLOCK_N=ASCEND_BWD_BLOCK_N,
                    BLOCK_K=ASCEND_GEMM_BLOCK_K,
                )

        ddown_proj = torch.zeros_like(down_proj)
        _moe_bwd_dW2_kernel[
            (
                gate_up_proj.shape[0] * triton.cdiv(intermediate_dim, ASCEND_DW_BLOCK_M),
                triton.cdiv(H, ASCEND_DW_BLOCK_N),
            )
        ](
            weighted_act,
            dO,
            x_gather_idx,
            expert_start_idx,
            ddown_proj,
            H_dim=H,
            I_dim=intermediate_dim,
            stride_wact_TK=weighted_act.stride(0),
            stride_wact_I=weighted_act.stride(1),
            stride_dout_T=dO.stride(0),
            stride_dout_H=dO.stride(1),
            stride_dW2_E=ddown_proj.stride(0),
            stride_dW2_H=ddown_proj.stride(1),
            stride_dW2_I=ddown_proj.stride(2),
            BLOCK_M=ASCEND_DW_BLOCK_M,
            BLOCK_N=ASCEND_DW_BLOCK_N,
            BLOCK_K=ASCEND_DW_BLOCK_K,
        )

        dx_expanded = torch.empty(TK, H, dtype=dO.dtype, device=dO.device)
        if num_m_tiles > 0:
            _moe_bwd_dX_expanded_kernel[(num_m_tiles,)](
                d_pre_act,
                gate_up_proj,
                expert_start_idx,
                tile_row_start,
                tile_expert,
                dx_expanded,
                H_dim=H,
                I_dim=intermediate_dim,
                stride_d_pre_TK=d_pre_act.stride(0),
                stride_d_pre_N=d_pre_act.stride(1),
                stride_w_E=gate_up_proj.stride(0),
                stride_w_N=gate_up_proj.stride(1),
                stride_w_K=gate_up_proj.stride(2),
                stride_dxe_TK=dx_expanded.stride(0),
                stride_dxe_H=dx_expanded.stride(1),
                BLOCK_M=BLOCK_M_TOKEN,
                BLOCK_N=ASCEND_GEMM_BLOCK_N,
                BLOCK_K=ASCEND_GEMM_BLOCK_K,
            )

        dx = _token_scatter_sum(dx_expanded, s_reverse_scatter_idx, T, K, H)

        dgate_up_proj = torch.zeros_like(gate_up_proj)
        _moe_bwd_dW1_kernel[
            (
                gate_up_proj.shape[0] * triton.cdiv(H, ASCEND_DW_BLOCK_M),
                triton.cdiv(2 * intermediate_dim, ASCEND_DW_BLOCK_N),
            )
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
            BLOCK_M=ASCEND_DW_BLOCK_M,
            BLOCK_N=ASCEND_DW_BLOCK_N,
            BLOCK_K=ASCEND_DW_BLOCK_K,
        )

        return dx, dgate_up_proj, ddown_proj, None, dS.view(T, K)
