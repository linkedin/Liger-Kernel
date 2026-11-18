"""
Fused MoE expert computation via Triton grouped GEMM (Ascend backend).

Forward: routing metadata → gather+GEMM → SwiGLU epilogue → down-proj → token aggregation
Backward: memory-efficient — recomputes dA' = dO@W2^T to avoid caching Y (TK×H bytes)
"""

import torch
import triton

from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count

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
from .fused_moe_kernels import _moe_router_tile_kernel
from .fused_moe_kernels import _swiglu_from_pre_act_kernel
from .fused_moe_kernels import _token_gather_weighted_sum_kernel

# Token-dimension tile size for M. Fixed (not autotuned) because tile_row_start,
# tile_expert, and the GEMM grid dim-0 depend on this value.
BLOCK_M_TOKEN = 32


# ---------------------------------------------------------------------------
# Routing metadata
# ---------------------------------------------------------------------------


def compute_routing_metadata(topk_indices: torch.Tensor, E: int, block_m_token: int = BLOCK_M_TOKEN):
    """Expert-major routing permutation and GEMM tile bounds."""
    T, K = topk_indices.shape
    TK = T * K
    device = topk_indices.device
    topk = topk_indices.contiguous().reshape(TK)

    expert_token_count = torch.zeros(E, dtype=torch.int32, device=device)
    expert_start_idx = torch.empty(E + 1, dtype=torch.int32, device=device)
    expert_tile_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    if E == 0:
        expert_start_idx.zero_()
        expert_tile_offset.zero_()
    else:
        nprog = min(get_npu_core_count(), TK) if TK > 0 else 1
        if TK > 0:
            _moe_router_histogram_kernel[(nprog,)](topk, expert_token_count, TK, E)
        _moe_router_prefix_sum_kernel[(1,)](
            expert_token_count,
            expert_start_idx,
            expert_tile_offset,
            E,
            BLOCK_M_TOKEN=block_m_token,
        )
        if TK > 0:
            cursor = torch.zeros(E, dtype=torch.int32, device=device)
            _moe_router_scatter_kernel[(nprog,)](
                topk,
                cursor,
                expert_start_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
                x_gather_idx,
                TK,
                K,
                E,
            )

    # Exact tile count stays on device. This is a host upper bound (slack < E).
    # Padding tiles start at row TK, past every expert, so the existing row mask drops them.
    num_m_tiles = 0 if TK <= 0 or E <= 0 else min(TK, (TK + E * (block_m_token - 1)) // block_m_token)
    tile_row_start = torch.full((num_m_tiles,), TK, dtype=torch.int32, device=device)
    tile_expert = torch.zeros(num_m_tiles, dtype=torch.int32, device=device)
    if num_m_tiles > 0:
        _moe_router_tile_kernel[(E,)](
            expert_start_idx,
            expert_tile_offset,
            tile_row_start,
            tile_expert,
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


def _row_launch_chunks(num_m_tiles, n_n_tiles):
    """One program per (tile, row) so tl.dot sees a single live row."""
    n_row = num_m_tiles * BLOCK_M_TOKEN
    step = max(1, ASCEND_MAX_GRID_PROGRAMS // max(int(n_n_tiles), 1))
    for base in range(0, n_row, step):
        yield min(step, n_row - base), base


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

    Forward: routing metadata → gather+GEMM → SwiGLU epilogue → down-proj → token aggregation
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
                H_dim=H,
                I_dim=intermediate_dim,
                stride_x_T=x.stride(0),
                stride_x_H=x.stride(1),
                stride_w_E=gate_up_proj.stride(0),
                stride_w_N=gate_up_proj.stride(1),
                stride_w_K=gate_up_proj.stride(2),
                stride_pre_TK=pre_act.stride(0),
                stride_pre_N=pre_act.stride(1),
                BLOCK_M=BLOCK_M_TOKEN,
                BLOCK_N=ASCEND_GEMM_BLOCK_N,
                BLOCK_K=ASCEND_GEMM_BLOCK_K,
            )
            _swiglu_from_pre_act_kernel[(min(TK, ASCEND_MAX_GRID_PROGRAMS),)](
                pre_act,
                post_act,
                TK,
                I_dim=intermediate_dim,
                stride_pre_TK=pre_act.stride(0),
                stride_pre_N=pre_act.stride(1),
                stride_post_TK=post_act.stride(0),
                stride_post_N=post_act.stride(1),
                BLOCK_N=ASCEND_GEMM_BLOCK_N,
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
            for m_count, m_off in _row_launch_chunks(num_m_tiles, n_i_tiles):
                _moe_bwd_down_proj_kernel[(m_count, n_i_tiles)](
                    dO,
                    x_gather_idx,
                    s_scatter_idx,
                    topk_weights_flat,
                    down_proj,
                    pre_act,
                    expert_start_idx,
                    tile_row_start,
                    tile_expert,
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
                    pid_base=m_off,
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
            n_h_tiles = triton.cdiv(H, ASCEND_GEMM_BLOCK_N)
            for m_count, m_off in _row_launch_chunks(num_m_tiles, n_h_tiles):
                _moe_bwd_dX_expanded_kernel[(m_count, n_h_tiles)](
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
                    pid_base=m_off,
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
