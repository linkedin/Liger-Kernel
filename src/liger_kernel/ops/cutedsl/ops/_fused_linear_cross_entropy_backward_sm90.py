"""Hopper (SM90) CuTe DSL fused-linear-cross-entropy **backward** GEMMs.

The FLCE gradients follow the "gradient-in-forward" contract used by the Triton
``LigerFusedLinearCrossEntropyFunction`` and by QuACK's
``chunked_linear_cross_entropy``: the autograd *forward* has already produced

    ``dZ[M, V] = (softmax(X @ W.T) - onehot(target)) * row_scale``

(``row_scale`` folds the ``mean``/``sum`` reduction in), and this module turns
that into the two parameter gradients with the shared WGMMA GEMM primitive
:class:`~liger_kernel.ops.cutedsl.ops._fused_linear_cross_entropy_forward_sm90._TileGemmSM90`:

    ``dX[M, H] = dZ[M, V] @ W[V, H]``      (A K-major over V, B MN-major over H)
    ``dW[V, H] = dZ.T[V, M] @ X[M, H]``    (both operands MN-major)

Both are expressed as ``C = A @ B.T`` by passing transposed *views* (never
materialised transposes) of ``dZ``/``W``/``X`` and telling the loader which dim
of each view is contiguous. The dX GEMM runs in autograd forward, while dW is
deferred to autograd backward so its epilogue can apply the upstream scale.
"""

import cuda.bindings.driver as cuda
import torch

from liger_kernel.ops.cutedsl.ops._fused_linear_cross_entropy_forward_sm90 import tile_gemm


def flce_dx(dz, weight, out_dtype=torch.bfloat16, output_scale=None):
    """``dX[M, H] = dZ[M, V] @ W[V, H]``.

    ``dz[M, V]`` is BF16 row-major (V contiguous).  ``weight[V, H]`` is BF16
    row-major (H contiguous); its transpose view ``[H, V]`` is passed as the
    MN-major B operand so ``C = A @ B.T = dZ @ W``.
    """
    with torch.cuda.device(dz.device):
        m, v = dz.shape
        h = weight.shape[1]
        dx = torch.empty(m, h, device=dz.device, dtype=out_dtype)
        weight_t = weight.transpose(0, 1)  # [H, V] view, H contiguous
        stream = cuda.CUstream(torch.cuda.current_stream(dz.device).cuda_stream)
        # A = dZ[M, V], K = V contiguous -> leading_dim 1
        # B = W.T[H, V], H contiguous     -> leading_dim 0
        tile_gemm(
            dz,
            weight_t,
            dx,
            a_leading=1,
            b_leading=0,
            stream=stream,
            output_scale=output_scale,
        )
        return dx


def flce_dw(dz, x, out_dtype=torch.bfloat16, output_scale=None):
    """``dW[V, H] = dZ.T[V, M] @ X[M, H]``.

    Both operands are MN-major transposed views: ``dZ.T[V, M]`` (V contiguous)
    and ``X.T[H, M]`` (H contiguous), so ``C = A @ B.T = dZ.T @ X``.
    """
    import os

    with torch.cuda.device(dz.device):
        m, v = dz.shape
        h = x.shape[1]
        dw = torch.empty(v, h, device=dz.device, dtype=out_dtype)
        dz_t = dz.transpose(0, 1)  # [V, M] view, V contiguous
        x_t = x.transpose(0, 1)  # [H, M] view, H contiguous
        stream = cuda.CUstream(torch.cuda.current_stream(dz.device).cuda_stream)
        # A = dZ.T[V, M], V contiguous -> leading_dim 0
        # B = X.T[H, M],  H contiguous -> leading_dim 0
        # dW scheduling. The default clustered persistent helper is selected inside
        # ``tile_gemm`` and fuses ``output_scale`` into its TMA-store epilogue.
        # ``FLCE_DW_CLUSTERED_PERSISTENT=0`` restores this static fallback.
        # ``swap_grid=True`` transposes the fallback launch grid so the GPU's
        # x-fastest CTA order makes consecutive CTAs share the same A tile (an M-tile
        # of dZ.T) across all N columns; that A tile is fetched from DRAM once and
        # reused out of L2, and the 32 MB B operand stays L2-resident.  This static
        # raster already delivers the QuACK-style L2 locality (77% L2 hit, 90.9%
        # tensor-pipe active, 6.47 ms NCU on M=4096,H=4096,V=128256) at zero extra
        # complexity.
        #
        # The older unclustered one-wave scheduler (``persistent=True`` -> the
        # ``kernel_persistent`` path with a continuous TMA pipeline + async TMA-store
        # epilogue) reaches an even better memory profile (81.8% L2 hit, 19% DRAM,
        # waves=1 -- matching QuACK's) but is net *slower* (7.07 ms, 83% tensor): its
        # shared-accumulator epilogue barriers block the single warp that also drives
        # the TMA producer, so the tensor cores idle ~17% at every tile boundary.
        # Beating the static raster from here needs a warp-specialized,
        # double-buffered-accumulator epilogue (dedicate a warp-group to the store so
        # the MMA WGs run the next tile uninterrupted); that path is left behind the
        # ``FLCE_DW_PERSISTENT=1`` env toggle for future work.  Default: static raster.
        persistent = os.environ.get("FLCE_DW_PERSISTENT", "0") != "0"
        tile_gemm(
            dz_t,
            x_t,
            dw,
            a_leading=0,
            b_leading=0,
            stream=stream,
            swap_grid=True,
            output_scale=output_scale,
            persistent=persistent,
        )
        return dw


__all__ = ["flce_dx", "flce_dw"]
