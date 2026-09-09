"""CuTe DSL (CUTLASS python) backend for ``rms_norm``.

Strategy
--------
Mirrors the sibling :mod:`liger_kernel.ops.backends._cutedsl.layer_norm`:

- **Forward** delegates to ``_cute_lib.rmsnorm_fwd`` — a self-contained
  inline CuTe DSL kernel adapted from Quack (Apache-2.0; see
  ``_cute_lib/NOTICE.md``). It carries a per-process ``cute.compile``
  cache so repeated calls reuse the compiled artifact.

- **Backward** uses an inline CuTe DSL kernel defined locally
  (``_LigerRMSNormCuTeDSLBackward``). It mirrors Quack's
  ``RMSNormBackward`` (one statistic — ``c = mean(wdy * x_hat)``) rather
  than the two-stat LayerNorm reduction. Per-SM ``(sm_count, N)`` fp32 ``dW``
  partials are reduced cross-SM on the **host** via ``.sum(dim=0)`` — the
  same pattern Liger uses in its Triton and cuTile backends. Compiled
  backward kernels are cached locally keyed by
  ``(input_dtype, weight_dtype, dx_dtype, N, has_weight)``.

Math (matches the Triton kernel in :mod:`liger_kernel.ops.rms_norm`)::

    rstd   = 1 / sqrt(mean(x^2) + eps)
    y      = (x * rstd) * (offset + w)
    w_eff  = w + offset                     (host-side, cheap)
    wdy    = (offset + w) * dy              (fp32)
    c      = mean(wdy * (x * rstd))
    dx     = rstd * (wdy - (x * rstd) * c)
    dW    += dy * (x * rstd)                (fp32, per-row)

Liger's RMSNorm has three casting modes which the inline forward kernel
does not directly expose; we adapt them as follows:

- ``"llama"`` (default): pass ``x`` and ``weight + offset`` to the kernel
  as-is. The kernel runs the reduction in fp32 internally; the final
  weight multiply happens at output dtype. This matches Triton.
- ``"gemma"``: pre-cast ``x`` to fp32 host-side; the kernel's output is
  then fp32 and we cast the result back to the input dtype after the
  kernel returns. Matches Triton's "everything in fp32, cast result back"
  path.
- ``"none"``: the kernel always promotes to fp32 internally, so a truly
  lossy "no casting" variant is not reachable. We still accept the mode
  for API parity, but the math runs through the same code path as
  ``"llama"`` — slightly more precise than the Triton ``none`` mode.

``offset`` is folded into the weight host-side (``weight + offset``)
before the kernel call — the inline forward has no offset argument, but
this single fused pre-add is far cheaper than the launch itself.

The backward uses the **effective** weight ``w + offset`` (we save it in
ctx). This matches Triton's ``W_row + offset`` line inside the bwd kernel.

Capability
----------
- Compute capability >= sm_90 (Hopper or newer).
- Requires only the ``cutlass`` Python package (the CuTe DSL utilities are
  inlined under ``_cute_lib/``; no runtime dependency on Quack).

Shapes and limits
-----------------
- ``x`` is flattened to 2D (``view(-1, N)``); arbitrary leading dims OK.
- Backward kernel is capped at ``N <= 8192``: beyond that the bwd
  triggers the cluster-reduce path which uses ``cute.arch.ProxyKind`` —
  a symbol that's not present on the cuda13.2 cutlass-cute wheels we
  install on B200 today. The test harness auto-skips when this raises.
- dtypes: fp16, bf16, fp32 for ``x`` and ``weight``.

Notes
-----
We deliberately do **not** put ``from __future__ import annotations`` here
(same reasoning as the LayerNorm sibling: keeps annotations live for
DSL introspection).

References
----------
- Quack (adapted under Apache-2.0): ``Dao-AILab/quack`` — ``quack/rmsnorm.py``
  is the source of the forward ``RMSNorm`` class, the ``_compile_rmsnorm_fwd``
  cache pattern, and the dW host-side ``.sum(0)`` pattern. See
  :mod:`._cute_lib` for the inlined copy.
  https://github.com/Dao-AILab/quack
- Triton reference: :mod:`liger_kernel.ops.rms_norm` — fixes the exact
  numerics we must reproduce (casting modes, offset semantics, dW row
  accumulation).
"""

import math

from functools import partial
from typing import Optional
from typing import Tuple
from typing import Type

# ---------------------------------------------------------------------------
# Top-level CuTe DSL imports. Identical pattern to the LayerNorm sibling
# (see its header for the rationale).
# ---------------------------------------------------------------------------
import cuda.bindings.driver as cuda  # noqa: F401  (referenced by cute.compile)
import cutlass
import cutlass.cute as cute
import torch

from cutlass import Float32
from cutlass import Int32
from cutlass import const_expr
from torch import Tensor

# Inlined CuTe DSL utilities (adapted from Quack, Apache-2.0). Importing
# ``_cute_lib`` does **not** require the upstream ``quack`` package.
import liger_kernel.ops.backends._cutedsl._cute_lib.copy_utils as copy_utils
import liger_kernel.ops.backends._cutedsl._cute_lib.layout_utils as layout_utils
import liger_kernel.ops.backends._cutedsl._cute_lib.utils as utils

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.backends.dispatch import emit_fallback_warning
from liger_kernel.ops._nvidia_shared import CASTING_MODE_GEMMA as _CASTING_MODE_GEMMA
from liger_kernel.ops._nvidia_shared import CASTING_MODE_LLAMA as _CASTING_MODE_LLAMA
from liger_kernel.ops._nvidia_shared import STR_TO_CASTING_MODE as _str_to_casting_mode
from liger_kernel.ops._nvidia_shared import to_local_if_dtensor as _to_local_if_dtensor
from liger_kernel.ops.backends._cutedsl._cute_lib.compile_utils import make_fake_tensor as fake_tensor
from liger_kernel.ops.backends._cutedsl._cute_lib.dtype_map import torch2cute_dtype_map
from liger_kernel.ops.backends._cutedsl._cute_lib.reduce import row_reduce
from liger_kernel.ops.backends._cutedsl._cute_lib.reduction_base import ReductionBase
from liger_kernel.ops.backends._cutedsl._cute_lib.rmsnorm_fwd import rmsnorm_fwd as _cutedsl_rmsnorm_fwd
from liger_kernel.ops.utils import device_context

# Same ceiling as the LayerNorm sibling — beyond this the kernel triggers
# the cluster-reduce path which uses ``cute.arch.ProxyKind`` (missing on
# the cuda13.2 cutlass-cute wheels on B200 today). Surface a RuntimeError
# so the test harness auto-skips for wider rows.
_BWD_MAX_TILE_CUTEDSL = 32768


# ===========================================================================
# Backward kernel — inline CuTe DSL
#
# Single-statistic variant of Quack's RMSNormBackward: we only need
# ``c = mean(wdy * x_hat)`` per row (LayerNorm needs ``c1`` AND ``c2``).
# ===========================================================================
class _LigerRMSNormCuTeDSLBackward(ReductionBase):
    """CuTe DSL implementation of RMSNorm backward.

    Layout: one CTA processes ``rows_per_program`` consecutive rows. Grid is
    ``(sm_count, cluster_n)``; each SM holds a per-SM ``(N,)`` fp32 partial
    for ``dW`` that the host reduces post-launch.

    The reduction-buffer slot count (``stage=2``) is intentional for
    double-buffering across rows even though we only reduce one statistic
    per row — matches Quack's structure.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        has_weight: bool = True,
        casting_mode: int = _CASTING_MODE_LLAMA,
    ):
        super().__init__(dtype, N, stage=2, reduction_dtype=Float32)
        self.has_weight = has_weight
        self.casting_mode = casting_mode
        # Beyond 16K we reload wdy from smem instead of holding the
        # fragment live — matches Quack's ``RMSNormBackward.reload_wdy``.
        self.reload_wdy = None if N <= 16 * 1024 else "smem"
        if self.N > 128 * 1024 and self.dtype.width >= 32:
            raise ValueError(
                "RMSNormBackward does not support N > 128k with dtype >= 32 bits (register file pressure)."
            )

    def _num_threads(self):
        return 128 if self.N <= 4096 else 256

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (256, 32), (512, 64), (4096, 128)]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        N = self.N
        for limit, cluster in [(8 * 1024, 1), (16 * 1024, 2), (32 * 1024, 4), (64 * 1024, 8)]:
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = 16

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # Input [M, N]
        mW: Optional[cute.Tensor],  # Effective weight (w + offset) [N,] or None
        mdO: cute.Tensor,  # dY [M, N]
        mRstd: cute.Tensor,  # RSTD [M,] (fp32)
        mdX: cute.Tensor,  # dX [M, N]
        mdW: Optional[cute.Tensor],  # dW partial [sm_count, N] fp32
        sm_count: Int32,
        mdS: Optional[cute.Tensor] = None,  # Residual grad dS_out [M, N] — FARN epilogue fold
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()

        largest_dtype_width = const_expr(max(*(t.element_type.width for t in [mX, mW, mdO, mdX, mdS] if t is not None)))
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)

        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size

        mW = layout_utils.expand(mW, dim=0, size=tiler_mn[0]) if const_expr(mW is not None) else None

        num_blocks = sm_count

        self.kernel(
            mX,
            mW,
            mdO,
            mRstd,
            mdX,
            mdW,
            tiler_mn,
            tiled_copy,
            threads_per_row,
            mdS,
        ).launch(
            grid=[num_blocks, self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if self.cluster_n > 1 else None,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mdO: cute.Tensor,
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: Optional[cute.Tensor],
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
        mdS: Optional[cute.Tensor] = None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_start, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mX.shape
        M = shape[0]
        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)

        idX = cute.make_identity_tensor(shape)

        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_ordered_layout((tiler_mn[0], tiler_mn[1], 2), order=(1, 0, 2))
        sX = smem.allocate_tensor(mX.element_type, smem_layout, byte_alignment=16)
        sdO = smem.allocate_tensor(mdO.element_type, smem_layout, byte_alignment=16)
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout, is_persistent=True)
        if const_expr(mbar_ptr is not None):
            mbar_full_ptr, mbar_empty_ptr = mbar_ptr, mbar_ptr + 2
        else:
            mbar_full_ptr, mbar_empty_ptr = None, None

        thr_copy_X = tiled_copy.get_slice(tidx)

        gX, gdO, gdX, cX = [
            cute.local_tile(mT, tiler_mn, (None, cluster_y)) if mT is not None else None for mT in (mX, mdO, mdX, idX)
        ]
        gW = cute.local_tile(mW, tiler_mn, (0, cluster_y)) if mW is not None else None
        gdW = cute.local_tile(mdW, (1, tiler_mn[1]), (bidx_start, cluster_y)) if const_expr(mdW is not None) else None
        gdS = cute.local_tile(mdS, tiler_mn, (None, cluster_y)) if const_expr(mdS is not None) else None

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgdO = thr_copy_X.partition_S(gdO)
        tXsdO = thr_copy_X.partition_D(sdO)
        tXgdX = thr_copy_X.partition_D(gdX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None, None]

        tXrX, tXrdO, tXrdX = [cute.make_fragment_like(thr[None, None, None, 0]) for thr in (tXgX, tXgdO, tXgdX)]

        tXpX = None if is_even_N else copy_utils.predicate_k(thr_copy_X.partition_S(cX[None, None, 0]), limit=shape[1])
        copy_pred = partial(copy_utils.copy, pred=tXpX)

        tXgdW, tXrdW = None, None
        if const_expr(mdW is not None):
            tXgdW = thr_copy_X.partition_S(gdW)
            # Per-SM dW partials always accumulate in fp32 for numerical
            # stability — matches Liger Triton's ``_dW``.
            tXrdW = cute.make_fragment_like(tXgdW, Float32)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE

        self._initialize_cluster(tidx, mbar_ptr, num_warps, is_persistent=True)

        tXrW = None
        if const_expr(mW is not None):
            tXgW = thr_copy_X.partition_S(gW)
            tXrW = cute.make_fragment_like(tXgW)
            if const_expr(not is_even_N):
                tXrW.fill(0.0)
            copy_pred(tXgW, tXrW)

        tXgdS, tXrdS = None, None
        if const_expr(mdS is not None):
            tXgdS = thr_copy_X.partition_S(gdS)
            tXrdS = cute.make_fragment_like(tXrX)

        # Prefetch the first batch of (X, dO).
        row = tXcX[None, None, None, bidx_start][0][0]
        if row < M:
            copy_pred(
                tXgX[None, None, None, bidx_start],
                tXsX[None, None, None, 0],
                is_async=True,
            )
            copy_pred(
                tXgdO[None, None, None, bidx_start],
                tXsdO[None, None, None, 0],
                is_async=True,
            )
        else:
            if const_expr(tiler_mn[0] > 1):
                utils.fill_oob(tXsX[None, None, None, 0], None, fill_value=mX.element_type.zero)
                utils.fill_oob(tXsdO[None, None, None, 0], None, fill_value=mdO.element_type.zero)
        cute.arch.cp_async_commit_group()

        if const_expr(self.cluster_n > 1):
            cute.arch.cluster_wait()

        if const_expr(mdW is not None):
            tXrdW.fill(0.0)

        stage = Int32(0)
        producer_phase = Int32(1)
        consumer_phase = Int32(0)

        for bidx in cutlass.range(bidx_start, cute.ceil_div(M, tiler_mn[0]), gdim):
            row = tXcX[None, None, None, bidx][0][0]

            # Prefetch next batch.
            if row + gdim * tiler_mn[0] < M:
                copy_pred(
                    tXgX[None, None, None, bidx + gdim],
                    tXsX[None, None, None, stage ^ 1],
                    is_async=True,
                )
                copy_pred(
                    tXgdO[None, None, None, bidx + gdim],
                    tXsdO[None, None, None, stage ^ 1],
                    is_async=True,
                )
            else:
                if const_expr(tiler_mn[0] > 1):
                    utils.fill_oob(
                        tXsX[None, None, None, stage ^ 1],
                        None,
                        fill_value=mX.element_type.zero,
                    )
                    utils.fill_oob(
                        tXsdO[None, None, None, stage ^ 1],
                        None,
                        fill_value=mdO.element_type.zero,
                    )
            cute.arch.cp_async_commit_group()

            rstd_val = cutlass.Float.zero
            if row < M or tiler_mn[0] == 1:
                rstd_val = mRstd[row]

            cute.arch.cp_async_wait_group(1)
            cute.autovec_copy(tXsX[None, None, None, stage], tXrX)
            x = tXrX.load().to(cute.Float32)
            cute.autovec_copy(tXsdO[None, None, None, stage], tXrdO)
            dout = tXrdO.load().to(cute.Float32)

            # RMSNorm: x_hat = x * rstd (no mean subtraction).
            x_hat = x * rstd_val

            wdy = dout
            if const_expr(mW is not None):
                # mW already contains (offset + weight) — folded host-side.
                wdy = wdy * tXrW.load().to(Float32)

            if const_expr(self.cluster_n > 1):
                cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)

            # Single reduction: c = mean(wdy * x_hat)
            mean_xhat_wdy = (
                row_reduce(
                    x_hat * wdy,
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, stage],
                    (mbar_full_ptr + stage if const_expr(self.cluster_n > 1) else None),
                    phase=consumer_phase,
                    init_val=0.0,
                )
                / shape[1]
            )

            if const_expr(self.cluster_n > 1):
                # Use ``fence_view_async_shared()`` (present in all cutlass-cute
                # wheels) instead of the older ``fence_proxy(ProxyKind.async_shared)``
                # which is absent from the cuda13.2 wheels we ship today.
                # Quack's RMSNormBackward uses this exact form.
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                lane_idx = cute.arch.lane_idx()
                if lane_idx < self.cluster_n:
                    cute.arch.mbarrier_arrive(mbar_empty_ptr + stage, peer_cta_rank_in_cluster=lane_idx)

            if const_expr(self.reload_wdy == "smem"):
                cute.autovec_copy(tXsdO[None, None, None, stage], tXrdO)
                dout = tXrdO.load().to(cute.Float32)
                wdy = dout
                if const_expr(mW is not None):
                    wdy = wdy * tXrW.load().to(Float32)

            # dx = rstd * (wdy - x_hat * c)
            dx = (wdy - x_hat * mean_xhat_wdy) * rstd_val

            if const_expr(mdS is not None):
                # FusedAddRMSNorm epilogue fold: dx += dS. Adding in fp32
                # before the single store round is strictly closer to exact
                # than the host's post-hoc two-tensor bf16 add it replaces
                # (round(a+b) vs round(round(a)+round(b))), and removes that
                # host pass's whole (M,N) read+write trip (~0.06ms at 8192^2
                # bf16 — the measured gap in the hd>4096 bwd cliff).
                copy_pred(tXgdS[None, None, None, bidx], tXrdS)
                # ``to()`` of a no-op dtype (fp32->fp32) returns the value
                # unchanged; a 2-byte cast is needed because ``TensorSSA +
                # element_type`` (round_to_nearest) is a binary_Elementwise
                # without a .to() chain.
                dx = dx + tXrdS.load().to(cute.Float32)

            tXrdX.store(dx.to(tXrdX.element_type))
            if row < M or tiler_mn[0] == 1:
                copy_pred(tXrdX, tXgdX[None, None, None, bidx])

            # Per-SM dW partial: dW += dY * x_hat. We use the raw ``dout``
            # (input dtype upcast to fp32), NOT the effective ``wdy``, so the
            # formula matches Liger Triton: ``dW_row += dY_row * (X_row * rstd)``.
            if const_expr(mdW is not None):
                x_hat_dw = (
                    x_hat.to(self.dtype).to(Float32) if const_expr(self.casting_mode == _CASTING_MODE_LLAMA) else x_hat
                )
                tXrdW.store(tXrdW.load() + dout * x_hat_dw)

            stage ^= 1
            if stage == 0:
                consumer_phase ^= 1
                producer_phase ^= 1

        # Reduce per-thread partials within the CTA: row 0 collects the
        # other rows' partials from smem and writes the final per-SM partial
        # to gmem. Identical scheme to LayerNorm / Quack RMSNormBackward.
        if const_expr(tiler_mn[0] > 1):
            if const_expr(mdW is not None):
                sdW = cute.make_tensor(
                    cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                )
                tXsdW = thr_copy_X.partition_D(sdW)
                cute.arch.barrier()
                row_in_tile = tXcX[None, None, None, 0][0][0]
                if row_in_tile > 0:
                    cute.autovec_copy(tXrdW, tXsdW)
                cute.arch.barrier()
                if row_in_tile == 0:
                    for i in cutlass.range_constexpr(1, const_expr(tiler_mn[0])):
                        tXrdW_other = cute.make_fragment_like(tXrdW)
                        tXsdW_other = cute.make_tensor(tXsdW.iterator + i * sdW.stride[0], tXsdW.layout)
                        cute.autovec_copy(tXsdW_other, tXrdW_other)
                        tXrdW.store(tXrdW.load() + tXrdW_other.load())
                    copy_pred(tXrdW, tXgdW)
                cute.arch.barrier()
        else:
            if const_expr(mdW is not None):
                copy_pred(tXrdW, tXgdW)

        if const_expr(self.cluster_n > 1):
            stage ^= 1
            if stage == 0:
                producer_phase ^= 1
            cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)


# ---------------------------------------------------------------------------
# Backward compile cache. ``cute.compile()`` is multi-second; we key on the
# minimal set of attributes that change codegen.
# ---------------------------------------------------------------------------
_BWD_COMPILE_CACHE: dict = {}


def _bwd_sm_count(N: int, device: torch.device) -> int:
    """SM-count heuristic mirroring Quack's ``_get_sm_count`` (RMSNorm
    variant). Identical to the LayerNorm sibling's helper but inlined here
    so the two backends stay decoupled.
    """
    sm_count_multiple = 16 if N <= 256 else (8 if N <= 1024 else (4 if N <= 2048 else (2 if N <= 4096 else 1)))
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    if N <= 8192:
        return sm_count * sm_count_multiple
    if N <= 16384:
        return sm_count // 2
    return sm_count * 2


def _get_bwd_kernel(
    x_dtype: torch.dtype,
    weight_dtype: Optional[torch.dtype],
    dx_dtype: torch.dtype,
    N: int,
    has_weight: bool,
    casting_mode: int,
    has_ds: bool = False,
):
    """Return a compiled backward kernel, building it on first miss."""
    key = (x_dtype, weight_dtype, dx_dtype, N, has_weight, casting_mode, has_ds)
    if key in _BWD_COMPILE_CACHE:
        return _BWD_COMPILE_CACHE[key]

    dtype = torch2cute_dtype_map[x_dtype]
    dx_cute_dtype = torch2cute_dtype_map[dx_dtype]
    weight_cute_dtype = torch2cute_dtype_map[weight_dtype] if weight_dtype is not None else None

    batch_sym = cute.sym_int()
    all_dtypes = [d for d in [dtype, dx_cute_dtype, weight_cute_dtype] if d is not None]
    div = math.gcd(N, *(128 // d.width for d in all_dtypes))

    x_cute = fake_tensor(dtype, (batch_sym, N), div)
    dout_cute = fake_tensor(dtype, (batch_sym, N), div)
    weight_cute = fake_tensor(weight_cute_dtype, (N,), div) if weight_cute_dtype is not None else None
    rstd_cute = fake_tensor(Float32, (batch_sym,))
    dx_cute = fake_tensor(dx_cute_dtype, (batch_sym, N), div)

    sm_sym = cute.sym_int()
    dw_cute = fake_tensor(Float32, (sm_sym, N), div) if has_weight else None
    ds_cute = fake_tensor(dtype, (batch_sym, N), div) if has_ds else None

    kernel = _LigerRMSNormCuTeDSLBackward(
        dtype,
        N,
        has_weight=has_weight,
        casting_mode=casting_mode,
    )
    compiled = cute.compile(
        kernel,
        x_cute,
        weight_cute,
        dout_cute,
        rstd_cute,
        dx_cute,
        dw_cute,
        Int32(0),
        ds_cute,
        options="--enable-tvm-ffi",
    )
    _BWD_COMPILE_CACHE[key] = compiled
    return compiled


# ===========================================================================
# Host-side launchers and autograd Function
# ===========================================================================
def _rms_norm_cutedsl_forward(
    x: Tensor,
    weight: Optional[Tensor],
    eps: float,
    offset: float,
    casting_mode: int,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor]:
    """Forward via the inlined ``_cute_lib.rmsnorm_fwd``.

    Returns ``(Y, X_flat, W_eff, RSTD)``. ``W_eff`` is the effective weight
    ``weight + offset`` (or ``None`` if ``weight is None``), kept around for
    the inline backward. ``RSTD`` is fp32 (the kernel always produces it
    in fp32 when ``store_rstd=True``).
    """
    shape = x.shape
    N = shape[-1]

    # The Quack fwd kernel promotes x/w to fp32 in-register
    # (``tXrX.load().to(cute.Float32)`` in ``_cute_lib/rmsnorm_fwd.py``) and
    # bf16/fp16->fp32 up-casts are EXACT, so for uniform-dtype gemma calls the
    # host-side fp32 promotion changes only HOW the kernel's fp32 stream is
    # produced, not its values: the in-register fp32 tensor is the same either
    # way. It costs a full (M,N) fp32 copy of x, a 2x wider x read inside the
    # kernel, an (M,N) fp32 round trip for out, plus (via the backward's
    # ``needs_dtype_bridge``) a full dy up-cast and an fp32 dx round trip.
    # ``casting_mode="none"`` already runs exactly this no-promotion path.
    # Keep the host promotion ONLY for mixed-dtype gemma calls (e.g. bf16 x
    # with an fp32 weight), matching Triton's forced-fp32 gemma math.
    _promote_gemma_fp32 = (
        casting_mode == _CASTING_MODE_GEMMA
        and x.dtype != torch.float32
        and not (x.dtype in (torch.float16, torch.bfloat16) and (weight is None or weight.dtype == x.dtype))
    )

    # Casting-mode handling: only mixed-dtype gemma calls pre-cast x to fp32
    # host-side. llama / none / uniform-dtype gemma pass straight through —
    # the kernel already promotes to fp32 in-register.
    if _promote_gemma_fp32:
        x_in = x.to(torch.float32)
    else:
        x_in = x
    x_flat = x_in.view(-1, N).contiguous()

    # Fold offset into the weight host-side — Quack has no offset arg, but
    # one extra elementwise add on a (N,) vector is essentially free.
    if weight is not None:
        if offset != 0.0:
            w_eff = weight + offset
        else:
            # Avoid copying when offset is the common 0.0 — Quack accepts
            # the original weight directly.
            w_eff = weight
        # Only the mixed-dtype gemma route promotes w_eff host-side;
        # otherwise the kernel up-casts in-register (bit-exact).
        if _promote_gemma_fp32 and w_eff.dtype != torch.float32:
            w_eff = w_eff.to(torch.float32)
        w_eff = w_eff.contiguous()
    else:
        w_eff = None

    out, _residual_out, rstd = _cutedsl_rmsnorm_fwd(
        x_flat,
        weight=w_eff,
        eps=eps,
        store_rstd=True,
    )

    # Back-cast if the promote route above widened the output (it is the
    # only path that can produce an out dtype != x.dtype).
    if out.dtype != x.dtype:
        out = out.to(x.dtype)

    return out.view(shape), x_flat, w_eff, rstd


def _rms_norm_cutedsl_backward(
    dy: Tensor,
    x_flat: Tensor,
    w_eff: Optional[Tensor],
    rstd: Tensor,
    in_place: bool,
    casting_mode: int,
    ds: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Backward via the inline CuTe DSL kernel.

    ``dy`` is reshaped to 2D to match ``x_flat``. Per-SM dW partials are
    allocated as fp32 ``(sm_count, N)`` and reduced to the final ``dW``
    host-side. This matches Liger's Triton and cuTile backends.

    ``ds`` (optional) is the FusedAddRMSNorm residual-grad stream: when
    given, ``dx += ds`` is folded into the kernel epilogue in fp32 (a
    compile-keyed extra [M, N] input, like the mW/mdW Optionals), replacing
    the caller's host-side add pass. Plain rms_norm passes ``None`` —
    the absent-arg specialization compiles dead-code-eliminated to the
    same kernel body as before the fold was added.
    """
    with device_context(x_flat.device):
        shape = dy.shape
        N = shape[-1]
        if N > _BWD_MAX_TILE_CUTEDSL:
            raise RuntimeError(
                f"cuTeDSL rms_norm backward only supports hidden dim <= "
                f"{_BWD_MAX_TILE_CUTEDSL}; got {N}. Use backend='triton' for "
                f"wider rows. (Cluster-reduce path requires a newer cutlass-cute.)"
            )
        dy_flat = dy.view(-1, N).contiguous()
        M = dy_flat.shape[0]

        # The mixed-dtype gemma route pre-casts x to fp32 in the forward, so
        # x_flat.dtype can be fp32 while dy comes in at the original input dtype
        # (uniform-dtype gemma no longer bridges — see the forward). The
        # compiled kernel is keyed by x_flat.dtype, so all in/out tensors it
        # touches must match. Promote dy and (later) demote dx accordingly.
        original_dy_dtype = dy_flat.dtype
        needs_dtype_bridge = dy_flat.dtype != x_flat.dtype
        if needs_dtype_bridge:
            dy_flat = dy_flat.to(x_flat.dtype)

        # ``in_place`` lets the caller reuse dY's storage as dX — Liger Triton
        # does the same. Safe here because each row is processed atomically by
        # one block; no aliasing tile is read after being written.
        if in_place and not needs_dtype_bridge:
            dx = dy_flat
        else:
            # Allocate in x_flat's dtype so the kernel signature matches.
            dx = torch.empty(dy_flat.shape, dtype=x_flat.dtype, device=dy_flat.device)

        sm_count = _bwd_sm_count(N, x_flat.device)
        # Saturate the grid: never launch more SMs than rows.
        sm_count = min(sm_count, max(M, 1))

        has_weight = w_eff is not None
        dw_partial = torch.empty((sm_count, N), dtype=torch.float32, device=x_flat.device) if has_weight else None

        ds_flat = None
        if ds is not None:
            ds_flat = ds.view(-1, N).contiguous()
            # Must match the kernel's compile-keyed dtype stream (x_flat.dtype);
            # FARN keeps dy-store dtype == x dtype through the gemma route, but
            # bridge defensively the same way dy is bridged above.
            if ds_flat.dtype != x_flat.dtype:
                ds_flat = ds_flat.to(x_flat.dtype)

        compiled = _get_bwd_kernel(
            x_dtype=x_flat.dtype,
            weight_dtype=w_eff.dtype if w_eff is not None else None,
            dx_dtype=dx.dtype,
            N=N,
            has_weight=has_weight,
            casting_mode=casting_mode,
            has_ds=ds_flat is not None,
        )

        # CuTe DSL compiled kernels read torch.cuda.current_stream() at launch;
        # the kernel ABI does NOT take a stream positional. Mirrors the
        # LayerNorm sibling's launch call.
        compiled(
            x_flat,
            w_eff,
            dy_flat,
            rstd,
            dx,
            dw_partial,
            sm_count,
            ds_flat,
        )

        if has_weight:
            dw = dw_partial.sum(dim=0)
        else:
            dw = None

        # Cast dx back to the autograd-graph's original dtype (gemma promoted).
        if needs_dtype_bridge:
            dx = dx.to(original_dy_dtype)

        return dx.view(shape), dw


class _LigerRMSNormCuTeDSLFunction(torch.autograd.Function):
    """Autograd wrapper. Saves enough state for the inline-kernel backward.

    We save ``x_flat`` (the 2D view passed to the kernel; fp32 when the
    mixed-dtype gemma route promoted host-side) and ``w_eff`` (``weight + offset``),
    so the backward doesn't redo any of the host-side pre-processing.
    """

    @staticmethod
    def forward(ctx, X, W, eps, offset, casting_mode, in_place, row_mode):
        # row_mode is the legacy Triton tuning knob; CuTe DSL ignores it.
        del row_mode

        X = _to_local_if_dtensor(X)
        if W is not None:
            W = _to_local_if_dtensor(W)

        X = X.contiguous()
        if W is not None:
            W = W.contiguous()

        # Normalize casting_mode to an int once, here.
        if not isinstance(casting_mode, int):
            if casting_mode not in _str_to_casting_mode:
                raise ValueError(f"Invalid casting mode: {casting_mode}")
            casting_mode_int = _str_to_casting_mode[casting_mode]
        else:
            casting_mode_int = casting_mode

        Y, X_flat, W_eff, RSTD = _rms_norm_cutedsl_forward(X, W, eps, offset, casting_mode_int)

        ctx.in_place = in_place
        ctx.elementwise_affine = W is not None
        ctx.casting_mode = casting_mode_int
        # Save the user's original weight dtype so we can cast dW back to it.
        ctx.weight_dtype = W.dtype if W is not None else None
        if W is not None:
            ctx.save_for_backward(X_flat, W_eff, RSTD)
        else:
            ctx.save_for_backward(X_flat, RSTD)
        return Y

    @staticmethod
    def backward(ctx, dY):
        dY = _to_local_if_dtensor(dY).contiguous()

        if ctx.elementwise_affine:
            X_flat, W_eff, RSTD = ctx.saved_tensors
        else:
            X_flat, RSTD = ctx.saved_tensors
            W_eff = None

        dX, dW = _rms_norm_cutedsl_backward(
            dY,
            X_flat,
            W_eff,
            RSTD,
            ctx.in_place,
            ctx.casting_mode,
        )
        if ctx.elementwise_affine and dW is not None:
            dW = dW.to(ctx.weight_dtype)

        # forward arity: (X, W, eps, offset, casting_mode, in_place, row_mode)
        return dX, dW, None, None, None, None, None


# ===========================================================================
# Public registration
# ===========================================================================
_CUTEDSL_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


@register_op(
    "rms_norm",
    impl_name="nvidia-cutedsl",
    capability=Capability(
        min_cc=(9, 0),
        modules=["cutlass.cute", "torch"],
    ),
    modes=("default",),
    default_mode="default",
    # Auto-select (rank 40 < Triton 50). Measured on B200 dispatch path: pinned cutedsl
    # beats pinned Triton fwd 1.18-1.57x, full 1.37-1.43x w/ lower err vs fp64; guards+min_cc keep Triton fallback.
    preference_rank=40,
    tolerances=_CUTEDSL_TOLERANCES,
    notes="Auto-select CuTe DSL RMSNorm for Hopper+ (sm_90+); self-contained inline fwd+bwd.",
)
def rms_norm_cutedsl(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    offset: float = 0.0,
    casting_mode: str = "llama",
    in_place: bool = False,
    row_mode: Optional[bool] = None,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """CuTe DSL RMSNorm dispatch entry point.

    ``mode`` is accepted only for API parity with the other backends; the
    only valid value is ``"default"`` (or ``None``).
    """
    if mode not in (None, "default"):
        raise ValueError(f"CuTe DSL rms_norm has only mode='default'; got mode={mode!r}.")
    vector_width = 16 // x.element_size()
    fallback_reason = None
    if x.shape[-1] % vector_width:
        fallback_reason = f"hidden size {x.shape[-1]} is not divisible by vector width {vector_width}"
    elif x.shape[-1] > _BWD_MAX_TILE_CUTEDSL:
        fallback_reason = f"hidden size {x.shape[-1]} exceeds CuTe DSL limit {_BWD_MAX_TILE_CUTEDSL}"
    if fallback_reason is not None:
        from liger_kernel.ops.backends._triton.rms_norm import rms_norm_triton

        emit_fallback_warning(
            "rms_norm",
            "nvidia-cutedsl",
            "nvidia-triton",
            fallback_reason,
        )
        return rms_norm_triton(
            x,
            weight,
            eps,
            offset,
            casting_mode,
            in_place,
            row_mode,
            mode=mode,
        )
    return _LigerRMSNormCuTeDSLFunction.apply(x, weight, eps, offset, casting_mode, in_place, row_mode)
