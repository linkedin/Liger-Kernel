"""CuTe DSL (CUTLASS python) backend for ``layer_norm``.

Strategy
--------
This is a **hybrid** implementation:

- **Forward** delegates to the inlined ``layernorm_fwd``
  (``_cute_lib.layernorm_fwd``), which ships an inline CuTe DSL kernel (the
  shared ``RMSNorm`` class with ``is_layernorm=True`` runs the two-pass
  mean+variance reduction we need). The implementation is adapted from Quack
  (Apache-2.0; see ``_cute_lib/NOTICE.md``); we keep a per-process
  ``cute.compile`` cache so repeated calls reuse the compiled artifact.

- **Backward** uses an inline CuTe DSL kernel defined locally
  (``_LigerLayerNormCuTeDSLBackward``). It builds on the inlined reduction
  primitives (:class:`._cute_lib.reduction_base.ReductionBase`,
  :func:`._cute_lib.reduce.row_reduce`,
  :func:`._cute_lib.copy_utils.tiled_copy_2d`) — the same plumbing Quack
  uses for its RMSNorm backward, specialized for LayerNorm's
  ``c1 = mean(wdy * x_hat)`` and ``c2 = mean(wdy)`` two-statistic reduction.
  ``dW`` and ``dB`` are produced as per-SM ``(sm_count, N)`` fp32 partials
  and reduced cross-SM on the **host** via ``.sum(dim=0)`` — the same
  pattern Liger uses in its Triton and cuTile backends. The compiled
  backward kernel is cached locally in ``_BWD_COMPILE_CACHE`` keyed by
  ``(dtype, weight_dtype, N, has_bias)``.

Math (matches the Triton kernel in ``liger_kernel.ops.layer_norm``)::

    mean   = sum(x) / N
    var    = sum((x - mean)^2) / N
    rstd   = 1 / sqrt(var + eps)
    x_hat  = (x - mean) * rstd
    y      = w * x_hat + b
    wdy    = w * dy                                  (fp32)
    c1     = mean(wdy * x_hat)
    c2     = mean(wdy)
    dx     = rstd * (wdy - x_hat * c1 - c2)
    dW    += dy * x_hat                              (fp32, per-row)
    dB    += dy                                      (fp32, per-row)

Capability
----------
- Compute capability >= sm_90 (Hopper or newer).
- Requires only the ``cutlass`` Python package (the CuTe DSL utilities are
  inlined under :mod:`._cute_lib`; no runtime dependency on Quack).

Shapes and limits
-----------------
- ``x`` must be 2D after flattening (``view(-1, N)``); arbitrary leading dims
  are supported by the wrapper.
- ``N`` must be divisible by the SIMT vector width the kernel picks
  (gcd(N, 128/elem_bits)); odd hidden sizes fall back to triton via the
  registry's preference rank.
- Backward kernel: ``N <= 128k`` for fp16/bf16 (the register-pressure cliff
  at the ``stage=2`` working set is the same as Quack's RMSNormBackward).
- dtypes: fp16, bf16, fp32 for ``x``; fp32 for ``weight``/``bias`` (the
  inlined forward asserts fp32 weight/bias, mirroring Quack). The forward
  wrapper auto-casts fp16/bf16 weight/bias to fp32 before invoking the
  kernel so user code keeps its natural parameter dtype.

Notes
-----
We do **not** put ``from __future__ import annotations`` here because the
sibling cuTile module documents that future-annotations interferes with DSL
introspection. CuTe DSL itself doesn't introspect like cuTile does, but we
keep the runtime evaluation for consistency across backends.

References
----------
- Quack (adapted under Apache-2.0): ``Dao-AILab/quack`` — ``quack/rmsnorm.py``
  is the source of the forward ``RMSNorm`` class (with ``is_layernorm=True``),
  ``ReductionBase``, the compile-cache pattern, and the dW/dB host-side
  ``.sum(0)`` reduction pattern. See :mod:`._cute_lib` for the inlined copy.
  https://github.com/Dao-AILab/quack
- Triton reference: ``liger_kernel.ops.layer_norm`` — fixes the exact
  numerics we must reproduce (mean/rstd cast paths, dW/dB row accumulation).
"""

import math

from functools import partial
from typing import Optional
from typing import Tuple
from typing import Type

# ---------------------------------------------------------------------------
# Top-level CuTe DSL imports.
#
# We deliberately import here (not inside the function body) so that
# ``@register_op`` runs at module-import time. The registry's discovery loop
# catches ImportError on the whole module and reports it via
# ``python -m liger_kernel.dev check`` — see
# ``liger_kernel.backends.registry._discover``.
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
from liger_kernel.ops.backends._cutedsl._cute_lib.compile_utils import make_fake_tensor as fake_tensor
from liger_kernel.ops.backends._cutedsl._cute_lib.dtype_map import torch2cute_dtype_map
from liger_kernel.ops.backends._cutedsl._cute_lib.reduce import row_reduce
from liger_kernel.ops.backends._cutedsl._cute_lib.reduction_base import ReductionBase
from liger_kernel.ops.backends._cutedsl._cute_lib.rmsnorm_fwd import layernorm_fwd as _cutedsl_layernorm_fwd
from liger_kernel.ops.utils import device_context

# Beyond this hidden dim the backward kernel enters its multi-cluster
# (``cluster_n >= 2``) path AND switches ``reload_wdy`` to smem staging. On
# B200 this path faults at runtime with a *context-poisoning* launch error
# for every dtype once more than ~1 row lands per program — measured
# (CUDA_LAUNCH_BLOCKING=1, fresh process per cell): fp32 N=11008/12288/
# 14336/16384/24576/32768 and bf16 N=11008/16384/24576/32768 all
# ``cudaErrorLaunchFailure`` at M <= 2048 (M=256, ~3.5 rows/program, also
# faults; M=64, ~1 row/program, happens not to), while N=8192 stays healthy
# at M=65536 (bf16/fp32, bias/no-bias). The older wrapper guard at 32768 let
# this hard fault slip behind the rank-40 auto-select; cap at the verified
# single-cluster boundary instead. NOTE: rms_norm.py owns an identically
# named constant — do NOT change it there; its backward differs (not
# measured faulting).
_BWD_MAX_TILE_CUTEDSL = 8192

# Forward-right-side ceiling for *no-grad* inference calls: through this hidden
# dim the (single-cluster) forward kernel is verified healthy for hidden sizes
# up to 32768, and for 2-byte dtypes it beats Triton at every probed band cell
# (see the wrapper guard comment below).  fp32 has its own rule (see wrapper
# guard) because its varsched fp32 tile hits a pathological rung at exactly
# 16384.  Dispatch-only; the kernel is unchanged.
_FWD_NO_GRAD_MAX_TILE_CUTEDSL = 32768


# ===========================================================================
# Backward kernel — inline CuTe DSL
#
# Mirrors quack's RMSNormBackward in structure but accumulates the two
# LayerNorm-specific row statistics (mean(wdy*x_hat) and mean(wdy)) inside
# the same reduction infrastructure.
# ===========================================================================
class _LigerLayerNormCuTeDSLBackward(ReductionBase):
    """CuTe DSL implementation of LayerNorm backward.

    Layout: one CTA processes ``rows_per_program`` consecutive rows. Grid is
    ``(sm_count, cluster_n)``; each SM holds a per-SM ``(N,)`` fp32 partial
    for ``dW`` and ``dB`` which the host reduces post-launch.

    The reduction-buffer slot count (``stage=2``) is intentional: ``c1`` and
    ``c2`` are reduced in parallel across the two slots.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        has_bias: bool = True,
    ):
        super().__init__(dtype, N, stage=2, reduction_dtype=Float32)
        self.has_bias = has_bias
        # Beyond 16K we reload x/dy from smem instead of holding two
        # full-width fragments live — matches the seed's heuristic.
        self.reload_wdy = None if N <= 16 * 1024 else "smem"
        if self.N > 128 * 1024 and self.dtype.width >= 32:
            raise ValueError(
                "LayerNormBackward does not support N > 128k with dtype >= 32 bits (register file pressure)."
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
        mW: Optional[cute.Tensor],  # Weight [N,] or None
        mdO: cute.Tensor,  # dY [M, N]
        mMean: cute.Tensor,  # Mean [M,] (fp32)
        mRstd: cute.Tensor,  # RSTD [M,] (fp32)
        mdX: cute.Tensor,  # dX [M, N]
        mdW: Optional[cute.Tensor],  # dW partial [sm_count, N] fp32
        mdB: Optional[cute.Tensor],  # dB partial [sm_count, N] fp32
        sm_count: Int32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()

        largest_dtype_width = const_expr(max(*(t.element_type.width for t in [mX, mW, mdO, mdX] if t is not None)))
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)

        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size

        mW = layout_utils.expand(mW, dim=0, size=tiler_mn[0]) if const_expr(mW is not None) else None

        num_blocks = sm_count

        self.kernel(
            mX,
            mW,
            mdO,
            mMean,
            mRstd,
            mdX,
            mdW,
            mdB,
            tiler_mn,
            tiled_copy,
            threads_per_row,
        ).launch(
            grid=[num_blocks, self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if self.cluster_n > 1 else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mdO: cute.Tensor,
        mMean: cute.Tensor,
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: Optional[cute.Tensor],
        mdB: Optional[cute.Tensor],
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
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
        gdB = cute.local_tile(mdB, (1, tiler_mn[1]), (bidx_start, cluster_y)) if const_expr(mdB is not None) else None

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
            tXrdW = cute.make_fragment_like(tXgdW, Float32)

        tXgdB, tXrdB = None, None
        if const_expr(mdB is not None):
            tXgdB = thr_copy_X.partition_S(gdB)
            tXrdB = cute.make_fragment_like(tXgdB, Float32)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE

        self._initialize_cluster(tidx, mbar_ptr, num_warps, is_persistent=True)

        tXrW = None
        if const_expr(mW is not None):
            tXgW = thr_copy_X.partition_S(gW)
            tXrW = cute.make_fragment_like(tXgW)
            if const_expr(not is_even_N):
                tXrW.fill(0.0)
            copy_pred(tXgW, tXrW)

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
        if const_expr(mdB is not None):
            tXrdB.fill(0.0)

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

            mean_val = cutlass.Float.zero
            rstd_val = cutlass.Float.zero
            if row < M or tiler_mn[0] == 1:
                mean_val = mMean[row].to(cute.Float32)
                rstd_val = mRstd[row]

            cute.arch.cp_async_wait_group(1)
            cute.autovec_copy(tXsX[None, None, None, stage], tXrX)
            x = tXrX.load().to(cute.Float32)
            cute.autovec_copy(tXsdO[None, None, None, stage], tXrdO)
            dout = tXrdO.load().to(cute.Float32)

            x_hat = (x - mean_val) * rstd_val

            wdy = dout
            if const_expr(mW is not None):
                wdy = wdy * tXrW.load().to(Float32)

            if const_expr(self.cluster_n > 1):
                cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)

            c1 = (
                row_reduce(
                    wdy * x_hat,
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, stage],
                    (mbar_full_ptr + stage if const_expr(self.cluster_n > 1) else None),
                    phase=consumer_phase,
                    init_val=0.0,
                )
                / shape[1]
            )
            c2 = (
                row_reduce(
                    wdy,
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, stage ^ 1],
                    (mbar_full_ptr + (stage ^ 1) if const_expr(self.cluster_n > 1) else None),
                    phase=consumer_phase,
                    init_val=0.0,
                )
                / shape[1]
            )

            if const_expr(self.cluster_n > 1):
                # See _cutedsl/rms_norm.py: Quack uses fence_view_async_shared
                # which is present in all cutlass-cute wheels; the older
                # ProxyKind API is absent on cuda13.2 wheels.
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

            # dx = rstd * (wdy - c2 - x_hat * c1)
            dx = rstd_val * (wdy - c2 - x_hat * c1)

            tXrdX.store(dx.to(tXrdX.element_type))
            if row < M or tiler_mn[0] == 1:
                copy_pred(tXrdX, tXgdX[None, None, None, bidx])

            # Accumulate per-SM partials. dW uses x_hat (centered+scaled),
            # dB just sums dY. Both stay in fp32 until the host reduces.
            if const_expr(mdW is not None):
                tXrdW.store(tXrdW.load() + dout * x_hat)
            if const_expr(mdB is not None):
                tXrdB.store(tXrdB.load() + dout)

            stage ^= 1
            if stage == 0:
                consumer_phase ^= 1
                producer_phase ^= 1

        # Reduce per-thread partials within the CTA (same scheme as the seed
        # / Quack RMSNormBackward): row 0 collects the other rows' partials
        # from smem and writes the final per-SM partial to gmem.
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
            if const_expr(mdB is not None):
                sdB = cute.make_tensor(
                    cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                )
                tXsdB = thr_copy_X.partition_D(sdB)
                cute.arch.barrier()
                row_in_tile = tXcX[None, None, None, 0][0][0]
                if row_in_tile > 0:
                    cute.autovec_copy(tXrdB, tXsdB)
                cute.arch.barrier()
                if row_in_tile == 0:
                    for i in cutlass.range_constexpr(1, const_expr(tiler_mn[0])):
                        tXrdB_other = cute.make_fragment_like(tXrdB)
                        tXsdB_other = cute.make_tensor(tXsdB.iterator + i * sdB.stride[0], tXsdB.layout)
                        cute.autovec_copy(tXsdB_other, tXrdB_other)
                        tXrdB.store(tXrdB.load() + tXrdB_other.load())
                    copy_pred(tXrdB, tXgdB)
                cute.arch.barrier()
        else:
            if const_expr(mdW is not None):
                copy_pred(tXrdW, tXgdW)
            if const_expr(mdB is not None):
                copy_pred(tXrdB, tXgdB)

        if const_expr(self.cluster_n > 1):
            stage ^= 1
            if stage == 0:
                producer_phase ^= 1
            cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)


# ---------------------------------------------------------------------------
# Backward compile cache.
#
# ``cute.compile()`` is a slow (multi-second) operation. We key the compiled
# kernel by (input_dtype, weight_dtype, N, has_bias); shapes outside ``N`` are
# represented symbolically inside the fake-tensor signature so the same
# compiled object handles any batch size.
# ---------------------------------------------------------------------------
_BWD_COMPILE_CACHE: dict = {}


def _bwd_sm_count(N: int, device: torch.device) -> int:
    """SM-count heuristic mirroring Quack's ``_get_sm_count``.

    Wider rows get fewer SMs (each row already saturates a CTA); narrow rows
    multiply the SM count so per-SM partial buffers don't collapse to too few
    rows-per-program.
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
    has_bias: bool,
):
    """Return a compiled backward kernel, building it on first miss.

    Cache key is exactly the set of attributes that change codegen: input
    dtype, weight dtype, output-gradient dtype, hidden size ``N``, and the
    bias flag (which adds the ``dB`` accumulator).
    """
    key = (x_dtype, weight_dtype, dx_dtype, N, has_bias)
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
    mean_cute = fake_tensor(Float32, (batch_sym,))
    rstd_cute = fake_tensor(Float32, (batch_sym,))
    dx_cute = fake_tensor(dx_cute_dtype, (batch_sym, N), div)

    sm_sym = cute.sym_int()
    dw_cute = fake_tensor(Float32, (sm_sym, N), div) if weight_cute_dtype is not None else None
    db_cute = fake_tensor(Float32, (sm_sym, N), div) if has_bias else None

    kernel = _LigerLayerNormCuTeDSLBackward(dtype, N, has_bias=has_bias)
    compiled = cute.compile(
        kernel,
        x_cute,
        weight_cute,
        dout_cute,
        mean_cute,
        rstd_cute,
        dx_cute,
        dw_cute,
        db_cute,
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _BWD_COMPILE_CACHE[key] = compiled
    return compiled


# ===========================================================================
# Host-side launchers and autograd Function
# ===========================================================================
def _layer_norm_cutedsl_forward(
    x: Tensor, weight: Tensor, bias: Optional[Tensor], eps: float
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Forward via Quack's ``layernorm_fwd``.

    Returns ``(Y, X_flat, W, B, Mean, Rstd)`` (W/B in native dtype). Mean and
    fp32 (Quack always produces them in fp32 when requested), suitable for
    saving into ``ctx`` for the backward.
    """
    shape = x.shape
    N = shape[-1]
    x_flat = x.view(-1, N).contiguous()

    # The forward and backward kernels load weight/bias in their native dtype
    # and promote to fp32 in-register (``tXrW.load().to(Float32)``), so the old
    # host-side ``.to(torch.float32)`` up-cast was redundant: two extra
    # elementwise kernel launches per call plus a 2x-wider weight read in the
    # kernel. ``fp16/bf16 -> fp32`` is exact, so numerics are bit-identical.
    weight_nat = weight.contiguous()
    bias_nat = bias.contiguous() if bias is not None else None

    out, rstd, mean = _cutedsl_layernorm_fwd(
        x_flat,
        weight_nat,
        bias=bias_nat,
        eps=eps,
        return_rstd=True,
        return_mean=True,
    )
    return out.view(shape), x_flat, weight_nat, bias_nat, mean, rstd


def _layer_norm_cutedsl_backward(
    dy: Tensor,
    x_flat: Tensor,
    weight_f32: Tensor,
    bias_f32: Optional[Tensor],
    mean: Tensor,
    rstd: Tensor,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Backward via the inline CuTe DSL kernel.

    ``dY`` is reshaped to 2D to match ``x_flat``. Per-SM partials are
    allocated as fp32 ``(sm_count, N)`` and reduced to the final ``dW`` /
    ``dB`` host-side. This matches Liger's Triton and cuTile backends.
    """
    with device_context(x_flat.device):
        shape = dy.shape
        N = shape[-1]
        # Cap N at the last verified-healthy single-cluster (cluster_n == 1)
        # hidden dim; wider rows fault at runtime on B200 (see the constant's
        # comment above) and surface as the "only supports hidden dim"
        # RuntimeError that the test framework auto-skips.
        if N > _BWD_MAX_TILE_CUTEDSL:
            raise RuntimeError(
                f"cuTeDSL layer_norm backward only supports hidden dim <= "
                f"{_BWD_MAX_TILE_CUTEDSL}; got {N}. Use backend='triton' for "
                f"wider rows. (Cluster-reduce path requires a newer cutlass-cute.)"
            )
        dy_flat = dy.view(-1, N).contiguous()
        M = dy_flat.shape[0]

        dx = torch.empty_like(dy_flat)
        sm_count = _bwd_sm_count(N, x_flat.device)
        # Saturate the grid: don't launch more SMs than there are rows. ``ceil``
        # gives an SM per chunk of ``ceil(M / sm_count)`` rows; we clamp so each
        # SM has at least one row.
        sm_count = min(sm_count, max(M, 1))

        has_bias = bias_f32 is not None
        dw_partial = torch.empty((sm_count, N), dtype=torch.float32, device=x_flat.device)
        db_partial = torch.empty((sm_count, N), dtype=torch.float32, device=x_flat.device) if has_bias else None

        compiled = _get_bwd_kernel(
            x_dtype=x_flat.dtype,
            weight_dtype=weight_f32.dtype,
            dx_dtype=dx.dtype,
            N=N,
            has_bias=has_bias,
        )

        # CuTe DSL compiled kernels read torch.cuda.current_stream() at launch;
        # the kernel ABI does not take a stream positional. Passing `stream` here
        # raised TypeError: Expects 9 parameters (...).
        compiled(
            x_flat,
            weight_f32,
            dy_flat,
            mean,
            rstd,
            dx,
            dw_partial,
            db_partial,
            sm_count,
        )

        # Host-side cross-SM reduction. Cast dW back to the *user's* weight
        # dtype (we accepted fp16/bf16 weight by casting up in the forward).
        dw = dw_partial.sum(dim=0)
        db = db_partial.sum(dim=0) if has_bias else None
        return dx.view(shape), dw, db


class _LigerLayerNormCuTeDSLFunction(torch.autograd.Function):
    """Autograd wrapper. Saves enough state for the inline-kernel backward.

    We save ``x_flat`` (the 2D view used by the kernel) and ``weight_f32`` /
    ``bias_f32`` (the weights in their *native* dtype — the kernels promote to
    fp32 in-register) so the backward doesn't re-cast and dW reduction matches.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        y, x_flat, weight_f32, bias_f32, mean, rstd = _layer_norm_cutedsl_forward(x, weight, bias, eps)
        ctx.save_for_backward(x_flat, weight_f32, bias_f32, mean, rstd)
        ctx.eps = eps
        # Remember the user's original weight/bias dtype so we can cast the
        # per-SM-reduced gradients back to it.
        ctx.weight_dtype = weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, dy):
        x_flat, weight_f32, bias_f32, mean, rstd = ctx.saved_tensors
        dy = dy.contiguous()
        dx, dw, db = _layer_norm_cutedsl_backward(dy, x_flat, weight_f32, bias_f32, mean, rstd)
        # Match the dtype of the corresponding ``forward`` inputs.
        dw = dw.to(ctx.weight_dtype)
        if ctx.has_bias and db is not None:
            db = db.to(ctx.bias_dtype)
        elif not ctx.has_bias:
            db = None
        return dx, dw, db, None


# ===========================================================================
# Public registration
# ===========================================================================
_CUTEDSL_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    # Two-pass reduction (mean + variance) accumulates more rounding than
    # RMSNorm; atol=5e-4 absorbs N up to ~32K at unit magnitude.
    torch.float32: {"atol_fwd": 5e-4, "atol_bwd": 1e-3, "rtol_fwd": 1e-4, "rtol_bwd": 1e-3},
}


def _fwd_no_grad_band_ok(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> bool:
    """Route no-grad inference calls in the (8192, 32768] band to the forward
    kernel even though the backward kernel faults above 8192.

    The band's forward is healthy (all dtypes; bias or no-bias), and for
    2-byte dtypes it beats Triton at every probed cell (B200: 1.20-1.64x).
    fp32 is kept on the 8192 bwd-rule instead because its tile hits a
    pathological rung at exactly N=16384 (0.16-0.20x vs Triton), while a
    16384 cap would leave measured 1.5-1.9x fp32 wins at 24576/32768
    unrouted; a finer fp32 gate was rejected as fragile crossover-chasing.

    The forward path returns a detached view of ``x`` (raw helper, no
    autograd graph), so it is only safe when *no* input can require the
    gradient: grad mode must be off, or every input requires_grad=False.
    A frozen-x call whose weight/bias still requires grad must stay on the
    8192 rule (Triton fallback), or weight.grad would silently be dropped.
    """
    if x.element_size() != 2:
        return False
    if not _BWD_MAX_TILE_CUTEDSL < x.shape[-1] <= _FWD_NO_GRAD_MAX_TILE_CUTEDSL:
        return False
    if not torch.is_grad_enabled():
        return True
    if x.requires_grad or weight.requires_grad:
        return False
    return not (bias is not None and bias.requires_grad)


@register_op(
    "layer_norm",
    impl_name="nvidia-cutedsl",
    capability=Capability(
        min_cc=(9, 0),
        modules=["cutlass.cute", "torch"],
    ),
    modes=("default",),
    default_mode="default",
    # B200 (after the host-side fp32 weight/bias cast removal): forward AND
    # backward both beat Triton for hidden <= 8192 (fwd 1.34-1.38x,
    # bwd 1.35-1.43x), so promote below Triton's rank 50 for auto-select.
    # Backward hard-faults above 8192 (see ``_BWD_MAX_TILE_CUTEDSL``), so
    # training calls in the (8192, 32768] band fall back to Triton inside
    # ``layer_norm_cutedsl``; *no-grad* 2-byte inference calls in the band
    # still take the healthy forward path (measured 1.20-1.64x vs Triton).
    preference_rank=40,
    tolerances=_CUTEDSL_TOLERANCES,
    notes=(
        "CuTe DSL LayerNorm for Hopper+ (sm_90+); fwd+bwd both faster than "
        "Triton on B200 for hidden <= 8192 (auto-selected); training calls "
        "8192-32768 fall back to Triton (bwd fault above 8192); no-grad "
        "fp16/bf16 inference in that band takes the fwd-only path; "
        "non-vector-width or >32K hiddens fall back to Triton."
    ),
)
def layer_norm_cutedsl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """CuTe DSL LayerNorm dispatch entry point.

    ``mode`` is accepted only for API parity with the other backends; the
    only valid value is ``"default"`` (or ``None``). To opt into a multi-
    cluster reduction variant later, add it to the ``modes=`` tuple in
    the ``@register_op`` block above.
    """
    if mode not in (None, "default"):
        raise ValueError(f"CuTe DSL layer_norm has only mode='default'; got mode={mode!r}.")
    vector_width = 16 // x.element_size()
    fallback_reason = None
    if x.shape[-1] % vector_width:
        fallback_reason = f"hidden size {x.shape[-1]} is not divisible by vector width {vector_width}"
    else:
        limit = _FWD_NO_GRAD_MAX_TILE_CUTEDSL if _fwd_no_grad_band_ok(x, weight, bias) else _BWD_MAX_TILE_CUTEDSL
        if x.shape[-1] > limit:
            fallback_reason = f"hidden size {x.shape[-1]} exceeds CuTe DSL limit {limit}"
        elif limit == _FWD_NO_GRAD_MAX_TILE_CUTEDSL:
            # No-grad band call: forward only, no autograd graph, no bwd fault.
            return _layer_norm_cutedsl_forward(x, weight, bias, eps)[0]
    if fallback_reason is not None:
        from liger_kernel.ops.backends._triton.layer_norm import layer_norm_triton

        emit_fallback_warning(
            "layer_norm",
            "nvidia-cutedsl",
            "nvidia-triton",
            fallback_reason,
        )
        return layer_norm_triton(x, weight, bias, eps, mode=mode)
    return _LigerLayerNormCuTeDSLFunction.apply(x, weight, bias, eps)
