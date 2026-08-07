"""SM90 fused scaled cross entropy **M-fast forward** kernel: the default
"MMA-warp issues TMA + dedicated SMEM-logits epilogue" kernel with an
**M-fast (N-outer / M-inner) persistent work order**.

This kernel backs ``m_tiles_per_cluster > 1`` in
:mod:`liger_kernel.ops.cutedsl.ops.fused_scaled_cross_entropy_sm90`; the public
``m_tiles_per_cluster`` is this module's ``slots_per_wave``.  It keeps several
``M128`` online-softmax states live so that one L2-resident ``W`` panel is
shared by several M tiles.  It is correct at every ``slots_per_wave`` but was
measured **slower** than the M-outer default at the representative
``M=4096, H=4096, V=131072`` shape, hence the default of 1.  Everything except
:func:`to_cute_tensor` is self-contained.

Work order
==========

The default forward (``_fused_scaled_cross_entropy_forward_sm90.py``) is
*already* statically persistent: a cluster owns ``pid_m = cluster_idx + m_slot *
num_persistent_clusters`` and walks its slots one at a time.  Its work nesting
is::

    for m_slot in owned_slots:          # M outer
        for n_round in range(N/(cluster_n*tile_n)):   # N inner
            for k in range(H/tile_k):
                WGMMA(M128 x N256 x K64)
        reduce + store row statistics for this m_slot

Softmax is a *row* reduction, so it only has to happen after **all** N tiles of
a row have been seen -- the N tiles themselves may be visited in any order.
This module therefore reverses the nesting, exactly as the dZ scheduler
(``_fused_scaled_cross_entropy_dz_sm90.py``) does for the
backward::

    for n_round in range(N/(cluster_n*tile_n)):       # N outer
        for m_slot in owned_slots:                    # M inner
            for k in range(H/tile_k):
                WGMMA(M128 x N256 x K64)
    for m_slot in owned_slots:                        # after ALL N work
        reduce + store row statistics

so the epilogue warp group carries **one online-softmax state per owned M
slot** across the whole N sweep instead of one state at a time.

Locality hypothesis under test (**disproven** -- see "Status: not the default")
------------------------------------------------------------------------------

At ``M=4096, H=4096, V=131072`` with ``cluster_n=16`` and 132 SMs, 7 clusters
are resident and each owns 4-5 ``M128`` slots.  Per N round a CTA touches one
``256 x 4096`` BF16 W panel = 2 MiB (32 MiB per cluster, and *all* clusters
touch the same 32 MiB in the same round).  The motivating hypothesis was that in
M-outer order that panel is re-fetched from DRAM once per M slot, so M-fast
order would cut W traffic ~4x by reusing the L2-resident panel across all owned
M slots.

The measurement says otherwise: because all resident clusters are on the *same*
N round in *both* orders, W is already fetched once and shared through L2, so
there is nothing to save; meanwhile the live X footprint grows from ~7 MiB
(1 block per cluster) to ~32 MiB (4-5 blocks per cluster), which together with
the 32 MiB W panel set overflows the 60 MiB L2 and makes X re-stream every
round.  Mainloop DRAM reads go **up** 4.3% and the kernel is 2.2-3.2% slower.
Full numbers and the mechanism are at the bottom of this docstring.

Everything else is held fixed
-----------------------------

Identical to the accepted forward, deliberately: WG0+WG1 cooperative
``atom_layout=(2,1,1)`` WGMMA over a ``(64, 256)`` atom (``M128 x N256 x K64``),
warp 0 of WG0 issuing every TMA under a live WGMMA group, ``wait_group(1)`` in
the steady state with a full drain only at the tile tail, WG2 as a dedicated
padded-SMEM (row stride ``tile_n + 8``) single-pass online-softmax epilogue,
3 stages, ``cluster_n = 16``, **no multicast**, 208/88 register split.
``cluster_n`` sweeps are explicitly *not* repeated here: ``cluster_n=4`` was
already measured at 10.44 ms / 421 TFLOPS against 7.02 ms / 627 for
``cluster_n=16`` under matched timing.

Implementation notes specific to the new order
==============================================

Waves
-----

Per-slot softmax state lives in registers, so the number of simultaneously live
slots must be a compile-time constant.  Owned slots are therefore processed in
*waves* of at most ``wave_slots = min(config.slots_per_wave, ceil(num_m_tiles /
num_persistent_clusters))`` slots (a host-computed constant, 4 at the target
shape, so the target shape is a single wave)::

    for wave:                        # dynamic, cluster-uniform
      for n_round:                   # dynamic
        for slot in range(cur):      # dynamic; cur = slots in this wave
          one M128 x N256 tile
      for slot in range(cur):        # reduce + store
    # register state: run_max[wave_slots], run_sum[wave_slots],
    #                 run_target[wave_slots], tgt_tile[..], tgt_col[..]

Only 1 wave is needed until a cluster owns more than ``slots_per_wave`` M
blocks, so the target shape never pays for the outer loop.

Keeping the epilogue code single-instance
-----------------------------------------

A naive implementation makes the slot loop ``range_constexpr`` so the register
arrays can be indexed statically -- which replicates the *entire* 256-column
softmax body ``wave_slots`` times (~4x SASS growth in WG2, straight into the
I-cache).  Instead the slot loop stays **dynamic** and the heavy body computes a
purely tile-local ``(tile_max, tile_sum, tile_target)`` in plain registers; only
the ~6-instruction fold into the per-slot state is wrapped in a
``range_constexpr`` + ``if slot == j`` select::

    for j in range_constexpr(wave_slots):
        if slot == j:
            new = fmax(run_max[j], tile_max)
            run_sum[j] = run_sum[j]*exp2((run_max[j]-new)*log2e)
                       + tile_sum  *exp2((tile_max  -new)*log2e)
            run_max[j] = new
            run_target[j] += tile_target

The same trick reads ``tgt_tile/tgt_col`` before the body and ``run_*`` before
the cluster reduction.  Net effect: register-resident per-slot state with the
epilogue body emitted exactly once, i.e. the same SASS footprint as the accepted
kernel.

The two-sentinel masking trick is preserved: the tile-local max starts at
``NEG_INF_F32 = -1e38`` while out-of-vocabulary columns are ``MASK_F32 =
-3e38``, so a fully-masked tile contributes ``exp2((-3e38 + 1e38) * log2 e) = 0``
rather than 1, and folding a ``(max=-1e38, sum=0)`` tile into any state is a
no-op (``0 * exp2(...) == 0``, never ``0 * inf``).

TMA stream
----------

The load stream is a **single linear stream over the whole persistent kernel**:
one prologue of ``stages`` loads at kernel entry, one refill per consumer
release, one ``producer_tail`` at exit.  The stream cursor is
``(k, slot, n_round, wave)`` advanced k-fastest, mirroring the consumer nesting
exactly, so ``load_index == release_index`` holds at every refill with no
modular arithmetic -- and unlike the accepted kernel there is no per-M-tile
prologue/drain at all, because M tiles are no longer loop boundaries.
``wait_group(1)`` and the tail-only ``wait_group(0)`` are unchanged.

Ragged tails
------------

``num_real_slots = ceil((num_m_tiles - cluster_idx) / num_persistent_clusters)``
depends only on ``blockIdx.z``, which is identical for all 16 CTAs of a cluster,
so every loop bound and every predicate above is **cluster-uniform**; named
barriers (``MMA_BARRIER_ID`` over 256 threads, ``EPI_BARRIER_ID`` over 128),
``__syncthreads``, the cluster barriers and both pipelines therefore stay
balanced even when the last cluster owns fewer M slots than the others.  All 16
ranks of a cluster are always on the same M slot for a given tile.

Status: not the default
-----------------------
The locality hypothesis is **disproven**.  Matched, interleaved measurements
put this order ~3 % behind the M-outer forward at the representative shape
(independently reconfirmed at ~3.8 %), because keeping several M tiles live per
cluster shrinks the per-tile ``W`` reuse window without buying back enough
``X`` reuse.  It is retained as the implementation of the public
``m_tiles_per_cluster > 1`` scheduling knob (``slots_per_wave``); the default
``m_tiles_per_cluster == 1`` dispatches to the M-outer forward instead.

"""

import inspect

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from cutlass import BFloat16
from cutlass import Float16
from cutlass import Float32
from cutlass import Int32
from cutlass import cutlass_dsl
from cutlass import pipeline
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline.helpers import pipeline_init_arrive
from cutlass.pipeline.helpers import pipeline_init_wait
from cutlass.utils import LayoutEnum
from cutlass.utils import hopper_helpers

from liger_kernel.ops.cutedsl.ops.utils import to_cute_tensor

THREADS_PER_CTA = 384
NUM_MMA_WARP_GROUPS = 2
WARPS_PER_WARP_GROUP = 4

# Named barriers.  0 is reserved for __syncthreads().
MMA_BARRIER_ID = 4  # both MMA warp groups, 256 threads
EPI_BARRIER_ID = 5  # epilogue warp group, 128 threads

LOG2_E = 1.4426950408889634
LN2 = 0.6931471805599453
NEG_INF_F32 = -1.0e38
MASK_F32 = -3.0e38
HOPPER_MAX_SMEM_BYTES = 227 * 1024

_compile_cache = {}
_stream_cache = {}
_max_active_clusters_cache = {}

try:
    _FMAX_NEEDS_RESULT_TYPE = next(iter(inspect.signature(nvvm.fmax).parameters)) == "res"
except Exception:
    _FMAX_NEEDS_RESULT_TYPE = False


@dsl_user_op
def _fmax(a, b, *, loc=None, ip=None) -> Float32:
    av = Float32(a).ir_value(loc=loc, ip=ip)
    bv = Float32(b).ir_value(loc=loc, ip=ip)
    if _FMAX_NEEDS_RESULT_TYPE:
        return Float32(nvvm.fmax(T.f32(), av, bv, loc=loc, ip=ip))
    return Float32(nvvm.fmax(av, bv, loc=loc, ip=ip))


@dsl_user_op
def _map_dsmem_addr(local_ptr, peer_rank, *, loc=None, ip=None) -> Int32:
    """Map a local SMEM pointer to the same object in a peer CTA's SMEM.

    ``llvm.nvvm.mapa`` ICEs this toolchain ("only supported when pointer size is
    >= 64 bits"), so the PTX is emitted directly, exactly as
    ``cutlass.cute.arch.map_dsmem_ptr`` does.
    """
    addr = llvm.ptrtoint(T.i32(), local_ptr.llvm_ptr, loc=loc, ip=ip)
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [addr, Int32(peer_rank).ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _asm(res_ty, args, tmpl, cons, side=True, *, loc=None, ip=None):
    return llvm.inline_asm(
        res_ty,
        args,
        tmpl,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _dsmem_load_f32(addr, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.f32 $0, [$1];",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _dsmem_atomic_add_f32(addr, value, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [
                Int32(addr).ir_value(loc=loc, ip=ip),
                Float32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.shared::cluster.add.f32 $0, [$1], $2;",
            "=f,r,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _dsmem_atomic_max_s32(addr, value, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [
                Int32(addr).ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.shared::cluster.max.s32 $0, [$1], $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _dsmem_atomic_min_u32(addr, value, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [
                Int32(addr).ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.shared::cluster.min.u32 $0, [$1], $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _dsmem_atomic_fmax(addr, value, *, loc=None, ip=None) -> Float32:
    """FP32 max via the monotonic-bit-pattern trick (``red.max.f32`` has no PTX)."""
    bits = llvm.bitcast(T.i32(), Float32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    old = cutlass_dsl.if_generate(
        Int32(bits) < 0,
        lambda: _dsmem_atomic_min_u32(addr, Int32(bits), loc=loc, ip=ip),
        lambda: _dsmem_atomic_max_s32(addr, Int32(bits), loc=loc, ip=ip),
        [],
        [Int32],
        loc=loc,
        ip=ip,
    )
    return Float32(llvm.bitcast(T.f32(), old.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


def _cute_stream():
    raw = torch.cuda.current_stream().cuda_stream
    stream = _stream_cache.get(raw)
    if stream is None:
        stream = cuda.CUstream(raw)
        _stream_cache[raw] = stream
    return stream


def _max_active_clusters(cluster_n):
    key = (torch.cuda.current_device(), cluster_n)
    value = _max_active_clusters_cache.get(key)
    if value is None:
        torch.cuda.current_stream()
        value = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_n)
        _max_active_clusters_cache[key] = value
    return value


@dataclass(frozen=True)
class ScaledCEForwardMFastConfig:
    """Tuning knobs for the M-fast (N-outer / M-inner) forward.

    Everything except ``slots_per_wave`` matches the accepted forward kernel's
    defaults and should not be swept here -- this experiment is about the work
    order only.

    ``slots_per_wave`` caps how many owned ``M128`` slots keep a live
    online-softmax state (and therefore how many M slots share one L2-resident W
    panel).  The kernel is compiled with ``wave_slots = min(slots_per_wave,
    ceil(num_m_tiles / num_persistent_clusters))``, so it never costs registers
    the shape cannot use.  ``slots_per_wave=1`` reproduces the accepted
    M-outer order exactly and is the in-module A/B control.

    Register budget: ``2 * 128 * mma_registers + 128 * epi_registers <= 64512``
    (requesting the full 65536-register file deadlocks ``setmaxnreg.inc``).
    """

    tile_m: int = 128
    tile_n: int = 256
    tile_k: int = 64
    stages: int = 3
    cluster_n: int = 16
    fast_math: bool = True
    logit_dtype: str = "f16"
    logit_pad: int = 8
    chunk_n: int = 32
    mma_registers: int = 208
    epi_registers: int = 88
    slots_per_wave: int = 8
    return_entropy: bool = False

    @property
    def logit_stride(self):
        return self.tile_n + self.logit_pad

    def smem_bytes(self):
        ab = (self.tile_m + self.tile_n) * self.tile_k * 2 * self.stages
        logits = self.tile_m * self.logit_stride * 2
        reduction = (3 + int(self.return_entropy)) * self.tile_m * 4
        barriers = 16 * self.stages + 16
        return ab + logits + reduction + barriers + 3 * 1024

    def register_total(self):
        return 2 * 128 * self.mma_registers + 128 * self.epi_registers


class _ScaledCEForwardMFastSM90:
    """M-fast persistent forward: N-outer / M-inner over each cluster's slots."""

    def __init__(self, config: ScaledCEForwardMFastConfig, wave_slots: int):
        self.tile_m = config.tile_m
        self.tile_n = config.tile_n
        self.tile_k = config.tile_k
        self.stages = config.stages
        self.cluster_n = config.cluster_n
        self.fast_math = config.fast_math
        self.logit_pad = config.logit_pad
        self.logit_stride = config.tile_n + config.logit_pad
        self.chunk_n = config.chunk_n
        self.mma_registers = config.mma_registers
        self.epi_registers = config.epi_registers
        self.return_entropy = config.return_entropy
        self.logit_dtype = Float16 if config.logit_dtype == "f16" else BFloat16
        self.wave_slots = int(wave_slots)
        self.tile_shape = (self.tile_m, self.tile_n, self.tile_k)
        self.buffer_align = 1024
        self.mma_barrier = pipeline.NamedBarrier(
            barrier_id=MMA_BARRIER_ID,
            num_threads=NUM_MMA_WARP_GROUPS * 128,
        )
        self.epi_barrier = pipeline.NamedBarrier(
            barrier_id=EPI_BARRIER_ID,
            num_threads=128,
        )

    @staticmethod
    def _make_tma(tensor, smem_layout_staged, smem_tile):
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        return cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=1,
        )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        target: cute.Tensor,
        lse: cute.Tensor,
        token_loss: cute.Tensor,
        entropy: cute.Tensor,
        inverse_temperature: Float32,
        ignore_index: Int32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        x_layout = LayoutEnum.from_tensor(x)
        weight_layout = LayoutEnum.from_tensor(weight)
        tiled_mma = hopper_helpers.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            x_layout.sm90_mma_major_mode(),
            weight_layout.sm90_mma_major_mode(),
            Float32,
            (NUM_MMA_WARP_GROUPS, 1, 1),
            (64, self.tile_n),
        )
        a_smem_layout = hopper_helpers.make_smem_layout_a(
            x_layout,
            self.tile_shape,
            BFloat16,
            self.stages,
        )
        b_smem_layout = hopper_helpers.make_smem_layout_b(
            weight_layout,
            self.tile_shape,
            BFloat16,
            self.stages,
        )
        tma_x, tma_tensor_x = self._make_tma(x, a_smem_layout, (self.tile_m, self.tile_k))
        tma_w, tma_tensor_w = self._make_tma(weight, b_smem_layout, (self.tile_n, self.tile_k))

        logit_dtype = self.logit_dtype

        @cute.struct
        class SharedStorage:
            pipeline: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            logits_pipeline: cute.struct.MemRange[cutlass.Int64, 2]
            sX: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(a_smem_layout)],
                self.buffer_align,
            ]
            sW: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(b_smem_layout)],
                self.buffer_align,
            ]
            sLogits: cute.struct.Align[
                cute.struct.MemRange[
                    logit_dtype,
                    self.tile_m * self.logit_stride,
                ],
                self.buffer_align,
            ]
            cluster_max: cute.struct.Align[cute.struct.MemRange[Float32, self.tile_m], 16]
            cluster_sum: cute.struct.Align[cute.struct.MemRange[Float32, self.tile_m], 16]
            cluster_target: cute.struct.Align[cute.struct.MemRange[Float32, self.tile_m], 16]
            cluster_weighted: cute.struct.Align[
                cute.struct.MemRange[Float32, self.tile_m if self.return_entropy else 1],
                16,
            ]

        self.shared_storage = SharedStorage
        cluster_shape = (1, self.cluster_n, 1)
        num_m_tiles = cute.ceil_div(x.shape[0], self.tile_m)
        num_persistent_clusters = cutlass.min(num_m_tiles, max_active_clusters)
        grid = (1, self.cluster_n, num_persistent_clusters)
        self.kernel(
            tma_x,
            tma_tensor_x,
            tma_w,
            tma_tensor_w,
            target,
            lse,
            token_loss,
            entropy,
            inverse_temperature,
            tiled_mma,
            a_smem_layout,
            b_smem_layout,
            ignore_index,
        ).launch(
            grid=grid,
            block=(THREADS_PER_CTA, 1, 1),
            cluster=cluster_shape,
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_x: cute.CopyAtom,
        x: cute.Tensor,
        tma_w: cute.CopyAtom,
        weight: cute.Tensor,
        target: cute.Tensor,
        lse: cute.Tensor,
        token_loss: cute.Tensor,
        entropy: cute.Tensor,
        inverse_temperature: Float32,
        tiled_mma: cute.TiledMma,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        ignore_index: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group = cute.arch.make_warp_uniform(tid // 128)
        lane = tid % 32
        local_warp = (tid % 128) // 32
        local_tid = tid % 128
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        _, _, cluster_idx = cute.arch.block_idx()
        _, _, num_persistent_clusters = cute.arch.grid_dim()
        num_m_tiles = cute.ceil_div(target.shape[0], self.tile_m)

        if warp == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_x)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_w)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sX = storage.sX.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sW = storage.sW.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sLogits = storage.sLogits.get_tensor(
            cute.make_layout(
                (self.tile_m, self.tile_n),
                stride=(self.logit_stride, 1),
            )
        )
        cluster_max = storage.cluster_max.get_tensor(cute.make_layout(self.tile_m))
        cluster_sum = storage.cluster_sum.get_tensor(cute.make_layout(self.tile_m))
        cluster_target = storage.cluster_target.get_tensor(cute.make_layout(self.tile_m))
        cluster_weighted = storage.cluster_weighted.get_tensor(
            cute.make_layout(self.tile_m if cutlass.const_expr(self.return_entropy) else 1)
        )

        cta_layout = cute.make_layout((1, self.cluster_n, 1))
        cluster_coord = cta_layout.get_flat_coord(cluster_rank)

        gX = cute.local_tile(
            x,
            cute.slice_(self.tile_shape, (None, 0, None)),
            (None, None, None),
        )
        gW = cute.local_tile(
            weight,
            cute.slice_(self.tile_shape, (0, None, None)),
            (None, None, None),
        )
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_x,
            0,
            cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(gX, 0, 2),
        )
        w_cta_layout = cute.make_layout(cute.slice_(cta_layout, (None, 0, 0)).shape)
        tWsW, tWgW = cute.nvgpu.cpasync.tma_partition(
            tma_w,
            cluster_coord[0],
            w_cta_layout,
            cute.group_modes(sW, 0, 2),
            cute.group_modes(gW, 0, 2),
        )

        tma_bytes = cute.size_in_bytes(BFloat16, cute.slice_(a_smem_layout, (None, None, 0))) + cute.size_in_bytes(
            BFloat16, cute.slice_(b_smem_layout, (None, None, 0))
        )

        mainloop = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.pipeline.data_ptr(),
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                NUM_MMA_WARP_GROUPS * WARPS_PER_WARP_GROUP,
            ),
            tx_count=tma_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )
        logits_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.logits_pipeline.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, self.cluster_n), is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=(1, self.cluster_n))

        num_k_tiles = cute.size(gX, mode=[3])
        num_k_blocks = self.tile_k // 16
        num_pair_tiles = cute.ceil_div(weight.shape[0], self.cluster_n * self.tile_n)

        # ---- M-slot ownership (cluster-uniform: depends only on blockIdx.z) ---
        num_real_slots = cute.ceil_div(num_m_tiles - cluster_idx, num_persistent_clusters)
        num_waves = cute.ceil_div(num_real_slots, self.wave_slots)

        root_max_addr = _map_dsmem_addr(storage.cluster_max.data_ptr(), 0)
        root_sum_addr = _map_dsmem_addr(storage.cluster_sum.data_ptr(), 0)
        root_target_addr = _map_dsmem_addr(storage.cluster_target.data_ptr(), 0)
        root_weighted_addr = _map_dsmem_addr(storage.cluster_weighted.data_ptr(), 0)

        if warp_group < NUM_MMA_WARP_GROUPS:
            # ================= MMA warp groups (WG0 + WG1) =================
            cute.arch.setmaxregister_increase(self.mma_registers)

            mma_wg_layout = cute.make_layout(NUM_MMA_WARP_GROUPS, stride=128)
            thr_mma = tiled_mma.get_slice(mma_wg_layout(Int32(warp_group)))
            tCrX = tiled_mma.make_fragment_A(thr_mma.partition_A(sX))
            tCrW = tiled_mma.make_fragment_B(thr_mma.partition_B(sW))
            accum = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((self.tile_m, self.tile_n)),
                Float32,
            )
            n_accum = cute.size(accum)
            row_base_in_tile = warp_group * 64 + local_warp * 16 + lane // 4
            col_base_in_tile = (lane % 4) * 2

            read_index = Int32(0)
            read_phase = Int32(0)
            rel_index = Int32(0)
            rel_phase = Int32(0)
            load_index = Int32(0)
            load_phase = Int32(1)
            logit_w_index = Int32(0)
            logit_w_phase = Int32(1)

            # TMA cursor: (k, slot-in-wave, n_round, wave), advanced k-fastest.
            load_k = Int32(0)
            load_j = Int32(0)
            load_round = Int32(0)
            load_wave = Int32(0)
            load_wave_slots = cutlass.min(Int32(self.wave_slots), num_real_slots)

            cute.arch.sync_threads()
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            # ---- prologue: S complete TMA stages before any WGMMA ----------
            if warp == 0:
                total_loads = Int32(num_k_tiles) * Int32(num_pair_tiles) * num_real_slots
                prologue_stages = cutlass.min(Int32(self.stages), total_loads)
                for _ in range(prologue_stages):
                    pstate = pipeline.PipelineState(self.stages, Int32(0), load_index, load_phase)
                    mainloop.producer_acquire(pstate)
                    bar = mainloop.producer_get_barrier(pstate)
                    load_pid_m = cluster_idx + (load_wave * self.wave_slots + load_j) * num_persistent_clusters
                    tXgX_m = tXgX[(None, load_pid_m, None, 0)]
                    cute.copy(
                        tma_x,
                        tXgX_m[(None, load_k)],
                        tXsX[(None, load_index)],
                        tma_bar_ptr=bar,
                        mcast_mask=0,
                    )
                    tWgW_n = tWgW[(None, load_round * self.cluster_n + cluster_rank, None, 0)]
                    cute.copy(
                        tma_w,
                        tWgW_n[(None, load_k)],
                        tWsW[(None, load_index)],
                        tma_bar_ptr=bar,
                    )
                    mainloop.producer_commit(pstate)
                    pstate.advance()
                    load_index = pstate.index
                    load_phase = pstate.phase
                    load_k += 1
                    if load_k == num_k_tiles:
                        load_k = Int32(0)
                        load_j += 1
                    if load_j == load_wave_slots:
                        load_j = Int32(0)
                        load_round += 1
                    if load_round == num_pair_tiles:
                        load_round = Int32(0)
                        load_wave += 1
                        load_wave_slots = cutlass.min(
                            Int32(self.wave_slots),
                            num_real_slots - load_wave * self.wave_slots,
                        )
                        if load_wave_slots < 1:
                            load_wave_slots = Int32(1)

            for wave in range(num_waves):
                cur_slots = cutlass.min(
                    Int32(self.wave_slots),
                    num_real_slots - wave * self.wave_slots,
                )

                for _n_round in range(num_pair_tiles):
                    for _slot in range(cur_slots):
                        # ------------- one M128 x N256 output tile ----------
                        accum.fill(0.0)
                        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

                        read_state = pipeline.PipelineState(self.stages, Int32(0), read_index, read_phase)
                        mainloop.consumer_wait(read_state)
                        cute.nvgpu.warpgroup.fence()
                        for k_block in cutlass.range_constexpr(num_k_blocks):
                            coord = (None, None, k_block, read_index)
                            cute.gemm(tiled_mma, accum, tCrX[coord], tCrW[coord], accum)
                        cute.nvgpu.warpgroup.commit_group()
                        read_state.advance()
                        read_index = read_state.index
                        read_phase = read_state.phase

                        for _ in range(1, num_k_tiles):
                            read_state = pipeline.PipelineState(self.stages, Int32(0), read_index, read_phase)
                            mainloop.consumer_wait(read_state)
                            cute.nvgpu.warpgroup.fence()
                            for k_block in cutlass.range_constexpr(num_k_blocks):
                                coord = (None, None, k_block, read_index)
                                cute.gemm(tiled_mma, accum, tCrX[coord], tCrW[coord], accum)
                            cute.nvgpu.warpgroup.commit_group()
                            # Retire only the previous group; the group just
                            # issued stays IN FLIGHT across release + refill.
                            cute.nvgpu.warpgroup.wait_group(1)
                            rstate = pipeline.PipelineState(self.stages, Int32(0), rel_index, rel_phase)
                            mainloop.consumer_release(rstate)
                            rstate.advance()
                            rel_index = rstate.index
                            rel_phase = rstate.phase
                            if warp == 0:
                                if load_wave < num_waves:
                                    pstate = pipeline.PipelineState(self.stages, Int32(0), load_index, load_phase)
                                    mainloop.producer_acquire(pstate)
                                    bar = mainloop.producer_get_barrier(pstate)
                                    load_pid_m = (
                                        cluster_idx + (load_wave * self.wave_slots + load_j) * num_persistent_clusters
                                    )
                                    tXgX_m = tXgX[(None, load_pid_m, None, 0)]
                                    cute.copy(
                                        tma_x,
                                        tXgX_m[(None, load_k)],
                                        tXsX[(None, load_index)],
                                        tma_bar_ptr=bar,
                                        mcast_mask=0,
                                    )
                                    tWgW_n = tWgW[(None, load_round * self.cluster_n + cluster_rank, None, 0)]
                                    cute.copy(
                                        tma_w,
                                        tWgW_n[(None, load_k)],
                                        tWsW[(None, load_index)],
                                        tma_bar_ptr=bar,
                                    )
                                    mainloop.producer_commit(pstate)
                                    pstate.advance()
                                    load_index = pstate.index
                                    load_phase = pstate.phase
                                    load_k += 1
                                    if load_k == num_k_tiles:
                                        load_k = Int32(0)
                                        load_j += 1
                                    if load_j == load_wave_slots:
                                        load_j = Int32(0)
                                        load_round += 1
                                    if load_round == num_pair_tiles:
                                        load_round = Int32(0)
                                        load_wave += 1
                                        load_wave_slots = cutlass.min(
                                            Int32(self.wave_slots),
                                            num_real_slots - load_wave * self.wave_slots,
                                        )
                                        if load_wave_slots < 1:
                                            load_wave_slots = Int32(1)
                            read_state.advance()
                            read_index = read_state.index
                            read_phase = read_state.phase

                        # ---- tile tail: the only WGMMA drain ---------------
                        cute.nvgpu.warpgroup.wait_group(0)
                        rstate = pipeline.PipelineState(self.stages, Int32(0), rel_index, rel_phase)
                        mainloop.consumer_release(rstate)
                        rstate.advance()
                        rel_index = rstate.index
                        rel_phase = rstate.phase
                        if warp == 0:
                            if load_wave < num_waves:
                                pstate = pipeline.PipelineState(self.stages, Int32(0), load_index, load_phase)
                                mainloop.producer_acquire(pstate)
                                bar = mainloop.producer_get_barrier(pstate)
                                load_pid_m = (
                                    cluster_idx + (load_wave * self.wave_slots + load_j) * num_persistent_clusters
                                )
                                tXgX_m = tXgX[(None, load_pid_m, None, 0)]
                                cute.copy(
                                    tma_x,
                                    tXgX_m[(None, load_k)],
                                    tXsX[(None, load_index)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=0,
                                )
                                tWgW_n = tWgW[(None, load_round * self.cluster_n + cluster_rank, None, 0)]
                                cute.copy(
                                    tma_w,
                                    tWgW_n[(None, load_k)],
                                    tWsW[(None, load_index)],
                                    tma_bar_ptr=bar,
                                )
                                mainloop.producer_commit(pstate)
                                pstate.advance()
                                load_index = pstate.index
                                load_phase = pstate.phase
                                load_k += 1
                                if load_k == num_k_tiles:
                                    load_k = Int32(0)
                                    load_j += 1
                                if load_j == load_wave_slots:
                                    load_j = Int32(0)
                                    load_round += 1
                                if load_round == num_pair_tiles:
                                    load_round = Int32(0)
                                    load_wave += 1
                                    load_wave_slots = cutlass.min(
                                        Int32(self.wave_slots),
                                        num_real_slots - load_wave * self.wave_slots,
                                    )
                                    if load_wave_slots < 1:
                                        load_wave_slots = Int32(1)

                        wstate = pipeline.PipelineState(1, Int32(0), logit_w_index, logit_w_phase)
                        logits_pipeline.producer_acquire(wstate)
                        for i in cutlass.range_constexpr(n_accum):
                            row = row_base_in_tile + ((i % 4) // 2) * 8
                            col = col_base_in_tile + (i // 4) * 8 + (i % 2)
                            sLogits[row, col] = self.logit_dtype(accum[i])
                        cute.arch.fence_proxy("async.shared", space="cta")
                        cute.arch.fence_acq_rel_cta()
                        self.mma_barrier.arrive_and_wait()
                        if tid == 0:
                            logits_pipeline.producer_commit(wstate)
                        wstate.advance()
                        logit_w_index = wstate.index
                        logit_w_phase = wstate.phase

                # ---- per-slot cluster reduction: barrier participation only --
                for _slot in range(cur_slots):
                    cute.arch.sync_threads()
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()

            if warp == 0:
                ptail = pipeline.PipelineState(self.stages, Int32(0), load_index, load_phase)
                mainloop.producer_tail(ptail)
        else:
            # ==================== epilogue warp group ======================
            cute.arch.setmaxregister_decrease(self.epi_registers)

            row = local_tid
            chunk = cute.make_rmem_tensor((self.chunk_n,), Float32)
            # Per-owned-M-slot online softmax state, register resident.
            run_max = cute.make_rmem_tensor((self.wave_slots,), Float32)
            run_sum = cute.make_rmem_tensor((self.wave_slots,), Float32)
            run_target = cute.make_rmem_tensor((self.wave_slots,), Float32)
            run_weighted = cute.make_rmem_tensor(
                (self.wave_slots if self.return_entropy else 1,),
                Float32,
            )
            tgt_tile = cute.make_rmem_tensor((self.wave_slots,), Int32)
            tgt_col = cute.make_rmem_tensor((self.wave_slots,), Int32)
            logit_r_index = Int32(0)
            logit_r_phase = Int32(0)

            cute.arch.sync_threads()
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            for wave in range(num_waves):
                cur_slots = cutlass.min(
                    Int32(self.wave_slots),
                    num_real_slots - wave * self.wave_slots,
                )
                slot_base = wave * self.wave_slots

                for j in cutlass.range_constexpr(self.wave_slots):
                    run_max[j] = Float32(NEG_INF_F32)
                    run_sum[j] = Float32(0.0)
                    run_target[j] = Float32(0.0)
                    if cutlass.const_expr(self.return_entropy):
                        run_weighted[j] = Float32(0.0)
                    tgt_tile[j] = Int32(-1)
                    tgt_col[j] = Int32(-1)
                    if cur_slots > j:
                        grow = (cluster_idx + (slot_base + j) * num_persistent_clusters) * self.tile_m + row
                        if grow < target.shape[0]:
                            tcol = Int32(target[grow])
                            if tcol >= 0:
                                tgt_tile[j] = tcol // self.tile_n
                                tgt_col[j] = tcol % self.tile_n

                for n_round in range(num_pair_tiles):
                    n_tile = n_round * self.cluster_n + cluster_rank
                    valid_cols = Int32(weight.shape[0]) - n_tile * self.tile_n

                    for slot in range(cur_slots):
                        # Pull this slot's target coordinates into plain
                        # registers so the heavy body below stays a single
                        # SASS instance (see module docstring).
                        cur_tgt_tile = Int32(-1)
                        cur_tgt_col = Int32(-1)
                        for j in cutlass.range_constexpr(self.wave_slots):
                            if slot == j:
                                cur_tgt_tile = tgt_tile[j]
                                cur_tgt_col = tgt_col[j]

                        rstate = pipeline.PipelineState(1, Int32(0), logit_r_index, logit_r_phase)
                        logits_pipeline.consumer_wait(rstate)
                        cute.arch.fence_acq_rel_cta()

                        tile_max = Float32(NEG_INF_F32)
                        tile_sum = Float32(0.0)
                        tile_target = Float32(0.0)
                        tile_weighted = Float32(0.0)
                        for ci in cutlass.range_constexpr(self.tile_n // self.chunk_n):
                            c0 = ci * self.chunk_n
                            chunk_max = Float32(MASK_F32)
                            if c0 + self.chunk_n <= valid_cols:
                                for jj in cutlass.range_constexpr(self.chunk_n):
                                    v = Float32(sLogits[row, c0 + jj]) * inverse_temperature
                                    chunk[jj] = v
                                    chunk_max = _fmax(chunk_max, v)
                            else:
                                for jj in cutlass.range_constexpr(self.chunk_n):
                                    v = Float32(sLogits[row, c0 + jj]) * inverse_temperature
                                    if c0 + jj >= valid_cols:
                                        v = Float32(MASK_F32)
                                    chunk[jj] = v
                                    chunk_max = _fmax(chunk_max, v)
                            new_max = _fmax(tile_max, chunk_max)
                            acc = Float32(0.0)
                            weighted = Float32(0.0)
                            for jj in cutlass.range_constexpr(self.chunk_n):
                                exp_value = cute.math.exp2(
                                    (chunk[jj] - new_max) * LOG2_E,
                                    fastmath=self.fast_math,
                                )
                                acc += exp_value
                                if cutlass.const_expr(self.return_entropy):
                                    weighted += exp_value * chunk[jj]
                            old_scale = cute.math.exp2(
                                (tile_max - new_max) * LOG2_E,
                                fastmath=self.fast_math,
                            )
                            tile_sum = tile_sum * old_scale + acc
                            if cutlass.const_expr(self.return_entropy):
                                tile_weighted = tile_weighted * old_scale + weighted
                            tile_max = new_max

                        if cur_tgt_tile == n_tile:
                            tile_target = Float32(sLogits[row, cur_tgt_col]) * inverse_temperature

                        self.epi_barrier.arrive_and_wait()
                        if local_tid == 0:
                            logits_pipeline.consumer_release(rstate)
                        rstate.advance()
                        logit_r_index = rstate.index
                        logit_r_phase = rstate.phase

                        # Fold the tile-local result into this slot's state.
                        for j in cutlass.range_constexpr(self.wave_slots):
                            if slot == j:
                                folded = _fmax(run_max[j], tile_max)
                                run_sum[j] = run_sum[j] * cute.math.exp2(
                                    (run_max[j] - folded) * LOG2_E,
                                    fastmath=self.fast_math,
                                ) + tile_sum * cute.math.exp2(
                                    (tile_max - folded) * LOG2_E,
                                    fastmath=self.fast_math,
                                )
                                if cutlass.const_expr(self.return_entropy):
                                    run_weighted[j] = run_weighted[j] * cute.math.exp2(
                                        (run_max[j] - folded) * LOG2_E,
                                        fastmath=self.fast_math,
                                    ) + tile_weighted * cute.math.exp2(
                                        (tile_max - folded) * LOG2_E,
                                        fastmath=self.fast_math,
                                    )
                                run_max[j] = folded
                                run_target[j] = run_target[j] + tile_target

                # ---- all N work for this wave is done: reduce every slot ----
                for slot in range(cur_slots):
                    cur_max = Float32(NEG_INF_F32)
                    cur_sum = Float32(0.0)
                    cur_target = Float32(0.0)
                    cur_weighted = Float32(0.0)
                    for j in cutlass.range_constexpr(self.wave_slots):
                        if slot == j:
                            cur_max = run_max[j]
                            cur_sum = run_sum[j]
                            cur_target = run_target[j]
                            if cutlass.const_expr(self.return_entropy):
                                cur_weighted = run_weighted[j]

                    global_row = (cluster_idx + (slot_base + slot) * num_persistent_clusters) * self.tile_m + row

                    if cluster_rank == 0:
                        cluster_max[row] = Float32(NEG_INF_F32)
                        cluster_sum[row] = Float32(0.0)
                        cluster_target[row] = Float32(0.0)
                        if cutlass.const_expr(self.return_entropy):
                            cluster_weighted[row] = Float32(0.0)
                    cute.arch.sync_threads()
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()

                    _dsmem_atomic_fmax(root_max_addr + row * 4, cur_max)
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()

                    global_max = _dsmem_load_f32(root_max_addr + row * 4)
                    scaled_sum = cur_sum * cute.math.exp2(
                        (cur_max - global_max) * LOG2_E,
                        fastmath=self.fast_math,
                    )
                    _dsmem_atomic_add_f32(root_sum_addr + row * 4, scaled_sum)
                    _dsmem_atomic_add_f32(root_target_addr + row * 4, cur_target)
                    if cutlass.const_expr(self.return_entropy):
                        scaled_weighted = cur_weighted * cute.math.exp2(
                            (cur_max - global_max) * LOG2_E,
                            fastmath=self.fast_math,
                        )
                        _dsmem_atomic_add_f32(root_weighted_addr + row * 4, scaled_weighted)
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()

                    if cluster_rank == 0:
                        if global_row < target.shape[0]:
                            row_lse = cluster_max[row] + cute.math.log2(cluster_sum[row], fastmath=self.fast_math) * LN2
                            lse[global_row] = row_lse
                            if target[global_row] == ignore_index:
                                token_loss[global_row] = Float32(0.0)
                                if cutlass.const_expr(self.return_entropy):
                                    entropy[global_row] = BFloat16(0.0)
                            else:
                                token_loss[global_row] = row_lse - cluster_target[row]
                                if cutlass.const_expr(self.return_entropy):
                                    entropy[global_row] = BFloat16(row_lse - cluster_weighted[row] / cluster_sum[row])

                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()


def _validate(x, weight, target):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(x.device) != (9, 0):
        raise RuntimeError("SM90 fused scaled cross entropy M-fast forward requires a Hopper (sm90) GPU")
    if x.device != weight.device or x.device != target.device:
        raise ValueError("input, weight, and target must be on the same CUDA device")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("SM90 fused scaled cross entropy M-fast forward supports BF16 input and weight only")
    if x.ndim != 2 or weight.ndim != 2 or target.ndim != 1:
        raise ValueError("expected input[BT,H], weight[V,H], and target[BT]")
    if x.shape[0] != target.shape[0] or x.shape[1] != weight.shape[1]:
        raise ValueError("input, weight, and target shapes are incompatible")


def _pad_hidden(x, weight, tile_k):
    hidden_size = x.shape[1]
    padded = (hidden_size + tile_k - 1) // tile_k * tile_k
    if padded == hidden_size:
        return x.contiguous(), weight.contiguous(), hidden_size
    return (
        torch.nn.functional.pad(x, (0, padded - hidden_size)),
        torch.nn.functional.pad(weight, (0, padded - hidden_size)),
        hidden_size,
    )


def scaled_ce_forward_mfast(
    _input,
    weight,
    target,
    temperature=1.0,
    ignore_index=-100,
    return_entropy=False,
    config: ScaledCEForwardMFastConfig = None,
):
    """M-fast (N-outer / M-inner) forward.  Returns per-token ``(nll[M], lse[M])``."""
    _validate(_input, weight, target)
    if config is None:
        config = ScaledCEForwardMFastConfig()
    if config.return_entropy != return_entropy:
        config = ScaledCEForwardMFastConfig(**{**config.__dict__, "return_entropy": return_entropy})
    if config.tile_m != 128:
        raise ValueError("tile_m must be 128 (one epilogue thread per row)")
    if config.tile_n % 8 != 0 or config.tile_n % config.chunk_n != 0:
        raise ValueError("tile_n must be a multiple of 8 and of chunk_n")
    if config.slots_per_wave < 1:
        raise ValueError("slots_per_wave must be >= 1")
    if config.smem_bytes() > HOPPER_MAX_SMEM_BYTES:
        raise ValueError(
            f"config needs ~{config.smem_bytes()} B of shared memory, "
            f"which exceeds the {HOPPER_MAX_SMEM_BYTES} B Hopper opt-in limit"
        )
    if config.register_total() > 64512:
        raise ValueError(f"register budget {config.register_total()} exceeds the usable 64512-register budget")

    target = target.contiguous()
    x_padded, weight_padded, _ = _pad_hidden(_input, weight, config.tile_k)
    bt = x_padded.shape[0]

    lse = torch.empty(bt, device=_input.device, dtype=torch.float32)
    token_loss = torch.empty(bt, device=_input.device, dtype=torch.float32)
    entropy = torch.empty(bt if return_entropy else 1, device=_input.device, dtype=torch.bfloat16)
    x_cute = to_cute_tensor(x_padded.unsqueeze(-1), leading_dim=1, assumed_align=16)
    weight_cute = to_cute_tensor(weight_padded.unsqueeze(-1), leading_dim=1, assumed_align=16)
    target_cute = to_cute_tensor(target, assumed_align=8)
    lse_cute = to_cute_tensor(lse, assumed_align=4)
    token_loss_cute = to_cute_tensor(token_loss, assumed_align=4)
    entropy_cute = to_cute_tensor(entropy, assumed_align=2)
    stream = _cute_stream()
    inverse_temperature = Float32(1.0 / temperature)
    max_active_clusters = _max_active_clusters(config.cluster_n)

    # ``wave_slots`` must be a compile-time constant (it sizes the per-slot
    # register arrays), so it is derived on the host from the launch geometry.
    num_m_tiles = (bt + config.tile_m - 1) // config.tile_m
    num_persistent_clusters = min(num_m_tiles, max_active_clusters)
    max_m_slots = (num_m_tiles + num_persistent_clusters - 1) // num_persistent_clusters
    wave_slots = max(1, min(config.slots_per_wave, max_m_slots))

    key = ("scaled_ce_forward_mfast", config, max_active_clusters, wave_slots)
    compiled = _compile_cache.get(key)
    if compiled is None:
        compiled = cute.compile(
            _ScaledCEForwardMFastSM90(config, wave_slots),
            x_cute,
            weight_cute,
            target_cute,
            lse_cute,
            token_loss_cute,
            entropy_cute,
            inverse_temperature,
            Int32(ignore_index),
            max_active_clusters,
            stream,
        )
        _compile_cache[key] = compiled
    compiled(
        x_cute,
        weight_cute,
        target_cute,
        lse_cute,
        token_loss_cute,
        entropy_cute,
        inverse_temperature,
        Int32(ignore_index),
        stream,
    )

    return token_loss, entropy if return_entropy else None, lse
