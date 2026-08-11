"""Hopper (SM90) CuTe DSL fused-linear-cross-entropy **forward** building blocks.

This module owns two device kernels used by
:mod:`liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy_sm90`:

``_TileGemmSM90``
    The static WGMMA fallback computes ``C[M, N] = A[M, K] @ B[N, K].T`` with
    BF16 operands and an FP32 accumulator. Each operand may be either *K-major*
    (the contraction dim is contiguous, ``leading_dim`` points at it) or
    *MN-major* (the M/N dim is contiguous). Its cooperative split-M topology
    (two WGMMA warp groups, ``M128 x N256 x K64``, 3 stages) drives dX and
    remains available for logits/dW with ``FLCE_LOGITS_PERSISTENT=0`` /
    ``FLCE_DW_CLUSTERED_PERSISTENT=0``:

    ==========  ==========================  ==============  ==============
    GEMM        maths                        A layout        B layout
    ==========  ==========================  ==============  ==============
    logits      ``X[M,H] @ W[V,H].T``        K-major (H)     K-major (H)
    dX          ``dZ[M,V] @ W[V,H]``         K-major (V)     MN-major (H)
    dW          ``dZ.T[V,M] @ X[M,H]``       MN-major (V)    MN-major (H)
    ==========  ==========================  ==============  ==============

    The default logits and dW phases use :mod:`_sm90_persistent_gemm`: a
    cluster-2, four-stage CUTLASS persistent template with one DMA and two WGMMA
    warp groups plus a TMA-store epilogue. dW's scalar autograd scale is fused
    into that epilogue.

    The static mainloop is the proven "warp-0-of-WG0 issues every TMA" software
    pipeline: a single linear ``cp.async.bulk.tensor`` load stream feeds a
    3-stage SMEM buffer, one WGMMA group stays in flight across each
    release+refill, and the pipeline is drained exactly once per output tile.

``_CrossEntropyClusterVector``
    The preferred one-read CE path. Eight clustered CTAs cooperatively process
    each row using aligned 128-bit copies, retain exponentials in registers, and
    merge partition statistics through ordered deterministic DSMEM peer loads.
    Shapes outside its bounded register/tail contract fall back to the original
    two-kernel ``_CrossEntropyPartials`` + ``_CrossEntropyDZ`` implementation.
    Both paths write ``dZ = (softmax(logits) - onehot(target)) * row_scale`` over
    the BF16 logits in place.

Only the pieces needed for a correct first functional FLCE are implemented:
Hopper SM90, BF16 contiguous ``x[M, H]`` / ``w[V, H]``, int64 ``target[M]``.
``H`` is padded up to ``TILE_N`` internally (a contraction-dim pad is exact);
``M`` and ``V`` are guarded to be tile-aligned by the caller.
"""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from cutlass import BFloat16
from cutlass import Float32
from cutlass import Int32
from cutlass import pipeline
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline.helpers import pipeline_init_arrive
from cutlass.pipeline.helpers import pipeline_init_wait
from cutlass.utils import LayoutEnum
from cutlass.utils import hopper_helpers

from liger_kernel.ops.cutedsl.ops.utils import to_cute_tensor

TILE_M = 128
TILE_N = 256
TILE_K = 64
STAGES = 3
NUM_MMA_WG = 2
THREADS = NUM_MMA_WG * 128

# Persistent dW epilogue: stage the FP32 accumulator into a small BF16 SMEM
# buffer one 64-wide column sub-tile at a time and TMA-store it asynchronously,
# double/quad-buffered, so the store overlaps the next tile's WGMMA mainloop
# instead of draining the pipeline (the synchronous SMEM->GMEM epilogue idled
# the tensor cores ~40% of the time under the one-CTA/SM persistent schedule).
DW_STORE_TILE_N = 256
DW_STORE_STAGES = 1
DW_SUBTILES = TILE_N // DW_STORE_TILE_N
DW_STORE_BARRIER_ID = 1


WARPS_PER_CE_CTA = 8
CE_THREADS = WARPS_PER_CE_CTA * 32
CE_SCRATCH_VALUES = WARPS_PER_CE_CTA * 3 + 3
CE_PARTITIONS = 8
CE_VECTOR = 8
CE_VECTOR_BITS = CE_VECTOR * 16
CE_VECTOR_TILE = CE_THREADS * CE_VECTOR
CE_VECTOR_VALUES_PER_THREAD = 64
NEG_INF_F32 = -3.0e38
LOG2_E = 1.4426950408889634
LN2 = 0.6931471805599453


@dsl_user_op
def _map_dsmem_addr(local_ptr, peer_rank, *, loc=None, ip=None) -> Int32:
    """Map a local SMEM pointer into a peer CTA's cluster-shared window."""
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


@dsl_user_op
def _dsmem_load_f32(addr, *, loc=None, ip=None) -> Float32:
    """Load one FP32 value from a deterministic peer DSMEM address."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.f32 $0, [$1];",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------
# WGMMA GEMM: C[M, N] = A[M, K] @ B[N, K].T
# ---------------------------------------------------------------------------
class _TileGemmSM90:
    """BF16 WGMMA GEMM with a K-major/MN-major agnostic operand loader."""

    def __init__(self, swap_grid=False, scale_output=False, persistent=False, num_sms=0):
        self.swap_grid = swap_grid
        self.scale_output = scale_output
        # ``persistent`` launches exactly ``num_sms`` CTAs (one wave) and streams
        # a flattened m-outer/n-inner tile schedule through a single continuous
        # TMA software pipeline.  Used for the dW GEMM, whose 1 GB A operand
        # (dZ.T[V, M]) otherwise thrashes L2 across ~121 scheduling waves.
        self.persistent = persistent
        self.num_sms = num_sms

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
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        output_scale: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        a_layout = LayoutEnum.from_tensor(a)
        b_layout = LayoutEnum.from_tensor(b)
        tile_shape = (TILE_M, TILE_N, TILE_K)
        tiled_mma = hopper_helpers.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            a_layout.sm90_mma_major_mode(),
            b_layout.sm90_mma_major_mode(),
            Float32,
            (NUM_MMA_WG, 1, 1),
            (64, TILE_N),
        )
        a_smem_layout = hopper_helpers.make_smem_layout_a(a_layout, tile_shape, BFloat16, STAGES)
        b_smem_layout = hopper_helpers.make_smem_layout_b(b_layout, tile_shape, BFloat16, STAGES)
        tma_a, ta = self._make_tma(a, a_smem_layout, (TILE_M, TILE_K))
        tma_b, tb = self._make_tma(b, b_smem_layout, (TILE_N, TILE_K))

        m = a.shape[0]
        n = b.shape[0]
        grid_m = cute.ceil_div(m, TILE_M)
        grid_n = cute.ceil_div(n, TILE_N)

        if cutlass.const_expr(self.persistent):
            # Persistent path: async TMA-store epilogue with a multi-stage BF16
            # store buffer (sD) instead of the synchronous sC drain.
            c_layout = LayoutEnum.from_tensor(c)
            store_smem_layout = hopper_helpers.make_smem_layout_epi(
                BFloat16, c_layout, (TILE_M, DW_STORE_TILE_N), DW_STORE_STAGES
            )
            tma_c_store, _tc = cute.nvgpu.cpasync.make_tiled_tma_atom(
                cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
                c,
                cute.slice_(store_smem_layout, (None, None, 0)),
                (TILE_M, DW_STORE_TILE_N),
            )

            @cute.struct
            class StoragePersistent:
                pipe: cute.struct.MemRange[cutlass.Int64, STAGES * 2]
                sA: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(a_smem_layout)], 1024]
                sB: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(b_smem_layout)], 1024]
                sD: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(store_smem_layout)], 1024]

            self.storage_t = StoragePersistent
            self.kernel_persistent(
                tma_a,
                ta,
                tma_b,
                tb,
                c,
                _tc,
                output_scale,
                tiled_mma,
                a_smem_layout,
                b_smem_layout,
                tma_c_store,
                store_smem_layout,
            ).launch(
                grid=(self.num_sms, 1, 1),
                block=(THREADS, 1, 1),
                stream=stream,
            )
            return

        @cute.struct
        class Storage:
            pipe: cute.struct.MemRange[cutlass.Int64, STAGES * 2]
            sA: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(a_smem_layout)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(b_smem_layout)], 1024]
            sC: cute.struct.Align[cute.struct.MemRange[BFloat16, TILE_M * (TILE_N + 8)], 1024]

        # Plain row-major epilogue SMEM tile with a +8 column pad.  The pad makes
        # consecutive rows land in different SMEM banks, so the strided WGMMA
        # register -> (row, col) scatter is bank-conflict free.
        c_smem_layout = cute.make_layout((TILE_M, TILE_N), stride=(TILE_N + 8, 1))
        self.storage_t = Storage
        grid = (grid_n, grid_m, 1) if self.swap_grid else (grid_m, grid_n, 1)
        self.kernel(
            tma_a,
            ta,
            tma_b,
            tb,
            c,
            output_scale,
            tiled_mma,
            a_smem_layout,
            b_smem_layout,
            c_smem_layout,
        ).launch(
            grid=grid,
            block=(THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_a: cute.CopyAtom,
        a: cute.Tensor,
        tma_b: cute.CopyAtom,
        b: cute.Tensor,
        c: cute.Tensor,
        output_scale: Optional[cute.Tensor],
        tiled_mma: cute.TiledMma,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        c_smem_layout: cute.Layout,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group = cute.arch.make_warp_uniform(tid // 128)
        lane = tid % 32
        local_warp = (tid % 128) // 32
        block_x, block_y, _ = cute.arch.block_idx()
        pid_m = block_y if self.swap_grid else block_x
        pid_n = block_x if self.swap_grid else block_y

        if warp == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_b)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.storage_t)
        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sC = storage.sC.get_tensor(c_smem_layout)

        tile_shape = (TILE_M, TILE_N, TILE_K)
        gA = cute.local_tile(a, cute.slice_(tile_shape, (None, 0, None)), (None, None, None))
        gB = cute.local_tile(b, cute.slice_(tile_shape, (0, None, None)), (None, None, None))
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )

        tma_bytes = cute.size_in_bytes(BFloat16, cute.slice_(a_smem_layout, (None, None, 0))) + cute.size_in_bytes(
            BFloat16, cute.slice_(b_smem_layout, (None, None, 0))
        )

        mainloop = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.pipe.data_ptr(),
            num_stages=STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, NUM_MMA_WG * 4),
            tx_count=tma_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=(1, 1))

        num_k_tiles = cute.size(gA, mode=[3])
        num_k_blocks = TILE_K // 16

        cute.arch.setmaxregister_increase(240)
        mma_wg_layout = cute.make_layout(NUM_MMA_WG, stride=128)
        thr_mma = tiled_mma.get_slice(mma_wg_layout(Int32(warp_group)))
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
        accum = cute.make_rmem_tensor(thr_mma.partition_shape_C((TILE_M, TILE_N)), Float32)
        n_accum = cute.size(accum)

        tAgA_m = tAgA[(None, pid_m, None, 0)]
        tBgB_n = tBgB[(None, pid_n, None, 0)]

        load_index = Int32(0)
        load_phase = Int32(1)
        read_index = Int32(0)
        read_phase = Int32(0)
        rel_index = Int32(0)
        rel_phase = Int32(0)
        load_k = Int32(0)

        # ---- prologue: STAGES loads on the single TMA warp ---------------
        if warp == 0:
            prologue = cutlass.min(Int32(STAGES), Int32(num_k_tiles))
            for _ in range(prologue):
                ps = pipeline.PipelineState(STAGES, Int32(0), load_index, load_phase)
                mainloop.producer_acquire(ps)
                bar = mainloop.producer_get_barrier(ps)
                cute.copy(tma_a, tAgA_m[(None, load_k)], tAsA[(None, load_index)], tma_bar_ptr=bar, mcast_mask=0)
                cute.copy(tma_b, tBgB_n[(None, load_k)], tBsB[(None, load_index)], tma_bar_ptr=bar)
                mainloop.producer_commit(ps)
                ps.advance()
                load_index = ps.index
                load_phase = ps.phase
                load_k += 1

        accum.fill(0.0)
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

        # ---- k == 0: no stage to release yet -----------------------------
        rs = pipeline.PipelineState(STAGES, Int32(0), read_index, read_phase)
        mainloop.consumer_wait(rs)
        cute.nvgpu.warpgroup.fence()
        for kb in cutlass.range_constexpr(num_k_blocks):
            cute.gemm(tiled_mma, accum, tCrA[(None, None, kb, read_index)], tCrB[(None, None, kb, read_index)], accum)
        cute.nvgpu.warpgroup.commit_group()
        rs.advance()
        read_index = rs.index
        read_phase = rs.phase

        # ---- steady state: release the previous stage, refill it ---------
        for _ in range(1, num_k_tiles):
            rs = pipeline.PipelineState(STAGES, Int32(0), read_index, read_phase)
            mainloop.consumer_wait(rs)
            cute.nvgpu.warpgroup.fence()
            for kb in cutlass.range_constexpr(num_k_blocks):
                cute.gemm(
                    tiled_mma, accum, tCrA[(None, None, kb, read_index)], tCrB[(None, None, kb, read_index)], accum
                )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(1)
            rst = pipeline.PipelineState(STAGES, Int32(0), rel_index, rel_phase)
            mainloop.consumer_release(rst)
            rst.advance()
            rel_index = rst.index
            rel_phase = rst.phase
            if warp == 0 and load_k < num_k_tiles:
                ps = pipeline.PipelineState(STAGES, Int32(0), load_index, load_phase)
                mainloop.producer_acquire(ps)
                bar = mainloop.producer_get_barrier(ps)
                cute.copy(tma_a, tAgA_m[(None, load_k)], tAsA[(None, load_index)], tma_bar_ptr=bar, mcast_mask=0)
                cute.copy(tma_b, tBgB_n[(None, load_k)], tBsB[(None, load_index)], tma_bar_ptr=bar)
                mainloop.producer_commit(ps)
                ps.advance()
                load_index = ps.index
                load_phase = ps.phase
                load_k += 1
            rs.advance()
            read_index = rs.index
            read_phase = rs.phase

        # ---- tile tail: the only full WGMMA drain ------------------------
        cute.nvgpu.warpgroup.wait_group(0)
        rst = pipeline.PipelineState(STAGES, Int32(0), rel_index, rel_phase)
        mainloop.consumer_release(rst)
        rst.advance()
        rel_index = rst.index
        rel_phase = rst.phase
        if warp == 0:
            ptail = pipeline.PipelineState(STAGES, Int32(0), load_index, load_phase)
            mainloop.producer_tail(ptail)

        # ---- epilogue: stage the FP32 WGMMA accumulator into plain SMEM via
        # the validated register -> (row, col) map, then drain SMEM -> GMEM with
        # fully coalesced stores.  The old path wrote each accumulator element
        # straight to GMEM through the strided WGMMA map, so a warp's 32 lanes
        # touched only a few columns per row (25% global store sector use);
        # staging through SMEM lets adjacent lanes write adjacent columns of one
        # row (100% sector use), which lifts the WGMMA tensor pipe from ~73% to
        # ~91% (logits) / ~84% (dW) active on H200.
        row_base = warp_group * 64 + local_warp * 16 + lane // 4
        col_base = (lane % 4) * 2
        c_dtype = c.element_type
        scale = Float32(output_scale[0]) if cutlass.const_expr(self.scale_output) else Float32(1.0)
        for i in cutlass.range_constexpr(n_accum):
            row = row_base + ((i % 4) // 2) * 8
            col = col_base + (i // 4) * 8 + (i % 2)
            sC[row, col] = c_dtype(accum[i] * scale)
        cute.arch.barrier()

        gC = cute.local_tile(c, (TILE_M, TILE_N), (pid_m, pid_n))
        # Element e = k * THREADS + tid: within any warp the 32 lanes cover 32
        # consecutive columns of the same row, so every store is coalesced.
        drain_iters = (TILE_M * TILE_N) // THREADS
        for k in cutlass.range_constexpr(drain_iters):
            e = k * THREADS + tid
            row = e // TILE_N
            col = e % TILE_N
            gC[row, col] = sC[row, col]

    @cute.kernel
    def kernel_persistent(
        self,
        tma_a: cute.CopyAtom,
        a: cute.Tensor,
        tma_b: cute.CopyAtom,
        b: cute.Tensor,
        c: cute.Tensor,
        c_store: cute.Tensor,
        output_scale: Optional[cute.Tensor],
        tiled_mma: cute.TiledMma,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        tma_c_store: cute.CopyAtom,
        store_smem_layout: cute.ComposedLayout,
    ):
        # Persistent one-wave scheduler for the dW GEMM.  ``num_sms`` CTAs are
        # launched; CTA ``cid`` owns the flattened output tiles ``cid, cid +
        # num_sms, cid + 2*num_sms, ...`` decoded m-outer / n-inner
        # (``pid_m = lin // grid_n``, ``pid_n = lin % grid_n``).  Because the CTAs
        # march the linear index in lockstep, every 132-tile window covers ~8 A
        # rows x all N columns, so each 1 MB A tile is fetched from DRAM once and
        # reused ``grid_n`` times out of L2 (and the whole B operand stays
        # resident), collapsing the dW DRAM footprint that the multi-wave launch
        # re-streamed 121 times.  The TMA load pipeline runs *continuously* across
        # tile boundaries (ring index/phase carried; compute drained only at each
        # epilogue), and the epilogue TMA-stores each 64-wide column sub-tile
        # asynchronously so the store overlaps the next tile's WGMMA mainloop
        # instead of stalling the tensor cores.
        num_sms = cutlass.const_expr(self.num_sms)
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group = cute.arch.make_warp_uniform(tid // 128)
        lane = tid % 32
        local_warp = (tid % 128) // 32
        block_x, _, _ = cute.arch.block_idx()
        cta_id = Int32(block_x)

        if warp == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_c_store)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.storage_t)
        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sD = storage.sD.get_tensor(store_smem_layout.outer, swizzle=store_smem_layout.inner)

        tile_shape = (TILE_M, TILE_N, TILE_K)
        gA = cute.local_tile(a, cute.slice_(tile_shape, (None, 0, None)), (None, None, None))
        gB = cute.local_tile(b, cute.slice_(tile_shape, (0, None, None)), (None, None, None))
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )
        # Output tiled into (TILE_M, DW_STORE_TILE_N) sub-tiles for the TMA store.
        # ``c_store`` is the store TMA's coord tensor (mirrors how the loads tile
        # the load TMAs' coord tensors ``a``/``b`` rather than the raw operands).
        gDW = cute.local_tile(c_store, (TILE_M, DW_STORE_TILE_N), (None, None))
        tDsD, tDgD = cute.nvgpu.cpasync.tma_partition(
            tma_c_store,
            0,
            cute.make_layout(1),
            cute.group_modes(sD, 0, 2),
            cute.group_modes(gDW, 0, 2),
        )

        tma_bytes = cute.size_in_bytes(BFloat16, cute.slice_(a_smem_layout, (None, None, 0))) + cute.size_in_bytes(
            BFloat16, cute.slice_(b_smem_layout, (None, None, 0))
        )

        mainloop = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.pipe.data_ptr(),
            num_stages=STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, NUM_MMA_WG * 4),
            tx_count=tma_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=(1, 1))

        store_bar = pipeline.NamedBarrier(barrier_id=DW_STORE_BARRIER_ID, num_threads=THREADS)

        num_k_tiles = cute.size(gA, mode=[3])
        num_k_blocks = TILE_K // 16

        cute.arch.setmaxregister_increase(240)
        mma_wg_layout = cute.make_layout(NUM_MMA_WG, stride=128)
        thr_mma = tiled_mma.get_slice(mma_wg_layout(Int32(warp_group)))
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
        accum = cute.make_rmem_tensor(thr_mma.partition_shape_C((TILE_M, TILE_N)), Float32)

        grid_m = cute.ceil_div(c.shape[0], TILE_M)
        grid_n = cute.ceil_div(c.shape[1], TILE_N)
        total_tiles = Int32(grid_m * grid_n)
        num_tiles_for_cta = (total_tiles - cta_id + Int32(num_sms - 1)) // Int32(num_sms)
        total_k = num_tiles_for_cta * Int32(num_k_tiles)

        # Epilogue constants (tile-independent).
        row_base = warp_group * 64 + local_warp * 16 + lane // 4
        col_base = (lane % 4) * 2
        c_dtype = c.element_type
        scale = Float32(output_scale[0]) if cutlass.const_expr(self.scale_output) else Float32(1.0)
        issuer = warp == 0
        # Per-thread FP32 accumulator registers that map into one 64-wide column
        # sub-tile: (TILE_M * DW_STORE_TILE_N) / THREADS.  For sub-tile ``sub``
        # these are the contiguous chunk ``accum[sub_frag*sub : +sub_frag]``.
        sub_frag = (TILE_M * DW_STORE_TILE_N) // THREADS

        load_index = Int32(0)
        load_phase = Int32(1)
        read_index = Int32(0)
        read_phase = Int32(0)
        rel_index = Int32(0)
        rel_phase = Int32(0)
        gload = Int32(0)
        gk_consumed = Int32(0)

        # ---- one-time prologue: STAGES loads (decoded from the global k idx) --
        if warp == 0:
            prologue = cutlass.min(Int32(STAGES), total_k)
            for _ in range(prologue):
                gtile = gload // Int32(num_k_tiles)
                gkk = gload % Int32(num_k_tiles)
                glin = cta_id + gtile * Int32(num_sms)
                gpm = glin // grid_n
                gpn = glin % grid_n
                ps = pipeline.PipelineState(STAGES, Int32(0), load_index, load_phase)
                mainloop.producer_acquire(ps)
                bar = mainloop.producer_get_barrier(ps)
                cute.copy(
                    tma_a,
                    tAgA[(None, gpm, None, 0)][(None, gkk)],
                    tAsA[(None, load_index)],
                    tma_bar_ptr=bar,
                    mcast_mask=0,
                )
                cute.copy(tma_b, tBgB[(None, gpn, None, 0)][(None, gkk)], tBsB[(None, load_index)], tma_bar_ptr=bar)
                mainloop.producer_commit(ps)
                ps.advance()
                load_index = ps.index
                load_phase = ps.phase
                gload += 1

        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

        # ---- persistent tile loop ---------------------------------------
        tile_ord = Int32(0)
        for _ in range(num_tiles_for_cta):
            lin = cta_id + tile_ord * Int32(num_sms)
            pid_m = lin // grid_n
            pid_n = lin % grid_n
            accum.fill(0.0)

            for _ in range(num_k_tiles):
                rs = pipeline.PipelineState(STAGES, Int32(0), read_index, read_phase)
                mainloop.consumer_wait(rs)
                cute.nvgpu.warpgroup.fence()
                for kb in cutlass.range_constexpr(num_k_blocks):
                    cute.gemm(
                        tiled_mma, accum, tCrA[(None, None, kb, read_index)], tCrB[(None, None, kb, read_index)], accum
                    )
                cute.nvgpu.warpgroup.commit_group()
                # Release the stage consumed on the previous global k step and
                # refill it, keeping exactly one WGMMA group in flight.
                if gk_consumed > 0:
                    cute.nvgpu.warpgroup.wait_group(1)
                    rst = pipeline.PipelineState(STAGES, Int32(0), rel_index, rel_phase)
                    mainloop.consumer_release(rst)
                    rst.advance()
                    rel_index = rst.index
                    rel_phase = rst.phase
                    if warp == 0 and gload < total_k:
                        gtile = gload // Int32(num_k_tiles)
                        gkk = gload % Int32(num_k_tiles)
                        glin = cta_id + gtile * Int32(num_sms)
                        gpm = glin // grid_n
                        gpn = glin % grid_n
                        ps = pipeline.PipelineState(STAGES, Int32(0), load_index, load_phase)
                        mainloop.producer_acquire(ps)
                        bar = mainloop.producer_get_barrier(ps)
                        cute.copy(
                            tma_a,
                            tAgA[(None, gpm, None, 0)][(None, gkk)],
                            tAsA[(None, load_index)],
                            tma_bar_ptr=bar,
                            mcast_mask=0,
                        )
                        cute.copy(
                            tma_b, tBgB[(None, gpn, None, 0)][(None, gkk)], tBsB[(None, load_index)], tma_bar_ptr=bar
                        )
                        mainloop.producer_commit(ps)
                        ps.advance()
                        load_index = ps.index
                        load_phase = ps.phase
                        gload += 1
                rs.advance()
                read_index = rs.index
                read_phase = rs.phase
                gk_consumed += 1

            # ---- tile tail: drain the last WGMMA, then async TMA store -----
            cute.nvgpu.warpgroup.wait_group(0)
            for sub in cutlass.range_constexpr(DW_SUBTILES):
                stage = sub % DW_STORE_STAGES
                # Throttle to <= DW_STORE_STAGES outstanding stores so this sD
                # stage is free to overwrite.
                if issuer:
                    cute.arch.cp_async_bulk_wait_group(DW_STORE_STAGES - 1, read=True)
                store_bar.arrive_and_wait()
                # Scatter this sub-tile's 64 columns of the FP32 accumulator into
                # the BF16 store buffer using the validated register->(row, col)
                # map (accum[sub_frag*sub : sub_frag*sub + sub_frag] == columns
                # [DW_STORE_TILE_N*sub, +DW_STORE_TILE_N)).
                for jj in cutlass.range_constexpr(sub_frag):
                    row = row_base + ((jj % 4) // 2) * 8
                    col = col_base + (jj // 4) * 8 + (jj % 2)
                    sD[row, col, stage] = c_dtype(accum[sub_frag * sub + jj] * scale)
                cute.arch.fence_proxy("async.shared", space="cta")
                store_bar.arrive_and_wait()
                if issuer:
                    cute.copy(
                        tma_c_store,
                        tDsD[(None, stage)],
                        tDgD[(None, pid_m, pid_n * DW_SUBTILES + sub)],
                    )
                    cute.arch.cp_async_bulk_commit_group()
            tile_ord += 1

        # ---- final trailing release + single pipeline drain -------------
        if gk_consumed > 0:
            rst = pipeline.PipelineState(STAGES, Int32(0), rel_index, rel_phase)
            mainloop.consumer_release(rst)
            rst.advance()
            rel_index = rst.index
            rel_phase = rst.phase
            if warp == 0:
                ptail = pipeline.PipelineState(STAGES, Int32(0), load_index, load_phase)
                mainloop.producer_tail(ptail)
        # Drain every outstanding TMA store before the kernel exits.
        if issuer:
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        store_bar.arrive_and_wait()
        if issuer:
            cute.arch.cp_async_bulk_wait_group(0)


# ---------------------------------------------------------------------------
# Cross entropy + dZ formation (warp per row)
# ---------------------------------------------------------------------------
class _CrossEntropyPartials:
    """Compute online-softmax state for one row/vocabulary partition per CTA."""

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        target: cute.Tensor,
        partials: cute.Tensor,
        stream: cuda.CUstream,
    ):
        m = target.shape[0]
        grid = (m, CE_PARTITIONS, 1)

        @cute.struct
        class Storage:
            scratch: cute.struct.Align[
                cute.struct.MemRange[Float32, CE_SCRATCH_VALUES],
                16,
            ]

        self.storage_t = Storage
        self.kernel(logits, target, partials).launch(
            grid=grid,
            block=(CE_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        logits: cute.Tensor,
        target: cute.Tensor,
        partials: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, partition, _ = cute.arch.block_idx()
        lane = tid % 32
        warp = tid // 32
        m = target.shape[0]
        v_size = logits.shape[1]

        if row < m:
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(self.storage_t)
            scratch = storage.scratch.get_tensor(cute.make_layout(CE_SCRATCH_VALUES))
            tgt = Int32(target[row])
            partition_size = cute.ceil_div(v_size, CE_PARTITIONS)
            partition_start = partition * partition_size
            partition_end = cutlass.min(partition_start + partition_size, v_size)

            run_max = Float32(NEG_INF_F32)
            run_sum = Float32(0.0)
            tgt_logit = Float32(0.0)
            v = partition_start + tid
            while v < partition_end:
                lg = Float32(logits[row, v])
                new_max = cute.arch.fmax(run_max, lg)
                run_sum = run_sum * cute.math.exp2((run_max - new_max) * LOG2_E, fastmath=True) + cute.math.exp2(
                    (lg - new_max) * LOG2_E, fastmath=True
                )
                run_max = new_max
                if v == tgt:
                    tgt_logit = lg
                v += CE_THREADS

            warp_max = cute.arch.warp_reduction_max(run_max)
            if lane == 0:
                scratch[warp] = warp_max
            cute.arch.sync_threads()

            if warp == 0:
                candidate_max = scratch[lane] if lane < WARPS_PER_CE_CTA else Float32(NEG_INF_F32)
                block_max = cute.arch.warp_reduction_max(candidate_max)
                if lane == 0:
                    scratch[3 * WARPS_PER_CE_CTA] = block_max
            cute.arch.sync_threads()

            block_max = scratch[3 * WARPS_PER_CE_CTA]
            local_sum = run_sum * cute.math.exp2(
                (run_max - block_max) * LOG2_E,
                fastmath=True,
            )
            warp_sum = cute.arch.warp_reduction_sum(local_sum)
            if lane == 0:
                scratch[WARPS_PER_CE_CTA + warp] = warp_sum
            cute.arch.sync_threads()

            warp_target = cute.arch.warp_reduction_sum(tgt_logit)
            if lane == 0:
                scratch[2 * WARPS_PER_CE_CTA + warp] = warp_target
            cute.arch.sync_threads()

            if warp == 0:
                candidate_sum = scratch[WARPS_PER_CE_CTA + lane] if lane < WARPS_PER_CE_CTA else Float32(0.0)
                candidate_target = scratch[2 * WARPS_PER_CE_CTA + lane] if lane < WARPS_PER_CE_CTA else Float32(0.0)
                block_sum = cute.arch.warp_reduction_sum(candidate_sum)
                block_target = cute.arch.warp_reduction_sum(candidate_target)
                if lane == 0:
                    partials[row, partition, 0] = block_max
                    partials[row, partition, 1] = block_sum
                    partials[row, partition, 2] = block_target


class _CrossEntropyDZ:
    """Merge partial softmax states and overwrite logits with scaled dZ."""

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        target: cute.Tensor,
        partials: cute.Tensor,
        nll: cute.Tensor,
        dz: cute.Tensor,
        row_scale: Float32,
        ignore_index: Int32,
        stream: cuda.CUstream,
    ):
        m = target.shape[0]
        grid = (m, 1, 1)

        @cute.struct
        class Storage:
            stats: cute.struct.Align[cute.struct.MemRange[Float32, 3], 16]

        self.storage_t = Storage
        self.kernel(logits, target, partials, nll, dz, row_scale, ignore_index).launch(
            grid=grid,
            block=(CE_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        logits: cute.Tensor,
        target: cute.Tensor,
        partials: cute.Tensor,
        nll: cute.Tensor,
        dz: cute.Tensor,
        row_scale: Float32,
        ignore_index: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = tid % 32
        warp = tid // 32
        m = target.shape[0]
        v_size = logits.shape[1]

        if row < m:
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(self.storage_t)
            stats = storage.stats.get_tensor(cute.make_layout(3))
            tgt = Int32(target[row])
            ignore = tgt == ignore_index

            if warp == 0:
                partial_max = partials[row, lane, 0] if lane < CE_PARTITIONS else Float32(NEG_INF_F32)
                block_max = cute.arch.warp_reduction_max(partial_max)
                partial_sum = (
                    partials[row, lane, 1]
                    * cute.math.exp2(
                        (partial_max - block_max) * LOG2_E,
                        fastmath=True,
                    )
                    if lane < CE_PARTITIONS
                    else Float32(0.0)
                )
                partial_target = partials[row, lane, 2] if lane < CE_PARTITIONS else Float32(0.0)
                block_sum = cute.arch.warp_reduction_sum(partial_sum)
                target_logit = cute.arch.warp_reduction_sum(partial_target)
                if lane == 0:
                    stats[0] = block_max
                    stats[1] = block_sum
                    stats[2] = target_logit
                    lse = block_max + cute.math.log2(block_sum, fastmath=True) * LN2
                    nll[row] = Float32(0.0) if ignore else lse - target_logit
            cute.arch.sync_threads()

            block_max = stats[0]
            inv_sum = 1.0 / stats[1]
            v = tid
            while v < v_size:
                lg = Float32(logits[row, v])
                g = cute.math.exp2((lg - block_max) * LOG2_E, fastmath=True) * inv_sum
                if v == tgt:
                    g = g - 1.0
                if ignore:
                    g = Float32(0.0)
                dz[row, v] = BFloat16(g * row_scale)
                v += CE_THREADS


class _CrossEntropyClusterVector:
    """One-read cluster CE using aligned 128-bit CuTe G2R/R2G tiled copies."""

    def __init__(self, partition_size):
        vector_tiles = partition_size // CE_VECTOR_TILE
        tail_values = partition_size - vector_tiles * CE_VECTOR_TILE
        self.partition_size = partition_size
        self.vector_tiles = vector_tiles
        self.tail_threads = tail_values // CE_VECTOR

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        target: cute.Tensor,
        nll: cute.Tensor,
        row_scale: Float32,
        ignore_index: Int32,
        stream: cuda.CUstream,
    ):
        g2r_atom = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            BFloat16,
            num_bits_per_copy=CE_VECTOR_BITS,
        )
        r2g_atom = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(),
            BFloat16,
            num_bits_per_copy=CE_VECTOR_BITS,
        )
        vector_copy_g2r = cute.make_tiled_copy_tv(
            g2r_atom,
            cute.make_layout((CE_THREADS, 1), stride=(1, CE_THREADS)),
            cute.make_layout((1, CE_VECTOR), stride=(CE_VECTOR, 1)),
        )
        vector_copy_r2g = cute.make_tiled_copy_tv(
            r2g_atom,
            cute.make_layout((CE_THREADS, 1), stride=(1, CE_THREADS)),
            cute.make_layout((1, CE_VECTOR), stride=(CE_VECTOR, 1)),
        )

        @cute.struct
        class Storage:
            scratch: cute.struct.Align[
                cute.struct.MemRange[Float32, WARPS_PER_CE_CTA],
                16,
            ]
            stats: cute.struct.Align[cute.struct.MemRange[Float32, 6], 16]

        self.storage_t = Storage
        self.kernel(
            logits,
            target,
            nll,
            row_scale,
            ignore_index,
            vector_copy_g2r,
            vector_copy_r2g,
        ).launch(
            grid=(CE_PARTITIONS * target.shape[0], 1, 1),
            block=(CE_THREADS, 1, 1),
            cluster=(CE_PARTITIONS, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _load_tail(
        self,
        g_partition: cute.Tensor,
        vector_copy_g2r: cute.TiledCopy,
        tid: Int32,
        retained: cute.Tensor,
    ):
        g_tail_linear = cute.local_tile(
            g_partition,
            (CE_VECTOR_TILE,),
            (self.vector_tiles,),
        )
        g_tail = cute.make_tensor(
            g_tail_linear.iterator,
            cute.make_layout((CE_THREADS, CE_VECTOR), stride=(CE_VECTOR, 1)),
        )
        tail_source = vector_copy_g2r.get_slice(tid).partition_S(g_tail)
        tail_fragment = cute.make_rmem_tensor_like(tail_source, BFloat16)
        cute.copy(vector_copy_g2r, tail_source, tail_fragment)
        tail_values = tail_fragment.load().to(Float32)
        for j in cutlass.range_constexpr(CE_VECTOR):
            retained[self.vector_tiles * CE_VECTOR + j] = tail_values[j]

    @cute.jit
    def _store_tail(
        self,
        g_partition: cute.Tensor,
        vector_copy_r2g: cute.TiledCopy,
        tid: Int32,
        retained: cute.Tensor,
        inv_sum: Float32,
        effective_scale: Float32,
    ):
        g_tail_linear = cute.local_tile(
            g_partition,
            (CE_VECTOR_TILE,),
            (self.vector_tiles,),
        )
        g_tail = cute.make_tensor(
            g_tail_linear.iterator,
            cute.make_layout((CE_THREADS, CE_VECTOR), stride=(CE_VECTOR, 1)),
        )
        tail_destination = vector_copy_r2g.get_slice(tid).partition_D(g_tail)
        tail_fragment = cute.make_rmem_tensor_like(tail_destination, BFloat16)
        for j in cutlass.range_constexpr(CE_VECTOR):
            tail_fragment[j] = BFloat16(retained[self.vector_tiles * CE_VECTOR + j] * inv_sum * effective_scale)
        cute.copy(vector_copy_r2g, tail_fragment, tail_destination)

    @cute.kernel
    def kernel(
        self,
        logits: cute.Tensor,
        target: cute.Tensor,
        nll: cute.Tensor,
        row_scale: Float32,
        ignore_index: Int32,
        vector_copy_g2r: cute.TiledCopy,
        vector_copy_r2g: cute.TiledCopy,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        partition = block % CE_PARTITIONS
        row = block // CE_PARTITIONS
        lane = tid % 32
        warp = tid // 32

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.storage_t)
        scratch = storage.scratch.get_tensor(cute.make_layout(WARPS_PER_CE_CTA))
        stats = storage.stats.get_tensor(cute.make_layout(6))

        pipeline_init_arrive(cluster_shape_mn=(CE_PARTITIONS, 1), is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=(CE_PARTITIONS, 1))

        partition_start = partition * self.partition_size
        partition_end = partition_start + self.partition_size
        tgt = Int32(target[row])
        ignore = tgt == ignore_index
        effective_scale = Float32(0.0) if ignore else row_scale

        g_row = logits[(row, None)]
        g_partition = cute.local_tile(
            g_row,
            (self.partition_size,),
            (partition,),
        )
        retained = cute.make_rmem_tensor((CE_VECTOR_VALUES_PER_THREAD,), Float32)
        retained.fill(Float32(NEG_INF_F32))
        thread_copy_g2r = vector_copy_g2r.get_slice(tid)

        for tile in cutlass.range_constexpr(self.vector_tiles):
            g_tile_linear = cute.local_tile(g_partition, (CE_VECTOR_TILE,), (tile,))
            g_tile = cute.make_tensor(
                g_tile_linear.iterator,
                cute.make_layout((CE_THREADS, CE_VECTOR), stride=(CE_VECTOR, 1)),
            )
            source = thread_copy_g2r.partition_S(g_tile)
            fragment = cute.make_rmem_tensor_like(source, BFloat16)
            cute.copy(vector_copy_g2r, source, fragment)
            values = fragment.load().to(Float32)
            for j in cutlass.range_constexpr(CE_VECTOR):
                retained[tile * CE_VECTOR + j] = values[j]

        if tid < self.tail_threads:
            self._load_tail(
                g_partition,
                vector_copy_g2r,
                tid,
                retained,
            )

        local_max = Float32(NEG_INF_F32)
        for i in cutlass.range_constexpr(CE_VECTOR_VALUES_PER_THREAD):
            local_max = cute.arch.fmax(local_max, retained[i])
        warp_max = cute.arch.warp_reduction_max(local_max)
        if lane == 0:
            scratch[warp] = warp_max
        cute.arch.sync_threads()
        if warp == 0:
            candidate = scratch[lane] if lane < WARPS_PER_CE_CTA else Float32(NEG_INF_F32)
            block_max = cute.arch.warp_reduction_max(candidate)
            if lane == 0:
                stats[0] = block_max
        cute.arch.sync_threads()

        # Peers publish their maxima before deterministic DSMEM reads.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if tid == 0:
            cluster_max = Float32(NEG_INF_F32)
            for peer in cutlass.range_constexpr(CE_PARTITIONS):
                peer_stats = _map_dsmem_addr(storage.stats.data_ptr(), peer)
                cluster_max = cute.arch.fmax(cluster_max, _dsmem_load_f32(peer_stats))
            stats[3] = cluster_max
        cute.arch.sync_threads()

        cluster_max = stats[3]
        local_sum = Float32(0.0)
        for i in cutlass.range_constexpr(CE_VECTOR_VALUES_PER_THREAD):
            exp_value = cute.math.exp2(
                (retained[i] - cluster_max) * LOG2_E,
                fastmath=True,
            )
            retained[i] = exp_value
            local_sum += exp_value
        warp_sum = cute.arch.warp_reduction_sum(local_sum)
        if lane == 0:
            scratch[warp] = warp_sum
        cute.arch.sync_threads()
        if warp == 0:
            candidate = scratch[lane] if lane < WARPS_PER_CE_CTA else Float32(0.0)
            block_sum = cute.arch.warp_reduction_sum(candidate)
            if lane == 0:
                stats[1] = block_sum
        if tid == 0:
            stats[2] = (
                Float32(logits[row, tgt])
                if tgt >= partition_start and tgt < partition_end and not ignore
                else Float32(0.0)
            )
        cute.arch.sync_threads()

        # Peers publish sums and target logits before the second DSMEM merge.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if tid == 0:
            cluster_sum = Float32(0.0)
            target_logit = Float32(0.0)
            for peer in cutlass.range_constexpr(CE_PARTITIONS):
                peer_stats = _map_dsmem_addr(storage.stats.data_ptr(), peer)
                cluster_sum += _dsmem_load_f32(peer_stats + 4)
                target_logit += _dsmem_load_f32(peer_stats + 8)
            stats[4] = cute.arch.rcp_approx(cluster_sum)
            if partition == 0:
                lse = cluster_max + cute.math.log2(cluster_sum, fastmath=True) * LN2
                nll[row] = Float32(0.0) if ignore else lse - target_logit
        cute.arch.sync_threads()

        # Keep every DSMEM window alive until every peer has finished reading it.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        inv_sum = stats[4]
        thread_copy_r2g = vector_copy_r2g.get_slice(tid)
        for tile in cutlass.range_constexpr(self.vector_tiles):
            g_tile_linear = cute.local_tile(g_partition, (CE_VECTOR_TILE,), (tile,))
            g_tile = cute.make_tensor(
                g_tile_linear.iterator,
                cute.make_layout((CE_THREADS, CE_VECTOR), stride=(CE_VECTOR, 1)),
            )
            destination = thread_copy_r2g.partition_D(g_tile)
            fragment = cute.make_rmem_tensor_like(destination, BFloat16)
            for j in cutlass.range_constexpr(CE_VECTOR):
                fragment[j] = BFloat16(retained[tile * CE_VECTOR + j] * inv_sum * effective_scale)
            cute.copy(vector_copy_r2g, fragment, destination)

        if tid < self.tail_threads:
            self._store_tail(
                g_partition,
                vector_copy_r2g,
                tid,
                retained,
                inv_sum,
                effective_scale,
            )

        # Select exactly one thread after every vector store has completed.
        cute.arch.sync_threads()
        if (
            not ignore
            and tgt >= partition_start
            and tgt < partition_end
            and tid == (tgt - partition_start) % CE_THREADS
        ):
            logits[row, tgt] = BFloat16(Float32(logits[row, tgt]) - row_scale)


# ---------------------------------------------------------------------------
# compile caches (compile once per (shape, layout) signature)
# ---------------------------------------------------------------------------
_gemm_cache = {}
_ce_partials_cache = {}
_ce_dz_cache = {}
_ce_cluster_vector_cache = {}


def _gemm_key(a, b, c, swap_grid, scale_output, persistent):
    return (
        tuple(a.shape),
        tuple(a.stride()),
        a.dtype,
        tuple(b.shape),
        tuple(b.stride()),
        b.dtype,
        tuple(c.shape),
        c.dtype,
        swap_grid,
        scale_output,
        persistent,
    )


def tile_gemm(
    a,
    b,
    c,
    a_leading,
    b_leading,
    stream,
    swap_grid=False,
    output_scale=None,
    persistent=False,
):
    """C[M,N] = A[M,K] @ B[N,K].T. ``*_leading`` names each operand's contiguous dim."""
    import os

    # Dispatch logits to the NVIDIA StaticPersistent WGMMA template for the
    # logits GEMM (K-major A+B, no output scale, default grid orientation).
    # The cluster-2 path raises tensor activity from 89.67% to 95.45% on H200.
    # Set FLCE_LOGITS_PERSISTENT=0 to retain the original static fallback.
    if (
        os.environ.get("FLCE_LOGITS_PERSISTENT", "1") != "0"
        and a_leading == 1
        and b_leading == 1
        and output_scale is None
        and not swap_grid
        and not persistent
    ):
        from liger_kernel.ops.cutedsl.ops._sm90_persistent_gemm import logits_persistent_gemm

        logits_persistent_gemm(a, b, c, a_leading, b_leading, stream)
        return

    # The same persistent cluster improves dW while preserving its scalar
    # autograd scale in the TMA epilogue. The old static raster remains
    # available independently of the legacy FLCE_DW_PERSISTENT experiment.
    if (
        os.environ.get("FLCE_DW_CLUSTERED_PERSISTENT", "1") != "0"
        and a_leading == 0
        and b_leading == 0
        and output_scale is not None
        and swap_grid
        and not persistent
    ):
        from liger_kernel.ops.cutedsl.ops._sm90_persistent_gemm import dw_persistent_gemm

        dw_persistent_gemm(a, b, c, a_leading, b_leading, stream, output_scale)
        return

    a_c = to_cute_tensor(a.unsqueeze(-1), leading_dim=a_leading, assumed_align=16)
    b_c = to_cute_tensor(b.unsqueeze(-1), leading_dim=b_leading, assumed_align=16)
    c_c = to_cute_tensor(c, assumed_align=(2 if c.dtype == torch.bfloat16 else 4))
    scale_c = to_cute_tensor(output_scale.reshape(1), assumed_align=4) if output_scale is not None else None
    scale_output = output_scale is not None
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count if persistent else 0
    key = _gemm_key(a, b, c, swap_grid, scale_output, persistent)
    compiled = _gemm_cache.get(key)
    if compiled is None:
        compiled = cute.compile(
            _TileGemmSM90(
                swap_grid=swap_grid,
                scale_output=scale_output,
                persistent=persistent,
                num_sms=num_sms,
            ),
            a_c,
            b_c,
            c_c,
            scale_c,
            stream,
        )
        _gemm_cache[key] = compiled
    compiled(a_c, b_c, c_c, scale_c, stream)


def cross_entropy_dz(logits, target, nll, row_scale, ignore_index, stream):
    """Fill ``nll[M]`` and overwrite ``logits`` in place with the scaled dZ."""
    import os

    partition_size = logits.shape[1] // CE_PARTITIONS
    use_vector_cluster = (
        os.environ.get("FLCE_CE_CLUSTER_VECTOR", "1") != "0"
        and logits.shape[1] % CE_PARTITIONS == 0
        and partition_size % CE_VECTOR == 0
        and partition_size <= CE_THREADS * CE_VECTOR_VALUES_PER_THREAD
    )
    l_c = (
        from_dlpack(logits.detach(), assumed_align=16)
        if use_vector_cluster
        else to_cute_tensor(logits, assumed_align=2)
    )
    t_c = to_cute_tensor(target, assumed_align=8)
    n_c = to_cute_tensor(nll, assumed_align=4)
    if use_vector_cluster:
        key = (tuple(logits.shape), tuple(target.shape), row_scale, ignore_index)
        compiled = _ce_cluster_vector_cache.get(key)
        args = (
            l_c,
            t_c,
            n_c,
            Float32(row_scale),
            Int32(ignore_index),
            stream,
        )
        if compiled is None:
            compiled = cute.compile(
                _CrossEntropyClusterVector(partition_size),
                *args,
            )
            _ce_cluster_vector_cache[key] = compiled
        compiled(*args)
        return

    partials = torch.empty(
        target.shape[0],
        CE_PARTITIONS,
        3,
        device=logits.device,
        dtype=torch.float32,
    )
    p_c = to_cute_tensor(partials, assumed_align=4)
    key = (tuple(logits.shape), tuple(target.shape))
    partials_compiled = _ce_partials_cache.get(key)
    partials_args = (l_c, t_c, p_c, stream)
    if partials_compiled is None:
        partials_compiled = cute.compile(_CrossEntropyPartials(), *partials_args)
        _ce_partials_cache[key] = partials_compiled
    partials_compiled(*partials_args)

    dz_compiled = _ce_dz_cache.get(key)
    dz_args = (
        l_c,
        t_c,
        p_c,
        n_c,
        l_c,
        Float32(row_scale),
        Int32(ignore_index),
        stream,
    )
    if dz_compiled is None:
        dz_compiled = cute.compile(_CrossEntropyDZ(), *dz_args)
        _ce_dz_cache[key] = dz_compiled
    dz_compiled(*dz_args)
