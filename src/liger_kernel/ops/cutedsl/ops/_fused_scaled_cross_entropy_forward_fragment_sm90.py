"""SM90 two-fragment fused scaled cross entropy forward.

Each cluster-M2 CTA computes an M128xN320 logical tile with a three-stage K64
pipeline. WG0 is the 24-register TMA producer and multicasts two N160 weight
panels to the cluster peer. WG1 and WG2 each own 64 token rows, two FP32 N160
accumulators, and the online-softmax epilogue.

The consumer read and release states are deliberately separate. Stage zero is
issued without a release. Each later stage is committed, ``wait_group(1)``
retires the previous WGMMA group, and only then is that previous AB stage
released. The tail uses ``wait_group(0)`` before the final release. This
one-stage lag prevents TMA from overwriting operands still consumed by WGMMA.

Each N160 accumulator is staged as two N80 slices through two padded FP16 SMEM
buffers, producing four online-softmax folds per logical N320 tile. Vocabulary
tiles are split over cluster pairs; each CTA writes compact FP32
``(max, sum, target[, weighted])`` statistics to HBM and a small finalizer
reduces the splits into per-token NLL/LSE and optional BF16 entropy. Entropy
uses the same online max rescaling as the softmax sum and is compiled out when
not requested.
"""

from dataclasses import dataclass
from dataclasses import replace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from cutlass import BFloat16
from cutlass import Float16
from cutlass import Float32
from cutlass import Int32
from cutlass import pipeline
from cutlass.pipeline.helpers import pipeline_init_arrive
from cutlass.pipeline.helpers import pipeline_init_wait
from cutlass.utils import LayoutEnum
from cutlass.utils import hopper_helpers

from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_utils_sm90 import HOPPER_MAX_SMEM_BYTES
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_utils_sm90 import LN2
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_utils_sm90 import LOG2_E
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_utils_sm90 import MASK_F32
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_utils_sm90 import NEG_INF_F32
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_utils_sm90 import _cute_stream
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_utils_sm90 import _fmax
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_utils_sm90 import _max_active_clusters
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_utils_sm90 import _pad_hidden
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_utils_sm90 import _validate
from liger_kernel.ops.cutedsl.ops.utils import to_cute_tensor

THREADS_PER_CTA = 384
PRODUCER_WARP_GROUP = 0
NUM_MMA_WARP_GROUPS = 2
WARPS_PER_WARP_GROUP = 4
TILE_M = 128
TILE_K = 64
STAGES = 3
CLUSTER_M = 2
ACCUMULATOR_N = 160
NUM_ACCUMULATORS = 2
LOGICAL_N = ACCUMULATOR_N * NUM_ACCUMULATORS
EPILOGUE_SLICE_N = 80
CHUNK_N = 40
LOGIT_BUFFERS = 2
LOGIT_PAD = 8
LOGIT_STRIDE = EPILOGUE_SLICE_N + LOGIT_PAD
EPILOGUE_BARRIER_ID = 4
PRODUCER_REGISTERS = 24
MMA_REGISTERS = 240
USABLE_REGISTER_BUDGET = 64512
SHARED_MEMORY_BYTES_BASE = (
    (TILE_M + LOGICAL_N) * TILE_K * 2 * STAGES
    + LOGIT_BUFFERS * TILE_M * LOGIT_STRIDE * 2
    + 3 * TILE_M * 4
    + 16 * STAGES
    + 3 * 1024
)

_compile_cache = {}


@dataclass(frozen=True)
class ScaledCEForwardFragmentConfig:
    """Vocabulary-split tuning for the fixed M128xN320 fragment kernel.

    ``return_entropy`` compile-time enables weighted softmax statistics.
    """

    fast_math: bool = True
    return_entropy: bool = False
    split_n: int = 0
    base_split_n: int = 0
    extra_m_pairs: int = 0
    target_cluster_pairs: int = 0
    max_split_n: int = 9

    def smem_bytes(self):
        return SHARED_MEMORY_BYTES_BASE + (TILE_M if self.return_entropy else 1) * 4

    def register_total(self):
        return 128 * PRODUCER_REGISTERS + NUM_MMA_WARP_GROUPS * 128 * MMA_REGISTERS


class _ScaledCEForwardFragmentSM90:
    def __init__(self, config: ScaledCEForwardFragmentConfig):
        self.fast_math = config.fast_math
        self.return_entropy = config.return_entropy
        self.split_n = config.split_n
        self.base_split_n = config.base_split_n
        self.extra_m_pairs = config.extra_m_pairs
        self.a_tile_shape = (TILE_M, ACCUMULATOR_N, TILE_K)
        self.b_tile_shape = (TILE_M, ACCUMULATOR_N, TILE_K)
        self.buffer_align = 1024
        self.epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=EPILOGUE_BARRIER_ID,
            num_threads=NUM_MMA_WARP_GROUPS * 128,
        )

    @staticmethod
    def _make_tma(tensor, smem_layout_staged, smem_tile, multicast=1):
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if multicast == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        return cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=multicast,
        )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        target: cute.Tensor,
        partial_max: cute.Tensor,
        partial_sum: cute.Tensor,
        partial_target: cute.Tensor,
        partial_weighted: cute.Tensor,
        inverse_temperature: Float32,
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
            (64, ACCUMULATOR_N),
        )
        a_smem_layout = hopper_helpers.make_smem_layout_a(
            x_layout,
            self.a_tile_shape,
            BFloat16,
            STAGES,
        )
        b_smem_layout = hopper_helpers.make_smem_layout_b(
            weight_layout,
            self.b_tile_shape,
            BFloat16,
            STAGES,
        )
        tma_x, tma_tensor_x = self._make_tma(x, a_smem_layout, (TILE_M, TILE_K))
        tma_w, tma_tensor_w = self._make_tma(
            weight,
            b_smem_layout,
            (ACCUMULATOR_N, TILE_K),
            multicast=CLUSTER_M,
        )

        @cute.struct
        class SharedStorage:
            pipeline: cute.struct.MemRange[cutlass.Int64, STAGES * 2]
            sX: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(a_smem_layout)],
                self.buffer_align,
            ]
            sW0: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(b_smem_layout)],
                self.buffer_align,
            ]
            sW1: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(b_smem_layout)],
                self.buffer_align,
            ]
            sLogits: cute.struct.Align[
                cute.struct.MemRange[
                    Float16,
                    LOGIT_BUFFERS * TILE_M * LOGIT_STRIDE,
                ],
                self.buffer_align,
            ]
            segment_max: cute.struct.Align[cute.struct.MemRange[Float32, TILE_M], 16]
            segment_sum: cute.struct.Align[cute.struct.MemRange[Float32, TILE_M], 16]
            segment_target: cute.struct.Align[cute.struct.MemRange[Float32, TILE_M], 16]
            segment_weighted: cute.struct.Align[
                cute.struct.MemRange[Float32, TILE_M if self.return_entropy else 1],
                16,
            ]

        self.shared_storage = SharedStorage
        num_m_tiles = cute.ceil_div(x.shape[0], TILE_M)
        num_m_pairs = cute.ceil_div(num_m_tiles, CLUSTER_M)
        num_cluster_pairs = num_m_pairs * self.base_split_n + self.extra_m_pairs
        grid = (CLUSTER_M, 1, num_cluster_pairs)
        self.kernel(
            tma_x,
            tma_tensor_x,
            tma_w,
            tma_tensor_w,
            target,
            partial_max,
            partial_sum,
            partial_target,
            partial_weighted,
            inverse_temperature,
            tiled_mma,
            a_smem_layout,
            b_smem_layout,
        ).launch(
            grid=grid,
            block=(THREADS_PER_CTA, 1, 1),
            cluster=(CLUSTER_M, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.jit
    def _stage_slice(
        self,
        accum: cute.Tensor,
        s_logits: cute.Tensor,
        buffer: Int32,
        row_base: Int32,
        col_base: Int32,
        slice_base: Int32,
    ):
        for i in cutlass.range_constexpr(cute.size(accum)):
            row = row_base + ((i % 4) // 2) * 8
            accum_col = col_base + (i // 4) * 8 + (i % 2)
            if accum_col >= slice_base and accum_col < slice_base + EPILOGUE_SLICE_N:
                s_logits[buffer, row, accum_col - slice_base] = Float16(accum[i])
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.fence_acq_rel_cta()
        self.epilogue_barrier.arrive_and_wait()

    @cute.jit
    def _fold_slice(
        self,
        s_logits: cute.Tensor,
        buffer: Int32,
        row: Int32,
        segment: Int32,
        slice_global_base: Int32,
        valid_cols: Int32,
        target_col: Int32,
        inverse_temperature: Float32,
        run_max: Float32,
        run_sum: Float32,
        run_target: Float32,
        run_weighted: Float32,
    ):
        segment_n = EPILOGUE_SLICE_N // 2
        segment_base = segment * segment_n
        values = cute.make_rmem_tensor((CHUNK_N,), Float32)
        for ci in cutlass.range_constexpr(segment_n // CHUNK_N):
            c0 = segment_base + ci * CHUNK_N
            chunk_max = Float32(MASK_F32)
            if c0 + CHUNK_N <= valid_cols:
                for j in cutlass.range_constexpr(CHUNK_N):
                    value = Float32(s_logits[buffer, row, c0 + j]) * inverse_temperature
                    values[j] = value
                    chunk_max = _fmax(chunk_max, value)
            else:
                for j in cutlass.range_constexpr(CHUNK_N):
                    value = Float32(s_logits[buffer, row, c0 + j]) * inverse_temperature
                    if c0 + j >= valid_cols:
                        value = Float32(MASK_F32)
                    values[j] = value
                    chunk_max = _fmax(chunk_max, value)

            new_max = _fmax(run_max, chunk_max)
            chunk_sum = Float32(0.0)
            if cutlass.const_expr(self.return_entropy):
                chunk_weighted = Float32(0.0)
                for j in cutlass.range_constexpr(CHUNK_N):
                    exp_value = cute.math.exp2(
                        (values[j] - new_max) * LOG2_E,
                        fastmath=self.fast_math,
                    )
                    chunk_sum += exp_value
                    chunk_weighted += exp_value * values[j]
                old_scale = cute.math.exp2(
                    (run_max - new_max) * LOG2_E,
                    fastmath=self.fast_math,
                )
                run_sum = run_sum * old_scale + chunk_sum
                run_weighted = run_weighted * old_scale + chunk_weighted
            else:
                for j in cutlass.range_constexpr(CHUNK_N):
                    chunk_sum += cute.math.exp2(
                        (values[j] - new_max) * LOG2_E,
                        fastmath=self.fast_math,
                    )
                run_sum = (
                    run_sum
                    * cute.math.exp2(
                        (run_max - new_max) * LOG2_E,
                        fastmath=self.fast_math,
                    )
                    + chunk_sum
                )
            run_max = new_max

        target_begin = slice_global_base + segment_base
        target_end = target_begin + segment_n
        if target_col >= target_begin and target_col < target_end:
            run_target += Float32(s_logits[buffer, row, target_col - slice_global_base]) * inverse_temperature
        return run_max, run_sum, run_target, run_weighted

    @cute.kernel
    def kernel(
        self,
        tma_x: cute.CopyAtom,
        x: cute.Tensor,
        tma_w: cute.CopyAtom,
        weight: cute.Tensor,
        target: cute.Tensor,
        output_partial_max: cute.Tensor,
        output_partial_sum: cute.Tensor,
        output_partial_target: cute.Tensor,
        output_partial_weighted: cute.Tensor,
        inverse_temperature: Float32,
        tiled_mma: cute.TiledMma,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group = cute.arch.make_warp_uniform(tid // 128)
        lane = tid % 32
        local_warp = (tid % 128) // 32
        local_tid = tid % 128
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        _, _, cluster_work = cute.arch.block_idx()

        num_m_tiles = cute.ceil_div(target.shape[0], TILE_M)
        if cutlass.const_expr(self.extra_m_pairs == 0):
            m_pair = cluster_work // self.base_split_n
            split_id = cluster_work % self.base_split_n
            split_count = self.base_split_n
        else:
            extra_work = self.extra_m_pairs * self.split_n
            m_pair = Int32(0)
            split_id = Int32(0)
            split_count = Int32(self.base_split_n)
            if cluster_work < extra_work:
                m_pair = cluster_work // self.split_n
                split_id = cluster_work % self.split_n
                split_count = Int32(self.split_n)
            else:
                normal_work = cluster_work - extra_work
                m_pair = self.extra_m_pairs + normal_work // self.base_split_n
                split_id = normal_work % self.base_split_n
                split_count = Int32(self.base_split_n)
        raw_pid_m = m_pair * CLUSTER_M + cluster_rank
        store_enabled = raw_pid_m < num_m_tiles
        pid_m = cutlass.min(raw_pid_m, num_m_tiles - 1)
        output_work = raw_pid_m * self.split_n + split_id

        if warp == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_x)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_w)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        s_x = storage.sX.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        s_w0 = storage.sW0.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        s_w1 = storage.sW1.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        s_logits = storage.sLogits.get_tensor(
            cute.make_layout(
                (LOGIT_BUFFERS, TILE_M, EPILOGUE_SLICE_N),
                stride=(TILE_M * LOGIT_STRIDE, LOGIT_STRIDE, 1),
            )
        )
        segment_max = storage.segment_max.get_tensor(cute.make_layout(TILE_M))
        segment_sum = storage.segment_sum.get_tensor(cute.make_layout(TILE_M))
        segment_target = storage.segment_target.get_tensor(cute.make_layout(TILE_M))
        segment_weighted = storage.segment_weighted.get_tensor(
            cute.make_layout(TILE_M if cutlass.const_expr(self.return_entropy) else 1)
        )

        g_x = cute.local_tile(
            x,
            cute.slice_(self.a_tile_shape, (None, 0, None)),
            (None, None, None),
        )
        g_w = cute.local_tile(
            weight,
            cute.slice_(self.b_tile_shape, (0, None, None)),
            (None, None, None),
        )
        one = cute.make_layout(1)
        t_x_s, t_x_g = cute.nvgpu.cpasync.tma_partition(
            tma_x,
            0,
            one,
            cute.group_modes(s_x, 0, 2),
            cute.group_modes(g_x, 0, 2),
        )

        cta_layout = cute.make_layout((CLUSTER_M, 1, 1))
        cluster_coord = cta_layout.get_flat_coord(cluster_rank)
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout, (None, 0, 0)).shape)
        b_mcast_mask = cute.make_layout_image_mask(cta_layout, cluster_coord, mode=0)
        b_mcast_rank = cluster_coord[0]
        t_w_s0, t_w_g0 = cute.nvgpu.cpasync.tma_partition(
            tma_w,
            b_mcast_rank,
            b_cta_layout,
            cute.group_modes(s_w0, 0, 2),
            cute.group_modes(g_w, 0, 2),
        )
        t_w_s1, t_w_g1 = cute.nvgpu.cpasync.tma_partition(
            tma_w,
            b_mcast_rank,
            b_cta_layout,
            cute.group_modes(s_w1, 0, 2),
            cute.group_modes(g_w, 0, 2),
        )
        x_stage_bytes = cute.size_in_bytes(BFloat16, cute.slice_(a_smem_layout, (None, None, 0)))
        w_stage_bytes = cute.size_in_bytes(BFloat16, cute.slice_(b_smem_layout, (None, None, 0)))
        mainloop = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.pipeline.data_ptr(),
            num_stages=STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                NUM_MMA_WARP_GROUPS * WARPS_PER_WARP_GROUP * CLUSTER_M,
            ),
            tx_count=x_stage_bytes + NUM_ACCUMULATORS * w_stage_bytes,
            cta_layout_vmnk=cute.make_layout((1, CLUSTER_M, 1, 1)),
            tidx=tid,
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(CLUSTER_M, 1), is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=(CLUSTER_M, 1))

        num_k_tiles = cute.size(g_x, mode=[3])
        num_logical_n_tiles = cute.ceil_div(weight.shape[0], LOGICAL_N)
        num_split_tiles = cute.ceil_div(num_logical_n_tiles - split_id, split_count)

        if warp_group == PRODUCER_WARP_GROUP:
            cute.arch.setmaxregister_decrease(PRODUCER_REGISTERS)
            load_index = Int32(0)
            load_phase = Int32(1)
            if warp == 0:
                t_x_g_m = t_x_g[(None, pid_m, None, 0)]
                for local_n in range(num_split_tiles):
                    logical_n = split_id + local_n * split_count
                    fragment_base = logical_n * NUM_ACCUMULATORS
                    for k_tile in range(num_k_tiles):
                        pstate = pipeline.PipelineState(
                            STAGES,
                            Int32(0),
                            load_index,
                            load_phase,
                        )
                        mainloop.producer_acquire(pstate)
                        barrier = mainloop.producer_get_barrier(pstate)
                        cute.copy(
                            tma_x,
                            t_x_g_m[(None, k_tile)],
                            t_x_s[(None, load_index)],
                            tma_bar_ptr=barrier,
                            mcast_mask=0,
                        )
                        cute.copy(
                            tma_w,
                            t_w_g0[(None, fragment_base + 0, None, 0)][(None, k_tile)],
                            t_w_s0[(None, load_index)],
                            tma_bar_ptr=barrier,
                            mcast_mask=b_mcast_mask,
                        )
                        cute.copy(
                            tma_w,
                            t_w_g1[(None, fragment_base + 1, None, 0)][(None, k_tile)],
                            t_w_s1[(None, load_index)],
                            tma_bar_ptr=barrier,
                            mcast_mask=b_mcast_mask,
                        )
                        mainloop.producer_commit(pstate)
                        pstate.advance()
                        load_index = pstate.index
                        load_phase = pstate.phase
                tail = pipeline.PipelineState(
                    STAGES,
                    Int32(0),
                    load_index,
                    load_phase,
                )
                mainloop.producer_tail(tail)
        else:
            cute.arch.setmaxregister_increase(MMA_REGISTERS)
            mma_wg = warp_group - 1
            mma_wg_layout = cute.make_layout(NUM_MMA_WARP_GROUPS, stride=128)
            thr_mma = tiled_mma.get_slice(mma_wg_layout(Int32(mma_wg)))
            r_x = tiled_mma.make_fragment_A(thr_mma.partition_A(s_x))
            r_w0 = tiled_mma.make_fragment_B(thr_mma.partition_B(s_w0))
            r_w1 = tiled_mma.make_fragment_B(thr_mma.partition_B(s_w1))
            accum_shape = thr_mma.partition_shape_C((TILE_M, ACCUMULATOR_N))
            accum0 = cute.make_rmem_tensor(accum_shape, Float32)
            accum1 = cute.make_rmem_tensor(accum_shape, Float32)

            row_base = mma_wg * 64 + local_warp * 16 + lane // 4
            col_base = (lane % 4) * 2
            epi_row = mma_wg * 64 + local_tid % 64
            epi_segment = local_tid // 64
            global_row = raw_pid_m * TILE_M + epi_row
            target_col = Int32(-1)
            if store_enabled and global_row < target.shape[0]:
                target_col = Int32(target[global_row])

            run_max = Float32(NEG_INF_F32)
            run_sum = Float32(0.0)
            run_target = Float32(0.0)
            run_weighted = Float32(0.0)
            read_index = Int32(0)
            read_phase = Int32(0)
            release_index = Int32(0)
            release_phase = Int32(0)
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            for local_n in range(num_split_tiles):
                logical_n = split_id + local_n * split_count
                accum0.fill(0.0)
                accum1.fill(0.0)

                # K stage 0: issue the first multi-fragment WGMMA group.  No AB
                # stage is released because no earlier group has retired.
                rstate = pipeline.PipelineState(
                    STAGES,
                    Int32(0),
                    read_index,
                    read_phase,
                )
                mainloop.consumer_wait(rstate)
                cute.nvgpu.warpgroup.fence()
                for k_block in cutlass.range_constexpr(TILE_K // 16):
                    coord = (None, None, k_block, read_index)
                    cute.gemm(tiled_mma, accum0, r_x[coord], r_w0[coord], accum0)
                    cute.gemm(tiled_mma, accum1, r_x[coord], r_w1[coord], accum1)
                cute.nvgpu.warpgroup.commit_group()
                rstate.advance()
                read_index = rstate.index
                read_phase = rstate.phase

                # The release state intentionally trails the read state.  Once
                # wait_group(1) retires the previous WGMMA group, and only
                # then, its AB stage is returned to the producer.
                for _ in range(1, num_k_tiles):
                    rstate = pipeline.PipelineState(
                        STAGES,
                        Int32(0),
                        read_index,
                        read_phase,
                    )
                    mainloop.consumer_wait(rstate)
                    cute.nvgpu.warpgroup.fence()
                    for k_block in cutlass.range_constexpr(TILE_K // 16):
                        coord = (None, None, k_block, read_index)
                        cute.gemm(tiled_mma, accum0, r_x[coord], r_w0[coord], accum0)
                        cute.gemm(tiled_mma, accum1, r_x[coord], r_w1[coord], accum1)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(1)
                    release_state = pipeline.PipelineState(
                        STAGES,
                        Int32(0),
                        release_index,
                        release_phase,
                    )
                    mainloop.consumer_release(release_state)
                    release_state.advance()
                    release_index = release_state.index
                    release_phase = release_state.phase
                    rstate.advance()
                    read_index = rstate.index
                    read_phase = rstate.phase

                # Drain the final live WGMMA group before releasing its stage.
                cute.nvgpu.warpgroup.wait_group(0)
                release_state = pipeline.PipelineState(
                    STAGES,
                    Int32(0),
                    release_index,
                    release_phase,
                )
                mainloop.consumer_release(release_state)
                release_state.advance()
                release_index = release_state.index
                release_phase = release_state.phase

                logical_base = logical_n * LOGICAL_N

                # Each N160 accumulator is folded as two N80 slices. Both
                # buffers are fully consumed before either is overwritten.
                self._stage_slice(accum0, s_logits, Int32(0), row_base, col_base, Int32(0))
                self._stage_slice(
                    accum0,
                    s_logits,
                    Int32(1),
                    row_base,
                    col_base,
                    Int32(EPILOGUE_SLICE_N),
                )
                run_max, run_sum, run_target, run_weighted = self._fold_slice(
                    s_logits,
                    Int32(0),
                    epi_row,
                    epi_segment,
                    logical_base,
                    Int32(weight.shape[0]) - logical_base,
                    target_col,
                    inverse_temperature,
                    run_max,
                    run_sum,
                    run_target,
                    run_weighted,
                )
                self.epilogue_barrier.arrive_and_wait()
                self._stage_slice(accum1, s_logits, Int32(0), row_base, col_base, Int32(0))
                run_max, run_sum, run_target, run_weighted = self._fold_slice(
                    s_logits,
                    Int32(1),
                    epi_row,
                    epi_segment,
                    logical_base + EPILOGUE_SLICE_N,
                    Int32(weight.shape[0]) - (logical_base + EPILOGUE_SLICE_N),
                    target_col,
                    inverse_temperature,
                    run_max,
                    run_sum,
                    run_target,
                    run_weighted,
                )
                self.epilogue_barrier.arrive_and_wait()
                self._stage_slice(
                    accum1,
                    s_logits,
                    Int32(1),
                    row_base,
                    col_base,
                    Int32(EPILOGUE_SLICE_N),
                )
                run_max, run_sum, run_target, run_weighted = self._fold_slice(
                    s_logits,
                    Int32(0),
                    epi_row,
                    epi_segment,
                    logical_base + ACCUMULATOR_N,
                    Int32(weight.shape[0]) - (logical_base + ACCUMULATOR_N),
                    target_col,
                    inverse_temperature,
                    run_max,
                    run_sum,
                    run_target,
                    run_weighted,
                )
                self.epilogue_barrier.arrive_and_wait()
                run_max, run_sum, run_target, run_weighted = self._fold_slice(
                    s_logits,
                    Int32(1),
                    epi_row,
                    epi_segment,
                    logical_base + ACCUMULATOR_N + EPILOGUE_SLICE_N,
                    Int32(weight.shape[0]) - (logical_base + ACCUMULATOR_N + EPILOGUE_SLICE_N),
                    target_col,
                    inverse_temperature,
                    run_max,
                    run_sum,
                    run_target,
                    run_weighted,
                )
                self.epilogue_barrier.arrive_and_wait()

            if epi_segment == 1:
                segment_max[epi_row] = run_max
                segment_sum[epi_row] = run_sum
                segment_target[epi_row] = run_target
                if cutlass.const_expr(self.return_entropy):
                    segment_weighted[epi_row] = run_weighted
            self.epilogue_barrier.arrive_and_wait()

            if epi_segment == 0 and store_enabled:
                other_max = segment_max[epi_row]
                combined_max = _fmax(run_max, other_max)
                if cutlass.const_expr(self.return_entropy):
                    run_scale = cute.math.exp2(
                        (run_max - combined_max) * LOG2_E,
                        fastmath=self.fast_math,
                    )
                    other_scale = cute.math.exp2(
                        (other_max - combined_max) * LOG2_E,
                        fastmath=self.fast_math,
                    )
                    combined_sum = run_sum * run_scale + segment_sum[epi_row] * other_scale
                    combined_weighted = run_weighted * run_scale + segment_weighted[epi_row] * other_scale
                else:
                    combined_sum = run_sum * cute.math.exp2(
                        (run_max - combined_max) * LOG2_E,
                        fastmath=self.fast_math,
                    ) + segment_sum[epi_row] * cute.math.exp2(
                        (other_max - combined_max) * LOG2_E,
                        fastmath=self.fast_math,
                    )
                output_partial_max[output_work, epi_row] = combined_max
                output_partial_sum[output_work, epi_row] = combined_sum
                output_partial_target[output_work, epi_row] = run_target + segment_target[epi_row]
                if cutlass.const_expr(self.return_entropy):
                    output_partial_weighted[output_work, epi_row] = combined_weighted


class _ScaledCEForwardFragmentFinalizeSM90:
    """Combine split vocabulary statistics into final NLL/LSE/entropy."""

    def __init__(self, split_n, base_split_n, extra_m_pairs, fast_math, return_entropy):
        self.split_n = split_n
        self.base_split_n = base_split_n
        self.extra_m_pairs = extra_m_pairs
        self.fast_math = fast_math
        self.return_entropy = return_entropy

    @cute.jit
    def __call__(
        self,
        partial_max: cute.Tensor,
        partial_sum: cute.Tensor,
        partial_target: cute.Tensor,
        partial_weighted: cute.Tensor,
        target: cute.Tensor,
        lse: cute.Tensor,
        token_loss: cute.Tensor,
        entropy: cute.Tensor,
        ignore_index: Int32,
        stream: cuda.CUstream,
    ):
        num_m_tiles = cute.ceil_div(target.shape[0], TILE_M)
        self.kernel(
            partial_max,
            partial_sum,
            partial_target,
            partial_weighted,
            target,
            lse,
            token_loss,
            entropy,
            ignore_index,
        ).launch(
            grid=(num_m_tiles, 1, 1),
            block=(TILE_M, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        partial_max: cute.Tensor,
        partial_sum: cute.Tensor,
        partial_target: cute.Tensor,
        partial_weighted: cute.Tensor,
        target: cute.Tensor,
        lse: cute.Tensor,
        token_loss: cute.Tensor,
        entropy: cute.Tensor,
        ignore_index: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        pid_m, _, _ = cute.arch.block_idx()
        row = pid_m * TILE_M + tid
        if cutlass.const_expr(self.extra_m_pairs == 0):
            split_count = self.base_split_n
        else:
            m_pair = pid_m // CLUSTER_M
            split_count = Int32(self.base_split_n)
            if m_pair < self.extra_m_pairs:
                split_count = Int32(self.split_n)
        row_max = Float32(NEG_INF_F32)
        row_target = Float32(0.0)
        for split_id in cutlass.range_constexpr(self.split_n):
            if split_id < split_count:
                work_id = pid_m * self.split_n + split_id
                row_max = _fmax(row_max, partial_max[work_id, tid])
                row_target += partial_target[work_id, tid]

        row_sum = Float32(0.0)
        row_weighted = Float32(0.0)
        for split_id in cutlass.range_constexpr(self.split_n):
            if split_id < split_count:
                work_id = pid_m * self.split_n + split_id
                if cutlass.const_expr(self.return_entropy):
                    split_scale = cute.math.exp2(
                        (partial_max[work_id, tid] - row_max) * LOG2_E,
                        fastmath=self.fast_math,
                    )
                    row_sum += partial_sum[work_id, tid] * split_scale
                    row_weighted += partial_weighted[work_id, tid] * split_scale
                else:
                    row_sum += partial_sum[work_id, tid] * cute.math.exp2(
                        (partial_max[work_id, tid] - row_max) * LOG2_E,
                        fastmath=self.fast_math,
                    )

        if row < target.shape[0]:
            row_lse = row_max + cute.math.log2(row_sum, fastmath=self.fast_math) * LN2
            lse[row] = row_lse
            if target[row] == ignore_index:
                token_loss[row] = Float32(0.0)
                if cutlass.const_expr(self.return_entropy):
                    entropy[row] = BFloat16(0.0)
            else:
                token_loss[row] = row_lse - row_target
                if cutlass.const_expr(self.return_entropy):
                    entropy[row] = BFloat16(row_lse - row_weighted / row_sum)


def _resolve_split(config, tokens, vocab_size):
    num_m_tiles = (tokens + TILE_M - 1) // TILE_M
    num_m_pairs = (num_m_tiles + CLUSTER_M - 1) // CLUSTER_M
    num_logical_n_tiles = (vocab_size + LOGICAL_N - 1) // LOGICAL_N
    if config.base_split_n:
        return (
            config.split_n,
            config.base_split_n,
            config.extra_m_pairs,
            num_m_tiles,
            num_logical_n_tiles,
        )
    if config.split_n:
        return config.split_n, config.split_n, 0, num_m_tiles, num_logical_n_tiles

    target_cluster_pairs = config.target_cluster_pairs or _max_active_clusters(CLUSTER_M)
    max_cluster_pairs = num_m_pairs * min(config.max_split_n, num_logical_n_tiles)
    num_cluster_pairs = max(
        num_m_pairs,
        min(target_cluster_pairs, max_cluster_pairs),
    )
    base_split_n, extra_m_pairs = divmod(num_cluster_pairs, num_m_pairs)
    split_n = base_split_n + int(extra_m_pairs != 0)
    return split_n, base_split_n, extra_m_pairs, num_m_tiles, num_logical_n_tiles


def scaled_ce_forward_fragment(
    _input,
    weight,
    target,
    temperature=1.0,
    ignore_index=-100,
    return_entropy=False,
    config: ScaledCEForwardFragmentConfig = None,
):
    """Run the fixed two-N160/four-N80 fragment forward."""
    _validate(_input, weight, target)
    if config is None:
        config = ScaledCEForwardFragmentConfig()
    if config.return_entropy != return_entropy:
        config = replace(config, return_entropy=return_entropy)
    if min(config.split_n, config.base_split_n, config.extra_m_pairs, config.target_cluster_pairs) < 0:
        raise ValueError("split and cluster tuning values must be non-negative")
    if config.max_split_n < 1:
        raise ValueError("max_split_n must be positive")
    if config.extra_m_pairs and not config.base_split_n:
        raise ValueError("extra_m_pairs requires an explicit base_split_n")
    if config.smem_bytes() > HOPPER_MAX_SMEM_BYTES:
        raise ValueError(
            f"fragment kernel needs approximately {config.smem_bytes()} B of shared memory, "
            f"over the {HOPPER_MAX_SMEM_BYTES} B Hopper limit"
        )
    if config.register_total() > USABLE_REGISTER_BUDGET:
        raise ValueError(
            f"register budget {config.register_total()} exceeds the usable {USABLE_REGISTER_BUDGET}-register budget"
        )

    target = target.contiguous()
    x_padded, weight_padded, _ = _pad_hidden(_input, weight, TILE_K)
    tokens = x_padded.shape[0]
    split_n, base_split_n, extra_m_pairs, num_m_tiles, num_logical_n_tiles = _resolve_split(
        config,
        tokens,
        weight_padded.shape[0],
    )
    if base_split_n < 1 or split_n > num_logical_n_tiles:
        raise ValueError("split_n must be between 1 and the number of logical vocabulary tiles")
    num_m_pairs = (num_m_tiles + CLUSTER_M - 1) // CLUSTER_M
    if extra_m_pairs >= num_m_pairs or split_n != base_split_n + int(extra_m_pairs != 0):
        raise ValueError("uneven split tuning must use split_n=base_split_n+1 for a proper subset of M pairs")
    config = replace(
        config,
        split_n=split_n,
        base_split_n=base_split_n,
        extra_m_pairs=extra_m_pairs,
    )

    partial_shape = (num_m_tiles * split_n, TILE_M)
    partial_max = torch.empty(partial_shape, device=_input.device, dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    partial_target = torch.empty_like(partial_max)
    partial_weighted = (
        torch.empty_like(partial_max)
        if return_entropy
        else torch.empty(
            1,
            device=_input.device,
            dtype=torch.float32,
        )
    )
    lse = torch.empty(tokens, device=_input.device, dtype=torch.float32)
    token_loss = torch.empty(tokens, device=_input.device, dtype=torch.float32)
    entropy = torch.empty(
        tokens if return_entropy else 1,
        device=_input.device,
        dtype=torch.bfloat16,
    )

    x_cute = to_cute_tensor(x_padded.unsqueeze(-1), leading_dim=1, assumed_align=16)
    weight_cute = to_cute_tensor(weight_padded.unsqueeze(-1), leading_dim=1, assumed_align=16)
    target_cute = to_cute_tensor(target, assumed_align=8)
    partial_max_cute = to_cute_tensor(partial_max, leading_dim=1, assumed_align=4)
    partial_sum_cute = to_cute_tensor(partial_sum, leading_dim=1, assumed_align=4)
    partial_target_cute = to_cute_tensor(partial_target, leading_dim=1, assumed_align=4)
    partial_weighted_cute = to_cute_tensor(
        partial_weighted,
        leading_dim=1 if return_entropy else None,
        assumed_align=4,
    )
    lse_cute = to_cute_tensor(lse, assumed_align=4)
    token_loss_cute = to_cute_tensor(token_loss, assumed_align=4)
    entropy_cute = to_cute_tensor(entropy, assumed_align=2)
    stream = _cute_stream()
    inverse_temperature = Float32(1.0 / temperature)

    key = ("scaled_ce_forward_fragment", config)
    compiled = _compile_cache.get(key)
    if compiled is None:
        compiled = cute.compile(
            _ScaledCEForwardFragmentSM90(config),
            x_cute,
            weight_cute,
            target_cute,
            partial_max_cute,
            partial_sum_cute,
            partial_target_cute,
            partial_weighted_cute,
            inverse_temperature,
            stream,
        )
        _compile_cache[key] = compiled
    compiled(
        x_cute,
        weight_cute,
        target_cute,
        partial_max_cute,
        partial_sum_cute,
        partial_target_cute,
        partial_weighted_cute,
        inverse_temperature,
        stream,
    )

    finalize_key = (
        "scaled_ce_forward_fragment_finalize",
        split_n,
        base_split_n,
        extra_m_pairs,
        config.fast_math,
        config.return_entropy,
    )
    finalize = _compile_cache.get(finalize_key)
    if finalize is None:
        finalize = cute.compile(
            _ScaledCEForwardFragmentFinalizeSM90(
                split_n,
                base_split_n,
                extra_m_pairs,
                config.fast_math,
                config.return_entropy,
            ),
            partial_max_cute,
            partial_sum_cute,
            partial_target_cute,
            partial_weighted_cute,
            target_cute,
            lse_cute,
            token_loss_cute,
            entropy_cute,
            Int32(ignore_index),
            stream,
        )
        _compile_cache[finalize_key] = finalize
    finalize(
        partial_max_cute,
        partial_sum_cute,
        partial_target_cute,
        partial_weighted_cute,
        target_cute,
        lse_cute,
        token_loss_cute,
        entropy_cute,
        Int32(ignore_index),
        stream,
    )
    return token_loss, entropy if return_entropy else None, lse
