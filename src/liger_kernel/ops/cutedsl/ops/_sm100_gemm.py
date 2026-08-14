# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Reusable SM100 persistent GEMM with operator-defined fragment epilogues."""

import fcntl
import functools
import inspect
import os
import tempfile

from contextlib import contextmanager
from contextlib import nullcontext
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch

from cutlass import Float32
from cutlass import const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive
from cutlass.pipeline import pipeline_init_wait
from cutlass.utils.blackwell_helpers import get_smem_store_op
from cutlass.utils.blackwell_helpers import get_tmem_load_op

from liger_kernel.ops.cutedsl.ops.utils import torch2cute_dtype_map

try:
    import tvm_ffi  # noqa: F401

    _TVM_FFI_AVAILABLE = True
except ImportError:
    _TVM_FFI_AVAILABLE = False

_MMA_M = 256
_MMA_N = 256
_MMA_K = 64
_CTA_M = _MMA_M // 2
_EPI_N = 32
_MMA_TILER = (_MMA_M, _MMA_N, _MMA_K)
_CTA_TILE = (_CTA_M, _MMA_N, _MMA_K)
_EPI_TILE = (_CTA_M, _EPI_N)
_CLUSTER_SHAPE_MN = (2, 1)

_NUM_AB_STAGES = 6
_NUM_ACC_STAGES = 2
_NUM_OUT_STAGES = 2
_NUM_TMEM_COLS = 512

# Four epilogue warps plus one dedicated UMMA warp and one TMA warp: six total.
# No warp is idle, so rounding the launch to eight would only consume resources.
_EPILOGUE_WARPS = (0, 1, 2, 3)
_MMA_WARP = 4
_TMA_WARP = 5
_THREADS = 32 * 6

_TMEM_ALLOC_BARRIER = 1
_TMEM_DEALLOC_BARRIER = 2
_EPILOGUE_BARRIER = 3

K_ALIGNMENT = _MMA_K

__all__ = [
    "K_ALIGNMENT",
    "run_epilogue_gemm",
]


def _device_guard(device):
    if device.index is None or device.index == torch.cuda.current_device():
        return nullcontext()
    return torch.cuda.device(device)


@cute.jit
def _run_tma_load(
    tma_atom_a,
    tma_atom_b,
    t_ag_a,
    t_as_a,
    t_bg_b,
    t_bs_b,
    a_full_mcast_mask,
    b_full_mcast_mask,
    ab_producer,
    tile_sched,
    work_tile,
    k_tile_count,
    atom_thr_size,
):
    while work_tile.is_valid_tile:
        tile_coord = work_tile.tile_idx
        m_tile = tile_coord[0] // atom_thr_size
        n_tile = tile_coord[1]
        batch_tile = tile_coord[2]
        t_ag_a_slice = t_ag_a[(None, m_tile, None, batch_tile)]
        t_bg_b_slice = t_bg_b[(None, n_tile, None, batch_tile)]
        ab_producer.reset()
        peek_ab_empty_status = ab_producer.try_acquire()
        for _ in cutlass.range(0, k_tile_count, 1, unroll=1):
            handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
            cute.copy(
                tma_atom_a,
                t_ag_a_slice[(None, handle.count)],
                t_as_a[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=a_full_mcast_mask,
            )
            cute.copy(
                tma_atom_b,
                t_bg_b_slice[(None, handle.count)],
                t_bs_b[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=b_full_mcast_mask,
            )
            peek_ab_empty_status = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_count:
                peek_ab_empty_status = ab_producer.try_acquire()
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    ab_producer.tail()


@cute.jit
def _run_umma_mainloop(
    tiled_mma,
    t_cr_a,
    t_cr_b,
    t_ct_acc_fake,
    tmem,
    ab_consumer,
    acc_pipeline,
    tile_sched,
    work_tile,
    k_tile_count,
    is_leader_cta,
):
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(Float32)
    t_ct_acc_base = cute.make_tensor(tmem_ptr, t_ct_acc_fake.layout)
    acc_producer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer,
        _NUM_ACC_STAGES,
    )

    while work_tile.is_valid_tile:
        t_ct_acc = t_ct_acc_base[(None, None, None, acc_producer_state.index)]
        ab_consumer.reset()
        peek_ab_full_status = cutlass.Boolean(1)
        if is_leader_cta:
            peek_ab_full_status = ab_consumer.try_wait()
            acc_pipeline.producer_acquire(acc_producer_state)

        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for _ in range(k_tile_count):
            if is_leader_cta:
                handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                for k_block in cutlass.range(
                    cute.size(t_cr_a, mode=[2]),
                    unroll_full=True,
                ):
                    coord = (None, None, k_block, handle.index)
                    cute.gemm(
                        tiled_mma,
                        t_ct_acc,
                        t_cr_a[coord],
                        t_cr_b[coord],
                        t_ct_acc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                handle.release()
                peek_ab_full_status = cutlass.Boolean(1)
                if handle.count + 1 < k_tile_count:
                    peek_ab_full_status = ab_consumer.try_wait()

        if is_leader_cta:
            acc_pipeline.producer_commit(acc_producer_state)
        acc_producer_state.advance()
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    acc_pipeline.producer_tail(acc_producer_state)


@cute.jit
def _make_tma_epilogue_partitions(tidx, t_acc, t_cg_c, s_out, io_dtype):
    """Partition accumulator fragments for fused conversion and TMA output."""
    copy_atom_t2r = get_tmem_load_op(
        _CTA_TILE,
        utils.LayoutEnum.ROW_MAJOR,
        io_dtype,
        Float32,
        _EPI_TILE,
        True,
    )
    t_acc_epi = cute.flat_divide(t_acc[((None, None), 0, 0, None)], _EPI_TILE)
    tiled_copy_t2r = tcgen05.make_tmem_copy(
        copy_atom_t2r,
        t_acc_epi[(None, None, 0, 0, 0)],
    )
    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
    t_tr_t_acc = thr_copy_t2r.partition_S(t_acc_epi)
    t_cg_c_epi = cute.flat_divide(
        t_cg_c[((None, None), 0, 0, None, None, None)],
        _EPI_TILE,
    )
    t_tr_g_c = thr_copy_t2r.partition_D(t_cg_c_epi)
    t_tr_r_acc = cute.make_rmem_tensor(
        t_tr_g_c[(None, None, None, 0, 0, 0, 0, 0)].shape,
        Float32,
    )
    t_tr_r_out = cute.make_rmem_tensor(t_tr_r_acc.shape, io_dtype)

    copy_atom_r2s = get_smem_store_op(
        utils.LayoutEnum.ROW_MAJOR,
        io_dtype,
        Float32,
        tiled_copy_t2r,
    )
    tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
    thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
    return (
        tiled_copy_t2r,
        t_tr_t_acc,
        t_tr_g_c,
        t_tr_r_acc,
        tiled_copy_r2s,
        tiled_copy_r2s.retile(t_tr_r_acc),
        tiled_copy_r2s.retile(t_tr_r_out),
        thr_copy_r2s.partition_D(s_out),
    )


@cute.kernel
def _kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    m_a_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    m_b_nkl: cute.Tensor,
    tma_atom_out: cute.CopyAtom,
    m_out_mnl: cute.Tensor,
    m_out_direct_mnl: cute.Tensor,
    cluster_layout_vmnk: cute.Layout,
    tile_sched_params: utils.PersistentTileSchedulerParams,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    out_smem_layout_staged: cute.ComposedLayout,
    num_ab_stages: cutlass.Constexpr,
    io_dtype: cutlass.Constexpr,
    epilogue: cutlass.Constexpr,
    use_tma_output: cutlass.Constexpr,
):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    if warp_idx == _TMA_WARP:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        if const_expr(use_tma_output):
            cpasync.prefetch_descriptor(tma_atom_out)

    atom_thr_size = cute.size(tiled_mma.thr_id.shape)
    bidx, _, _ = cute.arch.block_idx()
    mma_tile_coord_v = bidx % atom_thr_size
    is_leader_cta = mma_tile_coord_v == 0
    cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

    @cute.struct
    class SharedStorage:
        ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stages * 2]
        acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, _NUM_ACC_STAGES * 2]
        tmem_dealloc_mbar: cutlass.Int64
        tmem_holding_buf: cutlass.Int32

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    num_mcast_ctas_a = cute.size(cluster_layout_vmnk.shape[2])
    num_mcast_ctas_b = cute.size(cluster_layout_vmnk.shape[1])
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
        num_stages=num_ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            num_mcast_ctas_a + num_mcast_ctas_b - 1,
        ),
        tx_count=(
            cute.size_in_bytes(
                io_dtype,
                cute.slice_(a_smem_layout_staged, (None, None, None, 0)),
            )
            + cute.size_in_bytes(
                io_dtype,
                cute.slice_(b_smem_layout_staged, (None, None, None, 0)),
            )
        )
        * atom_thr_size,
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    ).make_participants()

    acc_pipeline = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
        num_stages=_NUM_ACC_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len(_EPILOGUE_WARPS) * atom_thr_size,
        ),
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    )

    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=_TMEM_ALLOC_BARRIER,
        num_threads=32 * len((_MMA_WARP, *_EPILOGUE_WARPS)),
    )
    tmem_dealloc_barrier = pipeline.NamedBarrier(
        barrier_id=_TMEM_DEALLOC_BARRIER,
        num_threads=32 * len(_EPILOGUE_WARPS),
    )
    epilogue_barrier = pipeline.NamedBarrier(
        barrier_id=_EPILOGUE_BARRIER,
        num_threads=32 * len(_EPILOGUE_WARPS),
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=_EPILOGUE_WARPS[0],
        is_two_cta=True,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )

    pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

    s_a = smem.allocate_tensor(
        element_type=io_dtype,
        layout=a_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=a_smem_layout_staged.inner,
    )
    s_b = smem.allocate_tensor(
        element_type=io_dtype,
        layout=b_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=b_smem_layout_staged.inner,
    )
    s_out = smem.allocate_tensor(
        element_type=io_dtype,
        layout=out_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=out_smem_layout_staged.inner,
    )

    a_full_mcast_mask = cpasync.create_tma_multicast_mask(
        cluster_layout_vmnk,
        block_in_cluster_coord_vmnk,
        mcast_mode=2,
    )
    b_full_mcast_mask = cpasync.create_tma_multicast_mask(
        cluster_layout_vmnk,
        block_in_cluster_coord_vmnk,
        mcast_mode=1,
    )

    g_a_mkl = cute.local_tile(
        m_a_mkl,
        cute.slice_(_MMA_TILER, (None, 0, None)),
        (None, None, None),
    )
    g_b_nkl = cute.local_tile(
        m_b_nkl,
        cute.slice_(_MMA_TILER, (0, None, None)),
        (None, None, None),
    )
    k_tile_count = cute.size(g_a_mkl, mode=[3])

    thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
    t_cg_a = thr_mma.partition_A(g_a_mkl)
    t_cg_b = thr_mma.partition_B(g_b_nkl)
    m_c_mnl = cute.make_identity_tensor((m_a_mkl.shape[0], m_b_nkl.shape[0], 1))
    g_c_mnl = cute.local_tile(
        m_c_mnl,
        cute.slice_(_MMA_TILER, (None, None, 0)),
        (None, None, None),
    )
    t_cg_c = thr_mma.partition_C(g_c_mnl)
    g_out_mnl = cute.local_tile(
        m_out_mnl,
        cute.slice_(_MMA_TILER, (None, None, 0)),
        (None, None, None),
    )
    t_cg_out = thr_mma.partition_C(g_out_mnl)

    a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
    t_as_a, t_ag_a = cpasync.tma_partition(
        tma_atom_a,
        block_in_cluster_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(s_a, 0, 3),
        cute.group_modes(t_cg_a, 0, 3),
    )
    b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
    t_bs_b, t_bg_b = cpasync.tma_partition(
        tma_atom_b,
        block_in_cluster_coord_vmnk[1],
        b_cta_layout,
        cute.group_modes(s_b, 0, 3),
        cute.group_modes(t_cg_b, 0, 3),
    )

    t_cr_a = tiled_mma.make_fragment_A(s_a)
    t_cr_b = tiled_mma.make_fragment_B(s_b)
    acc_shape = tiled_mma.partition_shape_C(_MMA_TILER[:2])
    t_ct_acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, _NUM_ACC_STAGES))

    pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
    tile_sched = utils.StaticPersistentTileScheduler.create(
        tile_sched_params,
        cute.arch.block_idx(),
        cute.arch.grid_dim(),
    )
    work_tile = tile_sched.initial_work_tile_info()

    if warp_idx == _TMA_WARP:
        _run_tma_load(
            tma_atom_a,
            tma_atom_b,
            t_ag_a,
            t_as_a,
            t_bg_b,
            t_bs_b,
            a_full_mcast_mask,
            b_full_mcast_mask,
            ab_producer,
            tile_sched,
            work_tile,
            k_tile_count,
            atom_thr_size,
        )

    if warp_idx == _MMA_WARP:
        _run_umma_mainloop(
            tiled_mma,
            t_cr_a,
            t_cr_b,
            t_ct_acc_fake,
            tmem,
            ab_consumer,
            acc_pipeline,
            tile_sched,
            work_tile,
            k_tile_count,
            is_leader_cta,
        )

    if warp_idx < _MMA_WARP:
        tmem.allocate(_NUM_TMEM_COLS)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        t_ct_acc_base = cute.make_tensor(tmem_ptr, t_ct_acc_fake.layout)
        (
            tiled_copy_t2r,
            t_tr_t_acc_base,
            t_tr_g_c_base,
            t_tr_r_acc,
            tiled_copy_r2s,
            t_rs_r_acc,
            t_rs_r_out,
            t_rs_s_out,
        ) = _make_tma_epilogue_partitions(tidx, t_ct_acc_base, t_cg_c, s_out, io_dtype)
        t_cg_out_transformed = utils.gemm.sm100.transform_partitioned_tensor_layout(t_cg_out)
        t_cg_out_epi = cute.flat_divide(t_cg_out_transformed, _EPI_TILE)
        b_sg_s_out, b_sg_g_out_partitioned = cpasync.tma_partition(
            tma_atom_out,
            0,
            cute.make_layout(1),
            cute.group_modes(s_out, 0, 2),
            cute.group_modes(t_cg_out_epi, 0, 2),
        )
        out_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=_NUM_OUT_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(_EPILOGUE_WARPS),
            ),
        )

        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            _NUM_ACC_STAGES,
        )
        m = m_a_mkl.shape[0]
        n = m_b_nkl.shape[0]

        while work_tile.is_valid_tile:
            tile_coord = work_tile.tile_idx
            m_tile = tile_coord[0] // atom_thr_size
            n_tile = tile_coord[1]
            batch_tile = tile_coord[2]
            t_tr_g_c = t_tr_g_c_base[(None, None, None, None, None, m_tile, n_tile, batch_tile)]
            t_tr_g_c = cute.group_modes(t_tr_g_c, 3, cute.rank(t_tr_g_c))
            b_sg_g_out = b_sg_g_out_partitioned[(None, None, None, m_tile, n_tile, batch_tile)]
            b_sg_g_out = cute.group_modes(
                b_sg_g_out,
                1,
                cute.rank(b_sg_g_out),
            )
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

            t_tr_t_acc = t_tr_t_acc_base[(None, None, None, None, None, acc_consumer_state.index)]
            acc_pipeline.consumer_wait(acc_consumer_state)
            t_tr_t_acc = cute.group_modes(t_tr_t_acc, 3, cute.rank(t_tr_t_acc))

            fragment_count = _MMA_N // _EPI_N
            for fragment_idx in cutlass.range_constexpr(fragment_count):
                cute.copy(
                    tiled_copy_t2r,
                    t_tr_t_acc[(None, None, None, fragment_idx)],
                    t_tr_r_acc,
                )
                if fragment_idx == fragment_count - 1:
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                epilogue(t_rs_r_acc, t_rs_r_out)
                if const_expr(not use_tma_output):
                    t_rs_g_c = tiled_copy_r2s.retile(t_tr_g_c[(None, None, None, fragment_idx)])
                    for element in cutlass.range(
                        cute.size(t_rs_r_acc),
                        unroll_full=True,
                    ):
                        output_coord = t_rs_g_c[element]
                        output_row = output_coord[0]
                        output_col = output_coord[1]
                        if output_row < m and output_col < n:
                            m_out_direct_mnl[output_row, output_col, 0] = t_rs_r_out[element]

                if const_expr(use_tma_output):
                    out_buffer = fragment_idx % _NUM_OUT_STAGES
                    if warp_idx == _EPILOGUE_WARPS[0]:
                        out_pipeline.producer_acquire()
                    epilogue_barrier.arrive_and_wait()
                    cute.copy(
                        tiled_copy_r2s,
                        t_rs_r_out,
                        t_rs_s_out[(None, None, None, out_buffer)],
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    epilogue_barrier.arrive_and_wait()
                    if warp_idx == _EPILOGUE_WARPS[0]:
                        cute.copy(
                            tma_atom_out,
                            b_sg_s_out[(None, out_buffer)],
                            b_sg_g_out[(None, fragment_idx)],
                        )
                        out_pipeline.producer_commit()

        if const_expr(use_tma_output) and warp_idx == _EPILOGUE_WARPS[0]:
            out_pipeline.producer_tail()
        tmem_dealloc_barrier.arrive_and_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


@cute.jit
def _host(
    m_a: cute.Tensor,
    m_b: cute.Tensor,
    m_out: cute.Tensor,
    num_ab_stages: cutlass.Constexpr,
    epilogue: cutlass.Constexpr,
    use_tma_output: cutlass.Constexpr,
    swizzle_size: cutlass.Constexpr,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    m_a_mkl = cute.make_tensor(
        m_a.iterator,
        cute.append(m_a.layout, cute.make_layout(1)),
    )
    m_b_nkl = cute.make_tensor(
        m_b.iterator,
        cute.append(m_b.layout, cute.make_layout(1)),
    )
    m_out_mnl = cute.make_tensor(
        m_out.iterator,
        cute.append(m_out.layout, cute.make_layout(1)),
    )
    io_dtype = m_a_mkl.element_type
    mma_op = tcgen05.MmaF16BF16Op(
        io_dtype,
        Float32,
        (_MMA_M, _MMA_N, 16),
        tcgen05.CtaGroup.TWO,
        tcgen05.OperandSource.SMEM,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(mma_op)
    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((*_CLUSTER_SHAPE_MN, 1)),
        (tiled_mma.thr_id.shape,),
    )

    a_smem_layout_staged = utils.sm100.make_smem_layout_a(
        tiled_mma,
        _MMA_TILER,
        io_dtype,
        num_ab_stages,
    )
    b_smem_layout_staged = utils.sm100.make_smem_layout_b(
        tiled_mma,
        _MMA_TILER,
        io_dtype,
        num_ab_stages,
    )
    out_smem_layout_staged = utils.sm100.make_smem_layout_epi(
        io_dtype,
        utils.LayoutEnum.ROW_MAJOR,
        _EPI_TILE,
        _NUM_OUT_STAGES,
    )

    a_op = utils.sm100.cluster_shape_to_tma_atom_A(
        _CLUSTER_SHAPE_MN,
        tiled_mma.thr_id,
    )
    a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        a_op,
        m_a_mkl,
        a_smem_layout,
        _MMA_TILER,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )

    b_op = utils.sm100.cluster_shape_to_tma_atom_B(
        _CLUSTER_SHAPE_MN,
        tiled_mma.thr_id,
    )
    b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        b_op,
        m_b_nkl,
        b_smem_layout,
        _MMA_TILER,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    out_smem_layout = cute.select(out_smem_layout_staged, mode=[0, 1])
    tma_atom_out, tma_tensor_out = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(),
        m_out_mnl,
        out_smem_layout,
        _EPI_TILE,
    )

    m_c_mnl = cute.make_identity_tensor((m_a_mkl.shape[0], m_b_nkl.shape[0], 1))
    g_c_mnl = cute.zipped_divide(
        m_c_mnl,
        tiler=(_CTA_M, _MMA_N),
    )
    tile_sched_params = utils.PersistentTileSchedulerParams(
        g_c_mnl[(0, (None, None, None))].shape,
        (*_CLUSTER_SHAPE_MN, 1),
        swizzle_size=swizzle_size,
        raster_along_m=swizzle_size == 1,
    )
    grid = utils.StaticPersistentTileScheduler.get_grid_shape(
        tile_sched_params,
        max_active_clusters,
    )
    _kernel(
        tiled_mma,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        tma_atom_out,
        tma_tensor_out,
        m_out_mnl,
        cluster_layout_vmnk,
        tile_sched_params,
        a_smem_layout_staged,
        b_smem_layout_staged,
        out_smem_layout_staged,
        num_ab_stages,
        io_dtype,
        epilogue,
        use_tma_output,
    ).launch(
        grid=grid,
        block=(_THREADS, 1, 1),
        cluster=(*_CLUSTER_SHAPE_MN, 1),
        stream=stream,
    )


_COMPILE_CACHE = {}
_STREAM_CACHE = {}
_GET_CURRENT_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


@functools.cache
def _max_active_clusters(device_index):
    cluster_sms = _CLUSTER_SHAPE_MN[0] * _CLUSTER_SHAPE_MN[1]
    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    return sm_count // cluster_sms


def _current_stream(device):
    if _GET_CURRENT_RAW_STREAM is not None:
        return _GET_CURRENT_RAW_STREAM(device.index)
    return torch.cuda.current_stream(device)


def _driver_stream(stream_like):
    raw_stream = stream_like if isinstance(stream_like, int) else stream_like.cuda_stream
    stream = _STREAM_CACHE.get(raw_stream)
    if stream is None:
        stream = cuda.CUstream(raw_stream)
        _STREAM_CACHE[raw_stream] = stream
    return stream


def _matrix_tensor(tensor):
    return (
        from_dlpack(tensor.detach(), assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=_MMA_K)
    )


def _dynamic_tensor(tensor, leading_dim, assumed_align):
    return from_dlpack(tensor.detach(), assumed_align=assumed_align).mark_layout_dynamic(leading_dim=leading_dim)


@contextmanager
def _compile_guard():
    """Serialize CUTLASS DSL compilation across tensor-parallel processes."""
    lock_dir = Path(tempfile.gettempdir()) / f"liger-kernel-{os.getuid()}"
    lock_dir.mkdir(mode=0o700, exist_ok=True)
    with (lock_dir / "cutedsl-sm100-gemm-compile.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _fake_matrix(dtype):
    cute_dtype = torch2cute_dtype_map[dtype]
    return cute.runtime.make_fake_tensor(
        cute_dtype,
        (cute.sym_int(), cute.sym_int(divisibility=_MMA_K)),
        stride=(cute.sym_int64(divisibility=_MMA_K), 1),
        assumed_align=16,
    )


def _fake_dynamic_tensor(tensor, leading_dim, assumed_align, stride_divisibility=1):
    cute_dtype = torch2cute_dtype_map[tensor.dtype]
    shape = tuple(cute.sym_int() for _ in tensor.shape)
    stride = tuple(
        1 if dim == leading_dim else cute.sym_int64(divisibility=stride_divisibility) for dim in range(tensor.ndim)
    )
    return cute.runtime.make_fake_tensor(
        cute_dtype,
        shape,
        stride=stride,
        assumed_align=assumed_align,
    )


@functools.cache
def _validate_epilogue_callback(epilogue):
    wrapped = getattr(epilogue, "__wrapped__", None)
    if wrapped is None or getattr(epilogue, "_dsl_cls", None) is None:
        raise TypeError("epilogue must be a module-level function decorated with @cute.jit.")
    if "<locals>" in wrapped.__qualname__ or wrapped.__closure__ or wrapped.__code__.co_freevars:
        raise ValueError("epilogue must be module-level and cannot close over local variables.")
    if len(inspect.signature(wrapped).parameters) != 2:
        raise TypeError("epilogue must accept exactly 2 parameters.")
    return wrapped.__module__, wrapped.__qualname__, wrapped


@functools.lru_cache(maxsize=64)
def _validate_epilogue_signature(
    a_shape,
    b_shape,
    out_shape,
    contiguous,
    aligned,
    devices,
    dtypes,
):
    for name, shape, is_contiguous, is_aligned, device in zip(
        ("a", "b", "out"),
        (a_shape, b_shape, out_shape),
        contiguous,
        aligned,
        devices,
        strict=True,
    ):
        if len(shape) != 2:
            raise ValueError(f"{name} must be a 2D tensor.")
        if device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor.")
        if not is_contiguous and name != "out":
            raise ValueError(f"{name} must be contiguous.")
        if not is_aligned:
            raise ValueError(f"{name} must be 16-byte aligned.")
    if devices[0] != devices[1] or devices[0] != devices[2]:
        raise ValueError("a, b, and out must be on the same CUDA device.")
    if dtypes[0] != dtypes[1] or dtypes[0] != dtypes[2]:
        raise TypeError("a, b, and out must have the same dtype.")
    if dtypes[0] not in (torch.float16, torch.bfloat16):
        raise TypeError("SM100 GEMM supports float16 and bfloat16 tensors.")
    if a_shape[0] == 0 or a_shape[1] == 0 or b_shape[0] == 0:
        raise ValueError("GEMM dimensions must be positive.")
    if a_shape[1] != b_shape[1]:
        raise ValueError(f"a and b K dimensions must match, got {a_shape[1]} and {b_shape[1]}.")
    if a_shape[1] % K_ALIGNMENT:
        raise ValueError(f"K must be divisible by {K_ALIGNMENT}, got {a_shape[1]}.")
    expected_shape = (a_shape[0], b_shape[0])
    if out_shape != expected_shape:
        raise ValueError(f"out must have shape {expected_shape}, got {out_shape}.")


def _validate_epilogue_inputs(a, b, out):
    if out.stride(1) != 1 or out.stride(0) < out.shape[1]:
        raise ValueError("out must be a dense row-major tensor or row-padded row-major view.")
    tensors = (a, b, out)
    _validate_epilogue_signature(
        *(tuple(tensor.shape) for tensor in tensors),
        tuple(tensor.is_contiguous() for tensor in tensors),
        tuple(tensor.data_ptr() % 16 == 0 for tensor in tensors),
        tuple(tensor.device for tensor in tensors),
        tuple(tensor.dtype for tensor in tensors),
    )


def _select_epilogue_config(a):
    """Select measured SM100 scheduling knobs while retaining the two-CTA kernel."""
    m_tiles = (a.shape[0] + _CTA_M - 1) // _CTA_M
    if a.shape[0] >= 4096 and m_tiles % 2 == 0:
        if a.shape[1] in (256, 2048):
            return 4, 2
        if a.shape[1] in (512, 1024):
            return 6, 2
    return _NUM_AB_STAGES, 1


def _run_epilogue_gemm(a, b, out, epilogue):
    _validate_epilogue_inputs(a, b, out)
    epilogue_key = _validate_epilogue_callback(epilogue)
    current_stream = _current_stream(a.device)
    num_ab_stages, swizzle_size = _select_epilogue_config(a)
    use_tma_output = out.stride(0) * out.element_size() % 16 == 0
    max_active_clusters = _max_active_clusters(a.device.index)

    key = (
        a.device,
        a.dtype,
        b.dtype,
        out.dtype,
        epilogue_key,
        use_tma_output,
        swizzle_size,
        num_ab_stages,
        max_active_clusters,
    )
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        if _TVM_FFI_AVAILABLE:
            # TVM-FFI is only the Python ABI used to pass torch tensors and the
            # current stream directly; cute.compile still produces a CUDA kernel.
            m_a = _fake_matrix(a.dtype)
            m_b = _fake_matrix(b.dtype)
            m_out = _fake_dynamic_tensor(
                out,
                leading_dim=1,
                assumed_align=16,
                stride_divisibility=8 if use_tma_output else 1,
            )
            stream = cute.runtime.make_fake_stream()
            options = "--enable-tvm-ffi"
        else:
            m_a = _matrix_tensor(a)
            m_b = _matrix_tensor(b)
            m_out = _dynamic_tensor(out, leading_dim=1, assumed_align=16)
            stream = _driver_stream(current_stream)
            options = None
        compile_args = (
            m_a,
            m_b,
            m_out,
            num_ab_stages,
            epilogue,
            use_tma_output,
            swizzle_size,
            max_active_clusters,
            stream,
        )
        with _compile_guard():
            if options is None:
                compiled = cute.compile(_host, *compile_args)
            else:
                compiled = cute.compile(_host, *compile_args, options=options)
        _COMPILE_CACHE[key] = compiled

    if _TVM_FFI_AVAILABLE:
        compiled(a, b, out, current_stream)
    else:
        m_a = _matrix_tensor(a)
        m_b = _matrix_tensor(b)
        m_out = _dynamic_tensor(out, leading_dim=1, assumed_align=16)
        stream = _driver_stream(current_stream)
        compiled(m_a, m_b, m_out, stream)


def run_epilogue_gemm(a, b, out, epilogue):
    """Run ``a @ b.T`` and apply ``epilogue(accumulator, output)`` per fragment."""
    with _device_guard(a.device):
        _run_epilogue_gemm(a, b, out, epilogue)
