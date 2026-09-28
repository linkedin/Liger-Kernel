# This file contains code adapted from Quack
# (https://github.com/Dao-AILab/quack), Apache License 2.0.
# Copyright (c) 2025 Wentao Guo, Ted Zadouri, Tri Dao.
# Modifications by Liger-Kernel contributors.
"""``ReductionBase`` — base class for the inlined backward kernels.

Inlined verbatim from ``quack.reduction_base`` with the only edit being
the import (``quack.copy_utils`` -> local sibling). The base class
exposes the plumbing for tiled-copy construction, reduction-buffer +
mbar allocation, and cluster initialisation that both the LayerNorm
and RMSNorm backward kernels build on.
"""

from typing import Optional
from typing import Tuple
from typing import Type

import cutlass
import cutlass.cute as cute

from cutlass import Float32
from cutlass import Int32
from cutlass import Int64
from cutlass import const_expr

from . import copy_utils


class ReductionBase:
    def __init__(self, dtype: Type[cutlass.Numeric], N: int, stage: int, reduction_dtype=Float32):
        self.dtype = dtype
        self.N = N
        self.stage = stage
        self.reduction_dtype = reduction_dtype

    def _threads_per_row(self):
        raise NotImplementedError()

    def _num_threads(self):
        return 128 if self.N <= 16384 else 256

    def _set_cluster_n(self):
        self.cluster_n = 1

    def _get_tiled_copy(self, vecsize: int = 1):
        assert self.N % vecsize == 0, f"Input N {self.N} is not divisible by vector size {vecsize}"
        threads_per_row = self._threads_per_row()
        num_threads = self._num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0
        num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n)
        tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = copy_utils.tiled_copy_2d(self.dtype, threads_per_row, num_threads, vecsize)
        return tiled_copy, tiler_mn, threads_per_row

    def _get_reduction_buffer_layout(self, tv_layout: cute.Layout, cluster_n: int):
        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        warps_per_row = (
            num_warps if cute.rank(tv_layout.shape[0]) == 1 else max(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
        )
        return cute.make_ordered_layout(
            (num_warps // warps_per_row, (warps_per_row, cluster_n), self.stage),
            order=(1, 0, 2),
        )

    def _allocate_reduction_buffer_and_mbar(
        self, smem: cutlass.utils.SmemAllocator, tv_layout: cute.Layout, is_persistent: bool = False
    ) -> Tuple[cute.Tensor, Optional[cute.Pointer]]:
        reduction_buffer = smem.allocate_tensor(
            self.reduction_dtype,
            self._get_reduction_buffer_layout(tv_layout, self.cluster_n),
            byte_alignment=8,
        )
        if const_expr(self.cluster_n > 1):
            mbar_ptr = smem.allocate_array(Int64, num_elems=self.stage if not is_persistent else self.stage * 2)
        else:
            mbar_ptr = None
        return reduction_buffer, mbar_ptr

    @cute.jit
    def _initialize_cluster(
        self,
        tidx: Int32,
        mbar_ptr: cute.Pointer,
        num_warps: int,
        is_persistent: bool = False,
    ):
        if const_expr(self.cluster_n > 1):
            if tidx < self.stage:  # Initialize full barrier
                cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
                if const_expr(is_persistent):  # Initialize empty barrier
                    cute.arch.mbarrier_init(mbar_ptr + self.stage + tidx, num_warps * self.cluster_n)
            cute.arch.mbarrier_init_fence()
            # Cluster arrive after barrier init
            cute.arch.cluster_arrive_relaxed()
