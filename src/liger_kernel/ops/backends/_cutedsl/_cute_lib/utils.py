# This file contains code adapted from Quack
# (https://github.com/Dao-AILab/quack), Apache License 2.0.
# Copyright (c) 2025 Wentao Guo, Ted Zadouri, Tri Dao.
# Modifications by Liger-Kernel contributors.
"""Minimal subset of ``quack.utils`` required by the inlined CuTe DSL backends.

Trimmed to: ``fill_oob`` (called from the inline backward kernels),
``elem_pointer``, ``store_shared_remote``, ``set_block_rank``,
``f32x2_to_i64``, ``i64_to_f32x2`` (all transitively reached through
``quack.reduce.row_reduce`` → ``cluster_reduce``).

Everything else (atomic helpers, prefix sum, ``store_shared_remote_x4``,
the cluster-launch-control wrapper, ``sqrt`` / ``fmin`` / ``ceil``,
``make_vector``) was excluded — neither the forward nor the backward
kernels touch them, and dropping them shrinks the inlined surface.
"""

from typing import Optional
from typing import Tuple
from typing import Union

import cutlass
import cutlass.cute as cute

from cutlass import Float32
from cutlass import Int32
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import vector
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def set_block_rank(smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None) -> Int32:
    """Map the given smem pointer to the address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@dsl_user_op
def store_shared_remote(
    val: Union[float, Float32, Int32, cutlass.Int64],
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *,
    loc=None,
    ip=None,
) -> None:
    remote_smem_ptr_i32 = set_block_rank(smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value()
    if const_expr(isinstance(val, float)):
        val = Float32(val)
    assert isinstance(val, (Float32, Int32, cutlass.Int64)), "val must be Float32, Int32, or Int64"
    suffix = {Float32: "f32", Int32: "s32", cutlass.Int64: "s64"}[type(val)]
    constraint = {Float32: "f", Int32: "r", cutlass.Int64: "l"}[type(val)]
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.{suffix} [$0], $1, [$2];",
        f"r,{constraint},r",
        has_side_effects=True,
        is_align_stack=False,
    )


@cute.jit
def fill_oob(tXsX: cute.Tensor, tXpX: Optional[cute.Tensor], fill_value: cute.Numeric) -> None:
    """Fill out-of-bounds values in shared memory tensor.

    Args:
        tXsX: Shared memory tensor to fill
        tXpX: Predicate tensor indicating valid elements
        fill_value: Value to fill OOB locations with
    """
    tXrX_fill = cute.make_rmem_tensor_like(tXsX[(None, 0), None, 0])
    tXrX_fill.fill(fill_value)
    for rest_v in cutlass.range_constexpr(tXsX.shape[0][1]):
        for rest_k in cutlass.range_constexpr(tXsX.shape[2]):
            if const_expr(tXpX is not None):
                if not tXpX[rest_v, 0, rest_k]:
                    cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])
            else:
                cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])


@dsl_user_op
def f32x2_to_i64(a: Float32, b: Float32, *, loc=None, ip=None) -> cutlass.Int64:
    vec_f32x2 = vector.from_elements(T.vector(2, T.f32()), (a.ir_value(), b.ir_value()), loc=loc, ip=ip)
    vec_i64x1 = vector.bitcast(T.vector(1, T.i64()), vec_f32x2)
    res = cutlass.Int64(vector.extract(vec_i64x1, dynamic_position=[], static_position=[0], loc=loc, ip=ip))
    return res


@dsl_user_op
def i64_to_f32x2(c: cutlass.Int64, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    vec_i64x1 = vector.from_elements(T.vector(1, T.i64()), (c.ir_value(),), loc=loc, ip=ip)
    vec_f32x2 = vector.bitcast(T.vector(2, T.f32()), vec_i64x1)
    res0 = Float32(vector.extract(vec_f32x2, dynamic_position=[], static_position=[0], loc=loc, ip=ip))
    res1 = Float32(vector.extract(vec_f32x2, dynamic_position=[], static_position=[1], loc=loc, ip=ip))
    return res0, res1
