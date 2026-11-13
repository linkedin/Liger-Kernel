"""TileLang RMSNorm forward and backward for Ascend 950.

SIMD row tiles handle rows that fit in UB; wider rows use column tiles.
Backward accumulates FP32 partial weight gradients and reduces them without
atomics. Row counts remain dynamic.
"""

import tilelang
import tilelang.ascend.language as T
import torch

from torch.autograd.function import once_differentiable

from liger_kernel.ops.tilelang.ascend.utils import ASCEND_950_UB_SIZE_BYTES
from liger_kernel.ops.tilelang.ascend.utils import DTYPE_MAP
from liger_kernel.ops.tilelang.ascend.utils import load_float
from liger_kernel.ops.tilelang.ascend.utils import store_float

_CASTING_MODES = {"llama": 0, "gemma": 1, "none": -1}
_UB_RESERVE_BYTES = 16 * 1024
# Per-core RSTD cache capacity in row-tile iterations.
_RSTD_CACHE_STEPS = 128
_VECTOR_CORE_COUNTS = {}


@T.macro
def _round_vector(value, dtype):
    if dtype != "float32":
        value = T.simd.vcvt(T.simd.vcvt(value, dtype, sat=False), "float32")
    return value


@T.macro
def _load_row_float(buf, row, col, hidden, dtype):
    width = (hidden + 63) // 64 * 64
    value = load_float(buf, row * width + col, dtype)
    if hidden % 64 != 0:
        # The UB row is padded, so the full vector load is safe. Clear its tail
        # before arithmetic: an uninitialized padded lane could contain NaN.
        mask = T.simd.update_mask(T.min(64, hidden - col))
        value = T.simd.vsel(value, T.simd.vdup(0.0, "float32"), mask)
    return value


@T.macro
def _row_rstd(x, r, hidden, casting_mode, eps, one):
    """Compute one row's inverse RMS with the requested rounding."""
    dtype = x.dtype
    width = (hidden + 63) // 64 * 64
    acc = T.simd.alloc_var("float32")
    acc = T.simd.vdup(0.0, "float32")
    if hidden % 256 == 0 and casting_mode != -1:
        accs = T.simd.alloc_local((4,), "float32")
        for j in T.unroll(4, explicit=True):
            accs[j] = T.simd.vdup(0.0, "float32")
        for group in T.serial(hidden // 256):
            for j in T.unroll(4, explicit=True):
                xv = _load_row_float(x, r, group * 256 + j * 64, hidden, dtype)
                T.simd.vmula(accs[j], xv, xv)
        acc = T.simd.vadd(T.simd.vadd(accs[0], accs[1]), T.simd.vadd(accs[2], accs[3]))
    else:
        for i in T.unroll(width // 64, unroll_factor=4):
            xv = _load_row_float(x, r, i * 64, hidden, dtype)
            if casting_mode == -1:
                squares = _round_vector(T.simd.vmul(xv, xv), dtype)
                acc = T.simd.vadd(acc, squares)
            else:
                T.simd.vmula(acc, xv, xv)
    total = T.simd.vcadd(acc)
    mean = T.simd.vmuls(T.simd.vdupv(total), 1.0 / hidden)
    if casting_mode == -1:
        mean = _round_vector(mean, dtype)
    variance = T.simd.vadds(mean, eps)
    if casting_mode == -1:
        variance = _round_vector(variance, dtype)
    root = T.simd.vsqrt(variance, precision="ftz_true")
    inv = T.simd.vdiv(one, root, precision="ftz_true")
    if casting_mode == -1:
        inv = _round_vector(inv, dtype)
    return inv


@T.macro
def _normalize_row(x, w, y, r, hidden, casting_mode, affine, offset, inv):
    """Normalize one row and apply its optional affine weight."""
    dtype = x.dtype
    weight_dtype = w.dtype
    width = (hidden + 63) // 64 * 64
    for i in T.unroll(width // 64, unroll_factor=4):
        xv = _load_row_float(x, r, i * 64, hidden, dtype)
        normalized = T.simd.vmul(xv, inv)
        if casting_mode != 1:
            normalized = _round_vector(normalized, dtype)
        if affine:
            wv = _load_row_float(w, 0, i * 64, hidden, weight_dtype)
            scale = T.simd.vadds(wv, offset)
            if casting_mode != 1:
                scale = _round_vector(scale, weight_dtype)
            result = T.simd.vmul(normalized, scale)
        else:
            result = normalized
        store_float(y, r * width + i * 64, result, dtype)


@T.macro
def _backward_scale(w, col, hidden, casting_mode, offset):
    weight_dtype = w.dtype
    scale = _load_row_float(w, 0, col, hidden, weight_dtype)
    # FP32 weights already include offset, prepared once before the pipeline.
    if weight_dtype != "float32":
        scale = T.simd.vadds(scale, offset)
        if casting_mode != 1:
            scale = _round_vector(scale, weight_dtype)
    return scale


@T.macro
def _scale_gradient(grad, scale, dtype, weight_dtype, casting_mode):
    grad = T.simd.vmul(grad, scale)
    if casting_mode == -1 or (casting_mode == 0 and weight_dtype != "float32"):
        grad = _round_vector(grad, dtype)
    return grad


@T.macro
def _row_correction(x, w, dy, rstd, correction, r, rstd_index, hidden, casting_mode, affine, offset):
    """Reduce one row's weighted gradient dot product for dX."""
    dtype = x.dtype
    weight_dtype = w.dtype
    width = (hidden + 63) // 64 * 64
    count = 4 if hidden % 256 == 0 and casting_mode != -1 else 1
    zero = T.simd.vdup(0.0, "float32")
    accs = T.simd.alloc_local((count,), "float32")
    for j in T.unroll(count, explicit=True):
        accs[j] = zero
    for group in T.serial(width // (64 * count)):
        for j in T.unroll(count, explicit=True):
            i = group * count + j
            xv = _load_row_float(x, r, i * 64, hidden, dtype)
            gv = _load_row_float(dy, r, i * 64, hidden, dtype)
            if affine:
                scale = _backward_scale(w, i * 64, hidden, casting_mode, offset)
                grad = _scale_gradient(gv, scale, dtype, weight_dtype, casting_mode)
            else:
                grad = gv
            T.simd.vmula(accs[j], grad, xv)
    if count == 4:
        total = T.simd.vadd(T.simd.vadd(accs[0], accs[1]), T.simd.vadd(accs[2], accs[3]))
    else:
        total = accs[0]
    total = T.simd.vdupv(T.simd.vcadd(total))
    inv = T.simd.vld(rstd[rstd_index, r], dist="BRC_B32")
    if casting_mode == -1:
        scaled = total
    else:
        scaled = T.simd.vmuls(T.simd.vmul(T.simd.vmul(total, inv), inv), -1.0 / hidden)
    T.simd.vsts(correction[r], scaled, dist="ONEPT_B32")


@T.macro
def _backward_rows(
    x, w, dy, dx, dw, rstd, correction, valid, rstd_index, hidden, casting_mode, affine, needs_dw, offset
):
    """Compute dX and accumulate dW, reusing each weight across rows."""
    dtype = x.dtype
    weight_dtype = w.dtype
    width = (hidden + 63) // 64 * 64
    for i in T.serial(width // 64):
        if affine:
            scale = _backward_scale(w, i * 64, hidden, casting_mode, offset)
        if needs_dw:
            acc_dw = T.simd.alloc_var("float32")
            acc_dw = T.simd.vld(dw[i * 64])
        for r in T.serial(valid):
            xv = _load_row_float(x, r, i * 64, hidden, dtype)
            gv = _load_row_float(dy, r, i * 64, hidden, dtype)
            inv = T.simd.vld(rstd[rstd_index, r], dist="BRC_B32")
            corr = T.simd.vld(correction[r], dist="BRC_B32")
            if affine:
                grad = _scale_gradient(gv, scale, dtype, weight_dtype, casting_mode)
            else:
                grad = gv
            result = T.simd.alloc_var("float32")
            if casting_mode == -1:
                correction_value = T.simd.vmul(T.simd.vmul(xv, inv), inv)
                correction_value = T.simd.vmuls(T.simd.vmul(correction_value, corr), 1.0 / hidden)
                result = T.simd.vmul(inv, T.simd.vsub(grad, correction_value))
            else:
                result = grad
                T.simd.vmula(result, xv, corr)
                result = T.simd.vmul(inv, result)
            store_float(dx, r * width + i * 64, result, dtype)
            if needs_dw:
                normalized = T.simd.vmul(xv, inv)
                if casting_mode != 1:
                    normalized = _round_vector(normalized, dtype)
                if weight_dtype == "float32":
                    T.simd.vmula(acc_dw, gv, normalized)
                else:
                    contribution = T.simd.vmul(gv, normalized)
                    if casting_mode != 1 and weight_dtype == dtype:
                        contribution = _round_vector(contribution, dtype)
                    acc_dw = T.simd.vadd(acc_dw, contribution)
        if needs_dw:
            T.simd.vsts(dw[i * 64], acc_dw)


@tilelang.jit(target="ascend", out_idx=[])
def rms_norm_forward_kernel(dtype, weight_dtype, hidden, cores, casting_mode, affine, row_tile, stages, bulk_rstd):
    rows = T.dynamic("rows", "int64")
    width = (hidden + 63) // 64 * 64

    @T.prim_func
    def main(
        X: T.Tensor((rows, hidden), dtype),
        W: T.Tensor((hidden,), weight_dtype),
        Y: T.Tensor((rows, hidden), dtype),
        RSTD: T.Tensor((rows,), "float32"),
        eps: T.float32,
        offset: T.float32,
    ):
        with T.Kernel(cores) as core:
            x = T.alloc_shared((row_tile * width,), dtype)
            w = T.alloc_shared((width,), weight_dtype)
            y = T.alloc_shared((row_tile * width,), dtype)
            rstd = T.alloc_shared((_RSTD_CACHE_STEPS if bulk_rstd else 1, 8), "float32")
            T.annotate_buffer_versions({x: stages, w: 1, y: stages, rstd: 1 if bulk_rstd else stages})
            if affine:
                T.copy(W, w[:hidden])
            x_rows = T.Tensor((row_tile, width), dtype, x.data)
            y_rows = T.Tensor((row_tile, width), dtype, y.data)
            for step in T.Pipelined(T.ceildiv(T.ceildiv(rows, row_tile) - core, cores), num_stages=stages):
                start = (step * cores + core) * row_tile
                valid = T.cast(T.min(row_tile, rows - start), "int32")
                T.copy(X[start : start + valid, :], x_rows[:valid, :hidden])
                with T.SimdVF():
                    one = T.simd.vdup(1.0, "float32")
                    for r in T.serial(valid):
                        inv = _row_rstd(x, r, hidden, casting_mode, eps, one)
                        T.simd.vsts(rstd[T.cast(step, "int32") if bulk_rstd else 0, r], inv, dist="ONEPT_B32")
                        _normalize_row(x, w, y, r, hidden, casting_mode, affine, offset, inv)
                T.copy(y_rows[:valid, :hidden], Y[start : start + valid, :], l2_cache_ctrl=0)
                if not bulk_rstd:
                    T.copy(rstd[0, :valid], RSTD[start : start + valid], l2_cache_ctrl=0)

            if bulk_rstd:
                rstd_view = T.Tensor((T.ceildiv(rows, cores * row_tile), cores * row_tile), "float32", RSTD.data)
                T.copy(
                    rstd[: T.ceildiv(rows, cores * row_tile), :row_tile],
                    rstd_view[:, core * row_tile : (core + 1) * row_tile],
                    l2_cache_ctrl=0,
                )

    return main


@tilelang.jit(target="ascend", out_idx=[])
def rms_norm_backward_kernel(
    dtype, weight_dtype, hidden, cores, casting_mode, affine, needs_dw, row_tile, stages, bulk_rstd
):
    rows = T.dynamic("rows", "int64")
    width = (hidden + 63) // 64 * 64
    reduce_cores = 1 << (cores - 1).bit_length()

    @T.prim_func
    def main(
        X: T.Tensor((rows, hidden), dtype),
        W: T.Tensor((hidden,), weight_dtype),
        DY: T.Tensor((rows, hidden), dtype),
        RSTD: T.Tensor((rows,), "float32"),
        DX: T.Tensor((rows, hidden), dtype),
        DW: T.Tensor((cores if needs_dw else 0, hidden), "float32"),
        DW_FINAL: T.Tensor((hidden,), weight_dtype),
        offset: T.float32,
    ):
        with T.Kernel(cores) as core:
            x = T.alloc_shared((row_tile * width,), dtype)
            w = T.alloc_shared((width,), weight_dtype)
            dy = T.alloc_shared((row_tile * width,), dtype)
            dx = T.alloc_shared((row_tile * width,), dtype)
            dw = T.alloc_shared((width,), "float32")
            rstd = T.alloc_shared((_RSTD_CACHE_STEPS if bulk_rstd else 1, 8), "float32")
            correction = T.alloc_shared((row_tile,), "float32")
            T.annotate_buffer_versions(
                {x: stages, w: 1, dy: stages, dx: stages, dw: 1, rstd: 1 if bulk_rstd else stages, correction: 1}
            )
            if affine:
                T.copy(W, w[:hidden])
            if needs_dw or (affine and weight_dtype == "float32"):
                with T.SimdVF():
                    zero = T.simd.vdup(0.0, "float32")
                    for i in T.serial(width // 64):
                        if needs_dw:
                            T.simd.vsts(dw[i * 64], zero)
                        if affine and weight_dtype == "float32":
                            scale = T.simd.vadds(T.simd.vld(w[i * 64]), offset)
                            T.simd.vsts(w[i * 64], scale)
            x_rows = T.Tensor((row_tile, width), dtype, x.data)
            dy_rows = T.Tensor((row_tile, width), dtype, dy.data)
            dx_rows = T.Tensor((row_tile, width), dtype, dx.data)
            if bulk_rstd:
                rstd_view = T.Tensor((T.ceildiv(rows, cores * row_tile), cores * row_tile), "float32", RSTD.data)
                T.copy(
                    rstd_view[:, core * row_tile : (core + 1) * row_tile],
                    rstd[: T.ceildiv(rows, cores * row_tile), :row_tile],
                )
            for step in T.Pipelined(T.ceildiv(T.ceildiv(rows, row_tile) - core, cores), num_stages=stages):
                start = (step * cores + core) * row_tile
                valid = T.cast(T.min(row_tile, rows - start), "int32")
                T.copy(X[start : start + valid, :], x_rows[:valid, :hidden])
                T.copy(DY[start : start + valid, :], dy_rows[:valid, :hidden])
                if not bulk_rstd:
                    T.copy(RSTD[start : start + valid], rstd[0, :valid])
                with T.SimdVF():
                    for r in T.serial(valid):
                        _row_correction(
                            x,
                            w,
                            dy,
                            rstd,
                            correction,
                            r,
                            T.cast(step, "int32") if bulk_rstd else 0,
                            hidden,
                            casting_mode,
                            affine,
                            offset,
                        )
                    T.simd.mem_bar("VST_VLD")
                    _backward_rows(
                        x,
                        w,
                        dy,
                        dx,
                        dw,
                        rstd,
                        correction,
                        valid,
                        T.cast(step, "int32") if bulk_rstd else 0,
                        hidden,
                        casting_mode,
                        affine,
                        needs_dw,
                        offset,
                    )
                T.copy(dx_rows[:valid, :hidden], DX[start : start + valid, :], l2_cache_ctrl=0)
            if needs_dw:
                T.copy(dw[:hidden], DW[core, :], l2_cache_ctrl=0)

        if needs_dw:
            with T.Kernel(T.ceildiv(hidden, 64)) as tile:
                partial = T.alloc_shared((reduce_cores, 64), "float32")
                result = T.alloc_shared((64,), weight_dtype)
                cols = T.min(64, hidden - tile * 64)
                T.copy(DW[:, tile * 64 : tile * 64 + cols], partial[:cores, :cols])
                with T.SimtVF(threads=min(128, reduce_cores * 64)):
                    values = T.alloc_fragment((reduce_cores, 64), "float32")
                    total = T.alloc_fragment((64,), "float32")
                    for r, i in T.Parallel(reduce_cores, 64):
                        values[r, i] = T.if_then_else(r < cores and i < cols, partial[r, i], T.float32(0))
                    T.reduce_sum(values, total, dim=0)
                    for i in T.Parallel(64):
                        result[i] = total[i]
                T.copy(result[:cols], DW_FINAL[tile * 64 : tile * 64 + cols], l2_cache_ctrl=0)

    return main


@T.macro
def _load_column_float(buf, col, valid, hidden, dtype):
    value = load_float(buf, col, dtype)
    if hidden % 64 != 0:
        mask = T.simd.update_mask(T.min(64, valid - col))
        value = T.simd.vsel(value, T.simd.vdup(0.0, "float32"), mask)
    return value


@tilelang.jit(target="ascend", out_idx=[])
def rms_norm_forward_tiled(dtype, weight_dtype, hidden, block, cores, casting_mode, affine):
    rows = T.dynamic("rows", "int64")

    @T.prim_func
    def main(
        X: T.Tensor((rows, hidden), dtype),
        W: T.Tensor((hidden,), weight_dtype),
        Y: T.Tensor((rows, hidden), dtype),
        RSTD: T.Tensor((rows,), "float32"),
        eps: T.float32,
        offset: T.float32,
    ):
        with T.Kernel(cores) as core:
            x = T.alloc_shared((block,), dtype)
            w = T.alloc_shared((block,), weight_dtype)
            y = T.alloc_shared((block,), dtype)
            stat = T.alloc_shared((64,), "float32")
            T.annotate_buffer_versions({x: 2, w: 2 if affine else 1, y: 2, stat: 1})
            for row in T.serial(T.cast(core, "int64"), rows, step=cores):
                for tile in T.Pipelined(T.ceildiv(hidden, block), num_stages=2):
                    start = tile * block
                    valid = T.min(block, hidden - start)
                    T.copy(X[row, start : start + valid], x[:valid])
                    with T.SimdVF():
                        acc = T.simd.alloc_var("float32")
                        if tile == 0:
                            acc = T.simd.vdup(0.0, "float32")
                        else:
                            acc = T.simd.vld(stat[0])
                        for i in T.serial(T.ceildiv(valid, 64)):
                            value = _load_column_float(x, i * 64, valid, hidden, dtype)
                            if casting_mode == -1:
                                acc = T.simd.vadd(acc, _round_vector(T.simd.vmul(value, value), dtype))
                            else:
                                T.simd.vmula(acc, value, value)
                        T.simd.vsts(stat[0], acc)
                with T.SimdVF():
                    total = T.simd.vdupv(T.simd.vcadd(T.simd.vld(stat[0])))
                    mean = T.simd.vmuls(total, 1.0 / hidden)
                    if casting_mode == -1:
                        mean = _round_vector(mean, dtype)
                    variance = T.simd.vadds(mean, eps)
                    if casting_mode == -1:
                        variance = _round_vector(variance, dtype)
                    inv = T.simd.vdiv(
                        T.simd.vdup(1.0, "float32"), T.simd.vsqrt(variance, precision="ftz_true"), precision="ftz_true"
                    )
                    if casting_mode == -1:
                        inv = _round_vector(inv, dtype)
                    T.simd.vsts(stat[0], inv, dist="ONEPT_B32")
                T.copy(stat[:1], RSTD[row : row + 1], l2_cache_ctrl=0)
                for tile in T.Pipelined(T.ceildiv(hidden, block), num_stages=2):
                    start = tile * block
                    valid = T.min(block, hidden - start)
                    T.copy(X[row, start : start + valid], x[:valid])
                    if affine:
                        T.copy(W[start : start + valid], w[:valid])
                    with T.SimdVF():
                        inv = T.simd.vld(stat[0], dist="BRC_B32")
                        for i in T.serial(T.ceildiv(valid, 64)):
                            value = _load_column_float(x, i * 64, valid, hidden, dtype)
                            normalized = T.simd.vmul(value, inv)
                            if casting_mode != 1:
                                normalized = _round_vector(normalized, dtype)
                            if affine:
                                scale = T.simd.vadds(_load_column_float(w, i * 64, valid, hidden, weight_dtype), offset)
                                if casting_mode != 1:
                                    scale = _round_vector(scale, weight_dtype)
                                result = T.simd.vmul(normalized, scale)
                            else:
                                result = normalized
                            store_float(y, i * 64, result, dtype)
                    T.copy(y[:valid], Y[row, start : start + valid], l2_cache_ctrl=0)

    return main


@tilelang.jit(target="ascend", out_idx=[])
def rms_norm_backward_tiled(dtype, weight_dtype, hidden, block, cores, casting_mode, affine, needs_dw):
    rows = T.dynamic("rows", "int64")
    width = (hidden + 63) // 64 * 64
    input_bytes = 4 if dtype == "float32" else 2
    weight_bytes = (4 if weight_dtype == "float32" else 2) if affine else 0
    cache_dw = (
        needs_dw
        and width * 4 + 2 * (3 * input_bytes + weight_bytes) * block <= ASCEND_950_UB_SIZE_BYTES - _UB_RESERVE_BYTES
    )
    # Fit the partial gradients and output in UB, using wider tiles for fewer rows.
    reduce_limit = (ASCEND_950_UB_SIZE_BYTES - _UB_RESERVE_BYTES) // (cores * 4 + weight_bytes)
    reduce_block = min(2048, 1 << (reduce_limit.bit_length() - 1))

    @T.macro
    def compute_tile(x, w, dy, dx, dw, rstd, stat, start, valid, offset, initialize_dw):
        """Specialize initialization outside the SIMD vector loop."""
        with T.SimdVF():
            inv = T.simd.vld(rstd[0], dist="BRC_B32")
            correction = T.simd.vld(stat[0], dist="BRC_B32")
            for i in T.serial(T.ceildiv(valid, 64)):
                value = _load_column_float(x, i * 64, valid, hidden, dtype)
                grad_y = _load_column_float(dy, i * 64, valid, hidden, dtype)
                grad = grad_y
                if affine:
                    scale = T.simd.vadds(_load_column_float(w, i * 64, valid, hidden, weight_dtype), offset)
                    if casting_mode != 1:
                        scale = _round_vector(scale, weight_dtype)
                    grad = T.simd.vmul(grad, scale)
                    if casting_mode != 1 and weight_dtype != "float32":
                        grad = _round_vector(grad, dtype)
                result = T.simd.vmul(inv, T.simd.vsub(grad, T.simd.vmul(value, correction)))
                store_float(dx, i * 64, result, dtype)
                if needs_dw:
                    normalized = T.simd.vmul(value, inv)
                    if casting_mode != 1:
                        normalized = _round_vector(normalized, dtype)
                    contribution = T.simd.vmul(grad_y, normalized)
                    if casting_mode != 1 and weight_dtype == dtype:
                        contribution = _round_vector(contribution, dtype)
                    acc_dw = T.simd.alloc_var("float32")
                    if cache_dw:
                        acc_dw = T.simd.vadd(T.simd.vld(dw[start + i * 64]), contribution)
                    elif initialize_dw:
                        acc_dw = contribution
                    else:
                        acc_dw = T.simd.vadd(T.simd.vld(dw[i * 64]), contribution)
                    T.simd.vsts(dw[(start if cache_dw else 0) + i * 64], acc_dw)

    @T.prim_func
    def main(
        X: T.Tensor((rows, hidden), dtype),
        W: T.Tensor((hidden,), weight_dtype),
        DY: T.Tensor((rows, hidden), dtype),
        RSTD: T.Tensor((rows,), "float32"),
        DX: T.Tensor((rows, hidden), dtype),
        DW: T.Tensor((cores if needs_dw else 0, hidden), "float32"),
        DW_FINAL: T.Tensor((hidden,), weight_dtype),
        offset: T.float32,
    ):
        with T.Kernel(cores) as core:
            x = T.alloc_shared((block,), dtype)
            w = T.alloc_shared((block,), weight_dtype)
            dy = T.alloc_shared((block,), dtype)
            dx = T.alloc_shared((block,), dtype)
            dw = T.alloc_shared((width if cache_dw else block,), "float32")
            rstd = T.alloc_shared((1,), "float32")
            stat = T.alloc_shared((64,), "float32")
            T.annotate_buffer_versions({x: 2, w: 2 if affine else 1, dy: 2, dx: 2, dw: 1, rstd: 1, stat: 1})
            if cache_dw:
                with T.SimdVF():
                    zero = T.simd.vdup(0.0, "float32")
                    for i in T.serial(width // 64):
                        T.simd.vsts(dw[i * 64], zero)
            for row in T.serial(T.cast(core, "int64"), rows, step=cores):
                T.copy(RSTD[row : row + 1], rstd)
                for tile in T.Pipelined(T.ceildiv(hidden, block), num_stages=2):
                    start = tile * block
                    valid = T.min(block, hidden - start)
                    T.copy(X[row, start : start + valid], x[:valid])
                    T.copy(DY[row, start : start + valid], dy[:valid])
                    if affine:
                        T.copy(W[start : start + valid], w[:valid])
                    with T.SimdVF():
                        acc = T.simd.alloc_var("float32")
                        if tile == 0:
                            acc = T.simd.vdup(0.0, "float32")
                        else:
                            acc = T.simd.vld(stat[0])
                        for i in T.serial(T.ceildiv(valid, 64)):
                            value = _load_column_float(x, i * 64, valid, hidden, dtype)
                            grad = _load_column_float(dy, i * 64, valid, hidden, dtype)
                            if affine:
                                scale = T.simd.vadds(_load_column_float(w, i * 64, valid, hidden, weight_dtype), offset)
                                if casting_mode != 1:
                                    scale = _round_vector(scale, weight_dtype)
                                grad = T.simd.vmul(grad, scale)
                                if casting_mode != 1 and weight_dtype != "float32":
                                    grad = _round_vector(grad, dtype)
                            T.simd.vmula(acc, grad, value)
                        T.simd.vsts(stat[0], acc)
                with T.SimdVF():
                    total = T.simd.vdupv(T.simd.vcadd(T.simd.vld(stat[0])))
                    inv = T.simd.vld(rstd[0], dist="BRC_B32")
                    correction = T.simd.vmuls(T.simd.vmul(T.simd.vmul(total, inv), inv), 1.0 / hidden)
                    T.simd.vsts(stat[0], correction, dist="ONEPT_B32")
                for tile in T.Pipelined(T.ceildiv(hidden, block), num_stages=2):
                    start = tile * block
                    valid = T.min(block, hidden - start)
                    T.copy(X[row, start : start + valid], x[:valid])
                    T.copy(DY[row, start : start + valid], dy[:valid])
                    if affine:
                        T.copy(W[start : start + valid], w[:valid])
                    if needs_dw and not cache_dw:
                        if row != core:
                            T.copy(DW[core, start : start + valid], dw[:valid])
                        if row == core:
                            compute_tile(x, w, dy, dx, dw, rstd, stat, start, valid, offset, True)
                        else:
                            compute_tile(x, w, dy, dx, dw, rstd, stat, start, valid, offset, False)
                    else:
                        compute_tile(x, w, dy, dx, dw, rstd, stat, start, valid, offset, False)
                    T.copy(dx[:valid], DX[row, start : start + valid], l2_cache_ctrl=0)
                    if needs_dw and not cache_dw:
                        T.copy(dw[:valid], DW[core, start : start + valid], l2_cache_ctrl=0)
            if cache_dw:
                T.copy(dw[:hidden], DW[core, :], l2_cache_ctrl=0)

        if needs_dw:
            with T.Kernel(T.ceildiv(hidden, reduce_block)) as weight_tile:
                partial = T.alloc_shared((cores, reduce_block), "float32")
                result = T.alloc_shared((reduce_block,), weight_dtype)
                cols = T.min(reduce_block, hidden - weight_tile * reduce_block)
                T.copy(DW[:, weight_tile * reduce_block : weight_tile * reduce_block + cols], partial[:, :cols])
                with T.SimdVF():
                    for i in T.serial(T.ceildiv(cols, 64)):
                        acc = T.simd.alloc_var("float32")
                        acc = T.simd.vdup(0.0, "float32")
                        for r in T.serial(cores):
                            acc = T.simd.vadd(acc, T.simd.vld(partial[r, i * 64]))
                        store_float(result, i * 64, acc, weight_dtype)
                T.copy(
                    result[:cols],
                    DW_FINAL[weight_tile * reduce_block : weight_tile * reduce_block + cols],
                    l2_cache_ctrl=0,
                )

    return main


def _casting_mode(mode):
    if isinstance(mode, str):
        if mode not in _CASTING_MODES:
            raise ValueError(f"Invalid casting mode: {mode!r}")
        mode = _CASTING_MODES[mode]
    if mode not in _CASTING_MODES.values():
        raise ValueError(f"Invalid casting mode: {mode!r}")
    return mode


def _validate_input(x, weight):
    if x.device.type != "npu" or x.dtype not in DTYPE_MAP:
        raise ValueError("Ascend TileLang RMSNorm requires an NPU FP16, BF16 or FP32 input")
    if x.ndim == 0 or x.shape[-1] <= 0:
        raise ValueError("Ascend TileLang RMSNorm requires a positive hidden dimension")
    if weight is not None:
        if weight.shape != (x.shape[-1],):
            raise ValueError("RMSNorm weight must be a vector matching the input hidden dimension")
        if weight.device != x.device or weight.dtype not in DTYPE_MAP:
            raise ValueError("RMSNorm weight must be FP16, BF16 or FP32 on the same NPU as the input")


def _vector_core_count(device):
    count = _VECTOR_CORE_COUNTS.get(device.index)
    if count is None:
        count = max(1, torch.npu.get_device_properties(device).vector_core_num)
        _VECTOR_CORE_COUNTS[device.index] = count
    return count


def _tiled_launch_config(x, weight, backward=False, needs_dw=True):
    hidden = x.shape[-1]
    budget = ASCEND_950_UB_SIZE_BYTES - _UB_RESERVE_BYTES
    weight_bytes = weight.dtype.itemsize if weight is not None else 0
    bytes_per_element = 2 * ((3 if backward else 2) * x.dtype.itemsize + weight_bytes)
    dw_bytes = ((hidden + 63) // 64 * 64) * 4 if backward and weight is not None and needs_dw else 0
    # Cache the full dW only when at least a 1K column tile still fits.
    if dw_bytes and budget - dw_bytes >= bytes_per_element * 1024:
        limit = (budget - dw_bytes) // bytes_per_element
    else:
        limit = budget // (bytes_per_element + (4 if dw_bytes else 0))
    block = min(max(128, 1 << (hidden - 1).bit_length()), 1 << (limit.bit_length() - 1))
    cores = min(x.shape[0], _vector_core_count(x.device))
    return block, cores


def _launch_config(x, weight, backward=False, needs_dw=True):
    # Include the last partial vector in both the row stride and UB budget.
    width = (x.shape[-1] + 63) // 64 * 64
    # Reserve space for the VF stack, the 4 KiB RSTD cache, and compiler scratch.
    budget = ASCEND_950_UB_SIZE_BYTES - _UB_RESERVE_BYTES
    fixed = width * ((weight.dtype.itemsize if weight is not None else 0) + (4 if backward and needs_dw else 0))
    stages = 2 if backward else 3
    copies = (3 if backward else 2) * stages
    limit = (budget - fixed) // (copies * width * x.dtype.itemsize)
    while limit < 1 and stages > 1:
        stages -= 1
        copies = (3 if backward else 2) * stages
        limit = (budget - fixed) // (copies * width * x.dtype.itemsize)
    if limit < 1:
        return None
    row_tile = min(4 if backward else 8, 1 << (limit.bit_length() - 1))
    cores = min((x.shape[0] + row_tile - 1) // row_tile, _vector_core_count(x.device))
    return cores, row_tile, stages


def rms_norm_forward(x, weight, eps, offset=0.0, casting_mode="llama", row_mode=None):
    """Return output, contiguous flattened input, inverse RMS and casting mode.

    ``row_mode`` is accepted for compatibility. The UB budget determines whether
    a row is processed in full or in column tiles. Empty batches need no launch.
    """
    _validate_input(x, weight)
    casting_mode = _casting_mode(casting_mode)
    shape = x.shape
    x = x.contiguous()
    if x.ndim != 2:
        x = x.view(-1, shape[-1])
    weight = weight.contiguous() if weight is not None else None
    y = torch.empty_like(x)
    rstd = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    if x.numel():
        with torch.npu.device(x.device):
            w_arg = weight if weight is not None else x[0]
            config = _launch_config(x, weight)
            if config is not None:
                cores, row_tile, stages = config
                kernel = rms_norm_forward_kernel(
                    DTYPE_MAP[x.dtype],
                    DTYPE_MAP[w_arg.dtype],
                    x.shape[1],
                    cores,
                    casting_mode,
                    weight is not None,
                    row_tile,
                    stages,
                    x.shape[0] % (cores * row_tile) == 0 and x.shape[0] <= _RSTD_CACHE_STEPS * cores * row_tile,
                )
            else:
                block, cores = _tiled_launch_config(x, weight)
                kernel = rms_norm_forward_tiled(
                    DTYPE_MAP[x.dtype],
                    DTYPE_MAP[w_arg.dtype],
                    x.shape[1],
                    block,
                    cores,
                    casting_mode,
                    weight is not None,
                )
            kernel(x, w_arg, y, rstd, float(eps), float(offset))
    return (y if len(shape) == 2 else y.view(shape)), x, rstd, casting_mode


def rms_norm_backward(dy, x, weight, rstd, offset, casting_mode, in_place=False, needs_dw=True):
    casting_mode = _casting_mode(casting_mode)
    shape = dy.shape
    dy = dy.contiguous()
    if dy.ndim != 2:
        dy = dy.view(-1, x.shape[-1])
    weight = weight.contiguous() if weight is not None else None
    dx = dy if in_place else torch.empty_like(dy)
    needs_dw = weight is not None and needs_dw
    dw = torch.empty_like(weight) if needs_dw else None
    if not x.numel():
        if dw is not None:
            dw.zero_()
        return (dx if not in_place and len(shape) == 2 else dx.view(shape)), dw
    with torch.npu.device(x.device):
        w_arg = weight if weight is not None else x[0]
        config = _launch_config(x, weight, backward=True, needs_dw=needs_dw)
        if config is not None:
            cores, row_tile, stages = config
            kernel = rms_norm_backward_kernel(
                DTYPE_MAP[x.dtype],
                DTYPE_MAP[w_arg.dtype],
                x.shape[1],
                cores,
                casting_mode,
                weight is not None,
                needs_dw,
                row_tile,
                stages,
                x.shape[0] % (cores * row_tile) == 0 and x.shape[0] <= _RSTD_CACHE_STEPS * cores * row_tile,
            )
        else:
            block, cores = _tiled_launch_config(x, weight, backward=True, needs_dw=needs_dw)
            kernel = rms_norm_backward_tiled(
                DTYPE_MAP[x.dtype],
                DTYPE_MAP[w_arg.dtype],
                x.shape[1],
                block,
                cores,
                casting_mode,
                weight is not None,
                needs_dw,
            )
        partial = torch.empty((cores if needs_dw else 0, x.shape[1]), device=x.device, dtype=torch.float32)
        kernel(x, w_arg, dy, rstd, dx, partial, dw if needs_dw else w_arg, float(offset))
    # A distinct view lets autograd reuse an in-place gradient without copying it.
    return (dx if not in_place and len(shape) == 2 else dx.view(shape)), dw


class LigerRMSNormFunction(torch.autograd.Function):
    """First-order RMSNorm autograd with optional upstream-gradient reuse."""

    @staticmethod
    def forward(ctx, x, weight, eps, offset=0.0, casting_mode="llama", in_place=True, row_mode=None):
        weight = weight.contiguous() if weight is not None else None
        y, x, rstd, casting_mode = rms_norm_forward(x, weight, eps, offset, casting_mode, row_mode)
        ctx.save_for_backward(x, weight, rstd)
        ctx.offset, ctx.casting_mode, ctx.in_place = float(offset), casting_mode, in_place
        return y

    @staticmethod
    def backward(ctx, dy):
        # Ordinary backward already runs with grad disabled. Keep the decorated
        # path for create_graph so double-backward errors retain Torch semantics.
        if torch.is_grad_enabled():
            return LigerRMSNormFunction._backward_once(ctx, dy)
        x, weight, rstd = ctx.saved_tensors
        dx, dw = rms_norm_backward(
            dy, x, weight, rstd, ctx.offset, ctx.casting_mode, ctx.in_place, ctx.needs_input_grad[1]
        )
        return (dx, dw, None, None, None, None, None)[: len(ctx.needs_input_grad)]

    @staticmethod
    @once_differentiable
    def _backward_once(ctx, dy):
        return LigerRMSNormFunction.backward(ctx, dy)
