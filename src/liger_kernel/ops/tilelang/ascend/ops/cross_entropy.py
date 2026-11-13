"""Ascend950 TileLang/PTO cross-entropy kernels and first-order autograd."""

import math

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

from tilelang.ascend.language import simd as S
from torch.autograd.function import once_differentiable

from liger_kernel.ops.tilelang.ascend.ops._cross_entropy_metadata import prepare_targets


@tilelang.jit
def simd_cross_entropy_kernel(
    V,
    dtype,
    cores,
    need_grad,
    stages=2,
    storage="auto",
    save_stats=False,
    return_z_loss=True,
    has_weight=False,
    smoothing=0.0,
    softcap=0.0,
    z_scale=0.0,
):
    """Compute CE using one row per core and compile-time optional features.

    Small rows stay in UB; larger rows stream through tiles while accumulating
    the row maximum and shifted exponential sum. Optionally save these
    statistics for backward instead of materializing the input gradient.
    """
    narrow = dtype == "bfloat16"
    # Class weights need an additional FP32 tile in each pipeline stage.
    resident_limit = 8192 if has_weight else 32000
    resident = V <= resident_limit
    tile = tilelang.cdiv(V, 64) * 64 if resident else (8192 if has_weight else (31680 if narrow else 15808))
    full_blocks, tail = divmod(V, tile)
    manual = storage == "manual" and not resident and stages == 2
    active_stages = 1 if resident else stages
    store_dist = "PK_B32" if narrow else "NORM_B32"

    def load(buf, offset, slot):
        if manual:
            raw = S.vld(buf[slot, offset], dist="UNPK_B16" if narrow else "NORM")
        else:
            raw = S.vld(buf[offset], dist="UNPK_B16" if narrow else "NORM")
        return S.vcvt(raw, T.float32, part=0) if narrow else raw

    def store(buf, offset, value, slot, mask):
        converted = S.vcvt(value, T.bfloat16) if narrow else value
        if manual:
            return S.vsts(buf[slot, offset], converted, mask, store_dist)
        else:
            return S.vsts(buf[offset], converted, mask, store_dist)

    def weight_value(w, offset, slot):
        return S.vld(w[slot, offset] if manual else w[offset], dist="NORM")

    def tanh(value, mask, return_derivative=False):
        # This form avoids overflow for large positive or negative logits.
        decay = S.vexp(S.vmuls(S.vabs(value, mask), -2.0 / softcap, mask), mask)
        one = S.vdup(1.0, T.float32, mask)
        divisor = S.vadd(one, decay, mask)
        magnitude = S.vdiv(S.vsub(one, decay, mask), divisor, mask, precision="exact")
        signed = S.vsel(S.vmuls(magnitude, -1.0, mask), magnitude, S.vcmps(value, 0.0, mask, "lt"))
        if return_derivative:
            # sech²(x/c) = 4e / (1+e)² avoids cancellation near saturation.
            derivative = S.vdiv(S.vmuls(decay, 4.0, mask), S.vmul(divisor, divisor, mask), mask, precision="exact")
            return signed, derivative
        return signed

    def cap(value, mask):
        return S.vmuls(tanh(value, mask), softcap, mask) if softcap > 0 else value

    # Row state: max, exponential sum, target logit, two reciprocals,
    # target weight, shifted weighted-logit sum, cumulative weight.
    # Slots 8:10 hold aligned statistics for deferred backward.
    @T.macro
    def fold(x, w, state, target, start, length, slot):
        with T.SimdVF():
            mask = S.pset(32)
            S.mem_bar("VST_VLD")
            zero = S.vdup(0.0, T.float32, mask)
            negative = S.vdup(-T.infinity("float32"), T.float32, mask)
            run_max = S.vld(state[0], dist="BRC_B32")
            run_sum = S.vld(state[1], dist="BRC_B32")
            run_target = S.vld(state[2], dist="BRC_B32")
            label = S.vld(target[0], dist="BRC_B32")
            lane_max = S.alloc_var(T.float32)
            lane_max = run_max
            lane_target = S.alloc_var(T.float32)
            lane_target = zero
            if has_weight:
                lane_target_weight = S.alloc_var(T.float32)
                lane_target_weight = zero
            for vec in T.serial(tilelang.cdiv(length, 64)):
                lane = S.vci(T.int32(0), T.int32)
                live = S.vcmps(lane, length - vec * 64, mask, "lt")
                value = S.vsel(cap(load(x, vec * 64, slot), mask), negative, live)
                lane_max = S.vmax(lane_max, value, mask)
                columns = S.vadds(lane, start + vec * 64, mask)
                chosen = S.vcmp(columns, label, live, "eq")
                lane_target = S.vadd(lane_target, S.vsel(value, zero, chosen), mask)
                if has_weight:
                    lane_target_weight = S.vadd(
                        lane_target_weight, S.vsel(weight_value(w, vec * 64, slot), zero, chosen), mask
                    )
            new_max = S.vdupv(S.vcmax(lane_max, mask), mask)
            lane_sum = S.alloc_var(T.float32)
            lane_sum = zero
            if smoothing > 0:
                lane_smooth = S.alloc_var(T.float32)
                lane_smooth = zero
                if has_weight:
                    lane_weights = S.alloc_var(T.float32)
                    lane_weights = zero
            for vec in T.serial(tilelang.cdiv(length, 64)):
                lane2 = S.vci(T.int32(0), T.int32)
                live2 = S.vcmps(lane2, length - vec * 64, mask, "lt")
                value2 = S.vsel(cap(load(x, vec * 64, slot), mask), negative, live2)
                lane_sum = S.vadd(lane_sum, S.vexp(S.vsub(value2, new_max, mask), mask), mask)
                if smoothing > 0:
                    shifted = S.vsel(S.vsub(value2, new_max, mask), zero, live2)
                    if has_weight:
                        wi = S.vsel(weight_value(w, vec * 64, slot), zero, live2)
                        shifted = S.vmul(shifted, wi, mask)
                        lane_weights = S.vadd(lane_weights, wi, mask)
                    lane_smooth = S.vadd(lane_smooth, shifted, mask)
            block_sum = S.vdupv(S.vcadd(lane_sum, mask), mask)
            merged = S.vadd(S.vmul(S.vexp(S.vsub(run_max, new_max, mask), mask), run_sum, mask), block_sum, mask)
            # Only the target lane can be nonzero; select it directly to avoid
            # an additional horizontal reduction.
            target_lane = S.vand(label, S.vdup(63, T.int32, mask), mask)
            merged_target = S.vadd(run_target, S.vselr(lane_target, target_lane), mask)
            S.vsts(state[0], new_max, mask, "ONEPT_B32")
            S.vsts(state[1], merged, mask, "ONEPT_B32")
            S.vsts(state[2], merged_target, mask, "ONEPT_B32")
            if has_weight:
                target_weight = S.vadd(S.vld(state[5], dist="BRC_B32"), S.vselr(lane_target_weight, target_lane), mask)
                S.vsts(state[5], target_weight, mask, "ONEPT_B32")
            if smoothing > 0:
                # Keep weighted logits relative to the running maximum, including
                # the correction when a later tile raises that maximum.
                previous_weights = S.vld(state[7], dist="BRC_B32")
                previous_smooth = S.vld(state[6], dist="BRC_B32")
                shift = S.vsel(
                    zero, S.vsub(run_max, new_max, mask), S.vcmps(run_max, -T.infinity("float32"), mask, "eq")
                )
                previous_smooth = S.vadd(previous_smooth, S.vmul(previous_weights, shift, mask), mask)
                block_smooth = S.vdupv(S.vcadd(lane_smooth, mask), mask)
                block_weights = (
                    S.vdupv(S.vcadd(lane_weights, mask), mask)
                    if has_weight
                    else S.vdup(T.cast(length, "float32"), T.float32, mask)
                )
                S.vsts(state[6], S.vadd(previous_smooth, block_smooth, mask), mask, "ONEPT_B32")
                S.vsts(state[7], S.vadd(previous_weights, block_weights, mask), mask, "ONEPT_B32")

    @T.macro
    def apply(x, dx, w, state, target, start, length, slot, ignore_index, denominator, valid_count, weight_sum):
        with T.SimdVF():
            mask = S.pset(32)
            S.mem_bar("VST_VLD")
            max_value = S.vld(state[0], dist="BRC_B32")
            label = S.vld(target[0], dist="BRC_B32")
            zero = S.vdup(0.0, T.float32, mask)
            one = S.vdup(1.0, T.float32, mask)
            if resident:
                # A resident row needs these reciprocals only once.
                sum_value = S.vld(state[1], dist="BRC_B32")
                apply_inv_sum = S.vdiv(one, sum_value, mask, precision="exact")
                apply_inv_den = S.vdiv(one, S.vdup(denominator, T.float32, mask), mask, precision="exact")
            else:
                # Reload row reciprocals from UB: SIMD SSA values cannot cross
                # a T.Pipelined region.
                apply_inv_sum = S.vld(state[3], dist="BRC_B32")
                apply_inv_den = S.vld(state[4], dist="BRC_B32")
            ignored = S.vcmps(label, ignore_index, mask, "eq")
            for vec in T.serial(tilelang.cdiv(length, 64)):
                lane = S.vci(T.int32(0), T.int32)
                live = S.vcmps(lane, length - vec * 64, mask, "lt")
                columns = S.vadds(lane, start + vec * 64, mask)
                value = load(x, vec * 64, slot)
                if softcap > 0:
                    cap_tanh, cap_derivative = tanh(value, mask, return_derivative=True)
                    value = S.vmuls(cap_tanh, softcap, mask)
                p = S.vmul(S.vexp(S.vsub(value, max_value, mask), mask), apply_inv_sum, mask)
                if has_weight or smoothing > 0 or z_scale > 0 or softcap > 0:
                    target_weight = S.vld(state[5], dist="BRC_B32") if has_weight else one
                    coefficient = S.vadds(
                        S.vmuls(target_weight, 1.0 - smoothing, mask), smoothing / V * weight_sum, mask
                    )
                    g = S.vmul(p, coefficient, mask)
                    if smoothing > 0:
                        wi = weight_value(w, vec * 64, slot) if has_weight else one
                        g = S.vsub(g, S.vmuls(wi, smoothing / V, mask), mask)
                    g = S.vsub(
                        g,
                        S.vsel(S.vmuls(target_weight, 1.0 - smoothing, mask), zero, S.vcmp(columns, label, mask, "eq")),
                        mask,
                    )
                    g = S.vmul(g, apply_inv_den, mask)
                    if z_scale > 0:
                        lse = S.vadd(max_value, S.vln(S.vld(state[1], dist="BRC_B32"), mask), mask)
                        z_derivative = S.vdiv(
                            S.vmuls(lse, 2.0 * z_scale, mask),
                            S.vdup(valid_count, T.float32, mask),
                            mask,
                            precision="exact",
                        )
                        g = S.vadd(g, S.vmul(z_derivative, p, mask), mask)
                    if softcap > 0:
                        g = S.vmul(g, cap_derivative, mask)
                else:
                    g = S.vmul(
                        S.vsub(p, S.vsel(one, zero, S.vcmp(columns, label, mask, "eq")), mask),
                        apply_inv_den,
                        mask,
                    )
                store(dx, vec * 64, S.vsel(zero, g, ignored), slot, live)

    @T.macro
    def row_body(
        row,
        x,
        dx,
        w,
        target,
        state,
        loss,
        zloss,
        X,
        Target,
        W,
        Loss,
        ZLoss,
        DX,
        Stats,
        ignore_index,
        denominator,
        valid_count,
        weight_sum,
    ):
        T.copy(Target[row : row + 1], target[:1])
        with T.SimdVF():
            mask = S.pset(32)
            S.vsts(state[0], S.vdup(-T.infinity("float32"), T.float32, mask), mask, "ONEPT_B32")
            S.vsts(state[1], S.vdup(0.0, T.float32, mask), mask, "ONEPT_B32")
            S.vsts(state[2], S.vdup(0.0, T.float32, mask), mask, "ONEPT_B32")
            if has_weight:
                S.vsts(state[5], S.vdup(0.0, T.float32, mask), mask, "ONEPT_B32")
            if smoothing > 0:
                S.vsts(state[6], S.vdup(0.0, T.float32, mask), mask, "ONEPT_B32")
                S.vsts(state[7], S.vdup(0.0, T.float32, mask), mask, "ONEPT_B32")
        if resident:
            T.copy(X[row, :V], x[:V], l2_cache_ctrl="NORMAL_PERS")
            if has_weight:
                T.copy(W[:V], w[:V])
            fold(x, w, state, target, 0, V, 0)
        else:
            for block in T.Pipelined(full_blocks, num_stages=active_stages):
                if manual:
                    if block % 2 == 0:
                        T.copy(X[row, block * tile : (block + 1) * tile], x[0, :], l2_cache_ctrl="NORMAL_PERS")
                        if has_weight:
                            T.copy(W[block * tile : (block + 1) * tile], w[0, :])
                        fold(x, w, state, target, block * tile, tile, 0)
                    else:
                        T.copy(X[row, block * tile : (block + 1) * tile], x[1, :], l2_cache_ctrl="NORMAL_PERS")
                        if has_weight:
                            T.copy(W[block * tile : (block + 1) * tile], w[1, :])
                        fold(x, w, state, target, block * tile, tile, 1)
                else:
                    T.copy(X[row, block * tile : (block + 1) * tile], x, l2_cache_ctrl="NORMAL_PERS")
                    if has_weight:
                        T.copy(W[block * tile : (block + 1) * tile], w)
                    fold(x, w, state, target, block * tile, tile, 0)
            if tail > 0:
                if manual:
                    T.copy(X[row, full_blocks * tile : V], x[0, :tail], l2_cache_ctrl="NORMAL_PERS")
                    if has_weight:
                        T.copy(W[full_blocks * tile : V], w[0, :tail])
                else:
                    T.copy(X[row, full_blocks * tile : V], x[:tail], l2_cache_ctrl="NORMAL_PERS")
                    if has_weight:
                        T.copy(W[full_blocks * tile : V], w[:tail])
                fold(x, w, state, target, full_blocks * tile, tail, 0)
        with T.SimdVF():
            mask = S.pset(32)
            S.mem_bar("VST_VLD")
            m = S.vld(state[0], dist="BRC_B32")
            d = S.vld(state[1], dist="BRC_B32")
            chosen = S.vld(state[2], dist="BRC_B32")
            label = S.vld(target[0], dist="BRC_B32")
            zero = S.vdup(0.0, T.float32, mask)
            ce = S.vadd(S.vsub(m, chosen, mask), S.vln(d, mask), mask)
            if has_weight:
                ce = S.vmul(ce, S.vld(state[5], dist="BRC_B32"), mask)
            if smoothing > 0:
                smooth = S.vsub(S.vmuls(S.vln(d, mask), weight_sum, mask), S.vld(state[6], dist="BRC_B32"), mask)
                ce = S.vadd(S.vmuls(ce, 1.0 - smoothing, mask), S.vmuls(smooth, smoothing / V, mask), mask)
            result = S.vdiv(ce, S.vdup(denominator, T.float32, mask), mask)
            z = zero
            if z_scale > 0:
                lse = S.vadd(m, S.vln(d, mask), mask)
                z = S.vdiv(S.vmuls(S.vmul(lse, lse, mask), z_scale, mask), S.vdup(valid_count, T.float32, mask), mask)
                result = S.vadd(result, z, mask)
            selected = S.vsel(zero, result, S.vcmps(label, ignore_index, mask, "eq"))
            S.vsts(
                loss[0],
                S.vcvt(selected, T.bfloat16) if narrow else selected,
                S.pset(32, "PAT_VL1"),
                store_dist,
                extent=1,
            )
            if return_z_loss:
                logged_z = S.vsel(zero, z, S.vcmps(label, ignore_index, mask, "eq"))
                S.vsts(
                    zloss[0],
                    S.vcvt(logged_z, T.bfloat16) if narrow else logged_z,
                    S.pset(32, "PAT_VL1"),
                    store_dist,
                    extent=1,
                )
            if save_stats:
                S.vsts(state[8], m, S.pset(32, "PAT_VL1"), "ONEPT_B32", extent=1)
                inv_sum = S.vdiv(S.vdup(1.0, T.float32, mask), d, mask, precision="exact")
                S.vsts(state[9], inv_sum, S.pset(32, "PAT_VL1"), "ONEPT_B32", extent=1)
        T.copy(loss[:1], Loss[row : row + 1])
        if return_z_loss:
            T.copy(zloss[:1], ZLoss[row : row + 1])
        if save_stats:
            T.copy(state[8:10], Stats[row, :2])
        if need_grad:
            if resident:
                apply(x, x, w, state, target, 0, V, 0, ignore_index, denominator, valid_count, weight_sum)
                T.copy(x[:V], DX[row, :V])
            else:
                # Compute row reciprocals once for reuse across all gradient tiles.
                with T.SimdVF():
                    mask = S.pset(32)
                    one = S.vdup(1.0, T.float32, mask)
                    sum_value = S.vld(state[1], dist="BRC_B32")
                    inv_sum = S.vdiv(one, sum_value, mask, precision="exact")
                    inv_den = S.vdiv(one, S.vdup(denominator, T.float32, mask), mask, precision="exact")
                    S.vsts(state[3], inv_sum, mask, "ONEPT_B32")
                    S.vsts(state[4], inv_den, mask, "ONEPT_B32")
                for block in T.Pipelined(full_blocks, num_stages=active_stages):
                    if manual:
                        if block % 2 == 0:
                            T.copy(X[row, block * tile : (block + 1) * tile], x[0, :], l2_cache_ctrl="NORMAL_PERS")
                            if has_weight:
                                T.copy(W[block * tile : (block + 1) * tile], w[0, :])
                            apply(
                                x,
                                dx,
                                w,
                                state,
                                target,
                                block * tile,
                                tile,
                                0,
                                ignore_index,
                                denominator,
                                valid_count,
                                weight_sum,
                            )
                            T.copy(dx[0, :], DX[row, block * tile : (block + 1) * tile])
                        else:
                            T.copy(X[row, block * tile : (block + 1) * tile], x[1, :], l2_cache_ctrl="NORMAL_PERS")
                            if has_weight:
                                T.copy(W[block * tile : (block + 1) * tile], w[1, :])
                            apply(
                                x,
                                dx,
                                w,
                                state,
                                target,
                                block * tile,
                                tile,
                                1,
                                ignore_index,
                                denominator,
                                valid_count,
                                weight_sum,
                            )
                            T.copy(dx[1, :], DX[row, block * tile : (block + 1) * tile])
                    else:
                        T.copy(X[row, block * tile : (block + 1) * tile], x, l2_cache_ctrl="NORMAL_PERS")
                        if has_weight:
                            T.copy(W[block * tile : (block + 1) * tile], w)
                        apply(
                            x,
                            dx,
                            w,
                            state,
                            target,
                            block * tile,
                            tile,
                            0,
                            ignore_index,
                            denominator,
                            valid_count,
                            weight_sum,
                        )
                        T.copy(dx, DX[row, block * tile : (block + 1) * tile])
                if tail > 0:
                    if manual:
                        T.copy(X[row, full_blocks * tile : V], x[0, :tail], l2_cache_ctrl="NORMAL_PERS")
                        if has_weight:
                            T.copy(W[full_blocks * tile : V], w[0, :tail])
                    else:
                        T.copy(X[row, full_blocks * tile : V], x[:tail], l2_cache_ctrl="NORMAL_PERS")
                        if has_weight:
                            T.copy(W[full_blocks * tile : V], w[:tail])
                    apply(
                        x,
                        dx,
                        w,
                        state,
                        target,
                        full_blocks * tile,
                        tail,
                        0,
                        ignore_index,
                        denominator,
                        valid_count,
                        weight_sum,
                    )
                    if manual:
                        T.copy(dx[0, :tail], DX[row, full_blocks * tile : V])
                    else:
                        T.copy(dx[:tail], DX[row, full_blocks * tile : V])

    @T.prim_func
    def main(
        X_handle: T.handle,
        Target_handle: T.handle,
        W_handle: T.handle,
        Loss_handle: T.handle,
        ZLoss_handle: T.handle,
        DX_handle: T.handle,
        ignore_index: T.int32,
        denominator: T.float32,
        valid_count: T.float32,
        weight_sum: T.float32,
        Stats_handle: T.handle,
        N: T.int64,
    ):
        X = T.match_buffer(X_handle, (N, V), dtype)
        Target = T.match_buffer(Target_handle, (N,), "int32")
        W = T.match_buffer(W_handle, (V if has_weight else 0,), "float32")
        Loss = T.match_buffer(Loss_handle, (N,), dtype)
        ZLoss = T.match_buffer(ZLoss_handle, (N,), dtype)
        DX = T.match_buffer(DX_handle, (N, V), dtype)
        Stats = T.match_buffer(Stats_handle, (N, 8) if save_stats else (0, 8), "float32")
        with T.Kernel(cores) as core:
            x = T.alloc_shared((2, tile) if manual else (tile,), dtype)
            w = (
                T.alloc_shared((2, tile) if manual else (tile,), "float32")
                if has_weight
                else T.alloc_shared((0,), "float32")
            )
            if not resident:
                dx = T.alloc_shared((2, tile) if manual else (tile,), dtype)
            target = T.alloc_shared((8,), "int32")
            state = T.alloc_shared((16,), "float32")
            loss = T.alloc_shared((16 if narrow else 8,), dtype)
            zloss = T.alloc_shared((16 if narrow else 8,), dtype)
            T.annotate_buffer_versions({target: 1, state: 1, loss: 1, zloss: 1})
            if not resident:
                if manual:
                    T.annotate_manual_multi_buffer(x, dx)
                    if has_weight:
                        T.annotate_manual_multi_buffer(w)
                else:
                    T.annotate_buffer_versions({x: active_stages, dx: active_stages})
                    if has_weight:
                        T.annotate_buffer_versions({w: active_stages})

            for wave in T.serial(N // cores):
                row_body(
                    wave * cores + core,
                    x,
                    x if resident else dx,
                    w,
                    target,
                    state,
                    loss,
                    zloss,
                    X,
                    Target,
                    W,
                    Loss,
                    ZLoss,
                    DX,
                    Stats,
                    ignore_index,
                    denominator,
                    valid_count,
                    weight_sum,
                )
            if core < N % cores:
                row_body(
                    (N // cores) * cores + core,
                    x,
                    x if resident else dx,
                    w,
                    target,
                    state,
                    loss,
                    zloss,
                    X,
                    Target,
                    W,
                    Loss,
                    ZLoss,
                    DX,
                    Stats,
                    ignore_index,
                    denominator,
                    valid_count,
                    weight_sum,
                )

    return main


@tilelang.jit
def simd_scale_gradient_kernel(V, dtype, cores, vector_grad, tile=8192, stages=2, storage="auto"):
    """Scale saved gradients by a scalar or per-row upstream gradient."""
    N = T.dynamic("rows", "int64")
    narrow = dtype == "bfloat16"
    manual = storage == "manual" and stages == 2
    total = N * V
    full_tasks = total // tile
    tail = T.cast(total % tile, "int32")
    shape = (2, tile) if manual else (tile,)
    store_dist = "PK_B32" if narrow else "NORM_B32"

    @T.macro
    def compute(x, y, grad, length, slot):
        with T.SimdVF():
            mask = S.pset(32)
            # Grad is cast once from its original dtype, then broadcast in fp32.
            raw_grad = S.vld(grad[0], dist="BRC_B16" if narrow else "BRC_B32")
            gr = S.vcvt(raw_grad, T.float32) if narrow else raw_grad
            for vec in T.serial(tilelang.cdiv(length, 64)):
                lane = S.vci(T.int32(0), T.int32)
                live = S.vcmps(lane, length - vec * 64, mask, "lt")
                if manual:
                    raw = S.vld(x[slot, vec * 64], dist="UNPK_B16" if narrow else "NORM")
                else:
                    raw = S.vld(x[vec * 64], dist="UNPK_B16" if narrow else "NORM")
                value = S.vcvt(raw, T.float32) if narrow else raw
                product = S.vmul(value, gr, mask)
                converted = S.vcvt(product, T.bfloat16) if narrow else product
                if manual:
                    S.vsts(y[slot, vec * 64], converted, live, store_dist)
                else:
                    S.vsts(y[vec * 64], converted, live, store_dist)

    @T.macro
    def task(task_id, length, slot, x, y, grad, DX, Out):
        begin = task_id * tile
        if manual:
            T.copy(T.Tensor((N * V,), dtype, DX.data)[begin : begin + length], x[slot, :length])
        else:
            T.copy(T.Tensor((N * V,), dtype, DX.data)[begin : begin + length], x[:length])
        compute(x, y, grad, length, slot)
        if manual:
            T.copy(y[slot, :length], T.Tensor((N * V,), dtype, Out.data)[begin : begin + length])
        else:
            T.copy(y[:length], T.Tensor((N * V,), dtype, Out.data)[begin : begin + length])

    @T.prim_func
    def main(
        DX: T.Tensor((N, V), dtype), Grad: T.Tensor((N if vector_grad else 1,), dtype), Out: T.Tensor((N, V), dtype)
    ):
        with T.Kernel(cores) as core:
            x = T.alloc_shared(shape, dtype)
            y = T.alloc_shared(shape, dtype)
            grad = T.alloc_shared((16,), dtype)
            T.annotate_buffer_versions({grad: 1})
            if manual:
                T.annotate_manual_multi_buffer(x, y)
            else:
                T.annotate_buffer_versions({x: stages, y: stages})

            if vector_grad:
                # A vector upstream requires preserving each row's factor; no
                # flattened task may span two rows with different factors.
                for wave in T.serial(tilelang.cdiv(N, cores)):
                    row = wave * cores + core
                    if row < N:
                        T.copy(Grad[row : row + 1], grad[:1])
                        for block in T.serial(tilelang.cdiv(V, tile)):
                            begin = row * V + block * tile
                            valid = T.min(tile, V - block * tile)
                            if manual:
                                T.copy(T.Tensor((N * V,), dtype, DX.data)[begin : begin + valid], x[0, :valid])
                            else:
                                T.copy(T.Tensor((N * V,), dtype, DX.data)[begin : begin + valid], x[:valid])
                            compute(x, y, grad, valid, 0)
                            if manual:
                                T.copy(y[0, :valid], T.Tensor((N * V,), dtype, Out.data)[begin : begin + valid])
                            else:
                                T.copy(y[:valid], T.Tensor((N * V,), dtype, Out.data)[begin : begin + valid])
            else:
                T.copy(Grad[:1], grad[:1])
                for wave in T.Pipelined(full_tasks // cores, num_stages=stages):
                    if manual:
                        if wave % 2 == 0:
                            task(wave * cores + core, tile, 0, x, y, grad, DX, Out)
                        else:
                            task(wave * cores + core, tile, 1, x, y, grad, DX, Out)
                    else:
                        task(wave * cores + core, tile, 0, x, y, grad, DX, Out)
                if core < full_tasks % cores:
                    task((full_tasks // cores) * cores + core, tile, 0, x, y, grad, DX, Out)
                if tail > 0:
                    if core == 0:
                        task(full_tasks, tail, 0, x, y, grad, DX, Out)

    return main


@tilelang.jit
def simd_deferred_gradient_kernel(V, dtype, cores, tile=8192, stages=2, storage="auto"):
    """Recompute CE gradients from logits, row maxima and inverse exponential sums."""
    narrow = dtype == "bfloat16"
    # Keep both input and output in UB for small rows. The two buffers avoid
    # racing an asynchronous MTE3 writeback against the next row's input copy.
    resident = V <= 32000
    tile = tilelang.cdiv(V, 64) * 64 if resident else (31680 if narrow else 15808)
    full_blocks, tail = divmod(V, tile)
    manual = storage == "manual" and stages == 2 and not resident
    active_stages = 1 if resident else stages
    shape = (2, tile) if manual else (tile,)
    store_dist = "PK_B32" if narrow else "NORM_B32"

    @T.macro
    def apply(x, dx, target, state, grad, length, start, slot, ignore_index):
        with T.SimdVF():
            mask = S.pset(32)
            S.mem_bar("VST_VLD")
            row_max = S.vld(state[0], dist="BRC_B32")
            inv_sum = S.vld(state[1], dist="BRC_B32")
            inv_den = S.vld(state[8], dist="BRC_B32")
            label = S.vld(target[0], dist="BRC_B32")
            raw_grad = S.vld(grad[0], dist="BRC_B16" if narrow else "BRC_B32")
            upstream = S.vcvt(raw_grad, T.float32) if narrow else raw_grad
            zero = S.vdup(0.0, T.float32, mask)
            one = S.vdup(1.0, T.float32, mask)
            ignored = S.vcmps(label, ignore_index, mask, "eq")
            for vec in T.serial(tilelang.cdiv(length, 64)):
                lane = S.vci(T.int32(0), T.int32)
                live = S.vcmps(lane, length - vec * 64, mask, "lt")
                columns = S.vadds(lane, start + vec * 64, mask)
                is_target = S.vcmp(columns, label, mask, "eq")
                raw = S.vld(
                    x[slot, vec * 64] if manual else x[vec * 64],
                    dist="UNPK_B16" if narrow else "NORM",
                )
                value = S.vcvt(raw, T.float32) if narrow else raw
                probability = S.vmul(S.vexp(S.vsub(value, row_max, mask), mask), inv_sum, mask)
                probability = S.vsub(probability, S.vsel(one, zero, is_target), mask)
                result = S.vmul(S.vmul(probability, upstream, mask), inv_den, mask)
                result = S.vsel(zero, result, ignored)
                converted = S.vcvt(result, T.bfloat16) if narrow else result
                if manual:
                    S.vsts(dx[slot, vec * 64], converted, live, store_dist)
                else:
                    S.vsts(dx[vec * 64], converted, live, store_dist)

    @T.prim_func
    def main(
        X_handle: T.handle,
        Target_handle: T.handle,
        Stats_handle: T.handle,
        Grad_handle: T.handle,
        DX_handle: T.handle,
        ignore_index: T.int32,
        denominator: T.float32,
        N: T.int64,
    ):
        X = T.match_buffer(X_handle, (N, V), dtype)
        Target = T.match_buffer(Target_handle, (N,), "int32")
        Stats = T.match_buffer(Stats_handle, (N, 8), "float32")
        Grad = T.match_buffer(Grad_handle, (1,), dtype)
        DX = T.match_buffer(DX_handle, (N, V), dtype)
        with T.Kernel(cores) as core:
            x = T.alloc_shared(shape, dtype)
            if not resident:
                dx = T.alloc_shared(shape, dtype)
            target = T.alloc_shared((8,), "int32")
            state = T.alloc_shared((16,), "float32")
            grad = T.alloc_shared((16,), dtype)
            T.annotate_buffer_versions({target: 1, state: 1, grad: 1})
            if manual:
                T.annotate_manual_multi_buffer(x, dx)
            elif not resident:
                T.annotate_buffer_versions({x: active_stages, dx: active_stages})

            for wave in T.serial(tilelang.cdiv(N, cores)):
                row = wave * cores + core
                if row < N:
                    T.copy(Target[row : row + 1], target[:1])
                    T.copy(Stats[row, :2], state[:2])
                    T.copy(Grad[:1], grad[:1])
                    with T.SimdVF():
                        mask = S.pset(32)
                        one = S.vdup(1.0, T.float32, mask)
                        inv_den = S.vdiv(
                            one,
                            S.vdup(denominator, T.float32, mask),
                            mask,
                            precision="exact",
                        )
                        # Keep the scalar reciprocal at a 32-byte aligned slot.
                        S.vsts(state[8], inv_den, mask, "ONEPT_B32")
                    if resident:
                        T.copy(X[row, :V], x[:V], l2_cache_ctrl="NORMAL_PERS")
                        apply(x, x, target, state, grad, V, 0, 0, ignore_index)
                        T.copy(x[:V], DX[row, :V])
                    else:
                        for block in T.Pipelined(full_blocks, num_stages=active_stages):
                            if manual:
                                if block % 2 == 0:
                                    T.copy(
                                        X[row, block * tile : (block + 1) * tile], x[0, :], l2_cache_ctrl="NORMAL_PERS"
                                    )
                                    apply(x, dx, target, state, grad, tile, block * tile, 0, ignore_index)
                                    T.copy(dx[0, :], DX[row, block * tile : (block + 1) * tile])
                                else:
                                    T.copy(
                                        X[row, block * tile : (block + 1) * tile], x[1, :], l2_cache_ctrl="NORMAL_PERS"
                                    )
                                    apply(x, dx, target, state, grad, tile, block * tile, 1, ignore_index)
                                    T.copy(dx[1, :], DX[row, block * tile : (block + 1) * tile])
                            else:
                                T.copy(X[row, block * tile : (block + 1) * tile], x, l2_cache_ctrl="NORMAL_PERS")
                                apply(x, dx, target, state, grad, tile, block * tile, 0, ignore_index)
                                T.copy(dx, DX[row, block * tile : (block + 1) * tile])
                        if tail > 0:
                            T.copy(
                                X[row, full_blocks * tile : V],
                                x[0, :tail] if manual else x[:tail],
                                l2_cache_ctrl="NORMAL_PERS",
                            )
                            apply(x, dx, target, state, grad, tail, full_blocks * tile, 0, ignore_index)
                            if manual:
                                T.copy(dx[0, :tail], DX[row, full_blocks * tile : V])
                            else:
                                T.copy(dx[:tail], DX[row, full_blocks * tile : V])

    return main


def cross_entropy_kernel(
    N,
    V,
    dtype,
    cores,
    tile,
    has_weight,
    smoothing,
    softcap,
    z_scale,
    need_grad,
    pipeline_stages=2,
    pipeline_storage="auto",
    save_stats=False,
    return_z_loss=True,
):
    return simd_cross_entropy_kernel(
        V,
        dtype,
        cores,
        need_grad,
        stages=pipeline_stages,
        storage=pipeline_storage,
        save_stats=save_stats,
        return_z_loss=return_z_loss,
        has_weight=has_weight,
        smoothing=smoothing,
        softcap=softcap,
        z_scale=z_scale,
    )


def scale_gradient_kernel(N, V, dtype, cores, vector_grad, tile=8192, pipeline_stages=2, pipeline_storage="auto"):
    # Match the streaming forward tile size to stay within UB capacity.
    chosen_tile = 31680 if dtype == "bfloat16" else 15808
    return simd_scale_gradient_kernel(
        V,
        dtype,
        min(56, tilelang.cdiv(N * V, chosen_tile)),
        vector_grad,
        chosen_tile,
        stages=pipeline_stages,
        storage=pipeline_storage,
    )


DTYPES = (torch.bfloat16, torch.float32)


def _validate(input, target, weight, ignore_index, z_scale, smoothing, reduction, softcap):
    if not isinstance(input, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("input and target must be tensors")
    if input.ndim != 2 or min(input.shape) <= 0:
        raise ValueError("input must have positive shape [N,V]")
    if input.dtype not in DTYPES:
        raise TypeError("input must be bfloat16 or float32")
    if input.device.type != "npu" or target.device != input.device:
        raise ValueError("input and target must be on the same NPU")
    if not torch.npu.get_device_name(input.device).startswith("Ascend950"):
        raise ValueError("ascend-tilelang CrossEntropy requires Ascend950")
    if target.shape != (input.shape[0],) or target.dtype not in (torch.int32, torch.int64):
        raise ValueError("target must be an integer vector of length N")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError("reduction must be none, sum or mean")
    if not isinstance(ignore_index, int) or not -(2**31) <= ignore_index < 2**31:
        raise ValueError("ignore_index must fit int32")
    if not math.isfinite(float(smoothing)) or not 0 <= smoothing <= 1:
        raise ValueError("label_smoothing must lie in [0,1]")
    if not math.isfinite(float(z_scale)) or z_scale < 0:
        raise ValueError("lse_square_scale must be finite and nonnegative")
    if softcap is not None and (not math.isfinite(float(softcap)) or softcap <= 0):
        raise ValueError("softcap must be positive and finite")
    if weight is not None:
        if not isinstance(weight, torch.Tensor):
            raise TypeError("weight must be a tensor or None")
        if weight.shape != (input.shape[1],) or weight.device != input.device or not weight.is_floating_point():
            raise ValueError("weight must be a floating vector of length V on the same NPU")
        if weight.requires_grad:
            raise ValueError("weight gradients are unsupported")


def _forward(
    input,
    target,
    weight,
    ignore_index,
    z_scale,
    smoothing,
    reduction,
    softcap,
    need_grad,
    defer_grad=False,
    return_z_loss=False,
):
    _validate(input, target, weight, ignore_index, z_scale, smoothing, reduction, softcap)
    N, V = input.shape
    labels, count = prepare_targets(target, V, ignore_index)
    if weight is None:
        weights = torch.empty((0,), dtype=torch.float32, device=input.device)
        weight_sum = float(V)
        denominator = float(count)
    else:
        weights = weight.float().contiguous()
        # Fixed-size gather avoids boolean-index/nonzero synchronization. The
        # labels have already been validated, so only ignored values are clamped.
        selected = weights[target.clamp(0, V - 1).long()]
        selected = torch.where(target != ignore_index, selected, 0.0)
        weight_sum, denominator = torch.stack((weights.sum(), selected.sum())).cpu().tolist()
    if reduction != "mean":
        denominator = 1.0
    normalizer = max(1, count) if reduction == "mean" else 1
    # All-ignored rows are zero by contract. Keep the ignored branch free of 0/0.
    if count == 0:
        denominator = 1.0
    x = input.contiguous()
    stats = torch.empty((N if defer_grad else 0, 8), device=input.device, dtype=torch.float32)
    loss = torch.empty((N,), device=input.device, dtype=input.dtype)
    zloss = torch.empty_like(loss) if return_z_loss else loss
    # The forward-only specialization never accesses DX; alias X to satisfy its static ABI.
    dx = torch.empty_like(x) if need_grad and not defer_grad else x
    with torch.npu.device(input.device):
        cores = min(N, torch.npu.get_device_properties(input.device).vector_core_num)
        kernel = cross_entropy_kernel(
            N,
            V,
            str(input.dtype).removeprefix("torch."),
            cores,
            8192,
            weight is not None,
            float(smoothing),
            float(softcap or 0),
            float(z_scale),
            need_grad and not defer_grad,
            pipeline_stages=2,
            pipeline_storage="manual",
            save_stats=defer_grad,
            return_z_loss=return_z_loss,
        )
        args = (
            x,
            labels,
            weights,
            loss,
            zloss,
            dx,
            ignore_index,
            float(denominator),
            float(normalizer),
            float(weight_sum),
        )
        kernel(*args, stats, N)
    if reduction != "none":
        loss = loss.sum()
        if return_z_loss:
            zloss = zloss.sum()
    if not return_z_loss:
        zloss = None
    if defer_grad:
        return loss, zloss, x, labels, stats, float(denominator)
    return loss, zloss, dx


def _validate_return_flags(return_z_loss, return_token_accuracy, return_predicted_tokens):
    for name, value in (
        ("return_z_loss", return_z_loss),
        ("return_token_accuracy", return_token_accuracy),
        ("return_predicted_tokens", return_predicted_tokens),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be bool")
    if return_token_accuracy or return_predicted_tokens:
        raise NotImplementedError("ascend-tilelang CrossEntropy does not support token accuracy or predicted tokens")


class LigerCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        target,
        weight=None,
        ignore_index=-100,
        z_scale=0.0,
        smoothing=0.0,
        reduction="mean",
        softcap=None,
        return_z_loss=False,
        return_token_accuracy=False,
        return_predicted_tokens=False,
    ):
        _validate_return_flags(return_z_loss, return_token_accuracy, return_predicted_tokens)
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a tensor")
        need_grad = input.requires_grad
        defer_grad = (
            need_grad
            and weight is None
            and float(z_scale) == 0
            and float(smoothing) == 0
            and softcap is None
            and reduction == "mean"
        )
        if defer_grad:
            loss, zloss, x, labels, stats, denominator = _forward(
                input,
                target,
                weight,
                ignore_index,
                z_scale,
                smoothing,
                reduction,
                softcap,
                False,
                defer_grad=True,
                return_z_loss=return_z_loss,
            )
            ctx.defer_grad = True
            ctx.ignore_index = ignore_index
            ctx.denominator = denominator
            ctx.save_for_backward(x, labels, stats)
        else:
            loss, zloss, dx = _forward(
                input,
                target,
                weight,
                ignore_index,
                z_scale,
                smoothing,
                reduction,
                softcap,
                need_grad,
                return_z_loss=return_z_loss,
            )
            ctx.defer_grad = False
        if need_grad and not defer_grad:
            ctx.save_for_backward(dx)
        if return_z_loss:
            ctx.mark_non_differentiable(zloss)
        return loss, zloss if return_z_loss else None, None, None

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, _grad_z, _grad_accuracy, _grad_tokens):
        if ctx.defer_grad:
            x, target, stats = ctx.saved_tensors
            N, V = x.shape
            result = torch.empty_like(x)
            upstream = grad_output.contiguous().reshape(-1)
            with torch.npu.device(x.device):
                cores = min(N, torch.npu.get_device_properties(x.device).vector_core_num)
                kernel = simd_deferred_gradient_kernel(
                    V,
                    str(x.dtype).removeprefix("torch."),
                    cores,
                    tile=8192,
                    stages=2,
                    storage="manual",
                )
                kernel(x, target, stats, upstream, result, ctx.ignore_index, float(ctx.denominator), N)
        else:
            (dx,) = ctx.saved_tensors
            if grad_output.ndim == 0 and torch.equal(grad_output, torch.ones((), dtype=dx.dtype, device=dx.device)):
                result = dx
            else:
                N, V = dx.shape
                result = torch.empty_like(dx)
                with torch.npu.device(dx.device):
                    cores = min(N, torch.npu.get_device_properties(dx.device).vector_core_num)
                    kernel = scale_gradient_kernel(
                        N,
                        V,
                        str(dx.dtype).removeprefix("torch."),
                        cores,
                        grad_output.ndim > 0,
                        pipeline_stages=2,
                        pipeline_storage="manual",
                    )
                    kernel(dx, grad_output.contiguous().reshape(-1), result)
        return (result,) + (None,) * (len(ctx.needs_input_grad) - 1)


def cross_entropy(
    input,
    target,
    weight=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
    return_z_loss=False,
):
    """Compute cross-entropy without modifying inputs or class weights.

    Logits have shape [N, V] and BF16/FP32 dtype. All-ignored inputs return
    zero loss and gradients. Class weights are constants; only first-order
    input gradients are supported. When requested, the separate z-loss output
    is detached for logging; z-loss regularization still contributes to loss.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a tensor")
    if not isinstance(return_z_loss, bool):
        raise TypeError("return_z_loss must be bool")
    if torch.is_grad_enabled() and input.requires_grad:
        loss, zloss, _, _ = LigerCrossEntropyFunction.apply(
            input, target, weight, ignore_index, lse_square_scale, label_smoothing, reduction, softcap, return_z_loss
        )
    else:
        loss, zloss, _ = _forward(
            input,
            target,
            weight,
            ignore_index,
            lse_square_scale,
            label_smoothing,
            reduction,
            softcap,
            False,
            return_z_loss=return_z_loss,
        )
    return (loss, zloss) if return_z_loss else loss
