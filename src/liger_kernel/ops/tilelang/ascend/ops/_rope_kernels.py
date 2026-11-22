"""Ascend C RoPE kernels for dense and strided Q/K tensors."""

import tilelang
import tilelang.ascend.language as T


def _simd_rows(half, dtype, phase_dtype, backward):
    chunks = tilelang.cdiv(half, 64)

    @T.macro
    def compute(x, c, s, y, valid):
        with T.SimdVF():
            convert_mask = T.simd.pset(16)
            for row in T.serial(valid):
                for chunk in T.serial(chunks):
                    lane = chunk * 64
                    mask = T.simd.update_mask(T.min(64, half - lane), 32)
                    if dtype == "float32":
                        a = T.simd.vld(x[row, lane])
                        bval = T.simd.vld(x[row, lane + half])
                    else:
                        ar = T.simd.vld(x[row, lane], dist="UNPK_B16")
                        br = T.simd.vld(x[row, lane + half], dist="UNPK_B16")
                        a = T.simd.vcvt(ar, "float32", convert_mask, part=0)
                        bval = T.simd.vcvt(br, "float32", convert_mask, part=0)
                    if phase_dtype == "float32":
                        ca = T.simd.vld(c[row, lane])
                        sa = T.simd.vld(s[row, lane])
                    else:
                        car = T.simd.vld(c[row, lane], dist="UNPK_B16")
                        sar = T.simd.vld(s[row, lane], dist="UNPK_B16")
                        ca = T.simd.vcvt(car, "float32", convert_mask, part=0)
                        sa = T.simd.vcvt(sar, "float32", convert_mask, part=0)
                    ac = T.simd.vmul(a, ca, mask)
                    bc = T.simd.vmul(bval, ca, mask)
                    bs = T.simd.vmul(bval, sa, mask)
                    ass = T.simd.vmul(a, sa, mask)
                    if backward:
                        first = T.simd.vadd(ac, bs, mask)
                        second = T.simd.vsub(bc, ass, mask)
                    else:
                        first = T.simd.vsub(ac, bs, mask)
                        second = T.simd.vadd(bc, ass, mask)
                    if dtype == "float32":
                        T.simd.vsts(y[row, lane], first, mask)
                        T.simd.vsts(y[row, lane + half], second, mask)
                    else:
                        # NOSAT preserves the reference's infinity on overflow.
                        first_out = T.simd.vcvt(first, dtype, mask, sat=False)
                        second_out = T.simd.vcvt(second, dtype, mask, sat=False)
                        T.simd.vsts(y[row, lane], first_out, mask, dist="PK_B32")
                        T.simd.vsts(y[row, lane + half], second_out, mask, dist="PK_B32")

    return compute


def _simd_group(half, dtype, phase_dtype, backward, *, unroll=False):
    chunks = tilelang.cdiv(half, 64)

    @T.macro
    def compute_group(x, c, s, y, valid, heads):
        with T.SimdVF():
            convert_mask = T.simd.pset(16)
            for row in T.serial(valid):
                for chunk in T.serial(chunks):
                    lane = chunk * 64
                    mask = T.simd.update_mask(T.min(64, half - lane), 32)
                    if phase_dtype == "float32":
                        ca = T.simd.vld(c[row, lane])
                        sa = T.simd.vld(s[row, lane])
                    else:
                        car = T.simd.vld(c[row, lane], dist="UNPK_B16")
                        sar = T.simd.vld(s[row, lane], dist="UNPK_B16")
                        ca = T.simd.vcvt(car, "float32", convert_mask, part=0)
                        sa = T.simd.vcvt(sar, "float32", convert_mask, part=0)
                    for hi in T.unroll(heads, explicit=True) if unroll else T.serial(heads):
                        if dtype == "float32":
                            a = T.simd.vld(x[hi, row, lane])
                            bval = T.simd.vld(x[hi, row, lane + half])
                        else:
                            ar = T.simd.vld(x[hi, row, lane], dist="UNPK_B16")
                            br = T.simd.vld(x[hi, row, lane + half], dist="UNPK_B16")
                            a = T.simd.vcvt(ar, "float32", convert_mask, part=0)
                            bval = T.simd.vcvt(br, "float32", convert_mask, part=0)
                        ac = T.simd.vmul(a, ca, mask)
                        bc = T.simd.vmul(bval, ca, mask)
                        bs = T.simd.vmul(bval, sa, mask)
                        ass = T.simd.vmul(a, sa, mask)
                        if backward:
                            first = T.simd.vadd(ac, bs, mask)
                            second = T.simd.vsub(bc, ass, mask)
                        else:
                            first = T.simd.vsub(ac, bs, mask)
                            second = T.simd.vadd(bc, ass, mask)
                        if dtype == "float32":
                            T.simd.vsts(y[hi, row, lane], first, mask)
                            T.simd.vsts(y[hi, row, lane + half], second, mask)
                        else:
                            # NOSAT preserves the reference's infinity on overflow.
                            first_out = T.simd.vcvt(first, dtype, mask, sat=False)
                            second_out = T.simd.vcvt(second, dtype, mask, sat=False)
                            T.simd.vsts(y[hi, row, lane], first_out, mask, dist="PK_B32")
                            T.simd.vsts(y[hi, row, lane + half], second_out, mask, dist="PK_B32")

    return compute_group


@tilelang.jit(target="ascend")
def _ungrouped_kernel(B, H, S, D, Cb, dtype, phase_dtype, backward, num_cores):
    """Build a non-mutating BNSD rotation (or its transpose Jacobian).

    Call with contiguous X, Cos, Sin, Y tensors. Y has X's shape/dtype.
    Cos/Sin have shape (Cb, S, D), where Cb is 1 or B. All shape arguments
    are compile-time constants; the tensor values remain runtime inputs.
    """
    half = D // 2
    chunks = tilelang.cdiv(half, 64)
    # vld is unpredicated: the last second-half load must remain in its UB row.
    # A 64-element pitch multiple also satisfies every supported dtype's DMA alignment.
    pitch = tilelang.cdiv(max(D, half + chunks * 64), 64) * 64
    item_bytes = 4 if dtype == "float32" else 2
    phase_bytes = 4 if phase_dtype == "float32" else 2
    # Normal vlds/vsts require 32-byte UB addresses (errcode 340 on unaligned
    # FP16 D80). Keep only these unsupported half-offsets on the generic route.
    use_simd = half * item_bytes % 32 == 0 and half * phase_bytes % 32 == 0
    if not use_simd:
        pitch = D
    ub_limit = (248 if use_simd else 216) * 1024
    rows = min(32, (ub_limit // (pitch * (2 * item_bytes + 2 * phase_bytes))) // 8 * 8)
    tiles = tilelang.cdiv(S, rows)
    tasks = B * H * tiles
    cores = min(num_cores, tasks)

    compute = _simd_rows(half, dtype, phase_dtype, backward)

    @T.prim_func
    def main(
        X: T.Tensor((B, H, S, D), dtype),
        Cos: T.Tensor((Cb, S, D), phase_dtype),
        Sin: T.Tensor((Cb, S, D), phase_dtype),
        Y: T.Tensor((B, H, S, D), dtype),
    ):
        with T.Kernel(cores) as core:
            x = T.alloc_shared((rows, pitch), dtype)
            c = T.alloc_shared((rows, pitch), phase_dtype)
            s = T.alloc_shared((rows, pitch), phase_dtype)
            y = T.alloc_shared((rows, pitch), dtype)
            for step in T.serial(tilelang.cdiv(tasks, cores)):
                task = step * cores + core
                if task < tasks:
                    b = task // (H * tiles)
                    h = (task // tiles) % H
                    begin = (task % tiles) * rows
                    valid = T.min(rows, S - begin)
                    cb = b if Cb != 1 else 0
                    T.copy(X[b, h, begin : begin + valid, :], x[:valid, :D])
                    T.copy(Cos[cb, begin : begin + valid, :], c[:valid, :D])
                    T.copy(Sin[cb, begin : begin + valid, :], s[:valid, :D])
                    if use_simd:
                        compute(x, c, s, y, valid)
                    else:
                        with T.SimtVF(threads=128):
                            for row, lane in T.Parallel(rows, half):
                                if row < valid:
                                    a = T.cast(x[row, lane], "float32")
                                    bval = T.cast(x[row, lane + half], "float32")
                                    ca = T.cast(c[row, lane], "float32")
                                    sa = T.cast(s[row, lane], "float32")
                                    if backward:
                                        y[row, lane] = T.cast(a * ca + bval * sa, dtype)
                                        y[row, lane + half] = T.cast(bval * ca - a * sa, dtype)
                                    else:
                                        y[row, lane] = T.cast(a * ca - bval * sa, dtype)
                                        y[row, lane + half] = T.cast(bval * ca + a * sa, dtype)
                    T.copy(y[:valid, :D], Y[b, h, begin : begin + valid, :])

    return main


@tilelang.jit(target="ascend")
def _pair_kernel(B, Hq, Hk, S, D, Cb, dtype, phase_dtype, backward, num_cores):
    """Single-stage combined-head tasks; Q/K scalar branches stay outside VF."""
    rows = 32
    half = D // 2
    chunks = tilelang.cdiv(half, 64)
    pitch = tilelang.cdiv(max(D, half + chunks * 64), 64) * 64
    tiles = tilelang.cdiv(S, rows)
    tasks = B * (Hq + Hk) * tiles
    cores = min(num_cores, tasks)

    compute = _simd_rows(half, dtype, phase_dtype, backward)

    @T.macro
    def process_head(Input, Output, Cos, Sin, x, y, c, s, b, h, begin, valid):
        cb = b if Cb != 1 else 0
        T.copy(Input[b, h, begin : begin + valid, :], x[:valid, :D])
        T.copy(Cos[cb, begin : begin + valid, :], c[:valid, :D])
        T.copy(Sin[cb, begin : begin + valid, :], s[:valid, :D])
        compute(x, c, s, y, valid)
        T.copy(y[:valid, :D], Output[b, h, begin : begin + valid, :])

    @T.prim_func
    def main(
        Q: T.Tensor((B, Hq, S, D), dtype),
        K: T.Tensor((B, Hk, S, D), dtype),
        Cos: T.Tensor((Cb, S, D), phase_dtype),
        Sin: T.Tensor((Cb, S, D), phase_dtype),
        Qout: T.Tensor((B, Hq, S, D), dtype),
        Kout: T.Tensor((B, Hk, S, D), dtype),
    ):
        with T.Kernel(cores) as core:
            x = T.alloc_shared((rows, pitch), dtype)
            y = T.alloc_shared((rows, pitch), dtype)
            c = T.alloc_shared((rows, pitch), phase_dtype)
            s = T.alloc_shared((rows, pitch), phase_dtype)
            T.annotate_buffer_versions({x: 1, y: 1, c: 1, s: 1})
            for wave in T.serial(tilelang.cdiv(tasks, cores)):
                task = wave * cores + core
                if task < tasks:
                    b = task // ((Hq + Hk) * tiles)
                    h = task // tiles % (Hq + Hk)
                    begin = task % tiles * rows
                    valid = T.min(rows, S - begin)
                    if h < Hq:
                        process_head(Q, Qout, Cos, Sin, x, y, c, s, b, h, begin, valid)
                    else:
                        process_head(K, Kout, Cos, Sin, x, y, c, s, b, h - Hq, begin, valid)

    return main


def rope_pair_kernel(B, Hq, Hk, S, D, Cb, dtype, phase_dtype, backward, num_cores):
    """Return a Q/K pair kernel for small aligned nonempty inputs, else None."""
    item_bytes = 4 if dtype == "float32" else 2
    phase_bytes = 4 if phase_dtype == "float32" else 2
    aligned = D // 2 * item_bytes % 32 == 0 and D // 2 * phase_bytes % 32 == 0
    if min(B, Hq, Hk, S) <= 0 or S > 256 or D > 128 or not aligned:
        return None
    return _pair_kernel(B, Hq, Hk, S, D, Cb, dtype, phase_dtype, backward, num_cores)


@tilelang.jit(target="ascend")
def _grouped_phase_kernel(
    B, H, S, D, Cb, dtype, phase_dtype, backward, num_cores, stages, strides=None, group=4, tile_rows=None
):
    """Stage raw G4; one VF reuses phase registers across all four heads."""
    rows = tile_rows if tile_rows is not None else (16 if dtype == "float32" else 32)
    half = D // 2
    chunks = tilelang.cdiv(half, 64)
    pitch = tilelang.cdiv(max(D, half + chunks * 64), 64) * 64
    groups = H // group
    tiles = S // rows
    tasks = B * groups * tiles
    cores = min(num_cores, tasks)
    full_waves = tasks // cores

    compute_group = _simd_group(half, dtype, phase_dtype, backward)

    @T.macro
    def tail_group(X, Cos, Sin, Y, xt, ct, st, yt, b, hbegin, begin, valid, heads):
        cb = b if Cb != 1 else 0
        T.copy(Cos[cb, begin : begin + valid, :], ct[:valid, :D], l2_cache_ctrl="NORMAL_PERS")
        T.copy(Sin[cb, begin : begin + valid, :], st[:valid, :D], l2_cache_ctrl="NORMAL_PERS")
        for hi in T.serial(heads):
            T.copy(X[b, hbegin + hi, begin : begin + valid, :], xt[0, :valid, :D], l2_cache_ctrl="NOTALLOC_KEEP")
            compute_group(xt, ct, st, yt, valid, 1)
            T.copy(yt[0, :valid, :D], Y[b, hbegin + hi, begin : begin + valid, :])

    @T.prim_func
    def main(
        X: T.Tensor((B, H, S, D), dtype) if strides is None else T.StridedTensor((B, H, S, D), strides, dtype),
        Cos: T.Tensor((Cb, S, D), phase_dtype),
        Sin: T.Tensor((Cb, S, D), phase_dtype),
        Y: T.Tensor((B, H, S, D), dtype),
    ):
        with T.Kernel(cores) as core:
            x = T.alloc_shared((group, rows, pitch), dtype)
            y = T.alloc_shared((group, rows, pitch), dtype)
            c = T.alloc_shared((rows, pitch), phase_dtype)
            s = T.alloc_shared((rows, pitch), phase_dtype)
            # Every buffer is changed at the next outer task, including C/S.
            T.annotate_buffer_versions({x: stages, y: stages, c: stages, s: stages})
            for wave in T.Pipelined(full_waves, num_stages=stages):
                task = wave * cores + core
                b = task // (groups * tiles)
                hbegin = task // tiles % groups * group
                begin = task % tiles * rows
                cb = b if Cb != 1 else 0
                T.copy(Cos[cb, begin : begin + rows, :], c[:, :D], l2_cache_ctrl="NORMAL_PERS")
                T.copy(Sin[cb, begin : begin + rows, :], s[:, :D], l2_cache_ctrl="NORMAL_PERS")
                for hi in T.unroll(group, explicit=True):
                    T.copy(X[b, hbegin + hi, begin : begin + rows, :], x[hi, :, :D], l2_cache_ctrl="NOTALLOC_KEEP")
                compute_group(x, c, s, y, rows, group)
                for hi in T.unroll(group, explicit=True):
                    T.copy(y[hi, :, :D], Y[b, hbegin + hi, begin : begin + rows, :])
            if tasks % cores != 0 or H % group != 0 or S % rows != 0:
                xt = T.alloc_shared((1, rows, pitch), dtype)
                yt = T.alloc_shared((1, rows, pitch), dtype)
                ct = T.alloc_shared((rows, pitch), phase_dtype)
                st = T.alloc_shared((rows, pitch), phase_dtype)
                T.annotate_buffer_versions({xt: 1, yt: 1, ct: 1, st: 1})
                if tasks % cores != 0:
                    task = full_waves * cores + core
                    if task < tasks:
                        b = task // (groups * tiles)
                        hbegin = task // tiles % groups * group
                        begin = task % tiles * rows
                        tail_group(X, Cos, Sin, Y, xt, ct, st, yt, b, hbegin, begin, rows, group)
                if H % group != 0:
                    head_tasks = B * (H % group) * tiles
                    for wave in T.serial(tilelang.cdiv(head_tasks, cores)):
                        task = wave * cores + core
                        if task < head_tasks:
                            b = task // ((H % group) * tiles)
                            h = groups * group + task // tiles % (H % group)
                            begin = task % tiles * rows
                            tail_group(X, Cos, Sin, Y, xt, ct, st, yt, b, h, begin, rows, 1)
                if S % rows != 0:
                    seq_tasks = B * H
                    for wave in T.serial(tilelang.cdiv(seq_tasks, cores)):
                        task = wave * cores + core
                        if task < seq_tasks:
                            b = task // H
                            h = task % H
                            tail_group(X, Cos, Sin, Y, xt, ct, st, yt, b, h, tiles * rows, S % rows, 1)

    return main


@tilelang.jit(target="ascend")
def _grouped_pair_phase_kernel(
    B,
    Hq,
    Hk,
    S,
    D,
    Cb,
    dtype,
    phase_dtype,
    backward,
    num_cores,
    strides_q=None,
    strides_k=None,
    group=4,
    tile_rows=32,
    stages=2,
):
    """Fuse Q/K launches while retaining grouped G4 phase reuse and R32 tiles."""
    rows = tile_rows
    half = D // 2
    chunks = tilelang.cdiv(half, 64)
    pitch = tilelang.cdiv(max(D, half + chunks * 64), 64) * 64
    groups_q = Hq // group
    groups_k = Hk // group
    groups_total = groups_q + groups_k
    tiles = S // rows
    tasks = B * groups_total * tiles
    cores = min(num_cores, tasks)
    full_waves = tasks // cores

    compute_group = _simd_group(half, dtype, phase_dtype, backward)

    @T.macro
    def process_group(Input, Output, Cos, Sin, x, y, c, s, b, hbegin, begin, valid):
        cb = b if Cb != 1 else 0
        T.copy(Cos[cb, begin : begin + valid, :], c[:valid, :D], l2_cache_ctrl="NORMAL_PERS")
        T.copy(Sin[cb, begin : begin + valid, :], s[:valid, :D], l2_cache_ctrl="NORMAL_PERS")
        for hi in T.unroll(group, explicit=True):
            T.copy(
                Input[b, hbegin + hi, begin : begin + valid, :],
                x[hi, :valid, :D],
                l2_cache_ctrl="NOTALLOC_KEEP",
            )
        compute_group(x, c, s, y, valid, group)
        for hi in T.unroll(group, explicit=True):
            T.copy(y[hi, :valid, :D], Output[b, hbegin + hi, begin : begin + valid, :])

    @T.prim_func
    def main(
        Q: T.Tensor((B, Hq, S, D), dtype) if strides_q is None else T.StridedTensor((B, Hq, S, D), strides_q, dtype),
        K: T.Tensor((B, Hk, S, D), dtype) if strides_k is None else T.StridedTensor((B, Hk, S, D), strides_k, dtype),
        Cos: T.Tensor((Cb, S, D), phase_dtype),
        Sin: T.Tensor((Cb, S, D), phase_dtype),
        Qout: T.Tensor((B, Hq, S, D), dtype),
        Kout: T.Tensor((B, Hk, S, D), dtype),
    ):
        with T.Kernel(cores) as core:
            x = T.alloc_shared((group, rows, pitch), dtype)
            y = T.alloc_shared((group, rows, pitch), dtype)
            c = T.alloc_shared((rows, pitch), phase_dtype)
            s = T.alloc_shared((rows, pitch), phase_dtype)
            T.annotate_buffer_versions({x: stages, y: stages, c: stages, s: stages})
            for wave in T.Pipelined(full_waves, num_stages=stages):
                task = wave * cores + core
                b = task // (groups_total * tiles)
                group_index = task // tiles % groups_total
                begin = task % tiles * rows
                if group_index < groups_q:
                    process_group(Q, Qout, Cos, Sin, x, y, c, s, b, group_index * group, begin, rows)
                else:
                    process_group(
                        K,
                        Kout,
                        Cos,
                        Sin,
                        x,
                        y,
                        c,
                        s,
                        b,
                        (group_index - groups_q) * group,
                        begin,
                        rows,
                    )
            if tasks % cores != 0:
                xt = T.alloc_shared((group, rows, pitch), dtype)
                yt = T.alloc_shared((group, rows, pitch), dtype)
                ct = T.alloc_shared((rows, pitch), phase_dtype)
                st = T.alloc_shared((rows, pitch), phase_dtype)
                T.annotate_buffer_versions({xt: 1, yt: 1, ct: 1, st: 1})
                task = full_waves * cores + core
                if task < tasks:
                    b = task // (groups_total * tiles)
                    group_index = task // tiles % groups_total
                    begin = task % tiles * rows
                    if group_index < groups_q:
                        process_group(
                            Q,
                            Qout,
                            Cos,
                            Sin,
                            xt,
                            yt,
                            ct,
                            st,
                            b,
                            group_index * group,
                            begin,
                            rows,
                        )
                    else:
                        process_group(
                            K,
                            Kout,
                            Cos,
                            Sin,
                            xt,
                            yt,
                            ct,
                            st,
                            b,
                            (group_index - groups_q) * group,
                            begin,
                            rows,
                        )

    return main


@tilelang.jit(target="ascend")
def _sequence_pair_phase_kernel(
    B,
    Hq,
    Hk,
    S,
    D,
    Cb,
    dtype,
    phase_dtype,
    backward,
    num_cores,
    strides_q=None,
    strides_k=None,
    group=4,
    tile_rows=32,
    stages=2,
):
    """Fuse Q/K and reuse each sequence tile's phase across all head groups."""
    rows = tile_rows
    half = D // 2
    chunks = tilelang.cdiv(half, 64)
    pitch = tilelang.cdiv(max(D, half + chunks * 64), 64) * 64
    groups_q = Hq // group
    groups_k = Hk // group
    tiles = S // rows
    tasks = B * tiles
    cores = min(num_cores, tasks)
    full_waves = tasks // cores

    compute_group = _simd_group(half, dtype, phase_dtype, backward, unroll=True)

    @T.macro
    def process_group(Input, Output, x, y, c, s, batch, hbegin, begin):
        for hi in T.unroll(group, explicit=True):
            T.copy(
                Input[batch, hbegin + hi, begin : begin + rows, :],
                x[hi, :, :D],
                l2_cache_ctrl="NOTALLOC_KEEP",
            )
        compute_group(x, c, s, y, rows, group)
        for hi in T.unroll(group, explicit=True):
            T.copy(y[hi, :, :D], Output[batch, hbegin + hi, begin : begin + rows, :])

    @T.macro
    def process_sequence(Q, K, Cos, Sin, Qout, Kout, x, y, c, s, task):
        batch = task // tiles
        begin = task % tiles * rows
        cb = batch if Cb != 1 else 0
        T.copy(Cos[cb, begin : begin + rows, :], c[:, :D], l2_cache_ctrl="NORMAL_PERS")
        T.copy(Sin[cb, begin : begin + rows, :], s[:, :D], l2_cache_ctrl="NORMAL_PERS")
        for head_group in T.serial(groups_q):
            process_group(Q, Qout, x, y, c, s, batch, head_group * group, begin)
        for head_group in T.serial(groups_k):
            process_group(K, Kout, x, y, c, s, batch, head_group * group, begin)

    @T.prim_func
    def main(
        Q: T.Tensor((B, Hq, S, D), dtype) if strides_q is None else T.StridedTensor((B, Hq, S, D), strides_q, dtype),
        K: T.Tensor((B, Hk, S, D), dtype) if strides_k is None else T.StridedTensor((B, Hk, S, D), strides_k, dtype),
        Cos: T.Tensor((Cb, S, D), phase_dtype),
        Sin: T.Tensor((Cb, S, D), phase_dtype),
        Qout: T.Tensor((B, Hq, S, D), dtype),
        Kout: T.Tensor((B, Hk, S, D), dtype),
    ):
        with T.Kernel(cores) as core:
            x = T.alloc_shared((group, rows, pitch), dtype)
            y = T.alloc_shared((group, rows, pitch), dtype)
            c = T.alloc_shared((rows, pitch), phase_dtype)
            s = T.alloc_shared((rows, pitch), phase_dtype)
            T.annotate_buffer_versions({x: stages, y: stages, c: stages, s: stages})
            for wave in T.Pipelined(full_waves, num_stages=stages):
                task = wave * cores + core
                process_sequence(Q, K, Cos, Sin, Qout, Kout, x, y, c, s, task)
            if tasks % cores != 0:
                task = full_waves * cores + core
                if task < tasks:
                    process_sequence(Q, K, Cos, Sin, Qout, Kout, x, y, c, s, task)

    return main


def rope_long_pair_kernel(B, Hq, Hk, S, D, Cb, dtype, phase_dtype, backward, num_cores, strides_q, strides_k):
    """Select a Q/K schedule from the shape and full buffered UB footprint."""
    group, rows = 4, 32
    item_bytes = 4 if dtype == "float32" else 2
    phase_bytes = 4 if phase_dtype == "float32" else 2
    half = D // 2
    if S < 2048 or S % rows or Hq % group or Hk % group or D > 128 or half * item_bytes % 32 or half * phase_bytes % 32:
        return None

    pitch = tilelang.cdiv(max(D, half + tilelang.cdiv(half, 64) * 64), 64) * 64
    per_stage = rows * pitch * (2 * group * item_bytes + 2 * phase_bytes)
    args = (B, Hq, Hk, S, D, Cb, dtype, phase_dtype, backward, num_cores, strides_q, strides_k, group, rows)
    sequence_tasks = B * (S // rows)
    if S >= 8192 and sequence_tasks >= num_cores and 3 * per_stage <= 248 * 1024:
        return _sequence_pair_phase_kernel(*args, 3)

    tasks = sequence_tasks * (Hq // group + Hk // group)
    cores = min(num_cores, tasks)
    tail_bytes = per_stage if tasks % cores else 0
    if 2 * per_stage + tail_bytes <= 248 * 1024:
        return _grouped_pair_phase_kernel(*args, 2)
    return None


@tilelang.jit(target="ascend")
def _narrow_register_kernel(B, H, S, D, Cb, backward, num_cores, stages):
    """Rotate contiguous 128-element BF16 windows using register permutations."""
    rows = 128 if D == 16 else 64
    pitch = 16 if D == 16 else 64
    half = D // 2
    window_rows = 128 // pitch
    tiles = S // rows
    tasks = B * H * tiles
    cores = min(num_cores, max(tasks, B * H))
    full_waves = tasks // cores

    @T.macro
    def compute(x, c, s, y, valid):
        with T.SimdVF():
            m16 = T.simd.pset(16)
            m32 = T.simd.pset(32)
            index = T.simd.vci(0, "int16")
            coord = T.simd.vand(index, T.simd.vdup(pitch - 1, "int16"), m16)
            first = T.simd.vcmps(coord, half, m16, op="lt")
            swap = T.simd.vsel(T.simd.vadds(index, half, m16), T.simd.vadds(index, -half, m16), first)
            active = T.simd.vcmps(coord, D, m16, op="lt")
            # Padding indices must stay within the allocated 128-lane window.
            swap_index = T.simd.vsel(swap, index, active)
            even_lane = T.simd.vmuls(T.simd.vci(0, "int32"), 2, m32)
            even_coord = T.simd.vand(even_lane, T.simd.vdup(pitch - 1, "int32"), m32)
            odd_coord = T.simd.vadds(even_coord, 1, m32)
            first_even = T.simd.vcmps(even_coord, half, m32, op="lt")
            first_odd = T.simd.vcmps(odd_coord, half, m32, op="lt")
            # Both rotation halves use the first half of the phase table.
            phase_index = T.simd.vsel(index, T.simd.vadds(index, -half, m16), first)
            phase_index = T.simd.vsel(phase_index, index, active)
            zero = T.simd.vdup(0, "bfloat16", m16)
            for window in T.serial(tilelang.cdiv(valid, window_rows)):
                row = window * window_rows
                # Normal unpredicated loads stay inside the full allocated UB.
                # Explicit register masking initializes every padding lane.
                xr = T.simd.vsel(T.simd.vld(x[row, 0], dist="NORM"), zero, active)
                cr = T.simd.vsel(T.simd.vld(c[row, 0], dist="NORM"), zero, active)
                sr = T.simd.vsel(T.simd.vld(s[row, 0], dist="NORM"), zero, active)
                rotated = T.simd.vselr(xr, swap_index)
                cr = T.simd.vselr(cr, phase_index)
                phase = T.simd.vselr(sr, phase_index)
                xe = T.simd.vcvt(xr, "float32", m16, part=0)
                xo = T.simd.vcvt(xr, "float32", m16, part=1)
                ce = T.simd.vcvt(cr, "float32", m16, part=0)
                co = T.simd.vcvt(cr, "float32", m16, part=1)
                re = T.simd.vcvt(rotated, "float32", m16, part=0)
                ro = T.simd.vcvt(rotated, "float32", m16, part=1)
                se = T.simd.vcvt(phase, "float32", m16, part=0)
                so = T.simd.vcvt(phase, "float32", m16, part=1)
                base_even = T.simd.vmul(xe, ce, m32)
                base_odd = T.simd.vmul(xo, co, m32)
                cross_even = T.simd.vmul(re, se, m32)
                cross_odd = T.simd.vmul(ro, so, m32)
                plus_even = T.simd.vadd(base_even, cross_even, m32)
                plus_odd = T.simd.vadd(base_odd, cross_odd, m32)
                minus_even = T.simd.vsub(base_even, cross_even, m32)
                minus_odd = T.simd.vsub(base_odd, cross_odd, m32)
                if backward:
                    ye = T.simd.vsel(plus_even, minus_even, first_even)
                    yo = T.simd.vsel(plus_odd, minus_odd, first_odd)
                else:
                    ye = T.simd.vsel(minus_even, plus_even, first_even)
                    yo = T.simd.vsel(minus_odd, plus_odd, first_odd)
                # Copied register merge from examples/ascend/flash_attention/core.py:
                # EVEN/ODD conversions zero the complementary BF16 positions.
                out_even = T.simd.vcvt(ye, "bfloat16", m32, sat=False, part=0)
                out_odd = T.simd.vcvt(yo, "bfloat16", m32, sat=False, part=1)
                bits = T.simd.vor(T.reinterpret(out_even, "uint16x128"), T.reinterpret(out_odd, "uint16x128"), m16)
                T.simd.vsts(y[row, 0], T.reinterpret(bits, "bfloat16x128"), m16, dist="NORM_B16")

    @T.macro
    def process(X, Cos, Sin, Y, x, c, s, y, task, valid, sequence_tail):
        b = task // H if sequence_tail else task // (H * max(1, tiles))
        h = task % H if sequence_tail else task // max(1, tiles) % H
        begin = tiles * rows if sequence_tail else task % max(1, tiles) * rows
        cb = b if Cb != 1 else 0
        T.copy(X[b, h, begin : begin + valid, :], x[:valid, :D], l2_cache_ctrl="NOTALLOC_KEEP")
        T.copy(Cos[cb, begin : begin + valid, :], c[:valid, :D], l2_cache_ctrl="NORMAL_PERS")
        T.copy(Sin[cb, begin : begin + valid, :], s[:valid, :D], l2_cache_ctrl="NORMAL_PERS")
        compute(x, c, s, y, valid)
        T.copy(y[:valid, :D], Y[b, h, begin : begin + valid, :])

    @T.prim_func
    def main(
        X: T.Tensor((B, H, S, D), "bfloat16"),
        Cos: T.Tensor((Cb, S, D), "bfloat16"),
        Sin: T.Tensor((Cb, S, D), "bfloat16"),
        Y: T.Tensor((B, H, S, D), "bfloat16"),
    ):
        with T.Kernel(cores) as core:
            x = T.alloc_shared((rows, pitch), "bfloat16")
            c = T.alloc_shared((rows, pitch), "bfloat16")
            s = T.alloc_shared((rows, pitch), "bfloat16")
            y = T.alloc_shared((rows, pitch), "bfloat16")
            T.annotate_buffer_versions({x: stages, c: stages, s: stages, y: stages})
            for wave in T.Pipelined(full_waves, num_stages=stages):
                process(X, Cos, Sin, Y, x, c, s, y, wave * cores + core, rows, False)
            if tasks % cores != 0 or S % rows != 0:
                xt = T.alloc_shared((rows, pitch), "bfloat16")
                ct = T.alloc_shared((rows, pitch), "bfloat16")
                st = T.alloc_shared((rows, pitch), "bfloat16")
                yt = T.alloc_shared((rows, pitch), "bfloat16")
                T.annotate_buffer_versions({xt: 1, ct: 1, st: 1, yt: 1})
                if tasks % cores != 0:
                    task = full_waves * cores + core
                    if task < tasks:
                        process(X, Cos, Sin, Y, xt, ct, st, yt, task, rows, False)
                if S % rows != 0:
                    for wave in T.serial(tilelang.cdiv(B * H, cores)):
                        task = wave * cores + core
                        if task < B * H:
                            process(X, Cos, Sin, Y, xt, ct, st, yt, task, S % rows, True)

    return main


def _dense_rope_kernel(B, H, S, D, Cb, dtype, phase_dtype, backward, num_cores):
    """Grouped outer task pipeline with complete existing public fallbacks."""
    if min(B, H, S) > 0 and dtype == phase_dtype == "bfloat16" and D in (16, 56):
        return _narrow_register_kernel(B, H, S, D, Cb, backward, num_cores, 2)
    rows = 16 if dtype == "float32" else 32
    item_bytes = 4 if dtype == "float32" else 2
    phase_bytes = 4 if phase_dtype == "float32" else 2
    half = D // 2
    aligned = half * item_bytes % 32 == 0 and half * phase_bytes % 32 == 0
    eligible = min(B, H, S) > 0 and H >= 4 and S >= rows and D <= 128 and aligned
    if eligible:
        pitch = tilelang.cdiv(max(D, half + tilelang.cdiv(half, 64) * 64), 64) * 64
        tasks = B * (H // 4) * (S // rows)
        cores = min(num_cores, tasks)
        tail = tasks % cores != 0 or H % 4 != 0 or S % rows != 0
        main_bytes = rows * pitch * (16 * item_bytes + 4 * phase_bytes)
        tail_bytes = rows * pitch * (2 * item_bytes + 2 * phase_bytes) if tail else 0
        eligible = main_bytes + tail_bytes <= 248 * 1024
    if eligible:
        return _grouped_phase_kernel(B, H, S, D, Cb, dtype, phase_dtype, backward, num_cores, 2)
    return _ungrouped_kernel(B, H, S, D, Cb, dtype, phase_dtype, backward, num_cores)


def _grouped_footprint(B, H, S, D, dtype, phase_dtype, num_cores, group, rows):
    """Complete SIMD UB allocation including physical row padding and tails."""
    item_bytes = 4 if dtype == "float32" else 2
    phase_bytes = 4 if phase_dtype == "float32" else 2
    half = D // 2
    pitch = tilelang.cdiv(max(D, half + tilelang.cdiv(half, 64) * 64), 64) * 64
    tasks = B * (H // group) * (S // rows)
    cores = min(num_cores, tasks)
    tail = tasks > 0 and (tasks % cores != 0 or H % group != 0 or S % rows != 0)
    main_bytes = rows * pitch * (4 * group * item_bytes + 4 * phase_bytes)
    tail_bytes = rows * pitch * (2 * item_bytes + 2 * phase_bytes) if tail else 0
    return tasks, main_bytes + tail_bytes


def rope_launch_route(B, H, S, D, dtype, phase_dtype, num_cores, bsnd=False):
    """Select by alignment, storage layout and complete allocation, never mode."""
    item_bytes = 4 if dtype == "float32" else 2
    phase_bytes = 4 if phase_dtype == "float32" else 2
    half = D // 2
    aligned = half * item_bytes % 32 == 0 and half * phase_bytes % 32 == 0
    if min(B, H, S) <= 0 or dtype not in ("float16", "bfloat16") or not aligned:
        return "dense"
    if D > 128:
        group, rows = (8 if H >= 8 else 4), 8
        if H >= 4:
            tasks, allocation = _grouped_footprint(B, H, S, D, dtype, phase_dtype, num_cores, group, rows)
            if tasks >= num_cores and allocation <= 248 * 1024:
                return "wide"
        return "dense"
    if bsnd and H >= 4:
        tasks, allocation = _grouped_footprint(B, H, S, D, dtype, phase_dtype, num_cores, 4, 32)
        if tasks > 0 and allocation <= 248 * 1024:
            return "narrow_stride"
    return "dense"


def rope_kernel(B, H, S, D, Cb, dtype, phase_dtype, backward, num_cores, strides=None):
    """Use exact dense recovery or shared narrow/wide grouped SIMD kernels."""
    route = rope_launch_route(B, H, S, D, dtype, phase_dtype, num_cores, strides is not None)
    if route == "wide":
        group = 8 if H >= 8 else 4
        return _grouped_phase_kernel(B, H, S, D, Cb, dtype, phase_dtype, backward, num_cores, 2, strides, group, 8)
    if route == "narrow_stride":
        return _grouped_phase_kernel(B, H, S, D, Cb, dtype, phase_dtype, backward, num_cores, 2, strides, 4, 32)
    return _dense_rope_kernel(B, H, S, D, Cb, dtype, phase_dtype, backward, num_cores)
