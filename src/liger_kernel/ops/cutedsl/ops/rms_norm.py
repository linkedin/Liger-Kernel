"""
CuTe DSL (NVIDIA CUTLASS Python DSL) RMSNorm.

This is a genuine ``cutlass.cute`` implementation — it imports ``cutlass.cute``,
emits ``@cute.kernel`` device kernels, compiles them host-side with
``cute.compile`` (cached), marshals torch tensors via DLPack, and launches on
torch's current CUDA stream. It contains **no** ``cuda.tile`` / cuTile code and
no Triton/PyTorch fallback dressed up as a kernel.

Behavior is a drop-in match for the default Triton implementation
(``liger_kernel.ops.rms_norm``): same public ``rms_norm_forward`` /
``rms_norm_backward`` / ``LigerRMSNormFunction`` signatures, same math, and the
same ``llama`` / ``gemma`` / ``none`` casting modes, offset, elementwise-affine
toggle, in-place backward, and DTensor gathering.

    y_i  = (x_i / RMS) * (offset + w_i),   RMS = sqrt(mean(x_i^2) + eps)
    dx   = rstd * [ m - (1/N) * rstd^2 * (m . x) * x ],   m = dy * (w + offset)
    dw   = sum_rows dy * (x * rstd)

The kernels are written for correctness and clarity first (fp32 accumulation, one
CTA per row for the reductions). The weight gradient is reduced in parallel across
row strips (one strip per SM, partials summed on the host — the same strategy as the
Triton backward) rather than a single thread walking every row. They keep the hidden
dimension fully dynamic, so a single compiled kernel serves every shape — regular or
irregular. Further performance tuning (fused dx/dw pass, vectorized/pipelined loads)
is left to the ``liger-kernel-perf`` workflow on real Hopper/Blackwell hardware.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch

from cutlass import Float32
from cutlass import Int32
from cutlass import const_expr

from liger_kernel.ops.cutedsl.ops.utils import to_cute_tensor

# Casting-mode ids — identical values to the Triton kernel so an ``int`` casting
# mode round-trips between the two backends unchanged.
_CASTING_MODE_NONE = -1
_CASTING_MODE_LLAMA = 0
_CASTING_MODE_GEMMA = 1

_str_to_casting_mode = {
    "llama": _CASTING_MODE_LLAMA,
    "gemma": _CASTING_MODE_GEMMA,
    "none": _CASTING_MODE_NONE,
}

# One CTA cooperates on a single row. 8 warps (256 threads) is a solid default for
# the memory-bound per-row reduction; the grid (= number of rows) supplies the
# parallelism, so we don't need a huge block.
_NUM_WARPS = 8
_THREADS = 32 * _NUM_WARPS

# Compiled-kernel cache keyed on everything the kernels bake (dtypes + constexpr
# flags). Without it every call would re-run ``cute.compile`` (tens of ms).
_compile_cache = {}

# Cache the CUstream wrapper keyed on torch's raw stream handle so we don't rebuild
# the cuda.CUstream object every launch (same trick as the cutedsl CE kernel).
_stream_cache = {}


def _cute_stream():
    raw = torch.cuda.current_stream().cuda_stream
    s = _stream_cache.get(raw)
    if s is None:
        s = cuda.CUstream(raw)
        _stream_cache[raw] = s
    return s


def _maybe_gather_dtensor(t):
    """Gather a DTensor to a full local tensor; pass plain tensors through unchanged.

    Accessing ``torch.distributed.tensor.DTensor`` directly assumes that submodule is
    eagerly imported, which is not guaranteed on every torch build — some raise
    ``AttributeError: module 'torch.distributed' has no attribute 'tensor'`` until it
    is imported explicitly. Import it defensively so RMSNorm works whether or not
    DTensor is available (mirrors the Triton op's TP-gather without the hard attribute
    access).
    """
    try:
        from torch.distributed.tensor import DTensor
    except Exception:
        return t
    if isinstance(t, DTensor):
        return t.full_tensor()
    return t


# =============================================================================
# Device-side helpers
# =============================================================================
@cute.jit
def _warp_reduce_sum(val: Float32) -> Float32:
    """Full-warp sum via butterfly shuffle; every lane ends with the warp total."""
    for i in cutlass.range_constexpr(5):  # log2(32) = 5 steps
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


# =============================================================================
# Device kernels
# =============================================================================
@cute.kernel
def _rms_norm_fwd_kernel(
    mX: cute.Tensor,  # (n_rows, n_cols) input
    mW: cute.Tensor,  # (n_cols,) weight (read only if ELEMENTWISE_AFFINE; else a dummy)
    mY: cute.Tensor,  # (n_rows, n_cols) output
    mRSTD: cute.Tensor,  # (n_rows,) fp32 reciprocal-RMS cache (consumed by backward)
    eps: Float32,
    offset: Float32,
    CASTING_MODE: cutlass.Constexpr,  # -1 none / 0 llama / 1 gemma
    ELEMENTWISE_AFFINE: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    lane = tid % 32
    warp = tid // 32
    row, _, _ = cute.arch.block_idx()

    # Cross-warp reduction scratch: one partial per warp.
    smem = cutlass.utils.SmemAllocator()
    sm_red = smem.allocate_tensor(Float32, cute.make_layout(_NUM_WARPS), byte_alignment=4)

    gX = mX[row, None]  # 1D (n_cols,) view of this row
    gY = mY[row, None]
    n_cols = gX.shape[0]
    num_col_tiles = (n_cols + _THREADS - 1) // _THREADS

    # --- pass 1: sum of squares (fp32 accumulation, matches llama/gemma; slightly
    # more accurate than Triton's in-dtype "none" path, covered by test tolerances).
    partial = Float32(0.0)
    for ct in cutlass.range(0, num_col_tiles):
        c = ct * _THREADS + tid
        if c < n_cols:
            xf = gX[c].to(Float32)
            partial = partial + xf * xf

    # warp reduce -> cross-warp reduce; every thread ends with the row's total.
    partial = _warp_reduce_sum(partial)
    if lane == 0:
        sm_red[warp] = partial
    cute.arch.barrier()
    total = Float32(0.0)
    for w in cutlass.range_constexpr(_NUM_WARPS):
        total = total + sm_red[w]

    mean_square = total / Float32(n_cols)
    rstd = cute.math.rsqrt(mean_square + eps)
    if tid == 0:
        mRSTD[row] = rstd.to(mRSTD.element_type)

    # --- pass 2: normalize + affine. Re-load x (cheap vs. carrying a dynamic-length
    # register tile); each thread writes only its own columns.
    for ct in cutlass.range(0, num_col_tiles):
        c = ct * _THREADS + tid
        if c < n_cols:
            xhat = gX[c].to(Float32) * rstd
            # llama casts the normalized value back to the input dtype *before* the
            # affine multiply (Triton parity); gemma/none stay in fp32 here.
            if const_expr(CASTING_MODE == _CASTING_MODE_LLAMA):
                xhat = xhat.to(mX.element_type).to(Float32)
            if const_expr(ELEMENTWISE_AFFINE):
                wv = mW[c].to(Float32)
                y = xhat * (offset + wv)
            else:
                y = xhat
            gY[c] = y.to(gY.element_type)


@cute.kernel
def _rms_norm_bwd_dx_kernel(
    mdY: cute.Tensor,  # (n_rows, n_cols) upstream grad
    mX: cute.Tensor,  # (n_rows, n_cols) saved input
    mW: cute.Tensor,  # (n_cols,) weight (read only if ELEMENTWISE_AFFINE)
    mRSTD: cute.Tensor,  # (n_rows,) fp32 reciprocal-RMS cache
    mdX: cute.Tensor,  # (n_rows, n_cols) input grad out (may alias mdY for in-place)
    offset: Float32,
    CASTING_MODE: cutlass.Constexpr,
    ELEMENTWISE_AFFINE: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    lane = tid % 32
    warp = tid // 32
    row, _, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    sm_red = smem.allocate_tensor(Float32, cute.make_layout(_NUM_WARPS), byte_alignment=4)

    gdY = mdY[row, None]
    gX = mX[row, None]
    gdX = mdX[row, None]
    n_cols = gX.shape[0]
    num_col_tiles = (n_cols + _THREADS - 1) // _THREADS
    rstd = mRSTD[row].to(Float32)

    # --- pass 1: dot(m, x) with m = dy * (w + offset)  [m = dy when non-affine].
    dot = Float32(0.0)
    for ct in cutlass.range(0, num_col_tiles):
        c = ct * _THREADS + tid
        if c < n_cols:
            xf = gX[c].to(Float32)
            dyf = gdY[c].to(Float32)
            if const_expr(ELEMENTWISE_AFFINE):
                mk = dyf * (mW[c].to(Float32) + offset)
            else:
                mk = dyf
            dot = dot + mk * xf

    dot = _warp_reduce_sum(dot)
    if lane == 0:
        sm_red[warp] = dot
    cute.arch.barrier()
    dot_total = Float32(0.0)
    for w in cutlass.range_constexpr(_NUM_WARPS):
        dot_total = dot_total + sm_red[w]

    # dx = rstd * (m - (1/N) * rstd^2 * dot * x)
    coef = (Float32(0.0) - rstd * rstd * dot_total) / Float32(n_cols)

    # --- pass 2: write dx. Reading dy/x again here (rather than caching a
    # dynamic-length tile) keeps the kernel shape-generic. In-place is safe: each
    # thread reads column c then writes the same column, and pass 1 (which reads
    # every column) is fully fenced from these writes by the barrier above.
    for ct in cutlass.range(0, num_col_tiles):
        c = ct * _THREADS + tid
        if c < n_cols:
            xf = gX[c].to(Float32)
            dyf = gdY[c].to(Float32)
            if const_expr(ELEMENTWISE_AFFINE):
                mk = dyf * (mW[c].to(Float32) + offset)
            else:
                mk = dyf
            dxk = rstd * (mk + coef * xf)
            gdX[c] = dxk.to(gdX.element_type)


@cute.kernel
def _rms_norm_bwd_dw_kernel(
    mdY: cute.Tensor,  # (n_rows, n_cols)
    mX: cute.Tensor,  # (n_rows, n_cols)
    mRSTD: cute.Tensor,  # (n_rows,) fp32
    mdW: cute.Tensor,  # (num_strips, n_cols) fp32 partial weight grads (one row per strip)
    rows_per_strip: Int32,  # ceil(n_rows / num_strips)
    CASTING_MODE: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    col_block, strip, _ = cute.arch.block_idx()
    c = col_block * _THREADS + tid

    n_rows = mX.shape[0]
    n_cols = mX.shape[1]

    # 2D grid: (column blocks) x (row strips). Each program reduces dW over just its
    # strip of rows for its columns, so the row reduction runs across the whole GPU
    # instead of one thread walking all n_rows serially. The num_strips per-strip
    # partials are summed on the host afterward (mirrors the Triton sm_count partials).
    # Consecutive threads own consecutive columns, so each row's loads stay coalesced.
    row_start = strip * rows_per_strip
    acc = Float32(0.0)
    for i in cutlass.range(0, rows_per_strip):
        r = row_start + i
        rstd = Float32(0.0)
        xf = Float32(0.0)
        dyf = Float32(0.0)
        if r < n_rows:
            rstd = mRSTD[r].to(Float32)
            if c < n_cols:
                xf = mX[r, None][c].to(Float32)
                dyf = mdY[r, None][c].to(Float32)
        xhat = xf * rstd
        # llama rounds x*rstd to the input dtype before accumulating (Triton parity).
        if const_expr(CASTING_MODE == _CASTING_MODE_LLAMA):
            xhat = xhat.to(mX.element_type).to(Float32)
        acc = acc + dyf * xhat
    if c < n_cols:
        mdW[strip, None][c] = acc.to(mdW.element_type)


# =============================================================================
# Host launch (compiled once per dtype/flag combo, then cached)
# =============================================================================
@cute.jit
def _rms_norm_fwd_host(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mY: cute.Tensor,
    mRSTD: cute.Tensor,
    eps: Float32,
    offset: Float32,
    CASTING_MODE: cutlass.Constexpr,
    ELEMENTWISE_AFFINE: cutlass.Constexpr,
    stream: cuda.CUstream = None,
):
    n_rows = mX.shape[0]
    smem_bytes = ((_NUM_WARPS * 4 + 15) // 16) * 16
    _rms_norm_fwd_kernel(mX, mW, mY, mRSTD, eps, offset, CASTING_MODE, ELEMENTWISE_AFFINE).launch(
        grid=[n_rows, 1, 1],
        block=[_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


@cute.jit
def _rms_norm_bwd_dx_host(
    mdY: cute.Tensor,
    mX: cute.Tensor,
    mW: cute.Tensor,
    mRSTD: cute.Tensor,
    mdX: cute.Tensor,
    offset: Float32,
    CASTING_MODE: cutlass.Constexpr,
    ELEMENTWISE_AFFINE: cutlass.Constexpr,
    stream: cuda.CUstream = None,
):
    n_rows = mX.shape[0]
    smem_bytes = ((_NUM_WARPS * 4 + 15) // 16) * 16
    _rms_norm_bwd_dx_kernel(mdY, mX, mW, mRSTD, mdX, offset, CASTING_MODE, ELEMENTWISE_AFFINE).launch(
        grid=[n_rows, 1, 1],
        block=[_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


@cute.jit
def _rms_norm_bwd_dw_host(
    mdY: cute.Tensor,
    mX: cute.Tensor,
    mRSTD: cute.Tensor,
    mdW: cute.Tensor,
    rows_per_strip: Int32,
    CASTING_MODE: cutlass.Constexpr,
    stream: cuda.CUstream = None,
):
    n_cols = mX.shape[1]
    num_strips = mdW.shape[0]
    num_col_blocks = (n_cols + _THREADS - 1) // _THREADS
    _rms_norm_bwd_dw_kernel(mdY, mX, mRSTD, mdW, rows_per_strip, CASTING_MODE).launch(
        grid=[num_col_blocks, num_strips, 1],
        block=[_THREADS, 1, 1],
        smem=0,
        stream=stream,
    )


def _launch_fwd(X, W, Y, RSTD, eps, offset, casting_mode, elementwise_affine):
    stream = _cute_stream()
    # Scalar (non-vectorized) access, so element-size alignment is all we assume — this
    # keeps the kernel correct for unaligned contiguous slices and irregular hidden dims.
    x_ct = to_cute_tensor(X, assumed_align=X.element_size())
    y_ct = to_cute_tensor(Y, assumed_align=Y.element_size())
    rstd_ct = to_cute_tensor(RSTD, assumed_align=4)  # fp32
    # Non-affine: reuse the fp32 RSTD handle as a dummy — the kernel never reads it.
    w_ct = to_cute_tensor(W, assumed_align=W.element_size()) if elementwise_affine else rstd_ct

    # Key on every dtype the kernel bakes: X (also Y), and W when affine (mW.element_type
    # is a compile-time specialization). Missing W.dtype would let a bf16-activations /
    # fp32-weight call reuse a kernel baked for a different weight width — see the same
    # guard in cross_entropy.py's compile key.
    key = ("fwd", X.dtype, W.dtype if elementwise_affine else None, casting_mode, elementwise_affine)
    if key not in _compile_cache:
        _compile_cache[key] = cute.compile(
            _rms_norm_fwd_host,
            x_ct,
            w_ct,
            y_ct,
            rstd_ct,
            float(eps),
            float(offset),
            casting_mode,
            elementwise_affine,
            stream,
        )
    _compile_cache[key](x_ct, w_ct, y_ct, rstd_ct, float(eps), float(offset), stream)


def _launch_bwd_dx(dY, X, W, RSTD, dX, offset, casting_mode, elementwise_affine):
    stream = _cute_stream()
    dy_ct = to_cute_tensor(dY, assumed_align=dY.element_size())
    x_ct = to_cute_tensor(X, assumed_align=X.element_size())
    rstd_ct = to_cute_tensor(RSTD, assumed_align=4)
    dx_ct = to_cute_tensor(dX, assumed_align=dX.element_size())
    w_ct = to_cute_tensor(W, assumed_align=W.element_size()) if elementwise_affine else rstd_ct

    # Key on every baked dtype: dY, X (also dX == dY.dtype), and W when affine.
    key = ("bwd_dx", X.dtype, dY.dtype, W.dtype if elementwise_affine else None, casting_mode, elementwise_affine)
    if key not in _compile_cache:
        _compile_cache[key] = cute.compile(
            _rms_norm_bwd_dx_host,
            dy_ct,
            x_ct,
            w_ct,
            rstd_ct,
            dx_ct,
            float(offset),
            casting_mode,
            elementwise_affine,
            stream,
        )
    _compile_cache[key](dy_ct, x_ct, w_ct, rstd_ct, dx_ct, float(offset), stream)


def _launch_bwd_dw(dY, X, RSTD, dW_partial, rows_per_strip, casting_mode):
    stream = _cute_stream()
    dy_ct = to_cute_tensor(dY, assumed_align=dY.element_size())
    x_ct = to_cute_tensor(X, assumed_align=X.element_size())
    rstd_ct = to_cute_tensor(RSTD, assumed_align=4)
    dw_ct = to_cute_tensor(dW_partial, assumed_align=4)  # fp32 (num_strips, n_cols)

    # Key on every baked dtype: dY and X (mdW is always fp32). The llama cast bakes
    # mX.element_type; the loads bake mdY.element_type. rows_per_strip is a runtime
    # arg (not baked), so one compiled kernel serves every shape.
    key = ("bwd_dw", X.dtype, dY.dtype, casting_mode)
    if key not in _compile_cache:
        _compile_cache[key] = cute.compile(
            _rms_norm_bwd_dw_host, dy_ct, x_ct, rstd_ct, dw_ct, int(rows_per_strip), casting_mode, stream
        )
    _compile_cache[key](dy_ct, x_ct, rstd_ct, dw_ct, int(rows_per_strip), stream)


# =============================================================================
# Public host API (matches liger_kernel.ops.rms_norm)
# =============================================================================
def rms_norm_forward(X, W, eps, offset, casting_mode, row_mode):
    """CuTe DSL RMSNorm forward.

    Returns ``(Y, X_2d, RSTD, BLOCK_SIZE, num_warps, casting_mode)`` — the
    ``BLOCK_SIZE`` / ``num_warps`` slots are kept for signature parity with the
    Triton op (this kernel doesn't need them) and are passed through to backward.
    ``row_mode`` is accepted for parity and has no effect here.
    """
    if not isinstance(casting_mode, int):
        assert casting_mode in _str_to_casting_mode, f"Invalid casting mode: {casting_mode}"
        casting_mode = _str_to_casting_mode[casting_mode]
    else:
        assert casting_mode in _str_to_casting_mode.values(), f"Invalid casting mode: {casting_mode}"

    shape = X.shape
    dim = shape[-1]
    # Contiguous before view (mirrors the Triton op's @ensure_contiguous): view(-1, dim)
    # requires a contiguous tensor, and the kernel indexes rows with unit column stride.
    X = X.contiguous().view(-1, dim)
    n_rows, n_cols = X.shape

    elementwise_affine = W is not None
    if elementwise_affine:
        assert X.shape[1] == W.shape[0], (
            "Incompatible hidden size dimension between tensor1.shape[1] and tensor2.shape[0]"
        )
        W = W.contiguous()

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    # RSTD is an internal cache consumed only by our own backward, so we store it in
    # fp32 unconditionally (simpler + more accurate than mirroring Triton's per-mode
    # RSTD dtype; it is never compared against the Triton reference).
    RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)

    _launch_fwd(X, W, Y, RSTD, eps, offset, casting_mode, elementwise_affine)

    return Y.view(*shape), X, RSTD, None, None, casting_mode


def rms_norm_backward(dY, X, W, RSTD, offset, casting_mode, BLOCK_SIZE, num_warps, in_place, row_mode):
    """CuTe DSL RMSNorm backward. Returns ``(dX, dW)`` (``dW`` is ``None`` when
    non-affine). ``BLOCK_SIZE`` / ``num_warps`` / ``row_mode`` are accepted for
    signature parity with the Triton op and are unused."""
    shape = dY.shape
    dim = shape[-1]
    dY = dY.contiguous().view(-1, dim)
    n_rows, n_cols = dY.shape

    elementwise_affine = W is not None

    if in_place is True:
        dX = dY
    else:
        dX = torch.empty_like(dY)

    # dW must be computed BEFORE dX: when in_place=True, dX aliases dY and the dX kernel
    # overwrites dY in place, but dW = sum_rows(dY * x * rstd) needs the original dY.
    # Neither gradient depends on the other, so ordering dW first is correct in both the
    # in-place and out-of-place cases.
    if elementwise_affine:
        # Parallelize the dW reduction across row strips: one strip per SM (capped at
        # n_rows), each producing a partial dW row, then sum the partials on the host —
        # mirrors the Triton backward's sm_count partials + `_dW.sum(dim=0)`.
        if X.device.type == "cuda":
            sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
        else:
            sm_count = 1
        num_strips = max(1, min(sm_count, n_rows))
        rows_per_strip = (n_rows + num_strips - 1) // num_strips
        dW_partial = torch.empty((num_strips, n_cols), dtype=torch.float32, device=W.device)
        _launch_bwd_dw(dY, X, RSTD, dW_partial, rows_per_strip, casting_mode)
        dW = dW_partial.sum(dim=0).to(W.dtype)
    else:
        dW = None

    _launch_bwd_dx(dY, X, W, RSTD, dX, offset, casting_mode, elementwise_affine)

    return dX.view(*shape), dW


class LigerRMSNormFunction(torch.autograd.Function):
    """
    CuTe DSL autograd wrapper for RMSNorm.

    Signature-compatible with ``liger_kernel.ops.rms_norm.LigerRMSNormFunction``:
    ``forward(X, W, eps, offset=0.0, casting_mode="llama", in_place=True, row_mode=None)``.
    See that class for the semantics of ``offset``, ``casting_mode`` and ``in_place``.
    """

    @staticmethod
    def forward(ctx, X, W, eps, offset=0.0, casting_mode="llama", in_place=True, row_mode=None):
        """
        X: (B, T, H) or (BxT, H)
        W: (H,)
        """
        # Gather a TP-sharded input to a local tensor before normalizing (safe when
        # torch.distributed.tensor isn't importable on this build).
        X = _maybe_gather_dtensor(X)

        Y, X, RSTD, BLOCK_SIZE, num_warps, casting_mode = rms_norm_forward(X, W, eps, offset, casting_mode, row_mode)
        ctx.offset = offset
        ctx.casting_mode = casting_mode
        ctx.in_place = in_place
        ctx.row_mode = row_mode
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.elementwise_affine = W is not None
        if W is not None:
            ctx.save_for_backward(X, W, RSTD)
        else:
            ctx.save_for_backward(X, RSTD)
        return Y

    @staticmethod
    def backward(ctx, dY):
        """
        Y: (B, T, H) or (BxT, H)
        """
        if ctx.elementwise_affine:
            X, W, RSTD = ctx.saved_tensors
        else:
            X, RSTD = ctx.saved_tensors
            W = None

        dY = _maybe_gather_dtensor(dY)

        dX, dW = rms_norm_backward(
            dY, X, W, RSTD, ctx.offset, ctx.casting_mode, ctx.BLOCK_SIZE, ctx.num_warps, ctx.in_place, ctx.row_mode
        )
        return dX, dW, None, None, None, None, None
