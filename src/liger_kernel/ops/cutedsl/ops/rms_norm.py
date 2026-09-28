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

The scalar forward and split backward kernels remain correctness fallbacks for
irregular, unaligned, and unsupported shapes. Aligned widths through 8192 use
vector paths: forward keeps X in registers across its reduction and output phase;
affine backward mirrors Triton's execution shape with one CTA per SM, contiguous
persistent row ranges, width-dependent warp counts, and register-resident X/dY
fragments reused for both dX and dW. The fast path is selected host-side only when
its alignment assumptions are true. Its compilation key therefore includes the
exact width, vector width, and launch geometry; the scalar path remains
shape-generic.
"""

import json
import os
import time

from collections import OrderedDict

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch

from cutlass import Float32
from cutlass import Int32
from cutlass import const_expr

from liger_kernel.ops.cutedsl.ops.rms_norm_fastpath import backward_warp_count
from liger_kernel.ops.cutedsl.ops.rms_norm_fastpath import fast_path_vector_width
from liger_kernel.ops.cutedsl.ops.rms_norm_fastpath import fwd_warp_count
from liger_kernel.ops.cutedsl.ops.utils import to_cute_tensor

# ---------------------------------------------------------------------------
# Tuning / debug env vars are read ONCE at import. These are launch-time knobs
# (set via `env VAR=...` before the process starts); nothing mutates them at
# runtime, so re-reading os.environ on every kernel launch was pure host
# overhead. cProfile showed ~12 os.environ lookups per fwd+bwd pass (~1us each),
# which dominates the launch path at the small shapes where this op is
# host-enqueue-bound rather than GPU-bound. Hoisting removes that per-call cost
# from every launch (forward and backward, fresh and cached tensors alike).
# ---------------------------------------------------------------------------
_DEBUG = bool(os.environ.get("LIGER_RMS_DEBUG"))
_COMPILE_BUCKET = int(os.environ.get("LIGER_RMS_COMPILE_BUCKET") or 0) or None
_RELOAD_POLICY = os.environ.get("LIGER_RMS_RELOAD_POLICY", "auto")
_FORCE_NO_FAST = bool(int(os.environ.get("LIGER_RMS_FORCE_NO_FAST") or 0))
_FORCE_SPLIT_BWD = bool(int(os.environ.get("LIGER_RMS_FORCE_SPLIT_BWD") or 0))
_AUTOTUNE_FILE = os.environ.get("LIGER_RMS_AUTOTUNE_FILE") or None
_FUSED_STRIP_MULT = int(os.environ.get("LIGER_RMS_FUSED_STRIP_MULT") or 0)
try:
    _BACKWARD_WARPS = int(os.environ.get("LIGER_RMS_BACKWARD_WARPS") or 0) or None
except Exception:
    _BACKWARD_WARPS = None


# Lightweight debug logger controlled by env var LIGER_RMS_DEBUG
def _rms_debug(msg):
    if _DEBUG:
        try:
            print(f"[RMS_DEBUG] {time.time():.6f} {msg}", flush=True)
        except Exception:
            pass


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

# The vector forward picks its warp count from the hidden width (see
# fwd_warp_count): wide rows use 8 warps so each thread holds fewer register-resident
# vector tiles, which lifts occupancy and hides DRAM-load latency on this memory-bound
# kernel; narrow rows keep 4 warps so threads stay busy and the reduction stays cheap.
# These constants are the narrow-row default / scalar-path fallback and the eligibility
# probe's assumed block size; the vector launch overrides them per call.
_FAST_NUM_WARPS = 4
_FAST_THREADS = 32 * _FAST_NUM_WARPS
_FAST_MAX_COLS = 8192

# Fused backward strip-count auto-tuning. The register-resident fused kernel runs
# one block per strip. Its register pressure (~122 regs/thread) caps the SM at
# ~2 resident blocks, so once each strip would otherwise process this many rows
# per SM serially (latency-bound regime), we launch two strips per SM instead of
# one. The threshold sits inside the B200-measured 13.8-27.7 rows/SM crossover
# window; below it, one strip per SM is as fast or faster.
_FUSED_DOUBLE_STRIP_ROWS_PER_SM = 16

# Compiled-kernel cache keyed on everything the kernels bake (dtypes + constexpr
# flags). Without it every call would re-run ``cute.compile`` (tens of ms).
_compile_cache = {}

# Cache the CUstream wrapper keyed on torch's raw stream handle so we don't rebuild
# the cuda.CUstream object every launch (same trick as the cutedsl CE kernel).
_stream_cache = {}

_tensor_cache: OrderedDict = OrderedDict()

# Cap on the number of marshaled cute-tensor wrappers kept alive. Keying on
# data_ptr means a freshly-allocated output (Y/RSTD/dX from torch.empty*) gets
# cached too, and the cute wrapper holds a DLPack reference that pins the storage.
# An UNBOUNDED cache therefore stops the caching allocator from recycling that
# address -- every subsequent call gets a brand-new address (a cudaMalloc storm;
# this was the ~5x "in_place=False" backward cliff). A bounded FIFO caps how many
# buffers can be pinned at once: stable tensors (weights, and any buffer whose
# address the allocator recycles) stay cached and marshal in ~0.4us instead of
# ~4us, while genuinely fresh addresses are evicted after _TENSOR_CACHE_CAP inserts
# so the allocator can reclaim them. Eviction only drops our redundant cute wrapper
# -- the torch tensor's own reference plus CUDA stream ordering govern the real
# storage lifetime, so an in-flight kernel is never affected.
#
# Cap sizing: the fused backward's working set is ~5-8 distinct handles per pass
# (dY, X, RSTD, dX/dW_partial, W). Cap must be >= that or the backward evicts a
# still-hot input mid-pass and loses the speed win (measured: cap=8 keeps bwd at
# 0.82x vs Triton, cap=4 regresses to 1.06x). But every extra slot also pins one
# more fresh output buffer, inflating peak memory (cap=16 -> ~2.1x Triton's peak,
# cap=8 -> ~1.8x). 8 is the knee: smallest cap that still holds the backward
# working set, so it keeps the full speed win while pinning the fewest buffers.
#
# Overridable via LIGER_RMS_TENSOR_CACHE_CAP for workloads that want to trade the
# other way: raise it (e.g. 16) for max launch speed, lower it (e.g. 4) to shave
# peak memory at the cost of the backward's cache-hit win.
try:
    _TENSOR_CACHE_CAP = int(os.environ.get("LIGER_RMS_TENSOR_CACHE_CAP") or 8)
except Exception:
    _TENSOR_CACHE_CAP = 8


def _to_cute_cached(t, assumed_align=16):
    key = (t.data_ptr(), t.dtype, tuple(t.shape), tuple(t.stride()), assumed_align)
    cached = _tensor_cache.get(key)
    if cached is not None:
        return cached
    result = to_cute_tensor(t, assumed_align=assumed_align)
    _tensor_cache[key] = result
    if len(_tensor_cache) > _TENSOR_CACHE_CAP:
        _tensor_cache.popitem(last=False)
    return result


_dw_partial_pool: dict = {}


def _get_dw_partial_buf(num_strips, n_cols, device):
    import torch

    dev_key = device.index if device.type == "cuda" else str(device)
    key = (dev_key, num_strips, n_cols)
    buf = _dw_partial_pool.get(key)
    if buf is None:
        buf = torch.empty((num_strips, n_cols), dtype=torch.float32, device=device)
        _dw_partial_pool[key] = buf
    return buf


def _cute_stream():
    raw = torch.cuda.current_stream().cuda_stream
    s = _stream_cache.get(raw)
    if s is None:
        s = cuda.CUstream(raw)
        _stream_cache[raw] = s
    return s


_sm_count_cache: dict = {}


def _get_sm_count(device):
    """multi_processor_count for a device, cached (the query is a ~1.3us driver call
    otherwise paid on every backward launch)."""
    if device.type != "cuda":
        return 1
    key = device.index if device.index is not None else torch.cuda.current_device()
    n = _sm_count_cache.get(key)
    if n is None:
        n = torch.cuda.get_device_properties(device).multi_processor_count
        _sm_count_cache[key] = n
    return n


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


@cute.jit
def _cta_reduce_sum_warp0(
    val: Float32,
    sm_warp: cute.Tensor,
    sm_result: cute.Tensor,
    lane: Int32,
    warp: Int32,
    NUM_WARPS: cutlass.Constexpr,
) -> Float32:
    """Reduce warp partials with warp 0 and broadcast one shared scalar.

    Only warp 0 reads the partials. This avoids the scalar fallback's shared-load
    loop in every thread while retaining a simple, reusable CTA
    reduction for the vector forward and backward paths.
    """
    val = _warp_reduce_sum(val)
    if lane == 0:
        sm_warp[warp] = val
    cute.arch.barrier()
    warp0_val = Float32(0.0)
    if warp == 0:
        if lane < NUM_WARPS:
            warp0_val = sm_warp[lane]
        warp0_val = _warp_reduce_sum(warp0_val)
        if lane == 0:
            sm_result[0] = warp0_val
    cute.arch.barrier()
    return sm_result[0]


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
def _rms_norm_fwd_vector_kernel(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mY: cute.Tensor,
    mRSTD: cute.Tensor,
    eps: Float32,
    offset: Float32,
    CASTING_MODE: cutlass.Constexpr,
    ELEMENTWISE_AFFINE: cutlass.Constexpr,
    N_COLS: cutlass.Constexpr,
    VEC: cutlass.Constexpr,
    NUM_VEC_TILES: cutlass.Constexpr,
    NUM_WARPS: cutlass.Constexpr,
    NUM_THREADS: cutlass.Constexpr,
):
    """Aligned one-row forward kernel with register-resident X fragments."""
    tid, _, _ = cute.arch.thread_idx()
    lane = tid % 32
    warp = tid // 32
    row, _, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    sm_warp = smem.allocate_tensor(Float32, cute.make_layout(NUM_WARPS), byte_alignment=4)
    sm_result = smem.allocate_tensor(Float32, cute.make_layout(1), byte_alignment=4)

    # Slicing a dynamic tensor loses its static alignment.  The host validates the
    # 16-byte base/row alignment before selecting this kernel, so reconstruct the
    # row views with that alignment for vector loads and stores.
    x_row = mX[row, None]
    y_row = mY[row, None]
    gX = cute.make_tensor(
        cute.make_ptr(mX.element_type, x_row.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16),
        cute.make_layout((N_COLS,)),
    )
    gY = cute.make_tensor(
        cute.make_ptr(mY.element_type, y_row.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16),
        cute.make_layout((N_COLS,)),
    )
    gXv = cute.tiled_divide(gX, (VEC,))
    gYv = cute.tiled_divide(gY, (VEC,))

    # Keep every X fragment live through normalization.  N_COLS is constexpr on the
    # fast path, so this is a compact fixed register tile (at most 16 vectors/thread).
    x_frags = cute.make_rmem_tensor((VEC, NUM_VEC_TILES), mX.element_type)
    out_frag = cute.make_rmem_tensor((VEC,), mY.element_type)
    partial = Float32(0.0)
    n_vec = N_COLS // VEC
    for ct in cutlass.range_constexpr(NUM_VEC_TILES):
        vec_idx = ct * NUM_THREADS + tid
        if vec_idx < n_vec:
            cute.autovec_copy(gXv[None, vec_idx], x_frags[None, ct])
            x_ssa = x_frags[None, ct].load().to(Float32)
            partial = partial + (x_ssa * x_ssa).reduce(cute.ReductionOp.ADD, Float32(0.0), 0)

    total = _cta_reduce_sum_warp0(partial, sm_warp, sm_result, lane, warp, NUM_WARPS)
    rstd = cute.math.rsqrt(total / Float32(N_COLS) + eps)
    if tid == 0:
        mRSTD[row] = rstd.to(mRSTD.element_type)

    # W is deliberately loaded only after the reduction, avoiding its liveness
    # across the X-square accumulation.
    if const_expr(ELEMENTWISE_AFFINE):
        gW = cute.make_tensor(
            cute.make_ptr(mW.element_type, mW.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((N_COLS,)),
        )
        gWv = cute.tiled_divide(gW, (VEC,))
        w_frag = cute.make_rmem_tensor((VEC,), mW.element_type)
    for ct in cutlass.range_constexpr(NUM_VEC_TILES):
        vec_idx = ct * NUM_THREADS + tid
        if vec_idx < n_vec:
            xhat = x_frags[None, ct].load().to(Float32) * rstd
            if const_expr(CASTING_MODE == _CASTING_MODE_LLAMA):
                xhat = xhat.to(mX.element_type).to(Float32)
            if const_expr(ELEMENTWISE_AFFINE):
                cute.autovec_copy(gWv[None, vec_idx], w_frag)
                xhat = xhat * (w_frag.load().to(Float32) + offset)
            out_frag.store(xhat.to(mY.element_type))
            cute.autovec_copy(out_frag, gYv[None, vec_idx])


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


@cute.kernel
def _rms_norm_bwd_fused_vector_kernel(
    mdY: cute.Tensor,  # (n_rows, n_cols) upstream grad
    mX: cute.Tensor,  # (n_rows, n_cols) saved input
    mW: cute.Tensor,  # (n_cols,) weight (affine only)
    mRSTD: cute.Tensor,  # (n_rows,) fp32 reciprocal-RMS cache
    mdX: cute.Tensor,  # (n_rows, n_cols) input grad out (may alias mdY for in-place)
    mdW: cute.Tensor,  # (num_strips, n_cols) fp32 partial weight grads (one row per strip)
    offset: Float32,
    CASTING_MODE: cutlass.Constexpr,
    N_COLS: cutlass.Constexpr,
    VEC: cutlass.Constexpr,
    NUM_VEC_TILES: cutlass.Constexpr,
    NUM_THREADS: cutlass.Constexpr,
    NUM_WARPS: cutlass.Constexpr,
):
    """Aligned affine backward matching Triton's persistent execution shape.

    Each CTA owns a contiguous row range, loads X/dY once into registers, and
    reuses those fragments after the row reduction for both dX and dW. The
    reduction barrier also fences every dY load before any in-place dX store.
    """
    tid, _, _ = cute.arch.thread_idx()
    lane = tid % 32
    warp = tid // 32
    strip, _, _ = cute.arch.block_idx()

    n_rows = mX.shape[0]
    num_strips = mdW.shape[0]
    n_vec = N_COLS // VEC

    smem = cutlass.utils.SmemAllocator()
    sm_warp = smem.allocate_tensor(Float32, cute.make_layout(NUM_WARPS), byte_alignment=4)
    sm_result = smem.allocate_tensor(Float32, cute.make_layout(1), byte_alignment=4)

    # W and dW are persistent row vectors, matching Triton's per-program state.
    gW = cute.make_tensor(
        cute.make_ptr(mW.element_type, mW.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16),
        cute.make_layout((N_COLS,)),
    )
    gWv = cute.tiled_divide(gW, (VEC,))
    w_frags = cute.make_rmem_tensor((VEC, NUM_VEC_TILES), mW.element_type)
    for ct in cutlass.range_constexpr(NUM_VEC_TILES):
        vec_idx = ct * NUM_THREADS + tid
        if vec_idx < n_vec:
            cute.autovec_copy(gWv[None, vec_idx], w_frags[None, ct])

    dw_acc = cute.make_rmem_tensor((VEC, NUM_VEC_TILES), Float32)
    dw_acc.fill(0.0)
    x_frags = cute.make_rmem_tensor((VEC, NUM_VEC_TILES), mX.element_type)
    dy_frags = cute.make_rmem_tensor((VEC, NUM_VEC_TILES), mdY.element_type)
    dx_frag = cute.make_rmem_tensor((VEC,), mdX.element_type)

    # Triton assigns one contiguous ceil-divided row range to each SM program.
    rows_per_strip = (n_rows + num_strips - 1) // num_strips
    row_start = strip * rows_per_strip
    for i in cutlass.range(0, rows_per_strip):
        r = row_start + i
        r_valid = r < n_rows
        # Constructing a row view is pointer arithmetic even when no copy follows.
        # Clamp the inactive final iteration to row 0 to keep that pointer in bounds.
        r_safe = Int32(0)
        if r_valid:
            r_safe = r
        rstd = Float32(0.0)
        if r_valid:
            rstd = mRSTD[r].to(Float32)

        # Rebuild row views with the host-validated alignment for vector copies.
        x_row = mX[r_safe, None]
        dy_row = mdY[r_safe, None]
        dx_row = mdX[r_safe, None]
        gX = cute.make_tensor(
            cute.make_ptr(mX.element_type, x_row.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((N_COLS,)),
        )
        gdY = cute.make_tensor(
            cute.make_ptr(mdY.element_type, dy_row.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((N_COLS,)),
        )
        gdX = cute.make_tensor(
            cute.make_ptr(mdX.element_type, dx_row.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((N_COLS,)),
        )
        gXv = cute.tiled_divide(gX, (VEC,))
        gdYv = cute.tiled_divide(gdY, (VEC,))
        gdXv = cute.tiled_divide(gdX, (VEC,))

        # Load every X/dY element before entering the CTA reduction. The fragments
        # remain live across the barrier and are reused below, as in Triton's SSA row.
        for ct in cutlass.range_constexpr(NUM_VEC_TILES):
            vec_idx = ct * NUM_THREADS + tid
            if r_valid and vec_idx < n_vec:
                cute.autovec_copy(gXv[None, vec_idx], x_frags[None, ct])
                cute.autovec_copy(gdYv[None, vec_idx], dy_frags[None, ct])

        dot = Float32(0.0)
        for ct in cutlass.range_constexpr(NUM_VEC_TILES):
            vec_idx = ct * NUM_THREADS + tid
            if r_valid and vec_idx < n_vec:
                dot = dot + (
                    x_frags[None, ct].load().to(Float32)
                    * dy_frags[None, ct].load().to(Float32)
                    * (w_frags[None, ct].load().to(Float32) + offset)
                ).reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
        dot_total = _cta_reduce_sum_warp0(dot, sm_warp, sm_result, lane, warp, NUM_WARPS)
        coef = (Float32(0.0) - rstd * rstd * dot_total) / Float32(N_COLS)

        # Reuse the original register fragments for dX and dW.
        for ct in cutlass.range_constexpr(NUM_VEC_TILES):
            vec_idx = ct * NUM_THREADS + tid
            if r_valid and vec_idx < n_vec:
                xf = x_frags[None, ct].load().to(Float32)
                dyf = dy_frags[None, ct].load().to(Float32)
                mk = dyf * (w_frags[None, ct].load().to(Float32) + offset)
                dx_frag.store((rstd * (mk + coef * xf)).to(mdX.element_type))
                cute.autovec_copy(dx_frag, gdXv[None, vec_idx])
                xhat = xf * rstd
                if const_expr(CASTING_MODE == _CASTING_MODE_LLAMA):
                    xhat = xhat.to(mX.element_type).to(Float32)
                dw_acc[None, ct].store(dw_acc[None, ct].load() + dyf * xhat)

    # Emit this strip's dW partials; the host sums the num_strips partial rows.
    dw_row = mdW[strip, None]
    gdW = cute.make_tensor(
        cute.make_ptr(mdW.element_type, dw_row.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16),
        cute.make_layout((N_COLS,)),
    )
    gdWv = cute.tiled_divide(gdW, (VEC,))
    dw_out = cute.make_rmem_tensor((VEC,), mdW.element_type)
    for ct in cutlass.range_constexpr(NUM_VEC_TILES):
        vec_idx = ct * NUM_THREADS + tid
        if vec_idx < n_vec:
            dw_out.store(dw_acc[None, ct].load().to(mdW.element_type))
            cute.autovec_copy(dw_out, gdWv[None, vec_idx])


# =============================================================================
# Host launch (compiled once per dtype/flag combo, then cached)
# =============================================================================
@cute.jit
def _rms_norm_fwd_vector_host(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mY: cute.Tensor,
    mRSTD: cute.Tensor,
    eps: Float32,
    offset: Float32,
    CASTING_MODE: cutlass.Constexpr,
    ELEMENTWISE_AFFINE: cutlass.Constexpr,
    N_COLS: cutlass.Constexpr,
    VEC: cutlass.Constexpr,
    NUM_VEC_TILES: cutlass.Constexpr,
    NUM_WARPS: cutlass.Constexpr,
    NUM_THREADS: cutlass.Constexpr,
    stream: cuda.CUstream = None,
):
    n_rows = mX.shape[0]
    smem_bytes = (((NUM_WARPS + 1) * 4 + 15) // 16) * 16
    _rms_norm_fwd_vector_kernel(
        mX,
        mW,
        mY,
        mRSTD,
        eps,
        offset,
        CASTING_MODE,
        ELEMENTWISE_AFFINE,
        N_COLS,
        VEC,
        NUM_VEC_TILES,
        NUM_WARPS,
        NUM_THREADS,
    ).launch(
        grid=[n_rows, 1, 1],
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


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


@cute.jit
def _rms_norm_bwd_fused_host(
    mdY: cute.Tensor,
    mX: cute.Tensor,
    mW: cute.Tensor,
    mRSTD: cute.Tensor,
    mdX: cute.Tensor,
    mdW: cute.Tensor,
    offset: Float32,
    CASTING_MODE: cutlass.Constexpr,
    N_COLS: cutlass.Constexpr,
    VEC: cutlass.Constexpr,
    NUM_VEC_TILES: cutlass.Constexpr,
    NUM_THREADS: cutlass.Constexpr,
    NUM_WARPS: cutlass.Constexpr,
    SMEM_BYTES: cutlass.Constexpr,
    stream: cuda.CUstream = None,
):
    num_strips = mdW.shape[0]
    _rms_norm_bwd_fused_vector_kernel(
        mdY,
        mX,
        mW,
        mRSTD,
        mdX,
        mdW,
        offset,
        CASTING_MODE,
        N_COLS,
        VEC,
        NUM_VEC_TILES,
        NUM_THREADS,
        NUM_WARPS,
    ).launch(
        grid=[num_strips, 1, 1],
        block=[NUM_THREADS, 1, 1],
        smem=SMEM_BYTES,
        stream=stream,
    )


def _is_16b_row_aligned(t):
    """Whether a contiguous 1D/2D tensor can safely use 16-byte row vectors."""
    if t is None or t.data_ptr() % 16 or t.stride(-1) != 1:
        return False
    return t.ndim < 2 or (t.stride(0) * t.element_size()) % 16 == 0


def _fast_vector_params(n_cols, *tensors, num_threads=_FAST_THREADS):
    """Return (VEC, vector-tiles/thread), or None for the scalar-safe fallback.

    Honor environment override LIGER_RMS_FORCE_NO_FAST=1 to disable the fast
    vectorized path so experiments can compare scalar/fallback performance.
    """
    if _FORCE_NO_FAST:
        if _DEBUG:
            _rms_debug("_fast_vector_params: fast path disabled by LIGER_RMS_FORCE_NO_FAST")
        return None

    if n_cols <= 0 or n_cols > _FAST_MAX_COLS or not all(_is_16b_row_aligned(t) for t in tensors):
        return None
    try:
        vec = fast_path_vector_width(*(t.element_size() for t in tensors))
    except ValueError:
        return None
    if n_cols % vec:
        return None
    return vec, (n_cols // vec + num_threads - 1) // num_threads


def _launch_fwd_vector(X, W, Y, RSTD, eps, offset, casting_mode, elementwise_affine, vec):
    """Launch the aligned vector forward specialization.

    The warp count (hence thread count and register-resident tiles per thread) is
    chosen from the hidden width by ``fwd_warp_count``; ``num_vec_tiles`` follows.
    """
    stream = _cute_stream()
    # Cache the marshaled handles for the INPUTS (X, W) -- their addresses are stable
    # across steps (weights always; activations under a reused-buffer harness), so they
    # hit the cache and marshal in ~0.4us. Y and RSTD are freshly allocated OUTPUTS:
    # caching them would pin the storage and stop the allocator from recycling the
    # address, so every call would churn a new address (misses + wasted pinning). Marshal
    # those uncached so their addresses stay reusable.
    x_ct = _to_cute_cached(X, assumed_align=16)
    y_ct = to_cute_tensor(Y, assumed_align=16)
    rstd_ct = to_cute_tensor(RSTD, assumed_align=4)
    w_ct = _to_cute_cached(W, assumed_align=16) if elementwise_affine else rstd_ct
    n_cols = X.shape[1]
    num_warps = fwd_warp_count(n_cols, vec)
    num_threads = 32 * num_warps
    num_vec_tiles = (n_cols // vec + num_threads - 1) // num_threads
    # Optionally bucket n_cols in the compile key to reduce cold-compile churn.
    bucket = _COMPILE_BUCKET
    n_cols_key = n_cols
    if bucket is not None and bucket > 0:
        n_cols_key = ((int(n_cols) + bucket - 1) // bucket) * bucket
    key = (
        "fwd_vec",
        n_cols_key,
        vec,
        num_vec_tiles,
        num_warps,
        X.dtype,
        W.dtype if elementwise_affine else None,
        casting_mode,
        elementwise_affine,
    )
    compiled = _compile_cache.get(key)
    _dbg = _DEBUG
    if _dbg:
        _rms_debug(
            f"_launch_fwd_vector key={key} (n_cols={n_cols} n_cols_key={n_cols_key}) cache_hit={compiled is not None}"
        )
    if compiled is None:
        compiled = cute.compile(
            _rms_norm_fwd_vector_host,
            x_ct,
            w_ct,
            y_ct,
            rstd_ct,
            float(eps),
            float(offset),
            casting_mode,
            elementwise_affine,
            n_cols,
            vec,
            num_vec_tiles,
            num_warps,
            num_threads,
            stream,
        )
        _compile_cache[key] = compiled
        if _dbg:
            _rms_debug(f"Compiled fwd_vec kernel for key: {key}")
    elif _dbg:
        _rms_debug(f"Reusing fwd_vec kernel for key: {key}")
    compiled(x_ct, w_ct, y_ct, rstd_ct, float(eps), float(offset), stream)


def _launch_fwd(X, W, Y, RSTD, eps, offset, casting_mode, elementwise_affine):
    fast_tensors = (X, Y, W) if elementwise_affine else (X, Y)
    fast_params = _fast_vector_params(X.shape[1], *fast_tensors)
    if fast_params is not None:
        # fast_params[0] is VEC; _launch_fwd_vector derives the width-aware warp/thread
        # count and num_vec_tiles itself, so only VEC is forwarded.
        _launch_fwd_vector(X, W, Y, RSTD, eps, offset, casting_mode, elementwise_affine, fast_params[0])
        return

    stream = _cute_stream()
    # Scalar (non-vectorized) access, so element-size alignment is all we assume — this
    # keeps the kernel correct for unaligned contiguous slices and irregular hidden dims.
    x_ct = _to_cute_cached(X, assumed_align=X.element_size())
    y_ct = _to_cute_cached(Y, assumed_align=Y.element_size())
    rstd_ct = _to_cute_cached(RSTD, assumed_align=4)  # fp32
    # Non-affine: reuse the fp32 RSTD handle as a dummy — the kernel never reads it.
    w_ct = _to_cute_cached(W, assumed_align=W.element_size()) if elementwise_affine else rstd_ct

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
    dy_ct = _to_cute_cached(dY, assumed_align=dY.element_size())
    x_ct = _to_cute_cached(X, assumed_align=X.element_size())
    rstd_ct = _to_cute_cached(RSTD, assumed_align=4)
    dx_ct = _to_cute_cached(dX, assumed_align=dX.element_size())
    w_ct = _to_cute_cached(W, assumed_align=W.element_size()) if elementwise_affine else rstd_ct

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
    dy_ct = _to_cute_cached(dY, assumed_align=dY.element_size())
    x_ct = _to_cute_cached(X, assumed_align=X.element_size())
    rstd_ct = _to_cute_cached(RSTD, assumed_align=4)
    dw_ct = _to_cute_cached(dW_partial, assumed_align=4)  # fp32 (num_strips, n_cols)

    # Key on every baked dtype: dY and X (mdW is always fp32). The llama cast bakes
    # mX.element_type; the loads bake mdY.element_type. rows_per_strip is a runtime
    # arg (not baked), so one compiled kernel serves every shape.
    key = ("bwd_dw", X.dtype, dY.dtype, casting_mode)
    if key not in _compile_cache:
        _compile_cache[key] = cute.compile(
            _rms_norm_bwd_dw_host, dy_ct, x_ct, rstd_ct, dw_ct, int(rows_per_strip), casting_mode, stream
        )
    _compile_cache[key](dy_ct, x_ct, rstd_ct, dw_ct, int(rows_per_strip), stream)


def _launch_bwd_fused(
    dY,
    X,
    W,
    RSTD,
    dX,
    dW_partial,
    offset,
    casting_mode,
    vec,
    num_vec_tiles,
    num_warps,
):
    """Launch the aligned register-resident affine backward specialization."""
    stream = _cute_stream()
    dy_ct = _to_cute_cached(dY, assumed_align=16)
    x_ct = _to_cute_cached(X, assumed_align=16)
    w_ct = _to_cute_cached(W, assumed_align=16)
    rstd_ct = _to_cute_cached(RSTD, assumed_align=4)
    dx_ct = _to_cute_cached(dX, assumed_align=16)
    dw_ct = _to_cute_cached(dW_partial, assumed_align=16)  # fp32 (num_strips, n_cols)
    n_cols = X.shape[1]
    num_threads = 32 * num_warps
    smem_bytes = (((num_warps + 1) * 4 + 15) // 16) * 16

    # The width and thread geometry size the register layouts and reduction
    # scratch, so every value is baked into the compiled specialization.
    # Optionally bucket the n_cols compile key to reduce cold-compile churn.
    bucket = _COMPILE_BUCKET
    n_cols_key = n_cols
    if bucket is not None and bucket > 0:
        # round up to the next bucket, keep original n_cols for actual bake
        n_cols_key = ((int(n_cols) + bucket - 1) // bucket) * bucket
    # Support an optional reload policy tag (registers|smem|gmem|auto) baked
    # into the compile-key so later device variants can be compiled per-policy.
    reload_policy = _RELOAD_POLICY
    key = (
        "bwd_fused_vec",
        n_cols_key,
        vec,
        num_vec_tiles,
        num_threads,
        num_warps,
        smem_bytes,
        reload_policy,
        X.dtype,
        dY.dtype,
        W.dtype,
        casting_mode,
    )
    if _DEBUG:
        _rms_debug(f"_launch_bwd_fused reload_policy={reload_policy}")
    cache_hit = key in _compile_cache
    if _DEBUG:
        _rms_debug(f"_launch_bwd_fused key={key} (n_cols={n_cols} n_cols_key={n_cols_key}) cache_hit={cache_hit}")
    # Warn when a non-auto policy is requested but no device variant exists yet.
    if reload_policy not in ("auto", "registers", "smem", "gmem"):
        if _DEBUG:
            _rms_debug(f"Unrecognized LIGER_RMS_RELOAD_POLICY={reload_policy}; falling back to 'auto' behavior")
    elif reload_policy in ("smem", "gmem"):
        # No specialized SMEM/GMEM variants implemented in this change; warn so
        # experimenters know they're tagging the compile key but still using the
        # current register-resident kernel implementation.
        if _DEBUG:
            _rms_debug(
                f"LIGER_RMS_RELOAD_POLICY={reload_policy} requested, but device-side variant not implemented; using current kernel implementation"
            )
    if not cache_hit:
        _compile_cache[key] = cute.compile(
            _rms_norm_bwd_fused_host,
            dy_ct,
            x_ct,
            w_ct,
            rstd_ct,
            dx_ct,
            dw_ct,
            float(offset),
            casting_mode,
            n_cols,
            vec,
            num_vec_tiles,
            num_threads,
            num_warps,
            smem_bytes,
            stream,
        )
        if _DEBUG:
            _rms_debug(f"Compiled kernel for key: {key}")
    else:
        if _DEBUG:
            _rms_debug(f"Reusing compiled kernel for key: {key}")
    _compile_cache[key](dy_ct, x_ct, w_ct, rstd_ct, dx_ct, dw_ct, float(offset), stream)


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

    if not elementwise_affine:
        # No weight gradient — a single dx kernel (one CTA per row) is all we need.
        _launch_bwd_dx(dY, X, W, RSTD, dX, offset, casting_mode, elementwise_affine)
        return dX.view(*shape), None

    # Affine: match Triton's one-program-per-SM persistent decomposition. Any
    # failed alignment/shape precondition selects the established split fallback.
    sm_count = _get_sm_count(X.device)
    # Allow explicit override of backward warp count for autotuning experiments
    backward_num_warps = backward_warp_count(n_cols)
    if _BACKWARD_WARPS is not None:
        backward_num_warps = _BACKWARD_WARPS
        if _DEBUG:
            _rms_debug(f"Overriding backward_num_warps with LIGER_RMS_BACKWARD_WARPS={_BACKWARD_WARPS}")
    num_threads = 32 * backward_num_warps

    # Allow forcing the split fallback for A/B testing: set LIGER_RMS_FORCE_SPLIT_BWD=1
    force_split = _FORCE_SPLIT_BWD

    # Honor a per-shape autotune file (JSON) only when explicitly requested via
    # the LIGER_RMS_AUTOTUNE_FILE environment variable. Expected format:
    # { "4096": { "time": ..., "config": { "warps":8, "bucket":16, "force_split":1 } }, ... }
    # NOTE: there is intentionally NO hardcoded default path -- a stray file must
    # never silently alter kernel selection (a prior /tmp default forced the slow
    # split fallback on every run).
    try:
        autotune_path = _AUTOTUNE_FILE
        if autotune_path and os.path.isfile(autotune_path):
            try:
                with open(autotune_path, "r") as f:
                    _autotune_map = json.load(f)
            except Exception:
                _autotune_map = {}
            cfg = None
            # keys may be strings or ints
            if str(n_cols) in _autotune_map:
                cfg = _autotune_map.get(str(n_cols))
            elif n_cols in _autotune_map:
                cfg = _autotune_map.get(n_cols)
            if cfg:
                # nested structure: {"time":..., "config": { ... }} or flat
                nested = cfg.get("config") if isinstance(cfg, dict) else None
                force_val = None
                if isinstance(nested, dict):
                    force_val = nested.get("force_split")
                elif isinstance(cfg, dict):
                    force_val = cfg.get("force_split")
                if int(force_val or 0):
                    force_split = True
                    if _DEBUG:
                        _rms_debug(f"Autotune file {autotune_path} requests force_split for n_cols={n_cols}")
    except Exception:
        # best-effort; ignore failures reading autotune file
        pass

    fast_params = None if force_split else _fast_vector_params(n_cols, dY, X, W, dX, num_threads=num_threads)
    num_strips = max(1, min(sm_count, n_rows))
    # The fused register-resident backward launches exactly one block per strip
    # (grid = num_strips). At one strip per SM its high register pressure (~122
    # regs/thread caps the SM at 2 resident blocks) leaves only a couple of
    # resident warps, so the kernel is occupancy/latency bound rather than
    # compute or memory bound. Once there is enough row-parallelism that each
    # strip would otherwise process many rows serially, launch two strips per SM
    # to fill that register-limited 2-blocks/SM ceiling and expose more waves.
    # (Measured on B200: ~1.2-1.5x faster for n_rows>=4096, neutral below; going
    # past 2 strips/SM cannot raise occupancy and only adds scheduling waves.)
    # LIGER_RMS_FUSED_STRIP_MULT overrides the auto choice with an explicit
    # strips-per-SM count. Only applies to the fused path; the split dw kernel
    # already saturates occupancy, so leave it untouched.
    if fast_params is not None:
        strip_mult = _FUSED_STRIP_MULT
        if strip_mult <= 0:
            # Auto: double once latency-bound (threshold sits inside the measured
            # 13.8-27.7 rows/strip crossover window), else one strip per SM.
            strip_mult = 2 if n_rows >= _FUSED_DOUBLE_STRIP_ROWS_PER_SM * sm_count else 1
        if strip_mult > 1:
            num_strips = max(1, min(n_rows, sm_count * strip_mult))
    rows_per_strip = (n_rows + num_strips - 1) // num_strips
    dW_partial = _get_dw_partial_buf(num_strips, n_cols, W.device)
    if _DEBUG:
        try:
            _rms_debug(
                json.dumps(
                    {
                        "n_rows": int(n_rows),
                        "n_cols": int(n_cols),
                        "sm_count": int(sm_count),
                        "backward_num_warps": int(backward_num_warps),
                        "num_threads": int(num_threads),
                        "force_split": bool(force_split),
                        "fast_params": str(fast_params),
                        "num_strips": int(num_strips),
                        "rows_per_strip": int(rows_per_strip),
                        "dW_partial_shape": tuple(dW_partial.shape),
                    }
                )
            )
        except Exception:
            _rms_debug(
                f"rms_norm_backward debug: n_rows={n_rows} n_cols={n_cols} force_split={force_split} fast_params={fast_params}"
            )

    if fast_params is not None and _is_16b_row_aligned(dW_partial):
        # One global read of each X/dY element; the row reduction fences those
        # register loads before any in-place dX stores.
        vec, num_vec_tiles = fast_params
        _launch_bwd_fused(
            dY,
            X,
            W,
            RSTD,
            dX,
            dW_partial,
            offset,
            casting_mode,
            vec,
            num_vec_tiles,
            backward_num_warps,
        )
    else:
        # dW runs BEFORE dx because in-place dx overwrites dY, but dW needs the
        # original upstream gradient.
        _launch_bwd_dw(dY, X, RSTD, dW_partial, rows_per_strip, casting_mode)
        _launch_bwd_dx(dY, X, W, RSTD, dX, offset, casting_mode, elementwise_affine)

    dW = dW_partial.sum(dim=0).to(W.dtype)
    return dX.view(*shape), dW


# ---------------------------------------------------------------------------
# Dispatch delegation (B200 optimization, iteration 2)
# ---------------------------------------------------------------------------
# The Quack-derived kernels in the multi-backend dispatcher
# (``ops/backends/_cutedsl/rms_norm.py``) measurably beat the inline kernels
# below on every benchmarked B200 shape (llama_3_8b, hidden 4096, bf16 @ seq
# 8192, wall ms: fwd 0.0618->0.0250, bwd 0.2067->0.0841, full 0.2683->0.1261
# — also faster than the Triton kernel there). They are, however, compile-
# gated: hidden dim must be divisible by the vector width (16 bytes /
# elem_size) and <= 32K (``_BWD_MAX_TILE_CUTEDSL``). Module-replacement users
# (``LIGER_KERNEL_IMPL=cutedsl``) therefore route through the guarded
# delegation below: fast Quack kernel when supported, the inline kernels
# otherwise — mirroring the guards in the dispatcher's ``rms_norm_cutedsl``.
_FAST_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_FAST_DISPATCH_FN = None


def _fast_dispatch_fn():
    """Deferred-import the dispatcher's autograd Function (lazy; no cycle —
    ``backends/_cutedsl/rms_norm`` only depends on ``ops/_nvidia_shared``)."""
    global _FAST_DISPATCH_FN
    if _FAST_DISPATCH_FN is None:
        from liger_kernel.ops.backends._cutedsl.rms_norm import _LigerRMSNormCuTeDSLFunction

        _FAST_DISPATCH_FN = _LigerRMSNormCuTeDSLFunction
    return _FAST_DISPATCH_FN


def _fast_dispatch_supported(X, W) -> bool:
    """True when the Quack-derived dispatcher kernel can run this call."""
    if not (isinstance(X, torch.Tensor) and X.is_cuda):
        return False
    if X.dtype not in _FAST_SUPPORTED_DTYPES:
        return False
    if W is not None and W.dtype not in _FAST_SUPPORTED_DTYPES:
        return False
    N = X.shape[-1]
    vecwidth = 16 // X.element_size()
    if N % vecwidth != 0:
        return False
    if N > 32768:  # _BWD_MAX_TILE_CUTEDSL in the dispatcher backend
        return False
    return True


class LigerRMSNormFunction(torch.autograd.Function):
    """
    CuTe DSL autograd wrapper for RMSNorm.

    Signature-compatible with ``liger_kernel.ops.rms_norm.LigerRMSNormFunction``:
    ``forward(X, W, eps, offset=0.0, casting_mode="llama", in_place=True, row_mode=None)``.
    See that class for the semantics of ``offset``, ``casting_mode`` and ``in_place``.

    Runs the fast Quack-derived dispatcher kernel when the shape/dtype allow it
    (see ``_fast_dispatch_supported``); otherwise falls back to the inline
    kernels in this module (also reachable directly via ``rms_norm_forward`` /
    ``rms_norm_backward``).
    """

    @staticmethod
    def forward(ctx, X, W, eps, offset=0.0, casting_mode="llama", in_place=True, row_mode=None):
        """
        X: (B, T, H) or (BxT, H)
        W: (H,)
        """
        if _fast_dispatch_supported(X, W):
            ctx._impl = "dispatch"
            # Re-use THIS autograd node: parameterize the shared ctx exactly as
            # the dispatcher's Function does, then let its backward handle dY.
            return _fast_dispatch_fn().forward(ctx, X, W, eps, offset, casting_mode, in_place, row_mode)

        ctx._impl = "inline"
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
        if ctx._impl == "dispatch":
            return _fast_dispatch_fn().backward(ctx, dY)

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
