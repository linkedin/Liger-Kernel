"""CuteDSL (NVIDIA CUTLASS Python DSL) implementation of Rotary Positional
Embedding (RoPE).

This is a drop-in alternative to the Triton kernel in
``liger_kernel.ops.rope``. It implements the HuggingFace Llama / Mistral RoPE
rotation (the "rotate-half" variant):

    forward:   q1' = q1 * cos - q2 * sin
               q2' = q2 * cos + q1 * sin
    backward:  d1' = d1 * cos + d2 * sin
               d2' = d2 * cos - d1 * sin

where ``q1`` / ``q2`` are the first / second halves of each head's ``head_dim``
slice and ``cos`` / ``sin`` are broadcast across the head axis.

Design notes (tuned on B200 / sm_100):
  * RoPE is purely memory-bound. Both q and k are read once and written once, so
    HBM traffic is the minimum 2x -- identical to the Triton kernel.
  * The kernel views q (resp. k) as a flat ``(RR, head_dim)`` matrix where
    ``RR`` collapses the (batch, head, seq) -- or (batch, seq, head) -- axes into
    a single "head-row" mode of stride ``head_dim``. The two ``head_dim // 2``
    halves ``q1`` / ``q2`` are addressed as strided views ``RR x hd_half``.
  * A 2-D ``TiledCopy`` (thread layout ``(TR, TC)``, value layout ``(VR, VEC)``)
    maps each thread to ``VR`` head-rows x ``VEC`` contiguous columns. Picking a
    *small* ``VEC`` (so a warp spans many head-rows along the row axis) keeps the
    HBM transactions fully coalesced; each thread then processes ``VR``
    independent rows, which exposes enough memory-level parallelism to hide load
    latency. This is ~2x the bandwidth of a naive one-token-per-CTA layout for
    bf16.
  * Every head-row shares the same ``cos`` / ``sin`` for a given token, so the
    cos/sin chunk is loaded *once per row* with a vectorized access and reused
    across the ``VEC`` rotation lanes -- avoiding a redundant per-element gather.
  * ``cos`` / ``sin`` are read in fp32 and the rotation math is done in fp32 for
    numerical parity with the Triton kernel, then cast back to the I/O dtype.
  * The kernel matches the Triton convention exactly: it rotates the
    ``head_dim // 2`` pairs ``(d, d + head_dim // 2)``; for an odd ``head_dim``
    the final element is left untouched, just like the reference.
  * TMA launches use direct PyTorch tensor arguments and the current framework
    stream when TVM FFI is available. The fallback creates fresh pointer bindings:
    changing a cached native memref's pointer does not change its TVM tensor.

The public API mirrors ``liger_kernel.ops.rope``:
``rope_forward`` / ``rope_backward`` / ``LigerRopeCuteDSLFunction``.
"""

from functools import lru_cache
from threading import Lock

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cute_utils
import torch

from cutlass.cute.nvgpu import cpasync

from liger_kernel.ops.cutedsl.ops.utils import _COMPILE_CACHE
from liger_kernel.ops.cutedsl.ops.utils import _dyn
from liger_kernel.ops.cutedsl.ops.utils import torch2cute_dtype_map

try:
    import tvm_ffi  # noqa: F401

    _TVM_FFI_PRESENT = True
except Exception:  # pragma: no cover - depends on optional deps
    _TVM_FFI_PRESENT = False

_TMA_COMPILE_LOCK = Lock()

# Cache the CUstream wrapper keyed on torch's raw stream handle so we don't
# rebuild the cuda.CUstream object every launch. The kernels MUST run on
# PyTorch's current stream (not the default/null stream) to preserve ordering
# under multi-stream execution (pipeline parallelism, grad checkpointing).
# Mirrors cross_entropy.py / rms_norm.py.
_stream_cache: dict = {}


def _cute_stream():
    raw = torch.cuda.current_stream().cuda_stream
    s = _stream_cache.get(raw)
    if s is None:
        s = cuda.CUstream(raw)
        _stream_cache[raw] = s
    return s


# ---------------------------------------------------------------------------
# Token-per-CTA kernel (mirrors the Triton design): one CTA owns a token, its
# hd_half threads each rotate one column-pair across EVERY q & k head, so cos/sin
# load once per token and reuse across all heads -- in forward AND backward.
# ---------------------------------------------------------------------------
@cute.kernel
def _rope_token_kernel(
    mQ: cute.Tensor,  # 4D (b, A, B, hd): (A,B)=(seq,head) fwd, (head,seq) bwd
    mK: cute.Tensor,
    mCos: cute.Tensor,  # (cos_bs, seq, hd)
    mSin: cute.Tensor,
    seq_len: cutlass.Int32,
    HD_HALF: cutlass.Constexpr,
    NQH: cutlass.Constexpr,
    NKH: cutlass.Constexpr,
    HEAD_PAR: cutlass.Constexpr,  # warp-groups cooperating on one token's heads
    QITERS: cutlass.Constexpr,  # ceil(NQH / HEAD_PAR)
    KITERS: cutlass.Constexpr,  # ceil(NKH / HEAD_PAR)
    SEQ_INNER: cutlass.Constexpr,
    COS_BCAST: cutlass.Constexpr,
    BACKWARD: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    tok, _, _ = cute.arch.block_idx()
    b = tok // seq_len
    s = tok % seq_len
    cb = 0 if cutlass.const_expr(COS_BCAST) else b
    # Each thread owns one column ``col`` of head-group ``hg``; the HEAD_PAR groups
    # split the heads so a token is processed by HEAD_PAR*HD_HALF threads in
    # parallel (instead of HD_HALF threads looping every head serially).
    col = tid % HD_HALF
    hg = tid // HD_HALF
    if hg < HEAD_PAR:
        c = mCos[cb, s, col].to(cutlass.Float32)
        sn = mSin[cb, s, col].to(cutlass.Float32)
        for i in cutlass.range_constexpr(QITERS):
            h = hg + i * HEAD_PAR
            if h < NQH:
                tq = mQ[b, s, h, None] if cutlass.const_expr(not SEQ_INNER) else mQ[b, h, s, None]
                x1 = tq[col].to(cutlass.Float32)
                x2 = tq[col + HD_HALF].to(cutlass.Float32)
                if cutlass.const_expr(BACKWARD):
                    tq[col] = (x1 * c + x2 * sn).to(mQ.element_type)
                    tq[col + HD_HALF] = (x2 * c - x1 * sn).to(mQ.element_type)
                else:
                    tq[col] = (x1 * c - x2 * sn).to(mQ.element_type)
                    tq[col + HD_HALF] = (x2 * c + x1 * sn).to(mQ.element_type)
        for i in cutlass.range_constexpr(KITERS):
            h = hg + i * HEAD_PAR
            if h < NKH:
                tk = mK[b, s, h, None] if cutlass.const_expr(not SEQ_INNER) else mK[b, h, s, None]
                x1 = tk[col].to(cutlass.Float32)
                x2 = tk[col + HD_HALF].to(cutlass.Float32)
                if cutlass.const_expr(BACKWARD):
                    tk[col] = (x1 * c + x2 * sn).to(mK.element_type)
                    tk[col + HD_HALF] = (x2 * c - x1 * sn).to(mK.element_type)
                else:
                    tk[col] = (x1 * c - x2 * sn).to(mK.element_type)
                    tk[col + HD_HALF] = (x2 * c + x1 * sn).to(mK.element_type)


def _head_par(nqh, hd_half):
    # Pick HEAD_PAR so block ~256 threads (8 warps) for good occupancy, without
    # exceeding the number of q heads.
    hp = max(1, min(nqh, 256 // max(1, hd_half)))
    return hp


def _make_launch_token(nqh, nkh, hd_half, seq_inner, cos_bcast, backward):
    head_par = _head_par(nqh, hd_half)
    qiters = (nqh + head_par - 1) // head_par
    kiters = (nkh + head_par - 1) // head_par
    block = max(32, hd_half * head_par)

    @cute.jit
    def launch(mQ, mK, mCos, mSin, n_tok: cutlass.Int32, seq_len: cutlass.Int32, stream: cuda.CUstream = None):
        _rope_token_kernel(
            mQ,
            mK,
            mCos,
            mSin,
            seq_len,
            hd_half,
            nqh,
            nkh,
            head_par,
            qiters,
            kiters,
            seq_inner,
            cos_bcast,
            backward,
        ).launch(grid=[n_tok, 1, 1], block=[block, 1, 1], stream=stream)

    return launch


# cos/sin are persistent rotary buffers reused every step; exporting them to a
# cute tensor costs ~5 us each, so cache the dlpack handle keyed by storage ptr.
_DLPACK_CACHE: dict = {}


def _dyn_cached(t: torch.Tensor):
    key = (t.data_ptr(), tuple(t.shape), t.dtype, t.device.index)
    h = _DLPACK_CACHE.get(key)
    if h is None:
        h = _dyn(t)
        if len(_DLPACK_CACHE) > 16:
            _DLPACK_CACHE.clear()
        _DLPACK_CACHE[key] = h
    return h


def _get_compiled_token(q4, k4, cos3, sin3, nqh, nkh, hd_half, seq_inner, cos_bcast, backward):
    key = (
        q4.dtype,
        k4.dtype,
        cos3.dtype,
        sin3.dtype,
        nqh,
        nkh,
        hd_half,
        seq_inner,
        cos_bcast,
        backward,
        "tok",
        q4.device.index,
    )
    fn = _COMPILE_CACHE.get(key)
    if fn is not None:
        return fn
    fn = cute.compile(
        _make_launch_token(nqh, nkh, hd_half, seq_inner, cos_bcast, backward),
        _dyn(q4),
        _dyn(k4),
        _dyn(cos3),
        _dyn(sin3),
        cutlass.Int32(1),
        cutlass.Int32(1),
        _cute_stream(),
    )
    _COMPILE_CACHE[key] = fn
    return fn


def _apply_token(q4, k4, cos3, sin3, nqh, nkh, hd_half, seq_inner, cos_bcast, seq_len, backward):
    """One CTA per token: cos/sin load once and reuse across all heads (q & k)."""
    n_tok = q4.shape[0] * seq_len
    fn = _get_compiled_token(q4, k4, cos3, sin3, nqh, nkh, hd_half, seq_inner, cos_bcast, backward)
    fn(
        _dyn(q4),
        _dyn(k4),
        _dyn_cached(cos3),
        _dyn_cached(sin3),
        cutlass.Int32(n_tok),
        cutlass.Int32(seq_len),
        _cute_stream(),
    )


# ---------------------------------------------------------------------------
# TMA (Tensor Memory Accelerator) kernel -- B200 / sm_100.
#
# RoPE is purely memory-bound and *in place* (each head-row is read once and
# written back to the SAME address). The synchronous per-thread ``cute.copy``
# idiom used by ``_rope_kernel`` stalls on the load->use dependency and tops out
# around 0.68x of HBM peak (it also silently demotes the transfer to 16-bit
# scalars). TMA instead bulk-copies a ``(TILE_M, head_dim)`` tile global->shared
# with a single async instruction issued by one warp, fully saturating HBM; the
# rotation math then runs on fast shared memory and the tile is bulk-copied back
# shared->global. This reaches ~0.93x of the in-place HBM ceiling -- matching the
# Triton kernel -- versus ~0.45x for the token-per-CTA kernel.
#
# Tiling: one CTA owns ``TILE_M`` contiguous head-rows. ``_TMA_TR * _TMA_VR`` must
# equal ``TILE_M`` and a 2-D ``TiledCopy`` (thread ``(TR, TC)``, value
# ``(VR, VEC)``) partitions the two ``hd_half`` halves of the shared tile exactly
# like ``_rope_kernel`` -- but the source is shared memory, so the partition is
# coalesced and bank-conflict free. ``VEC`` is a full 128-bit access.
# ---------------------------------------------------------------------------
_TMA_TILE_M = 64
_TMA_TR = 16
_TMA_VR = 4  # _TMA_TR * _TMA_VR == _TMA_TILE_M


def _tma_supported(dtype: torch.dtype, head_dim: int, device: torch.device = None) -> bool:
    if not torch.cuda.is_available():
        return False
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return _tma_supported_cached(dtype, head_dim, device)


@lru_cache(maxsize=128)
def _tma_supported_cached(dtype: torch.dtype, head_dim: int, device: torch.device) -> bool:
    """The TMA kernel requires a 128-bit-vectorizable, 16-byte-box-aligned tile.

    ``head_dim`` must be even, ``hd_half`` a multiple of the 128-bit vector width,
    and the tile's innermost box (``head_dim`` elements) a multiple of 16 bytes so
    the bulk copy is legal. Odd / awkward ``head_dim`` (e.g. 92) falls back to the
    portable token kernel.
    """
    # TMA (bulk async copy) is Hopper (sm_90)+ / Blackwell only. The vectorizable
    # shape checks below also pass on older archs, so gate on the compute
    # capability first and fall back to the portable token kernel on Ampere and
    # earlier instead of failing at kernel compile/launch. Query the *input
    # tensor's* device (not the current/default one) so heterogeneous multi-GPU
    # processes route each tensor by its own arch.
    if torch.cuda.get_device_capability(device)[0] < 9:
        return False
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if head_dim % 2 != 0:
        return False
    hd_half = head_dim // 2
    vec = max(1, 128 // torch.finfo(dtype).bits)
    bits = torch.finfo(dtype).bits
    if hd_half % vec != 0:
        return False
    # innermost TMA box must be 16-byte aligned
    if (head_dim * bits) % 128 != 0:
        return False
    # The TMA launch uses block = 16 * (hd_half / vec) threads; a CUDA block is
    # capped at 1024 threads. Oversized heads (e.g. bf16 head_dim=2048 -> 2048
    # threads) would fail to launch, so fall back to the portable token kernel.
    if 16 * (hd_half // vec) > 1024:
        return False
    return True


@cute.kernel
def _rope_tma_kernel(
    tma_load: cute.CopyAtom,
    g_load: cute.Tensor,  # local-tiled TMA source view  (TILE_M, hd, ntiles)
    tma_store: cute.CopyAtom,
    g_store: cute.Tensor,  # local-tiled TMA dest view
    smem_layout: cute.Layout,  # (TILE_M, hd)
    mCos: cute.Tensor,  # (cos_bs, seq, hd)
    mSin: cute.Tensor,
    RR: cutlass.Int32,  # true head-row count (for cos/sin clamp)
    seq_len: cutlass.Int32,
    SS: cutlass.Constexpr,  # smem struct type
    nbytes: cutlass.Constexpr,  # bytes per tile (TMA expect-tx)
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
    HD_HALF: cutlass.Constexpr,
    NH: cutlass.Constexpr,
    VR: cutlass.Constexpr,
    VEC: cutlass.Constexpr,
    SEQ_INNER: cutlass.Constexpr,
    COS_BCAST: cutlass.Constexpr,
    BACKWARD: cutlass.Constexpr,
):
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    smem = cute_utils.SmemAllocator()
    st = smem.allocate(SS)
    bar = st.bar.data_ptr()
    with cute.arch.elect_one():
        cute.arch.mbarrier_init(bar, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    sm = st.data.get_tensor(smem_layout)
    tAsA, tAgA = cpasync.tma_partition(
        tma_load, 0, cute.make_layout(1), cute.group_modes(sm, 0, 2), cute.group_modes(g_load, 0, 2)
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_store, 0, cute.make_layout(1), cute.group_modes(sm, 0, 2), cute.group_modes(g_store, 0, 2)
    )

    # Bulk load global->shared: a single warp issues the TMA collectively.
    if tidx < 32:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar, nbytes)
        cute.copy(tma_load, tAgA[(None, bidx, 0)], tAsA, tma_bar_ptr=bar)
    cute.arch.mbarrier_wait(bar, 0)

    # Coalesced TV-partition of the two hd_half halves of the shared tile.
    sm1 = cute.make_tensor(sm.iterator, cute.make_layout((_TMA_TILE_M, HD_HALF), stride=(2 * HD_HALF, 1)))
    sm2 = cute.make_tensor(sm.iterator + HD_HALF, cute.make_layout((_TMA_TILE_M, HD_HALF), stride=(2 * HD_HALF, 1)))
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), sm.element_type)
    tc = cute.make_tiled_copy_tv(atom, thr_layout, val_layout).get_slice(tidx)
    idC = cute.make_identity_tensor((_TMA_TILE_M, HD_HALF))
    tX1 = tc.partition_S(sm1)
    tX2 = tc.partition_S(sm2)
    crd = tc.partition_S(idC)
    f1 = cute.make_fragment_like(tX1)
    f2 = cute.make_fragment_like(tX2)
    cute.autovec_copy(tX1, f1)
    cute.autovec_copy(tX2, f2)

    tile_base = bidx * _TMA_TILE_M
    for r in cutlass.range_constexpr(VR):
        base = r * VEC
        row = crd[base][0]
        col0 = crd[base][1]
        rr = tile_base + row
        rr_safe = rr if rr < RR else 0
        if cutlass.const_expr(SEQ_INNER):
            s = rr_safe % seq_len
            b = rr_safe // (NH * seq_len)
        else:
            s = (rr_safe // NH) % seq_len
            b = rr_safe // (NH * seq_len)
        cb = 0 if cutlass.const_expr(COS_BCAST) else b
        tcos = cute.local_tile(mCos[cb, s, None], (VEC,), (col0 // VEC,))
        tsin = cute.local_tile(mSin[cb, s, None], (VEC,), (col0 // VEC,))
        fc = cute.make_fragment_like(tcos)
        fs = cute.make_fragment_like(tsin)
        cute.autovec_copy(tcos, fc)
        cute.autovec_copy(tsin, fs)
        for k in cutlass.range_constexpr(VEC):
            j = base + k
            x1 = f1[j].to(cutlass.Float32)
            x2 = f2[j].to(cutlass.Float32)
            cval = fc[k].to(cutlass.Float32)
            sval = fs[k].to(cutlass.Float32)
            if cutlass.const_expr(BACKWARD):
                o1 = x1 * cval + x2 * sval
                o2 = x2 * cval - x1 * sval
            else:
                o1 = x1 * cval - x2 * sval
                o2 = x2 * cval + x1 * sval
            f1[j] = o1.to(sm.element_type)
            f2[j] = o2.to(sm.element_type)
    cute.autovec_copy(f1, tX1)
    cute.autovec_copy(f2, tX2)

    # shared->global bulk store (generic-proxy writes must be visible to TMA).
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier()
    if tidx < 32:
        cute.copy(tma_store, tBsB, tBgB[(None, bidx, 0)])
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0)


@cute.kernel
def _rope_tma_qk_kernel(
    tma_load_q: cute.CopyAtom,
    g_load_q: cute.Tensor,
    tma_store_q: cute.CopyAtom,
    g_store_q: cute.Tensor,
    tma_load_k: cute.CopyAtom,
    g_load_k: cute.Tensor,
    tma_store_k: cute.CopyAtom,
    g_store_k: cute.Tensor,
    smem_layout: cute.Layout,  # (TILE_M, hd)
    mCos: cute.Tensor,
    mSin: cute.Tensor,
    RRq: cutlass.Int32,
    RRk: cutlass.Int32,
    ntiles_q: cutlass.Int32,
    SEQ_LEN: cutlass.Constexpr,
    SS: cutlass.Constexpr,
    nbytes: cutlass.Constexpr,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
    HD_HALF: cutlass.Constexpr,
    NHQ: cutlass.Constexpr,
    NHK: cutlass.Constexpr,
    VR: cutlass.Constexpr,
    VEC: cutlass.Constexpr,
    SEQ_INNER: cutlass.Constexpr,
    COS_BCAST: cutlass.Constexpr,
    BACKWARD: cutlass.Constexpr,
):
    """Rotate ONE tile (q or k) per CTA in a single fused launch.

    Grid spans ``ntiles_q + ntiles_k`` tiles: CTAs ``[0, ntiles_q)`` rotate q tiles,
    the rest rotate k tiles. Using a single 16 KB shared tile per CTA (q OR k --
    never both) keeps occupancy at the full ~12 blocks/SM; loading both tiles per
    CTA instead needs 32 KB and halves occupancy (measured 37.5% vs 75%), which
    bottlenecks HBM. Folding k into q's launch removes the second launch's fixed
    overhead that otherwise caps the small k pass at ~0.4x of HBM peak.
    """
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    smem = cute_utils.SmemAllocator()
    st = smem.allocate(SS)
    bar = st.bar.data_ptr()
    with cute.arch.elect_one():
        cute.arch.mbarrier_init(bar, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    sm = st.data.get_tensor(smem_layout)
    qAs, qAg = cpasync.tma_partition(
        tma_load_q, 0, cute.make_layout(1), cute.group_modes(sm, 0, 2), cute.group_modes(g_load_q, 0, 2)
    )
    qBs, qBg = cpasync.tma_partition(
        tma_store_q, 0, cute.make_layout(1), cute.group_modes(sm, 0, 2), cute.group_modes(g_store_q, 0, 2)
    )
    kAs, kAg = cpasync.tma_partition(
        tma_load_k, 0, cute.make_layout(1), cute.group_modes(sm, 0, 2), cute.group_modes(g_load_k, 0, 2)
    )
    kBs, kBg = cpasync.tma_partition(
        tma_store_k, 0, cute.make_layout(1), cute.group_modes(sm, 0, 2), cute.group_modes(g_store_k, 0, 2)
    )

    is_q = bidx < ntiles_q
    local = bidx if is_q else bidx - ntiles_q
    nh = NHQ if is_q else NHK
    RR = RRq if is_q else RRk
    tile_base = local * _TMA_TILE_M

    # Bulk load global->shared (one warp issues the TMA collectively).
    if tidx < 32:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar, nbytes)
        if is_q:
            cute.copy(tma_load_q, qAg[(None, local, 0)], qAs, tma_bar_ptr=bar)
        else:
            cute.copy(tma_load_k, kAg[(None, local, 0)], kAs, tma_bar_ptr=bar)
    # TV-partition (pure address arithmetic -- valid before the bulk data lands).
    sm1 = cute.make_tensor(sm.iterator, cute.make_layout((_TMA_TILE_M, HD_HALF), stride=(2 * HD_HALF, 1)))
    sm2 = cute.make_tensor(sm.iterator + HD_HALF, cute.make_layout((_TMA_TILE_M, HD_HALF), stride=(2 * HD_HALF, 1)))
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), sm.element_type)
    tc = cute.make_tiled_copy_tv(atom, thr_layout, val_layout).get_slice(tidx)
    idC = cute.make_identity_tensor((_TMA_TILE_M, HD_HALF))
    tX1 = tc.partition_S(sm1)
    tX2 = tc.partition_S(sm2)
    crd = tc.partition_S(idC)

    if cutlass.const_expr(not SEQ_INNER and NHQ % VR == 0 and NHK % VR == 0):
        # Both passes run on (b, s, h, hd) storage (backward transposes its grads to
        # this native layout): a thread's VR rows are CONTIGUOUS and (since every head
        # count is a multiple of VR) fall in the same head-block, i.e. the SAME (b, s)
        # token -- so cos/sin is identical for all of them. Loading it once instead of
        # VR times cuts the cos/sin gather 4x. That gather is the whole gap to the
        # pure-copy ceiling (cos-free fused = 9.07 vs 7.43 TB/s), dominated by a
        # long-scoreboard stall on the global loads. The single load is issued BEFORE
        # waiting on the q/k bulk transfer (it depends only on tile coordinates, not
        # data), so its L2 latency overlaps the TMA HBM stream instead of stalling the
        # rotate math afterwards.
        row0 = crd[0][0]
        col0 = crd[0][1]
        rr0 = tile_base + row0
        rr0 = rr0 if rr0 < RR else 0
        s = (rr0 // nh) % SEQ_LEN
        cb = 0 if cutlass.const_expr(COS_BCAST) else rr0 // (nh * SEQ_LEN)
        tcos = cute.local_tile(mCos[cb, s, None], (VEC,), (col0 // VEC,))
        tsin = cute.local_tile(mSin[cb, s, None], (VEC,), (col0 // VEC,))
        fc = cute.make_fragment_like(tcos)
        fs = cute.make_fragment_like(tsin)
        cute.autovec_copy(tcos, fc)
        cute.autovec_copy(tsin, fs)

        cute.arch.mbarrier_wait(bar, 0)
        f1 = cute.make_fragment_like(tX1)
        f2 = cute.make_fragment_like(tX2)
        cute.autovec_copy(tX1, f1)
        cute.autovec_copy(tX2, f2)
        for r in cutlass.range_constexpr(VR):
            base = r * VEC
            for kk in cutlass.range_constexpr(VEC):
                j = base + kk
                x1 = f1[j].to(cutlass.Float32)
                x2 = f2[j].to(cutlass.Float32)
                cval = fc[kk].to(cutlass.Float32)
                sval = fs[kk].to(cutlass.Float32)
                if cutlass.const_expr(BACKWARD):
                    o1 = x1 * cval + x2 * sval
                    o2 = x2 * cval - x1 * sval
                else:
                    o1 = x1 * cval - x2 * sval
                    o2 = x2 * cval + x1 * sval
                f1[j] = o1.to(sm.element_type)
                f2[j] = o2.to(sm.element_type)
    else:
        # General path (backward, or forward when a head count is not a multiple of
        # VR): each of a thread's VR rows is a DISTINCT token, so cos/sin cannot be
        # deduplicated. Issue all VR cos/sin gathers BEFORE waiting on the q/k bulk
        # transfer -- their addresses depend only on tile coordinates, so the L2
        # gather latency (the dominant long-scoreboard stall, ncu: 874 samples on
        # the backward pass) overlaps the TMA HBM stream instead of stalling the
        # rotate math afterwards.
        fcs = []
        fss = []
        for r in cutlass.range_constexpr(VR):
            base = r * VEC
            row = crd[base][0]
            col0 = crd[base][1]
            rr = tile_base + row
            rr_safe = rr if rr < RR else 0
            if cutlass.const_expr(SEQ_INNER):
                s = rr_safe % SEQ_LEN
            else:
                s = (rr_safe // nh) % SEQ_LEN
            cb = 0 if cutlass.const_expr(COS_BCAST) else rr_safe // (nh * SEQ_LEN)
            tcos = cute.local_tile(mCos[cb, s, None], (VEC,), (col0 // VEC,))
            tsin = cute.local_tile(mSin[cb, s, None], (VEC,), (col0 // VEC,))
            fc = cute.make_fragment_like(tcos)
            fs = cute.make_fragment_like(tsin)
            cute.autovec_copy(tcos, fc)
            cute.autovec_copy(tsin, fs)
            fcs.append(fc)
            fss.append(fs)

        cute.arch.mbarrier_wait(bar, 0)
        f1 = cute.make_fragment_like(tX1)
        f2 = cute.make_fragment_like(tX2)
        cute.autovec_copy(tX1, f1)
        cute.autovec_copy(tX2, f2)
        for r in cutlass.range_constexpr(VR):
            base = r * VEC
            fc = fcs[r]
            fs = fss[r]
            for kk in cutlass.range_constexpr(VEC):
                j = base + kk
                x1 = f1[j].to(cutlass.Float32)
                x2 = f2[j].to(cutlass.Float32)
                cval = fc[kk].to(cutlass.Float32)
                sval = fs[kk].to(cutlass.Float32)
                if cutlass.const_expr(BACKWARD):
                    o1 = x1 * cval + x2 * sval
                    o2 = x2 * cval - x1 * sval
                else:
                    o1 = x1 * cval - x2 * sval
                    o2 = x2 * cval + x1 * sval
                f1[j] = o1.to(sm.element_type)
                f2[j] = o2.to(sm.element_type)
    cute.autovec_copy(f1, tX1)
    cute.autovec_copy(f2, tX2)

    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier()
    if tidx < 32:
        if is_q:
            cute.copy(tma_store_q, qBs, qBg[(None, local, 0)])
        else:
            cute.copy(tma_store_k, kBs, kBg[(None, local, 0)])
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0)


def _make_launch_tma_qk(nqh, nkh, hd_half, vec, seq_inner, cos_bcast, backward, seq_len):
    head_dim = 2 * hd_half
    tr, vr = _TMA_TR, _TMA_VR
    tcols = hd_half // vec
    block = tr * tcols

    @cute.jit
    def launch(mQ, mK, mCos, mSin, RRq: cutlass.Int32, RRk: cutlass.Int32, stream: cuda.CUstream = None):
        thr = cute.make_layout((tr, tcols), stride=(tcols, 1))
        val = cute.make_layout((vr, vec), stride=(vec, 1))
        dtype = mQ.element_type
        sl = cute.make_layout((_TMA_TILE_M, head_dim), stride=(head_dim, 1))

        @cute.struct
        class SS:
            bar: cute.struct.MemRange[cutlass.Int64, 1]
            data: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sl)], 128]

        nbytes = cute.size_in_bytes(dtype, sl)
        tiler = (_TMA_TILE_M, head_dim)
        lq, sq = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mQ, sl, tiler)
        sq_atom, dq = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mQ, sl, tiler)
        lk, sk = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mK, sl, tiler)
        sk_atom, dk = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mK, sl, tiler)
        g_lq = cute.local_tile(sq, (_TMA_TILE_M, head_dim), (None, None))
        g_dq = cute.local_tile(dq, (_TMA_TILE_M, head_dim), (None, None))
        g_lk = cute.local_tile(sk, (_TMA_TILE_M, head_dim), (None, None))
        g_dk = cute.local_tile(dk, (_TMA_TILE_M, head_dim), (None, None))
        ntiles_q = cute.ceil_div(mQ.shape[0], _TMA_TILE_M)
        ntiles_k = cute.ceil_div(mK.shape[0], _TMA_TILE_M)
        _rope_tma_qk_kernel(
            lq,
            g_lq,
            sq_atom,
            g_dq,
            lk,
            g_lk,
            sk_atom,
            g_dk,
            sl,
            mCos,
            mSin,
            RRq,
            RRk,
            ntiles_q,
            seq_len,
            SS,
            nbytes,
            thr,
            val,
            hd_half,
            nqh,
            nkh,
            vr,
            vec,
            seq_inner,
            cos_bcast,
            backward,
        ).launch(grid=[ntiles_q + ntiles_k, 1, 1], block=[block, 1, 1], stream=stream)

    return launch


def _make_launch_tma_pointers(launch, head_dim, cos_shape, cos_stride, sin_stride):
    @cute.jit
    def pointer_launch(
        q_ptr: cute.Pointer,
        k_ptr: cute.Pointer,
        cos_ptr: cute.Pointer,
        sin_ptr: cute.Pointer,
        rows_q: cutlass.Int32,
        rows_k: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        q = cute.make_tensor(q_ptr, cute.make_layout((rows_q, head_dim), stride=(head_dim, 1)))
        k = cute.make_tensor(k_ptr, cute.make_layout((rows_k, head_dim), stride=(head_dim, 1)))
        cos = cute.make_tensor(cos_ptr, cute.make_layout(cos_shape, stride=cos_stride))
        sin = cute.make_tensor(sin_ptr, cute.make_layout(cos_shape, stride=sin_stride))
        launch(q, k, cos, sin, rows_q, rows_k, stream)

    return pointer_launch


def _tensor_pointer(t, assumed_align):
    return cute.runtime.make_ptr(
        torch2cute_dtype_map[t.dtype], t.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=assumed_align
    )


_INT32_CACHE: dict = {}


def _int32_cached(v: int):
    # RR (row counts) and seq_len are config-constant; constructing a cutlass.Int32
    # costs ~0.8us each. Cache them so the hot launch path reuses the scalar object.
    o = _INT32_CACHE.get(v)
    if o is None:
        if len(_INT32_CACHE) > 64:
            _INT32_CACHE.clear()
        o = cutlass.Int32(v)
        _INT32_CACHE[v] = o
    return o


def _fake_tma_matrix(dtype, head_dim):
    return cute.runtime.make_fake_tensor(
        torch2cute_dtype_map[dtype],
        (cute.sym_int(), head_dim),
        stride=(head_dim, 1),
        assumed_align=16,
    )


def _fake_rotary_table(t):
    return cute.runtime.make_fake_tensor(
        torch2cute_dtype_map[t.dtype],
        tuple(t.shape),
        stride=tuple(t.stride()),
        assumed_align=t.element_size(),
    )


def _apply_qk_tma(q, k, cos3, sin3, nqh, nkh, hd_half, vec, seq_inner, cos_bcast, seq_len, backward):
    """Rotate q and k in a single fused TMA launch (k rides along on q's grid)."""
    device_index = q.device.index
    if torch.cuda.current_device() != device_index:
        with torch.cuda.device(q.device):
            return _apply_qk_tma(q, k, cos3, sin3, nqh, nkh, hd_half, vec, seq_inner, cos_bcast, seq_len, backward)
    head_dim = 2 * hd_half
    use_tvm_ffi = _TVM_FFI_PRESENT
    key = (
        q.dtype,
        k.dtype,
        cos3.dtype,
        sin3.dtype,
        nqh,
        nkh,
        hd_half,
        vec,
        seq_inner,
        cos_bcast,
        backward,
        tuple(cos3.shape),
        tuple(cos3.stride()),
        tuple(sin3.stride()),
        device_index,
        use_tvm_ffi,
        "tma_qk",
    )
    q2d, k2d = q.reshape(-1, head_dim), k.reshape(-1, head_dim)
    if not use_tvm_ffi:
        # Fresh bindings also work when global TVM FFI is enabled: patching a
        # cached native memref does not update its underlying TVM tensor.
        args = (
            _tensor_pointer(q2d, 16),
            _tensor_pointer(k2d, 16),
            _tensor_pointer(cos3, cos3.element_size()),
            _tensor_pointer(sin3, sin3.element_size()),
            _int32_cached(q2d.shape[0]),
            _int32_cached(k2d.shape[0]),
            _cute_stream(),
        )
    fn = _COMPILE_CACHE.get(key)
    if fn is None:
        with _TMA_COMPILE_LOCK:
            fn = _COMPILE_CACHE.get(key)
            if fn is None:
                launch = _make_launch_tma_qk(nqh, nkh, hd_half, vec, seq_inner, cos_bcast, backward, seq_len)
                if use_tvm_ffi:
                    fn = cute.compile(
                        launch,
                        _fake_tma_matrix(q.dtype, head_dim),
                        _fake_tma_matrix(k.dtype, head_dim),
                        _fake_rotary_table(cos3),
                        _fake_rotary_table(sin3),
                        cutlass.Int32(1),
                        cutlass.Int32(1),
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options="--enable-tvm-ffi",
                    )
                else:
                    pointer_launch = _make_launch_tma_pointers(
                        launch, head_dim, tuple(cos3.shape), tuple(cos3.stride()), tuple(sin3.stride())
                    )
                    fn = cute.compile(pointer_launch, *args)
                _COMPILE_CACHE[key] = fn
    if use_tvm_ffi:
        fn(q2d, k2d, cos3, sin3, q2d.shape[0], k2d.shape[0])
    else:
        fn(*args)


# ---------------------------------------------------------------------------
# Public functional API
# ---------------------------------------------------------------------------
def rope_forward(q, k, cos, sin):
    # q,k arrive as (bsz, n_head, seq, hd) but are physically stored as
    # (bsz, seq, n_head, hd) (they come from a projection's .transpose(1,2)).
    # Transposing back exposes that NATIVE contiguous storage for free, so the
    # following .contiguous() is a no-op for the standard layout -- exactly like
    # the Triton kernel, avoiding an 80MB transpose copy. We then run the fast
    # token-per-CTA kernel (cos/sin once per token, q+k in one launch).
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    hd_half = head_dim // 2

    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    # The vision rotary helper (liger_rotary_pos_emb_vision) passes 2-D
    # (seq_len, head_dim) tables; add a leading batch dim so the reshape and the
    # cos_bcast broadcast path below apply, matching the Triton reference.
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    cos_batch_size = cos.shape[0]
    cos_bcast = cos_batch_size == 1

    cos3 = cos.view(cos_batch_size, seq_len, head_dim)
    sin3 = sin.view(cos_batch_size, seq_len, head_dim)

    # storage is (bsz, seq, n_head, hd): token=(b, s, :, :) -> seq NOT innermost.
    if _tma_supported(q.dtype, head_dim, q.device):
        vec = max(1, 128 // torch.finfo(q.dtype).bits)
        _apply_qk_tma(q, k, cos3, sin3, n_q_head, n_kv_head, hd_half, vec, False, cos_bcast, seq_len, False)
    else:
        _apply_token(q, k, cos3, sin3, n_q_head, n_kv_head, hd_half, False, cos_bcast, seq_len, False)

    return q.transpose(1, 2), k.transpose(1, 2), cos, sin


def rope_backward(dq, dk, cos, sin):
    # dq,dk arrive as (bsz, n_head, seq, hd) -- grads of the forward outputs.
    #
    # Fast path: an already-contiguous grad (eager attn, or a .contiguous() after
    # RoPE) is head-major/seq-innermost, so rotate it in place via SEQ_INNER
    # (kernel derives seq as row % seq_len) -- no copy, ~2-5x faster backward.
    #
    # Standard path (SDPA): grad is a transpose view of (bsz, seq, n_head, hd)
    # storage. Return it as a transpose view of that same storage so AccumulateGrad
    # stays coalesced (a head-major grad would make add_ ~2.9x slower). Like Triton.
    batch_size, n_q_head, seq_len, head_dim = dq.shape
    n_kv_head = dk.shape[1]
    hd_half = head_dim // 2

    cos = cos.contiguous()
    sin = sin.contiguous()
    # 2-D (seq_len, head_dim) rotary tables (vision RoPE) -> add the batch dim.
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos_batch_size = cos.shape[0]
    cos_bcast = cos_batch_size == 1
    cos3 = cos.view(cos_batch_size, seq_len, head_dim)
    sin3 = sin.view(cos_batch_size, seq_len, head_dim)

    if dq.is_contiguous() and dk.is_contiguous():
        # native (bsz, n_head, seq, hd) storage: seq IS innermost -> SEQ_INNER=True.
        # Both kernels rotate it in place; the token kernel indexes with the tensor's
        # own strides, so this fast path applies even when TMA isn't supported.
        if _tma_supported(dq.dtype, head_dim, dq.device):
            tvec = max(1, 128 // torch.finfo(dq.dtype).bits)
            _apply_qk_tma(dq, dk, cos3, sin3, n_q_head, n_kv_head, hd_half, tvec, True, cos_bcast, seq_len, True)
        else:
            _apply_token(dq, dk, cos3, sin3, n_q_head, n_kv_head, hd_half, True, cos_bcast, seq_len, True)
        return dq, dk

    # transpose(1,2) exposes the (bsz, seq, n_head, hd) storage; .contiguous() is a
    # no-op for the standard transpose-view grad (zero copy) and a safety net so the
    # kernel's flat .view(-1, head_dim) stays valid for any other incoming layout.
    dq = dq.transpose(1, 2).contiguous()
    dk = dk.transpose(1, 2).contiguous()
    if _tma_supported(dq.dtype, head_dim, dq.device):
        tvec = max(1, 128 // torch.finfo(dq.dtype).bits)
        _apply_qk_tma(dq, dk, cos3, sin3, n_q_head, n_kv_head, hd_half, tvec, False, cos_bcast, seq_len, True)
    else:
        _apply_token(dq, dk, cos3, sin3, n_q_head, n_kv_head, hd_half, False, cos_bcast, seq_len, True)

    return dq.transpose(1, 2), dk.transpose(1, 2)


class LigerRopeCuteDSLFunction(torch.autograd.Function):
    """CuteDSL equivalent of ``LigerRopeFunction``.

    Implements the HuggingFace Llama & Mistral RoPE rotation. Numerics match the
    Triton kernel (rotation math is done in fp32).
    """

    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        sin size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        """
        q, k, cos, sin = rope_forward(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        """
        dq size: (bsz, n_q_head, seq_len, head_dim)
        dk size: (bsz, n_kv_head, seq_len, head_dim)
        """
        cos, sin = ctx.saved_tensors
        dq, dk = rope_backward(dq, dk, cos, sin)
        return dq, dk, None, None, None, None


# Expose under the default op name so selecting this backend
# (``LIGER_KERNEL_IMPL=cutedsl``) transparently overrides the Triton
# ``LigerRopeFunction`` in ``liger_kernel.ops``. The descriptive
# ``LigerRopeCuteDSLFunction`` name is kept for the standalone test / head-to-head
# benchmark, which compare it side-by-side with the Triton implementation.
LigerRopeFunction = LigerRopeCuteDSLFunction

__all__ = [
    "LigerRopeFunction",
    "LigerRopeCuteDSLFunction",
    "rope_forward",
    "rope_backward",
]
