"""CuteDSL (NVIDIA CUTLASS Python DSL) implementation of the SwiGLU activation.

This is a drop-in alternative to the Triton kernels in
``liger_kernel.ops.swiglu``. It computes the element-wise SwiGLU gate

    forward:   c  = silu(a * gate_multiplier) * b
    backward:  da = dc * (silu(a') * (1 - sig(a')) + sig(a')) * b * gate_multiplier
               db = dc * silu(a')          with a' = a * gate_multiplier

where ``silu(x) = x * sigmoid(x)``.

Design notes (tuned on B200 / sm_100):
  * The problem is memory-bound, so the kernel is a flat element-wise pass that
    issues a single **128-bit (vectorized) load/store** per thread per tensor.
    ``vec = 128 // dtype.width`` contiguous elements are processed per thread.
    Vectorization is obtained with an *ordered* thread/value layout
    (``make_ordered_layout((1, NT), order=(1, 0))`` + ``val=(1, vec)``) and a
    copy atom carrying ``num_bits_per_copy = vec * dtype.width``; this is the
    same recipe the tuned ``rms_norm`` CuteDSL kernel uses and is what lifts
    achieved DRAM throughput past the Triton baseline on B200.
  * The sigmoid math runs in fp32 using Blackwell **packed-f32x2** SFU ops
    (``mul_packed_f32x2`` / ``add_packed_f32x2`` / ``exp_packed_f32x2``, which
    process two fp32 lanes per instruction) plus ``rcp_approx`` for the
    reciprocal. This roughly halves the issued math instructions versus the
    scalar ``1/(1+exp2(-x))`` form, which profiling showed to be issue-bound.
  * The *fast* path compiles **once per dtype** against an abstract
    (fake) tensor with a dynamic batch dimension and is invoked with
    ``--enable-tvm-ffi`` so PyTorch tensors are passed **directly** to the
    compiled function. This removes the per-call ``from_dlpack`` /
    memref-construction host overhead (~26us -> ~4us per launch on B200).
    It is used whenever ``numel`` is a multiple of the CTA tile
    (``_NUM_THREADS * vec``) -- i.e. every realistic workload.
  * A *general* predicated fallback handles arbitrary ``numel`` for odd shapes:
    it uses scalar per-element predicated loads/stores (128-bit vectorized copies
    are incompatible with per-element bounds masking) but still applies
    **packed-f32x2** SFU math when available — the predicated load zeroes OOB
    elements, so pairwise computation on them is safe and the predicated store
    prevents writing them back. When TVM-FFI is available the fallback is also
    compiled with ``--enable-tvm-ffi`` against a 1-D ``sym_int`` fake tensor so
    PyTorch tensors are passed directly (no ``from_dlpack`` overhead); the 1-D
    shape needs no divisibility because the reshape to ``(rows, tile)`` is only
    required by the vectorized copy atom, not by TVM-FFI itself.
  * Backward writes the gradients in place into the saved ``a`` / ``b`` buffers,
    exactly like the Triton kernel, so peak memory matches.

The public API mirrors ``liger_kernel.ops.swiglu``:
``swiglu_forward`` / ``swiglu_backward`` / ``LigerSiLUMulCuteDSLFunction``.
"""

import torch

try:
    import torch.distributed.tensor  # noqa: F401
except ImportError:
    pass

import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as carch
import cutlass.cute.math as cute_math

from cutlass.cute.runtime import from_dlpack

from liger_kernel.ops.backends._cutedsl._cute_lib.compile_utils import make_fake_tensor
from liger_kernel.ops.backends._cutedsl._cute_lib.dtype_map import torch2cute_dtype_map
from liger_kernel.ops.utils import ensure_contiguous

# log2(e); sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + exp2(-x * LOG2E))
_LOG2E = 1.4426950408889634

# Threads per CTA. 128 maximizes achieved bandwidth for the single-128-bit-load
# pattern on B200 (see benchmark/scripts/benchmark_swiglu_cutedsl.py).
_NUM_THREADS = 128

# Whether the fast (tvm-ffi / packed-math / vectorized) path is available. It
# requires the ``apache-tvm-ffi`` package and the Blackwell packed-f32x2 ops.
try:
    _ = (carch.mul_packed_f32x2, carch.add_packed_f32x2, carch.exp_packed_f32x2, carch.rcp_approx)
    import tvm_ffi  # noqa: F401

    _FAST_PATH_AVAILABLE = True
# NOTE: must be a single exception name, not a tuple. The CuteDSL AST preprocessor
# parses this module's source and only handles ``ast.Name`` / bare except handlers
# (``handler.type.id``); a tuple ``except (A, B):`` raises AttributeError at compile.
except Exception:  # pragma: no cover - depends on optional deps / GPU arch
    _FAST_PATH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared device math (Blackwell packed-f32x2 SFU)
# ---------------------------------------------------------------------------
# These helpers are the single source of truth for the packed-f32x2 SwiGLU math.
# They operate pairwise on register fragments (2 fp32 lanes per SFU instruction)
# and are inlined into both the predicated and the vectorized fast-path kernels.
@cute.jit
def _silu_mul_fwd_packed(
    frgA: cute.Tensor,
    frgB: cute.Tensor,
    frgC: cute.Tensor,
    gate_mult: cutlass.Float32,
):
    """SwiGLU forward: ``out = silu(a * gate) * b`` written in place into ``frgC``.

    Safe on the predicated path too: out-of-bounds fragment elements are zeroed
    by the predicated load, so computing on them is harmless and the predicated
    store prevents writing them back.
    """
    g = gate_mult
    out_dtype = frgC.element_type
    n = cute.size(frgA)
    for i in cutlass.range_constexpr(0, n, 2):
        a0 = frgA[i].to(cutlass.Float32)
        a1 = frgA[i + 1].to(cutlass.Float32)
        b0 = frgB[i].to(cutlass.Float32)
        b1 = frgB[i + 1].to(cutlass.Float32)
        # a' = a * gate
        a0, a1 = carch.mul_packed_f32x2((a0, a1), (g, g))
        # sigmoid(a') = 1 / (1 + exp(-a'))
        na0, na1 = carch.mul_packed_f32x2((a0, a1), (-1.0, -1.0))
        e0, e1 = carch.exp_packed_f32x2((na0, na1))
        d0, d1 = carch.add_packed_f32x2((e0, e1), (1.0, 1.0))
        s0 = carch.rcp_approx(d0)
        s1 = carch.rcp_approx(d1)
        # silu = a' * sigmoid; out = silu * b
        si0, si1 = carch.mul_packed_f32x2((a0, a1), (s0, s1))
        r0, r1 = carch.mul_packed_f32x2((si0, si1), (b0, b1))
        frgC[i] = r0.to(out_dtype)
        frgC[i + 1] = r1.to(out_dtype)


@cute.jit
def _silu_mul_bwd_packed(
    frgDC: cute.Tensor,
    frgA: cute.Tensor,
    frgB: cute.Tensor,
    frgDA: cute.Tensor,
    frgDB: cute.Tensor,
    gate_mult: cutlass.Float32,
):
    """SwiGLU backward: grads w.r.t. gate (``frgDA``) and up (``frgDB``).

    Same OOB safety as the forward helper on the predicated path.
    """
    g = gate_mult
    da_dtype = frgDA.element_type
    db_dtype = frgDB.element_type
    n = cute.size(frgA)
    for i in cutlass.range_constexpr(0, n, 2):
        dc0 = frgDC[i].to(cutlass.Float32)
        dc1 = frgDC[i + 1].to(cutlass.Float32)
        a0 = frgA[i].to(cutlass.Float32)
        a1 = frgA[i + 1].to(cutlass.Float32)
        b0 = frgB[i].to(cutlass.Float32)
        b1 = frgB[i + 1].to(cutlass.Float32)
        # a' = a * gate
        a0, a1 = carch.mul_packed_f32x2((a0, a1), (g, g))
        # sigmoid(a')
        na0, na1 = carch.mul_packed_f32x2((a0, a1), (-1.0, -1.0))
        e0, e1 = carch.exp_packed_f32x2((na0, na1))
        d0, d1 = carch.add_packed_f32x2((e0, e1), (1.0, 1.0))
        s0 = carch.rcp_approx(d0)
        s1 = carch.rcp_approx(d1)
        # silu = a' * sig ; db = dc * silu
        si0, si1 = carch.mul_packed_f32x2((a0, a1), (s0, s1))
        db0, db1 = carch.mul_packed_f32x2((dc0, dc1), (si0, si1))
        # t = silu * (1 - sig) + sig
        ms0, ms1 = carch.mul_packed_f32x2((s0, s1), (-1.0, -1.0))
        oms0, oms1 = carch.add_packed_f32x2((ms0, ms1), (1.0, 1.0))
        u0, u1 = carch.mul_packed_f32x2((si0, si1), (oms0, oms1))
        t0, t1 = carch.add_packed_f32x2((u0, u1), (s0, s1))
        # da = dc * t * b * gate
        da0, da1 = carch.mul_packed_f32x2((dc0, dc1), (t0, t1))
        da0, da1 = carch.mul_packed_f32x2((da0, da1), (b0, b1))
        da0, da1 = carch.mul_packed_f32x2((da0, da1), (g, g))
        frgDA[i] = da0.to(da_dtype)
        frgDA[i + 1] = da1.to(da_dtype)
        frgDB[i] = db0.to(db_dtype)
        frgDB[i + 1] = db1.to(db_dtype)


# ---------------------------------------------------------------------------
# Device kernels
# ---------------------------------------------------------------------------
@cute.kernel
def _swiglu_fwd_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,
    shape: cute.Shape,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
    gate_mult: cutlass.Float32,
    PREDICATED: cutlass.Constexpr,
    PACKED_MATH: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None,), bidx)
    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]

    copy_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    copy_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)

    thr_load = cute.make_tiled_copy_tv(copy_load, thr_layout, val_layout).get_slice(tidx)
    thr_store = cute.make_tiled_copy_tv(copy_store, thr_layout, val_layout).get_slice(tidx)

    thrA = thr_load.partition_S(blkA)
    thrB = thr_load.partition_S(blkB)
    thrC = thr_store.partition_S(blkC)

    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)

    if cutlass.const_expr(PREDICATED):
        blkCrd = cC[blk_coord]
        thrCrd = thr_store.partition_S(blkCrd)
        frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
        for i in cutlass.range_constexpr(cute.size(frgPred)):
            frgPred[i] = cute.elem_less(thrCrd[i], shape)
        cute.copy(copy_load, thrA, frgA, pred=frgPred)
        cute.copy(copy_load, thrB, frgB, pred=frgPred)
    else:
        cute.copy(copy_load, thrA, frgA)
        cute.copy(copy_load, thrB, frgB)

    if cutlass.const_expr(PACKED_MATH):
        _silu_mul_fwd_packed(frgA, frgB, frgC, gate_mult)
    else:
        a = frgA.load().to(cutlass.Float32) * gate_mult
        b = frgB.load().to(cutlass.Float32)
        sig = 1.0 / (1.0 + cute_math.exp2(-a * _LOG2E))
        res = (a * sig) * b
        frgC.store(res.to(gC.element_type))

    if cutlass.const_expr(PREDICATED):
        cute.copy(copy_store, frgC, thrC, pred=frgPred)
    else:
        cute.copy(copy_store, frgC, thrC)


@cute.kernel
def _swiglu_bwd_kernel(
    gDC: cute.Tensor,
    gA: cute.Tensor,
    gB: cute.Tensor,
    cC: cute.Tensor,
    shape: cute.Shape,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
    gate_mult: cutlass.Float32,
    PREDICATED: cutlass.Constexpr,
    PACKED_MATH: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None,), bidx)
    blkDC = gDC[blk_coord]
    blkA = gA[blk_coord]
    blkB = gB[blk_coord]

    copy_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    copy_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)

    thr_load = cute.make_tiled_copy_tv(copy_load, thr_layout, val_layout).get_slice(tidx)
    thr_store = cute.make_tiled_copy_tv(copy_store, thr_layout, val_layout).get_slice(tidx)

    thrDC = thr_load.partition_S(blkDC)
    thrA = thr_load.partition_S(blkA)
    thrB = thr_load.partition_S(blkB)
    thrDA = thr_store.partition_S(blkA)
    thrDB = thr_store.partition_S(blkB)

    frgDC = cute.make_fragment_like(thrDC)
    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgDA = cute.make_fragment_like(thrDA)
    frgDB = cute.make_fragment_like(thrDB)

    if cutlass.const_expr(PREDICATED):
        blkCrd = cC[blk_coord]
        thrCrd = thr_load.partition_S(blkCrd)
        frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
        for i in cutlass.range_constexpr(cute.size(frgPred)):
            frgPred[i] = cute.elem_less(thrCrd[i], shape)
        cute.copy(copy_load, thrDC, frgDC, pred=frgPred)
        cute.copy(copy_load, thrA, frgA, pred=frgPred)
        cute.copy(copy_load, thrB, frgB, pred=frgPred)
    else:
        cute.copy(copy_load, thrDC, frgDC)
        cute.copy(copy_load, thrA, frgA)
        cute.copy(copy_load, thrB, frgB)

    if cutlass.const_expr(PACKED_MATH):
        _silu_mul_bwd_packed(frgDC, frgA, frgB, frgDA, frgDB, gate_mult)
    else:
        dc = frgDC.load().to(cutlass.Float32)
        a = frgA.load().to(cutlass.Float32) * gate_mult
        b = frgB.load().to(cutlass.Float32)

        sig = 1.0 / (1.0 + cute_math.exp2(-a * _LOG2E))
        silu = a * sig
        db = dc * silu
        da = dc * (silu * (1.0 - sig) + sig) * b * gate_mult

        frgDA.store(da.to(gA.element_type))
        frgDB.store(db.to(gB.element_type))

    if cutlass.const_expr(PREDICATED):
        cute.copy(copy_store, frgDA, thrA, pred=frgPred)
        cute.copy(copy_store, frgDB, thrB, pred=frgPred)
    else:
        cute.copy(copy_store, frgDA, thrA)
        cute.copy(copy_store, frgDB, thrB)


# ---------------------------------------------------------------------------
# Device kernels - fast path (vectorized 128-bit copies + packed-f32x2 math)
# ---------------------------------------------------------------------------
@cute.kernel
def _swiglu_fwd_vec_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tiled_copy: cute.TiledCopy,
    gate_mult: cutlass.Float32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    blk_coord = ((None, None), bidx)

    thr = tiled_copy.get_slice(tidx)
    thrA = thr.partition_S(gA[blk_coord])
    thrB = thr.partition_S(gB[blk_coord])
    thrC = thr.partition_S(gC[blk_coord])

    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)

    cute.copy(tiled_copy, thrA, frgA)
    cute.copy(tiled_copy, thrB, frgB)

    _silu_mul_fwd_packed(frgA, frgB, frgC, gate_mult)

    cute.copy(tiled_copy, frgC, thrC)


@cute.kernel
def _swiglu_bwd_vec_kernel(
    gDC: cute.Tensor,
    gA: cute.Tensor,
    gB: cute.Tensor,
    tiled_copy: cute.TiledCopy,
    gate_mult: cutlass.Float32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    blk_coord = ((None, None), bidx)

    thr = tiled_copy.get_slice(tidx)
    thrDC = thr.partition_S(gDC[blk_coord])
    thrA = thr.partition_S(gA[blk_coord])
    thrB = thr.partition_S(gB[blk_coord])

    frgDC = cute.make_fragment_like(thrDC)
    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgDA = cute.make_fragment_like(thrA)
    frgDB = cute.make_fragment_like(thrB)

    cute.copy(tiled_copy, thrDC, frgDC)
    cute.copy(tiled_copy, thrA, frgA)
    cute.copy(tiled_copy, thrB, frgB)

    _silu_mul_bwd_packed(frgDC, frgA, frgB, frgDA, frgDB, gate_mult)

    cute.copy(tiled_copy, frgDA, thrA)
    cute.copy(tiled_copy, frgDB, thrB)


# ---------------------------------------------------------------------------
# Host (jit) launchers
# ---------------------------------------------------------------------------
def _make_fwd_vec(vec: int):
    @cute.jit
    def fwd(mA, mB, mC, gate_mult: cutlass.Float32):
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mA.element_type, num_bits_per_copy=vec * mA.element_type.width
        )
        thr_layout = cute.make_ordered_layout((1, _NUM_THREADS), order=(1, 0))
        val_layout = cute.make_layout((1, vec))
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        tiler = (1, _NUM_THREADS * vec)
        gA = cute.zipped_divide(mA, tiler)
        gB = cute.zipped_divide(mB, tiler)
        gC = cute.zipped_divide(mC, tiler)

        _swiglu_fwd_vec_kernel(gA, gB, gC, tiled_copy, gate_mult).launch(
            grid=[cute.size(gC, mode=[1]), 1, 1],
            block=[_NUM_THREADS, 1, 1],
        )

    return fwd


def _make_bwd_vec(vec: int):
    @cute.jit
    def bwd(mDC, mA, mB, gate_mult: cutlass.Float32):
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mA.element_type, num_bits_per_copy=vec * mA.element_type.width
        )
        thr_layout = cute.make_ordered_layout((1, _NUM_THREADS), order=(1, 0))
        val_layout = cute.make_layout((1, vec))
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        tiler = (1, _NUM_THREADS * vec)
        gDC = cute.zipped_divide(mDC, tiler)
        gA = cute.zipped_divide(mA, tiler)
        gB = cute.zipped_divide(mB, tiler)

        _swiglu_bwd_vec_kernel(gDC, gA, gB, tiled_copy, gate_mult).launch(
            grid=[cute.size(gA, mode=[1]), 1, 1],
            block=[_NUM_THREADS, 1, 1],
        )

    return bwd


# ---------------------------------------------------------------------------
# Host (jit) launchers - fallback path (predicated, scalar or packed math)
# ---------------------------------------------------------------------------
def _make_fwd(vec: int, predicated: bool, packed_math: bool):
    @cute.jit
    def fwd(mA, mB, mC, gate_mult: cutlass.Float32):
        thr_layout = cute.make_layout(_NUM_THREADS, stride=vec)
        val_layout = cute.make_layout(vec, stride=1)
        tiler, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        gA = cute.zipped_divide(mA, tiler)
        gB = cute.zipped_divide(mB, tiler)
        gC = cute.zipped_divide(mC, tiler)
        idC = cute.make_identity_tensor(mC.shape)
        cC = cute.zipped_divide(idC, tiler=tiler)

        _swiglu_fwd_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout, gate_mult, predicated, packed_math).launch(
            grid=[cute.size(gC, mode=[1]), 1, 1],
            block=[cute.size(tv_layout, mode=[0]), 1, 1],
        )

    return fwd


def _make_bwd(vec: int, predicated: bool, packed_math: bool):
    @cute.jit
    def bwd(mDC, mA, mB, gate_mult: cutlass.Float32):
        thr_layout = cute.make_layout(_NUM_THREADS, stride=vec)
        val_layout = cute.make_layout(vec, stride=1)
        tiler, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        gDC = cute.zipped_divide(mDC, tiler)
        gA = cute.zipped_divide(mA, tiler)
        gB = cute.zipped_divide(mB, tiler)
        idC = cute.make_identity_tensor(mA.shape)
        cC = cute.zipped_divide(idC, tiler=tiler)

        _swiglu_bwd_kernel(
            gDC, gA, gB, cC, mA.shape, thr_layout, val_layout, gate_mult, predicated, packed_math
        ).launch(
            grid=[cute.size(gA, mode=[1]), 1, 1],
            block=[cute.size(tv_layout, mode=[0]), 1, 1],
        )

    return bwd


# (dtype, vec, predicated, packed_math, "fwd"|"bwd") -> compiled callable
_COMPILE_CACHE: dict = {}
# (dtype, "fwd"|"bwd") -> compiled fast-path callable
_VEC_COMPILE_CACHE: dict = {}
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _vec_for_dtype(dtype: torch.dtype) -> int:
    """Number of elements in a 128-bit vectorized access for ``dtype``."""
    return max(1, 128 // (torch.finfo(dtype).bits))


def _validate_supported_dtype(dtype: torch.dtype):
    if dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"swiglu_cutedsl supports only {_SUPPORTED_DTYPES}; got {dtype}.")


def _dispatch_plan(numel: int, vec: int, tile: int):
    """Single source of truth for kernel selection.

    Returns ``(use_fast, predicated, packed_math)``:
      * ``use_fast``    -- take the vectorized 128-bit + TVM-FFI fast path
      * ``predicated``  -- the general kernel must bounds-mask (numel not tile-aligned)
      * ``packed_math`` -- the general kernel may use Blackwell packed-f32x2 SFU math
    """
    use_fast = _FAST_PATH_AVAILABLE and vec >= 2 and (numel % tile) == 0
    predicated = (numel % tile) != 0
    packed_math = _FAST_PATH_AVAILABLE and vec >= 2
    return use_fast, predicated, packed_math


def _dyn(t: torch.Tensor):
    # ``from_dlpack`` refuses tensors that require grad; the kernels operate on
    # raw storage inside ``autograd.Function`` so detaching is safe.
    return from_dlpack(t.detach()).mark_layout_dynamic()


class _DynCaller:
    """Wraps a non-TVM-FFI compiled function so callers always pass raw tensors."""

    def __init__(self, compiled):
        self._compiled = compiled

    def __call__(self, a, b, c, gm):
        return self._compiled(_dyn(a), _dyn(b), _dyn(c), gm)


def _get_compiled(kind: str, ref: torch.Tensor, vec: int, predicated: bool, packed_math: bool):
    key = (ref.dtype, vec, predicated, packed_math, kind)
    fn = _COMPILE_CACHE.get(key)
    if fn is not None:
        return fn
    gm = cutlass.Float32(1.0)
    maker = _make_fwd(vec, predicated, packed_math) if kind == "fwd" else _make_bwd(vec, predicated, packed_math)
    if _FAST_PATH_AVAILABLE:
        # TVM-FFI: 1-D sym_int fake tensor; PyTorch tensors passed directly — no
        # from_dlpack overhead. Divisibility=1 since the predicated path makes no
        # alignment guarantee beyond element size.
        cute_dtype = torch2cute_dtype_map[ref.dtype]
        fake = make_fake_tensor(cute_dtype, (cute.sym_int(),), 1)
        fn = cute.compile(maker, fake, fake, fake, gm, options="--enable-tvm-ffi")
    else:
        compiled = cute.compile(maker, _dyn(ref), _dyn(ref), _dyn(ref), gm)
        fn = _DynCaller(compiled)
    _COMPILE_CACHE[key] = fn
    return fn


def _get_compiled_vec(kind: str, dtype: torch.dtype, vec: int):
    """Compile (once per dtype) the vectorized + packed-math fast path.

    The kernel is built against an abstract tensor with a dynamic batch
    dimension and a static inner width (``_NUM_THREADS * vec``), and is invoked
    with ``--enable-tvm-ffi`` so PyTorch tensors are passed straight through.
    """
    key = (dtype, kind)
    fn = _VEC_COMPILE_CACHE.get(key)
    if fn is not None:
        return fn

    cute_dtype = torch2cute_dtype_map[dtype]
    inner = _NUM_THREADS * vec
    gm = cutlass.Float32(1.0)

    def fake():
        # dynamic rows, static inner width; ``vec`` divisibility => 128-bit align
        return make_fake_tensor(cute_dtype, (cute.sym_int(), inner), vec)

    if kind == "fwd":
        fn = cute.compile(_make_fwd_vec(vec), fake(), fake(), fake(), gm, options="--enable-tvm-ffi")
    else:
        fn = cute.compile(_make_bwd_vec(vec), fake(), fake(), fake(), gm, options="--enable-tvm-ffi")
    _VEC_COMPILE_CACHE[key] = fn
    return fn


# ---------------------------------------------------------------------------
# Public functional API
# ---------------------------------------------------------------------------
def swiglu_forward(a, b, gate_multiplier: float = 1.0):
    if a.dtype != b.dtype:
        raise TypeError(f"a and b must have the same dtype, got {a.dtype} and {b.dtype}.")
    _validate_supported_dtype(a.dtype)

    ori_shape = a.shape
    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols).reshape(-1)
    b = b.view(-1, n_cols).reshape(-1)
    c = torch.empty_like(a)

    numel = a.numel()
    vec = _vec_for_dtype(a.dtype)
    tile = _NUM_THREADS * vec
    use_fast, predicated, packed_math = _dispatch_plan(numel, vec, tile)

    if use_fast:
        fn = _get_compiled_vec("fwd", a.dtype, vec)
        rows = numel // tile
        fn(
            a.view(rows, tile),
            b.view(rows, tile),
            c.view(rows, tile),
            cutlass.Float32(float(gate_multiplier)),
        )
    else:
        fn = _get_compiled("fwd", a, vec, predicated, packed_math)
        fn(a, b, c, cutlass.Float32(float(gate_multiplier)))

    return a.view(-1, n_cols), b.view(-1, n_cols), c.view(*ori_shape)


def swiglu_backward(a, b, dc, gate_multiplier: float = 1.0):
    if a.dtype != b.dtype or a.dtype != dc.dtype:
        raise TypeError(f"a, b, and dc must have the same dtype, got {a.dtype}, {b.dtype}, and {dc.dtype}.")
    _validate_supported_dtype(dc.dtype)

    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols).reshape(-1)
    a = a.view(-1, n_cols).reshape(-1)
    b = b.view(-1, n_cols).reshape(-1)

    numel = dc.numel()
    vec = _vec_for_dtype(dc.dtype)
    tile = _NUM_THREADS * vec
    use_fast, predicated, packed_math = _dispatch_plan(numel, vec, tile)

    if use_fast:
        fn = _get_compiled_vec("bwd", dc.dtype, vec)
        rows = numel // tile
        fn(
            dc.view(rows, tile),
            a.view(rows, tile),
            b.view(rows, tile),
            cutlass.Float32(float(gate_multiplier)),
        )
    else:
        fn = _get_compiled("bwd", dc, vec, predicated, packed_math)
        fn(dc, a, b, cutlass.Float32(float(gate_multiplier)))

    return a.view(*ori_shape), b.view(*ori_shape)


class LigerSiLUMulCuteDSLFunction(torch.autograd.Function):
    """CuteDSL equivalent of ``LigerSiLUMulFunction``.

    Supports the same ``gate_multiplier`` / ``down_multiplier`` semantics. The
    DTensor handling intentionally mirrors the Triton implementation.
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b, gate_multiplier: float = 1.0, down_multiplier: float = 1.0):
        gate_multiplier = float(gate_multiplier)
        down_multiplier = float(down_multiplier)
        ctx.gate_multiplier = gate_multiplier
        ctx.down_multiplier = down_multiplier

        if isinstance(a, torch.distributed.tensor.DTensor) or isinstance(b, torch.distributed.tensor.DTensor):
            device_mesh, placements = (
                (a.device_mesh, a.placements)
                if isinstance(a, torch.distributed.tensor.DTensor)
                else (b.device_mesh, b.placements)
            )
            if not isinstance(a, torch.distributed.tensor.DTensor):
                a = torch.distributed.tensor.distribute_tensor(a, device_mesh=device_mesh, placements=placements)
            if not isinstance(b, torch.distributed.tensor.DTensor):
                b = torch.distributed.tensor.distribute_tensor(b, device_mesh=device_mesh, placements=placements)
            a_local, b_local, c_local = swiglu_forward(a.to_local(), b.to_local(), gate_multiplier)
            if down_multiplier != 1.0:
                c_local = c_local * down_multiplier
            ctx.save_for_backward(a_local, b_local)
            ctx.dtensor_metadata = (device_mesh, placements)
            return torch.distributed.tensor.DTensor.from_local(c_local, device_mesh, placements)
        else:
            a, b, c = swiglu_forward(a, b, gate_multiplier)
            if down_multiplier != 1.0:
                c = c * down_multiplier
            ctx.save_for_backward(a, b)
            ctx.dtensor_metadata = None
            return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        gate_multiplier = ctx.gate_multiplier
        down_multiplier = ctx.down_multiplier

        if ctx.dtensor_metadata is not None:
            device_mesh, placements = ctx.dtensor_metadata
            dc_local = (
                dc.to_local()
                if isinstance(dc, torch.distributed.tensor.DTensor)
                else torch.distributed.tensor.distribute_tensor(
                    dc, device_mesh=device_mesh, placements=placements
                ).to_local()
            )
            if down_multiplier != 1.0:
                dc_local = dc_local * down_multiplier
            a_local, b_local = swiglu_backward(a, b, dc_local, gate_multiplier)
            return (
                torch.distributed.tensor.DTensor.from_local(a_local, device_mesh, placements),
                torch.distributed.tensor.DTensor.from_local(b_local, device_mesh, placements),
                None,
                None,
            )

        if down_multiplier != 1.0:
            dc = dc * down_multiplier
        a, b = swiglu_backward(a, b, dc, gate_multiplier)
        return a, b, None, None
