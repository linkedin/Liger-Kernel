"""CuteDSL (NVIDIA CUTLASS Python DSL) implementation of the SwiGLU activation.

This is a drop-in alternative to the Triton kernels in
``liger_kernel.ops.swiglu``. It computes the element-wise SwiGLU gate

    forward:   c  = silu(a * gate_multiplier) * b
    backward:  da = dc * (silu(a') * (1 - sig(a')) + sig(a')) * b * gate_multiplier
               db = dc * silu(a')          with a' = a * gate_multiplier

where ``silu(x) = x * sigmoid(x)``.

Design notes (tuned on B200 / sm_100):
  * The problem is purely memory-bound, so the kernel is a flat 1-D element-wise
    pass that issues a single 128-bit (vectorized) load/store per thread per
    tensor. ``vec = 128 // dtype.width`` elements are processed contiguously.
  * ``sigmoid`` uses the fast ``exp2`` path (SFU) instead of ``exp``; the
    sigmoid math is done in fp32 for numerical parity with the Triton kernel.
  * Two kernel variants are compiled per dtype and cached: a *fast* variant with
    no bounds predication (used when ``numel`` is a multiple of the CTA tile)
    and a *general* predicated variant that handles any ``numel``.
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
import cutlass.cute.math as cute_math

from cutlass.cute.runtime import from_dlpack

from liger_kernel.ops.utils import ensure_contiguous

# log2(e); sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + exp2(-x * LOG2E))
_LOG2E = 1.4426950408889634

# Threads per CTA. 128 maximizes achieved bandwidth for the single-128-bit-load
# pattern on B200 (see benchmark/scripts/benchmark_swiglu_cutedsl.py).
_NUM_THREADS = 128


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
# Host (jit) launchers
# ---------------------------------------------------------------------------
def _make_fwd(vec: int, predicated: bool):
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

        _swiglu_fwd_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout, gate_mult, predicated).launch(
            grid=[cute.size(gC, mode=[1]), 1, 1],
            block=[cute.size(tv_layout, mode=[0]), 1, 1],
        )

    return fwd


def _make_bwd(vec: int, predicated: bool):
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

        _swiglu_bwd_kernel(gDC, gA, gB, cC, mA.shape, thr_layout, val_layout, gate_mult, predicated).launch(
            grid=[cute.size(gA, mode=[1]), 1, 1],
            block=[cute.size(tv_layout, mode=[0]), 1, 1],
        )

    return bwd


# (dtype, vec, predicated, "fwd"|"bwd") -> compiled callable
_COMPILE_CACHE: dict = {}


def _vec_for_dtype(dtype: torch.dtype) -> int:
    """Number of elements in a 128-bit vectorized access for ``dtype``."""
    return max(1, 128 // (torch.finfo(dtype).bits))


def _dyn(t: torch.Tensor):
    # ``from_dlpack`` refuses tensors that require grad; the kernels operate on
    # raw storage inside ``autograd.Function`` so detaching is safe.
    return from_dlpack(t.detach()).mark_layout_dynamic()


def _get_compiled(kind: str, ref: torch.Tensor, vec: int, predicated: bool):
    key = (ref.dtype, vec, predicated, kind)
    fn = _COMPILE_CACHE.get(key)
    if fn is not None:
        return fn
    gm = cutlass.Float32(1.0)
    if kind == "fwd":
        fn = cute.compile(_make_fwd(vec, predicated), _dyn(ref), _dyn(ref), _dyn(ref), gm)
    else:
        fn = cute.compile(_make_bwd(vec, predicated), _dyn(ref), _dyn(ref), _dyn(ref), gm)
    _COMPILE_CACHE[key] = fn
    return fn


# ---------------------------------------------------------------------------
# Public functional API
# ---------------------------------------------------------------------------
def swiglu_forward(a, b, gate_multiplier: float = 1.0):
    ori_shape = a.shape
    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols).reshape(-1)
    b = b.view(-1, n_cols).reshape(-1)
    c = torch.empty_like(a)

    numel = a.numel()
    vec = _vec_for_dtype(a.dtype)
    tile = _NUM_THREADS * vec
    predicated = (numel % tile) != 0

    fn = _get_compiled("fwd", a, vec, predicated)
    fn(_dyn(a), _dyn(b), _dyn(c), cutlass.Float32(float(gate_multiplier)))

    return a.view(-1, n_cols), b.view(-1, n_cols), c.view(*ori_shape)


def swiglu_backward(a, b, dc, gate_multiplier: float = 1.0):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols).reshape(-1)
    a = a.view(-1, n_cols).reshape(-1)
    b = b.view(-1, n_cols).reshape(-1)

    numel = dc.numel()
    vec = _vec_for_dtype(dc.dtype)
    tile = _NUM_THREADS * vec
    predicated = (numel % tile) != 0

    fn = _get_compiled("bwd", dc, vec, predicated)
    fn(_dyn(dc), _dyn(a), _dyn(b), cutlass.Float32(float(gate_multiplier)))

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
                else torch.distributed.tensor.distribute_tensor(dc, device_mesh=device_mesh, placements=placements)
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
