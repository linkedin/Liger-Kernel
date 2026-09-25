"""CuTe DSL (CUTLASS python) backend for ``geglu``.

Strategy
--------
GeGLU is a purely elementwise op (no reductions), structurally identical to
SwiGLU but with a GELU (tanh approximation) activation instead of SiLU.  The
math (matching :mod:`liger_kernel.ops.geglu`)::

    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c       = gelu(a) * b

The CuTe DSL kernel mirrors the SwiGLU sibling (:mod:`._cutedsl.swiglu`)
exactly — same TiledCopy + ``cp.async`` load/store pattern — with only the
activation math swapped.  The tanh is computed via the identity
``tanh(x) = 2 * sigmoid(2x) - 1`` using the native ``exp2`` HW instruction
(same ``rcp_approx`` path as the SwiGLU sigmoid), avoiding any external tanh
intrinsic.

Capability
----------
- Compute capability >= sm_90 (Hopper or newer), matching the swiglu sibling.
- Requires only the ``cutlass`` Python package.

Shapes and limits
-----------------
- ``a`` and ``b`` are flattened to 2D ``(M, N)``; arbitrary leading dims OK.
- dtypes: fp16, bf16, fp32.

References
----------
- Triton reference: :mod:`liger_kernel.ops.geglu` — fixes the exact numerics
  (fp32 tanh, GELU tanh approximation constants).
- CuTe DSL structure: :mod:`._cutedsl.swiglu` — the TiledCopy + autovec_copy
  load/store pattern.
"""

import math

from functools import partial
from typing import Optional
from typing import Tuple
from typing import Type

# ---------------------------------------------------------------------------
# Top-level CuTe DSL imports. Identical pattern to the swiglu sibling.
# ---------------------------------------------------------------------------
import cuda.bindings.driver as cuda  # noqa: F401  (referenced by cute.compile)
import cutlass
import cutlass.cute as cute
import torch

from cutlass import const_expr
from torch import Tensor

# Inlined CuTe DSL utilities (adapted from Quack, Apache-2.0).
import liger_kernel.ops.backends._cutedsl._cute_lib.copy_utils as copy_utils

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.ops._nvidia_shared import to_local_if_dtensor as _to_local_if_dtensor
from liger_kernel.ops.backends._cutedsl._cute_lib.compile_utils import make_fake_tensor as fake_tensor
from liger_kernel.ops.backends._cutedsl._cute_lib.dtype_map import torch2cute_dtype_map

# sqrt(2 / pi) — the GELU tanh approximation constant.
_SQRT_2_OVER_PI = 0.7978845608028654
# 0.044715 — the GELU tanh approximation cubic coefficient.
_GELU_CUBIC_COEFF = 0.044715


def _cutedsl_tanh(x):
    """Compute tanh through the vector-aware fast-math MUFU path."""
    return cute.math.tanh(x, fastmath=True)


def _cutedsl_gelu(x):
    """GELU tanh approximation: ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``."""
    x_cubed = x * x * x
    tanh_arg = _SQRT_2_OVER_PI * (x + _GELU_CUBIC_COEFF * x_cubed)
    tanh_result = _cutedsl_tanh(tanh_arg)
    return 0.5 * x * (1.0 + tanh_result)


def _cutedsl_gelu_grad(x):
    """Gradient of the GELU tanh approximation w.r.t. x.

    ``gelu'(x) = 0.5 * (1 + tanh(z)) + 0.5 * x * (1 - tanh(z)^2) * (sqrt(2/pi) * (1 + 3 * 0.044715 * x^2))``
    where ``z = sqrt(2/pi) * (x + 0.044715 * x^3)``.
    """
    x_sq = x * x
    x_cubed = x_sq * x
    tanh_arg = _SQRT_2_OVER_PI * (x + _GELU_CUBIC_COEFF * x_cubed)
    tanh_result = _cutedsl_tanh(tanh_arg)
    tanh_sq = tanh_result * tanh_result
    term1 = 0.5 * (1.0 + tanh_result)
    term2 = 0.5 * x * (1.0 - tanh_sq) * (_SQRT_2_OVER_PI * (1.0 + 3.0 * _GELU_CUBIC_COEFF * x_sq))
    return term1 + term2


# ===========================================================================
# Forward kernel — elementwise: c = gelu(a) * b
# ===========================================================================
class _LigerGeGLUCuTeDSLForward:
    """CuTe DSL GeGLU forward.

    Pure elementwise: load a tile of ``a`` and ``b`` into registers, compute
    ``gelu(a) * b`` in fp32, store the result.  Identical structure to the
    SwiGLU forward kernel.
    """

    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        self.dtype = dtype
        self.N = N
        self.cluster_n = 1

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        # 128 threads/row for ALL N > 16384 (merging with the mid-N rung):
        # paired with num_threads == 128 (1 row per block) and the 2-chunk
        # grid-y cap, mirrored verbatim from the committed swiglu retune
        # (2a956b1), where a B200 tile sweep beat BOTH the 1024-thread cap-2
        # rung and Triton's lean Blackwell tiled path at every N >= 8192
        # (swiglu kernel med ms @32768: 0.2454 at 128thr vs 0.3127 prior /
        # 0.2554 Triton). All-elementwise math => the retile is bit-identical
        # (see _get_tiled_copy; an A/B retime measurement showed the
        # same whole-row-spill bwd win as swiglu @14336/16384).
        return 128

    def _num_threads(self):
        # 1 row per block on every mid/wide rung (N>6144): threads_per_row
        # is already 128 for 6144<N<=16384, and the N>16384 rung now uses
        # 128 threads/row as well. Small-N rungs keep their tpr<128 fanout.
        return 128

    def _get_tiled_copy(self, vecsize: int = 1):
        threads_per_row = self._threads_per_row()
        num_threads = self._num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0
        if self.N >= 8192:
            # Cap the N-chunks each block loops over at 2 and move the rest
            # onto grid-y (see __call__ and kernel). Now applies to the whole
            # N>=8192 rung (8192 itself included after the B200 gate-boundary
            # A/B mirroring swiglu 4cc3075: the uncapped 8-chunk whole-row
            # loop = 64 live fp32 elems/thread measured ~11% (bf16,
            # 1.11-1.12x full fwd+bwd @M=8192 and 16384) / ~12% (fp16, 1.13x
            # @M=4096) slower than cap-2 at N=8192; fp32's vecsize-4
            # 32-elem/thread path re-routes neutral at 1.004x), not just
            # N>16384: the uncapped whole-row loop spilled registers at
            # mid-N, exactly as in the swiglu retune (2a956b1). All math is
            # elementwise per element -- no cross-N reduction -- so
            # N-blocking is bit-identical (16 elems/thread live at cap=2 x
            # vecsize=8 for bf16). N < 8192 keeps the small-rung fanout tile.
            num_blocks_N = min(cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n), 2)
        else:
            num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n)
        tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = copy_utils.tiled_copy_2d(self.dtype, threads_per_row, num_threads, vecsize)
        return tiled_copy, tiler_mn, threads_per_row

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mA.element_type == self.dtype
        largest_dtype_width = const_expr(max(t.element_type.width for t in [mA, mB, mC]))
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, _ = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        self.kernel(mA, mB, mC, tiler_mn, tiled_copy).launch(
            grid=[cute.ceil_div(mA.shape[0], tiler_mn[0]), cute.ceil_div(self.N, tiler_mn[1]), 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bnidx, _ = cute.arch.block_idx()

        shape = mA.shape
        idX = cute.make_identity_tensor(shape)
        gA, gB, gC, cX = [cute.local_tile(mT, tiler_mn, (bidx, bnidx)) for mT in (mA, mB, mC, idX)]

        thr_copy = tiled_copy.get_slice(tidx)
        tAgA = thr_copy.partition_S(gA)
        tBgB = thr_copy.partition_S(gB)
        tCgC = thr_copy.partition_D(gC)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tArA, tBrB, tCrC = [cute.make_rmem_tensor_like(thr) for thr in (tAgA, tBgB, tCgC)]

        is_even_N = const_expr(shape[1] % tiler_mn[1] == 0)
        tXpX = None if is_even_N else copy_utils.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        copy = partial(copy_utils.copy, pred=tXpX)

        if tXcX[0][0] < shape[0]:
            copy(tAgA, tArA)
            copy(tBgB, tBrB)
        a = tArA.load().to(cute.Float32)
        b = tBrB.load()
        gelu_a = _cutedsl_gelu(a)
        c = gelu_a.to(b.element_type) * b
        tCrC.store(c)
        if tXcX[0][0] < shape[0]:
            copy(tCrC, tCgC)


# ===========================================================================
# Backward kernel — elementwise: da, db from dc, a, b
# ===========================================================================
class _LigerGeGLUCuTeDSLBackward:
    """CuTe DSL GeGLU backward.

    Given ``dc``, ``a``, ``b``, compute::

        gelu_a  = gelu(a)
        db      = dc * gelu_a
        da      = dc * b * gelu'(a)
    """

    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        self.dtype = dtype
        self.N = N
        self.cluster_n = 1

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        # 128 threads/row for ALL N > 16384 (merging with the mid-N rung):
        # paired with num_threads == 128 (1 row per block) and the 2-chunk
        # grid-y cap, mirrored verbatim from the committed swiglu retune
        # (2a956b1), where a B200 tile sweep beat BOTH the 1024-thread cap-2
        # rung and Triton's lean Blackwell tiled path at every N >= 8192
        # (swiglu kernel med ms @32768: 0.2454 at 128thr vs 0.3127 prior /
        # 0.2554 Triton). All-elementwise math => the retile is bit-identical
        # (see _get_tiled_copy; an A/B retime measurement showed the
        # same whole-row-spill bwd win as swiglu @14336/16384).
        return 128

    def _num_threads(self):
        # 1 row per block on every mid/wide rung (N>6144): threads_per_row
        # is already 128 for 6144<N<=16384, and the N>16384 rung now uses
        # 128 threads/row as well. Small-N rungs keep their tpr<128 fanout.
        return 128

    def _get_tiled_copy(self, vecsize: int = 1):
        threads_per_row = self._threads_per_row()
        num_threads = self._num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0
        if self.N >= 8192:
            # Cap the N-chunks each block loops over at 2 and move the rest
            # onto grid-y (see __call__ and kernel). Now applies to the whole
            # N>=8192 rung (8192 itself included after the B200 gate-boundary
            # A/B mirroring swiglu 4cc3075: the uncapped 8-chunk whole-row
            # loop = 64 live fp32 elems/thread measured ~11% (bf16,
            # 1.11-1.12x full fwd+bwd @M=8192 and 16384) / ~12% (fp16, 1.13x
            # @M=4096) slower than cap-2 at N=8192; fp32's vecsize-4
            # 32-elem/thread path re-routes neutral at 1.004x), not just
            # N>16384: the uncapped whole-row loop spilled registers at
            # mid-N, exactly as in the swiglu retune (2a956b1). All math is
            # elementwise per element -- no cross-N reduction -- so
            # N-blocking is bit-identical (16 elems/thread live at cap=2 x
            # vecsize=8 for bf16). N < 8192 keeps the small-rung fanout tile.
            num_blocks_N = min(cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n), 2)
        else:
            num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n)
        tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = copy_utils.tiled_copy_2d(self.dtype, threads_per_row, num_threads, vecsize)
        return tiled_copy, tiler_mn, threads_per_row

    @cute.jit
    def __call__(
        self,
        mDC: cute.Tensor,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mDA: cute.Tensor,
        mDB: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mDC.element_type == self.dtype
        largest_dtype_width = const_expr(max(t.element_type.width for t in [mDC, mA, mB, mDA, mDB]))
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, _ = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        self.kernel(mDC, mA, mB, mDA, mDB, tiler_mn, tiled_copy).launch(
            grid=[cute.ceil_div(mDC.shape[0], tiler_mn[0]), cute.ceil_div(self.N, tiler_mn[1]), 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mDC: cute.Tensor,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mDA: cute.Tensor,
        mDB: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bnidx, _ = cute.arch.block_idx()

        shape = mDC.shape
        idX = cute.make_identity_tensor(shape)
        gDC, gA, gB, gDA, gDB, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, bnidx)) for mT in (mDC, mA, mB, mDA, mDB, idX)
        ]

        thr_copy = tiled_copy.get_slice(tidx)
        tDCgDC = thr_copy.partition_S(gDC)
        tAgA = thr_copy.partition_S(gA)
        tBgB = thr_copy.partition_S(gB)
        tDAgDA = thr_copy.partition_D(gDA)
        tDBgDB = thr_copy.partition_D(gDB)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tDCrDC, tArA, tBrB, tDArDA, tDBrDB = [
            cute.make_rmem_tensor_like(thr) for thr in (tDCgDC, tAgA, tBgB, tDAgDA, tDBgDB)
        ]

        is_even_N = const_expr(shape[1] % tiler_mn[1] == 0)
        tXpX = None if is_even_N else copy_utils.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        copy = partial(copy_utils.copy, pred=tXpX)

        if tXcX[0][0] < shape[0]:
            copy(tDCgDC, tDCrDC)
            copy(tAgA, tArA)
            copy(tBgB, tBrB)
        dc = tDCrDC.load().to(cute.Float32)
        a = tArA.load().to(cute.Float32)
        b = tBrB.load().to(cute.Float32)

        gelu_a = _cutedsl_gelu(a)
        db = dc * gelu_a
        da = dc * b * _cutedsl_gelu_grad(a)

        tDArDA.store(da.to(tDArDA.element_type))
        tDBrDB.store(db.to(tDBrDB.element_type))
        if tXcX[0][0] < shape[0]:
            copy(tDArDA, tDAgDA)
            copy(tDBrDB, tDBgDB)


# ---------------------------------------------------------------------------
# Compile caches. ``cute.compile()`` is multi-second; key on the minimal set
# of attributes that change codegen.
# ---------------------------------------------------------------------------
_FWD_COMPILE_CACHE: dict = {}
_BWD_COMPILE_CACHE: dict = {}


def _get_fwd_kernel(a_dtype: torch.dtype, b_dtype: torch.dtype, c_dtype: torch.dtype, N: int):
    """Return a compiled forward kernel, building it on first miss."""
    key = (a_dtype, b_dtype, c_dtype, N)
    if key in _FWD_COMPILE_CACHE:
        return _FWD_COMPILE_CACHE[key]

    dtype = torch2cute_dtype_map[a_dtype]
    b_cute_dtype = torch2cute_dtype_map[b_dtype]
    c_cute_dtype = torch2cute_dtype_map[c_dtype]
    batch_sym = cute.sym_int()
    div = math.gcd(128 // dtype.width, N)
    a_cute = fake_tensor(dtype, (batch_sym, N), div)
    b_cute = fake_tensor(b_cute_dtype, (batch_sym, N), div)
    c_cute = fake_tensor(c_cute_dtype, (batch_sym, N), div)

    compiled = cute.compile(
        _LigerGeGLUCuTeDSLForward(dtype, N),
        a_cute,
        b_cute,
        c_cute,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _FWD_COMPILE_CACHE[key] = compiled
    return compiled


def _get_bwd_kernel(
    dc_dtype: torch.dtype,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    da_dtype: torch.dtype,
    db_dtype: torch.dtype,
    N: int,
):
    """Return a compiled backward kernel, building it on first miss."""
    key = (dc_dtype, a_dtype, b_dtype, da_dtype, db_dtype, N)
    if key in _BWD_COMPILE_CACHE:
        return _BWD_COMPILE_CACHE[key]

    dtype = torch2cute_dtype_map[dc_dtype]
    a_cute_dtype = torch2cute_dtype_map[a_dtype]
    b_cute_dtype = torch2cute_dtype_map[b_dtype]
    da_cute_dtype = torch2cute_dtype_map[da_dtype]
    db_cute_dtype = torch2cute_dtype_map[db_dtype]
    batch_sym = cute.sym_int()
    div = math.gcd(128 // dtype.width, N)
    dc_cute = fake_tensor(dtype, (batch_sym, N), div)
    a_cute = fake_tensor(a_cute_dtype, (batch_sym, N), div)
    b_cute = fake_tensor(b_cute_dtype, (batch_sym, N), div)
    da_cute = fake_tensor(da_cute_dtype, (batch_sym, N), div)
    db_cute = fake_tensor(db_cute_dtype, (batch_sym, N), div)

    compiled = cute.compile(
        _LigerGeGLUCuTeDSLBackward(dtype, N),
        dc_cute,
        a_cute,
        b_cute,
        da_cute,
        db_cute,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _BWD_COMPILE_CACHE[key] = compiled
    return compiled


# ===========================================================================
# Host-side launchers and autograd Function
# ===========================================================================
def _geglu_cutedsl_forward(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Forward via the inline CuTe DSL kernel.

    Returns ``(a, b, c)`` where ``a`` and ``b`` are the contiguous inputs saved
    for the backward.
    """
    shape = a.shape
    N = shape[-1]
    a_flat = a.view(-1, N).contiguous()
    b_flat = b.view(-1, N).contiguous()
    c_flat = torch.empty_like(a_flat)

    compiled = _get_fwd_kernel(a_flat.dtype, b_flat.dtype, c_flat.dtype, N)
    compiled(a_flat, b_flat, c_flat)
    c = c_flat.view(shape)
    return a_flat, b_flat, c


def _geglu_cutedsl_backward(dc: Tensor, a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """Backward via the inline CuTe DSL kernel.

    Returns ``(da, db)``.
    """
    shape = dc.shape
    N = shape[-1]
    dc_flat = dc.view(-1, N).contiguous()
    a_flat = a.view(-1, N).contiguous()
    b_flat = b.view(-1, N).contiguous()
    da_flat = torch.empty_like(a_flat)
    db_flat = torch.empty_like(b_flat)

    compiled = _get_bwd_kernel(dc_flat.dtype, a_flat.dtype, b_flat.dtype, da_flat.dtype, db_flat.dtype, N)
    compiled(dc_flat, a_flat, b_flat, da_flat, db_flat)

    da = da_flat.view(shape)
    db = db_flat.view(shape)
    return da, db


class _LigerGeGLUCuTeDSLFunction(torch.autograd.Function):
    """Autograd wrapper. Saves ``a`` and ``b`` for the backward."""

    @staticmethod
    def forward(ctx, a, b):
        a = _to_local_if_dtensor(a)
        b = _to_local_if_dtensor(b)
        a = a.contiguous()
        b = b.contiguous()
        a_saved, b_saved, c = _geglu_cutedsl_forward(a, b)
        ctx.save_for_backward(a_saved, b_saved)
        return c

    @staticmethod
    def backward(ctx, dc):
        dc = _to_local_if_dtensor(dc).contiguous()
        a, b = ctx.saved_tensors
        da, db = _geglu_cutedsl_backward(dc, a, b)
        return da, db


# ===========================================================================
# Public registration
# ===========================================================================
_CUTEDSL_GELU_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {
        "atol_fwd": 1e-4,
        "atol_bwd": 1e-4,
        "rtol_fwd": 1e-4,
        "rtol_bwd": 1e-4,
    },  # B300: CuTeDSL fast sigmoid/tanh gives ~1.5e-5 fp32 fwd error (tf32-class); 1e-5 was too strict
}


@register_op(
    "geglu",
    impl_name="nvidia-cutedsl",
    capability=Capability(
        min_cc=(9, 0),
        modules=["cutlass.cute", "torch"],
    ),
    modes=("default",),
    default_mode="default",
    # Auto-preferred when satisfied (lowest rank).  GeGLU is elementwise so the
    # CuTe DSL kernel avoids the Triton launch overhead and uses native exp2
    # for the tanh.
    preference_rank=10,
    tolerances=_CUTEDSL_GELU_TOLERANCES,
    notes="CuTe DSL GeGLU for Hopper+ (sm_90+); self-contained inline fwd+bwd.",
)
def geglu_cutedsl(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """CuTe DSL GeGLU dispatch entry point.

    ``mode`` is accepted only for API parity with the other backends; the only
    valid value is ``"default"`` (or ``None``).
    """
    if mode not in (None, "default"):
        raise ValueError(f"CuTe DSL geglu has only mode='default'; got mode={mode!r}.")
    return _LigerGeGLUCuTeDSLFunction.apply(a, b)
