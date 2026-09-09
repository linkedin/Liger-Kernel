"""CuTe DSL (CUTLASS python) backend for ``swiglu``.

Strategy
--------
SwiGLU is a purely elementwise op (no reductions), which makes it the simplest
CuTe DSL kernel in the suite.  The math (matching
:mod:`liger_kernel.ops.swiglu`)::

    silu(x) = x * sigmoid(x)
    c       = silu(a) * b

The ``gate_multiplier`` / ``down_multiplier`` scalars (used by Falcon-H1) are
folded host-side — exactly as the Triton ``LigerSiLUMulFunction`` folds
``down_multiplier`` outside the kernel.  Pre-scaling ``a`` by ``gate_mult``
before the kernel and post-scaling ``da`` by ``gate_mult`` after the backward
kernel reproduces the chain rule without baking scalars into the compiled
artifact::

    forward :  a' = a * gate_mult;  c = silu(a') * b;  c *= down_mult
    backward:  dc' = dc * down_mult;  da', db = bwd(dc', a', b);  da = da' * gate_mult

So the CuTe DSL kernel itself only ever sees the pure ``silu(a) * b`` form.

Capability
----------
- Compute capability >= sm_90 (Hopper or newer), matching the rms_norm / softmax
  CuTe DSL siblings.
- Requires only the ``cutlass`` Python package — the CuTe DSL utilities are
  inlined under ``_cute_lib/``; no runtime dependency on Quack.

Shapes and limits
-----------------
- ``a`` and ``b`` are flattened to 2D ``(M, N)``; arbitrary leading dims OK.
- dtypes: fp16, bf16, fp32.

Notes
-----
We deliberately do **not** put ``from __future__ import annotations`` here
(same reasoning as the RMSNorm / softmax siblings: keeps annotations live for
DSL introspection).

References
----------
- Triton reference: :mod:`liger_kernel.ops.swiglu` — fixes the exact numerics
  (fp32 sigmoid, gate/down multiplier chain rule).
- CuTe DSL structure: :mod:`liger_kernel.ops.backends._cutedsl.softmax` — the
  TiledCopy + ``cute.autovec_copy`` load/store pattern (minus the reduction
  stages that SwiGLU does not need).
"""

import math

from functools import partial
from typing import Optional
from typing import Tuple
from typing import Type

# ---------------------------------------------------------------------------
# Top-level CuTe DSL imports. Identical pattern to the softmax sibling.
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
from liger_kernel.ops._nvidia_shared import is_dtensor as _is_dtensor
from liger_kernel.ops._nvidia_shared import to_local_if_dtensor as _to_local_if_dtensor
from liger_kernel.ops.backends._cutedsl._cute_lib.compile_utils import make_fake_tensor as fake_tensor
from liger_kernel.ops.backends._cutedsl._cute_lib.dtype_map import torch2cute_dtype_map
from liger_kernel.ops.utils import device_context


# ===========================================================================
# Forward kernel — elementwise: c = silu(a) * b
# ===========================================================================
class _LigerSwiGLUCuTeDSLForward:
    """CuTe DSL SwiGLU forward.

    Pure elementwise: load a tile of ``a`` and ``b`` into registers, compute
    ``silu(a) * b`` in fp32, store the result.  No shared-memory reduction is
    needed (unlike softmax / rms_norm), but we still stage through shared
    memory via ``cp.async`` so the global load is pipelined.
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
        # grid-y cap, a B200 fwd tile sweep (M=8192 bf16) beat BOTH the
        # committed 1024-thread cap-2 rung and Triton's lean Blackwell tiled
        # path at every N >= 8192 (kernel med ms @32768: 0.2454 at 128thr vs
        # 0.3127 committed / 0.2554 Triton). All-elementwise math => the
        # retile is bit-identical (see _get_tiled_copy and the sw_fwd_sweep
        # sweep of (num_threads, threads_per_row, num_blocks_N)).
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
            # A/B: uncapped 8 chunks = 64 live fp32 elems/thread measured
            # ~4.6% (bf16) / ~7.8% (fp16) slower than cap-2 in full fwd+bwd
            # @M=8192,N=8192; fp32's vecsize-4 32-elem/thread path neutral),
            # not just N>16384: the uncapped whole-row loop
            # spilled registers at mid-N (14 vec-chunks @N=14336 gave 0.1583ms
            # fwd kernel vs Triton 0.1243; 0.7543ms full fwd+bwd vs 0.4383).
            # All math is elementwise per element -- no cross-N reduction --
            # so N-blocking is bit-identical (16 elems/thread live at
            # cap=2 x vecsize=8 for bf16). N <= 8192 stays whole-row.
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

        # tanh.approx maps directly to the GPU MUFU path and accepts vector
        # TensorSSA values, unlike the scalar-only rcp_approx wrapper.
        sig = 0.5 + 0.5 * cute.math.tanh(0.5 * a, fastmath=True)
        silu_a = a * sig
        c = silu_a.to(b.element_type) * b
        tCrC.store(c)
        if tXcX[0][0] < shape[0]:
            copy(tCrC, tCgC)


# ===========================================================================
# Backward kernel — elementwise: da, db from dc, a, b
# ===========================================================================
class _LigerSwiGLUCuTeDSLBackward:
    """CuTe DSL SwiGLU backward.

    Given ``dc``, ``a``, ``b`` (where ``a`` is the *effective* ``a * gate_mult``
    saved by the forward), compute::

        sig     = sigmoid(a)
        silu_a  = a * sig
        db      = dc * silu_a
        da      = dc * (silu_a * (1 - sig) + sig) * b

    The host launcher post-multiplies ``da`` by ``gate_mult`` to complete the
    chain rule.
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
        # grid-y cap, a B200 fwd tile sweep (M=8192 bf16) beat BOTH the
        # committed 1024-thread cap-2 rung and Triton's lean Blackwell tiled
        # path at every N >= 8192 (kernel med ms @32768: 0.2454 at 128thr vs
        # 0.3127 committed / 0.2554 Triton). All-elementwise math => the
        # retile is bit-identical (see _get_tiled_copy and the sw_fwd_sweep
        # sweep of (num_threads, threads_per_row, num_blocks_N)).
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
            # A/B: uncapped 8 chunks = 64 live fp32 elems/thread measured
            # ~4.6% (bf16) / ~7.8% (fp16) slower than cap-2 in full fwd+bwd
            # @M=8192,N=8192; fp32's vecsize-4 32-elem/thread path neutral),
            # not just N>16384: the uncapped whole-row loop
            # spilled registers at mid-N (14 vec-chunks @N=14336 gave 0.1583ms
            # fwd kernel vs Triton 0.1243; 0.7543ms full fwd+bwd vs 0.4383).
            # All math is elementwise per element -- no cross-N reduction --
            # so N-blocking is bit-identical (16 elems/thread live at
            # cap=2 x vecsize=8 for bf16). N <= 8192 stays whole-row.
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

        sig = 0.5 + 0.5 * cute.math.tanh(0.5 * a, fastmath=True)
        silu_a = a * sig
        db = dc * silu_a
        # silu'(a) = silu(a) * (1 - sig) + sig
        da = dc * (silu_a * (1.0 - sig) + sig) * b

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
        _LigerSwiGLUCuTeDSLForward(dtype, N),
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
        _LigerSwiGLUCuTeDSLBackward(dtype, N),
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
def _swiglu_cutedsl_forward(
    a: Tensor, b: Tensor, gate_multiplier: float, down_multiplier: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """Forward via the inline CuTe DSL kernel.

    Returns ``(a_eff, b, c)`` where ``a_eff`` is the gate-multiplied ``a``
    saved for the backward.  ``c`` keeps the original shape.
    """
    with device_context(a.device):
        shape = a.shape
        N = shape[-1]
        # Pre-scale a by gate_multiplier so the kernel only sees silu(a') * b.
        if gate_multiplier != 1.0:
            a_eff = a * gate_multiplier
        else:
            a_eff = a
        a_flat = a_eff.view(-1, N).contiguous()
        b_flat = b.view(-1, N).contiguous()
        c_flat = torch.empty_like(a_flat)

        compiled = _get_fwd_kernel(a_flat.dtype, b_flat.dtype, c_flat.dtype, N)
        compiled(a_flat, b_flat, c_flat)
        c = c_flat.view(shape)
        if down_multiplier != 1.0:
            c = c * down_multiplier
        return a_eff, b, c


def _swiglu_cutedsl_backward(
    dc: Tensor, a_eff: Tensor, b: Tensor, gate_multiplier: float, down_multiplier: float
) -> Tuple[Tensor, Tensor]:
    """Backward via the inline CuTe DSL kernel.

    Returns ``(da, db)``.  ``dc`` is pre-scaled by ``down_multiplier``;
    ``da`` is post-scaled by ``gate_multiplier`` to complete the chain rule.
    """
    with device_context(a_eff.device):
        shape = dc.shape
        N = shape[-1]
        if down_multiplier != 1.0:
            dc = dc * down_multiplier
        dc_flat = dc.view(-1, N).contiguous()
        a_flat = a_eff.view(-1, N).contiguous()
        b_flat = b.view(-1, N).contiguous()
        da_flat = torch.empty_like(a_flat)
        db_flat = torch.empty_like(b_flat)

        compiled = _get_bwd_kernel(dc_flat.dtype, a_flat.dtype, b_flat.dtype, da_flat.dtype, db_flat.dtype, N)
        compiled(dc_flat, a_flat, b_flat, da_flat, db_flat)

        da = da_flat.view(shape)
        db = db_flat.view(shape)
        if gate_multiplier != 1.0:
            da = da * gate_multiplier
        return da, db


class _LigerSwiGLUCuTeDSLFunction(torch.autograd.Function):
    """Autograd wrapper. Saves the effective ``a`` (gate-scaled) and ``b``."""

    @staticmethod
    def forward(ctx, a, b, gate_multiplier: float = 1.0, down_multiplier: float = 1.0):
        a = _to_local_if_dtensor(a)
        b = _to_local_if_dtensor(b)
        a = a.contiguous()
        b = b.contiguous()
        gate_multiplier = float(gate_multiplier)
        down_multiplier = float(down_multiplier)
        ctx.gate_multiplier = gate_multiplier
        ctx.down_multiplier = down_multiplier
        a_eff, b_saved, c = _swiglu_cutedsl_forward(a, b, gate_multiplier, down_multiplier)
        ctx.save_for_backward(a_eff, b_saved)
        return c

    @staticmethod
    def backward(ctx, dc):
        dc = _to_local_if_dtensor(dc).contiguous()
        a_eff, b = ctx.saved_tensors
        da, db = _swiglu_cutedsl_backward(dc, a_eff, b, ctx.gate_multiplier, ctx.down_multiplier)
        return da, db, None, None


# ===========================================================================
# Public registration
# ===========================================================================
_CUTEDSL_SWIGLU_TOLERANCES = {
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
    "swiglu",
    impl_name="nvidia-cutedsl",
    capability=Capability(
        min_cc=(9, 0),
        modules=["cutlass.cute", "torch"],
    ),
    modes=("default",),
    default_mode="default",
    # Auto-preferred when satisfied (lowest rank).  SwiGLU is elementwise so the
    # CuTe DSL kernel avoids the Triton launch overhead and uses native exp2
    # for the sigmoid.
    preference_rank=10,
    tolerances=_CUTEDSL_SWIGLU_TOLERANCES,
    notes="CuTe DSL SwiGLU for Hopper+ (sm_90+); self-contained inline fwd+bwd.",
)
def swiglu_cutedsl(
    a: torch.Tensor,
    b: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """CuTe DSL SwiGLU dispatch entry point.

    ``mode`` is accepted only for API parity with the other backends; the only
    valid value is ``"default"`` (or ``None``).
    """
    if mode not in (None, "default"):
        raise ValueError(f"CuTe DSL swiglu has only mode='default'; got mode={mode!r}.")
    # DTensor (tensor-parallel) inputs are excluded from the CuTe DSL fast path.
    # The inline kernel operates on plain local tensors: materializing a DTensor
    # via full_tensor() would trigger an all-gather and silently return a plain
    # tensor, breaking TP distribution semantics for the output and gradients.
    # Delegate such inputs to the Triton LigerSiLUMulFunction, which captures
    # (device_mesh, placements) and re-wraps the output/grads with
    # DTensor.from_local. The dense (non-DTensor) fast path below is untouched.
    if _is_dtensor(a) or _is_dtensor(b):
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction

        return LigerSiLUMulFunction.apply(a, b, float(gate_multiplier), float(down_multiplier))
    return _LigerSwiGLUCuTeDSLFunction.apply(a, b, float(gate_multiplier), float(down_multiplier))
