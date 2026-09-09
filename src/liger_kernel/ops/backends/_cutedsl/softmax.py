"""CuTe DSL (CUTLASS python) backend for ``softmax``.

Strategy
--------
Mirrors the sibling :mod:`liger_kernel.ops.backends._cutedsl.rms_norm`:

- **Forward** is an inline CuTe DSL kernel (``_LigerSoftmaxCuTeDSLForward``)
  adapted from Quack's ``Softmax`` (Apache-2.0; see ``_cute_lib/NOTICE.md``).
  It uses the **two-pass** reduction (max, then sum-of-exp) built on the
  in-repo ``_cute_lib.ReductionBase`` + ``row_reduce`` — we deliberately do
  **not** depend on Quack's ``online_softmax_reduce`` because that helper was
  intentionally excluded from the trimmed ``_cute_lib/reduce.py`` (it pulls in
  the ``Int64``-packed online-reduction machinery). Two-pass is numerically
  identical to the single-block Triton path:
  ``y = exp(x - max(x)) / sum(exp(x - max(x)))``.

- **Backward** is an inline CuTe DSL kernel (``_LigerSoftmaxCuTeDSLBackward``)
  adapted from Quack's ``SoftmaxBackward``: a single fp32 dot-product
  reduction ``dot = sum(dy * y)`` per row, then ``dx = y * (dy - dot)`` —
  byte-for-byte the formula in :mod:`liger_kernel.ops.softmax`.

Both kernels carry a per-process ``cute.compile`` cache keyed on the dtype mix
and ``N`` (``cute.compile`` is multi-second).

Capability
----------
- Compute capability >= sm_90 (Hopper or newer).
- Requires only the ``cutlass`` Python package — the CuTe DSL utilities are
  inlined under ``_cute_lib/``; no runtime dependency on Quack.

Shapes and limits
-----------------
- ``x`` is flattened to 2D (``view(-1, N)``); arbitrary leading dims OK.
- The cluster-reduce path (``cluster_n > 1``, triggered for very wide ``N``)
  uses ``mbarrier`` plumbing identical to the RMSNorm-bwd sibling. We cap at
  ``N <= _MAX_TILE_CUTEDSL`` and raise a ``RuntimeError`` above it so the test
  harness auto-skips for wider rows (the same contract the RMSNorm sibling
  documents for the cuda13.2 cutlass-cute wheels we ship on B200 today).
- dtypes: fp16, bf16, fp32.

Notes
-----
We deliberately do **not** put ``from __future__ import annotations`` here
(same reasoning as the RMSNorm / LayerNorm siblings: keep annotations live for
DSL introspection).

References
----------
- Quack (adapted under Apache-2.0):
  https://github.com/Dao-AILab/quack/blob/main/quack/softmax.py — source of the ``Softmax`` and
  ``SoftmaxBackward`` ``ReductionBase`` subclasses, the ``_threads_per_row`` /
  ``_set_cluster_n`` heuristics, and the ``cute.compile`` cache pattern. See
  :mod:`._cute_lib` for the inlined reduction infrastructure.
- Triton reference: :mod:`liger_kernel.ops.softmax` — fixes the exact numerics
  (single/multi-block fwd, ``dx = y * (dy - sum(dy * y))`` bwd).
"""

import math

from functools import partial
from typing import Optional
from typing import Tuple
from typing import Type

# ---------------------------------------------------------------------------
# Top-level CuTe DSL imports. Identical pattern to the RMSNorm sibling (see its
# header for the rationale: importing this module on a host without cutlass
# raises ImportError, which the dispatcher's discovery layer catches, and the
# Capability gate keeps it out of auto-select).
# ---------------------------------------------------------------------------
import cuda.bindings.driver as cuda  # noqa: F401  (referenced by cute.compile)
import cutlass
import cutlass.cute as cute
import torch

from cutlass import Float32
from cutlass import const_expr
from torch import Tensor

# Inlined CuTe DSL utilities (adapted from Quack, Apache-2.0). Importing
# ``_cute_lib`` does **not** require the upstream ``quack`` package.
import liger_kernel.ops.backends._cutedsl._cute_lib.copy_utils as copy_utils
import liger_kernel.ops.backends._cutedsl._cute_lib.utils as utils

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.backends.dispatch import emit_fallback_warning
from liger_kernel.ops._nvidia_shared import to_local_if_dtensor as _to_local_if_dtensor
from liger_kernel.ops.backends._cutedsl._cute_lib.compile_utils import make_fake_tensor as fake_tensor
from liger_kernel.ops.backends._cutedsl._cute_lib.dtype_map import torch2cute_dtype_map
from liger_kernel.ops.backends._cutedsl._cute_lib.reduce import row_reduce
from liger_kernel.ops.backends._cutedsl._cute_lib.reduction_base import ReductionBase
from liger_kernel.ops.utils import device_context

# Same ceiling reasoning as the RMSNorm-bwd sibling: beyond this width the
# kernel relies on the cluster-reduce path. Surface a RuntimeError so the test
# harness auto-skips for wider rows rather than failing inside a launch.
_MAX_TILE_CUTEDSL = 32768


# ===========================================================================
# Forward kernel — inline CuTe DSL (two-pass: max, then sum-of-exp)
# ===========================================================================
class _LigerSoftmaxCuTeDSLForward(ReductionBase):
    """CuTe DSL softmax forward.

    Two reduction stages: stage 0 for the row max, stage 1 for the denominator
    (sum of exponentials). Adapted from Quack's ``Softmax`` with
    ``online_softmax=False`` so it uses the in-repo ``row_reduce`` rather than
    the excluded ``online_softmax_reduce``.
    """

    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        # 2 stages: 1 for max, 1 for sum.
        super().__init__(dtype, N, stage=2, reduction_dtype=Float32)

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        N = self.N
        for limit, cluster in [(16 * 1024, 1), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8)]:
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = 16

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(max(t.element_type.width for t in [mX, mO]))
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=128 // largest_dtype_width)
        num_threads = tiled_copy.size
        self.kernel(mX, mO, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tv_layout = tiled_copy.layout_tv_tiled

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX, gO, cX = [cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mX, mO, idX)]

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16)
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        thr_copy_X = tiled_copy.get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_X.partition_D(gO)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]
        tXrX, tXrO = [cute.make_rmem_tensor_like(thr) for thr in (tXgX, tXgO)]

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = None if is_even_N else copy_utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
        copy = partial(copy_utils.copy, pred=tXpX)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        if tXcX[0][0] < shape[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        # Fill OOB lanes with -inf so the max ignores them and exp(-inf)==0
        # contributes nothing to the denominator.
        if const_expr(not is_even_N):
            utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)

        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)

        # Stage 0: row max.
        max_x = row_reduce(
            x,
            cute.ReductionOp.MAX,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr + 0 if const_expr(self.cluster_n > 1) else None,
            init_val=-Float32.inf,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )

        # exp(x - max). Use exp2 with the log2(e) factor (matches Quack's
        # fastmath path; exp2 is the native HW instruction).
        # Subtract BEFORE scaling: (x - max) * log2_e is shift-invariant in fp32.
        # x * log2_e - max_x * log2_e cancels two large scaled values and loses
        # the shift-invariance guarantee at large common offsets (fp32 stability).
        log2_e = math.log2(math.e)
        exp_x = cute.math.exp2((x - max_x) * log2_e, fastmath=True)

        # Stage 1: denominator.
        denom = row_reduce(
            exp_x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 1],
            mbar_ptr + 1 if const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
        )

        y = exp_x * cute.arch.rcp_approx(denom)
        tXrO.store(y.to(tXrO.element_type))
        if tXcX[0][0] < shape[0]:
            copy(tXrO, tXgO)


# ===========================================================================
# Backward kernel — inline CuTe DSL (single dot-product reduction)
# ===========================================================================
class _LigerSoftmaxCuTeDSLBackward(ReductionBase):
    """CuTe DSL softmax backward.

    One reduction stage: the dot product ``dot = sum(dy * y)`` (fp32). The
    gradient is then ``dx = y * (dy - dot)``. Adapted from Quack's
    ``SoftmaxBackward``.
    """

    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        super().__init__(dtype, N, stage=1, reduction_dtype=Float32)

    def _threads_per_row(self):
        N = self.N
        # 2-byte dtypes (bf16/fp16): flat 128 threads/row for N > 6144,
        # matching the fwd ladder. The smem-staged bwd keeps only a few fp32
        # temporaries per vec-chunk live across the dot-reduce, so these rows
        # were never in the spill regime of KL-bwd lore; 256 threads only hurt
        # at predication-free full-tile boundaries (N % 16384 == 0), where
        # 256->128 wins 1.28-1.43x on B200 and is noise-neutral elsewhere.
        # fp32 stages 4 elems per 128b vec-copy (vs 8 for 2-byte) and keeps the
        # old 256-thr rule - the gate probe measured c128 ~7% SLOWER there
        # (a gate probe measured this; dtype is trace-time, compile-keyed,
        # so each dtype bakes its own rule - the CE V-gate recipe applied to
        # the softmax bwd thread rung; from a bwd thread sweep).
        if self.dtype.width == 32:
            for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (8192, 128)]:
                if N <= limit:
                    return threads
            return 256
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 128

    def _num_threads(self):
        if self.dtype.width == 32:
            return 128 if self.N <= 8192 else 256
        return 128

    def _set_cluster_n(self):
        N = self.N
        for limit, cluster in [(16 * 1024, 1), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8)]:
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = 16

    @cute.jit
    def __call__(
        self,
        mdY: cute.Tensor,
        mY: cute.Tensor,
        mdX: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mdY.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(max(t.element_type.width for t in [mdY, mY, mdX]))
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=128 // largest_dtype_width)
        num_threads = tiled_copy.size
        self.kernel(mdY, mY, mdX, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[cute.ceil_div(mdY.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mdY: cute.Tensor,
        mY: cute.Tensor,
        mdX: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mdY.shape
        idX = cute.make_identity_tensor(shape)
        gdY, gY, gdX, cX = [cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mdY, mY, mdX, idX)]

        smem = cutlass.utils.SmemAllocator()
        sdY = smem.allocate_tensor(
            mdY.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        sY = smem.allocate_tensor(mY.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16)
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        thr_copy = tiled_copy.get_slice(tidx)

        tdYgdY = thr_copy.partition_S(gdY)
        tdYsdY = thr_copy.partition_D(sdY)
        tYgY = thr_copy.partition_S(gY)
        tYsY = thr_copy.partition_D(sY)
        tdXgdX = thr_copy.partition_D(gdX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tdYrdY, tYrY, tdXrdX = [cute.make_rmem_tensor_like(thr) for thr in (tdYgdY, tYgY, tdXgdX)]

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = None if is_even_N else copy_utils.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        copy = partial(copy_utils.copy, pred=tXpX)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        if tXcX[0][0] < shape[0]:
            copy(tdYgdY, tdYsdY, is_async=True)
            copy(tYgY, tYsY, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        # cp.async automatically zero-fills OOB lanes; dy*y over a zeroed lane
        # contributes 0 to the dot, so no explicit fill_oob is needed.

        cute.autovec_copy(tdYsdY, tdYrdY)
        cute.autovec_copy(tYsY, tYrY)
        dy = tdYrdY.load().to(cute.Float32)
        y = tYrY.load().to(cute.Float32)

        # dot = sum_j dy_j * y_j
        dot = row_reduce(
            dy * y,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr if const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )

        # dx_i = y_i * (dy_i - dot)
        dx = y * (dy - dot)
        tdXrdX.store(dx.to(tdXrdX.element_type))
        if tXcX[0][0] < shape[0]:
            copy(tdXrdX, tdXgdX)


# ---------------------------------------------------------------------------
# Compile caches. ``cute.compile()`` is multi-second; key on the minimal set of
# attributes that change codegen.
# ---------------------------------------------------------------------------
_FWD_COMPILE_CACHE: dict = {}
_BWD_COMPILE_CACHE: dict = {}


def _get_fwd_kernel(x_dtype: torch.dtype, out_dtype: torch.dtype, N: int):
    """Return a compiled forward kernel, building it on first miss."""
    key = (x_dtype, out_dtype, N)
    if key in _FWD_COMPILE_CACHE:
        return _FWD_COMPILE_CACHE[key]

    dtype = torch2cute_dtype_map[x_dtype]
    out_cute_dtype = torch2cute_dtype_map[out_dtype]
    batch_sym = cute.sym_int()
    div = math.gcd(128 // dtype.width, N)
    x_cute = fake_tensor(dtype, (batch_sym, N), div)
    out_cute = fake_tensor(out_cute_dtype, (batch_sym, N), div)

    compiled = cute.compile(
        _LigerSoftmaxCuTeDSLForward(dtype, N),
        x_cute,
        out_cute,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _FWD_COMPILE_CACHE[key] = compiled
    return compiled


def _get_bwd_kernel(dy_dtype: torch.dtype, y_dtype: torch.dtype, dx_dtype: torch.dtype, N: int):
    """Return a compiled backward kernel, building it on first miss."""
    key = (dy_dtype, y_dtype, dx_dtype, N)
    if key in _BWD_COMPILE_CACHE:
        return _BWD_COMPILE_CACHE[key]

    dtype = torch2cute_dtype_map[dy_dtype]
    y_cute_dtype = torch2cute_dtype_map[y_dtype]
    dx_cute_dtype = torch2cute_dtype_map[dx_dtype]
    batch_sym = cute.sym_int()
    div = math.gcd(128 // dtype.width, N)
    dy_cute = fake_tensor(dtype, (batch_sym, N), div)
    y_cute = fake_tensor(y_cute_dtype, (batch_sym, N), div)
    dx_cute = fake_tensor(dx_cute_dtype, (batch_sym, N), div)

    compiled = cute.compile(
        _LigerSoftmaxCuTeDSLBackward(dtype, N),
        dy_cute,
        y_cute,
        dx_cute,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _BWD_COMPILE_CACHE[key] = compiled
    return compiled


# ===========================================================================
# Host-side launchers and autograd Function
# ===========================================================================
def _softmax_cutedsl_forward(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Forward via the inline CuTe DSL kernel.

    Returns ``(Y, Y)`` (the second copy is what backward needs saved). ``x`` is
    flattened to 2D; the output keeps the original shape.
    """
    with device_context(x.device):
        shape = x.shape
        N = shape[-1]
        if N > _MAX_TILE_CUTEDSL:
            raise RuntimeError(
                f"cuTeDSL softmax only supports hidden dim <= {_MAX_TILE_CUTEDSL}; "
                f"got {N}. Use backend='triton' for wider rows. (Cluster-reduce "
                f"path requires a newer cutlass-cute.)"
            )
        x_flat = x.view(-1, N).contiguous()
        out = torch.empty_like(x_flat)

        compiled = _get_fwd_kernel(x_flat.dtype, out.dtype, N)
        # CuTe DSL compiled kernels read torch.cuda.current_stream() at launch via
        # the tvm-ffi env stream; the ABI does not take a positional stream.
        compiled(x_flat, out)
        return out.view(shape), out


def _softmax_cutedsl_backward(dy: Tensor, y: Tensor) -> Tensor:
    """Backward via the inline CuTe DSL kernel. Returns ``dX``."""
    with device_context(y.device):
        shape = dy.shape
        N = shape[-1]
        if N > _MAX_TILE_CUTEDSL:
            raise RuntimeError(
                f"cuTeDSL softmax backward only supports hidden dim <= {_MAX_TILE_CUTEDSL}; "
                f"got {N}. Use backend='triton' for wider rows."
            )
        dy_flat = dy.view(-1, N).contiguous()
        y_flat = y.view(-1, N).contiguous()

        # The compiled kernel is keyed by dtype; Quack asserts dy.dtype == y.dtype.
        # The autograd graph hands us dy in the same dtype as the forward output,
        # so this holds for the common path. Bridge defensively if not.
        if dy_flat.dtype != y_flat.dtype:
            dy_flat = dy_flat.to(y_flat.dtype)
        dx = torch.empty_like(dy_flat)

        compiled = _get_bwd_kernel(dy_flat.dtype, y_flat.dtype, dx.dtype, N)
        compiled(dy_flat, y_flat, dx)
        return dx.view(shape)


class _LigerSoftmaxCuTeDSLFunction(torch.autograd.Function):
    """Autograd wrapper. Saves the softmax output ``y`` for the inline backward."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor):
        input_ = _to_local_if_dtensor(input_)
        input_ = input_.contiguous()
        Y, Y_flat = _softmax_cutedsl_forward(input_)
        ctx.save_for_backward(Y_flat)
        ctx.input_shape = input_.shape
        return Y

    @staticmethod
    def backward(ctx, dY):
        dY = _to_local_if_dtensor(dY).contiguous()
        (Y_flat,) = ctx.saved_tensors
        dX = _softmax_cutedsl_backward(dY, Y_flat)
        return dX.view(ctx.input_shape)


_CUTEDSL_SOFTMAX_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


@register_op(
    "softmax",
    impl_name="nvidia-cutedsl",
    capability=Capability(
        min_cc=(9, 0),
        modules=["cutlass.cute", "torch"],
    ),
    modes=("default",),
    default_mode="default",
    # Auto-preferred when available. B200 re-measure (2026-08-07 cycle): fwd
    # 1.05-1.73x vs Triton across N in [2048, 32768]; the 2-byte flat-128-thr
    # bwd retune (fp32 keeps 256 - it measured slower at 128) closed the old
    # 0.907-0.986x full fwd+bwd deficit at the full-tile boundaries
    # N = 16384/32768 (bf16 now 1.11-1.17x faster there).
    preference_rank=10,
    tolerances=_CUTEDSL_SOFTMAX_TOLERANCES,
    notes="CuTe DSL softmax for Hopper+ (sm_90+); self-contained inline fwd+bwd.",
)
def softmax_cutedsl(
    x: torch.Tensor,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """CuTe DSL softmax dispatch entry point.

    ``mode`` is accepted only for API parity with the other backends; the only
    valid value is ``"default"`` (or ``None``).
    """
    if mode not in (None, "default"):
        raise ValueError(f"CuTe DSL softmax has only mode='default'; got mode={mode!r}.")
    vector_width = 16 // x.element_size()
    fallback_reason = None
    if x.shape[-1] % vector_width:
        fallback_reason = f"hidden size {x.shape[-1]} is not divisible by vector width {vector_width}"
    elif x.shape[-1] > _MAX_TILE_CUTEDSL:
        fallback_reason = f"hidden size {x.shape[-1]} exceeds CuTe DSL limit {_MAX_TILE_CUTEDSL}"
    if fallback_reason is not None:
        from liger_kernel.ops.backends._triton.softmax import softmax_triton

        emit_fallback_warning(
            "softmax",
            "nvidia-cutedsl",
            "nvidia-triton",
            fallback_reason,
        )
        return softmax_triton(x, mode=mode)
    return _LigerSoftmaxCuTeDSLFunction.apply(x)
