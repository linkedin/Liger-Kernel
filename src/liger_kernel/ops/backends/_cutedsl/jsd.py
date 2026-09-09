"""CuTe DSL (CUTLASS python) backend for the ``jsd_loss_and_grad`` primitive.

The ``jsd_loss_and_grad`` primitive is the non-autograd inner kernel used by
:class:`liger_kernel.ops.fused_linear_jsd.LigerFusedLinearJSDFunction`.  It
computes the per-element JSD loss and gradient for one chunk of logits,
returning ``(loss, dx)`` where ``loss`` is a ``(BT, V)`` fp32 tile (summed
elsewhere) and ``dx`` overwrites the student log-probabilities.

Strategy
--------
The JSD computation is **element-wise** (no row reduction is needed inside
this primitive — the caller sums the loss tile).  This makes the CuTe DSL
kernel structurally identical to the SwiGLU / GeGLU elementwise kernels:
load tiles of ``X`` (student log-prob) and ``Y`` (teacher log-prob) into
registers, compute the loss + gradient in fp32, store both outputs.

Math (matching :mod:`liger_kernel.ops.jsd._jsd_kernel`)::

    Q = exp(X)                           # student probabilities
    P = exp(Y)                           # teacher probabilities

    # General JSD (0 < beta < 1):
    M       = beta * P + (1 - beta) * Q
    log_M   = log(M)
    loss    = beta * P * Y + (1 - beta) * Q * X - M * log_M
    dX      = (1 - beta) * Q * (X - log_M)

    # Forward KL (beta == 0):
    loss    = P * (Y - X)
    dX      = -P

    # Reverse KL (beta == 1):
    loss    = Q * (X - Y)
    dX      = loss + Q

    loss *= (1 / n_non_ignore)
    dX   *= (1 / n_non_ignore)

The max-subtraction for numerical stability in the Triton kernel is omitted
here — log-probabilities are typically in ``[-30, 0]`` so ``exp`` does not
overflow.  For bf16/fp16 inputs ``exp``/``log`` are computed via native
``exp2``/``log2`` (``exp(x) = exp2(x * log2(e))``); for fp32 inputs the kernel
switches to libdevice-precise ``exp``/``log`` (no fastmath) to match
``torch.exp`` fp32 within the strict public JSD contract (1e-7 atol).

Rows where ``label == ignore_index`` are zeroed in-kernel: the label stream is
threaded through the CuTe DSL kernel as a compile-keyed ``Optional`` operand,
and ignored rows are ``fill(0.0)``-ed before the gmem store, so no host-side
``nonzero()``/scatter and no device->host sync are needed. Kernels compiled for
``shift_labels is None`` or packed batches (``n_non_ignore == BT``) pass
``None`` for the label operand and dead-code-eliminate the zero-fill.

Capability
----------
- Compute capability >= sm_90 (Hopper or newer).
- Requires only the ``cutlass`` Python package.

References
----------
- Triton reference: :mod:`liger_kernel.ops.jsd._jsd_kernel`
- Triton primitive: :mod:`liger_kernel.ops.backends._triton.jsd.jsd_loss_and_grad_triton`
- CuTe DSL elementwise: :mod:`._cutedsl.swiglu`
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

from cutlass import Float32
from cutlass import Int64
from cutlass import const_expr
from torch import Tensor

# Inlined CuTe DSL utilities (adapted from Quack, Apache-2.0).
import liger_kernel.ops.backends._cutedsl._cute_lib.copy_utils as copy_utils

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.backends.dispatch import emit_fallback_warning
from liger_kernel.ops._nvidia_shared import to_local_if_dtensor as _to_local_if_dtensor  # noqa: F401
from liger_kernel.ops.backends._cutedsl._cute_lib.compile_utils import make_fake_tensor as fake_tensor
from liger_kernel.ops.backends._cutedsl._cute_lib.dtype_map import torch2cute_dtype_map
from liger_kernel.ops.backends._triton.jsd import jsd_loss_and_grad_triton
from liger_kernel.ops.jsd import LigerJSDFunction


def _max_capability() -> int:
    # Runtime dispatch-plane key (e.g. 100 = sm_100/B200, 103 = sm_103/B300).
    # Called per wrapper invocation (never memoized) so mock-cc suites key
    # separately per call, same idiom as capability.is_satisfied.
    if torch.cuda.is_available():
        try:
            maj, minor = torch.cuda.get_device_capability()
        except (IndexError, RuntimeError, AssertionError):  # pragma: no cover
            return 0
        return 10 * maj + minor
    return 0


# Dispatch-plane preference ranks (auto-select picks the LOWEST rank among
# available impls -- backends/dispatch.py resolution rule 5).
#
# On Hopper and B200 (sm_90/sm_100) this file's impls stay at rank 20, which is
# what they have always been (cuTile is Blackwell-only: min_cc=(10,0), so on
# B200 cuTile rank 5 has silently won auto ahead of this 20 since the cutile
# backend landed; B200's cuTile-vs-cutedsl split was never mapped and stays
# untouched by the plane doctrine).
#
# On sm_103/B300 the end-to-end pinned map (
# full fwd+bwd through LigerJSDFunction.apply with jsd_impl pinned, ABAB x3,
# per-iter-sync CUDA-event medians) measured this file's kernels FASTER than
# the auto-winning cuTile at every mapped cell -- bf16 2.27-3.55x, fp16
# 2.22-3.08x, fp32 1.25-1.33x (V 8192..128256) -- EXCEPT one isolated fp32
# V==16384 cliff cell where cutedsl runs at 0.1997x of cuTile (also 3.3-5.3x
# behind Triton; the in-kernel wide-tile re-route was REFUTED at ratio ~1.0,
# so the fp32 band at exactly 16384 is off-roofline for the precise-libdevice
# kernels on sm_103). Rank 2 (below cuTile's 5) fixes the routing; the single
# losing cell is handed to Triton (== cuTile within 1.6% pinned at the band,
# a pinned band probe) by the _B300_FP32_SIDESTEP_VOCAB guard in
# jsd_loss_and_grad_cutedsl below.
_RANK_HOPPER_B200 = 20
_RANK_B300 = 2

_B300_FP32_SIDESTEP_VOCAB = 16384


# log2(e) and ln(2) for exp2/log2 conversions (native HW instructions).
_LOG2_E = math.log2(math.e)
_LN2 = math.log(2.0)


def _cutedsl_exp_fast(x):
    """exp via native exp2: ``exp(x) = exp2(x * log2(e))`` (``ex2.approx``)."""
    return cute.math.exp2(x * _LOG2_E, fastmath=True)


def _cutedsl_log_fast(x):
    """log via native log2: ``log(x) = log2(x) * ln(2)`` (``lg2.approx``)."""
    return cute.math.log2(x, fastmath=True) * _LN2


def _cutedsl_exp_precise(x):
    """exp via ``math.exp`` without fastmath: libdevice ``__nv_expf`` (~1 ulp).

    This matches ``torch.exp`` fp32 (which calls ``expf``/``__nv_expf``), so
    fp32 tensors land exactly on the strict public JSD contract that the
    ``ex2.approx`` fastmath path (~2 ulp) misses (fwd scalar drifted ~4e-7
    vs the 1e-7 atol of ``test/transformers/test_jsd.py``).
    """
    return cute.math.exp(x, fastmath=False)


def _cutedsl_log_precise(x):
    """log via ``math.log`` without fastmath: libdevice ``__nv_logf``."""
    return cute.math.log(x, fastmath=False)


# ===========================================================================
# Kernel — elementwise: loss + dX from student (X) and teacher (Y) log-probs
# ===========================================================================
class _LigerJSDLossAndGradCuTeDSL:
    """CuTe DSL JSD loss + gradient primitive.

    Pure elementwise: load tiles of ``X`` (student log-prob) and ``Y``
    (teacher log-prob), compute loss + dX in fp32, store both.  The ``beta``
    value is a constructor parameter so separate kernels are compiled for
    forward KL (beta=0), reverse KL (beta=1), and general JSD.
    """

    def __init__(self, dtype: Type[cutlass.Numeric], N: int, beta: float):
        self.dtype = dtype
        self.N = N
        self.beta = beta
        self.cluster_n = 1
        # fp32 needs a precise exp/log to hit the strict 1e-7 public JSD
        # contract (bf16/fp16 tolerate the native ex2.approx/lg2.approx path,
        # rounding away its ~2-ulp error on store). The decision is made at
        # trace time from the input dtype, so each compiled kernel bakes in a
        # single math path.
        if self.dtype.width == 32:
            self._exp = _cutedsl_exp_precise
            self._log = _cutedsl_log_precise
        else:
            self._exp = _cutedsl_exp_fast
            self._log = _cutedsl_log_fast

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        # 1024 for N > 16384 (not 256): at 256 threads/row each thread carries
        # >=125 fp32 (or 250 bf16) elements and spills registers. Measured on
        # B200: bf16 @N=32768 goes 0.61ms -> 0.13ms (4.8x), fp32 0.18 -> 0.17ms.
        return 1024

    def _num_threads(self):
        return 128 if self.N <= 16384 else 1024

    def _get_tiled_copy(self, vecsize: int = 1):
        threads_per_row = self._threads_per_row()
        num_threads = self._num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0
        if self.N > 16384:
            # Wide-vocab N-blocking: cap the chunks each block loops over at
            # 2 and move the remaining N chunk-groups onto grid-y (see
            # __call__ and kernel). All math is elementwise per element -- no
            # cross-N reduction -- so N-blocking is bit-identical, but it
            # keeps the per-thread live-register footprint bounded: the old
            # whole-row-per-block config makes each thread carry
            # ceil(N/vecsize/1024)*vecsize*~6 fp32 temporaries
            # (VEC=4 bf16 -> 128 fp32 regs @V=128256), spilling to local
            # memory. Cap sweep on B200 (caps 1/2/4/8 x bf16/fp32 @
            # N=32768/65536/128256, every nonzero cap bit-identical): cap=2
            # (8 elems/thread live) wins at ALL shapes/dtypes -- cap=4 (the
            # previous value) is 1.09-1.14x slower, cap=8 worse, and cap=1
            # is also slower (grid-y gets too deep; fp32 even miscompiles).
            num_blocks_N = min(cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n), 2)
        else:
            num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n)
        tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = copy_utils.tiled_copy_2d(self.dtype, threads_per_row, num_threads, vecsize)
        return tiled_copy, tiler_mn, threads_per_row

    def _row_math(self, tXrX, tYrY, scale, tLossRLoss, tDXrDX):
        # Plain (undecorated) helper executed at trace time and inlined into the
        # kernel body -- same proven trace-time idiom as ``self._exp`` /
        # ``self._log`` above. Shared by the packed and labeled kernel
        # specializations so the beta-math chain lives in exactly one place.
        x = tXrX.load().to(cute.Float32)  # X = log Q (student)
        y = tYrY.load().to(cute.Float32)  # Y = log P (teacher)

        beta = const_expr(self.beta)
        if const_expr(beta == 0.0):
            # Forward KL: loss = P * (Y - X), dX = -P
            p = self._exp(y)
            loss = p * (y - x)
            dx = -p
        elif const_expr(beta == 1.0):
            # Reverse KL: loss = Q * (X - Y), dX = loss + Q
            q = self._exp(x)
            loss = q * (x - y)
            dx = loss + q
        else:
            # General JSD
            q = self._exp(x)
            p = self._exp(y)
            beta_p = beta * p
            one_minus_beta_q = (1.0 - beta) * q
            m = beta_p + one_minus_beta_q
            log_m = self._log(m)
            loss = beta_p * y + one_minus_beta_q * x - m * log_m
            dx = one_minus_beta_q * (x - log_m)

        loss = loss * scale
        dx = dx * scale

        tLossRLoss.store(loss)
        tDXrDX.store(dx.to(tDXrDX.element_type))

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mLoss: cute.Tensor,
        mDX: cute.Tensor,
        mLabel: Optional[cute.Tensor],  # ignore-row labels (BT,) int64, or None
        scale: Float32,
        ignore_index: Int64,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        largest_dtype_width = const_expr(max(t.element_type.width for t in [mX, mY, mLoss, mDX]))
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, _ = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        self.kernel(mX, mY, mLoss, mDX, mLabel, scale, ignore_index, tiler_mn, tiled_copy).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), cute.ceil_div(self.N, tiler_mn[1]), 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mLoss: cute.Tensor,
        mDX: cute.Tensor,
        mLabel: Optional[cute.Tensor],
        scale: Float32,
        ignore_index: Int64,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bnidx, _ = cute.arch.block_idx()

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX, gY, gLoss, gDX, cX = [cute.local_tile(mT_, tiler_mn, (bidx, bnidx)) for mT_ in (mX, mY, mLoss, mDX, idX)]

        thr_copy = tiled_copy.get_slice(tidx)
        tXgX = thr_copy.partition_S(gX)
        tYgY = thr_copy.partition_S(gY)
        tLossGLoss = thr_copy.partition_D(gLoss)
        tDXgDX = thr_copy.partition_D(gDX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tXrX, tYrY, tLossRLoss, tDXrDX = [cute.make_rmem_tensor_like(thr) for thr in (tXgX, tYgY, tLossGLoss, tDXgDX)]

        # Same as the old full-row check (shape[1] == tiler_n * cluster_n) on
        # the small rung (tiler_n >= N there); on the wide rung each grid-y
        # block is full iff N % tiler_n == 0, in which case no N predication is
        # needed at all (predicate_k uses global col indices from idX).
        is_even_N = const_expr(shape[1] % tiler_mn[1] == 0)
        tXpX = None if is_even_N else copy_utils.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        copy = partial(copy_utils.copy, pred=tXpX)

        # One scalar label read per thread. tiled_copy_2d lays the thread layout
        # out as (num_rows_per_block, threads_per_row), so each thread covers
        # exactly one row: tXcX[0][0] (this thread's element-row coord, already
        # used for the row guard below) is that row's index.
        label_row = tXcX[0][0]

        if const_expr(mLabel is None):
            # Packed / no-label specialization: byte-identical trace to the
            # c848ced body (the label path is never traced in this variant).
            if tXcX[0][0] < shape[0]:
                copy(tXgX, tXrX)
                copy(tYgY, tYrY)
            self._row_math(tXrX, tYrY, scale, tLossRLoss, tDXrDX)
            if tXcX[0][0] < shape[0]:
                copy(tLossRLoss, tLossGLoss)
                copy(tDXrDX, tDXgDX)
        elif const_expr(self.dtype.width == 32):
            # Labeled fp32 kernels: Triton HAS_LABEL-class early exit -- an
            # ignored row skips the X/Y gmem loads AND the precise-libdevice
            # exp/log math (c848ced folded the +0.0 stores into the epilogue
            # but still paid the full read and math on every ignored row
            # before zeroing it). fp32-only by measurement on B300 (ABAB x2
            # fresh-process, ALL_BITWISE both constructions): the skip wins
            # +7.1-7.6% at 1/3-ignore V=32000 and +27% at 90%-ignore for fp32
            # -- its precise-libdevice body is expensive enough that skipping
            # it beats load-uniformity loss -- while the fast-exp bf16/fp16
            # path loses 5-9% at 1/3-ignore (uniform loads win) and only +5%
            # at 90% (still behind Triton there), so 2-byte labeled kernels
            # keep the c848ced trace below. The stores/copies write the same
            # +0.0 fragments at the same sites -- a pure control decision,
            # never an arithmetic reorder -- so every fp32 cell is bitwise ==
            # the c848ced body (probe enforces ALL_BITWISE).
            if label_row < shape[0]:
                if mLabel[label_row] == ignore_index:
                    tLossRLoss.fill(0.0)
                    tDXrDX.fill(tDXrDX.element_type.zero)
                    copy(tLossRLoss, tLossGLoss)
                    copy(tDXrDX, tDXgDX)
                else:
                    copy(tXgX, tXrX)
                    copy(tYgY, tYrY)
                    self._row_math(tXrX, tYrY, scale, tLossRLoss, tDXrDX)
                    copy(tLossRLoss, tLossGLoss)
                    copy(tDXrDX, tDXgDX)
        else:
            # Labeled 2-byte kernels: byte-identical trace to the c848ced body
            # (math unconditional under the row guard, epilogue fill overwrite,
            # guarded stores) -- the measured never-win regime for the skip.
            if tXcX[0][0] < shape[0]:
                copy(tXgX, tXrX)
                copy(tYgY, tYrY)
            self._row_math(tXrX, tYrY, scale, tLossRLoss, tDXrDX)
            if label_row < shape[0]:
                if mLabel[label_row] == ignore_index:
                    tLossRLoss.fill(0.0)
                    tDXrDX.fill(tDXrDX.element_type.zero)
            if tXcX[0][0] < shape[0]:
                copy(tLossRLoss, tLossGLoss)
                copy(tDXrDX, tDXgDX)
        # Rows >= shape[0] (partial last M-block, has_label or packed): no
        # loads and no stores -- identical predication coverage to c848ced,
        # whose single row guard gated every copy the same way.


# ---------------------------------------------------------------------------
# Compile cache.
# ---------------------------------------------------------------------------
_COMPILE_CACHE: dict = {}


def _get_kernel(
    x_dtype: torch.dtype,
    y_dtype: torch.dtype,
    loss_dtype: torch.dtype,
    dx_dtype: torch.dtype,
    N: int,
    beta: float,
    has_label: bool = False,
):
    """Return a compiled kernel, building it on first miss."""
    key = (x_dtype, y_dtype, loss_dtype, dx_dtype, N, beta, has_label)
    if key in _COMPILE_CACHE:
        return _COMPILE_CACHE[key]

    dtype = torch2cute_dtype_map[x_dtype]
    y_cute_dtype = torch2cute_dtype_map[y_dtype]
    loss_cute_dtype = torch2cute_dtype_map[loss_dtype]
    dx_cute_dtype = torch2cute_dtype_map[dx_dtype]
    batch_sym = cute.sym_int()
    div = math.gcd(128 // dtype.width, N)
    x_cute = fake_tensor(dtype, (batch_sym, N), div)
    y_cute = fake_tensor(y_cute_dtype, (batch_sym, N), div)
    loss_cute = fake_tensor(loss_cute_dtype, (batch_sym, N), div)
    dx_cute = fake_tensor(dx_cute_dtype, (batch_sym, N), div)
    label_cute = fake_tensor(torch2cute_dtype_map[torch.int64], (batch_sym,)) if has_label else None

    compiled = cute.compile(
        _LigerJSDLossAndGradCuTeDSL(dtype, N, beta),
        x_cute,
        y_cute,
        loss_cute,
        dx_cute,
        label_cute,
        Float32(0.0),
        Int64(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _COMPILE_CACHE[key] = compiled
    return compiled


# ===========================================================================
# Host-side launcher
# ===========================================================================
def _jsd_loss_and_grad_cutedsl(
    student_prob: Tensor,
    teacher_prob: Tensor,
    beta: float,
    n_non_ignore: int,
    shift_labels: Optional[Tensor] = None,
    ignore_index: int = -100,
) -> Tuple[Tensor, Tensor]:
    """Compute per-element JSD loss + dx via the inline CuTe DSL kernel.

    Returns ``(loss, dx)`` where ``loss`` is a fresh ``(BT, V)`` fp32 tensor
    and ``dx`` overwrites ``student_prob`` in-place (matching the Triton
    primitive's contract).

    ``shift_labels`` (optional) is the ignore-row label stream: when given AND a
    real ignored row exists (``n_non_ignore < BT``), it is threaded into the
    compiled kernel which zeroes those rows in the epilogue. ``None`` labels or a
    packed batch (``n_non_ignore == BT``) pass ``None`` -- the absent-arg
    specialization dead-code-eliminates the zero-fill.
    """
    BT, V = student_prob.shape
    x_flat = student_prob.contiguous()
    y_flat = teacher_prob.contiguous()
    # empty, not zeros: the elementwise kernel stores loss+dX for EVERY
    # in-tensor element unconditionally (predication only masks columns >= V
    # and rows >= BT, both outside the (BT, V) allocation), so the zeros
    # memset is a redundant (BT, V) fp32 write the kernel fully overwrites.
    loss = torch.empty((BT, V), dtype=torch.float32, device=student_prob.device)
    dx = x_flat  # in-place overwrite (matches Triton contract)

    label_arg = None
    if shift_labels is not None:
        labels_i64 = shift_labels.contiguous().to(torch.int64)
        # Thread labels whenever THIS (chunk-local) label tensor actually
        # contains an ignored row. The previous `n_non_ignore < BT` compared the
        # GLOBAL non-ignore count with the per-chunk BT, so multi-chunk batches
        # with a few ignores (global_n_non_ignore >= chunk_BT) silently dropped
        # the label stream and let ignored rows contribute loss/gradient.
        if bool((labels_i64 == ignore_index).any()):
            label_arg = labels_i64

    scale = Float32(1.0 / max(n_non_ignore, 1))
    compiled = _get_kernel(
        x_flat.dtype, y_flat.dtype, loss.dtype, dx.dtype, V, float(beta), has_label=label_arg is not None
    )
    compiled(x_flat, y_flat, loss, dx, label_arg, scale, int(ignore_index))

    return loss, dx


# ===========================================================================
# Public registration
# ===========================================================================
@register_op(
    "jsd_loss_and_grad",
    impl_name="nvidia-cutedsl",
    capability=Capability(
        min_cc=(9, 0),
        modules=["cutlass.cute", "torch"],
    ),
    modes=("default",),
    default_mode="default",
    preference_rank=_RANK_B300 if _max_capability() > 100 else _RANK_HOPPER_B200,
    notes=(
        "Per-chunk JSD primitive (returns per-row loss tile + per-element dx). "
        "CuTe DSL elementwise kernel for Hopper+ (sm_90+). Used by fused_linear_jsd. Auto preference_rank 20 on sm_90/sm_100, 2 on sm_103+ (see rank constants)."
    ),
)
def jsd_loss_and_grad_cutedsl(
    student_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    shift_labels: Optional[torch.Tensor],
    beta: float,
    ignore_index: int,
    n_non_ignore: float,
    *,
    mode: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-element loss + dx for one chunk via CuTe DSL.

    Mirrors the Triton primitive's contract: writes dx **in-place into**
    ``student_prob`` and returns ``(loss, dx)``.

    Args:
        student_prob: ``(BT, V)`` log Q (student). Will be overwritten with dx.
        teacher_prob: ``(BT, V)`` log P (teacher).
        shift_labels: optional ``(BT,)`` mask; rows where the label equals
            ``ignore_index`` contribute zero loss and zero gradient.
        beta: mixing coefficient in [0, 1].
        ignore_index: label value to ignore.
        n_non_ignore: pre-computed count of non-ignored rows (caller's job).

    Returns:
        ``(loss, dx)`` where ``loss.shape == (BT, V)`` (fp32) and
        ``dx is student_prob`` (the in-place write).
    """
    if mode not in (None, "default"):
        raise ValueError(f"jsd_loss_and_grad_cutedsl: only mode='default'; got {mode!r}")

    # fp32 V==16384 side-step (sm_103+): the one cell where the auto-routing
    # flip above would REGRESS. The CuTe DSL kernel runs ~0.20x vs both cuTile
    # and pinned Triton here (measured 4.99ms vs 1.00/0.99ms at BT=8192;
    # wide-tile re-route refuted at ratio ~1.0), while Triton == cuTile within
    # 1.6% pinned at the band. Delegate to the Triton primitive (identical
    # in-place dx / fp32 (BT, V) loss contract) and let the rest of the map
    # keep the rank flip.
    if (
        _max_capability() > 100
        and student_prob.element_size() == 4
        and teacher_prob.element_size() == 4
        and student_prob.shape[-1] == _B300_FP32_SIDESTEP_VOCAB
    ):
        emit_fallback_warning(
            "jsd_loss_and_grad",
            "nvidia-cutedsl",
            "nvidia-triton",
            "fp32 V==16384 on sm_103+: CuTe DSL kernel measured ~0.20x vs cuTile/Triton at this one cell",
        )
        return jsd_loss_and_grad_triton(
            student_prob,
            teacher_prob,
            shift_labels,
            float(beta),
            int(ignore_index),
            n_non_ignore,
            mode=mode,
        )

    # Ignore-row zeroing is folded into the kernel epilogue (see
    # _jsd_loss_and_grad_cutedsl): the compare + both (BT, V) boolean-mask
    # scatters that the host used to run on labeled-with-ignores shapes are
    # deleted. Packed batches (n_non_ignore == BT) still skip the label stream
    # entirely; no-label calls are unaffected.
    loss, dx = _jsd_loss_and_grad_cutedsl(
        student_prob,
        teacher_prob,
        float(beta),
        int(round(n_non_ignore)),
        shift_labels,
        int(ignore_index),
    )

    return loss, dx


# ===========================================================================
# Standalone JSD op — autograd-aware wrapper.
#
# The CuTe DSL acceleration happens inside ``LigerJSDFunction.forward``:
# it calls ``dispatch("jsd_loss_and_grad", ...)`` for the inner JSD kernel,
# which selects CuTe DSL on Hopper and cuTile on Blackwell when available.
# ===========================================================================
_CUTEDSL_JSD_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-3, "rtol_fwd": 1e-3, "rtol_bwd": 1e-3},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 2e-2, "rtol_fwd": 1e-2, "rtol_bwd": 1e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-5, "rtol_fwd": 1e-5, "rtol_bwd": 1e-5},
}


@register_op(
    "jsd",
    impl_name="nvidia-cutedsl",
    capability=Capability(
        min_cc=(9, 0),
        modules=["cutlass.cute", "torch"],
    ),
    modes=("default",),
    default_mode="default",
    preference_rank=_RANK_B300 if _max_capability() > 100 else _RANK_HOPPER_B200,
    tolerances=_CUTEDSL_JSD_TOLERANCES,
    notes=(
        "CuTe DSL JSD for Hopper+ (sm_90+); inner JSD via CuTe DSL elementwise kernel."
        " Auto preference_rank 20 on sm_90/sm_100, 2 on sm_103+; the fp32 V==16384 band is handed to Triton by the primitive-level side-step guard."
    ),
)
def jsd_cutedsl(
    _input: torch.Tensor,
    target: torch.Tensor,
    shift_labels: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    ignore_index: int = -100,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """CuTe DSL JSD dispatch entry point.

    Delegates to ``LigerJSDFunction``, which internally dispatches the
    ``jsd_loss_and_grad`` primitive through the dispatcher — picking up the
    CuTe DSL kernel on Hopper+.

    ``mode`` is accepted only for API parity with the other backends; the only
    valid value is ``"default"`` (or ``None``).
    """
    if mode not in (None, "default"):
        raise ValueError(f"CuTe DSL jsd has only mode='default'; got mode={mode!r}.")
    return LigerJSDFunction.apply(
        _input,
        target,
        shift_labels,
        beta,
        ignore_index,
        "nvidia-cutedsl",
        mode,
    )
