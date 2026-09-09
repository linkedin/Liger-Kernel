"""CuTe DSL (CUTLASS python) backend for ``kl_div``.

Strategy
--------
KL-divergence loss is a row-reduction followed by a cross-row reduction::

    loss_row = sum_v( target * (log(max(target, eps)) - y_pred) )   # not log_target
    loss_row = sum_v( exp(target) * (target - y_pred) )             # log_target
    loss     = reduce(loss_row)                                      # sum / mean / batchmean

The CuTe DSL forward kernel uses the ``ReductionBase`` infrastructure (same as
the rms_norm / layer_norm forward kernels) to load tiles of ``y_pred`` and
``target`` into shared memory, compute the per-element loss in fp32, and reduce
per row via ``row_reduce``.

The backward is purely elementwise::

    grad = -target            # not log_target
    grad = -exp(target)       # log_target

with ``grad_output`` and the reduction factor folded into the kernel epilogue
as one dynamic fp32 scalar.  The backward kernel follows the SwiGLU elementwise
pattern (TiledCopy + ``cp.async``).

Capability
----------
- Compute capability >= sm_90 (Hopper or newer).
- Forward kernel is capped at ``V <= 32768`` (shared-memory tile limit; same
  ceiling as the rms_norm backward). Larger V raises ``RuntimeError`` so the
  dispatcher falls back to Triton.

References
----------
- Triton reference: :mod:`liger_kernel.ops.kl_div`
- CuTe DSL reduction: :mod:`._cute_lib.reduction_base.ReductionBase`
- CuTe DSL elementwise: :mod:`._cutedsl.swiglu`
"""

import math

from functools import partial
from typing import Optional
from typing import Type

# ---------------------------------------------------------------------------
# Top-level CuTe DSL imports. Identical pattern to the rms_norm sibling.
# ---------------------------------------------------------------------------
import cuda.bindings.driver as cuda  # noqa: F401  (referenced by cute.compile)
import cutlass
import cutlass.cute as cute
import torch

from cutlass import Float32
from cutlass import const_expr
from torch import Tensor

# Inlined CuTe DSL utilities (adapted from Quack, Apache-2.0).
import liger_kernel.ops.backends._cutedsl._cute_lib.copy_utils as copy_utils

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.backends.dispatch import emit_fallback_warning
from liger_kernel.ops._nvidia_shared import to_local_if_dtensor as _to_local_if_dtensor
from liger_kernel.ops.backends._cutedsl._cute_lib.compile_utils import make_fake_tensor as fake_tensor
from liger_kernel.ops.backends._cutedsl._cute_lib.dtype_map import torch2cute_dtype_map
from liger_kernel.ops.backends._cutedsl._cute_lib.reduce import row_reduce
from liger_kernel.ops.backends._cutedsl._cute_lib.reduction_base import ReductionBase
from liger_kernel.ops.utils import device_context

# log2(e) for the exp2-based exp (native HW instruction).
_LOG2_E = math.log2(math.e)

# Same ceiling as the rms_norm backward — beyond this the tile doesn't fit
# in shared memory without the cluster-reduce path.
_FWD_MAX_TILE_CUTEDSL = 32768

# The FORWARD stages a full row of EACH input (y_pred + target) in dynamic
# shared memory, i.e. ~ V * (es_y + es_t) bytes plus the reduction buffer.
# That fits the 232448-byte sm_100 per-CTA cap across the 28672..32768 band
# for 2-byte dtypes (128KB @32768) and for mixed-dtype calls (bf16+fp32 =
# 192KB @32768), but overflows when BOTH inputs are 4-byte (vecsize 4):
# fp32 fwd kernels at V=32000 / V=32768 COMPILE then fault at launch with
# cudaErrorInvalidValue ("launch shared memory exceeds current GPU arch
# sm_100a allowed. Allocated: 262176 bytes. Max: 232448" — measured on B200
# by the wide-N fwd sweep; sY+sT alone = 2*4*32768 = 262144). Guard on the
# measured power-of-two ceiling below: an all-4-byte call in the
# (28672, 32768] band takes the Triton fallback in the public wrapper
# instead of faulting; every other dtype mix keeps the CuTe DSL kernel over
# the full band, byte-unchanged from before.
_FWD_MAX_TILE_CUTEDSL_FP32 = 28672


def _max_capability() -> int:
    # Runtime dispatch-plane key (e.g. 100 = sm_100/B200, 103 = sm_103/B300).
    # Called per wrapper invocation (never memoized) so mock-cc suites key
    # separately per call, same idiom as capability.is_satisfied and the
    # fused_add_rms_norm 305238c plane-guard.
    if torch.cuda.is_available():
        try:
            maj, minor = torch.cuda.get_device_capability()
        except (IndexError, RuntimeError, AssertionError):  # pragma: no cover
            return 0
        return 10 * maj + minor
    return 0


# B300 sm_103 no-grad fwd-only fp32-pair Triton-preference band. The full
# fwd+bwd decision map has CuTe DSL winning
# every fp32 rung 1.29-1.47x, and 2-byte dtypes keep their 1.20-1.91x wins
# in the no-grad regime too (pinned backends,
# M=8192, ABAB x3, per-iter-sync CUDA-event medians) -- but fp32-pair
# no-grad fwd-only shows a rope-class regime inversion on sm_103:
# ratio(T/C) = 0.9306x @16384, 0.9740x @24576, 0.7624x @28672 (reps tight
# <2%), ~0.93-1.31x absolute ms deficit. The bwd never runs in this regime,
# so the wrapper routes these calls to Triton. B200 sm_100 keeps the CuTe
# route (never B300-class measured there; same sm_103-only scoping as
# fused_add_rms_norm 305238c / jsd e8e4ed5 / fused_linear_jsd b977b0e).
_B300_FP32_NO_GRAD_TRITON_LO = 16384
_B300_FP32_NO_GRAD_TRITON_HI = 28672


# ===========================================================================
# Forward kernel — row reduction: loss_row = sum(loss_element)
# ===========================================================================
class _LigerKLDivCuTeDSLForward(ReductionBase):
    """CuTe DSL KL-divergence forward.

    One CTA per row.  Loads tiles of ``y_pred`` (log-space) and ``target``
    into shared memory, computes the per-element loss in fp32, and reduces
    per row via ``row_reduce``.
    """

    def __init__(self, dtype: Type[cutlass.Numeric], N: int, log_target: bool):
        super().__init__(dtype, N, stage=1)
        self.log_target = log_target

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        self.cluster_n = 1

    @cute.jit
    def __call__(
        self,
        mY: cute.Tensor,  # y_pred (BT, V) in log-space
        mT: cute.Tensor,  # target (BT, V)
        mLoss: cute.Tensor,  # output (BT,) per-row loss
        eps: Float32,
        stream: cuda.CUstream,
    ):
        assert mY.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(max(t.element_type.width for t in [mY, mT, mLoss]))
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        self.kernel(mY, mT, mLoss, eps, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[cute.ceil_div(mY.shape[0], tiler_mn[0]), 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mY: cute.Tensor,
        mT: cute.Tensor,
        mLoss: cute.Tensor,
        eps: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        shape = mY.shape
        idX = cute.make_identity_tensor(shape)
        gY, gT, cX = [cute.local_tile(mT_, tiler_mn, (bidx, 0)) for mT_ in (mY, mT, idX)]

        smem = cutlass.utils.SmemAllocator()
        sY = smem.allocate_tensor(mY.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16)
        sT = smem.allocate_tensor(mT.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16)
        tv_layout = tiled_copy.layout_tv_tiled
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        thr_copy = tiled_copy.get_slice(tidx)
        tYgY = thr_copy.partition_S(gY)
        tYsY = thr_copy.partition_D(sY)
        tTgT = thr_copy.partition_S(gT)
        tTsT = thr_copy.partition_D(sT)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]

        tYrY, tTrT = [cute.make_rmem_tensor_like(thr) for thr in (tYgY, tTgT)]

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = None if is_even_N else copy_utils.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        copy = partial(copy_utils.copy, pred=tXpX)

        row = tXcX[0][0]
        if row < shape[0]:
            copy(tYgY, tYsY, is_async=True)
            copy(tTgT, tTsT, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        cute.autovec_copy(tYsY, tYrY)
        cute.autovec_copy(tTsT, tTrT)
        y = tYrY.load().to(cute.Float32)
        t = tTrT.load().to(cute.Float32)

        # KL(target || y_pred):
        #   not log_target: target * (log(max(target, eps)) - y_pred)
        #   log_target:     exp(target) * (target - y_pred)
        if const_expr(not self.log_target):
            t_safe = cute.where(t > eps, t, eps)
            loss_elem = t * (cute.math.log2(t_safe, fastmath=True) * Float32(1.0 / _LOG2_E) - y)
        else:
            t_exp = cute.math.exp2(t * _LOG2_E, fastmath=True)
            loss_elem = t_exp * (t - y)

        loss_sum = row_reduce(
            loss_elem,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            init_val=0.0,
        )

        # Write per-row loss. Only the thread at column 0 writes.
        if tXcX[0][1] == 0 and row < shape[0]:
            mLoss[row] = loss_sum


# ===========================================================================
# Backward kernel — elementwise: grad = -target or -exp(target)
# ===========================================================================
class _LigerKLDivCuTeDSLBackward:
    """CuTe DSL KL-divergence backward.

    Pure elementwise: load a tile of ``target``, compute ``-target`` (or
    ``-exp(target)`` if ``log_target``), store the result.
    """

    def __init__(self, dtype: Type[cutlass.Numeric], N: int, log_target: bool):
        self.dtype = dtype
        self.N = N
        self.log_target = log_target
        self.cluster_n = 1

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _num_threads(self):
        return 128 if self.N <= 16384 else 256

    def _get_tiled_copy(self, vecsize: int = 1):
        threads_per_row = self._threads_per_row()
        num_threads = self._num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0
        num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n)
        tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = copy_utils.tiled_copy_2d(self.dtype, threads_per_row, num_threads, vecsize)
        return tiled_copy, tiler_mn, threads_per_row

    @cute.jit
    def __call__(
        self,
        mT: cute.Tensor,
        mGrad: cute.Tensor,
        scale: Float32,
        stream: cuda.CUstream,
    ):
        assert mT.element_type == self.dtype
        largest_dtype_width = const_expr(max(t.element_type.width for t in [mT, mGrad]))
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, _ = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        self.kernel(mT, mGrad, scale, tiler_mn, tiled_copy).launch(
            grid=[cute.ceil_div(mT.shape[0], tiler_mn[0]), 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mT: cute.Tensor,
        mGrad: cute.Tensor,
        scale: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        shape = mT.shape
        idX = cute.make_identity_tensor(shape)
        gT, gGrad, cX = [cute.local_tile(mT_, tiler_mn, (bidx, 0)) for mT_ in (mT, mGrad, idX)]

        thr_copy = tiled_copy.get_slice(tidx)
        tTgT = thr_copy.partition_S(gT)
        tGradGGrad = thr_copy.partition_D(gGrad)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tTrT, tGradRGrad = [cute.make_rmem_tensor_like(thr) for thr in (tTgT, tGradGGrad)]

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = None if is_even_N else copy_utils.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        copy = partial(copy_utils.copy, pred=tXpX)

        if tXcX[0][0] < shape[0]:
            copy(tTgT, tTrT)
        t = tTrT.load().to(cute.Float32)

        if const_expr(not self.log_target):
            grad = -t
        else:
            grad = -cute.math.exp2(t * _LOG2_E, fastmath=True)

        # ``scale`` = grad_output / reduction_divisor, one host-supplied dynamic scalar: the
        # upstream-grad multiply AND the batchmean (/BT) / mean (/(BT*V)) division are folded in
        # here, in fp32, with a single store rounding. This removes the two full (BT,V)
        # post-kernel elementwise passes (``* grad_output`` and ``/ BT``) plus the ``torch.equal``
        # host sync and device ``torch.tensor(1.0)`` allocation the old wrapper paid every call.
        grad = grad * scale

        tGradRGrad.store(grad.to(tGradRGrad.element_type))
        if tXcX[0][0] < shape[0]:
            copy(tGradRGrad, tGradGGrad)


# ---------------------------------------------------------------------------
# Compile caches.
# ---------------------------------------------------------------------------
_FWD_COMPILE_CACHE: dict = {}
_BWD_COMPILE_CACHE: dict = {}


def _get_fwd_kernel(y_dtype: torch.dtype, t_dtype: torch.dtype, N: int, log_target: bool, eps: float):
    """Return a compiled forward kernel, building it on first miss."""
    key = (y_dtype, t_dtype, N, log_target, float(eps))
    if key in _FWD_COMPILE_CACHE:
        return _FWD_COMPILE_CACHE[key]

    dtype = torch2cute_dtype_map[y_dtype]
    t_cute_dtype = torch2cute_dtype_map[t_dtype]
    batch_sym = cute.sym_int()
    div = math.gcd(128 // dtype.width, N)
    y_cute = fake_tensor(dtype, (batch_sym, N), div)
    t_cute = fake_tensor(t_cute_dtype, (batch_sym, N), div)
    loss_cute = fake_tensor(Float32, (batch_sym,))

    compiled = cute.compile(
        _LigerKLDivCuTeDSLForward(dtype, N, log_target),
        y_cute,
        t_cute,
        loss_cute,
        Float32(float(eps)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _FWD_COMPILE_CACHE[key] = compiled
    return compiled


def _get_bwd_kernel(t_dtype: torch.dtype, grad_dtype: torch.dtype, N: int, log_target: bool):
    """Return a compiled backward kernel, building it on first miss."""
    key = (t_dtype, grad_dtype, N, log_target)
    if key in _BWD_COMPILE_CACHE:
        return _BWD_COMPILE_CACHE[key]

    dtype = torch2cute_dtype_map[t_dtype]
    grad_cute_dtype = torch2cute_dtype_map[grad_dtype]
    batch_sym = cute.sym_int()
    div = math.gcd(128 // dtype.width, N)
    t_cute = fake_tensor(dtype, (batch_sym, N), div)
    grad_cute = fake_tensor(grad_cute_dtype, (batch_sym, N), div)

    compiled = cute.compile(
        _LigerKLDivCuTeDSLBackward(dtype, N, log_target),
        t_cute,
        grad_cute,
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _BWD_COMPILE_CACHE[key] = compiled
    return compiled


# ===========================================================================
# Host-side launchers and autograd Function
# ===========================================================================
def _kl_div_cutedsl_forward(
    y_pred: Tensor,
    y_true: Tensor,
    log_target: bool,
    reduction: str,
    eps: float,
) -> Tensor:
    """Forward via the inline CuTe DSL reduction kernel.

    Computes per-row loss sums, then applies the cross-row reduction
    (sum / mean / batchmean) host-side — matching the Triton kernel's
    post-kernel reduction logic.
    """
    with device_context(y_pred.device):
        BT, V = y_pred.shape
        both_fp32 = y_pred.element_size() > 2 and y_true.element_size() > 2
        fwd_limit = _FWD_MAX_TILE_CUTEDSL_FP32 if both_fp32 else _FWD_MAX_TILE_CUTEDSL
        if V > fwd_limit:
            raise RuntimeError(
                f"cuTeDSL kl_div forward only supports V <= {fwd_limit} for "
                f"4-byte-pair inputs; got {V}. Use impl='nvidia-triton' for wider rows."
            )

        y_flat = y_pred.contiguous()
        t_flat = y_true.contiguous()
        loss_per_row = torch.empty(BT, dtype=torch.float32, device=y_pred.device)

        compiled = _get_fwd_kernel(y_flat.dtype, t_flat.dtype, V, log_target, eps)
        compiled(y_flat, t_flat, loss_per_row, Float32(float(eps)))

        # Cross-row reduction (matches Triton's host-side logic).
        if reduction == "batchmean":
            return loss_per_row.sum() / BT
        elif reduction == "sum":
            return loss_per_row.sum(dim=0)
        elif reduction == "mean":
            return loss_per_row.sum() / (BT * V)
        else:  # "none"
            return loss_per_row


def _kl_div_cutedsl_backward(
    y_true: Tensor,
    grad_output: Tensor,
    log_target: bool,
    reduction: str,
) -> Tensor:
    """Backward via the inline CuTe DSL elementwise kernel.

    Computes ``(-target) * scale`` (``-exp(target) * scale`` when ``log_target``) with the
    scalar ``scale = grad_output / reduction_divisor`` folded INTO the kernel epilogue in fp32 —
    one rounding at the store. The old wrapper ran the unscaled kernel and then paid a
    ``torch.equal`` host sync + device ``torch.tensor(1.0)`` allocation every backward, plus up
    to TWO extra full (BT,V) elementwise passes (``* grad_output`` and ``/ BT``) for the default
    batchmean / mean reductions. Folding removes both passes and the allocation, and is strictly
    MORE accurate (old: bf16 store -> bf16 mul -> bf16 div; new: one fp32 multiply -> one store).
    """
    with device_context(y_true.device):
        BT, V = y_true.shape
        t_flat = y_true.contiguous()
        new_grads = torch.empty_like(t_flat)

        # Reduction divisor (matches torch/Triton semantics): batchmean -> /BT, mean -> /(BT*V),
        # sum -> 1 ("none" never reaches here — the public wrapper delegates it to Triton).
        if reduction == "batchmean":
            red_div = float(BT)
        elif reduction == "mean":
            red_div = float(BT * V)
        else:
            red_div = 1.0

        compiled = _get_bwd_kernel(t_flat.dtype, new_grads.dtype, V, log_target)
        if grad_output.numel() != 1:
            # Rare elementwise upstream grad: keep the old broadcast path exactly (per-element go).
            compiled(t_flat, new_grads, Float32(1.0))
            derivative = new_grads * grad_output
            if red_div != 1.0:
                derivative = derivative / red_div
            return derivative

        # Scalar upstream grad (every scalar-loss autograd backward): a single host read of the
        # value (the old torch.equal path synced as well) and one fused fp32 multiply in-kernel.
        scale = float(grad_output) / red_div
        compiled(t_flat, new_grads, Float32(scale))
        return new_grads


class _LigerKLDivCuTeDSLFunction(torch.autograd.Function):
    """Autograd wrapper for KL-divergence loss via CuTe DSL."""

    @staticmethod
    def forward(ctx, y_pred, y_true, reduction="batchmean", log_target=False, eps=1e-10):
        y_pred = _to_local_if_dtensor(y_pred).contiguous()
        y_true = _to_local_if_dtensor(y_true).contiguous()
        ctx.save_for_backward(y_true)
        ctx.reduction = reduction
        ctx.log_target = log_target
        return _kl_div_cutedsl_forward(y_pred, y_true, log_target, reduction, eps)

    @staticmethod
    def backward(ctx, grad_output):
        (y_true,) = ctx.saved_tensors
        derivative = _kl_div_cutedsl_backward(y_true, grad_output, ctx.log_target, ctx.reduction)
        return derivative, None, None, None, None


# ===========================================================================
# Public registration
# ===========================================================================
_CUTEDSL_KLDIV_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


@register_op(
    "kl_div",
    impl_name="nvidia-cutedsl",
    capability=Capability(
        min_cc=(9, 0),
        modules=["cutlass.cute", "torch"],
    ),
    modes=("default",),
    default_mode="default",
    preference_rank=10,
    tolerances=_CUTEDSL_KLDIV_TOLERANCES,
    notes=(
        "CuTe DSL KL-divergence for Hopper+ (sm_90+); row-reduction fwd + "
        "elementwise bwd. On B300 sm_103, no-grad 4-byte-pair calls with V in "
        "[16384, 28672] route to Triton (fwd-only regime inversion measured "
        "0.76-0.97x there; 2-byte and grad-carrying calls keep CuTe)."
    ),
)
def kl_div_cutedsl(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduction: str = "batchmean",
    log_target: bool = False,
    eps: float = 1e-10,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """CuTe DSL KL-divergence dispatch entry point.

    ``mode`` is accepted only for API parity with the other backends; the only
    valid value is ``"default"`` (or ``None``).
    """
    if mode not in (None, "default"):
        raise ValueError(f"CuTe DSL kl_div has only mode='default'; got mode={mode!r}.")
    if reduction not in ("none", "sum", "mean", "batchmean"):
        # Match the canonical Triton path, which rejects unknown reductions via
        # `_str_to_reduction_mode[reduction]`. Without this the CuTe DSL forward
        # would silently fall through to a row-sum and compute a different loss.
        raise ValueError(f"reduction must be one of 'none', 'sum', 'mean', 'batchmean'; got {reduction!r}.")
    if reduction == "none":
        from liger_kernel.ops.backends._triton.kl_div import kl_div_triton

        emit_fallback_warning(
            "kl_div",
            "nvidia-cutedsl",
            "nvidia-triton",
            "reduction='none' requires per-element output",
        )
        return kl_div_triton(
            y_pred,
            y_true,
            reduction,
            log_target,
            eps,
            mode=mode,
        )
    both_fp32 = y_pred.element_size() > 2 and y_true.element_size() > 2
    fwd_limit = _FWD_MAX_TILE_CUTEDSL_FP32 if both_fp32 else _FWD_MAX_TILE_CUTEDSL
    if y_pred.shape[-1] > fwd_limit:
        from liger_kernel.ops.backends._triton.kl_div import kl_div_triton

        emit_fallback_warning(
            "kl_div",
            "nvidia-cutedsl",
            "nvidia-triton",
            f"vocab size {y_pred.shape[-1]} exceeds CuTe DSL fwd limit {fwd_limit} for 4-byte-pair inputs",
        )
        return kl_div_triton(
            y_pred,
            y_true,
            reduction,
            log_target,
            eps,
            mode=mode,
        )
    if (
        both_fp32
        and _B300_FP32_NO_GRAD_TRITON_LO <= y_pred.shape[-1] <= _B300_FP32_NO_GRAD_TRITON_HI
        and not (y_pred.requires_grad or y_true.requires_grad)
        and _max_capability() > 100
    ):
        from liger_kernel.ops.backends._triton.kl_div import kl_div_triton

        emit_fallback_warning(
            "kl_div",
            "nvidia-cutedsl",
            "nvidia-triton",
            f"no-grad 4-byte-pair input in the B300 sm_103 Triton-preference band "
            f"[{_B300_FP32_NO_GRAD_TRITON_LO}, {_B300_FP32_NO_GRAD_TRITON_HI}] "
            f"(V={y_pred.shape[-1]}); forward-only CuTe DSL measured 0.76-0.97x "
            f"vs Triton there",
        )
        return kl_div_triton(
            y_pred,
            y_true,
            reduction,
            log_target,
            eps,
            mode=mode,
        )
    return _LigerKLDivCuTeDSLFunction.apply(y_pred, y_true, reduction, log_target, eps)
