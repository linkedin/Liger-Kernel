# This file contains code adapted from Quack
# (https://github.com/Dao-AILab/quack), Apache License 2.0.
# Copyright (c) 2025 Wentao Guo, Ted Zadouri, Tri Dao.
# Modifications by Liger-Kernel contributors.
"""Inlined CuTe DSL RMSNorm / LayerNorm forward kernel.

Source: ``quack/rmsnorm.py`` lines 36-470 — specifically the ``RMSNorm``
class (which doubles as the LayerNorm forward when constructed with
``is_layernorm=True``), the ``_compile_rmsnorm_fwd`` function (here
renamed and inlined as a module-level dict cache), and the public
``rmsnorm_fwd`` / ``layernorm_fwd`` wrappers.

Differences vs the upstream module:

- We **drop** the ``@torch.library.custom_op`` registration wrapper
  (``quack::_rmsnorm_fwd``). Liger's outer ``torch.autograd.Function``
  already handles the dispatch — adding a custom op layer here would
  collide with Quack's name if both were installed, and the custom-op
  isn't necessary outside ``torch.compile`` tracing.
- We **drop** the persistent on-disk cache (``@jit_cache``). Liger's
  CuTe-DSL backends already keep a per-process dict for the
  **backward** kernel; we use the same pattern for the forward.
- Residual / bias / per-head paths are **kept** in the kernel body
  (changing the kernel signature would force a recompile of the
  numerically equivalent reduced path; carrying the optional
  arguments through is essentially free).
"""

import math

from typing import Optional
from typing import Tuple
from typing import Type

import cuda.bindings.driver as cuda  # noqa: F401  (kernel signature references cuda.CUstream via cute.compile)
import cutlass
import cutlass.cute as cute
import torch

from cutlass import Float32
from cutlass import const_expr
from torch import Tensor

from . import copy_utils
from . import layout_utils
from .compile_utils import make_fake_tensor as fake_tensor
from .dtype_map import torch2cute_dtype_map
from .reduce import row_reduce
from .reduction_base import ReductionBase


# ===========================================================================
# Forward kernel
# ===========================================================================
class RMSNorm(ReductionBase):
    """RMSNorm / LayerNorm forward — single-kernel implementation.

    When ``is_layernorm=True`` the kernel runs the two-pass
    mean+variance reduction; otherwise it runs the RMSNorm
    sum-of-squares reduction.
    """

    def __init__(self, dtype: Type[cutlass.Numeric], N: int, is_layernorm: bool = False):
        super().__init__(dtype, N, stage=2 if is_layernorm else 1)
        self.is_layernorm = is_layernorm
        self.reload_from = None if N <= (16384 if is_layernorm else 8192) else "smem"
        self.delay_w_load = False

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        # cutlass-cute moved Arch from top-level to the ``arch`` submodule in
        # the cuda13.2 wheel series. Support both for forward/back compat.
        try:
            from cutlass.base_dsl import Arch
        except ImportError:
            from cutlass.base_dsl.arch import Arch

        arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
        # SM8x (Ampere/Ada) lacks cluster support
        if arch < Arch.sm_90:
            self.cluster_n = 1
            return
        # SM12x supports cluster up to 8
        max_cluster = 8 if arch.major == 12 else 16
        N = self.N
        # cluster_n = 4 is faster and cluster_n = 2 for N=64k for some reason
        # Similarly cluster_n = 8 is faster for N=128k
        if arch.major == 12 and const_expr(self.dtype.width >= 32):
            # SM12x 99 KB SMEM: fp32 needs tighter clustering (conservative for residual case)
            thresholds = [(8 * 1024, 1), (16 * 1024, 2), (32 * 1024, 4), (64 * 1024, 8)]
        elif const_expr(self.dtype.width == 16):
            thresholds = [(16 * 1024, 1), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8)]
        else:
            thresholds = [(32 * 1024, 1), (64 * 1024, 2), (128 * 1024, 4), (256 * 1024, 8)]
        for limit, cluster in thresholds:
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = max_cluster

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # (b, N) or (b, H, N)
        mW: Optional[cute.Tensor],  # (N,) or (H, N)
        mB: Optional[cute.Tensor],  # (N,) or (H, N)
        mRes: Optional[cute.Tensor],  # (b, N) or (b, H, N)
        mO: cute.Tensor,  # (b, N) or (b, H, N)
        mResO: Optional[cute.Tensor],
        mRstd: Optional[cute.Tensor],
        mMean: Optional[cute.Tensor],
        eps: Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(*(t.element_type.width for t in [mX, mRes, mW, mB, mO, mResO] if t is not None))
        )
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        mW, mB = [
            layout_utils.expand(mT, dim=0, size=tiler_mn[0]) if const_expr(mT is not None) else None for mT in (mW, mB)
        ]
        mRstd, mMean = [
            layout_utils.expand(mT, dim=cute.rank(mT), size=self.N) if const_expr(mT is not None) else None
            for mT in (mRstd, mMean)
        ]
        num_heads = mX.shape[1] if const_expr(cute.rank(mX) == 3) else 1
        self.kernel(mX, mW, mB, mRes, mO, mResO, mRstd, mMean, eps, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, num_heads],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mB: Optional[cute.Tensor],
        mRes: Optional[cute.Tensor],
        mO: cute.Tensor,
        mResO: Optional[cute.Tensor],
        mRstd: Optional[cute.Tensor],
        mMean: Optional[cute.Tensor],
        eps: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        from functools import partial

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, bidz = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16)
        if const_expr(mRes is not None):
            sRes = smem.allocate_tensor(
                mRes.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        # Slice per head
        if const_expr(cute.rank(mX) == 3):
            mX, mW, mB, mRes, mO, mResO, mRstd, mMean = [
                mT[None, bidz, None] if const_expr(mT is not None) else None
                for mT in (mX, mW, mB, mRes, mO, mResO, mRstd, mMean)
            ]

        shape = (cute.size(mX, mode=[0]), cute.size(mX, mode=[1]))
        idX = cute.make_identity_tensor(shape)
        # Slice for CTAs
        gX, gRes, gO, gResO, gRstd, gMean, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) if mT is not None else None
            for mT in (mX, mRes, mO, mResO, mRstd, mMean, idX)
        ]
        gW, gB = [
            cute.local_tile(mT, tiler_mn, (0, cluster_y)) if const_expr(mT is not None) else None for mT in (mW, mB)
        ]

        thr_copy_X = tiled_copy.get_slice(tidx)

        tXgW = thr_copy_X.partition_S(gW) if const_expr(mW is not None) else None
        tXgB = thr_copy_X.partition_S(gB) if const_expr(mB is not None) else None
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        if const_expr(mRes is not None):
            tXgRes = thr_copy_X.partition_S(gRes)
            tXsRes = thr_copy_X.partition_D(sRes)
        tXgO = thr_copy_X.partition_D(gO)
        if const_expr(mResO is not None):
            tXgResO = thr_copy_X.partition_D(gResO)
        tXrRstd = thr_copy_X.partition_D(gRstd) if const_expr(mRstd is not None) else None
        tXrMean = thr_copy_X.partition_D(gMean) if const_expr(mMean is not None) else None
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        # allocate fragments for gmem->rmem
        tXrW = cute.make_rmem_tensor_like(tXgW) if const_expr(mW is not None) else None
        tXrB = cute.make_rmem_tensor_like(tXgB) if const_expr(mB is not None) else None
        tXrX, tXrO = [cute.make_rmem_tensor_like(t) for t in (tXgX, tXgO)]
        if const_expr(mRes is not None):
            tXrRes = cute.make_rmem_tensor_like(tXgRes)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = copy_utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1]) if not is_even_N else None
        # Each copy will use the same predicate
        copy = partial(copy_utils.copy, pred=tXpX)

        row = tXcX[0][0]
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
            if const_expr(mRes is not None):
                copy(tXgRes, tXsRes, is_async=True)
        cute.arch.cp_async_commit_group()

        if const_expr(not self.delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)
            if const_expr(mB is not None):
                copy(tXgB, tXrB)

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)
        if const_expr(mRes is not None):
            cute.autovec_copy(tXsRes, tXrRes)
            x += tXrRes.load().to(cute.Float32)
        if const_expr(mResO is not None):
            tXrResO = cute.make_rmem_tensor_like(tXgResO)
            tXrResO.store(x.to(tXrResO.element_type))
            if row < shape[0]:
                copy(tXrResO, tXgResO)

        mean, rstd = None, None
        if const_expr(self.is_layernorm):
            # LayerNorm: compute mean first, then variance
            sum_x = row_reduce(
                x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr + 0 if const_expr(self.cluster_n > 1) else None,
                init_val=0.0,
                hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
            )
            mean = sum_x / shape[1]
            if const_expr(mMean is not None):
                # Only the thread corresponding to column 0 writes out the mean to gmem
                if (
                    tXcX[0][1] == 0
                    and row < shape[0]
                    and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
                ):
                    tXrMean[0] = mean
            if const_expr(self.reload_from == "smem"):
                cute.autovec_copy(tXsX, tXrX)
                x = tXrX.load().to(cute.Float32)
                if const_expr(mRes is not None):
                    cute.autovec_copy(tXsRes, tXrRes)
                    x += tXrRes.load().to(cute.Float32)
            elif const_expr(self.reload_from == "gmem"):
                copy(tXgX, tXrX)
                x = tXrX.load().to(cute.Float32)
                if const_expr(mRes is not None):
                    copy(tXgRes, tXrRes)
                    x += tXrRes.load().to(cute.Float32)
            sum_sq_x_sub_mean = row_reduce(
                (x - mean) * (x - mean),
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 1],
                mbar_ptr + 1 if const_expr(self.cluster_n > 1) else None,
                init_val=0.0,
            )
            rstd = cute.math.rsqrt(sum_sq_x_sub_mean / shape[1] + eps, fastmath=True)
        else:
            # RMSNorm: compute sum of squares directly
            mean = const_expr(0.0)
            sum_sq_x = row_reduce(
                x * x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                init_val=0.0,
                hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
            )
            rstd = cute.math.rsqrt(sum_sq_x / shape[1] + eps, fastmath=True)
        if const_expr(mRstd is not None):
            # Only the thread corresponding to column 0 writes out the rstd to gmem
            if tXcX[0][1] == 0 and row < shape[0] and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0):
                tXrRstd[0] = rstd
        if const_expr(self.delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)
            if const_expr(mB is not None):
                copy(tXgB, tXrB)
        if const_expr(self.reload_from == "smem" or self.reload_from == "gmem"):
            if const_expr(self.reload_from == "smem"):
                cute.autovec_copy(tXsX, tXrX)
                if const_expr(mRes is not None):
                    cute.autovec_copy(tXsRes, tXrRes)
            else:
                copy(tXgX, tXrX)
                if const_expr(mRes is not None):
                    copy(tXgRes, tXrRes)
            x = tXrX.load().to(cute.Float32)
            if const_expr(mRes is not None):
                x += tXrRes.load().to(cute.Float32)
        x_hat = (x - mean) * rstd if const_expr(self.is_layernorm) else x * rstd
        y = x_hat
        if const_expr(mW is not None):
            y *= tXrW.load().to(cute.Float32)
        if const_expr(mB is not None):
            y += tXrB.load().to(cute.Float32)
        tXrO.store(y.to(tXrO.element_type))
        if row < shape[0]:
            copy(tXrO, tXgO)


# ---------------------------------------------------------------------------
# Compile cache. In-memory only — keep startup latency low for downstream
# callers but skip the persistent ``.o`` cache the upstream module ships
# (which depends on ``tvm_ffi`` and writes to ``/tmp``).
# ---------------------------------------------------------------------------
_FWD_COMPILE_CACHE: dict = {}


def _compile_fwd(
    dtype,
    out_dtype,
    res_dtype,
    weight_dtype,
    bias_dtype,
    res_out_dtype,
    N,
    has_rstd,
    has_mean,
    is_layernorm,
    per_head,
):
    """Compile the RMSNorm/LayerNorm forward kernel for a given dtype mix.

    Cache key is the full argument tuple — ``cute.compile`` is multi-second
    so we deduplicate across calls. Matches the body of
    ``quack.rmsnorm._compile_rmsnorm_fwd``.
    """
    key = (
        dtype,
        out_dtype,
        res_dtype,
        weight_dtype,
        bias_dtype,
        res_out_dtype,
        N,
        has_rstd,
        has_mean,
        is_layernorm,
        per_head,
    )
    if key in _FWD_COMPILE_CACHE:
        return _FWD_COMPILE_CACHE[key]

    batch_sym = cute.sym_int()
    head_sym = cute.sym_int() if per_head else None
    batch_shape = (batch_sym, head_sym) if per_head else (batch_sym,)
    all_dtypes = [dtype, out_dtype, res_dtype, weight_dtype, bias_dtype, res_out_dtype]
    div = math.gcd(N, *(128 // dt.width for dt in all_dtypes if dt is not None))
    x_cute, out_cute, res_cute, res_out_cute = [
        fake_tensor(dt, (*batch_shape, N), div) for dt in [dtype, out_dtype, res_dtype, res_out_dtype]
    ]
    weight_shape = (head_sym, N) if per_head else (N,)
    weight_cute, bias_cute = [fake_tensor(dt, weight_shape, div) for dt in [weight_dtype, bias_dtype]]
    rstd_cute = fake_tensor(Float32, batch_shape) if has_rstd else None
    mean_cute = fake_tensor(Float32, batch_shape) if has_mean else None
    compiled = cute.compile(
        RMSNorm(dtype, N, is_layernorm=is_layernorm),
        x_cute,
        weight_cute,
        bias_cute,
        res_cute,
        out_cute,
        res_out_cute,
        rstd_cute,
        mean_cute,
        Float32(0),  # eps, just for compilation
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _FWD_COMPILE_CACHE[key] = compiled
    return compiled


def _run_fwd(
    x: Tensor,
    weight: Optional[Tensor],
    out: Tensor,
    bias: Optional[Tensor],
    rstd: Optional[Tensor],
    mean: Optional[Tensor],
    residual: Optional[Tensor],
    residual_out: Optional[Tensor],
    eps: float,
    is_layernorm: bool,
) -> None:
    """Run the compiled kernel. Mirrors ``quack.rmsnorm._rmsnorm_fwd`` but
    drops the ``torch.library.custom_op`` registration — we are always
    invoked from inside an ``autograd.Function``, never traced by Dynamo.
    """
    supported_types = {torch.float16, torch.bfloat16, torch.float32}
    assert x.dtype in supported_types, "Unsupported dtype"
    if weight is not None:
        assert weight.dtype in supported_types, "Weight must be float32, float16 or bfloat16"
    if residual is not None:
        assert residual.dtype in supported_types, "Residual must be float16, bfloat16, or float32"

    N = x.size(-1)
    per_head = (weight is not None and weight.dim() == 2) or (bias is not None and bias.dim() == 2)
    dtype, out_dtype, weight_dtype, bias_dtype, res_dtype, res_out_dtype = [
        torch2cute_dtype_map[t.dtype] if t is not None else None for t in [x, out, weight, bias, residual, residual_out]
    ]
    _compile_fwd(
        dtype,
        out_dtype,
        res_dtype,
        weight_dtype,
        bias_dtype,
        res_out_dtype,
        N,
        rstd is not None,
        mean is not None,
        is_layernorm,
        per_head,
    )(x, weight, bias, residual, out, residual_out, rstd, mean, eps)


# ===========================================================================
# Public host-side wrappers — drop-in for ``quack.rmsnorm.{rmsnorm_fwd,layernorm_fwd}``.
# ===========================================================================
def rmsnorm_fwd(
    x: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    store_rstd: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """RMSNorm forward. Drop-in replacement for ``quack.rmsnorm.rmsnorm_fwd``.

    Returns ``(out, residual_out, rstd)``. ``residual_out`` aliases ``x``
    when there is no residual addition and no dtype change — matches the
    upstream contract.
    """
    out_dtype = x.dtype if out_dtype is None else out_dtype
    out = torch.empty_like(x, dtype=out_dtype)
    rstd = torch.empty(*x.shape[:-1], device=x.device, dtype=torch.float32) if store_rstd else None
    if residual is not None:
        residual_dtype = residual.dtype
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty_like(x, dtype=residual_dtype if residual_dtype is not None else x.dtype)
    else:
        residual_out = None
    _run_fwd(x, weight, out, bias, rstd, None, residual, residual_out, eps, False)
    # residual_out is None if residual is None and residual_dtype == input_dtype
    if residual_out is None:
        residual_out = x
    return out, residual_out, rstd


def layernorm_fwd(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    eps: float = 1e-6,
    return_rstd: bool = False,
    return_mean: bool = False,
):
    """LayerNorm forward. Drop-in replacement for ``quack.rmsnorm.layernorm_fwd``.

    Uses the unified RMSNorm/LayerNorm kernel (``is_layernorm=True`` flag
    flips the two-pass mean+variance reduction).
    """
    assert x.dim() == 2, "Input must be 2D"
    assert weight.dim() == 1, "Weight must be 1D"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    # The unified RMSNorm kernel promotes weight/bias to fp32 in-register
    # (``tXrW.load().to(cute.Float32)``) and ``_run_fwd`` already accepts
    # fp16/bf16/fp32, so callers may pass them in their native dtype. The
    # fp16/bf16 -> fp32 in-register promotion is exact and numerically
    # identical to a host-side ``.to(torch.float32)`` up-cast.
    assert weight.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported weight dtype"
    if bias is not None:
        assert bias.dim() == 1, "Bias must be 1D"
        assert bias.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported bias dtype"

    M, N = x.shape
    device = x.device
    out = torch.empty_like(x)
    rstd = torch.empty(M, device=device, dtype=torch.float32) if return_rstd else None
    mean = torch.empty(M, device=device, dtype=torch.float32) if return_mean else None

    _run_fwd(x, weight, out, bias, rstd, mean, None, None, eps, True)

    if return_rstd and return_mean:
        return out, rstd, mean
    elif return_rstd:
        return out, rstd
    elif return_mean:
        return out, mean
    return out
