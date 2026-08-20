"""Parallel final partial-dW reduction for CuTeDSL RMSNorm backward."""

import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from cutlass import Float32

from liger_kernel.ops.cutedsl.ops.rms_norm import _compile_cache
from liger_kernel.ops.cutedsl.ops.rms_norm import _cute_stream
from liger_kernel.ops.cutedsl.ops.rms_norm import _to_cute_cached
from liger_kernel.ops.cutedsl.ops.utils import to_cute_tensor

_THREADS = 32
_REDUCTION_WARPS = int(os.environ.get("LIGER_RMS_DW_REDUCTION_WARPS") or 32)


@cute.kernel
def _reduce_dw_partials_kernel(
    mdW: cute.Tensor,
    mdWOut: cute.Tensor,
):
    """One warp reduces one contiguous 32-column tile over every strip."""
    tid, _, _ = cute.arch.thread_idx()
    column_block, _, _ = cute.arch.block_idx()
    column = column_block * _THREADS + tid
    n_strips = mdW.shape[0]
    n_cols = mdW.shape[1]

    accumulator = Float32(0.0)
    if column < n_cols:
        for strip in cutlass.range(0, n_strips):
            accumulator = accumulator + mdW[strip, column].to(Float32)
        mdWOut[column] = accumulator.to(mdWOut.element_type)


@cute.kernel
def _reduce_dw_partials_parallel_kernel(
    mdW: cute.Tensor,
    mdWOut: cute.Tensor,
    NUM_WARPS: cutlass.Constexpr,
):
    """Split each column's strip reduction across multiple warps."""
    tid, _, _ = cute.arch.thread_idx()
    lane = tid % _THREADS
    warp = tid // _THREADS
    column_block, _, _ = cute.arch.block_idx()
    column = column_block * _THREADS + lane
    n_strips = mdW.shape[0]
    n_cols = mdW.shape[1]

    accumulator = Float32(0.0)
    num_iterations = (n_strips + NUM_WARPS - 1) // NUM_WARPS
    if column < n_cols:
        for iteration in cutlass.range(0, num_iterations):
            strip = iteration * NUM_WARPS + warp
            if strip < n_strips:
                accumulator = accumulator + mdW[strip, column].to(Float32)

    smem = cutlass.utils.SmemAllocator()
    partials = smem.allocate_tensor(
        Float32,
        cute.make_layout((NUM_WARPS, _THREADS), stride=(_THREADS, 1)),
        byte_alignment=16,
    )
    partials[warp, lane] = accumulator
    cute.arch.barrier()

    if warp == 0 and column < n_cols:
        total = Float32(0.0)
        for source_warp in cutlass.range_constexpr(NUM_WARPS):
            total = total + partials[source_warp, lane]
        mdWOut[column] = total.to(mdWOut.element_type)


@cute.jit
def _reduce_dw_partials_host(
    mdW: cute.Tensor,
    mdWOut: cute.Tensor,
    stream: cuda.CUstream = None,
):
    _reduce_dw_partials_kernel(mdW, mdWOut).launch(
        grid=[cute.ceil_div(mdW.shape[1], _THREADS), 1, 1],
        block=[_THREADS, 1, 1],
        stream=stream,
    )


@cute.jit
def _reduce_dw_partials_parallel_host(
    mdW: cute.Tensor,
    mdWOut: cute.Tensor,
    NUM_WARPS: cutlass.Constexpr,
    stream: cuda.CUstream = None,
):
    _reduce_dw_partials_parallel_kernel(mdW, mdWOut, NUM_WARPS).launch(
        grid=[cute.ceil_div(mdW.shape[1], _THREADS), 1, 1],
        block=[_THREADS * NUM_WARPS, 1, 1],
        smem=NUM_WARPS * _THREADS * 4,
        stream=stream,
    )


def reduce_dw_partials(dW_partial: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    """Reduce FP32 strip partials and fuse the final output-dtype conversion."""
    if _REDUCTION_WARPS not in (1, 2, 4, 8, 16, 32):
        raise ValueError(f"LIGER_RMS_DW_REDUCTION_WARPS must be one of 1, 2, 4, 8, 16, or 32, got {_REDUCTION_WARPS}")
    output = torch.empty(dW_partial.shape[1], dtype=output_dtype, device=dW_partial.device)
    partial_ct = _to_cute_cached(dW_partial, assumed_align=16)
    output_ct = to_cute_tensor(output, assumed_align=16)
    key = (
        "rms_norm_dw_reduce",
        dW_partial.shape[0],
        dW_partial.shape[1],
        dW_partial.dtype,
        output.dtype,
        _REDUCTION_WARPS,
    )
    stream = _cute_stream()
    if key not in _compile_cache:
        if _REDUCTION_WARPS == 1:
            _compile_cache[key] = cute.compile(
                _reduce_dw_partials_host,
                partial_ct,
                output_ct,
                stream,
            )
        else:
            _compile_cache[key] = cute.compile(
                _reduce_dw_partials_parallel_host,
                partial_ct,
                output_ct,
                _REDUCTION_WARPS,
                stream,
            )
    _compile_cache[key](partial_ct, output_ct, stream)
    return output
