# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Softmax kernel (cuTile backend).

Row-wise softmax with an exp2-based online formulation. Two regimes, all autotuned
(occupancy + num_worker_warps) and cached per shape:
  - single-chunk: whole row in one tile. f32 uses a TMA load; bf16/fp16 use gather (aligned
    fast path when n_cols is a power of 2).
  - multi-chunk: 2-pass online softmax that re-reads the row; block size chosen for L2 reuse.

The single-chunk threshold and multi-chunk block size both differ by direction. Forward (light:
max/sum/exp/div) scales to a 32768 tile; backward (holds y+dy+dot+dx) spills past 16384, so larger
rows use the multi-chunk path. The multi-chunk backward re-read is L2-bound, so it uses a larger
block (8192 vs the forward's 4096) to shrink the pass1->pass2 gap. Backward: dx = y*(dy - dot),
dot = sum(y*dy).
"""

import math

from types import SimpleNamespace

import cuda.tile as ct
import torch

from cuda.tile.tune import exhaustive_search

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2

INV_LOG2 = 1.0 / math.log(2)

# Single-chunk handles the whole row in one tile. Forward is light (max/sum/exp/div over one
# tile) and scales to a 32768 tile; backward holds y+dy+dot+dx and spills past 16384, so it
# uses a lower threshold and routes larger rows to the (autotuned) multi-chunk online loop.
_SINGLE_CHUNK_MAX_N_FWD = 32768
_SINGLE_CHUNK_MAX_N_BWD = 16384

# Forward autotune: exp2+APPROX kernel has different register profile; nww=8/occ>=5
# cause register spills at large N. Safe space: occ=[2,3,4], nww=4 only.
_SOFTMAX_FWD_TUNE_CONFIGS = [SimpleNamespace(occ=o, nww=4) for o in [2, 3, 4]]

# Backward autotune: no exp, lighter register pressure — nww=8 beneficial at large N.
_SOFTMAX_BWD_TUNE_CONFIGS = [SimpleNamespace(occ=o, nww=n) for o in [2, 3, 4, 5, 6, 8] for n in [4, 8]]

# Multi-chunk block sizes differ by direction. The online loop re-reads the row in pass 2, so
# its cost is dominated by L2 reuse on that re-read. For backward, BLOCK=8192 (vs 4096) halves
# the pass1->pass2 gap and reaches ~parity with the Triton single-block backward across large N.
# Forward is lighter and keeps BLOCK=4096 (8192 regresses the largest rows). occupancy+nww are
# autotuned per shape on top.
_MULTI_CHUNK_BLOCK_SIZE_FWD = 4096
_MULTI_CHUNK_BLOCK_SIZE_BWD = 8192
_SOFTMAX_MULTI_TUNE_CONFIGS = [SimpleNamespace(occ=o, nww=n) for o in [2, 3, 4, 6, 8] for n in [4, 8]]

# Per-process cache: (path, n_cols, BLOCK_SIZE[, aligned]) -> tuned kernel
_SOFTMAX_FWD_TUNE_CACHE: dict = {}
_SOFTMAX_BWD_TUNE_CACHE: dict = {}
_SOFTMAX_MULTI_TUNE_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Single-chunk forward kernels (plain functions, wrapped by ct.kernel below).
# No hardcoded occupancy — autotuned at runtime.
# ---------------------------------------------------------------------------


@ct.kernel
def _softmax_single_tma(Y, X, n_cols: ct.Constant[int], BLOCK_SIZE: ct.Constant[int]):
    """f32, TMA load with NEG_INF padding. Scatter for write (check_bounds=True handles tail)."""
    row_idx = ct.bid(0)
    x_tile = ct.astype(
        ct.load(X, index=(row_idx, 0), shape=(1, BLOCK_SIZE), padding_mode=ct.PaddingMode.NEG_INF).reshape(
            (BLOCK_SIZE,)
        ),
        ct.float32,
    )
    global_max = ct.max(x_tile, 0, keepdims=False)
    exp_tile = ct.exp2(ct.mul(x_tile - global_max, INV_LOG2, flush_to_zero=True), flush_to_zero=True)
    y_tile = ct.truediv(
        exp_tile, ct.sum(exp_tile, 0, keepdims=False), rounding_mode=ct.RoundingMode.APPROX, flush_to_zero=True
    )
    ct.scatter(Y, (row_idx, ct.arange(BLOCK_SIZE, dtype=ct.int32)), ct.astype(y_tile, Y.dtype), check_bounds=True)


@ct.kernel
def _softmax_single_gather(Y, X, n_cols: ct.Constant[int], BLOCK_SIZE: ct.Constant[int], ALIGNED: ct.Constant[bool]):
    """bf16/fp16. ALIGNED=True: check_bounds=False (power-of-2 n_cols). ALIGNED=False: padded gather."""
    row_idx = ct.bid(0)
    col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32)
    if ALIGNED:
        x_tile = ct.astype(ct.gather(X, (row_idx, col_idx), check_bounds=False), ct.float32)
    else:
        x_tile = ct.astype(ct.gather(X, (row_idx, col_idx), check_bounds=True, padding_value=-math.inf), ct.float32)
    global_max = ct.max(x_tile, 0, keepdims=False)
    exp_tile = ct.exp2(ct.mul(x_tile - global_max, INV_LOG2, flush_to_zero=True), flush_to_zero=True)
    y_tile = ct.truediv(
        exp_tile, ct.sum(exp_tile, 0, keepdims=False), rounding_mode=ct.RoundingMode.APPROX, flush_to_zero=True
    )
    if ALIGNED:
        ct.scatter(Y, (row_idx, col_idx), ct.astype(y_tile, Y.dtype), check_bounds=False)
    else:
        ct.scatter(Y, (row_idx, col_idx), ct.astype(y_tile, Y.dtype), check_bounds=True)


def _get_tuned_single_kernel(n_cols: int, BLOCK_SIZE: int, n_rows: int, dtype: torch.dtype, stream, y2d, x2d):
    """Autotune occupancy+nww on first call; return cached kernel on subsequent calls."""
    is_tma = dtype == torch.float32
    is_aligned = n_cols == BLOCK_SIZE

    if is_tma:
        key = ("tma", n_cols, BLOCK_SIZE)
        base, args_fn = _softmax_single_tma, lambda cfg: (y2d, x2d, int(n_cols), int(BLOCK_SIZE))
    else:
        key = ("gather", n_cols, BLOCK_SIZE, is_aligned)
        base = _softmax_single_gather
        args_fn = lambda cfg: (y2d, x2d, int(n_cols), int(BLOCK_SIZE), is_aligned)

    if key in _SOFTMAX_FWD_TUNE_CACHE:
        return _SOFTMAX_FWD_TUNE_CACHE[key]

    result = exhaustive_search(
        _SOFTMAX_FWD_TUNE_CONFIGS,
        stream,
        lambda cfg: (n_rows, 1, 1),
        base,
        args_fn,
        lambda cfg: {"occupancy": cfg.occ, "num_worker_warps": cfg.nww},
        quiet=True,
    )
    best = result.best.config
    _SOFTMAX_FWD_TUNE_CACHE[key] = base.replace_hints(occupancy=best.occ, num_worker_warps=best.nww)
    return _SOFTMAX_FWD_TUNE_CACHE[key]


# ---------------------------------------------------------------------------
# Multi-chunk forward kernel (n_cols > _SINGLE_CHUNK_MAX_N)
# ---------------------------------------------------------------------------


@ct.kernel
def _softmax_fwd_ct(Y, X, n_cols: ct.Constant[int], BLOCK_SIZE: ct.Constant[int]):
    """2-pass online softmax. BLOCK_SIZE=4096 caps register pressure regardless of n_cols."""
    row_idx = ct.bid(0)
    n_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    running_max = ct.full((1,), -math.inf, dtype=ct.float32)
    running_sum = ct.full((1,), 0.0, dtype=ct.float32)
    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        x_tile = ct.astype(ct.gather(X, (row_idx, col_idx), check_bounds=True, padding_value=-math.inf), ct.float32)
        blk_max = ct.max(x_tile, 0, keepdims=True)
        new_max = ct.maximum(running_max, blk_max)
        running_sum = running_sum * ct.exp2((running_max - new_max) * INV_LOG2) + ct.sum(
            ct.exp2((x_tile - new_max) * INV_LOG2), 0, keepdims=True
        )
        running_max = new_max

    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        x_tile = ct.astype(ct.gather(X, (row_idx, col_idx), check_bounds=True, padding_value=-math.inf), ct.float32)
        y_tile = ct.exp2((x_tile - running_max) * INV_LOG2) / running_sum
        ct.scatter(Y, (row_idx, col_idx), ct.astype(y_tile, Y.dtype), check_bounds=True)


# ---------------------------------------------------------------------------
# Backward kernels
# ---------------------------------------------------------------------------


@ct.kernel
def _softmax_bwd_ct(DX, DY, Y, n_cols: ct.Constant[int], BLOCK_SIZE: ct.Constant[int], ALIGNED: ct.Constant[bool]):
    """Multi-chunk backward. dx = y * (dy - dot), dot = sum(y*dy). 2-pass fold."""
    row_idx = ct.bid(0)
    n_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    dot_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        if ALIGNED:
            y_tile = ct.astype(ct.gather(Y, (row_idx, col_idx), check_bounds=False), ct.float32)
            dy_tile = ct.astype(ct.gather(DY, (row_idx, col_idx), check_bounds=False), ct.float32)
        else:
            y_tile = ct.astype(ct.gather(Y, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
            dy_tile = ct.astype(ct.gather(DY, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        dot_tile = ct.add(dot_tile, y_tile * dy_tile)

    dot = ct.sum(dot_tile, 0, keepdims=False)
    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        if ALIGNED:
            y_tile = ct.astype(ct.gather(Y, (row_idx, col_idx), check_bounds=False), ct.float32)
            dy_tile = ct.astype(ct.gather(DY, (row_idx, col_idx), check_bounds=False), ct.float32)
            ct.scatter(DX, (row_idx, col_idx), ct.astype(y_tile * (dy_tile - dot), DX.dtype), check_bounds=False)
        else:
            y_tile = ct.astype(ct.gather(Y, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
            dy_tile = ct.astype(ct.gather(DY, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
            ct.scatter(DX, (row_idx, col_idx), ct.astype(y_tile * (dy_tile - dot), DX.dtype), check_bounds=True)


@ct.kernel
def _softmax_bwd_fused(DX, DY, Y, n_cols: ct.Constant[int], BLOCK_SIZE: ct.Constant[int], ALIGNED: ct.Constant[bool]):
    """Single-chunk fused backward: loads Y+DY once, computes dot in registers, then dx.

    ALIGNED=True (power-of-2 n_cols): check_bounds=False eliminates predicate overhead.
    """
    row_idx = ct.bid(0)
    col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32)
    if ALIGNED:
        y_tile = ct.astype(ct.gather(Y, (row_idx, col_idx), check_bounds=False), ct.float32)
        dy_tile = ct.astype(ct.gather(DY, (row_idx, col_idx), check_bounds=False), ct.float32)
        dot = ct.sum(y_tile * dy_tile, 0, keepdims=False)
        ct.scatter(DX, (row_idx, col_idx), ct.astype(y_tile * (dy_tile - dot), DX.dtype), check_bounds=False)
    else:
        y_tile = ct.astype(ct.gather(Y, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        dy_tile = ct.astype(ct.gather(DY, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        dot = ct.sum(y_tile * dy_tile, 0, keepdims=False)
        ct.scatter(DX, (row_idx, col_idx), ct.astype(y_tile * (dy_tile - dot), DX.dtype), check_bounds=True)


def _get_tuned_bwd_kernel(n_cols: int, BLOCK_SIZE: int, n_rows: int, dtype: torch.dtype, device):
    """Autotune occupancy+nww for single-chunk backward; cache by (n_cols, BLOCK_SIZE, aligned)."""
    is_aligned = n_cols == BLOCK_SIZE
    key = (n_cols, BLOCK_SIZE, is_aligned)
    if key in _SOFTMAX_BWD_TUNE_CACHE:
        return _SOFTMAX_BWD_TUNE_CACHE[key]

    # Use a fresh stream for timing — autograd backward stream is not safe for exhaustive_search.
    tune_stream = torch.cuda.Stream(device=device)
    dx = torch.empty(n_rows, BLOCK_SIZE, dtype=dtype, device=device)
    dy = torch.empty(n_rows, BLOCK_SIZE, dtype=dtype, device=device)
    y = torch.empty(n_rows, BLOCK_SIZE, dtype=dtype, device=device)

    result = exhaustive_search(
        _SOFTMAX_BWD_TUNE_CONFIGS,
        tune_stream,
        lambda cfg: (n_rows, 1, 1),
        _softmax_bwd_fused,
        lambda cfg: (dx, dy, y, int(n_cols), int(BLOCK_SIZE), is_aligned),
        lambda cfg: {"occupancy": cfg.occ, "num_worker_warps": cfg.nww},
        quiet=True,
    )
    best = result.best.config
    _SOFTMAX_BWD_TUNE_CACHE[key] = _softmax_bwd_fused.replace_hints(occupancy=best.occ, num_worker_warps=best.nww)
    return _SOFTMAX_BWD_TUNE_CACHE[key]


def _get_tuned_multi_fwd_kernel(n_cols: int, BLOCK_SIZE: int, n_rows: int, stream, y2d, x2d):
    """Autotune occupancy+nww for the multi-chunk forward; cache by (n_cols, BLOCK_SIZE)."""
    key = ("multi_fwd", n_cols, BLOCK_SIZE)
    if key in _SOFTMAX_MULTI_TUNE_CACHE:
        return _SOFTMAX_MULTI_TUNE_CACHE[key]
    result = exhaustive_search(
        _SOFTMAX_MULTI_TUNE_CONFIGS,
        stream,
        lambda cfg: (n_rows, 1, 1),
        _softmax_fwd_ct,
        lambda cfg: (y2d, x2d, int(n_cols), int(BLOCK_SIZE)),
        lambda cfg: {"occupancy": cfg.occ, "num_worker_warps": cfg.nww},
        quiet=True,
    )
    best = result.best.config
    _SOFTMAX_MULTI_TUNE_CACHE[key] = _softmax_fwd_ct.replace_hints(occupancy=best.occ, num_worker_warps=best.nww)
    return _SOFTMAX_MULTI_TUNE_CACHE[key]


def _get_tuned_multi_bwd_kernel(n_cols: int, BLOCK_SIZE: int, n_rows: int, aligned: bool, dtype, device):
    """Autotune occupancy+nww for the multi-chunk backward; cache by (n_cols, BLOCK_SIZE, aligned)."""
    key = ("multi_bwd", n_cols, BLOCK_SIZE, aligned)
    if key in _SOFTMAX_MULTI_TUNE_CACHE:
        return _SOFTMAX_MULTI_TUNE_CACHE[key]
    tune_stream = torch.cuda.Stream(device=device)
    dx = torch.empty(n_rows, n_cols, dtype=dtype, device=device)
    dy = torch.empty(n_rows, n_cols, dtype=dtype, device=device)
    y = torch.empty(n_rows, n_cols, dtype=dtype, device=device)
    result = exhaustive_search(
        _SOFTMAX_MULTI_TUNE_CONFIGS,
        tune_stream,
        lambda cfg: (n_rows, 1, 1),
        _softmax_bwd_ct,
        lambda cfg: (dx, dy, y, int(n_cols), int(BLOCK_SIZE), aligned),
        lambda cfg: {"occupancy": cfg.occ, "num_worker_warps": cfg.nww},
        quiet=True,
    )
    best = result.best.config
    _SOFTMAX_MULTI_TUNE_CACHE[key] = _softmax_bwd_ct.replace_hints(occupancy=best.occ, num_worker_warps=best.nww)
    return _SOFTMAX_MULTI_TUNE_CACHE[key]


# ---------------------------------------------------------------------------
# Host-side dispatch
# ---------------------------------------------------------------------------


def _softmax_forward_ct(x: torch.Tensor):
    *batch, n_cols = x.shape
    x2d = x.contiguous().view(-1, n_cols)
    n_rows = x2d.shape[0]
    y2d = torch.empty_like(x2d)
    stream = torch.cuda.current_stream()

    if n_cols <= _SINGLE_CHUNK_MAX_N_FWD:
        BLOCK_SIZE = min(_next_power_of_2(n_cols), 65536)
        is_tma = x2d.dtype == torch.float32
        is_aligned = n_cols == BLOCK_SIZE
        kernel = _get_tuned_single_kernel(n_cols, BLOCK_SIZE, n_rows, x2d.dtype, stream, y2d, x2d)
        if is_tma:
            ct.launch(stream, (n_rows, 1, 1), kernel, (y2d, x2d, int(n_cols), int(BLOCK_SIZE)))
        else:
            ct.launch(stream, (n_rows, 1, 1), kernel, (y2d, x2d, int(n_cols), int(BLOCK_SIZE), is_aligned))
    else:
        kernel = _get_tuned_multi_fwd_kernel(n_cols, _MULTI_CHUNK_BLOCK_SIZE_FWD, n_rows, stream, y2d, x2d)
        ct.launch(stream, (n_rows, 1, 1), kernel, (y2d, x2d, int(n_cols), _MULTI_CHUNK_BLOCK_SIZE_FWD))

    return y2d.view(*batch, n_cols)


def _softmax_backward_ct(dy: torch.Tensor, y: torch.Tensor):
    *batch, n_cols = dy.shape
    dy2d = dy.contiguous().view(-1, n_cols)
    y2d = y.contiguous().view(-1, n_cols)
    n_rows = dy2d.shape[0]
    dx2d = torch.empty_like(dy2d)
    stream = torch.cuda.current_stream()

    if n_cols <= _SINGLE_CHUNK_MAX_N_BWD:
        BLOCK_SIZE = min(_next_power_of_2(n_cols), 65536)
        aligned = n_cols == BLOCK_SIZE
        kernel = _get_tuned_bwd_kernel(n_cols, BLOCK_SIZE, n_rows, dx2d.dtype, dx2d.device)
        ct.launch(stream, (n_rows, 1, 1), kernel, (dx2d, dy2d, y2d, int(n_cols), int(BLOCK_SIZE), aligned))
    else:
        aligned = (n_cols % _MULTI_CHUNK_BLOCK_SIZE_BWD) == 0
        kernel = _get_tuned_multi_bwd_kernel(
            n_cols, _MULTI_CHUNK_BLOCK_SIZE_BWD, n_rows, aligned, dx2d.dtype, dx2d.device
        )
        ct.launch(stream, (n_rows, 1, 1), kernel, (dx2d, dy2d, y2d, int(n_cols), _MULTI_CHUNK_BLOCK_SIZE_BWD, aligned))

    return dx2d.view(*batch, n_cols)


class LigerSoftmaxFunction(torch.autograd.Function):
    """CuTile autograd wrapper for row-wise softmax."""

    @staticmethod
    def forward(ctx, input_):
        y = _softmax_forward_ct(input_)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        return _softmax_backward_ct(grad_output, y)
