"""cuTile (NVIDIA cuda.tile) backend for ``softmax``.

This module implements row-wise softmax with NVIDIA's cuTile DSL and registers
it via ``@register_op``. It exposes three forward kernel variants, selected by
the ``mode`` argument:

- ``"standard"`` — one row per program block, single tile per row. Loaded via
  ``ct.gather`` with ``padding_value=-inf`` so the max/sum reductions ignore
  the padded lanes. Best for narrow-to-mid rows that fit in a single tile.
- ``"static_persistent"`` — ``NUM_SMS * occupancy`` persistent blocks each
  striding over rows. Amortises launch overhead when ``M`` is much larger than
  ``NUM_SMS``. Single tile per row (``TILE_SIZE = next_pow2(N)``).
- ``"chunked"`` — a 3-pass online kernel for very wide rows (``N`` larger than
  the per-tile cap). Pass 1 finds the row max, pass 2 accumulates the
  denominator, pass 3 writes ``exp(x - m) / d``. Numerically identical to the
  multi-block Triton path.

Backward is a single persistent kernel: ``dx = y * (dy - sum(dy * y))`` with the
dot-product reduction in fp32. It mirrors the Triton backward exactly. For wide
rows the backward kernel runs the same 3-pass-free two-pass structure
(accumulate the dot, then write dx chunk by chunk).

References
----------
- cuTile softmax reference: NVIDIA TileGym
  ``src/tilegym/ops/cutile/softmax.py`` — source of the ``_softmax_kernel``
  (gather + ``padding_value=-inf``), ``_softmax_kernel_tma`` (single
  ``ct.load`` per row), and ``_softmax_kernel_chunked`` (3-pass online) idioms,
  plus the ``min(NUM_SM * k, n_rows)`` persistent-grid occupancy hint.
- Triton reference: ``liger_kernel.ops.softmax`` — defines the exact numerics
  we reproduce bit-for-bit: single-block ``y = exp(x - m) / sum(exp(x - m))``,
  multi-block online running ``(m, d)``, and the backward
  ``dx = y * (dy - sum(dy * y))``.
"""

# NOTE: do NOT use `from __future__ import annotations` here.
# cuTile's @ct.kernel introspects the wrapped function's __annotations__ to
# discover which parameters are Constants (i.e., compile-time specialized).
# PEP-563 future annotations stringifies all annotations, which makes cuTile
# silently miss the Constant flags, so a *single* compiled binary gets reused
# across all callers — yielding catastrophically wrong results when N or
# TILE_SIZE changes between launches. (Same caveat as the cuTile rms_norm /
# jsd siblings; see their headers for the full repro.)

import math

from typing import Optional

import cuda.tile as ct
import torch

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.ops._nvidia_shared import cutile_compiler_available as _cutile_compiler_available
from liger_kernel.ops._nvidia_shared import next_pow2 as _next_pow2
from liger_kernel.ops._nvidia_shared import num_sms as _num_sms
from liger_kernel.ops._nvidia_shared import to_local_if_dtensor as _to_local_if_dtensor
from liger_kernel.ops.utils import ensure_contiguous

_ConstInt = ct.Constant[int]

# Single-tile kernels keep a full row resident in registers/smem. Above this
# width we route to the chunked (multi-pass) kernels, mirroring the Triton
# backend's ``n_cols <= BLOCK_SIZE`` single-block vs multi-block split. The cap
# is a power of two so ``next_pow2(N) <= _MAX_SINGLE_TILE`` is the exact gate.
_MAX_SINGLE_TILE = 8192
# Chunk width used by the multi-pass (chunked) kernels. A power of two so the
# per-chunk ``ct.arange`` tiles align with the gather/scatter bounds.
_CHUNK_SIZE = 4096


def _launch(device: torch.device, grid, kernel, args) -> None:
    """Device-safe cuTile launch tied to the input tensor's CUDA device.

    Dispatch selects this backend from the first CUDA tensor's device, so in a
    multi-GPU process the process-current device can differ. Guarding the
    device (and using that device's stream) keeps the launch on the tensor's
    GPU. Mirrors ``ops/cutile/ops/fused_linear_cross_entropy.py``.
    """
    with torch.cuda.device(device):
        ct.launch(torch.cuda.current_stream(device), grid, kernel, args)


# ===========================================================================
# Forward kernels — single tile per row (``standard`` mode)
# ===========================================================================
@ct.kernel(occupancy=ct.ByTarget(sm_100=4))
def _fwd_standard(
    X,
    Y,
    n_cols: _ConstInt,
    TILE_SIZE: _ConstInt,
):
    """One row per program block. fp32 reduction; cast back on store.

    Loaded via ``ct.gather`` with ``padding_value=-inf`` so the padded lanes
    (when ``n_cols < TILE_SIZE``) contribute ``exp(-inf) == 0`` to the
    denominator and never win the max.
    """
    row_idx = ct.bid(0)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    x = ct.gather(X, (row_idx, offsets), check_bounds=True, padding_value=-math.inf)
    x_f32 = ct.astype(x, ct.float32)

    row_max = ct.max(x_f32, 0, keepdims=True)
    numerator = ct.exp(x_f32 - row_max)
    denominator = ct.sum(numerator, 0, keepdims=True)
    y = numerator / denominator

    ct.scatter(Y, (row_idx, offsets), ct.astype(y, X.dtype), check_bounds=True)


# ===========================================================================
# Forward kernels — static persistent (``static_persistent`` mode)
#
# NUM_SMS*occupancy persistent blocks, each striding over rows by num_blocks.
# Single tile per row. This is the dominant perf variant when M >> NUM_SMS
# (the TileGym reference's default ``_softmax_kernel``).
# ===========================================================================
@ct.kernel(occupancy=ct.ByTarget(sm_100=4))
def _fwd_persistent(
    X,
    Y,
    n_rows: _ConstInt,
    n_cols: _ConstInt,
    TILE_SIZE: _ConstInt,
):
    """Static persistent: each block processes multiple rows."""
    pid = ct.bid(0)
    num_blocks = ct.num_blocks(0)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    row_idx = pid
    while row_idx < n_rows:
        x = ct.gather(X, (row_idx, offsets), check_bounds=True, padding_value=-math.inf)
        x_f32 = ct.astype(x, ct.float32)

        row_max = ct.max(x_f32, 0, keepdims=True)
        numerator = ct.exp(x_f32 - row_max)
        denominator = ct.sum(numerator, 0, keepdims=True)
        y = numerator / denominator

        ct.scatter(Y, (row_idx, offsets), ct.astype(y, X.dtype), check_bounds=True)
        row_idx = row_idx + num_blocks


# ===========================================================================
# Forward kernel — chunked 3-pass (``chunked`` mode) for very wide rows
#
# Pass 1: running max over chunks.
# Pass 2: accumulate denominator = sum_chunks(sum(exp(chunk - row_max))).
# Pass 3: write exp(chunk - row_max) / denominator.
#
# Numerically equivalent to the Triton multi-block kernel (which folds the
# online running (m, d) — here we split into max-then-sum, which yields the
# same final m and the same d up to fp32 rounding).
# ===========================================================================
@ct.kernel(occupancy=ct.ByTarget(sm_100=4))
def _fwd_chunked(
    X,
    Y,
    n_rows: _ConstInt,
    n_cols: _ConstInt,
    CHUNK_SIZE: _ConstInt,
):
    """Static persistent + 3-pass online softmax for wide rows."""
    pid = ct.bid(0)
    num_blocks = ct.num_blocks(0)
    num_chunks = (n_cols + CHUNK_SIZE - 1) // CHUNK_SIZE
    col_base = ct.arange(CHUNK_SIZE, dtype=ct.int32)

    row_idx = pid
    while row_idx < n_rows:
        row_max = ct.full((1,), -math.inf, dtype=ct.float32)
        denominator = ct.full((1,), 0.0, dtype=ct.float32)

        # Pass 1: row max.
        for chunk_idx in range(num_chunks):
            cols = col_base + chunk_idx * CHUNK_SIZE
            chunk = ct.gather(X, (row_idx, cols), check_bounds=True, padding_value=-math.inf)
            chunk = ct.astype(chunk, ct.float32)
            chunk_max = ct.max(chunk, 0, keepdims=True)
            row_max = ct.maximum(row_max, chunk_max)

        # Pass 2: denominator.
        for chunk_idx in range(num_chunks):
            cols = col_base + chunk_idx * CHUNK_SIZE
            chunk = ct.gather(X, (row_idx, cols), check_bounds=True, padding_value=-math.inf)
            chunk = ct.astype(chunk, ct.float32)
            numerator = ct.exp(chunk - row_max)
            denominator = denominator + ct.sum(numerator, 0, keepdims=True)

        # Pass 3: write softmax.
        for chunk_idx in range(num_chunks):
            cols = col_base + chunk_idx * CHUNK_SIZE
            chunk = ct.gather(X, (row_idx, cols), check_bounds=True, padding_value=-math.inf)
            chunk = ct.astype(chunk, ct.float32)
            numerator = ct.exp(chunk - row_max)
            y = numerator / denominator
            ct.scatter(Y, (row_idx, cols), ct.astype(y, X.dtype), check_bounds=True)

        row_idx = row_idx + num_blocks


# ===========================================================================
# Backward kernels — dx = y * (dy - sum(dy * y))
# ===========================================================================
@ct.kernel(occupancy=ct.ByTarget(sm_100=4))
def _bwd_persistent(
    dY,
    Y,
    dX,
    n_rows: _ConstInt,
    n_cols: _ConstInt,
    TILE_SIZE: _ConstInt,
):
    """Static persistent backward, single tile per row.

    The dot-product ``sum(dy * y)`` is reduced in fp32 (matching Triton's
    ``tl.float32`` accumulator). Padded lanes contribute ``0 * 0 == 0`` to the
    dot because both dY and Y gather with ``padding_value=0``.
    """
    pid = ct.bid(0)
    num_blocks = ct.num_blocks(0)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    row_idx = pid
    while row_idx < n_rows:
        dy = ct.gather(dY, (row_idx, offsets), check_bounds=True, padding_value=0.0)
        y = ct.gather(Y, (row_idx, offsets), check_bounds=True, padding_value=0.0)

        dy_f32 = ct.astype(dy, ct.float32)
        y_f32 = ct.astype(y, ct.float32)

        dot = ct.sum(dy_f32 * y_f32, 0, keepdims=True)
        dx = y_f32 * (dy_f32 - dot)

        ct.scatter(dX, (row_idx, offsets), ct.astype(dx, dX.dtype), check_bounds=True)
        row_idx = row_idx + num_blocks


@ct.kernel(occupancy=ct.ByTarget(sm_100=4))
def _bwd_chunked(
    dY,
    Y,
    dX,
    n_rows: _ConstInt,
    n_cols: _ConstInt,
    CHUNK_SIZE: _ConstInt,
):
    """Two-pass backward for wide rows.

    Pass 1: accumulate ``dot = sum(dy * y)`` over chunks (fp32).
    Pass 2: write ``dx = y * (dy - dot)`` chunk by chunk.

    Matches the Triton ``_softmax_multi_block_backward_kernel`` exactly.
    """
    pid = ct.bid(0)
    num_blocks = ct.num_blocks(0)
    num_chunks = (n_cols + CHUNK_SIZE - 1) // CHUNK_SIZE
    col_base = ct.arange(CHUNK_SIZE, dtype=ct.int32)

    row_idx = pid
    while row_idx < n_rows:
        dot = ct.full((1,), 0.0, dtype=ct.float32)

        # Pass 1: accumulate the dot product.
        for chunk_idx in range(num_chunks):
            cols = col_base + chunk_idx * CHUNK_SIZE
            dy = ct.gather(dY, (row_idx, cols), check_bounds=True, padding_value=0.0)
            y = ct.gather(Y, (row_idx, cols), check_bounds=True, padding_value=0.0)
            dot = dot + ct.sum(ct.astype(dy, ct.float32) * ct.astype(y, ct.float32), 0, keepdims=True)

        # Pass 2: write dx.
        for chunk_idx in range(num_chunks):
            cols = col_base + chunk_idx * CHUNK_SIZE
            dy = ct.gather(dY, (row_idx, cols), check_bounds=True, padding_value=0.0)
            y = ct.gather(Y, (row_idx, cols), check_bounds=True, padding_value=0.0)
            dx = ct.astype(y, ct.float32) * (ct.astype(dy, ct.float32) - dot)
            ct.scatter(dX, (row_idx, cols), ct.astype(dx, dX.dtype), check_bounds=True)

        row_idx = row_idx + num_blocks


_VALID_MODES = ("standard", "static_persistent", "chunked")


def _select_mode(mode: Optional[str], n_rows: int, n_cols: int, device: torch.device) -> str:
    """Pick a forward kernel variant when ``mode`` is ``None``.

    Heuristic (mirrors the cuTile rms_norm sibling + TileGym reference):
      - Wide rows (``next_pow2(N) > _MAX_SINGLE_TILE``) must use ``chunked``
        — a single tile would exceed the per-row register/smem budget.
      - Otherwise, when ``M`` is much larger than ``NUM_SMS`` use
        ``static_persistent`` to amortise launch overhead.
      - Otherwise fall back to the simple one-row-per-program ``standard``.

    Validates positive dims so caller shape bugs surface here as a clean
    ValueError rather than a cryptic CUDA grid error.
    """
    if mode is not None:
        # Explicit single-tile modes cannot handle a row wider than the
        # per-row tile budget; reject up-front rather than compiling/launching
        # an oversized tile that silently corrupts results. Only ``chunked``
        # streams arbitrarily wide rows.
        if mode in ("standard", "static_persistent") and _next_pow2(n_cols) > _MAX_SINGLE_TILE:
            raise ValueError(
                f"cuTile softmax mode={mode!r} only supports rows with "
                f"next_pow2(N) <= {_MAX_SINGLE_TILE}; got N={n_cols} "
                f"(next_pow2={_next_pow2(n_cols)}). Use mode='chunked' or auto "
                f"selection (mode=None) for wider rows."
            )
        return mode

    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(f"softmax_cutile: invalid shape ({n_rows}, {n_cols}); both dims must be positive.")

    if _next_pow2(n_cols) > _MAX_SINGLE_TILE:
        return "chunked"
    if n_rows > _num_sms(device) * 2:
        return "static_persistent"
    return "standard"


def _persistent_grid(n_rows: int, device: torch.device) -> tuple:
    """Saturate at most ``NUM_SMS * 4`` blocks but never more than ``n_rows``.

    Matches TileGym's ``num_programs = min(NUM_SM * 4, n_rows)`` occupancy hint.
    """
    return (min(_num_sms(device) * 4, max(n_rows, 1)), 1, 1)


# ---------------------------------------------------------------------------
# Host-side launchers
# ---------------------------------------------------------------------------
def _softmax_forward(x: torch.Tensor, mode: Optional[str]):
    """Launch the forward kernel; return ``(Y, selected_mode)``."""
    *batch, n_cols = x.shape
    x2d = x.contiguous().view(-1, n_cols)
    n_rows = x2d.shape[0]

    selected = _select_mode(mode, n_rows, n_cols, x2d.device)
    if selected not in _VALID_MODES:
        raise ValueError(f"cuTile softmax: unknown mode {selected!r}; expected one of {_VALID_MODES}")

    y2d = torch.empty_like(x2d)
    device = x2d.device

    if selected == "standard":
        tile_size = _next_pow2(n_cols)
        _launch(device, (n_rows, 1, 1), _fwd_standard, (x2d, y2d, n_cols, tile_size))
    elif selected == "static_persistent":
        tile_size = _next_pow2(n_cols)
        _launch(device, _persistent_grid(n_rows, device), _fwd_persistent, (x2d, y2d, n_rows, n_cols, tile_size))
    else:  # chunked
        chunk = min(_CHUNK_SIZE, _next_pow2(n_cols))
        _launch(device, _persistent_grid(n_rows, device), _fwd_chunked, (x2d, y2d, n_rows, n_cols, chunk))

    return y2d.view(*batch, n_cols), selected


def _softmax_backward(dy: torch.Tensor, y: torch.Tensor, selected_mode: str):
    """Launch the backward kernel; return ``dX``."""
    *batch, n_cols = dy.shape
    dy2d = dy.contiguous().view(-1, n_cols)
    y2d = y.contiguous().view(-1, n_cols)
    n_rows = dy2d.shape[0]
    dx2d = torch.empty_like(dy2d)
    device = dy2d.device

    if selected_mode == "chunked":
        chunk = min(_CHUNK_SIZE, _next_pow2(n_cols))
        _launch(device, _persistent_grid(n_rows, device), _bwd_chunked, (dy2d, y2d, dx2d, n_rows, n_cols, chunk))
    else:
        tile_size = _next_pow2(n_cols)
        _launch(device, _persistent_grid(n_rows, device), _bwd_persistent, (dy2d, y2d, dx2d, n_rows, n_cols, tile_size))

    return dx2d.view(*batch, n_cols)


# ---------------------------------------------------------------------------
# autograd.Function — mode is threaded through ctx so backward picks the
# matching kernel layout (single-tile vs chunked).
# ---------------------------------------------------------------------------
class _LigerSoftmaxCuTileFunction(torch.autograd.Function):
    """cuTile softmax. ``mode`` flows in as a positional arg (resolved in the
    registration wrapper) so this class stays ``.apply()``-compatible.
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, input_: torch.Tensor, mode: Optional[str]):
        input_ = _to_local_if_dtensor(input_)
        y, selected_mode = _softmax_forward(input_, mode)
        ctx.save_for_backward(y)
        ctx.selected_mode = selected_mode
        return y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        grad_output = _to_local_if_dtensor(grad_output)
        dx = _softmax_backward(grad_output, y, ctx.selected_mode)
        # forward arity: (input_, mode)
        return dx, None


_CUTILE_SOFTMAX_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


@register_op(
    "softmax",
    impl_name="nvidia-cutile",
    capability=Capability(
        min_cc=(10, 0),
        modules=["cuda.tile", "torch"],
        predicate=_cutile_compiler_available,
    ),
    modes=("standard", "static_persistent", "chunked"),
    default_mode="static_persistent",
    # Keep cuTile as the last fallback. CuTe DSL is faster on measured
    # production shapes; cuTile remains useful for very wide rows where the
    # single-tile DSL paths are range-capped.
    preference_rank=80,
    tolerances=_CUTILE_SOFTMAX_TOLERANCES,
    notes="cuTile static-persistent + chunked softmax for Blackwell (sm_100).",
)
def softmax_cutile(
    x: torch.Tensor,
    *,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """Dispatch entry-point for the cuTile softmax backend.

    Resolves ``mode`` (auto if ``None``), then forwards to
    :class:`_LigerSoftmaxCuTileFunction`. The wrapper exists because the
    dispatcher passes ``mode`` as a kwarg and ``Function.apply`` does not
    accept unknown kwargs.
    """
    if mode is not None and mode not in _VALID_MODES:
        raise ValueError(f"cuTile softmax: unknown mode {mode!r}; valid modes are {_VALID_MODES}.")
    return _LigerSoftmaxCuTileFunction.apply(x, mode)
