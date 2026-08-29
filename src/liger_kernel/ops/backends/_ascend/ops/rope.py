"""Ascend RoPE with a layout-aware dispatch.

HuggingFace Qwen/LLaMA build q/k as ``(B, S, H, D).transpose(1, 2)``: logical
BNSD, storage BSHD. Transposing back to BSND is a view (no copy). That path
loads cos/sin once per (batch, s_tile) and reuses them across heads, matching
CANN RotaryPositionEmbeddingGrad.

Already-contiguous BNSD stays on the native-BNSD kernels: forcing a BSND
transpose there copies ~40% of Liger time.

q and k stay sequential on the default stream. Dual-stream overlap fights over
the same 48 AIV cores and corrupts k; a fused kernel with divergent q/k
pointers is predicated into illegal GM access. Short aligned shapes (and the
BSND path) use one two-phase launch (q then k) to hide the second launch.

rotate_half splits each row into (D/2, D/2). Two half-width loads on a
contiguous (S, D) plane are strided (skip the other half every row) and
saturate MTE2. The BNSD fast path loads the full (BLOCK_S, D) row, splits in
UB with extract_slice, and writes one contiguous store.

Four launch kernels cover the old ten via constexpr flags (MASKED, USE_FULLROW,
USE_FLAT). Binary specializations stay separate; Python dispatch never emits
MASKED together with USE_FULLROW/USE_FLAT.
"""

import functools

import torch
import triton
import triton.language as tl

from triton.language.extra.cann.extension import extract_slice
from triton.language.extra.cann.extension import insert_slice

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import get_npu_core_count


@triton.jit
def _rotate_half(left, right, cos_vals, sin_vals, BACKWARD_PASS: tl.constexpr):
    """Shared rotate_half body. Specialized kernels only differ in load/store."""
    if not BACKWARD_PASS:
        new_left = left * cos_vals - right * sin_vals
        new_right = right * cos_vals + left * sin_vals
    else:
        new_left = left * cos_vals + right * sin_vals
        new_right = right * cos_vals - left * sin_vals
    return new_left, new_right


@triton.jit
def _get_work_tile_coord(tile_id, n_s_tiles, n_heads):
    seq_idx = tile_id % n_s_tiles
    tmp = tile_id // n_s_tiles
    head_idx = tmp % n_heads
    batch_idx = tmp // n_heads
    return batch_idx, head_idx, seq_idx


@triton.jit
def _load_cos_sin(
    cos_ptr,
    sin_ptr,
    cos_base,
    sin_base,
    s_off,
    d_off,
    cos_stride_s,
    sin_stride_s,
    s0,
    seq_len,
    half_hd,
    MASKED: tl.constexpr,
):
    if MASKED:
        s_mask = (s0 + s_off) < seq_len
        d_mask = d_off < half_hd
        block_mask = s_mask[:, None] & d_mask[None, :]
        cos_vals = tl.load(
            cos_ptr + cos_base + s_off[:, None] * cos_stride_s + d_off[None, :],
            mask=block_mask,
            other=0,
        )
        sin_vals = tl.load(
            sin_ptr + sin_base + s_off[:, None] * sin_stride_s + d_off[None, :],
            mask=block_mask,
            other=0,
        )
    else:
        cos_vals = tl.load(cos_ptr + cos_base + s_off[:, None] * cos_stride_s + d_off[None, :])
        sin_vals = tl.load(sin_ptr + sin_base + s_off[:, None] * sin_stride_s + d_off[None, :])
    return cos_vals, sin_vals


@triton.jit
def _apply_rope_halves(
    x_ptr,
    x_out_ptr,
    x_base,
    x_stride_s,
    s_off,
    d_off,
    half_hd,
    cos_vals,
    sin_vals,
    s0,
    seq_len,
    BACKWARD_PASS: tl.constexpr,
    MASKED: tl.constexpr,
):
    if MASKED:
        s_mask = (s0 + s_off) < seq_len
        d_mask = d_off < half_hd
        block_mask = s_mask[:, None] & d_mask[None, :]
        x_left = tl.load(
            x_ptr + x_base + s_off[:, None] * x_stride_s + d_off[None, :],
            mask=block_mask,
            other=0,
        )
        x_right = tl.load(
            x_ptr + x_base + s_off[:, None] * x_stride_s + (d_off + half_hd)[None, :],
            mask=block_mask,
            other=0,
        )
        new_left, new_right = _rotate_half(x_left, x_right, cos_vals, sin_vals, BACKWARD_PASS)
        tl.store(x_out_ptr + x_base + s_off[:, None] * x_stride_s + d_off[None, :], new_left, mask=block_mask)
        tl.store(
            x_out_ptr + x_base + s_off[:, None] * x_stride_s + (d_off + half_hd)[None, :], new_right, mask=block_mask
        )
    else:
        x_left = tl.load(x_ptr + x_base + s_off[:, None] * x_stride_s + d_off[None, :])
        x_right = tl.load(x_ptr + x_base + s_off[:, None] * x_stride_s + (d_off + half_hd)[None, :])
        new_left, new_right = _rotate_half(x_left, x_right, cos_vals, sin_vals, BACKWARD_PASS)
        tl.store(x_out_ptr + x_base + s_off[:, None] * x_stride_s + d_off[None, :], new_left)
        tl.store(x_out_ptr + x_base + s_off[:, None] * x_stride_s + (d_off + half_hd)[None, :], new_right)


@triton.jit
def _apply_rope_fullrow(
    x_ptr,
    x_out_ptr,
    x_base,
    x_stride_s,
    s_off,
    cos_vals,
    sin_vals,
    hd: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
):
    # arange(0, hd) lives here so parent kernels never see a non-pow2 hd.
    d_full = tl.arange(0, hd)
    x = tl.load(x_ptr + x_base + s_off[:, None] * x_stride_s + d_full[None, :])
    left = extract_slice(x, (0, 0), (BLOCK_S, BLOCK_D), (1, 1))
    right = extract_slice(x, (0, BLOCK_D), (BLOCK_S, BLOCK_D), (1, 1))
    new_left, new_right = _rotate_half(left, right, cos_vals, sin_vals, BACKWARD_PASS)
    y = insert_slice(x, new_left, (0, 0), (BLOCK_S, BLOCK_D), (1, 1))
    y = insert_slice(y, new_right, (0, BLOCK_D), (BLOCK_S, BLOCK_D), (1, 1))
    tl.store(x_out_ptr + x_base + s_off[:, None] * x_stride_s + d_full[None, :], y)


@triton.jit
def _apply_rope_flat(
    x_ptr,
    x_out_ptr,
    x_base,
    row,
    d_off,
    half_hd,
    hd: tl.constexpr,
    cos_vals,
    sin_vals,
    BACKWARD_PASS: tl.constexpr,
):
    x_left = tl.load(x_ptr + x_base + row[:, None] * hd + d_off[None, :])
    x_right = tl.load(x_ptr + x_base + row[:, None] * hd + (d_off + half_hd)[None, :])
    new_left, new_right = _rotate_half(x_left, x_right, cos_vals, sin_vals, BACKWARD_PASS)
    tl.store(x_out_ptr + x_base + row[:, None] * hd + d_off[None, :], new_left)
    tl.store(x_out_ptr + x_base + row[:, None] * hd + (d_off + half_hd)[None, :], new_right)


@triton.jit
def _rope_bnsd_tile(
    x_ptr,
    x_out_ptr,
    cos_ptr,
    sin_ptr,
    x_base,
    cos_base,
    sin_base,
    x_stride_s,
    cos_stride_s,
    sin_stride_s,
    s_off,
    d_off,
    half_hd,
    seq_len,
    s0,
    hd: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MASKED: tl.constexpr,
    USE_FULLROW: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
):
    cos_vals, sin_vals = _load_cos_sin(
        cos_ptr, sin_ptr, cos_base, sin_base, s_off, d_off, cos_stride_s, sin_stride_s, s0, seq_len, half_hd, MASKED
    )
    if USE_FULLROW:
        _apply_rope_fullrow(
            x_ptr, x_out_ptr, x_base, x_stride_s, s_off, cos_vals, sin_vals, hd, BLOCK_S, BLOCK_D, BACKWARD_PASS
        )
    else:
        _apply_rope_halves(
            x_ptr,
            x_out_ptr,
            x_base,
            x_stride_s,
            s_off,
            d_off,
            half_hd,
            cos_vals,
            sin_vals,
            s0,
            seq_len,
            BACKWARD_PASS,
            MASKED,
        )


@triton.jit
def _triton_rope_bnsd(
    x_ptr,
    x_out_ptr,
    cos_ptr,
    sin_ptr,
    x_stride_b,
    x_stride_h,
    x_stride_s,
    cos_stride_b,
    cos_stride_s,
    sin_stride_b,
    sin_stride_s,
    seq_len,
    n_heads,
    n_s_tiles,
    total_tiles,
    hd: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    MASKED: tl.constexpr,
    USE_FULLROW: tl.constexpr,
):
    """BNSD single-tensor. Replaces aligned / fullrow / masked."""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    s_off = tl.arange(0, BLOCK_S)
    d_off = tl.arange(0, BLOCK_D)
    half_hd = hd // 2
    for tile_id in tl.range(pid, total_tiles, num_progs):
        batch_idx, head_idx, seq_idx = _get_work_tile_coord(tile_id, n_s_tiles, n_heads)
        s0 = seq_idx * BLOCK_S
        cos_base = batch_idx * cos_stride_b + s0 * cos_stride_s
        sin_base = batch_idx * sin_stride_b + s0 * sin_stride_s
        x_base = batch_idx * x_stride_b + head_idx * x_stride_h + s0 * x_stride_s
        _rope_bnsd_tile(
            x_ptr,
            x_out_ptr,
            cos_ptr,
            sin_ptr,
            x_base,
            cos_base,
            sin_base,
            x_stride_s,
            cos_stride_s,
            sin_stride_s,
            s_off,
            d_off,
            half_hd,
            seq_len,
            s0,
            hd,
            BLOCK_S,
            BLOCK_D,
            MASKED,
            USE_FULLROW,
            BACKWARD_PASS,
        )


@triton.jit
def _triton_rope_bnsd_qk(
    q_ptr,
    q_out_ptr,
    k_ptr,
    k_out_ptr,
    cos_ptr,
    sin_ptr,
    q_stride_b,
    q_stride_h,
    q_stride_s,
    k_stride_b,
    k_stride_h,
    k_stride_s,
    cos_stride_b,
    cos_stride_s,
    sin_stride_b,
    sin_stride_s,
    n_q_heads,
    n_k_heads,
    n_s_tiles,
    q_tiles,
    k_tiles,
    hd: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    USE_FULLROW: tl.constexpr,
):
    """Sequential q-then-k phases in one launch. Replaces aligned_qk / fullrow_qk.

    Pointers are never predicated against the other tensor, so addresses stay
    in-bounds.
    """
    pid = tl.program_id(0).to(tl.int64)
    num_progs = tl.num_programs(0).to(tl.int64)
    s_off = tl.arange(0, BLOCK_S)
    d_off = tl.arange(0, BLOCK_D)
    half_hd = hd // 2
    seq_len = 0
    for tile_id in tl.range(pid, q_tiles, num_progs):
        batch_idx, head_idx, seq_idx = _get_work_tile_coord(tile_id, n_s_tiles, n_q_heads)
        s0 = seq_idx * BLOCK_S
        cos_base = batch_idx * cos_stride_b + s0 * cos_stride_s
        sin_base = batch_idx * sin_stride_b + s0 * sin_stride_s
        x_base = batch_idx * q_stride_b + head_idx * q_stride_h + s0 * q_stride_s
        _rope_bnsd_tile(
            q_ptr,
            q_out_ptr,
            cos_ptr,
            sin_ptr,
            x_base,
            cos_base,
            sin_base,
            q_stride_s,
            cos_stride_s,
            sin_stride_s,
            s_off,
            d_off,
            half_hd,
            seq_len,
            s0,
            hd,
            BLOCK_S,
            BLOCK_D,
            False,
            USE_FULLROW,
            BACKWARD_PASS,
        )
    for tile_id in tl.range(pid, k_tiles, num_progs):
        batch_idx, head_idx, seq_idx = _get_work_tile_coord(tile_id, n_s_tiles, n_k_heads)
        s0 = seq_idx * BLOCK_S
        cos_base = batch_idx * cos_stride_b + s0 * cos_stride_s
        sin_base = batch_idx * sin_stride_b + s0 * sin_stride_s
        x_base = batch_idx * k_stride_b + head_idx * k_stride_h + s0 * k_stride_s
        _rope_bnsd_tile(
            k_ptr,
            k_out_ptr,
            cos_ptr,
            sin_ptr,
            x_base,
            cos_base,
            sin_base,
            k_stride_s,
            cos_stride_s,
            sin_stride_s,
            s_off,
            d_off,
            half_hd,
            seq_len,
            s0,
            hd,
            BLOCK_S,
            BLOCK_D,
            False,
            USE_FULLROW,
            BACKWARD_PASS,
        )


@triton.jit
def _triton_rope_bsnd(
    x_ptr,
    x_out_ptr,
    cos_ptr,
    sin_ptr,
    x_stride_b,
    x_stride_s,
    x_stride_h,
    cos_stride_b,
    cos_stride_s,
    sin_stride_b,
    sin_stride_s,
    n_heads,
    seq_len,
    n_s_tiles,
    total_tiles,
    hd: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    MASKED: tl.constexpr,
    USE_FLAT: tl.constexpr,
):
    """Contiguous BSND (B, S, H, D). Replaces bsnd / bsnd_flat / bsnd_masked.

    Three mutually exclusive tile paths (Python sets MASKED and USE_FLAT):

    * **MASKED** (``aligned=False``): ``seq_len % BLOCK_S != 0`` or
      ``BLOCK_D != head_dim // 2``. Tail sequence positions get load/store
      masks; cos/sin are loaded once per (batch, s_tile) and reused in a
      per-head ``tl.range`` loop.

    * **USE_FLAT** (``aligned and _can_flat_bsnd``): ``n_heads`` and
      ``BLOCK_S * n_heads`` are both unpadded powers of two. The
      ``(BLOCK_S, H, D)`` tile is flattened to ``(BLOCK_S * H, D)`` so each
      head row is contiguous in GM — one unmasked vector load/store per tile,
      no head loop, no tail masks. Typical training shapes (e.g. H=32/64).

    * **Head-loop** (``aligned and not USE_FLAT``): full tiles, no tail masks,
      but H is not a power of two (e.g. GQA H=8) so rows cannot be flattened;
      cos/sin load once per (batch, s_tile), then ``tl.range(0, n_heads)``.
    """
    pid = tl.program_id(0).to(tl.int64)
    num_progs = tl.num_programs(0).to(tl.int64)
    d_off = tl.arange(0, BLOCK_D)
    half_hd = hd // 2
    if USE_FLAT:
        BLOCK_ROWS: tl.constexpr = BLOCK_S * BLOCK_H
        row = tl.arange(0, BLOCK_ROWS)
        s_idx = row // BLOCK_H
        for tile_id in tl.range(pid, total_tiles, num_progs):
            s_tile = tile_id % n_s_tiles
            batch_idx = tile_id // n_s_tiles
            s0 = s_tile * BLOCK_S
            x_base = batch_idx * x_stride_b + s0 * x_stride_s
            cos_base = batch_idx * cos_stride_b + s0 * cos_stride_s
            sin_base = batch_idx * sin_stride_b + s0 * sin_stride_s
            cos_vals = tl.load(cos_ptr + cos_base + s_idx[:, None] * cos_stride_s + d_off[None, :])
            sin_vals = tl.load(sin_ptr + sin_base + s_idx[:, None] * sin_stride_s + d_off[None, :])
            _apply_rope_flat(x_ptr, x_out_ptr, x_base, row, d_off, half_hd, hd, cos_vals, sin_vals, BACKWARD_PASS)
    else:
        s_off = tl.arange(0, BLOCK_S)
        for tile_id in tl.range(pid, total_tiles, num_progs):
            s_tile = tile_id % n_s_tiles
            batch_idx = tile_id // n_s_tiles
            s0 = s_tile * BLOCK_S
            cos_base = batch_idx * cos_stride_b + s0 * cos_stride_s
            sin_base = batch_idx * sin_stride_b + s0 * sin_stride_s
            x_base = batch_idx * x_stride_b + s0 * x_stride_s
            cos_vals, sin_vals = _load_cos_sin(
                cos_ptr,
                sin_ptr,
                cos_base,
                sin_base,
                s_off,
                d_off,
                cos_stride_s,
                sin_stride_s,
                s0,
                seq_len,
                half_hd,
                MASKED,
            )
            for h in tl.range(0, n_heads):
                x_row = x_base + h * x_stride_h
                _apply_rope_halves(
                    x_ptr,
                    x_out_ptr,
                    x_row,
                    x_stride_s,
                    s_off,
                    d_off,
                    half_hd,
                    cos_vals,
                    sin_vals,
                    s0,
                    seq_len,
                    BACKWARD_PASS,
                    MASKED,
                )


@triton.jit
def _triton_rope_bsnd_qk(
    q_ptr,
    q_out_ptr,
    k_ptr,
    k_out_ptr,
    cos_ptr,
    sin_ptr,
    q_stride_b,
    q_stride_s,
    q_stride_h,
    k_stride_b,
    k_stride_s,
    k_stride_h,
    cos_stride_b,
    cos_stride_s,
    sin_stride_b,
    sin_stride_s,
    n_q_heads,
    n_k_heads,
    n_s_tiles,
    total_tiles,
    hd: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_HQ: tl.constexpr,
    BLOCK_HK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    USE_FLAT: tl.constexpr,
):
    """Sequential q-then-k with one cos/sin load per (batch, s_tile). Replaces bsnd_qk / bsnd_qk_flat."""
    pid = tl.program_id(0).to(tl.int64)
    num_progs = tl.num_programs(0).to(tl.int64)
    d_off = tl.arange(0, BLOCK_D)
    half_hd = hd // 2
    if USE_FLAT:
        Q_ROWS: tl.constexpr = BLOCK_S * BLOCK_HQ
        K_ROWS: tl.constexpr = BLOCK_S * BLOCK_HK
        q_row = tl.arange(0, Q_ROWS)
        k_row = tl.arange(0, K_ROWS)
        q_s = q_row // BLOCK_HQ
        k_s = k_row // BLOCK_HK
        for tile_id in tl.range(pid, total_tiles, num_progs):
            s_tile = tile_id % n_s_tiles
            batch_idx = tile_id // n_s_tiles
            s0 = s_tile * BLOCK_S
            q_base = batch_idx * q_stride_b + s0 * q_stride_s
            k_base = batch_idx * k_stride_b + s0 * k_stride_s
            cos_base = batch_idx * cos_stride_b + s0 * cos_stride_s
            sin_base = batch_idx * sin_stride_b + s0 * sin_stride_s
            q_cos = tl.load(cos_ptr + cos_base + q_s[:, None] * cos_stride_s + d_off[None, :])
            q_sin = tl.load(sin_ptr + sin_base + q_s[:, None] * sin_stride_s + d_off[None, :])
            _apply_rope_flat(q_ptr, q_out_ptr, q_base, q_row, d_off, half_hd, hd, q_cos, q_sin, BACKWARD_PASS)
            k_cos = tl.load(cos_ptr + cos_base + k_s[:, None] * cos_stride_s + d_off[None, :])
            k_sin = tl.load(sin_ptr + sin_base + k_s[:, None] * sin_stride_s + d_off[None, :])
            _apply_rope_flat(k_ptr, k_out_ptr, k_base, k_row, d_off, half_hd, hd, k_cos, k_sin, BACKWARD_PASS)
    else:
        s_off = tl.arange(0, BLOCK_S)
        seq_len = 0
        for tile_id in tl.range(pid, total_tiles, num_progs):
            s_tile = tile_id % n_s_tiles
            batch_idx = tile_id // n_s_tiles
            s0 = s_tile * BLOCK_S
            cos_base = batch_idx * cos_stride_b + s0 * cos_stride_s
            sin_base = batch_idx * sin_stride_b + s0 * sin_stride_s
            cos_vals, sin_vals = _load_cos_sin(
                cos_ptr,
                sin_ptr,
                cos_base,
                sin_base,
                s_off,
                d_off,
                cos_stride_s,
                sin_stride_s,
                s0,
                seq_len,
                half_hd,
                False,
            )
            q_base = batch_idx * q_stride_b + s0 * q_stride_s
            for h in tl.range(0, n_q_heads):
                x_row = q_base + h * q_stride_h
                _apply_rope_halves(
                    q_ptr,
                    q_out_ptr,
                    x_row,
                    q_stride_s,
                    s_off,
                    d_off,
                    half_hd,
                    cos_vals,
                    sin_vals,
                    s0,
                    seq_len,
                    BACKWARD_PASS,
                    False,
                )
            k_base = batch_idx * k_stride_b + s0 * k_stride_s
            for h in tl.range(0, n_k_heads):
                x_row = k_base + h * k_stride_h
                _apply_rope_halves(
                    k_ptr,
                    k_out_ptr,
                    x_row,
                    k_stride_s,
                    s_off,
                    d_off,
                    half_hd,
                    cos_vals,
                    sin_vals,
                    s0,
                    seq_len,
                    BACKWARD_PASS,
                    False,
                )


@functools.lru_cache(maxsize=64)
def _rope_block_s(seq_len: int, head_dim: int) -> int:
    half = max(1, triton.next_power_of_2(head_dim // 2))
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.70,
        dtype_size=4,
        memory_multiplier=8.0,
        shapes=((seq_len, half),),
        tiling_dims=(0,),
    )
    if tile_shapes:
        block_s = max(8, tile_shapes[0][0])
    else:
        block_s = 32
    return min(block_s, triton.next_power_of_2(seq_len), 64)


def _as_bnsd_cos_sin(cos, sin):
    """Accept (seq, dim) vision tables or (batch, seq, dim) tables."""
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    if cos.dim() != 3:
        raise ValueError(f"Unsupported cos rank {cos.dim()}, expected 2 or 3")
    return cos, sin


def _cos_sin_strides(cos, sin):
    cos_bs = cos.shape[0]
    cos_stride_b = 0 if cos_bs == 1 else cos.stride(0)
    sin_stride_b = 0 if cos_bs == 1 else sin.stride(0)
    return cos_stride_b, cos.stride(1), sin_stride_b, sin.stride(1)


def _ensure_innermost_contiguous(t):
    return t if t.stride(-1) == 1 else t.contiguous()


def _as_contiguous_bnsd(t):
    """Force standard contiguous (B, H, S, D).

    HuggingFace Qwen3 (and most LLaMA-style models) build q/k as
    ``view(B, S, H, D).transpose(1, 2)`` so the logical layout is BNSD but
    the storage is still BSHD (``stride(-1)==1`` yet ``not is_contiguous()``).
    ``empty_like`` on NPU then allocates a *contiguous* buffer while the
    kernel still indexes with the input's strided BSHD layout, so stores
    land in the wrong places and FlashAttention sees garbage (loss ~ln(V)).
    """
    return t if t.is_contiguous() else t.contiguous()


def _is_free_bsnd(t) -> bool:
    """True when ``t.transpose(1, 2)`` is a contiguous BSND view (no copy)."""
    return t is not None and t.dim() == 4 and t.stride(-1) == 1 and t.transpose(1, 2).is_contiguous()


@functools.lru_cache(maxsize=128)
def _bsnd_block_s(seq_len: int, n_heads: int, head_dim: int, dtype_size: int = 2) -> int:
    """S-tile for the BSND kernel.

    Flattened (BLOCK_S * H, D/2) tiles must fit UB; the head-loop fallback
    is smaller, so this heuristic is sized for the flattened path.
    """
    half = max(1, triton.next_power_of_2(head_dim // 2))
    n_heads = max(1, n_heads)
    # 6 live (BLOCK_S, H, D/2) tiles at 4 bytes so fp32 cannot UB-overflow.
    denom = 6 * n_heads * half * 4
    max_s = max(1, (96 * 1024) // denom) if denom else 1
    block_s = 1
    cap = min(32, triton.next_power_of_2(max(1, seq_len)), max_s)
    while block_s * 2 <= cap:
        block_s *= 2
    cores = get_npu_core_count()
    while block_s > 1 and (seq_len // block_s) < cores:
        block_s //= 2
    return max(1, block_s)


def _can_flat_bsnd(n_heads: int, block_s: int) -> bool:
    """True when the BSND tile can be flattened to (BLOCK_S * H, D) rows.

    Requires unpadded power-of-two ``n_heads`` and ``block_s * n_heads`` so
    ``tl.arange(0, BLOCK_S * BLOCK_H)`` is legal and every head row in the
    tile is contiguous in GM. When False but ``seq_len % block_s == 0``, the
    head-loop path still runs without tail masks (see ``MASKED`` in
    ``_triton_rope_bsnd``).
    """
    rows = block_s * n_heads
    return n_heads == triton.next_power_of_2(n_heads) and rows == triton.next_power_of_2(rows)


def _use_fullrow(head_dim: int, block_d: int, seq_len: int, block_s: int) -> bool:
    """Full-row DMA needs a power-of-two hd so tl.arange(0, hd) is legal.

    Tiny BLOCK_S (functional S=2) trips a vector-core exception in
    extract_slice / insert_slice; keep those shapes on the half-load kernel.
    """
    return (
        block_s >= 32
        and head_dim >= 64
        and seq_len % block_s == 0
        and head_dim == block_d * 2
        and head_dim == triton.next_power_of_2(head_dim)
    )


def _launch_one(x, x_out, cos, sin, backward: bool, block_s: int, block_d: int):
    batch_size, n_heads, seq_len, head_dim = x.shape
    n_s_tiles = triton.cdiv(seq_len, block_s)
    total_tiles = batch_size * n_heads * n_s_tiles
    aligned = seq_len % block_s == 0 and block_d == (head_dim // 2)
    use_fullrow = _use_fullrow(head_dim, block_d, seq_len, block_s)
    cos_stride_b, cos_stride_s, sin_stride_b, sin_stride_s = _cos_sin_strides(cos, sin)
    # Oversubscribing (num_cores * 2) lengthens this kernel; 48 persistent
    # programs with a grid-stride loop is the measured peak.
    grid_size = max(1, min(get_npu_core_count(), total_tiles))
    _triton_rope_bnsd[(grid_size,)](
        x,
        x_out,
        cos,
        sin,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        cos_stride_b,
        cos_stride_s,
        sin_stride_b,
        sin_stride_s,
        seq_len,
        n_heads,
        n_s_tiles,
        total_tiles,
        head_dim,
        BLOCK_S=block_s,
        BLOCK_D=block_d,
        BACKWARD_PASS=backward,
        MASKED=not aligned,
        USE_FULLROW=use_fullrow,
    )


def _launch_qk_aligned(q, q_out, k, k_out, cos, sin, backward: bool, block_s: int, block_d: int):
    batch_size, n_q_heads, seq_len, head_dim = q.shape
    n_k_heads = k.shape[1]
    n_s_tiles = triton.cdiv(seq_len, block_s)
    q_tiles = batch_size * n_q_heads * n_s_tiles
    k_tiles = batch_size * n_k_heads * n_s_tiles
    grid_size = max(1, min(get_npu_core_count(), q_tiles + k_tiles))
    cos_stride_b, cos_stride_s, sin_stride_b, sin_stride_s = _cos_sin_strides(cos, sin)
    _triton_rope_bnsd_qk[(grid_size,)](
        q,
        q_out,
        k,
        k_out,
        cos,
        sin,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        cos_stride_b,
        cos_stride_s,
        sin_stride_b,
        sin_stride_s,
        n_q_heads,
        n_k_heads,
        n_s_tiles,
        q_tiles,
        k_tiles,
        head_dim,
        BLOCK_S=block_s,
        BLOCK_D=block_d,
        BACKWARD_PASS=backward,
        USE_FULLROW=_use_fullrow(head_dim, block_d, seq_len, block_s),
    )


def _launch_bsnd_one(x_bsnd, x_out_bsnd, cos, sin, backward: bool, block_s: int, block_d: int):
    batch_size, seq_len, n_heads, head_dim = x_bsnd.shape
    n_s_tiles = triton.cdiv(seq_len, block_s)
    total_tiles = batch_size * n_s_tiles
    aligned = seq_len % block_s == 0 and block_d == (head_dim // 2)
    use_flat = aligned and _can_flat_bsnd(n_heads, block_s)
    grid_size = max(1, min(get_npu_core_count(), total_tiles))
    cos_stride_b, cos_stride_s, sin_stride_b, sin_stride_s = _cos_sin_strides(cos, sin)
    _triton_rope_bsnd[(grid_size,)](
        x_bsnd,
        x_out_bsnd,
        cos,
        sin,
        x_bsnd.stride(0),
        x_bsnd.stride(1),
        x_bsnd.stride(2),
        cos_stride_b,
        cos_stride_s,
        sin_stride_b,
        sin_stride_s,
        n_heads,
        seq_len,
        n_s_tiles,
        total_tiles,
        head_dim,
        BLOCK_S=block_s,
        BLOCK_H=n_heads,
        BLOCK_D=block_d,
        BACKWARD_PASS=backward,
        MASKED=not aligned,
        USE_FLAT=use_flat,
    )


def _launch_bsnd_qk(q_bsnd, q_out, k_bsnd, k_out, cos, sin, backward: bool, block_s: int, block_d: int):
    batch_size, seq_len, n_q_heads, head_dim = q_bsnd.shape
    n_k_heads = k_bsnd.shape[2]
    n_s_tiles = triton.cdiv(seq_len, block_s)
    total_tiles = batch_size * n_s_tiles
    grid_size = max(1, min(get_npu_core_count(), total_tiles))
    use_flat = _can_flat_bsnd(n_q_heads, block_s) and _can_flat_bsnd(n_k_heads, block_s)
    cos_stride_b, cos_stride_s, sin_stride_b, sin_stride_s = _cos_sin_strides(cos, sin)
    _triton_rope_bsnd_qk[(grid_size,)](
        q_bsnd,
        q_out,
        k_bsnd,
        k_out,
        cos,
        sin,
        q_bsnd.stride(0),
        q_bsnd.stride(1),
        q_bsnd.stride(2),
        k_bsnd.stride(0),
        k_bsnd.stride(1),
        k_bsnd.stride(2),
        cos_stride_b,
        cos_stride_s,
        sin_stride_b,
        sin_stride_s,
        n_q_heads,
        n_k_heads,
        n_s_tiles,
        total_tiles,
        head_dim,
        BLOCK_S=block_s,
        BLOCK_HQ=n_q_heads,
        BLOCK_HK=n_k_heads,
        BLOCK_D=block_d,
        BACKWARD_PASS=backward,
        USE_FLAT=use_flat,
    )


def _hf_like(t):
    """Allocate BNSD output with HF BSHD storage (``transpose(1,2)`` is BSND).

    ``torch.empty_like(t)`` would allocate standard contiguous ``(B, H, S, D)``
    storage because ``t`` is already logically BNSD. The kernel indexes with
    ``t``'s strided BSHD layout, so a contiguous ``(B, H, S, D)`` buffer would
    store results at the wrong offsets.
    """
    b, h, s, d = t.shape
    return torch.empty((b, s, h, d), device=t.device, dtype=t.dtype).transpose(1, 2)


def _launch_bsnd(q, k, cos, sin, backward: bool, q_out=None, k_out=None):
    """RoPE on the contiguous BSND view of HF-layout BNSD tensors.

    Accepts optional ``q_out``/``k_out`` in BNSD layout (same as ``_launch_bnsd``).
    Pass the input tensor for in-place store (backward); omit for a fresh buffer.
    """
    q_bsnd = q.transpose(1, 2) if q is not None else None
    k_bsnd = k.transpose(1, 2) if k is not None else None
    if q is not None:
        if q_out is None:
            q_out = _hf_like(q)
        q_out_bsnd = q_out.transpose(1, 2)
    else:
        q_out = None
        q_out_bsnd = None
    if k is not None:
        if k_out is None:
            k_out = _hf_like(k)
        k_out_bsnd = k_out.transpose(1, 2)
    else:
        k_out = None
        k_out_bsnd = None
    ref = q_bsnd if q_bsnd is not None else k_bsnd
    head_dim = ref.shape[-1]
    seq_len = ref.shape[1]
    block_d = triton.next_power_of_2(max(1, head_dim // 2))
    block_s = _bsnd_block_s(
        seq_len,
        max(q.shape[1] if q is not None else 1, k.shape[1] if k is not None else 1),
        head_dim,
        ref.element_size(),
    )
    aligned = seq_len % block_s == 0 and block_d == (head_dim // 2)
    if q_bsnd is not None and k_bsnd is not None and aligned:
        _launch_bsnd_qk(q_bsnd, q_out_bsnd, k_bsnd, k_out_bsnd, cos, sin, backward, block_s, block_d)
    else:
        if q_bsnd is not None:
            _launch_bsnd_one(q_bsnd, q_out_bsnd, cos, sin, backward, block_s, block_d)
        if k_bsnd is not None:
            _launch_bsnd_one(k_bsnd, k_out_bsnd, cos, sin, backward, block_s, block_d)
    if head_dim % 2:
        mid = head_dim - 1
        if q_out is not None:
            q_out[..., mid] = q[..., mid]
        if k_out is not None:
            k_out[..., mid] = k[..., mid]
    return q_out, k_out, cos, sin


def _launch_bnsd(q, k, cos, sin, backward: bool, q_out=None, k_out=None):
    q = _as_contiguous_bnsd(q) if q is not None else None
    k = _as_contiguous_bnsd(k) if k is not None else None
    ref = q if q is not None else k
    head_dim = ref.shape[-1]
    seq_len = ref.shape[2]
    block_d = triton.next_power_of_2(max(1, head_dim // 2))
    block_s = _rope_block_s(seq_len, head_dim)
    aligned = seq_len % block_s == 0 and block_d == (head_dim // 2)
    if q is not None and (q_out is None or not q_out.is_contiguous()):
        q_out = torch.empty_like(q)
    if k is not None and (k_out is None or not k_out.is_contiguous()):
        k_out = torch.empty_like(k)
    if q is not None and k is not None and aligned:
        n_s_tiles = triton.cdiv(seq_len, block_s)
        q_tiles = q.shape[0] * q.shape[1] * n_s_tiles
        k_tiles = k.shape[0] * k.shape[1] * n_s_tiles
        if q_tiles + k_tiles <= get_npu_core_count() * 4:
            _launch_qk_aligned(q, q_out, k, k_out, cos, sin, backward, block_s, block_d)
        else:
            _launch_one(q, q_out, cos, sin, backward, block_s, block_d)
            _launch_one(k, k_out, cos, sin, backward, block_s, block_d)
    else:
        if q is not None:
            _launch_one(q, q_out, cos, sin, backward, block_s, block_d)
        if k is not None:
            _launch_one(k, k_out, cos, sin, backward, block_s, block_d)
    if head_dim % 2:
        mid = head_dim - 1
        if q is not None:
            q_out[..., mid] = q[..., mid]
        if k is not None:
            k_out[..., mid] = k[..., mid]
    return q_out, k_out, cos, sin


def _launch_rope(q, k, cos, sin, backward: bool, q_out=None, k_out=None):
    if q is None and k is None:
        return None, None, cos, sin
    cos, sin = _as_bnsd_cos_sin(cos, sin)
    cos = _ensure_innermost_contiguous(cos)
    sin = _ensure_innermost_contiguous(sin)
    use_bsnd = (q is None or _is_free_bsnd(q)) and (k is None or _is_free_bsnd(k))
    if use_bsnd:
        return _launch_bsnd(q, k, cos, sin, backward, q_out=q_out, k_out=k_out)
    return _launch_bnsd(q, k, cos, sin, backward, q_out=q_out, k_out=k_out)


def rope_forward(q, k, cos, sin):
    return _launch_rope(q, k, cos, sin, backward=False)


def rope_backward(dq, dk, cos, sin):
    dq_out, dk_out, _, _ = _launch_rope(dq, dk, cos, sin, backward=True, q_out=dq, k_out=dk)
    return dq_out, dk_out


class LigerRopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        sin size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        """
        ctx.set_materialize_grads(False)
        q, k, cos, sin = rope_forward(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        """
        dq size: (bsz, n_q_head, seq_len, head_dim)
        dk size: (bsz, n_kv_head, seq_len, head_dim)
        """
        # set_materialize_grads(False) passes None for unused outputs. Both-None
        # is the unused-outputs guard; a single None is handled by _launch_rope.
        if dq is None and dk is None:
            return None, None, None, None, None, None
        cos, sin = ctx.saved_tensors
        dq, dk = rope_backward(dq, dk, cos, sin)
        return dq, dk, None, None, None, None
