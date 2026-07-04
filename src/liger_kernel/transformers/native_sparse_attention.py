"""
Native Sparse Attention (NSA).

Reference: Yuan et al., "Native Sparse Attention: Hardware-Aligned and Natively
Trainable Sparse Attention", https://arxiv.org/abs/2502.11089

This module provides a pure-PyTorch, fully-differentiable reference implementation
of NSA. It is the exactness oracle that the Triton kernels (added in follow-up
work) are validated against; every branch is exposed as a standalone function so
each can later be swapped for a fused kernel independently.

The structure (compression -> compressed-attention scoring -> top-n block
selection -> selected block-sparse attention -> sliding window -> gated combine)
follows the paper and is informed by the public reference implementations
XunhaoLai/native-sparse-attention-triton (Apache-2.0) and
lucidrains/native-sparse-attention-pytorch (MIT). No code is copied verbatim.

Scope: the trainable, parallel (training / prefill) path over dense
``[batch, heads, seq_len, head_dim]`` tensors. NSA is defined for grouped-query
attention (GQA); all query heads in a KV group share one block selection, which
is what makes the sparse-attention kernel hardware-aligned. The autoregressive
decode path and variable-length (``cu_seqlens``) inputs are out of scope here.
"""

import math

import torch
import torch.nn as nn


def _accum_dtype(dtype: torch.dtype) -> torch.dtype:
    """Score/softmax accumulation dtype: fp32 for low-precision inputs, fp64 kept as-is."""
    return dtype if dtype == torch.float64 else torch.float32


def _expand_kv_heads(x: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    """Repeat each KV head to cover its query-head group (GQA).

    x: ``[batch, num_kv_heads, seq_len, dim]`` -> ``[batch, num_query_heads, seq_len, dim]``.
    Query head ``h`` uses KV head ``h // group`` where ``group = num_query_heads // num_kv_heads``.
    """
    num_kv_heads = x.shape[1]
    if num_kv_heads == num_query_heads:
        return x
    group = num_query_heads // num_kv_heads
    return x.repeat_interleave(group, dim=1)


def _masked_softmax(scores: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """Row-softmax over ``keep_mask==True`` entries; fully-masked rows return all-zero.

    Uses a large finite fill (not ``-inf``) so no row ever produces ``NaN`` in the
    forward or backward pass, then explicitly zeros rows that had no valid key. For
    a row with at least one valid key the result is bit-identical to masking with
    ``-inf`` (the fill underflows to exactly ``0`` after the internal max-subtraction).
    ``scores`` is expected to already be ``float32``.
    """
    keep_mask = keep_mask.expand_as(scores)
    scores = scores.masked_fill(~keep_mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    any_valid = keep_mask.any(dim=-1, keepdim=True)
    return torch.where(any_valid, weights, torch.zeros_like(weights))


def compress_kv(
    x: torch.Tensor,
    intra_block_pe: torch.Tensor,
    mlp: nn.Module,
    block_size: int,
    stride: int,
) -> torch.Tensor:
    """Compress consecutive blocks of keys/values into block-level representations.

    Implements ``phi(k_{id+1:id+l})`` (paper Eq. 7): each length-``block_size`` window
    (stride ``stride``) gets a learnable intra-block positional encoding added, is
    flattened, and is passed through the learnable MLP ``phi`` to a single vector.

    x: ``[batch, heads, seq_len, dim]`` -> ``[batch, heads, num_blocks, dim]`` with
    ``num_blocks = floor((seq_len - block_size) / stride) + 1`` (0 when ``seq_len < block_size``).
    """
    batch, heads, seq_len, dim = x.shape
    if seq_len < block_size:
        return x.new_zeros((batch, heads, 0, dim))
    # windows: [batch, heads, num_blocks, dim, block_size] -> [..., block_size, dim]
    windows = x.unfold(dimension=2, size=block_size, step=stride).transpose(-1, -2)
    windows = windows + intra_block_pe  # broadcast [block_size, dim]
    num_blocks = windows.shape[2]
    flat = windows.reshape(batch, heads, num_blocks, block_size * dim)
    return mlp(flat)


def compressed_scores(
    q: torch.Tensor,
    k_cmp: torch.Tensor,
    scale: float,
    block_size: int,
    stride: int,
) -> torch.Tensor:
    """Softmax scores of full queries over compressed keys (paper Eq. 8).

    Returns ``p_cmp`` ``[batch, num_query_heads, seq_len, num_blocks]`` (all-zero
    rows for queries with no visible block). A compressed block ``j`` spans original
    positions ``[j*stride, j*stride + block_size - 1]`` and is visible to query ``t``
    only when it lies fully in the past (``j*stride + block_size - 1 <= t``). Split
    out from :func:`compressed_attention` so the Triton path can reuse the score
    matrix (needed for selection routing) without recomputing the softmax.
    """
    batch, num_q_heads, seq_len, _ = q.shape
    num_blocks = k_cmp.shape[2]
    if num_blocks == 0:
        return q.new_zeros((batch, num_q_heads, seq_len, 0))

    k_cmp = _expand_kv_heads(k_cmp, num_q_heads)

    # Upcast operands *before* the matmul so a low-precision (e.g. fp16) product
    # cannot saturate to inf prior to the accumulation cast.
    acc = _accum_dtype(q.dtype)
    scores = torch.matmul(q.to(acc), k_cmp.to(acc).transpose(-1, -2)) * scale

    # Causal mask over compressed blocks: block j visible to query t iff its last
    # covered original position is <= t.
    device = q.device
    query_pos = torch.arange(seq_len, device=device).view(seq_len, 1)
    block_last = torch.arange(num_blocks, device=device).view(1, num_blocks) * stride + (block_size - 1)
    visible = (block_last <= query_pos).view(1, 1, seq_len, num_blocks)  # [1, 1, seq_len, num_blocks]

    # Early queries (no compressed block visible yet) get an all-zero score row.
    return _masked_softmax(scores, visible)


def compressed_attention(
    q: torch.Tensor,
    k_cmp: torch.Tensor,
    v_cmp: torch.Tensor,
    scale: float,
    block_size: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention of full queries over compressed KV (paper Eq. 8).

    Returns ``(out_cmp, p_cmp)`` where ``out_cmp`` is the compression-branch output
    ``[batch, num_query_heads, seq_len, dim]`` and ``p_cmp`` is the softmax score
    matrix ``[batch, num_query_heads, seq_len, num_blocks]`` reused to score
    selection blocks.
    """
    batch, num_q_heads, seq_len, _ = q.shape
    num_blocks = k_cmp.shape[2]
    if num_blocks == 0:
        out = q.new_zeros((batch, num_q_heads, seq_len, v_cmp.shape[-1]))
        p_cmp = q.new_zeros((batch, num_q_heads, seq_len, 0))
        return out, p_cmp

    p_cmp = compressed_scores(q, k_cmp, scale, block_size, stride)
    v_cmp = _expand_kv_heads(v_cmp, num_q_heads)
    out = torch.matmul(p_cmp.to(v_cmp.dtype), v_cmp)
    return out, p_cmp


def selection_block_scores(
    p_cmp: torch.Tensor,
    num_kv_heads: int,
    compress_block_size: int,
    compress_stride: int,
    selection_block_size: int,
    num_selection_blocks: int,
) -> torch.Tensor:
    """Derive per-selection-block importance from compression scores.

    Implements paper Eq. 9 (mapping compression scores to selection blocks when
    block sizes differ) followed by Eq. 10 (summing importance across the query
    heads in each GQA group so the whole group selects the same blocks).

    p_cmp: ``[batch, num_query_heads, seq_len, num_cmp_blocks]``
    returns: ``[batch, num_kv_heads, seq_len, num_selection_blocks]``.
    """
    batch, num_q_heads, seq_len, num_cmp = p_cmp.shape
    lp = selection_block_size // compress_stride  # l' / d
    lc = compress_block_size // compress_stride  # l / d

    p_slc = p_cmp.new_zeros((batch, num_q_heads, seq_len, num_selection_blocks))
    if num_cmp > 0:
        base = torch.arange(num_selection_blocks, device=p_cmp.device) * lp  # [num_selection_blocks]
        for m in range(lp):
            for n in range(lc):
                idx = base + m + n
                valid = idx < num_cmp
                gathered = p_cmp[..., idx.clamp(max=num_cmp - 1)]
                p_slc = p_slc + gathered * valid.to(p_slc.dtype)

    # GQA group reduction (Eq. 10): sum over the query heads sharing each KV head.
    group = num_q_heads // num_kv_heads
    p_slc = p_slc.view(batch, num_kv_heads, group, seq_len, num_selection_blocks).sum(dim=2)
    return p_slc


def select_blocks(
    p_slc: torch.Tensor,
    selection_block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Top-n block selection with forced initial + local blocks (paper Eq. 11-12).

    p_slc: ``[batch, num_kv_heads, seq_len, num_selection_blocks]`` importance scores.
    Returns a boolean mask of the same shape marking selected blocks. Future blocks
    (start position > query position) are never selected; ``init_blocks`` leading
    blocks and ``local_blocks`` blocks ending at the query's own block are always
    forced in. Selection is discrete and carries no gradient (hard routing).
    """
    batch, num_kv_heads, seq_len, num_blocks = p_slc.shape
    device = p_slc.device

    block_ids = torch.arange(num_blocks, device=device)
    query_block = torch.arange(seq_len, device=device) // selection_block_size  # [seq_len]

    # Causal: a block is eligible for query t iff it starts at or before t.
    causal = block_ids.view(1, num_blocks) <= query_block.view(seq_len, 1)  # [seq_len, num_blocks]

    forced_init = block_ids.view(1, num_blocks) < init_blocks  # [1, num_blocks]
    local_lo = (query_block - (local_blocks - 1)).clamp(min=0).view(seq_len, 1)
    local = (block_ids.view(1, num_blocks) >= local_lo) & (
        block_ids.view(1, num_blocks) <= query_block.view(seq_len, 1)
    )
    forced = (forced_init | local) & causal  # [seq_len, num_blocks]
    forced = forced.view(1, 1, seq_len, num_blocks).expand(batch, num_kv_heads, seq_len, num_blocks)

    # Rank the remaining eligible blocks by importance, forced blocks always win.
    neg_inf = torch.finfo(p_slc.dtype).min
    scores = p_slc.masked_fill(~causal.view(1, 1, seq_len, num_blocks), neg_inf)
    scores = scores.masked_fill(forced, torch.finfo(p_slc.dtype).max)

    k = min(topk, num_blocks)
    topk_idx = scores.topk(k, dim=-1).indices  # [batch, num_kv_heads, seq_len, k]
    selected = torch.zeros_like(p_slc, dtype=torch.bool)
    selected.scatter_(-1, topk_idx, True)
    # Drop any non-causal blocks that were pulled in by topk padding.
    selected = selected & causal.view(1, 1, seq_len, num_blocks)
    return selected


def selected_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected: torch.Tensor,
    selection_block_size: int,
    scale: float,
) -> torch.Tensor:
    """Exact causal attention restricted to the selected KV blocks.

    q: ``[batch, num_query_heads, seq_len, dim]``; k/v: full-length
    ``[batch, num_kv_heads, seq_len, dim]``; selected: block mask from
    :func:`select_blocks`. Reference builds an explicit key-position mask; the
    Triton kernel gathers only the selected blocks.
    """
    batch, num_q_heads, seq_len, _ = q.shape
    num_kv_heads = k.shape[1]
    device = q.device

    # Map each key position to its selection block, then to the per-query mask.
    key_block = torch.arange(seq_len, device=device) // selection_block_size  # [seq_len]
    # selected: [batch, num_kv_heads, seq_len(query), num_blocks] -> gather per key pos
    key_mask = selected[..., key_block]  # [batch, num_kv_heads, seq_len(query), seq_len(key)]

    causal = torch.arange(seq_len, device=device).view(seq_len, 1) >= torch.arange(seq_len, device=device).view(
        1, seq_len
    )
    key_mask = key_mask & causal.view(1, 1, seq_len, seq_len)
    key_mask = _expand_mask_kv_heads(key_mask, num_q_heads, num_kv_heads)

    return _masked_attention(q, _expand_kv_heads(k, num_q_heads), _expand_kv_heads(v, num_q_heads), key_mask, scale)


def sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    scale: float,
) -> torch.Tensor:
    """Exact causal attention over the last ``window_size`` tokens."""
    batch, num_q_heads, seq_len, _ = q.shape
    device = q.device
    query_pos = torch.arange(seq_len, device=device).view(seq_len, 1)
    key_pos = torch.arange(seq_len, device=device).view(1, seq_len)
    mask = (key_pos <= query_pos) & (key_pos > query_pos - window_size)  # [seq_len, seq_len]
    mask = mask.view(1, 1, seq_len, seq_len)
    return _masked_attention(q, _expand_kv_heads(k, num_q_heads), _expand_kv_heads(v, num_q_heads), mask, scale)


def _expand_mask_kv_heads(mask: torch.Tensor, num_query_heads: int, num_kv_heads: int) -> torch.Tensor:
    if num_kv_heads == num_query_heads:
        return mask
    group = num_query_heads // num_kv_heads
    return mask.repeat_interleave(group, dim=1)


def _masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Softmax attention with a boolean keep-mask; masked-out rows return zero."""
    acc = _accum_dtype(q.dtype)
    scores = torch.matmul(q.to(acc), k.to(acc).transpose(-1, -2)) * scale
    weights = _masked_softmax(scores, mask)
    return torch.matmul(weights.to(v.dtype), v)


def native_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    k_cmp: torch.Tensor,
    v_cmp: torch.Tensor,
    *,
    num_kv_heads: int,
    compress_block_size: int,
    compress_stride: int,
    selection_block_size: int,
    num_selected_blocks: int,
    init_blocks: int,
    local_blocks: int,
    window_size: int,
    scale: float,
) -> torch.Tensor:
    """Combine the three NSA branches with per-head gates (paper Eq. 5).

    q: ``[batch, num_query_heads, seq_len, dim]``; k/v: ``[batch, num_kv_heads, seq_len, dim]``;
    gate: ``[batch, num_query_heads, seq_len, 3]`` in ``[0, 1]`` (compression, selection, sliding);
    k_cmp/v_cmp: precomputed compressed KV from :func:`compress_kv`.
    Returns ``[batch, num_query_heads, seq_len, dim]``.
    """
    seq_len = q.shape[2]

    out_cmp, p_cmp = compressed_attention(q, k_cmp, v_cmp, scale, compress_block_size, compress_stride)

    num_selection_blocks = math.ceil(seq_len / selection_block_size)
    p_slc = selection_block_scores(
        p_cmp,
        num_kv_heads,
        compress_block_size,
        compress_stride,
        selection_block_size,
        num_selection_blocks,
    )
    # Block selection is discrete (hard top-n), exactly as in the NSA paper, so no
    # gradient flows through the choice itself. This is by design, not a limitation:
    # the block scorer (q, k_cmp, and the compression MLP phi) is still trained,
    # because those same tensors feed the differentiable compression branch above.
    selected = select_blocks(
        p_slc.detach(),
        selection_block_size,
        num_selected_blocks,
        init_blocks,
        local_blocks,
    )
    out_slc = selected_attention(q, k, v, selected, selection_block_size, scale)
    out_win = sliding_window_attention(q, k, v, window_size, scale)

    g_cmp, g_slc, g_win = gate[..., 0:1], gate[..., 1:2], gate[..., 2:3]
    return g_cmp * out_cmp + g_slc * out_slc + g_win * out_win


def native_sparse_attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    k_cmp: torch.Tensor,
    v_cmp: torch.Tensor,
    *,
    num_kv_heads: int,
    compress_block_size: int,
    compress_stride: int,
    selection_block_size: int,
    num_selected_blocks: int,
    init_blocks: int,
    local_blocks: int,
    window_size: int,
    scale: float,
) -> torch.Tensor:
    """Triton-kernel-backed NSA combine; numerically matches :func:`native_sparse_attention`.

    The three attention branches run as fused FlashAttention-2 kernels (compressed,
    selected block-sparse, sliding window); compression ``phi``, the selection scoring
    and top-n routing, and the gate blend stay in PyTorch (small, and the routing is
    non-differentiable by design). Same signature and result as the pure-torch combine,
    which is its correctness oracle. Requires CUDA/ROCm/XPU + Triton.
    """
    # Deferred so the pure-torch reference above stays importable without Triton.
    from liger_kernel.ops.nsa_compressed_attention import nsa_compressed_attention
    from liger_kernel.ops.nsa_selected_attention import nsa_selected_attention
    from liger_kernel.ops.nsa_sliding_attention import nsa_sliding_attention

    batch, num_q_heads, seq_len, dim = q.shape

    num_cmp_blocks = k_cmp.shape[2]
    if num_cmp_blocks > 0:
        out_cmp = nsa_compressed_attention(q, k_cmp, v_cmp, compress_block_size, compress_stride, scale)
        p_cmp = compressed_scores(q, k_cmp, scale, compress_block_size, compress_stride)
    else:
        out_cmp = torch.zeros_like(q)
        p_cmp = q.new_zeros((batch, num_q_heads, seq_len, 0))

    num_selection_blocks = math.ceil(seq_len / selection_block_size)
    p_slc = selection_block_scores(
        p_cmp,
        num_kv_heads,
        compress_block_size,
        compress_stride,
        selection_block_size,
        num_selection_blocks,
    )
    selected = select_blocks(
        p_slc.detach(),
        selection_block_size,
        num_selected_blocks,
        init_blocks,
        local_blocks,
    )
    out_slc = nsa_selected_attention(q, k, v, selected, selection_block_size, scale)
    out_win = nsa_sliding_attention(q, k, v, window_size, scale)

    g_cmp, g_slc, g_win = gate[..., 0:1], gate[..., 1:2], gate[..., 2:3]
    return g_cmp * out_cmp + g_slc * out_slc + g_win * out_win


class LigerNativeSparseAttention(nn.Module):
    """Native Sparse Attention (arXiv 2502.11089).

    Applies grouped-query NSA to ``[batch, seq_len, hidden_size]`` inputs: coarse
    compressed attention, fine top-n block selection, and a sliding window,
    blended by learned per-head gates.

    On CUDA/ROCm/XPU the three attention branches run as fused FlashAttention-2
    Triton kernels (:func:`native_sparse_attention_kernel`); on CPU, or when
    ``use_kernel=False``, it falls back to the pure-PyTorch reference
    (:func:`native_sparse_attention`) that also serves as the kernels' correctness
    oracle. Both paths are fully differentiable and numerically equivalent.

    Args:
        hidden_size: model dimension.
        num_heads: number of query heads.
        num_kv_heads: number of key/value heads (GQA); must divide ``num_heads``.
        head_dim: per-head dimension; defaults to ``hidden_size // num_heads``.
        compress_block_size: compression block length ``l`` (paper default 32).
        compress_stride: compression sliding stride ``d`` (paper default 16).
        selection_block_size: selection block size ``l'`` (paper default 64).
        num_selected_blocks: number of selected blocks ``n`` incl. forced (paper default 16).
        init_blocks: leading blocks always selected (paper default 1).
        local_blocks: trailing local blocks always selected (paper default 2).
        sliding_window_size: sliding window ``w`` (paper default 512).
        compress_mlp_hidden_dim: hidden width of the compression MLP ``phi``; defaults
            to ``4 * head_dim``. ``phi`` maps a flattened ``block_size * head_dim`` window
            through one hidden layer (GELU) down to ``head_dim`` (paper Eq. 7).
        bias: whether the q/k/v/o projections use a bias.
        scale: attention scale; defaults to ``1 / sqrt(head_dim)``.
        use_kernel: ``True`` forces the Triton kernels, ``False`` forces the
            pure-torch reference, ``None`` (default) auto-selects — kernels on a
            non-CPU device, torch on CPU.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int = None,
        compress_block_size: int = 32,
        compress_stride: int = 16,
        selection_block_size: int = 64,
        num_selected_blocks: int = 16,
        init_blocks: int = 1,
        local_blocks: int = 2,
        sliding_window_size: int = 512,
        compress_mlp_hidden_dim: int = None,
        bias: bool = False,
        scale: float = None,
        use_kernel: bool = None,
    ):
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads}).")
        if compress_block_size % compress_stride != 0:
            raise ValueError(
                f"compress_block_size ({compress_block_size}) must be divisible by compress_stride ({compress_stride})."
            )
        if selection_block_size % compress_stride != 0:
            raise ValueError(
                f"selection_block_size ({selection_block_size}) must be divisible by "
                f"compress_stride ({compress_stride})."
            )
        if num_selected_blocks < init_blocks + local_blocks:
            raise ValueError(
                f"num_selected_blocks ({num_selected_blocks}) must be >= "
                f"init_blocks + local_blocks ({init_blocks + local_blocks})."
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.compress_block_size = compress_block_size
        self.compress_stride = compress_stride
        self.selection_block_size = selection_block_size
        self.num_selected_blocks = num_selected_blocks
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.sliding_window_size = sliding_window_size
        self.scale = scale if scale is not None else 1.0 / math.sqrt(self.head_dim)
        self.use_kernel = use_kernel

        q_out = num_heads * self.head_dim
        kv_out = num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden_size, q_out, bias=bias)
        self.k_proj = nn.Linear(hidden_size, kv_out, bias=bias)
        self.v_proj = nn.Linear(hidden_size, kv_out, bias=bias)
        self.o_proj = nn.Linear(q_out, hidden_size, bias=bias)

        # Per-head, per-branch gates (compression, selection, sliding).
        self.gate_proj = nn.Linear(hidden_size, num_heads * 3, bias=bias)

        # Learnable KV compression phi (paper Eq. 7): intra-block positional encoding
        # added to each window, then a 2-layer MLP over the flattened block.
        self.compress_mlp_hidden_dim = (
            compress_mlp_hidden_dim if compress_mlp_hidden_dim is not None else 4 * self.head_dim
        )
        self.k_intra_block_pe = nn.Parameter(torch.zeros(compress_block_size, self.head_dim))
        self.v_intra_block_pe = nn.Parameter(torch.zeros(compress_block_size, self.head_dim))
        self.k_compress = self._build_compress_mlp(bias)
        self.v_compress = self._build_compress_mlp(bias)

    def _build_compress_mlp(self, bias: bool) -> nn.Module:
        in_dim = self.compress_block_size * self.head_dim
        return nn.Sequential(
            nn.Linear(in_dim, self.compress_mlp_hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(self.compress_mlp_hidden_dim, self.head_dim, bias=bias),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if attention_mask is not None:
            raise NotImplementedError("LigerNativeSparseAttention does not support attention_mask yet.")

        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(hidden_states)).view(batch, seq_len, self.num_heads, 3).transpose(1, 2)

        k_cmp = compress_kv(k, self.k_intra_block_pe, self.k_compress, self.compress_block_size, self.compress_stride)
        v_cmp = compress_kv(v, self.v_intra_block_pe, self.v_compress, self.compress_block_size, self.compress_stride)

        if self.use_kernel is None:
            # Auto: use the Triton kernels on an accelerator when the config is
            # kernel-compatible (Triton has no fp64; selection blocks tile in 16/32/64).
            use_kernel = (
                hidden_states.device.type != "cpu"
                and hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and self.selection_block_size % 16 == 0
            )
        else:
            use_kernel = self.use_kernel
        combine = native_sparse_attention_kernel if use_kernel else native_sparse_attention
        out = combine(
            q,
            k,
            v,
            gate,
            k_cmp,
            v_cmp,
            num_kv_heads=self.num_kv_heads,
            compress_block_size=self.compress_block_size,
            compress_stride=self.compress_stride,
            selection_block_size=self.selection_block_size,
            num_selected_blocks=self.num_selected_blocks,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            window_size=self.sliding_window_size,
            scale=self.scale,
        )

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, compress_block_size={self.compress_block_size}, "
            f"compress_stride={self.compress_stride}, selection_block_size={self.selection_block_size}, "
            f"num_selected_blocks={self.num_selected_blocks}, init_blocks={self.init_blocks}, "
            f"local_blocks={self.local_blocks}, sliding_window_size={self.sliding_window_size}, "
            f"compress_mlp_hidden_dim={self.compress_mlp_hidden_dim}, scale={self.scale}"
        )
