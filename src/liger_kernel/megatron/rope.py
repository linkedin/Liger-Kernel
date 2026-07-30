"""Megatron-Core compatible RoPE backed by the Liger Triton kernel.

Megatron applies rotary positional embeddings through a single dispatcher,
``megatron.core.models.common.embeddings.rope_utils.apply_rotary_pos_emb``,
which is called **once per tensor** (first for the query, then for the key).
Liger's Triton RoPE kernel (:class:`liger_kernel.ops.rope.LigerRopeFunction`)
is instead a *fused* q/k kernel that rotates both tensors in one launch.

To reuse the existing, battle-tested Liger kernel without writing a second
Triton kernel, :class:`LigerMegatronRopeFunction` rotates a single tensor by
passing it as the ``q`` argument and a one-head throwaway ``k`` (negligible
extra work — one head out of typically dozens). The public entry point
:func:`liger_apply_rotary_pos_emb_bshd` handles the layout / precision
conversion between Megatron's convention and Liger's:

  * Megatron ``t``     : ``[seq, batch, heads, head_dim]`` (SBHD)
  * Megatron ``freqs`` : ``[seq, 1, 1, rot_dim]`` (already the doubled
    ``cat(freqs, freqs)`` produced by ``RotaryEmbedding``)
  * Liger ``q``        : ``[batch, heads, seq, head_dim]`` (BHSD)
  * Liger ``cos``/``sin`` : ``[1, seq, rot_dim]``

Only the standard non-fused, non-interleaved, non-multi-latent, non-packed path
is handled here. The monkey-patch delegates every other path (fused TE/Apex
RoPE, packed ``thd`` sequences, interleaved rotation, multi-latent attention,
``mscale != 1``, per-batch ``freqs``) back to Megatron's native implementation
so numerics never silently change.
"""

from __future__ import annotations

import torch

from liger_kernel.ops.rope import rope_backward
from liger_kernel.ops.rope import rope_forward


class LigerMegatronRopeFunction(torch.autograd.Function):
    """Single-tensor RoPE autograd op reusing Liger's fused q/k Triton kernel.

    Megatron rotates the query and key in two separate calls, so a fused q/k
    kernel does not map directly. We reuse :func:`liger_kernel.ops.rope.rope_forward`
    by treating the input as ``q`` and slicing a one-head throwaway ``k`` from
    it; the kernel copies its inputs (transpose + ``.contiguous()``) before
    writing, so the throwaway never aliases the real tensor and the input is
    left untouched.

    Args:
        t: ``[batch, heads, seq, head_dim]`` tensor to rotate.
        cos: ``[1, seq, head_dim]`` cosine table (doubled, ``[cos, cos]``).
        sin: ``[1, seq, head_dim]`` sine table (doubled, ``[sin, sin]``).

    Returns:
        The rotated tensor, same shape/dtype as ``t``.
    """

    @staticmethod
    def forward(ctx, t, cos, sin):
        # One-head throwaway "k". rope_forward makes contiguous copies of both
        # tensors before the kernel writes, so slicing from `t` is safe.
        dummy_k = t[:, :1]
        t_out, _dummy_out, cos, sin = rope_forward(t, dummy_k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return t_out

    @staticmethod
    def backward(ctx, dt):
        cos, sin = ctx.saved_tensors
        dummy_dk = dt[:, :1]
        dt_out, _dummy_dk_out = rope_backward(dt, dummy_dk, cos, sin)
        return dt_out, None, None


def liger_apply_rotary_pos_emb_bshd(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Liger drop-in for Megatron's ``_apply_rotary_pos_emb_bshd`` (standard path).

    The ``bshd`` suffix mirrors Megatron's upstream helper name; the tensor
    received here uses Megatron's ``[sequence, batch, heads, head_dim]`` (SBHD)
    layout.

    Equivalent to Megatron-Core's

        t = (t * cos(freqs)) + (rotate_half(t) * sin(freqs))

    for the non-interleaved, non-multi-latent case, but the elementwise rotation
    runs in Liger's Triton kernel. Partial rotary embeddings (``rot_dim < head_dim``)
    are supported: the trailing ``t_pass`` slice is carried through unchanged.

    Args:
        t: ``[seq, batch, heads, head_dim]`` query or key tensor.
        freqs: ``[seq, 1, 1, rot_dim]`` rotary frequencies (Megatron's doubled
            ``cat(freqs, freqs)``).

    Returns:
        Rotated tensor, same shape/dtype as ``t``.
    """
    rot_dim = freqs.shape[-1]

    # Split off the (possibly empty) non-rotary tail so partial-rotary configs work.
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # Megatron computes cos/sin in `t`'s dtype (see _apply_rotary_pos_emb_bshd);
    # match that so the monkey-patch is a bit-faithful drop-in. freqs is
    # [seq, 1, 1, rot_dim]; Liger wants [1, seq, rot_dim].
    cos = torch.cos(freqs).to(t.dtype)[:, 0, 0, :].unsqueeze(0)
    sin = torch.sin(freqs).to(t.dtype)[:, 0, 0, :].unsqueeze(0)

    # SBHD -> BHSD for the Liger kernel, then back.
    t_liger = t_rot.permute(1, 2, 0, 3)
    rotated = LigerMegatronRopeFunction.apply(t_liger, cos, sin)
    rotated = rotated.permute(2, 0, 1, 3)

    # cat also re-materializes a contiguous tensor (Megatron's native path does
    # the same); handles the full-rotary case where t_pass is empty.
    return torch.cat((rotated, t_pass), dim=-1)
