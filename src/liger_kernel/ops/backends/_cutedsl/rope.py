"""CuTe DSL (CUTLASS python) backend for ``rope``.

Strategy
--------
The dispatcher registration delegates directly to the fused ops-layer kernels
(:func:`liger_kernel.ops.cutedsl.ops.rope.rope_forward` and
:func:`rope_backward`) — the same token-per-CTA / TMA path that powers the
wholesale ``LIGER_KERNEL_IMPL=cutedsl`` backend. Those kernels fuse the whole
rotation (q + k in one launch, cos/sin read per token, no host-side
split/expand/concatenate), so the dispatcher inherits the wholesale layer's
performance profile instead of paying for host-side tensor transforms.

This replaces the original dispatcher-local implementation, which transposed
and flattened q/k, split head halves with four ``.contiguous()`` copies,
expanded ``cos``/``sin`` to a per-head ``(M, hd//2)`` layout (hundreds of MB
for Llama-sized batches), ran two launches, and concatenated the halves back
— ~12 device copies per call and ~3.8x slower than the Triton reference on
H200. The ops-layer path measures within noise of Triton on the same shapes.

Routing
-------
Promotion to this backend is shape- and regime-gated (preference rank 40,
ahead of Triton's 50); everything outside the gate falls back to the Triton
kernel via the in-wrapper fallback below (same pattern as layer_norm).

Gate = 2-byte dtype AND ``q.requires_grad`` AND ``E >= 131072`` AND
``head_dim >= 128``, where ``E = bsz * n_heads_q * seq_len`` is the total q
head-row count (the routing key). Only the grad-carrying full fwd+bwd pair in
this large-shape zone is routed here; the no-grad fwd-only path and smaller
shapes keep Triton.

Numerics: the ops-layer kernels compute the rotation in fp32 internally and
store back at the input dtype, matching the Triton reference; parity is pinned
by ``test/transformers/test_cutedsl_rope.py``.

In-place convention: identical to the Triton ``LigerRopeFunction`` — for the
standard projection-transpose layout the forward views native storage, so the
caller's q/k storage may be rotated in place.

Capability
----------
- Compute capability >= sm_90 (Hopper or newer), matching the rms_norm / softmax
  CuTe DSL siblings; the TMA tile path activates on sm_100 (Blackwell) and falls
  back to the token kernel otherwise.
- Requires only the ``cutlass`` Python package.

Shapes and limits
-----------------
- ``q`` shape ``(bsz, n_q_head, seq_len, head_dim)``; ``k`` shape
  ``(bsz, n_kv_head, seq_len, head_dim)``; ``head_dim`` must be even.
- ``cos`` / ``sin`` shape ``(1, seq_len, head_dim)`` or ``(bsz, seq_len, head_dim)``
  (2-D vision tables are handled by the ops layer).
- dtypes: fp16, bf16, fp32.

Notes
-----
We deliberately do **not** put ``from __future__ import annotations`` here
(same reasoning as the RMSNorm / softmax siblings).
"""

from typing import Optional

import torch

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.ops._nvidia_shared import to_local_if_dtensor as _to_local_if_dtensor
from liger_kernel.ops.cutedsl.ops.rope import rope_backward as _ops_rope_backward
from liger_kernel.ops.cutedsl.ops.rope import rope_forward as _ops_rope_forward


# ===========================================================================
# Autograd Function
# ===========================================================================
class _LigerRopeCuTeDSLFunction(torch.autograd.Function):
    """Autograd wrapper delegating to the fused ops-layer kernels.

    Mirrors the Triton ``LigerRopeFunction`` contract exactly: forward applies
    ``rope_forward`` and saves the returned ``cos``/``sin``; backward applies
    ``rope_backward`` (handles both contiguous fast-path and SDPA-style
    transpose-view grads).
    """

    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        del position_ids, unsqueeze_dim  # accepted for API parity with Triton
        q = _to_local_if_dtensor(q)
        k = _to_local_if_dtensor(k)
        cos = _to_local_if_dtensor(cos)
        sin = _to_local_if_dtensor(sin)
        q, k, cos, sin = _ops_rope_forward(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        dq = _to_local_if_dtensor(dq)
        dk = _to_local_if_dtensor(dk)
        cos, sin = ctx.saved_tensors
        dq_out, dk_out = _ops_rope_backward(dq, dk, cos, sin)
        return dq_out, dk_out, None, None, None, None


# ===========================================================================
# Public registration
# ===========================================================================
_CUTEDSL_ROPE_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}

# Auto-select gate (B200 map in module docstring).  "_Q_ROWS" is the total
# q head-row count E = bsz*n_heads_q*seq_len (== q.numel()/head_dim), NOT the
# plain token count: E-keying is measured ((1,8192)~(4,2048)~(2,4096) win;
# (1,2048) at E=65536 loses 0.88; wash rung E=98304 at 0.968; threshold cell
# E=131072 wins 1.054-1.115 in three probes).  Strict head_dim >= 128 (the
# hd=64 arm has a losing rung adjacent to a single win -- fragile).
_ROPE_AUTO_MIN_Q_ROWS = 131072
_ROPE_AUTO_MIN_HEAD_DIM = 128


def _rope_auto_select_ok(q: torch.Tensor) -> bool:
    if q.element_size() != 2 or q.shape[-1] < _ROPE_AUTO_MIN_HEAD_DIM or not q.requires_grad:
        return False
    q_rows = 1
    for d in q.shape[:-1]:
        q_rows *= d
    return q_rows >= _ROPE_AUTO_MIN_Q_ROWS


@register_op(
    "rope",
    impl_name="nvidia-cutedsl",
    capability=Capability(
        min_cc=(9, 0),
        modules=["cutlass.cute", "torch"],
    ),
    modes=("default",),
    default_mode="default",
    # Delegates to the fused ops-layer token/TMA kernels — the earlier
    # dispatcher-local expand/concatenate path was ~3.8x slower than Triton on
    # H200; the fused path measures at parity, and is 2.2-6.2x faster than the
    # deleted inline expand-cos/sin multi-launch path on B200 in the measured
    # win zone.  Rank 40 (auto-select ahead of Triton at rank 50) but only for
    # that zone: grad-carrying full fwd+bwd, 2-byte dtypes, q head-rows
    # >= 131072, head_dim >= 128 (B200 2026-08-09 map in the module docstring);
    # the wrapper silently Triton-falls-back outside it.
    preference_rank=40,
    tolerances=_CUTEDSL_ROPE_TOLERANCES,
    notes=(
        "CuTe DSL RoPE for Hopper+ (sm_90+) via the fused ops-layer token/TMA "
        "kernels.  Rank 40 ahead of Triton in the measured zone (2-byte, grad "
        "on, q head-rows>=131072, head_dim>=128); wrapper Triton-falls-back "
        "outside it.  Even head_dim required; odd dims raise ValueError."
    ),
)
def rope_cutedsl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
    *,
    mode: Optional[str] = None,
):
    """CuTe DSL RoPE dispatch entry point.

    ``mode`` is accepted only for API parity with the other backends; the only
    valid value is ``"default"`` (or ``None``).  ``position_ids`` /
    ``unsqueeze_dim`` are accepted for signature parity with
    ``LigerRopeFunction`` and are unused (the cos/sin broadcast is implicit —
    identical to the Triton kernel).
    """
    if mode not in (None, "default"):
        raise ValueError(f"CuTe DSL rope has only mode='default'; got mode={mode!r}.")
    if q.shape[-1] % 2 != 0 or k.shape[-1] % 2 != 0:
        raise ValueError(f"CuTe DSL RoPE requires even q/k head dimensions; got q={q.shape[-1]}, k={k.shape[-1]}")
    if not _rope_auto_select_ok(q):
        # Outside the measured win zone (small shapes / no-grad fwd-only /
        # fp32): the Triton kernel is faster on B200 -- keep it, silently
        # (neither backend was ever selected here, so this is plain routing).
        from liger_kernel.ops.backends._triton.rope import rope_triton

        return rope_triton(q, k, cos, sin, position_ids, unsqueeze_dim, mode=mode)
    return _LigerRopeCuTeDSLFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)
