"""CuTe DSL (CUTLASS python) backend for ``fused_add_rms_norm``.

Strategy
--------
``fused_add_rms_norm`` implements the residual-add + RMSNorm pattern used in
every transformer decoder layer::

    S = X + R            (residual add)
    Y = rms_norm(S, W)   (normalisation)

Fusing the add into the norm kernel saves a separate kernel launch and a full
memory round-trip for ``S`` — critical for training throughput.

The CuTe DSL implementation reuses the **same** forward and backward kernels
as the standalone ``rms_norm`` CuTeDSL backend:

- **Forward** delegates to :func:`_cute_lib.rmsnorm_fwd.rmsnorm_fwd` with
  ``residual=R``.  The Quack-derived kernel already fuses the residual add
  in-register (``x += residual``) before the sum-of-squares reduction, and
  writes the summed ``S`` to ``residual_out``.  This is exactly the
  ``fused_add_rms_norm`` forward — no new kernel is needed.

- **Backward** delegates to the inline ``_LigerRMSNormCuTeDSLBackward`` kernel
  from :mod:`._cutedsl.rms_norm`.  The RMSNorm backward formula is identical
  whether or not the input was the result of a residual add — the only
  difference is that ``dX`` and ``dR`` are **both** equal to the RMSNorm input
  gradient (because ``S = X + R`` ⇒ ``dS/dX = dS/dR = 1``).  An optional
  ``dS_out`` (gradient flowing to the residual output ``S``) is added
  host-side after the kernel.

Math (matches the Triton kernel in :mod:`liger_kernel.ops.fused_add_rms_norm`)::

    S     = X + R
    rstd  = 1 / sqrt(mean(S^2) + eps)
    Y     = (S * rstd) * (offset + W)

    # backward (dY flows through rms_norm; dS_out flows through the residual)
    wdy   = (offset + W) * dY             (fp32)
    c     = mean(wdy * (S * rstd))
    dx_rms= rstd * (wdy - (S * rstd) * c)
    dX    = dx_rms + dS_out
    dR    = dX                            (residual gradient equals dX)
    dW   += dY * (S * rstd)               (fp32, per-row)

Casting modes (llama / gemma / none) and ``offset`` are handled identically to
the ``rms_norm`` CuTeDSL sibling — see that file's docstring for details.

Capability
----------
- Compute capability >= sm_90 (Hopper or newer), matching the rms_norm sibling.
- Backward kernel is capped at ``N <= 32768`` (same cluster-reduce limitation).

References
----------
- Triton reference: :mod:`liger_kernel.ops.fused_add_rms_norm`
- CuTe DSL forward: :mod:`._cute_lib.rmsnorm_fwd` (residual parameter)
- CuTe DSL backward: :mod:`._cutedsl.rms_norm._LigerRMSNormCuTeDSLBackward`
"""

from typing import Optional
from typing import Tuple

import torch

from torch import Tensor

from liger_kernel.backends import Capability
from liger_kernel.backends import register_op
from liger_kernel.backends.dispatch import emit_fallback_warning
from liger_kernel.ops._nvidia_shared import CASTING_MODE_GEMMA as _CASTING_MODE_GEMMA
from liger_kernel.ops._nvidia_shared import STR_TO_CASTING_MODE as _str_to_casting_mode
from liger_kernel.ops._nvidia_shared import to_local_if_dtensor as _to_local_if_dtensor
from liger_kernel.ops.backends._cutedsl._cute_lib.rmsnorm_fwd import rmsnorm_fwd as _cutedsl_rmsnorm_fwd
from liger_kernel.ops.backends._cutedsl.rms_norm import _BWD_MAX_TILE_CUTEDSL
from liger_kernel.ops.backends._cutedsl.rms_norm import _rms_norm_cutedsl_backward

# The FORWARD stages a full row of X AND of the residual R in dynamic shared
# memory (``sX``/``sRes`` in :mod:`._cute_lib.rmsnorm_fwd`, reload_from="smem"
# for N > 8192), i.e. ~ (es_x + es_r) * N bytes plus the reduction buffer. That
# fits the 232448-byte sm_100 per-CTA cap for 2-byte pairs (128KB @32768) and
# mixed 2-byte/fp32 pairs (192KB @32768), but overflows when the STAGED pair is
# all-4-byte: fp32 fwd kernels at N=32768 COMPILE then fault at launch with
# cudaErrorInvalidValue ("Allocated: 262176 > 232448" — measured on B200 by the
# post-dS-fold crossover map's last cell; same class as kl_div 2a73598). X is
# staged fp32 when it IS fp32 or when the gemma mixed-dtype host promotion
# widens it; R is staged as-passed. Guard on the measured power-of-two ceiling
# (2*4*28672 = 229376 <= cap): an all-4-byte staged pair above it takes the
# Triton fallback in the registered wrapper instead of faulting; every other
# dtype mix keeps the CuTe DSL kernel byte-unchanged. Also closes a live trap
# in the inference-only 2-byte band: a no-grad gemma call with an fp32 residual
# staged (4,4)*N bytes and routed there over the full (8192, 32768] range.
_FWD_SMEM_CAP_TILE_FP32_PAIR = 28672


def _farn_fwd_stages_fp32_pair(X: Tensor, R: Tensor, W: Tensor, casting_mode) -> bool:
    """True when the fwd kernel would stage an all-4-byte (X, R) pair in smem.

    Mirrors the launcher's ``_promote_gemma_fp32`` decision: X is staged fp32
    when it already is fp32 or when the gemma mixed-dtype host promotion
    widens it (X 2-byte and W/R not both of X dtype); R is staged with its
    passed dtype (the helper never host-casts it).
    """
    cm = _str_to_casting_mode.get(casting_mode, casting_mode) if isinstance(casting_mode, str) else casting_mode
    stages_fp32_x = X.element_size() == 4 or (
        cm == _CASTING_MODE_GEMMA
        and X.dtype != torch.float32
        and not (X.dtype in (torch.float16, torch.bfloat16) and W.dtype == X.dtype and R.dtype == X.dtype)
    )
    return stages_fp32_x and R.element_size() == 4


# ===========================================================================
# Host-side launchers and autograd Function
# ===========================================================================
def _fused_add_rms_norm_cutedsl_forward(
    X: Tensor,
    R: Tensor,
    W: Tensor,
    eps: float,
    offset: float,
    casting_mode: int,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor]:
    """Forward via the inlined ``_cute_lib.rmsnorm_fwd`` with ``residual=R``.

    Returns ``(Y, S, W_eff, RSTD)`` where ``S = X + R`` is the summed residual
    (saved for the backward), ``W_eff`` is ``weight + offset``, and ``RSTD``
    is fp32.
    """
    shape = X.shape
    N = shape[-1]

    # Same redundant-promotion removal as the plain rms_norm sibling: the
    # kernel up-casts X/R/W to fp32 in-register with an EXACT bf16/fp16->fp32
    # map, so for uniform-dtype gemma calls the host round trip only burns
    # bandwidth (a (M,N) fp32 copy of X in, a 2x wider X kernel read, an (M,N)
    # fp32 round trip for out — the backward likewise bridges dy/dx through
    # fp32). Keep the promotion only for mixed-dtype gemma calls, matching
    # Triton's forced-fp32 gemma math.
    _promote_gemma_fp32 = (
        casting_mode == _CASTING_MODE_GEMMA
        and X.dtype != torch.float32
        and not (X.dtype in (torch.float16, torch.bfloat16) and W.dtype == X.dtype and R.dtype == X.dtype)
    )

    # Guard the sm_100 smem cap (constant block comment above): an all-4-byte
    # staged (X, R) pair above the ceiling compiles then faults at launch
    # (cudaErrorInvalidValue — recoverable, same class as kl_div 2a73598).
    # Registered-wrapper calls in this band are Triton-routed upstream; direct
    # Function/helper callers get a clean pre-launch raise instead.
    es_x_staged = 4 if (_promote_gemma_fp32 or X.element_size() == 4) else X.element_size()
    if N > _FWD_SMEM_CAP_TILE_FP32_PAIR and es_x_staged == 4 and R.element_size() == 4:
        raise RuntimeError(
            f"CuTe DSL fused_add_rms_norm fwd stages X+R in dynamic smem "
            f"({(es_x_staged + R.element_size()) * N} bytes > 232448 sm_100 cap) "
            f"for an all-fp32 staged pair at hidden {N}; use hidden <= "
            f"{_FWD_SMEM_CAP_TILE_FP32_PAIR} or the Triton path."
        )

    if _promote_gemma_fp32:
        X_in = X.to(torch.float32)
    else:
        X_in = X
    X_flat = X_in.view(-1, N).contiguous()
    R_flat = R.view(-1, N).contiguous()

    # Fold offset into the weight host-side.
    if offset != 0.0:
        w_eff = W + offset
    else:
        w_eff = W
    # Only the mixed-dtype gemma route promotes w_eff host-side; the
    # kernel up-casts in-register otherwise (bit-exact).
    if _promote_gemma_fp32 and w_eff.dtype != torch.float32:
        w_eff = w_eff.to(torch.float32)
    w_eff = w_eff.contiguous()

    # rmsnorm_fwd with residual=R fuses the add into the kernel.
    # Returns (out, residual_out=S, rstd).
    out, S, rstd = _cutedsl_rmsnorm_fwd(
        X_flat,
        weight=w_eff,
        residual=R_flat,
        eps=eps,
        store_rstd=True,
    )

    # Back-cast if the promote route above widened the output (it is the
    # only path that can produce an out dtype != X.dtype).
    if out.dtype != X.dtype:
        out = out.to(X.dtype)

    # S should match the original input dtype (the residual output is written
    # at the kernel's input dtype, which for gemma is fp32). Cast back so the
    # backward signature is consistent.
    if S.dtype != X.dtype:
        S = S.to(X.dtype)

    return out.view(shape), S.view(shape), w_eff, rstd


def _fused_add_rms_norm_cutedsl_backward(
    dY: Tensor,
    dS_out: Optional[Tensor],
    S: Tensor,
    W_eff: Tensor,
    RSTD: Tensor,
    in_place: bool,
    casting_mode: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Backward via the inline CuTe DSL rms_norm backward kernel.

    The RMSNorm backward is computed on ``S`` (the summed residual).  The
    residual-add makes ``dX = dR``; an optional ``dS_out`` (gradient flowing
    to the residual output) is added host-side after the kernel.
    """
    # Reuse the rms_norm backward: it computes dX_rms from (dY, S, W_eff, RSTD).
    # ``dS_out`` is folded into the kernel epilogue (dx += dS in fp32 before the
    # single store round — strictly closer to exact than a post-hoc two-tensor
    # bf16 add, and it removes the host pass's full (M,N) read+write trip: the
    # per-kernel attribution probe measured that add pass at 0.032-0.061ms for
    # hd 4096..8192 @M=8192 bf16, ~15% of the fused profile at 8192). This
    # accounts for the residual path: S = X + R, so dX += dS_out and dR = dX.
    #
    # The saved ``S`` keeps the *original* input shape (e.g. 3-D (B, T, N) from
    # decoder-layer usage) while the compiled backward kernel is 2-D only and is
    # launched with ``S`` handed straight through as the ``x_flat`` operand, so a
    # 3-D ``S`` raises "ValueError: expected ndim=2". Flatten every (M, N)
    # operand to 2-D rows here (the ``ds=`` epilogue fold path) and restore the
    # original shape on the returned gradient afterwards. (``dY`` and ``ds`` are
    # additionally re-flattened inside ``_rms_norm_cutedsl_backward``; flattening
    # them here too keeps all operands consistent.)
    shape = dY.shape
    N = shape[-1]
    dX, dW = _rms_norm_cutedsl_backward(
        dY.view(-1, N).contiguous() if dY.dim() > 2 else dY,
        S.view(-1, N) if S.dim() > 2 else S,
        W_eff,
        RSTD,
        in_place,
        casting_mode,
        ds=(dS_out.view(-1, N) if (dS_out is not None and dS_out.dim() > 2) else dS_out),
    )
    if dY.dim() > 2:
        dX = dX.view(shape)

    # dR = dX (the residual add makes the gradient flow equally to both).
    dR = dX
    return dX, dR, dW


class _LigerFusedAddRMSNormCuTeDSLFunction(torch.autograd.Function):
    """Autograd wrapper for fused_add_rms_norm via CuTe DSL.

    Saves ``S`` (the summed residual), ``W_eff``, and ``RSTD`` for the backward.
    """

    @staticmethod
    def forward(ctx, X, R, W, eps, offset=0.0, casting_mode="llama", in_place=False):
        X = _to_local_if_dtensor(X)
        R = _to_local_if_dtensor(R)
        W = _to_local_if_dtensor(W)

        X = X.contiguous()
        R = R.contiguous()
        W = W.contiguous()

        # Normalize casting_mode to an int once.
        if not isinstance(casting_mode, int):
            if casting_mode not in _str_to_casting_mode:
                raise ValueError(f"Invalid casting mode: {casting_mode}")
            casting_mode_int = _str_to_casting_mode[casting_mode]
        else:
            casting_mode_int = casting_mode

        Y, S, W_eff, RSTD = _fused_add_rms_norm_cutedsl_forward(X, R, W, eps, offset, casting_mode_int)

        ctx.in_place = in_place
        ctx.weight_dtype = W.dtype
        ctx.casting_mode = casting_mode_int
        ctx.save_for_backward(S, W_eff, RSTD)
        return Y, S

    @staticmethod
    def backward(ctx, dY, dS_out):
        dY = _to_local_if_dtensor(dY).contiguous()
        if dS_out is not None:
            dS_out = _to_local_if_dtensor(dS_out).contiguous()

        S, W_eff, RSTD = ctx.saved_tensors
        dX, dR, dW = _fused_add_rms_norm_cutedsl_backward(
            dY,
            dS_out,
            S,
            W_eff,
            RSTD,
            ctx.in_place,
            ctx.casting_mode,
        )
        dW = dW.to(ctx.weight_dtype)

        # forward arity: (X, R, W, eps, offset, casting_mode, in_place)
        return dX, dR, dW, None, None, None, None


# ===========================================================================
# Public registration
# ===========================================================================
# Auto-select routing (B200 sm_100 plane). Plain (in_place=False) grad-carrying
# calls route to CuTe DSL in two hidden-size bands; everything else keeps Triton:
#   - Lower band: hidden <= _AUTORANK_MAX_HIDDEN(_2B) -- CuTe DSL wins for all
#     dtypes; it loses just above, so the cap sits at the crossover.
#   - Upper band: hidden in [_AUTORANK_UPPER_MIN_HIDDEN, _BWD_MAX_TILE_CUTEDSL]
#     -- CuTe DSL wins again at wide hidden; the intervening rungs stay Triton.
# The only faulting cell (all-fp32 staged X+R pairs above the sm_100 fwd smem
# cap) is caught first by the _FWD_SMEM_CAP_TILE_FP32_PAIR guard. in_place=True
# keeps Triton (its dX=X alias erases the win). The dtype gate is retained so
# the 2-byte and fp32 caps can diverge on a future re-measurement.
_AUTORANK_MAX_HIDDEN = 8192
_AUTORANK_MAX_HIDDEN_2B = 8192
# Lower edge of the re-routed wide-hidden window (see the block above): hidden
# in (crossover guard, this) stays Triton; [this, 32768] plain calls go CuTe DSL.
_AUTORANK_UPPER_MIN_HIDDEN = 24576


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


# B300 sm_103 plane constants. The wrapper selects this set at call time via
# _max_capability() > 100; the DSL compile keys kernel tiles by capability
# regardless. Same two-band routing as B200, with the upper band opening one
# rung earlier (20480 vs 24576).
_B300_AUTORANK_MAX_HIDDEN = 8192
_B300_AUTORANK_MAX_HIDDEN_2B = 8192
_B300_AUTORANK_UPPER_MIN_HIDDEN = 20480


def _farn_fwd_no_grad_ok(X: torch.Tensor, R: torch.Tensor, W: torch.Tensor, in_place: bool) -> bool:
    """True when an inference-only 2-byte call sits in the wide-hidden band.

    The ``_AUTORANK_MAX_HIDDEN*`` crossover constants were tuned on the fused
    fwd+bwd pair (the Triton bwd wins past the crossover see b0c33e1), so a
    call that can never backprop should not be routed by them. For 2-byte
    dtypes in (``_AUTORANK_MAX_HIDDEN_2B``, ``_BWD_MAX_TILE_CUTEDSL``] the raw
    Quack fused forward still beats Triton forward-only (measured on B200:
    fwd host/wall 1.42-1.72x at M<=512, CUDA-event ~1.07x at 2048x32768,
    tie elsewhere, and strictly more accurate — err-vs-fp64 y 1.65e-3 vs
    Triton's 2.9e-3); fp32 is kept out (raw fwd loses 0.82x at 4096x16384,
    ties at 2048x24576 — fragile 2-cell crossover, same call as the
    layer_norm a0bb846 no-grad band).

    Safety gates: 2-byte only; hidden in the band; ``in_place=False`` (the
    raw path serves new tensors, while Triton's in_place mutates ``X`` — keep
    that observable contract on the fallback); and no autograd graph may be
    demanded — either grad mode is off or NONE of X/R/W requires grad. A
    frozen-X call whose weight still needs grad is REFUSED (falls to the
    crossover guard -> Triton) so a gradient is never silently dropped.
    """
    if in_place or X.element_size() != 2:
        return False
    n = X.shape[-1]
    if n <= _AUTORANK_MAX_HIDDEN_2B or n > _BWD_MAX_TILE_CUTEDSL:
        return False
    return not torch.is_grad_enabled() or not (X.requires_grad or R.requires_grad or W.requires_grad)


_CUTEDSL_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


@register_op(
    "fused_add_rms_norm",
    impl_name="nvidia-cutedsl",
    capability=Capability(
        min_cc=(9, 0),
        modules=["cutlass.cute", "torch"],
    ),
    modes=("default",),
    default_mode="default",
    # Auto-select (rank 40 < Triton 50): on B200 (M=8192, POST-dS-fold) CuTe DSL
    # wins full fwd+bwd at hidden<=8192 for all dtypes (all dtypes cross back to
    # Triton in (8192, 11008], wash through 16384) AND in the [24576, 32768]
    # upper band (1.39-1.63x; the fp32@32768 fwd-smem cell stays Triton via the
    # staged-pair guard ordered first below; in_place band calls keep Triton).
    preference_rank=40,
    tolerances=_CUTEDSL_TOLERANCES,
    notes="CuTe DSL fused_add_rms_norm for Hopper+; auto-selected at hidden<=8192 (all dtypes, post-dS-fold) and in the 24576-32768 upper band (plain, grad-carrying calls; fp32@32768 stays Triton via the fwd smem guard); the (8192, 24576) mid band falls back to Triton).",
)
def fused_add_rms_norm_cutedsl(
    X: torch.Tensor,
    R: torch.Tensor,
    W: torch.Tensor,
    eps: float,
    offset: float = 0.0,
    casting_mode: str = "llama",
    in_place: bool = False,
    *,
    mode: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CuTe DSL fused_add_rms_norm dispatch entry point.

    ``mode`` is accepted only for API parity with the other backends; the only
    valid value is ``"default"`` (or ``None``).
    """
    if mode not in (None, "default"):
        raise ValueError(f"CuTe DSL fused_add_rms_norm has only mode='default'; got mode={mode!r}.")
    vector_width = 16 // X.element_size()
    fallback_reason = None
    if _max_capability() > 100:
        _xmax = _B300_AUTORANK_MAX_HIDDEN
        _xmax_2b = _B300_AUTORANK_MAX_HIDDEN_2B
        _xupper = _B300_AUTORANK_UPPER_MIN_HIDDEN
    else:
        _xmax = _AUTORANK_MAX_HIDDEN
        _xmax_2b = _AUTORANK_MAX_HIDDEN_2B
        _xupper = _AUTORANK_UPPER_MIN_HIDDEN
    if X.shape[-1] % vector_width:
        fallback_reason = f"hidden size {X.shape[-1]} is not divisible by vector width {vector_width}"
    elif _farn_fwd_stages_fp32_pair(X, R, W, casting_mode) and X.shape[-1] > _FWD_SMEM_CAP_TILE_FP32_PAIR:
        fallback_reason = (
            f"hidden size {X.shape[-1]} with an all-fp32 staged X+R pair exceeds"
            f" the sm_100 fwd smem cap tile {_FWD_SMEM_CAP_TILE_FP32_PAIR}"
        )
    elif _farn_fwd_no_grad_ok(X, R, W, in_place):
        # Inference-only wide band: serve the raw fused forward directly
        # (identical code to _LigerFusedAddRMSNormCuTeDSLFunction.forward,
        # minus the autograd saves). Mirrors layer_norm a0bb846.
        X_l = _to_local_if_dtensor(X).contiguous()
        R_l = _to_local_if_dtensor(R).contiguous()
        W_l = _to_local_if_dtensor(W).contiguous()
        if not isinstance(casting_mode, int):
            if casting_mode not in _str_to_casting_mode:
                raise ValueError(f"Invalid casting mode: {casting_mode}")
            casting_mode_int = _str_to_casting_mode[casting_mode]
        else:
            casting_mode_int = casting_mode
        Y, S, _, _ = _fused_add_rms_norm_cutedsl_forward(X_l, R_l, W_l, eps, offset, casting_mode_int)
        return Y, S
    elif X.shape[-1] > (_xmax_2b if X.element_size() == 2 else _xmax) and not (
        # Upper-band window: post-fold the wide-hidden deficit re-inverts from
        # 24576 up for every dtype (bf16/fp16 1.39-1.63x, fp32 1.53x @24576).
        # All-fp32 staged pairs above 28672 never reach here (smem-cap guard
        # ordered first); Triton's in-place dX=X alias erases the win, so
        # in_place=True band calls stay on the Triton fallback.
        _xupper <= X.shape[-1] <= _BWD_MAX_TILE_CUTEDSL and not in_place
    ):
        _xover = _xmax_2b if X.element_size() == 2 else _xmax
        fallback_reason = (
            f"hidden size {X.shape[-1]} exceeds B200 fwd+bwd crossover {_xover}; Triton bwd is faster at wide hidden"
        )
    elif X.shape[-1] > _BWD_MAX_TILE_CUTEDSL:
        fallback_reason = f"hidden size {X.shape[-1]} exceeds CuTe DSL limit {_BWD_MAX_TILE_CUTEDSL}"
    if fallback_reason is not None:
        from liger_kernel.ops.backends._triton.fused_add_rms_norm import fused_add_rms_norm_triton

        emit_fallback_warning(
            "fused_add_rms_norm",
            "nvidia-cutedsl",
            "nvidia-triton",
            fallback_reason,
        )
        return fused_add_rms_norm_triton(
            X,
            R,
            W,
            eps,
            offset,
            casting_mode,
            in_place,
            mode=mode,
        )
    return _LigerFusedAddRMSNormCuTeDSLFunction.apply(X, R, W, eps, offset, casting_mode, in_place)
