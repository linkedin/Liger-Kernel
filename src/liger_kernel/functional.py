"""Functional (autograd-aware) wrappers over the implementation dispatcher.

These thin wrappers exist so user code can write::

    from liger_kernel.functional import rms_norm
    y = rms_norm(x, weight, eps=1e-6)
    y = rms_norm(x, weight, eps=1e-6, impl="nvidia-cutile", mode="static_persistent")

The wrapper's only job is to forward to ``dispatch(op_name, ...)``. All real
logic lives in the registered implementations.

Vocabulary
----------
*Implementation* names use the hyphenated ``<backend>-<dsl>`` form:
``nvidia-triton``, ``nvidia-cutile``, ``nvidia-cutedsl``, ``ascend-triton``, …

Legacy callers can still pass ``backend="cutile"`` (bare DSL name); the
dispatcher translates ``cutile`` → ``nvidia-cutile`` automatically.
"""

from __future__ import annotations

from typing import Optional
from typing import Tuple

import torch

from liger_kernel.backends import dispatch

# Declare where dispatcher should look for op impls. Imported lazily inside
# discover() so missing DSL packages don't break ``import liger_kernel``.
from liger_kernel.backends.registry import declare_op_locations

declare_op_locations(
    "rms_norm",
    (
        "liger_kernel.ops.backends._triton.rms_norm",
        "liger_kernel.ops.backends._cutedsl.rms_norm",
    ),
)

declare_op_locations(
    "layer_norm",
    (
        "liger_kernel.ops.backends._triton.layer_norm",
        "liger_kernel.ops.backends._cutedsl.layer_norm",
    ),
)

# JSD ops: Triton (universal) + cuTile (B200; 8x fwd speedup vs Triton). The
# ``jsd_loss_and_grad`` primitive is exposed so composed ops (fused_linear_jsd)
# route through the dispatcher and pick up cuTile when the user opts in.
declare_op_locations(
    "jsd",
    ("liger_kernel.ops.backends._triton.jsd",),
)
declare_op_locations(
    "jsd_loss_and_grad",
    ("liger_kernel.ops.backends._triton.jsd",),
)

# Softmax: Triton (universal) + cuTile (Blackwell) + CuTe DSL (Hopper+).
declare_op_locations(
    "softmax",
    ("liger_kernel.ops.backends._triton.softmax",),
)

# SwiGLU: Triton (universal) + CuTe DSL (Hopper+). SwiGLU is a purely
# elementwise op (silu(a) * b) so the CuTe DSL kernel avoids Triton launch
# overhead and uses native exp2 for the sigmoid.
declare_op_locations(
    "swiglu",
    ("liger_kernel.ops.backends._triton.swiglu",),
)

# RoPE: Triton (universal) + CuTe DSL (Hopper+). RoPE is an elementwise
# rotation; the CuTe DSL kernel shares one rotate primitive for fwd+bwd.
declare_op_locations(
    "rope",
    ("liger_kernel.ops.backends._triton.rope",),
)

# CrossEntropy: Triton (universal) + CuTe DSL (Hopper+). The CuTe DSL kernel
# uses an online-softmax reduction adapted from Quack.
declare_op_locations(
    "cross_entropy",
    ("liger_kernel.ops.backends._triton.cross_entropy",),
)

# cross_entropy_loss_and_grad: per-chunk CE primitive used by
# fused_linear_cross_entropy so the composed op routes through the dispatcher
# and picks up CuTe DSL on Hopper+.
declare_op_locations(
    "cross_entropy_loss_and_grad",
    ("liger_kernel.ops.backends._triton.cross_entropy",),
)

# fused_add_rms_norm: Triton (universal) + opt-in CuTe DSL (Hopper+). The
# CuTe DSL variant reuses rmsnorm_fwd with a residual input, while Triton
# remains the convergence-safe automatic default.
declare_op_locations(
    "fused_add_rms_norm",
    (
        "liger_kernel.ops.backends._triton.fused_add_rms_norm",
        "liger_kernel.ops.backends._cutedsl.fused_add_rms_norm",
    ),
)

# GeGLU: Triton (universal) + CuTe DSL (Hopper+). Elementwise GELU-tanh(a)*b;
# mirrors the SwiGLU CuTe DSL pattern.
declare_op_locations(
    "geglu",
    ("liger_kernel.ops.backends._triton.geglu",),
)

# KL Divergence: Triton (universal) + CuTe DSL (Hopper+). Forward uses a
# row-reduction (ReductionBase); backward is elementwise.
declare_op_locations(
    "kl_div",
    ("liger_kernel.ops.backends._triton.kl_div",),
)

# fused_linear_jsd: Triton (universal) + cuTile (Blackwell). The CuTe DSL
# registration remains explicit-only because the composed op upcasts its inner
# JSD inputs to fp32, which safely falls back to Triton.
declare_op_locations(
    "fused_linear_jsd",
    ("liger_kernel.ops.backends._triton.fused_linear_jsd",),
)

# fused_linear_cross_entropy: Triton (universal) + CuTe DSL (Hopper+). The
# CuTe DSL variant delegates to LigerFusedLinearCrossEntropyFunction which
# internally dispatches the cross_entropy_loss_and_grad primitive (picking up
# the CuTe DSL kernel on Hopper+).
declare_op_locations(
    "fused_linear_cross_entropy",
    ("liger_kernel.ops.backends._triton.fused_linear_cross_entropy",),
)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    offset: float = 0.0,
    casting_mode: str = "llama",
    in_place: bool = False,
    row_mode: Optional[bool] = None,
    impl: Optional[str] = None,
    mode: Optional[str] = None,
    backend: Optional[str] = None,  # legacy back-compat alias for impl=
) -> torch.Tensor:
    """RMSNorm: ``weight * x / sqrt(mean(x^2) + eps)``.

    Args:
        x: input tensor, shape ``(*, hidden)``.
        weight: scale parameter, shape ``(hidden,)``.
        eps: numerical stabilizer.
        offset: 0.0 for Llama-style; 1.0 for Gemma-style (weight is treated
            as ``(offset + weight)`` inside the kernel).
        casting_mode: ``"llama"`` (cast to fp32 only inside reduction),
            ``"gemma"`` (cast everything to fp32), or ``"none"`` (no casting).
        in_place: if True, the gradient computation reuses ``x``'s storage.
            Has no effect on the forward output.
        row_mode: legacy Liger Triton kernel knob; passed through unchanged.
        impl: explicit implementation name in the new hyphenated form,
            e.g. ``"nvidia-cutile"``, ``"nvidia-cutedsl"``, ``"nvidia-triton"``.
            Legacy bare names (``"cutile"``, ``"cutedsl"``, ``"triton"``) are
            also accepted and translated.
        backend: Legacy back-compat alias for ``impl=``. Use ``impl=`` in new code.
        mode: implementation-specific kernel-variant selection (e.g.,
            ``"static_persistent"`` for cuTile).
    """
    if impl is None and backend is not None:
        impl = backend
    return dispatch(
        "rms_norm",
        x,
        weight,
        eps,
        offset,
        casting_mode,
        in_place,
        row_mode,
        impl=impl,
        mode=mode,
    )


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    *,
    impl: Optional[str] = None,
    mode: Optional[str] = None,
    backend: Optional[str] = None,  # legacy back-compat alias for impl=
) -> torch.Tensor:
    """LayerNorm: ``weight * (x - mean) / sqrt(var + eps) + bias``.

    Args:
        x: input tensor, shape ``(*, hidden)``.
        weight: scale parameter, shape ``(hidden,)``.
        bias: optional shift parameter, shape ``(hidden,)``. If ``None``, a
            zeros tensor is passed through to the impl (the Triton kernel
            requires a real tensor; cuTile / CuTe DSL skip the add when bias
            is the registered zero buffer).
        eps: numerical stabilizer.
        impl: explicit implementation name (``"nvidia-cutile"``, etc.). Legacy
            bare names accepted and translated.
        backend: Legacy back-compat alias for ``impl=``.
        mode: implementation-specific kernel-variant selection.
    """
    if bias is None:
        bias = torch.zeros_like(weight)
    if impl is None and backend is not None:
        impl = backend
    return dispatch(
        "layer_norm",
        x,
        weight,
        bias,
        eps,
        impl=impl,
        mode=mode,
    )


def jsd(
    _input: torch.Tensor,
    target: torch.Tensor,
    shift_labels: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    ignore_index: int = -100,
    *,
    impl: Optional[str] = None,
    mode: Optional[str] = None,
    backend: Optional[str] = None,  # legacy back-compat alias for impl=
) -> torch.Tensor:
    """Generalized Jensen-Shannon Divergence loss.

    .. math::
        JSD_\\beta(P || Q) = \\beta \\cdot KL(P || M) + (1 - \\beta) \\cdot KL(Q || M),
        \\quad M = \\beta P + (1 - \\beta) Q

    Args:
        _input: student log-probabilities, shape ``(BT, V)`` in log-space.
        target: teacher log-probabilities, shape ``(BT, V)`` in log-space.
        shift_labels: optional ``(BT,)`` mask. Rows where the label equals
            ``ignore_index`` contribute zero loss and zero gradient.
        beta: mixing coefficient in ``[0, 1]``. ``0`` -> forward KL,
            ``1`` -> reverse KL, ``0.5`` -> symmetric JSD.
        ignore_index: label value to ignore.
        impl: explicit implementation (``"nvidia-cutile"`` / ``"nvidia-triton"``).
            Legacy bare names (``"cutile"``, ``"triton"``) are also accepted.
        backend: Legacy back-compat alias for ``impl=``.
        mode: implementation-specific kernel-variant selection.
    """
    if impl is None and backend is not None:
        impl = backend
    return dispatch(
        "jsd",
        _input,
        target,
        shift_labels,
        beta,
        ignore_index,
        impl=impl,
        mode=mode,
    )


def softmax(
    x: torch.Tensor,
    *,
    impl: Optional[str] = None,
    mode: Optional[str] = None,
    backend: Optional[str] = None,  # legacy back-compat alias for impl=
) -> torch.Tensor:
    """Row-wise softmax over the last dimension.

    Computes ``y = exp(x - max(x)) / sum(exp(x - max(x)))`` reducing over the
    last axis, with the max/sum reductions performed in fp32 (matching every
    registered Liger backend).

    Args:
        x: input tensor, shape ``(*, hidden)``. Softmax is taken over the last
            dimension.
        impl: explicit implementation name in the hyphenated form, e.g.
            ``"nvidia-cutile"``, ``"nvidia-cutedsl"``, ``"nvidia-triton"``.
            Legacy bare names (``"cutile"``, ``"cutedsl"``, ``"triton"``) are
            also accepted and translated.
        backend: Legacy back-compat alias for ``impl=``. Use ``impl=`` in new code.
        mode: implementation-specific kernel-variant selection (e.g.,
            ``"static_persistent"`` or ``"chunked"`` for cuTile).
    """
    if impl is None and backend is not None:
        impl = backend
    return dispatch(
        "softmax",
        x,
        impl=impl,
        mode=mode,
    )


def swiglu(
    a: torch.Tensor,
    b: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
    *,
    impl: Optional[str] = None,
    mode: Optional[str] = None,
    backend: Optional[str] = None,  # legacy back-compat alias for impl=
) -> torch.Tensor:
    """SwiGLU activation: ``silu(a) * b`` where ``silu(x) = x * sigmoid(x)``.

    ``gate_multiplier`` and ``down_multiplier`` (used by Falcon-H1) pre-scale
    ``a`` and post-scale the output, respectively.

    Args:
        a: gate tensor (pre-activation), shape ``(*, N)``.
        b: up tensor, shape ``(*, N)``.
        gate_multiplier: scalar pre-multiplier on ``a`` before the activation.
        down_multiplier: scalar post-multiplier on the output.
        impl: explicit implementation name (``"nvidia-cutedsl"``,
            ``"nvidia-triton"``). Legacy bare names accepted and translated.
        backend: Legacy back-compat alias for ``impl=``.
        mode: implementation-specific kernel-variant selection.
    """
    if impl is None and backend is not None:
        impl = backend
    return dispatch(
        "swiglu",
        a,
        b,
        float(gate_multiplier),
        float(down_multiplier),
        impl=impl,
        mode=mode,
    )


def rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
    *,
    impl: Optional[str] = None,
    mode: Optional[str] = None,
    backend: Optional[str] = None,  # legacy back-compat alias for impl=
):
    """Rotary Positional Embedding (RoPE).

    Applies the HuggingFace Llama/Mistral rotation to ``q`` and ``k``::

        q1' = q1 * cos - q2 * sin
        q2' = q2 * cos + q1 * sin

    where ``q1``/``q2`` are the first/second halves of ``head_dim``.

    Args:
        q: query tensor, shape ``(bsz, n_q_head, seq_len, head_dim)``.
        k: key tensor, shape ``(bsz, n_kv_head, seq_len, head_dim)``.
        cos: cosine tensor, shape ``(1, seq_len, head_dim)`` or
            ``(bsz, seq_len, head_dim)``.
        sin: sine tensor, same shape as ``cos``.
        position_ids: accepted for API parity (unused by the kernels).
        unsqueeze_dim: accepted for API parity (unused by the kernels).
        impl: explicit implementation name (``"nvidia-cutedsl"``,
            ``"nvidia-triton"``). Legacy bare names accepted and translated.
        backend: Legacy back-compat alias for ``impl=``.
        mode: implementation-specific kernel-variant selection.
    """
    if impl is None and backend is not None:
        impl = backend
    return dispatch(
        "rope",
        q,
        k,
        cos,
        sin,
        position_ids,
        unsqueeze_dim,
        impl=impl,
        mode=mode,
    )


def cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
    *,
    impl: Optional[str] = None,
    mode: Optional[str] = None,
    backend: Optional[str] = None,  # legacy back-compat alias for impl=
):
    """Cross-entropy loss with optional z-loss, token accuracy, and softcap.

    Args:
        input: logits, shape ``(BT, V)``.
        target: target indices, shape ``(BT,)``.
        weight: per-class weight (placeholder, not yet implemented).
        ignore_index: label value to ignore.
        lse_square_scale: z-loss coefficient (``lse_square_scale * logsumexp^2``).
        label_smoothing: label smoothing factor in ``[0, 1]``.
        reduction: ``"mean"`` or ``"sum"`` or ``"none"``.
        softcap: optional logit softcap (``tanh(x / softcap) * softcap``).
        return_z_loss: if True, also return the z-loss component.
        return_token_accuracy: if True, also return token-level accuracy.
        return_predicted_tokens: if True, also return predicted token indices.
        impl: explicit implementation name (``"nvidia-cutedsl"``,
            ``"nvidia-triton"``). Legacy bare names accepted and translated.
        backend: Legacy back-compat alias for ``impl=``.
        mode: implementation-specific kernel-variant selection.

    Returns:
        ``(loss, z_loss, token_accuracy, predicted_tokens)``. Optional outputs
        are ``None`` unless their corresponding return flags are enabled.
    """
    if impl is None and backend is not None:
        impl = backend
    return dispatch(
        "cross_entropy",
        input,
        target,
        weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
        return_token_accuracy,
        return_predicted_tokens,
        impl=impl,
        mode=mode,
    )


def fused_add_rms_norm(
    X: torch.Tensor,
    R: torch.Tensor,
    W: torch.Tensor,
    eps: float = 1e-6,
    offset: float = 0.0,
    casting_mode: str = "llama",
    in_place: bool = False,
    *,
    impl: Optional[str] = None,
    mode: Optional[str] = None,
    backend: Optional[str] = None,  # legacy back-compat alias for impl=
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused add + RMSNorm: ``Y = RMSNorm(X + R)`` returning ``(Y, S)`` where
    ``S = X + R`` (the residual output for the next layer).

    Fuses the residual add into the norm kernel — saves a separate kernel
    launch and a memory round-trip. Used in every transformer layer.

    Args:
        X: input tensor, shape ``(*, hidden)``.
        R: residual tensor, same shape as ``X``.
        W: weight parameter, shape ``(hidden,)``.
        eps: numerical stabilizer.
        offset: 0.0 for Llama-style; 1.0 for Gemma-style.
        casting_mode: ``"llama"``, ``"gemma"``, or ``"none"``.
        in_place: if True, backward reuses ``X``'s storage.
        impl: explicit implementation name (``"nvidia-cutedsl"``,
            ``"nvidia-triton"``). Legacy bare names accepted and translated.
        backend: Legacy back-compat alias for ``impl=``.
        mode: implementation-specific kernel-variant selection.
    """
    if impl is None and backend is not None:
        impl = backend
    return dispatch(
        "fused_add_rms_norm",
        X,
        R,
        W,
        eps,
        offset,
        casting_mode,
        in_place,
        impl=impl,
        mode=mode,
    )


def geglu(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    impl: Optional[str] = None,
    mode: Optional[str] = None,
    backend: Optional[str] = None,  # legacy back-compat alias for impl=
) -> torch.Tensor:
    """GeGLU activation: ``gelu_tanh(a) * b``.

    Args:
        a: gate tensor (pre-activation), shape ``(*, N)``.
        b: up tensor, shape ``(*, N)``.
        impl: explicit implementation name (``"nvidia-cutedsl"``,
            ``"nvidia-triton"``). Legacy bare names accepted and translated.
        backend: Legacy back-compat alias for ``impl=``.
        mode: implementation-specific kernel-variant selection.
    """
    if impl is None and backend is not None:
        impl = backend
    return dispatch(
        "geglu",
        a,
        b,
        impl=impl,
        mode=mode,
    )


def kl_div(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduction: str = "batchmean",
    log_target: bool = False,
    eps: float = 1e-10,
    *,
    impl: Optional[str] = None,
    mode: Optional[str] = None,
    backend: Optional[str] = None,  # legacy back-compat alias for impl=
) -> torch.Tensor:
    """Kullback-Leibler divergence loss.

    Args:
        y_pred: predicted log-probabilities, shape ``(BT, V)``.
        y_true: target probabilities (or log-probs if ``log_target=True``).
        reduction: ``"batchmean"``, ``"mean"``, ``"sum"``, or ``"none"``.
        log_target: if True, ``y_true`` is treated as log-probabilities.
        eps: small value to avoid division by zero.
        impl: explicit implementation name (``"nvidia-cutedsl"``,
            ``"nvidia-triton"``). Legacy bare names accepted and translated.
        backend: Legacy back-compat alias for ``impl=``.
        mode: implementation-specific kernel-variant selection.
    """
    if impl is None and backend is not None:
        impl = backend
    return dispatch(
        "kl_div",
        y_pred,
        y_true,
        reduction,
        log_target,
        eps,
        impl=impl,
        mode=mode,
    )


def fused_linear_jsd(
    student_input: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_input: torch.Tensor,
    teacher_weight: torch.Tensor,
    shift_labels: Optional[torch.Tensor] = None,
    jsd_beta: float = 0.5,
    ignore_index: int = -100,
    temperature: float = 1.0,
    *,
    impl: Optional[str] = None,
    mode: Optional[str] = None,
    backend: Optional[str] = None,  # legacy back-compat alias for impl=
) -> torch.Tensor:
    """Fused linear + JSD loss for RL training (GRPO).

    Computes the JSD loss between student and teacher logits without
    materializing the full log-softmax tensors. Matmuls use cuBLAS; the inner
    JSD computation dispatches through ``jsd_loss_and_grad`` (picking up the
    CuTe DSL kernel on Hopper+).

    Args:
        student_input: student hidden states, shape ``(BT, H)``.
        student_weight: student projection weight, shape ``(V, H)``.
        teacher_input: teacher hidden states, shape ``(BT, H)``.
        teacher_weight: teacher projection weight, shape ``(V, H)``.
        shift_labels: optional ``(BT,)`` mask. Rows where the label equals
            ``ignore_index`` contribute zero loss and zero gradient.
        jsd_beta: JSD mixing coefficient in ``[0, 1]``.
        ignore_index: label value to ignore.
        temperature: softmax temperature scaling.
        impl: explicit implementation name (``"nvidia-cutedsl"``,
            ``"nvidia-triton"``). Legacy bare names accepted and translated.
        backend: Legacy back-compat alias for ``impl=``.
        mode: implementation-specific kernel-variant selection.
    """
    if impl is None and backend is not None:
        impl = backend
    return dispatch(
        "fused_linear_jsd",
        student_input,
        student_weight,
        teacher_input,
        teacher_weight,
        shift_labels,
        jsd_beta,
        ignore_index,
        temperature,
        impl=impl,
        mode=mode,
    )


def fused_linear_cross_entropy(
    _input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ce_weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    accum_dtype: Optional[torch.dtype] = None,
    use_token_scaling: bool = False,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
    *,
    impl: Optional[str] = None,
    mode: Optional[str] = None,
    backend: Optional[str] = None,  # legacy back-compat alias for impl=
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fused linear + cross-entropy loss.

    Computes the cross-entropy loss between logits (``_input @ weight.T``) and
    targets without materializing the full logits tensor. Matmuls use cuBLAS;
    the inner CE computation dispatches through ``cross_entropy_loss_and_grad``
    (picking up the CuTe DSL kernel on Hopper+).

    Args:
        _input: hidden states, shape ``(BT, H)``.
        weight: projection weight, shape ``(V, H)``.
        target: target indices, shape ``(BT,)``.
        bias: optional bias, shape ``(V,)``.
        ce_weight: optional per-class weight, shape ``(V,)``.
        ignore_index: label value to ignore.
        lse_square_scale: z-loss coefficient (``lse_square_scale * logsumexp^2``).
        label_smoothing: label smoothing factor in ``[0, 1]``.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
        softcap: optional logit softcap (``tanh(x / softcap) * softcap``).
        return_z_loss: if True, also return the z-loss component.
        accum_dtype: dtype for gradient accumulation buffers.
        use_token_scaling: whether to scale each token's loss by its predicted probability.
        return_token_accuracy: if True, also return token-level accuracy.
        return_predicted_tokens: if True, also return predicted token indices.
        impl: explicit implementation name (``"nvidia-cutedsl"``,
            ``"nvidia-triton"``). Legacy bare names accepted and translated.
        backend: Legacy back-compat alias for ``impl=``.
        mode: implementation-specific kernel-variant selection.

    Returns:
        ``(loss, z_loss, token_accuracy, predicted_tokens)`` where the latter
        three are ``None`` when not requested.
    """
    if impl is None and backend is not None:
        impl = backend
    return dispatch(
        "fused_linear_cross_entropy",
        _input,
        weight,
        target,
        bias,
        ce_weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
        accum_dtype,
        use_token_scaling,
        return_token_accuracy,
        return_predicted_tokens,
        impl=impl,
        mode=mode,
    )


__all__ = [
    "cross_entropy",
    "fused_add_rms_norm",
    "fused_linear_cross_entropy",
    "fused_linear_jsd",
    "geglu",
    "jsd",
    "kl_div",
    "layer_norm",
    "rms_norm",
    "rope",
    "softmax",
    "swiglu",
]
