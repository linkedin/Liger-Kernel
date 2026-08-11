"""
Hopper (SM90) CuTe DSL **fused scaled cross entropy**.

This operator fuses the classifier projection ``Z = X @ W.T`` with the cross
entropy softmax so the ``[M, V]`` logits are never written to HBM in the
forward pass.  It is deliberately *narrower* than the Triton
``LigerFusedLinearCrossEntropyFunction``:

* the forward returns the per-token negative log-likelihood ``nll[M]`` and,
  when requested, differentiable per-token vocabulary entropy -- never a
  ``mean``/``sum`` reduction;
* the backward consumes the **per-token upstream gradient** ``grad_output[M]``,
  which is exactly the row scale of ``dZ``.  A scalar ``grad_output`` is
  rejected instead of being silently interpreted as a reduction;
* there is no ``bias``, ``ce_weight``, label smoothing, ``softcap``,
  ``lse_square_scale``, z-loss or token-scaling support.

Anything that needs a reduction composes it in PyTorch, e.g.::

    nll = LigerFusedScaledCrossEntropySM90Function.apply(x, w, target)
    loss = nll.sum() / (target != -100).sum().clamp_min(1)

which keeps the reduction (and therefore the per-token upstream gradient) in
autograd where it is cheap and unambiguous.

Requirements
------------
Hopper (compute capability 9.0), BF16 ``input``/``weight``, ``input[M, H]``,
``weight[V, H]``, ``target[M]`` int64.  ``H`` and ``V`` may be ragged; they are
padded internally and the returned gradients are sliced back to the caller's
shapes and dtypes.

Kernels
-------
=========================  ==================================================
phase                      module
=========================  ==================================================
forward                    ``_fused_scaled_cross_entropy_forward_fragment_sm90``
backward dZ+dX+dW          ``_fused_scaled_cross_entropy_backward_fused_sm90``
=========================  ==================================================

The fixed forward uses cluster-M2 weight multicast, two N160 accumulators and
four N80 online-softmax folds per logical N320 vocabulary tile.
``m_tiles_per_cluster`` remains accepted for API compatibility, but every
positive value selects this same forward.  Profiled long-sequence shapes use
a split-N lookup table; all other shapes retain the hardware-derived split.

Backward runs in one persistent cluster-2 CUDA kernel.  Its device-side wave
loop executes dZ, dX, and dW with phase-local schedulers over a phase-serial
shared-memory arena: dZ uses its static persistent scheduler, dX interprets each
cluster pair as W-multicast peers, and dW treats both CTAs as independent
output-tile workers with transposed (``order=(1, 0, 2)``) TMA.  A GPU-wide HBM
atomic barrier publishes dZ and protects the reusable workspace between waves.

``temperature`` (default ``1.0``) follows Verl semantics: softmax and NLL use
``logits / temperature`` and backward applies the corresponding
``1 / temperature`` chain-rule factor.

Backward always processes eight ``M128`` tiles (1024 tokens) per device-side
wave, reusing one BF16 dZ buffer.  At the representative shape this holds a
~256 MiB workspace instead of the ~1 GiB full dZ.  Wave zero plain-stores BF16
dW; later waves use Hopper BF16 TMA reduce-add.  FP32 WGMMA accumulators are
rounded in the epilogue, with no FP32 HBM accumulator or final cast.
``LIGER_SCALED_CE_SM90_VALIDATE_TARGETS=0`` disables the (host-synchronising)
out-of-range target check.
"""

import math
import os

import torch

from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_backward_fused_sm90 import FusedBackwardConfig
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_backward_fused_sm90 import fused_backward_one_kernel
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_forward_fragment_sm90 import ScaledCEForwardFragmentConfig
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_forward_fragment_sm90 import scaled_ce_forward_fragment
from liger_kernel.ops.cutedsl.ops.utils import ensure_cuda_context
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd

# The forward and every backward phase tile the hidden dimension by 64, so one
# padding of H covers the full operation.
HIDDEN_TILE = 64
# dZ and dX both tile the token dimension by 128, so a backward wave is counted
# in M128 tiles.
BACKWARD_M_TILE = 128
# 8 M128 tiles == 1024 tokens per wave: a ~256 MiB BF16 dZ buffer at
# V = 131072 instead of the ~1 GiB a 4096-token batch would need.
BACKWARD_M_TILES_PER_WAVE = 8
VALIDATE_TARGETS_ENV = "LIGER_SCALED_CE_SM90_VALIDATE_TARGETS"

# Measured H100 winners for the representative large-vocabulary workload.
# Unlisted shapes retain the fragment kernel's hardware-derived split.
_FORWARD_CONFIG_BY_SHAPE = {
    (16384, 4096, 131072, False): ScaledCEForwardFragmentConfig(split_n=4),
    (16384, 4096, 131072, True): ScaledCEForwardFragmentConfig(split_n=2),
    (32768, 4096, 131072, False): ScaledCEForwardFragmentConfig(split_n=2),
    (32768, 4096, 131072, True): ScaledCEForwardFragmentConfig(split_n=2),
}


def _select_forward_config(tokens, hidden_size, vocab_size, return_entropy):
    return _FORWARD_CONFIG_BY_SHAPE.get((tokens, hidden_size, vocab_size, return_entropy))


def _validate_temperature(temperature):
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a real number")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")


def _validate_targets_enabled() -> bool:
    return os.environ.get(VALIDATE_TARGETS_ENV, "1") not in ("0", "false", "False")


def _validate_inputs(_input, weight, target, ignore_index, m_tiles_per_cluster):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(_input.device) != (9, 0):
        raise RuntimeError("CuTe DSL fused scaled cross entropy requires a Hopper compute capability 9.0 GPU")
    if _input.device != weight.device or _input.device != target.device:
        raise ValueError("input, weight, and target must be on the same CUDA device")
    if _input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("CuTe DSL fused scaled cross entropy supports BF16 input and weight only")
    if target.dtype != torch.int64:
        raise TypeError("target must be an int64 tensor")
    if _input.ndim != 2 or weight.ndim != 2 or target.ndim != 1:
        raise ValueError("expected input[M,H], weight[V,H], and target[M]")
    if _input.shape[0] != target.shape[0] or _input.shape[1] != weight.shape[1]:
        raise ValueError(
            f"input {tuple(_input.shape)}, weight {tuple(weight.shape)} and target "
            f"{tuple(target.shape)} shapes are incompatible"
        )
    if not isinstance(m_tiles_per_cluster, int) or isinstance(m_tiles_per_cluster, bool):
        raise TypeError("m_tiles_per_cluster must be an int")
    if m_tiles_per_cluster < 1:
        raise ValueError("m_tiles_per_cluster must be >= 1")
    if not _validate_targets_enabled():
        return
    vocab_size = weight.shape[0]
    out_of_range = ((target < 0) | (target >= vocab_size)) & (target != ignore_index)
    if bool(out_of_range.any()):
        raise ValueError(f"target contains values outside [0, {vocab_size}) that are not ignore_index={ignore_index}")


def _validate_grad_output(grad_output, tokens):
    if not isinstance(grad_output, torch.Tensor):
        raise TypeError("grad_output must be a tensor of per-token gradients")
    if grad_output.ndim != 1 or grad_output.shape[0] != tokens:
        raise ValueError(
            "CuTe DSL fused scaled cross entropy backward expects the per-token upstream gradient of shape "
            f"[{tokens}], got {tuple(grad_output.shape)}. Apply any reduction outside the kernel so that "
            "autograd supplies a per-token gradient."
        )


def _pad_hidden(x, weight):
    """Pad ``H`` up to the shared ``HIDDEN_TILE`` so every phase tiles exactly."""
    hidden_size = x.shape[1]
    padded = (hidden_size + HIDDEN_TILE - 1) // HIDDEN_TILE * HIDDEN_TILE
    if padded == hidden_size:
        return x.contiguous(), weight.contiguous(), hidden_size
    pad = (0, padded - hidden_size)
    return (
        torch.nn.functional.pad(x.contiguous(), pad),
        torch.nn.functional.pad(weight.contiguous(), pad),
        hidden_size,
    )


def fused_scaled_cross_entropy_forward(
    _input,
    weight,
    target,
    temperature=1.0,
    ignore_index=-100,
    m_tiles_per_cluster=1,
    return_entropy=False,
):
    """Per-token fused scaled cross entropy forward.

    Returns ``(nll, entropy, lse, x_padded, weight_padded, hidden_size)``.
    ``nll`` is the ``[M]`` FP32 negative log-likelihood, ``entropy`` is BF16
    when requested (otherwise ``None``), and ``lse`` is the FP32 log-sum-exp
    reused by backward.
    """
    _validate_inputs(_input, weight, target, ignore_index, m_tiles_per_cluster)
    _validate_temperature(temperature)
    if not isinstance(return_entropy, bool):
        raise TypeError("return_entropy must be a bool")
    ensure_cuda_context()
    x_padded, weight_padded, hidden_size = _pad_hidden(_input, weight)
    target = target.contiguous()

    nll, entropy, lse = scaled_ce_forward_fragment(
        x_padded,
        weight_padded,
        target,
        temperature,
        ignore_index,
        return_entropy,
        config=_select_forward_config(
            x_padded.shape[0],
            x_padded.shape[1],
            weight_padded.shape[0],
            return_entropy,
        ),
    )
    return nll, entropy, lse, x_padded, weight_padded, hidden_size


def fused_scaled_cross_entropy_backward(
    grad_output,
    x_padded,
    weight_padded,
    target,
    lse,
    temperature,
    ignore_index,
    hidden_size,
    entropy=None,
    grad_entropy=None,
    input_dtype=torch.bfloat16,
):
    """Backward of :func:`fused_scaled_cross_entropy_forward`.

    ``grad_output`` is the ``[M]`` upstream gradient of the per-token NLL and is
    divided by ``temperature`` and used as the row scale of ``dZ``.  Returns
    ``(grad_input[M, H], grad_weight[V, H])`` in the caller's dtypes.

    Backward streams tokens through a reusable BF16 dZ buffer in fixed
    1024-token waves.
    """
    tokens = x_padded.shape[0]
    _validate_grad_output(grad_output, tokens)
    _validate_temperature(temperature)
    # Backward usually runs on an autograd worker thread, which has no driver
    # context bound; the CuTe DSL occupancy helpers need one.
    ensure_cuda_context()

    scale = (grad_output.detach().to(torch.float32) / temperature).contiguous()
    entropy_scale = None
    if grad_entropy is not None:
        entropy_scale = (grad_entropy.detach().to(torch.float32) / temperature).contiguous()
    total_m_tiles = (tokens + BACKWARD_M_TILE - 1) // BACKWARD_M_TILE
    m_tiles_per_wave = min(BACKWARD_M_TILES_PER_WAVE, total_m_tiles)
    return fused_backward_one_kernel(
        scale,
        x_padded,
        weight_padded,
        target,
        lse,
        entropy,
        entropy_scale,
        ignore_index,
        hidden_size,
        input_dtype,
        temperature,
        config=FusedBackwardConfig(m_tiles_per_wave=m_tiles_per_wave),
    )


class LigerFusedScaledCrossEntropySM90Function(torch.autograd.Function):
    """``apply(_input, weight, target, temperature=1.0, ignore_index=-100,
    m_tiles_per_cluster=1, return_entropy=False)``.

    Forward returns the per-token NLL ``[M]`` (FP32).  Backward expects the
    per-token upstream gradient ``[M]``.  ``m_tiles_per_cluster`` is retained
    for API compatibility; backward always uses fixed 1024-token waves.
    """

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        temperature=1.0,
        ignore_index=-100,
        m_tiles_per_cluster=1,
        return_entropy=False,
    ):
        ctx.set_materialize_grads(False)
        _validate_temperature(temperature)
        nll, entropy, lse, x_padded, weight_padded, hidden_size = fused_scaled_cross_entropy_forward(
            _input,
            weight,
            target,
            temperature,
            ignore_index,
            m_tiles_per_cluster,
            return_entropy,
        )
        saved_entropy = entropy if entropy is not None else torch.empty(0, device=_input.device, dtype=torch.bfloat16)
        ctx.save_for_backward(x_padded, weight_padded, target, lse, saved_entropy)
        ctx.ignore_index = ignore_index
        ctx.temperature = temperature
        ctx.hidden_size = hidden_size
        ctx.input_dtype = _input.dtype
        ctx.return_entropy = return_entropy
        return (nll, entropy) if return_entropy else nll

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_entropy=None):
        x_padded, weight_padded, target, lse, entropy = ctx.saved_tensors
        if grad_output is None:
            grad_output = torch.zeros(target.shape[0], device=target.device, dtype=torch.float32)
        grad_input, grad_weight = fused_scaled_cross_entropy_backward(
            grad_output,
            x_padded,
            weight_padded,
            target,
            lse,
            ctx.temperature,
            ctx.ignore_index,
            ctx.hidden_size,
            entropy=entropy if ctx.return_entropy else None,
            grad_entropy=grad_entropy,
            input_dtype=ctx.input_dtype,
        )
        return grad_input, grad_weight, None, None, None, None, None


__all__ = [
    "LigerFusedScaledCrossEntropySM90Function",
    "fused_scaled_cross_entropy_backward",
    "fused_scaled_cross_entropy_forward",
]
