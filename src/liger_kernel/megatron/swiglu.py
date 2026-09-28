"""Megatron-Core compatible SwiGLU backed by the Liger Triton SiLU-multiply kernel.

``LigerMegatronSwiGLU`` is an ``nn.Module`` mirroring ``bias_swiglu_impl``'s signature,
for hand-assembled specs (Mode 2) where the caller owns the MLP class. Mode 1 does not use
it: the monkey patch installs a small adapter over
``megatron.core.fusions.fused_bias_swiglu.SwiGLUFunction`` instead -- see
``monkey_patch._patch_swiglu_function``.

Both cover the path ``MLP.forward`` and ``SharedExpertMLP.forward`` take when
``config.bias_activation_fusion=True``, ``config.gated_linear_unit=True`` and
``config.activation_func is F.silu``.

Megatron's native implementation is a chain of ``@jit_fuser`` (TorchScript) helpers::

    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2

Liger fuses the whole thing into a single Triton kernel that computes sigmoid / silu /
multiply in registers, and recomputes ``silu`` in the backward instead of saving it.

This uses ``LigerFusedGateUpSiLUMulFunction`` rather than the two-tensor
``LigerSiLUMulFunction``, because Megatron hands over gate and up concatenated in one
buffer. Splitting them with ``torch.chunk`` yields non-contiguous views that
``@ensure_contiguous`` materializes into two full-size copies, plus a ``torch.cat`` to
reassemble the gradient. Measured on H100 at ``[2048*4, 2*32768]`` bf16, that bridge ran
3.4x slower than the fused kernel and 2.1x slower than Megatron's own TorchScript path --
a regression dressed up as an optimization. The fused kernel reads both halves from the
single buffer via a column offset instead.

Unsupported configurations (bias, FP8 input store, CPU activation offload) transparently
fall back to the native implementation, so enabling Liger can never change behavior for
those users.
"""

from __future__ import annotations

import logging

from typing import Callable
from typing import Optional

import torch
import torch.nn as nn

from liger_kernel.ops import LigerFusedGateUpSiLUMulFunction

logger = logging.getLogger(__name__)


def _unsupported_reason(
    bias: Optional[torch.Tensor],
    fp8_input_store: bool,
    cpu_offload_input: bool,
) -> Optional[str]:
    """Return a non-empty reason string if Liger cannot serve this call, else None."""
    if bias is not None:
        # Liger's kernel has no bias term; fusing it is tracked as follow-up work.
        return "add_bias_linear=True (bias is not None)"
    if fp8_input_store:
        # Megatron stores activations as fp8 and restores dtype in backward;
        # incompatible with Liger's in-place gradient writes.
        return "config.activation_func_fp8_input_store=True"
    if cpu_offload_input:
        # In-place gradient writes into a CPU-offloaded buffer are unvalidated.
        return "CPU activation offloading enabled"
    return None


class LigerMegatronSwiGLU(nn.Module):
    """``bias_swiglu_impl``-compatible SwiGLU using Liger's Triton kernel.

    The call signature matches Megatron's exactly -- including positional order, since
    both ``MLP.forward`` and ``SharedExpertMLP.forward`` call it positionally::

        bias_swiglu_impl(input, bias, fp8_input_store, cpu_offload_input)

    Args:
        fallback_impl: Callable used for configurations Liger cannot serve (bias, FP8
            input store, CPU offload). The monkey patch passes Megatron's captured
            original here. When ``None`` (direct Mode 2 construction) an unsupported
            configuration raises instead of silently changing numerics.
        in_place: Write the backward gradient into the fc1 output buffer instead of
            allocating a new one. Saves one activation-sized allocation -- measured on
            H100 at ``ffn_local=32768`` it moves peak memory from 20% above Megatron to
            20% below it -- at the cost of destroying the fc1 output during backward.

            Default False, matching ``LigerMegatronRMSNorm``. In the standard dense-MLP
            path nothing reads the fc1 output after the activation (``linear_fc1``'s
            backward needs its *input*, and ``linear_fc2``'s needs the activation
            *output*), so this is expected to be safe; it is opt-in because that has not
            been validated against every recompute / offload / CUDA-graph combination.

            Incompatible with backpropagating through the same graph twice
            (``retain_graph=True`` followed by a second ``backward``, or double-backward).
            That is detected and raises; see ``LigerFusedGateUpSiLUMulFunction.backward``
            for why autograd's own version-counter check cannot catch it.
    """

    def __init__(self, fallback_impl: Optional[Callable] = None, in_place: bool = False):
        super().__init__()
        self.fallback_impl = fallback_impl
        self.in_place = in_place
        # Reasons already logged, so a fallback taken every microbatch logs once, not
        # once per step.
        self._logged_fallbacks: set[str] = set()

    def _fallback(
        self,
        reason: str,
        input: torch.Tensor,
        bias: Optional[torch.Tensor],
        fp8_input_store: bool,
        cpu_offload_input: bool,
    ) -> torch.Tensor:
        if self.fallback_impl is None:
            raise RuntimeError(
                f"LigerMegatronSwiGLU cannot serve this call: {reason}. Liger's SwiGLU "
                "kernel supports neither a fused bias term, FP8 input storage, nor CPU "
                "activation offloading. Either construct this module with "
                "`fallback_impl=megatron.core.fusions.fused_bias_swiglu.bias_swiglu_impl` "
                "so unsupported calls defer to Megatron, or use "
                "`apply_liger_kernel_to_megatron(swiglu=True)`, which wires that fallback "
                "up automatically."
            )
        if reason not in self._logged_fallbacks:
            self._logged_fallbacks.add(reason)
            logger.info(
                "Liger SwiGLU is falling back to Megatron's native bias_swiglu_impl: %s. "
                "Numerics and memory behavior are unchanged for this configuration.",
                reason,
            )
        return self.fallback_impl(input, bias, fp8_input_store, cpu_offload_input)

    def forward(
        self,
        input: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        fp8_input_store: bool = False,
        cpu_offload_input: bool = False,
    ) -> torch.Tensor:
        reason = _unsupported_reason(bias, fp8_input_store, cpu_offload_input)
        if reason is not None:
            return self._fallback(reason, input, bias, fp8_input_store, cpu_offload_input)

        ori_shape = input.shape
        if len(ori_shape) not in (2, 3):
            raise ValueError(
                f"LigerMegatronSwiGLU expects a 2D [tokens, 2*ffn] or 3D [seq, batch, 2*ffn] "
                f"input, matching Megatron's bias_swiglu_impl; got shape {tuple(ori_shape)}."
            )
        if ori_shape[-1] % 2 != 0:
            raise ValueError(
                "LigerMegatronSwiGLU expects the gate and up projections concatenated along "
                f"the last dimension, so it must be even; got {ori_shape[-1]}. This usually "
                "means config.gated_linear_unit is False, in which case Megatron does not "
                "route through bias_swiglu_impl at all."
            )

        # Mirror Megatron's own reshape-compute-restore so 3D activations behave
        # identically.
        output = LigerFusedGateUpSiLUMulFunction.apply(input.view(-1, ori_shape[-1]), self.in_place)
        return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
