"""Megatron-Core compatible RMSNorm backed by the Liger Triton kernel."""

from __future__ import annotations

import numbers

import torch

# Force-import the submodule so liger_kernel.ops.rms_norm can resolve
# torch.distributed.tensor.DTensor on torch 2.11+, where the subpackage is no
# longer auto-loaded as an attribute of torch.distributed.
import torch.distributed.tensor  # noqa: F401

from torch.nn import Parameter
from torch.nn import init

from liger_kernel.ops import LigerRMSNormFunction


class LigerMegatronRMSNorm(torch.nn.Module):
    """RMSNorm module conforming to Megatron-Core's ``LayerNormBuilder`` protocol.

    Drop-in for ``megatron.core.transformer.torch_norm.WrappedTorchNorm`` and
    ``megatron.core.fusions.fused_layer_norm.FusedLayerNorm``. The constructor
    accepts the same keyword arguments those types accept so it can be slotted
    into a ``TransformerLayerSubmodules`` without further glue.

    Args:
        config: TransformerConfig. Duck-typed; only ``config.normalization``,
            ``config.sequence_parallel`` and ``config.layernorm_zero_centered_gamma``
            are read.
        hidden_size: Trailing dimension to normalize over.
        eps: Variance epsilon (matches ``layernorm_epsilon`` from the config).
        persist_layer_norm: Accepted for interface compatibility, unused.
        zero_centered_gamma: If True, the weight is stored centered at 0 and
            the kernel applies ``(1 + w)`` as the effective scale. Mirrors
            ``apex.normalization.FusedLayerNorm``'s ``zero_centered_gamma``
            semantics. Defaults to the value of
            ``config.layernorm_zero_centered_gamma`` if present.
        normalization: Accepted for interface compatibility; must be ``"RMSNorm"``.
    """

    def __init__(
        self,
        config,
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "RMSNorm",
    ):
        super().__init__()
        cfg_norm = getattr(config, "normalization", "RMSNorm")
        if cfg_norm != "RMSNorm":
            raise ValueError(f"LigerMegatronRMSNorm requires config.normalization='RMSNorm'; got {cfg_norm!r}.")

        self.config = config
        self.zero_centered_gamma = bool(zero_centered_gamma or getattr(config, "layernorm_zero_centered_gamma", False))

        if isinstance(hidden_size, numbers.Integral):
            hidden_size = (hidden_size,)
        self.hidden_size = torch.Size(hidden_size)
        self.eps = eps

        # Liger's kernel applies (offset + w) * x_normalized. Zero-centered gamma
        # is implemented by storing w around 0 and folding +1 into the kernel via
        # offset; this matches apex.normalization.FusedLayerNorm's semantics.
        self._offset = 1.0 if self.zero_centered_gamma else 0.0
        self.weight = Parameter(torch.empty(*self.hidden_size))
        self.reset_parameters()

        # Megatron's distributed optimizer inspects this attribute to decide
        # whether to all-reduce the parameter's gradient across TP ranks
        # under sequence parallelism. Match the flag-on-weight convention
        # used by FusedLayerNorm.
        #
        # TODO(sp): under pure TP (no SP) every TP rank holds the full
        # ``[s, b, h]`` activation and runs the same RMSNorm — wasting
        # `tp_world_size`x of compute and activation memory. The fix is for
        # the user to enable SP, in which case the activation arrives
        # already sharded as ``[s/tp, b, h]`` and the per-token RMSNorm
        # runs at 1/tp the cost. This wrapper does not block SP (we set
        # the gradient-reduction marker below), but the SP path has not
        # been E2E verified yet.
        sp = bool(getattr(config, "sequence_parallel", False))
        setattr(self.weight, "sequence_parallel", sp)

    def reset_parameters(self) -> None:
        if self.zero_centered_gamma:
            init.zeros_(self.weight)
        else:
            init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # in_place=False because Megatron holds activation references for
        # recompute (TransformerLayer.recompute_input_layernorm path) and
        # CUDA-graph capture; in-place writes would corrupt them.
        return LigerRMSNormFunction.apply(
            x,
            self.weight,
            self.eps,
            self._offset,
            "llama",  # casting_mode
            False,  # in_place
            None,  # row_mode
        )

    def extra_repr(self) -> str:
        return f"hidden_size={tuple(self.hidden_size)}, eps={self.eps}, zero_centered_gamma={self.zero_centered_gamma}"
