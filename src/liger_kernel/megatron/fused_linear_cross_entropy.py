"""Megatron-facing module for tensor-parallel fused linear cross entropy."""

from __future__ import annotations

import torch
import torch.nn as nn

from liger_kernel.ops import LigerMegatronFusedLinearCrossEntropyFunction


class LigerMegatronFusedLinearCrossEntropy(nn.Module):
    """Fuse a vocab-sharded output projection with per-token cross entropy.

    ``hidden`` is replicated across TP ranks and ``weight`` contains the local
    contiguous vocabulary shard. The output shape matches ``target``.
    """

    def __init__(
        self,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_group=None,
    ) -> torch.Tensor:
        return LigerMegatronFusedLinearCrossEntropyFunction.apply(
            hidden,
            weight,
            target,
            bias,
            tp_group,
            self.ignore_index,
        )

    def extra_repr(self) -> str:
        return f"ignore_index={self.ignore_index}"


def liger_megatron_fused_linear_cross_entropy_output_processor(
    *,
    hidden_states: torch.Tensor,
    output_layer,
    output_weight: torch.Tensor | None,
    labels: torch.Tensor,
    runtime_gather_output: bool | None,
    config,
    **_,
) -> torch.Tensor:
    """Megatron ``GPTModel`` output processor for the native TP output layer."""
    unsupported = []
    if type(output_layer).__name__ != "ColumnParallelLinear":
        unsupported.append("the output layer is not Megatron's native ColumnParallelLinear")
    if getattr(output_layer, "sequence_parallel", False):
        unsupported.append("sequence_parallel=True")
    if getattr(output_layer, "gradient_accumulation_fusion", False):
        unsupported.append("gradient_accumulation_fusion=True")
    if getattr(output_layer, "disable_grad_reduce", False):
        unsupported.append("output-layer dgrad reduction is disabled")
    if getattr(output_layer, "explicit_expert_comm", False):
        unsupported.append("the output layer uses explicit expert communication")
    if getattr(output_layer, "skip_bias_add", False):
        unsupported.append("the output layer returns bias separately")
    if getattr(config, "defer_embedding_wgrad_compute", False):
        unsupported.append("defer_embedding_wgrad_compute=True")
    if getattr(config, "mtp_num_layers", None):
        unsupported.append("MTP is enabled")
    if getattr(config, "use_mup", False):
        unsupported.append("MuP output scaling is enabled")

    gather_output = (
        getattr(output_layer, "gather_output", False) if runtime_gather_output is None else runtime_gather_output
    )
    if gather_output:
        unsupported.append("the output layer gathers TP logits")
    if unsupported:
        raise RuntimeError(
            "Liger Megatron FLCE does not support this GPT output configuration: " + "; ".join(unsupported)
        )

    weight = output_weight if output_weight is not None else getattr(output_layer, "weight", None)
    if weight is None:
        raise RuntimeError("Liger Megatron FLCE requires an output weight tensor.")

    labels_sb = labels.transpose(0, 1).contiguous()
    loss_sb = LigerMegatronFusedLinearCrossEntropyFunction.apply(
        hidden_states,
        weight,
        labels_sb,
        getattr(output_layer, "bias", None),
        getattr(output_layer, "tp_group", None),
        -100,
    )
    return loss_sb.transpose(0, 1).contiguous()
