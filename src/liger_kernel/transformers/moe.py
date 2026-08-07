"""Frontend ``nn.Module`` for the fused expert-parallel MoE op.

This module exposes :class:`LigerExpertParallelFusedMoe`, a thin ``nn.Module``
wrapper around the fused expert-parallel MoE autograd function. It owns no
weights of its own — the per-expert MLP matrices are supplied by the caller at
:meth:`forward` time — so it is purely an op wrapper, not a parameterized layer.
The autograd op currently lives in the ``cute`` backend
(``liger_kernel.ops.cute.ops.moe``); ``cute`` is just the backend implementation
and the frontend stays backend-neutral.

The op is backed by the separate, optional ``liger_cute_kernels`` (lck) wheel.
Importing it eagerly here means that pulling in this module on an environment
without the lck wheel fails fast with a clear, actionable error rather than
blowing up later inside ``forward``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Optional

import torch
import torch.nn as nn

# Import the backend autograd op eagerly so a missing/unbuildable backend
# sub-package surfaces immediately as a clear error at import time. The op module
# loads the native ``liger_cute_kernels.tvm_ffi`` facade at its own import, which
# raises ImportError when the lck wheel is absent.
try:
    from liger_kernel.ops.cute.ops.moe import moe_fused
except ImportError as exc:  # pragma: no cover - depends on a CUDA/lck build
    raise ImportError(
        "LigerExpertParallelFusedMoe requires the fused MoE backend, which is "
        "backed by the separate liger_cute_kernels (lck) wheel. Install the "
        "matching lck wheel for your CUDA/torch environment, or build it locally "
        "(see the liger_cute_kernels/ module at the repo root)."
    ) from exc

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

__all__ = ["LigerExpertParallelFusedMoe"]


class LigerExpertParallelFusedMoe(nn.Module):
    """Thin ``nn.Module`` wrapper around the fused expert-parallel MoE op.

    The module holds no parameters: the activations, routing, and all per-expert
    weights are passed into :meth:`forward`. Its only state is the expert-parallel
    ``process_group`` used to resolve the NVSHMEM team for remote experts.

    Expected weight layout (all ``bf16``, contiguous), with ``E`` experts, hidden
    size ``H`` and intermediate size ``I``:

      - ``all_B``: ``[E, I, H]`` — gate projection.
      - ``all_C``: ``[E, I, H]`` — up projection.
      - ``all_A``: ``[E, H, I]`` — down projection.

    Args:
        process_group: Expert-parallel ``ProcessGroup`` whose ranks hold the
            remote experts. ``None`` (the default) runs purely local
            (NVSHMEM_TEAM_WORLD); for multi-rank expert parallelism pass the EP
            group. A proper subgroup must first be prepared collectively on all
            NVSHMEM PEs with ``liger_cute_kernels.nvshmem.resolve_team(pg)``
            during distributed setup.
    """

    def __init__(self, process_group: Optional["ProcessGroup"] = None) -> None:
        super().__init__()
        self.process_group = process_group

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        all_B: torch.Tensor,
        all_C: torch.Tensor,
        all_A: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        """Run the fused MoE forward.

        Args:
            hidden_states: Input activations of shape ``(..., H)``. Leading dims
                are flattened to a token axis and restored on the output.
            expert_indices: Selected expert ids, shape ``(..., top_k)`` (cast to
                int32).
            expert_weights: Per-selection combine weights, shape ``(..., top_k)``.
            all_B: Gate-projection weights, ``[E_local, I, H]``.
            all_C: Up-projection weights, ``[E_local, I, H]``.
            all_A: Down-projection weights, ``[E_local, H, I]``.
            num_experts: Total number of experts **across the whole EP group**.
                This is required, not inferred from ``all_B``: under expert
                parallelism each rank holds only its shard (``all_B.shape[0]`` is
                the local expert count ``E_local``, not the global total), and the
                kernel routes against the global expert id space.

        Returns:
            Output tensor with the same shape as ``hidden_states``.
        """
        hidden_dim = hidden_states.shape[-1]
        # top_k is reliably the routing fan-out (expert_indices' last dim), so it
        # is inferred; num_experts cannot be — see the arg docstring.
        top_k = expert_indices.shape[-1]

        orig_shape = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_dim)

        expert_indices_2d = expert_indices.reshape(-1, top_k).to(torch.int32)
        expert_weights_2d = expert_weights.reshape(-1, top_k)

        out = moe_fused(
            x,
            expert_indices_2d,
            expert_weights_2d,
            all_B,
            all_C,
            all_A,
            num_experts,
            top_k,
            self.process_group,
        )
        return out.reshape(orig_shape)
