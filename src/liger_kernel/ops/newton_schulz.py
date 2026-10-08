"""
Newton-Schulz operator for Liger-Kernel.
Used to compute the approximate polar decomposition of weight gradient matrices
in the Muon optimizer.
"""

import os

import torch

from liger_kernel.ops.cutedsl.ops.newton_schulz import _reference_newton_schulz_5
from liger_kernel.ops.cutedsl.ops.newton_schulz import cutedsl_newton_schulz_forward

# Check if cutedsl is selected via environment or available
_IMPL = os.environ.get("LIGER_KERNEL_IMPL", "").lower()


def liger_newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes 5th-order Newton-Schulz polar decomposition of matrix G.

    Args:
        G: 2D Tensor to orthogonalize.
        steps: Number of Newton-Schulz iterations (default: 5).
        eps: Epsilon for norm stabilization (default: 1e-7).

    Returns:
        Orthogonalized 2D Tensor with the same shape and dtype as G.
    """
    assert len(G.shape) == 2, f"liger_newton_schulz expects a 2D matrix, got shape {G.shape}"

    if G.is_cuda and (_IMPL == "cutedsl" or G.is_cuda):
        return cutedsl_newton_schulz_forward(G, steps=steps, eps=eps)
    return _reference_newton_schulz_5(G, steps=steps, eps=eps)
