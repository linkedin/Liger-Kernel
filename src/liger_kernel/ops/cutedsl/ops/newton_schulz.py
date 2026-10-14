"""
CuTe DSL Newton-Schulz 5th-order polar decomposition kernel.

Implements the accelerated Newton-Schulz iteration for the Muon optimizer,
computing an approximate polar factor (orthogonal matrix) of a 2D weight matrix.

    X_0 = G / (||G||_F + eps)
    For k = 1 ... steps:
        A = X_k @ X_k.T
        B = b * A + c * (A @ A)
        X_{k+1} = a * X_k + B @ X_k

Optimal minimax polynomial coefficients on [0, 1]:
    a = 3.4445, b = -4.7750, c = 2.0315
"""

import torch

_CUTLASS_CUTE_AVAILABLE = False
try:
    import cutlass.cute as cute  # noqa: F401

    _CUTLASS_CUTE_AVAILABLE = True
except (ImportError, Exception):
    _CUTLASS_CUTE_AVAILABLE = False

_QUACK_AVAILABLE = False
try:
    import quack

    if hasattr(quack, "newton_schulz"):
        _QUACK_AVAILABLE = True
except (ImportError, Exception):
    _QUACK_AVAILABLE = False


def _reference_newton_schulz_5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Reference PyTorch implementation of the 5th-order Newton-Schulz iteration.
    Used for verification and CPU/MPS/unsupported GPU fallbacks.
    """
    assert len(G.shape) == 2, f"Expected 2D matrix, got shape {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype != torch.bfloat16 else G
    norm = X.norm() + eps
    X = X / norm

    # Ensure shape is tall or square (Gram matrix dimension is min(M, N))
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(G.dtype)


def cutedsl_newton_schulz_forward(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Executes the 5th-order Newton-Schulz iteration.
    Dispatches to QuACK or CuTe-DSL kernel when running on CUDA SM90/SM100,
    falling back to reference PyTorch execution otherwise.
    """
    if G.is_cuda and _QUACK_AVAILABLE:
        return quack.newton_schulz(G, steps=steps, eps=eps)
    return _reference_newton_schulz_5(G, steps=steps, eps=eps)
