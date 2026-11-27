"""
Tests for CuTe-DSL Newton-Schulz iteration and LigerMuon optimizer.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from liger_kernel.ops.cutedsl.ops.newton_schulz import _reference_newton_schulz_5
from liger_kernel.ops.newton_schulz import liger_newton_schulz
from liger_kernel.transformers.muon import LigerMuon


@pytest.mark.parametrize(
    "m, n",
    [
        (128, 128),
        (256, 64),
        (64, 256),
        (512, 128),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_newton_schulz_numerical_parity(m, n, dtype):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = torch.randn(m, n, device=device, dtype=dtype)
    out_liger = liger_newton_schulz(G, steps=5)
    out_ref = _reference_newton_schulz_5(G, steps=5)

    assert out_liger.shape == G.shape
    assert out_liger.dtype == G.dtype
    assert torch.allclose(out_liger, out_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dim", [64, 128])
def test_newton_schulz_properties(dim):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = torch.randn(dim, dim, device=device, dtype=torch.float32)
    X = liger_newton_schulz(G, steps=5)

    # In Muon quintic Newton-Schulz, singular values are bounded in [0, 2.0]
    # (empirically ~0.5 to 1.5 for well-conditioned components)
    s_orth = torch.linalg.svdvals(X)
    assert s_orth.max().item() <= 2.0, f"Max singular value exceeds bound: {s_orth.max()}"
    assert not torch.isnan(X).any(), "Found NaNs in Newton-Schulz output"
    assert not torch.isinf(X).any(), "Found Infs in Newton-Schulz output"
    assert X.shape == G.shape
    assert X.dtype == G.dtype


def test_liger_muon_optimizer_convergence():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Linear(64, 128, bias=True),
        nn.GELU(),
        nn.Linear(128, 10, bias=True),
    ).to(device)

    optimizer = LigerMuon(
        model.parameters(),
        lr=0.02,
        momentum=0.95,
        adamw_lr=1e-3,
    )

    x = torch.randn(32, 64, device=device)
    target = torch.randint(0, 10, (32,), device=device)

    losses = []
    for _ in range(15):
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss must decrease significantly over training (> 30% reduction)
    assert losses[-1] < losses[0] * 0.7, f"Loss did not decrease sufficiently: initial={losses[0]}, final={losses[-1]}"
