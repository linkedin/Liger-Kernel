"""Dispatcher-integration tests for the ``kl_loss_and_grad`` primitive.

The fused-linear KL composed op used to launch ``_kl_div_kernel`` directly,
bypassing impl selection — the same class of bug as linkedin/Liger-Kernel#1228
that ``jsd_loss_and_grad`` fixed for ``fused_linear_jsd``. These tests pin two
things:

1. the Triton primitive's numerics against a PyTorch reference (per-row loss +
   dL/dlogits, with and without ``ignore_index`` masking), and
2. that the composed op routes its inner computation through the dispatcher.

End-to-end fused-linear-KL correctness (shapes, dtypes, reductions, AMP) lives
in ``test/transformers/test_fused_linear_kl_div.py``.

Mirrors ``test/ops/test_fused_linear_jsd.py`` in structure: collects cleanly on
a CPU-only box (the conftest ``autouse`` fixture skips when CUDA is unavailable).
"""

from __future__ import annotations

import pytest
import torch

# Importing functional registers the discovery map so available_backends works.
import liger_kernel.functional  # noqa: F401
import liger_kernel.ops.fused_linear_kl_div as flkl_ops

from liger_kernel.backends.dispatch import available_backends
from liger_kernel.backends.dispatch import dispatch
from liger_kernel.ops.backends._triton.kl_div import kl_loss_and_grad_triton

from .conftest import get_available_backends_for_op

KL_PRIMITIVE_SHAPES = [
    (32, 512),  # (chunk_BT, V) — single BLOCK_SIZE launch
    (64, 4096),  # V > 2048 → multi-iteration row loop
]
KL_PRIMITIVE_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

_REGISTERED_BACKENDS = get_available_backends_for_op("kl_loss_and_grad")

# All kernel math runs on fp32 values upcast in registers from the native
# logits, and the reference upcasts the same values, so tolerances are far
# tighter than the composed op's end-to-end tables (no GEMM inside the
# primitive; only reduction-order and exp/log ULP differences remain).
_TOLERANCES = {
    torch.float16: {"atol": 1e-5, "rtol": 1e-4},
    torch.bfloat16: {"atol": 1e-5, "rtol": 1e-4},
    torch.float32: {"atol": 1e-6, "rtol": 1e-5},
}


def _kl_loss_and_grad_ref(logits, target, shift_labels, ignore_index, temperature, eps, scale):
    """PyTorch reference for one chunk: per-row scaled loss + dL/dlogits.

    Matches the kernel's algebra: q is clamped only inside the log (0 * log 0
    = 0), while the sums and the gradient use the raw q.
    """
    x = logits.float() / temperature
    lse = torch.logsumexp(x, dim=-1)
    q = target.float()

    a1 = (q * (q.clamp_min(eps).log() - x)).sum(dim=-1)
    a2 = q.sum(dim=-1)
    loss_rows = (a1 + lse * a2) * scale

    softmax = torch.softmax(x, dim=-1)
    dx = -scale * (q - softmax * a2.unsqueeze(-1)) / temperature

    if shift_labels is not None:
        ignored = shift_labels == ignore_index
        loss_rows = loss_rows.masked_fill(ignored, 0.0)
        dx = torch.where(ignored.unsqueeze(-1), torch.zeros_like(dx), dx)
    return loss_rows, dx


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
@pytest.mark.parametrize("with_labels", [False, True])
@pytest.mark.parametrize("dtype", KL_PRIMITIVE_DTYPES)
@pytest.mark.parametrize("shape", KL_PRIMITIVE_SHAPES)
def test_kl_loss_and_grad_correctness(backend, shape, dtype, with_labels):
    """Primitive parity against the PyTorch reference, per-row loss + dx."""
    if backend == "__none__":
        pytest.skip("No kl_loss_and_grad backends registered in this environment")

    BT, V = shape
    device = "cuda"
    g = torch.Generator(device="cpu").manual_seed(0)

    logits_cpu = torch.randn(BT, V, dtype=torch.float32, generator=g)
    target_cpu = torch.softmax(torch.randn(BT, V, dtype=torch.float32, generator=g), dim=-1)

    logits = logits_cpu.to(device=device, dtype=dtype)
    target = target_cpu.to(device=device, dtype=dtype)

    shift_labels = None
    if with_labels:
        # Two ignored rows exercise the kernel's early-return path.
        labels_cpu = torch.zeros(BT, dtype=torch.long)
        labels_cpu[:2] = -100
        shift_labels = labels_cpu.to(device)

    ignore_index = -100
    temperature = 1.7
    eps = 1e-10
    scale = 1.0 / BT  # batchmean-style
    tols = _TOLERANCES[dtype]

    # The primitive overwrites its logits input in place with dx, so clone for
    # the reference before dispatching.
    logits_ref = logits.clone()

    loss_rows, dx = dispatch(
        "kl_loss_and_grad",
        logits,
        target,
        shift_labels,
        ignore_index,
        temperature,
        eps,
        scale,
        backend=backend,
    )
    loss_rows_ref, dx_ref = _kl_loss_and_grad_ref(
        logits_ref, target, shift_labels, ignore_index, temperature, eps, scale
    )

    torch.testing.assert_close(
        loss_rows,
        loss_rows_ref,
        atol=tols["atol"],
        rtol=tols["rtol"],
        msg=lambda m: f"[kl_loss_and_grad/{backend} shape={shape} dtype={dtype} labels={with_labels}] loss: {m}",
    )
    torch.testing.assert_close(
        dx.float(),
        dx_ref,
        atol=tols["atol"],
        rtol=tols["rtol"],
        msg=lambda m: f"[kl_loss_and_grad/{backend} shape={shape} dtype={dtype} labels={with_labels}] dx: {m}",
    )


def test_kl_loss_and_grad_available_backends_includes_triton():
    """Sanity: the Triton implementation should always be available."""
    impls = available_backends("kl_loss_and_grad")
    assert any(b in ("triton", "nvidia-triton") for b in impls), f"expected 'triton' / 'nvidia-triton' in {impls}"


@pytest.mark.parametrize("V", [257, 4096, 65537])
@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("offset", [20.0, 100.0, -100.0, 10000.0])
def test_kl_loss_and_grad_common_logit_offset(V, temperature, offset):
    # Dyadic logits and temperatures make each translation exact in fp32:
    # this isolates loss cancellation from input rounding / GEMM precision (#1539).
    index = torch.arange(V, device="cuda")
    logits = ((index % 65).float() - 32.0).unsqueeze(0) / 8.0
    target = torch.softmax(logits.double() / temperature + 0.03 * index.double().sin(), dim=-1).float()
    q = target.double()
    expected = (q * (q.log() - (logits.double() / temperature).log_softmax(-1))).sum(-1)

    loss, dx = kl_loss_and_grad_triton(logits.clone(), target, None, -100, temperature, 1e-10, 1.0)
    shifted_loss, shifted_dx = kl_loss_and_grad_triton(logits + offset, target, None, -100, temperature, 1e-10, 1.0)

    torch.testing.assert_close(loss.double(), expected, atol=1e-7, rtol=5e-3)
    torch.testing.assert_close(shifted_loss.double(), expected, atol=1e-7, rtol=5e-3)
    torch.testing.assert_close(shifted_loss, loss, atol=0.0, rtol=0.0)
    torch.testing.assert_close(shifted_dx, dx, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("temperature", [0.5, 2.0, -1.0])
@pytest.mark.parametrize("scale", [0.0, 0.125, 1.0])
def test_kl_loss_and_grad_shifted_target_semantics(temperature, scale):
    # Targets need not sum to one. Preserve raw target mass, zero targets,
    # epsilon clamping, ignored rows, and in-place gradients after centering.
    V = 257
    logits = ((torch.arange(4 * V, device="cuda") % 65).float().reshape(4, V) - 32.0) / 8.0 + 100.0
    target = torch.softmax(logits.double() / temperature, dim=-1).float()
    target *= torch.tensor([0.75, 1.0, 1.25, 0.0], device="cuda").unsqueeze(-1)
    target[:, ::7] = 0.0
    labels = torch.tensor([0, -100, 0, 0], device="cuda")
    eps = 1e-3

    x = logits.double().requires_grad_(True)
    q = target.double()
    expected = (q * (q.clamp_min(eps).log() - (x / temperature).log_softmax(-1))).sum(-1) * scale
    expected = expected.masked_fill(labels == -100, 0.0)
    expected.sum().backward()

    loss, dx = kl_loss_and_grad_triton(logits, target, labels, -100, temperature, eps, scale)

    assert dx is logits
    torch.testing.assert_close(loss.double(), expected, atol=5e-7, rtol=1e-5)
    torch.testing.assert_close(dx.double(), x.grad, atol=1e-7, rtol=1e-5)
    assert torch.count_nonzero(loss[labels == -100]) == 0
    assert torch.count_nonzero(dx[labels == -100]) == 0


def test_fused_linear_kl_default_path_dispatches_inner_kl(monkeypatch):
    """The composed op must route its per-chunk work through the dispatcher."""
    observed_ops = []
    real_dispatch = flkl_ops.dispatch

    def tracking_dispatch(op_name, *args, **kwargs):
        if op_name == "kl_loss_and_grad":
            observed_ops.append(op_name)
        return real_dispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(flkl_ops, "dispatch", tracking_dispatch)

    BT, H, V = 32, 64, 512
    student_input = torch.randn(BT, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    student_weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.softmax(torch.randn(BT, V, device="cuda", dtype=torch.float32), dim=-1).to(torch.bfloat16)

    flkl_ops.LigerFusedLinearKLDivFunction.apply(student_input, student_weight, target).backward()

    assert observed_ops, "fused_linear_kl_div must dispatch its inner kernel as 'kl_loss_and_grad'"
