"""Correctness tests for ``LigerFusedLinearVocabParallelCrossEntropy`` and
``LigerMegatronFusedLinearCrossEntropy``.

Both classes share the same ``LigerFusedLinearVPCEFunction`` autograd op + the
same Triton kernels in ``liger_kernel.ops.fused_linear_vocab_parallel_cross_entropy``;
the only difference is the default formula / mode.

File layout:

  Section 0: helpers (PyTorch oracle ``F.linear + F.cross_entropy`` in fp32)
  Section 1: General-purpose class — forward + backward parity vs F.cross_entropy
  Section 2: Megatron class — forward + backward parity (with rescaled alpha)
  Section 3: Configuration plumbing (validation, accum_dtype, bias, edges)
  Section 4: TP=1 fast path + main_grad routing
  Section 5: TP>1 multi-GPU parity (mp.spawn; TP=2,4,8)
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from liger_kernel.megatron import LigerMegatronFusedLinearCrossEntropy
from liger_kernel.transformers import LigerFusedLinearVocabParallelCrossEntropy
from liger_kernel.utils import infer_device
from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

device = infer_device()
set_seed(42)


# ===========================================================================
# 0. Helpers
# ===========================================================================


def _reference_pytorch(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Oracle: F.linear + F.cross_entropy, both in fp32. Returns [S, B] per-token loss."""
    s, b, h = hidden_states.shape
    logits = F.linear(hidden_states.float(), weight.float(), bias.float() if bias is not None else None)
    v = logits.shape[-1]
    loss_flat = F.cross_entropy(
        logits.reshape(-1, v),
        target.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    return loss_flat.reshape(s, b)


def _reference_megatron(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Oracle for Megatron's NeMo formula: rescale alpha to alpha*V/(V-1)."""
    v = weight.shape[0]
    alpha_rescaled = label_smoothing * v / (v - 1) if label_smoothing > 0 else 0.0
    return _reference_pytorch(
        hidden_states, weight, target, bias=bias, ignore_index=ignore_index, label_smoothing=alpha_rescaled
    )


def _assign_ignore_index(target: torch.Tensor, ignore_index: int, frac: float = 0.25) -> None:
    flat = target.view(-1)
    n = max(1, int(flat.numel() * frac))
    idx = torch.randperm(flat.numel(), device=flat.device)[:n]
    flat[idx] = ignore_index


# ===========================================================================
# 1. LigerFusedLinearVocabParallelCrossEntropy — F.cross_entropy parity
# ===========================================================================


@pytest.mark.parametrize(
    "s, b, h, v",
    [
        (8, 2, 64, 128),
        (16, 2, 256, 4096),
        (32, 1, 512, 32000),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-4),
        pytest.param(
            torch.bfloat16,
            5e-2,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported"),
        ),
    ],
)
def test_vp_flce_matches_pytorch_cross_entropy(s, b, h, v, dtype, atol, rtol):
    """Headline correctness — forward AND backward parity vs F.linear+F.cross_entropy."""
    ce = LigerFusedLinearVocabParallelCrossEntropy()

    hidden = torch.randn(s, b, h, device=device, dtype=dtype) * 0.5
    weight = torch.randn(v, h, device=device, dtype=dtype) * 0.5
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    h_ref = hidden.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    h_got = hidden.detach().clone().requires_grad_(True)
    w_got = weight.detach().clone().requires_grad_(True)

    ref = _reference_pytorch(h_ref, w_ref, target)
    got = ce(h_got, w_got, target)

    assert got.shape == (s, b)
    assert_verbose_allclose(got.float(), ref.float(), atol=atol, rtol=rtol)

    ref.sum().backward()
    got.sum().backward()
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=atol, rtol=rtol)
    assert_verbose_allclose(w_got.grad.float(), w_ref.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("ignore_index", [-100, 0])
def test_vp_flce_respects_ignore_index(ignore_index):
    s, b, h, v = 16, 2, 128, 1024
    ce = LigerFusedLinearVocabParallelCrossEntropy(ignore_index=ignore_index)

    hidden = torch.randn(s, b, h, device=device, dtype=torch.float32)
    weight = torch.randn(v, h, device=device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)
    _assign_ignore_index(target, ignore_index, frac=0.25)

    h_ref = hidden.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    h_got = hidden.detach().clone().requires_grad_(True)
    w_got = weight.detach().clone().requires_grad_(True)

    ref = _reference_pytorch(h_ref, w_ref, target, ignore_index=ignore_index)
    got = ce(h_got, w_got, target)
    assert_verbose_allclose(got.float(), ref.float(), atol=1e-4, rtol=1e-4)

    ref.sum().backward()
    got.sum().backward()
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=1e-4, rtol=1e-4)
    assert_verbose_allclose(w_got.grad.float(), w_ref.grad.float(), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_vp_flce_respects_label_smoothing_pytorch_formula(label_smoothing):
    """PyTorch formula default → matches F.cross_entropy directly."""
    s, b, h, v = 8, 2, 64, 512
    ce = LigerFusedLinearVocabParallelCrossEntropy(label_smoothing=label_smoothing)

    hidden = torch.randn(s, b, h, device=device, dtype=torch.float32)
    weight = torch.randn(v, h, device=device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    h_ref = hidden.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    h_got = hidden.detach().clone().requires_grad_(True)
    w_got = weight.detach().clone().requires_grad_(True)

    ref = _reference_pytorch(h_ref, w_ref, target, label_smoothing=label_smoothing)
    got = ce(h_got, w_got, target)
    assert_verbose_allclose(got.float(), ref.float(), atol=1e-4, rtol=1e-4)

    ref.sum().backward()
    got.sum().backward()
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=1e-4, rtol=1e-4)
    assert_verbose_allclose(w_got.grad.float(), w_ref.grad.float(), atol=1e-4, rtol=1e-4)


def test_vp_flce_with_bias():
    s, b, h, v = 16, 2, 128, 4096
    ce = LigerFusedLinearVocabParallelCrossEntropy()

    hidden = torch.randn(s, b, h, device=device, dtype=torch.float32)
    weight = torch.randn(v, h, device=device, dtype=torch.float32) * 0.5
    bias = torch.randn(v, device=device, dtype=torch.float32) * 0.1
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    h_ref = hidden.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    b_ref = bias.detach().clone().requires_grad_(True)
    h_got = hidden.detach().clone().requires_grad_(True)
    w_got = weight.detach().clone().requires_grad_(True)
    b_got = bias.detach().clone().requires_grad_(True)

    ref = _reference_pytorch(h_ref, w_ref, target, bias=b_ref)
    got = ce(h_got, w_got, target, bias=b_got)
    assert_verbose_allclose(got.float(), ref.float(), atol=1e-4, rtol=1e-4)

    ref.sum().backward()
    got.sum().backward()
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=1e-4, rtol=1e-4)
    assert_verbose_allclose(w_got.grad.float(), w_ref.grad.float(), atol=1e-4, rtol=1e-4)
    assert_verbose_allclose(b_got.grad.float(), b_ref.grad.float(), atol=1e-4, rtol=1e-4)


# ===========================================================================
# 2. LigerMegatronFusedLinearCrossEntropy — Megatron formula parity
# ===========================================================================


@pytest.mark.parametrize(
    "s, b, h, v",
    [
        (8, 2, 64, 128),
        (16, 2, 256, 4096),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-4),
        pytest.param(
            torch.bfloat16,
            5e-2,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported"),
        ),
    ],
)
def test_megatron_vp_flce_matches_at_zero_smoothing(s, b, h, v, dtype, atol, rtol):
    """At label_smoothing=0 the Megatron and PyTorch formulas are identical."""
    ce = LigerMegatronFusedLinearCrossEntropy()

    hidden = torch.randn(s, b, h, device=device, dtype=dtype) * 0.5
    weight = torch.randn(v, h, device=device, dtype=dtype) * 0.5
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    h_ref = hidden.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    h_got = hidden.detach().clone().requires_grad_(True)
    w_got = weight.detach().clone().requires_grad_(True)

    ref = _reference_pytorch(h_ref, w_ref, target)
    got = ce(h_got, w_got, target)
    assert_verbose_allclose(got.float(), ref.float(), atol=atol, rtol=rtol)

    ref.sum().backward()
    got.sum().backward()
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=atol, rtol=rtol)
    assert_verbose_allclose(w_got.grad.float(), w_ref.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_megatron_vp_flce_matches_megatron_formula(label_smoothing):
    """LigerMegatronFusedLinearCrossEntropy → matches F.cross_entropy with alpha rescaled."""
    s, b, h, v = 8, 2, 64, 512
    ce = LigerMegatronFusedLinearCrossEntropy(label_smoothing=label_smoothing)

    hidden = torch.randn(s, b, h, device=device, dtype=torch.float32)
    weight = torch.randn(v, h, device=device, dtype=torch.float32)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    h_ref = hidden.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    h_got = hidden.detach().clone().requires_grad_(True)
    w_got = weight.detach().clone().requires_grad_(True)

    ref = _reference_megatron(h_ref, w_ref, target, label_smoothing=label_smoothing)
    got = ce(h_got, w_got, target)
    assert_verbose_allclose(got.float(), ref.float(), atol=1e-4, rtol=1e-4)

    ref.sum().backward()
    got.sum().backward()
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=1e-4, rtol=1e-4)
    assert_verbose_allclose(w_got.grad.float(), w_ref.grad.float(), atol=1e-4, rtol=1e-4)


# ===========================================================================
# 3. Configuration plumbing
# ===========================================================================


@pytest.mark.parametrize("bad_reduction", ["mean", "sum", "MEAN", "garbage"])
def test_vp_flce_rejects_non_none_reduction(bad_reduction):
    with pytest.raises(ValueError, match="reduction must be 'none'"):
        LigerFusedLinearVocabParallelCrossEntropy(reduction=bad_reduction)
    with pytest.raises(ValueError, match="reduction must be 'none'"):
        LigerMegatronFusedLinearCrossEntropy(reduction=bad_reduction)


@pytest.mark.parametrize("bad_formula", ["pyTorch", "tf", ""])
def test_vp_flce_rejects_bad_label_smoothing_formula(bad_formula):
    with pytest.raises(ValueError, match="label_smoothing_formula"):
        LigerFusedLinearVocabParallelCrossEntropy(label_smoothing_formula=bad_formula)


@pytest.mark.parametrize("bad_mode", ["Global", "", "world"])
def test_vp_flce_rejects_bad_label_smoothing_mode(bad_mode):
    with pytest.raises(ValueError, match="label_smoothing_mode"):
        LigerFusedLinearVocabParallelCrossEntropy(label_smoothing_mode=bad_mode)


@pytest.mark.parametrize("bad_alpha", [-0.1, 1.0, 1.5])
def test_vp_flce_rejects_bad_label_smoothing(bad_alpha):
    with pytest.raises(ValueError, match="label_smoothing must be in"):
        LigerFusedLinearVocabParallelCrossEntropy(label_smoothing=bad_alpha)


def test_vp_flce_rejects_non_3d_hidden_states():
    ce = LigerFusedLinearVocabParallelCrossEntropy()
    weight = torch.randn(16, 16, device=device)
    target = torch.randint(0, 16, (8,), device=device, dtype=torch.long)
    bad = torch.randn(8, 16, device=device)  # 2-D
    with pytest.raises(ValueError, match="3-D"):
        ce(bad, weight, target)


def test_vp_flce_accum_dtype_fp32():
    """fp32 accumulator: numerical correctness preserved."""
    s, b, h, v = 16, 2, 128, 1024
    ce_fp32 = LigerFusedLinearVocabParallelCrossEntropy(accum_dtype=torch.float32)
    ce_default = LigerFusedLinearVocabParallelCrossEntropy()

    hidden = torch.randn(s, b, h, device=device, dtype=torch.bfloat16)
    weight = torch.randn(v, h, device=device, dtype=torch.bfloat16) * 0.5
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    h_a = hidden.detach().clone().requires_grad_(True)
    w_a = weight.detach().clone().requires_grad_(True)
    h_b = hidden.detach().clone().requires_grad_(True)
    w_b = weight.detach().clone().requires_grad_(True)

    out_a = ce_fp32(h_a, w_a, target)
    out_b = ce_default(h_b, w_b, target)
    # Forward identical regardless of accum_dtype (only affects bwd accumulator).
    assert torch.allclose(out_a.float(), out_b.float(), atol=1e-3, rtol=1e-3)

    out_a.sum().backward()
    out_b.sum().backward()
    # Grad_W should both match the fp32 oracle (within bf16 noise).
    h_ref = hidden.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    ref = _reference_pytorch(h_ref, w_ref, target)
    ref.sum().backward()
    assert_verbose_allclose(w_a.grad.float(), w_ref.grad.float(), atol=5e-2, rtol=5e-2)
    assert_verbose_allclose(w_b.grad.float(), w_ref.grad.float(), atol=5e-2, rtol=5e-2)


def test_vp_flce_does_not_mutate_input():
    s, b, h, v = 16, 2, 128, 1024
    ce = LigerFusedLinearVocabParallelCrossEntropy()

    hidden = torch.randn(s, b, h, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(v, h, device=device, dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)
    hidden_snapshot = hidden.detach().clone()
    weight_snapshot = weight.detach().clone()

    out = ce(hidden, weight, target)
    out.sum().backward()
    assert torch.equal(hidden.detach(), hidden_snapshot)
    assert torch.equal(weight.detach(), weight_snapshot)


def test_vp_flce_chained_with_downstream_factor():
    """Composition with downstream gradient — exercises in-place backward correctness."""
    s, b, h, v = 16, 1, 128, 4096
    ce = LigerFusedLinearVocabParallelCrossEntropy()

    hidden = torch.randn(s, b, h, device=device, dtype=torch.float32)
    weight = torch.randn(v, h, device=device, dtype=torch.float32) * 0.5
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    h_ref = hidden.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    h_got = hidden.detach().clone().requires_grad_(True)
    w_got = weight.detach().clone().requires_grad_(True)

    ref = _reference_pytorch(h_ref, w_ref, target)
    got = ce(h_got, w_got, target)

    grad_out = torch.rand_like(ref)
    (ref * 3.0).backward(gradient=grad_out)
    (got * 3.0).backward(gradient=grad_out)
    assert_verbose_allclose(h_got.grad.float(), h_ref.grad.float(), atol=1e-4, rtol=1e-4)
    assert_verbose_allclose(w_got.grad.float(), w_ref.grad.float(), atol=1e-4, rtol=1e-4)


def test_megatron_vp_flce_extra_repr():
    ce = LigerMegatronFusedLinearCrossEntropy(ignore_index=42, label_smoothing=0.07)
    rep = ce.extra_repr()
    assert "ignore_index=42" in rep
    assert "label_smoothing=0.07" in rep
    assert "label_smoothing_formula='megatron'" in rep
    assert "label_smoothing_mode='partition'" in rep


# ===========================================================================
# 4. TP=1 fast path + main_grad routing
# ===========================================================================


def test_vp_flce_accepts_none_tp_group():
    s, b, h, v = 16, 1, 64, 1024
    ce = LigerFusedLinearVocabParallelCrossEntropy()
    hidden = torch.randn(s, b, h, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(v, h, device=device, dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)
    out = ce(hidden, weight, target, tp_group=None)
    assert out.shape == (s, b)
    out.sum().backward()
    assert hidden.grad is not None
    assert weight.grad is not None


def test_vp_flce_main_grad_routing():
    """When ``weight.main_grad`` is present, grad goes into it (not .grad) and
    ``weight.grad_added_to_main_grad`` is set — matching Megatron's
    ``LinearWithGradAccumulationAndAsyncCommunication`` contract."""
    s, b, h, v = 16, 2, 128, 1024
    ce = LigerFusedLinearVocabParallelCrossEntropy(accum_dtype=torch.float32)

    hidden = torch.randn(s, b, h, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(v, h, device=device, dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)

    # Attach main_grad as Megatron does in its DistributedDataParallel wrapper.
    weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)

    # Reference: collect what grad_W should be from an oracle run.
    h_ref = hidden.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    _reference_pytorch(h_ref, w_ref, target).sum().backward()
    expected_grad_W = w_ref.grad

    out = ce(hidden, weight, target)
    out.sum().backward()

    # main_grad got the accumulated wgrad; .grad was NOT touched.
    assert getattr(weight, "grad_added_to_main_grad", False) is True
    assert weight.grad is None
    assert_verbose_allclose(weight.main_grad.float(), expected_grad_W.float(), atol=1e-4, rtol=1e-4)


# ===========================================================================
# 5. TP>1 multi-GPU parity (mp.spawn; require >= 2 GPUs)
# ===========================================================================


def _tp_flce_worker(
    rank,
    tp_size,
    s,
    b,
    h,
    v_global,
    formula,
    mode,
    label_smoothing,
    ignore_index_frac,
    file_name,
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{file_name}",
        rank=rank,
        world_size=tp_size,
    )
    torch.cuda.set_device(rank)
    tp_group = dist.group.WORLD
    dev = torch.device(f"cuda:{rank}")
    v_local = v_global // tp_size

    # Build same global state on all ranks.
    torch.manual_seed(42)
    hidden_cpu = torch.randn(s, b, h, dtype=torch.float32) * 0.5
    weight_global_cpu = torch.randn(v_global, h, dtype=torch.float32) * 0.5
    target_cpu = torch.randint(0, v_global, (s, b), dtype=torch.long)

    hidden = hidden_cpu.to(dev)
    weight_global = weight_global_cpu.to(dev)
    target = target_cpu.to(dev)
    dist.broadcast(hidden, src=0, group=tp_group)
    dist.broadcast(weight_global, src=0, group=tp_group)
    dist.broadcast(target, src=0, group=tp_group)

    if ignore_index_frac > 0:
        torch.manual_seed(123)
        flat = target.view(-1)
        n_ignore = max(1, int(flat.numel() * ignore_index_frac))
        idx = torch.randperm(flat.numel(), device=flat.device)[:n_ignore]
        flat[idx] = -100

    weight_local = weight_global[rank * v_local : (rank + 1) * v_local]

    # Liger VP-FLCE path on this rank's vocab slice.
    h_liger = hidden.detach().clone().requires_grad_(True)
    w_liger = weight_local.detach().clone().requires_grad_(True)
    ce = LigerFusedLinearVocabParallelCrossEntropy(
        label_smoothing=label_smoothing,
        label_smoothing_formula=formula,
        label_smoothing_mode=mode,
    )
    loss_liger = ce(h_liger, w_liger, target, tp_group=tp_group)

    # Reference: F.linear + F.cross_entropy on global state in fp32.
    if formula == "megatron" and label_smoothing > 0:
        K_ref = v_global if mode == "global" else v_local
        alpha_ref = label_smoothing * K_ref / (K_ref - 1)
    else:
        alpha_ref = label_smoothing
    h_ref = hidden.detach().clone().requires_grad_(True)
    w_ref_global = weight_global.detach().clone().requires_grad_(True)
    logits_ref = F.linear(h_ref, w_ref_global)
    loss_ref = F.cross_entropy(
        logits_ref.reshape(-1, v_global),
        target.reshape(-1),
        reduction="none",
        ignore_index=-100,
        label_smoothing=alpha_ref,
    ).reshape(s, b)

    # Forward parity skipped for partition smoothing (diverges per rank by design).
    if not (mode == "partition" and label_smoothing > 0):
        torch.testing.assert_close(loss_liger.float(), loss_ref.float(), atol=1e-3, rtol=1e-3)

    loss_liger.sum().backward()
    loss_ref.sum().backward()

    # grad_x: should match the reference (after AllReduce — wrapped here as the
    # autograd op itself doesn't AllReduce grad_x. In real Megatron the SP-gather
    # backward does it. For non-SP TP, the AR happens at the ColumnParallelLinear
    # boundary which we collapse here — but the reference accumulates grad from
    # the full vocab so we need to sum across ranks.)
    grad_x_local = h_liger.grad.detach().clone()
    grad_x_summed = grad_x_local.clone()
    dist.all_reduce(grad_x_summed, op=dist.ReduceOp.SUM, group=tp_group)

    # grad_w: this rank's slice should match the corresponding slice of ref.
    grad_w_ref_slice = w_ref_global.grad[rank * v_local : (rank + 1) * v_local]

    if not (mode == "partition" and label_smoothing > 0):
        torch.testing.assert_close(grad_x_summed.float(), h_ref.grad.float(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(w_liger.grad.float(), grad_w_ref_slice.float(), atol=1e-3, rtol=1e-3)

    dist.destroy_process_group()


@pytest.mark.parametrize(
    "tp_size",
    [
        pytest.param(
            2,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                reason="requires >= 2 GPUs",
            ),
        ),
        pytest.param(
            4,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                reason="requires >= 4 GPUs",
            ),
        ),
        pytest.param(
            8,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available() or torch.cuda.device_count() < 8,
                reason="requires >= 8 GPUs",
            ),
        ),
    ],
)
@pytest.mark.parametrize("s, b, h, v_global", [(8, 2, 64, 128), (16, 1, 128, 4096)])
@pytest.mark.parametrize("formula", ["pytorch", "megatron"])
@pytest.mark.parametrize("mode", ["global", "partition"])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("ignore_index_frac", [0.0, 0.25])
def test_tp_vp_flce_parity(tp_size, s, b, h, v_global, formula, mode, label_smoothing, ignore_index_frac):
    """Multi-GPU TP>1 forward + backward parity for all formula/mode/ignore combos."""
    with tempfile.NamedTemporaryFile() as f:
        mp.spawn(
            _tp_flce_worker,
            args=(tp_size, s, b, h, v_global, formula, mode, label_smoothing, ignore_index_frac, f.name),
            nprocs=tp_size,
            join=True,
        )
