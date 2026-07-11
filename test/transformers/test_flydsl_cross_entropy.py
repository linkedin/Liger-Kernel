"""Parity checks for the optional FlyDSL cross-entropy backend.

Scaffolding mirrors ``test_cutedsl_cross_entropy.py``; uses ``test.utils`` helpers.
"""

import os
import subprocess
import sys
import textwrap

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

_TOL = {
    torch.float32: (1e-5, 1e-5),
    torch.bfloat16: (5e-5, 5e-2),
    torch.float16: (5e-3, 5e-2),
}
_DTYPES = [
    pytest.param(
        torch.bfloat16,
        marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this device"),
    ),
    torch.float16,
    torch.float32,
]
_DTYPE_IDS = ["bf16", "fp16", "fp32"]

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="flydsl CE requires a GPU")
hip_preferred = pytest.mark.skipif(
    torch.cuda.is_available() and not getattr(torch.version, "hip", None),
    reason="flydsl targets ROCm/AMD (torch.version.hip)",
)


# =============================================================================
# Imports (skip — don't fail — when the optional backend isn't installed)
# =============================================================================
def _flydsl_ce():
    try:
        from liger_kernel.ops.flydsl.ops.cross_entropy import LigerCrossEntropyFunction
    except ImportError as exc:
        pytest.skip(f"flydsl backend not importable: {exc}")
    return LigerCrossEntropyFunction


def _triton_ce():
    from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction

    return LigerCrossEntropyFunction


# =============================================================================
# Helpers
# =============================================================================
def _apply(
    fn,
    _input,
    target,
    *,
    weight=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
    return_z_loss=False,
    return_token_accuracy=False,
    return_predicted_tokens=False,
):
    return fn.apply(
        _input,
        target,
        weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
        return_token_accuracy,
        return_predicted_tokens,
    )


def _run_or_skip(thunk):
    try:
        return thunk()
    except NotImplementedError as exc:
        pytest.skip(f"flydsl CE branch not implemented yet: {exc}")


def _run(
    fn,
    base_logits,
    target,
    dtype,
    *,
    reduction="mean",
    ignore_index=-100,
    lse_square_scale=0.0,
    softcap=None,
    return_z_loss=False,
    weight=None,
    label_smoothing=0.0,
    return_token_accuracy=False,
    return_predicted_tokens=False,
    requires_grad=True,
    grad_vec=None,
    grad_scale=None,
):
    _input = base_logits.clone().detach().to(dtype).requires_grad_(requires_grad)
    loss, z_loss, token_acc, pred = _apply(
        fn,
        _input,
        target,
        weight=weight,
        ignore_index=ignore_index,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        reduction=reduction,
        softcap=softcap,
        return_z_loss=return_z_loss,
        return_token_accuracy=return_token_accuracy,
        return_predicted_tokens=return_predicted_tokens,
    )
    grad = None
    if requires_grad:
        if reduction == "none":
            go = torch.ones_like(loss) if grad_vec is None else grad_vec.to(loss.dtype)
            loss.backward(gradient=go)
        elif grad_scale is not None:
            (loss * grad_scale).backward()
        else:
            loss.backward()
        grad = _input.grad.detach().float()
    return (
        loss.detach().float(),
        None if z_loss is None else z_loss.detach().float(),
        grad,
        None if token_acc is None else token_acc.detach().float(),
        None if pred is None else pred.detach(),
    )


def _assert_close(out, ref, atol, rtol, what):
    if out is None and ref is None:
        return
    assert out is not None and ref is not None, (
        f"{what}: one side is None (out={out is not None}, ref={ref is not None})"
    )
    assert_verbose_allclose(out, ref, atol=atol, rtol=rtol, extra_info=f"[{what}]")


def _assert_ce_parity(base_logits, target, dtype, *, check_grad=True, **opts):
    atol, rtol = _TOL[dtype]
    ref = _run(_triton_ce(), base_logits, target, dtype, **opts)
    out = _run_or_skip(lambda: _run(_flydsl_ce(), base_logits, target, dtype, **opts))
    _assert_close(out[0], ref[0], atol, rtol, "loss")
    if opts.get("return_z_loss"):
        _assert_close(out[1], ref[1], atol, rtol, "z_loss")
    if check_grad and opts.get("requires_grad", True):
        _assert_close(out[2], ref[2], atol, rtol, "grad")
    if opts.get("return_token_accuracy"):
        _assert_close(out[3], ref[3], atol, rtol, "token_accuracy")
    if opts.get("return_predicted_tokens"):
        assert torch.equal(out[4], ref[4]), "predicted_tokens mismatch vs Triton (exact int match expected)"


def _basic_inputs(BT=64, V=1024):
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    return base, target


def _scatter_ignored(target, frac, ignore_index=-100):
    BT = target.shape[0]
    k = max(1, int(BT * frac))
    target[torch.randperm(BT, device=target.device)[:k]] = ignore_index
    return target


# =============================================================================
# A. Parity vs Triton — implemented branches (auto-skip while a feature is a stub)
# =============================================================================
@cuda_required
@hip_preferred
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("BT, V", [(4, 1024), (128, 1024)], ids=["bt4", "bt128"])
def test_ce_core_matches_triton(BT, V, reduction, dtype):
    set_seed()
    base, target = _basic_inputs(BT, V)
    _assert_ce_parity(base, target, dtype, reduction=reduction)


@cuda_required
@hip_preferred
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ce_core_matches_torch_reference(reduction):
    set_seed()
    BT, V = 64, 1024
    base, target = _basic_inputs(BT, V)
    out = _run_or_skip(
        lambda: _run(_flydsl_ce(), base, target, torch.float32, reduction=reduction, requires_grad=False)
    )
    ref = F.cross_entropy(base, target, reduction=reduction).detach().float()
    _assert_close(out[0], ref, 1e-3, 1e-3, f"loss vs torch ({reduction})")


@cuda_required
@hip_preferred
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ce_ignore_index_matches_triton(reduction, dtype):
    set_seed()
    BT, V, ignore_index = 128, 1024, -100
    base, target = _basic_inputs(BT, V)
    _scatter_ignored(target, frac=1 / 3, ignore_index=ignore_index)
    _assert_ce_parity(base, target, dtype, reduction=reduction, ignore_index=ignore_index)


@cuda_required
@hip_preferred
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_ce_all_ignored_matches_triton(reduction):
    set_seed()
    BT, V, ignore_index = 64, 1024, -100
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.full((BT,), ignore_index, device="cuda", dtype=torch.long)
    out = _run_or_skip(
        lambda: _run(_flydsl_ce(), base, target, torch.float32, reduction=reduction, ignore_index=ignore_index)
    )
    ref = _run(_triton_ce(), base, target, torch.float32, reduction=reduction, ignore_index=ignore_index)
    _assert_close(out[0], ref[0], *_TOL[torch.float32], "loss(all-ignored)")
    assert torch.equal(out[0], torch.zeros_like(out[0])), "all-ignored loss must be exactly 0"
    assert out[2] is not None and out[2].abs().max().item() == 0.0, "all-ignored grad must be exactly 0"


@cuda_required
@hip_preferred
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("return_z_loss", [True, False], ids=["zret", "znoret"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ce_z_loss_matches_triton(reduction, return_z_loss, dtype):
    set_seed()
    base, target = _basic_inputs(128, 1024)
    _assert_ce_parity(base, target, dtype, reduction=reduction, lse_square_scale=1e-4, return_z_loss=return_z_loss)


@cuda_required
@hip_preferred
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ce_softcap_matches_triton(reduction, dtype):
    set_seed()
    base, target = _basic_inputs(128, 1024)
    _assert_ce_parity(base, target, dtype, reduction=reduction, softcap=30.0)


@cuda_required
@hip_preferred
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this device"),
        ),
        torch.float32,
    ],
    ids=["bf16", "fp32"],
)
def test_ce_combined_features_matches_triton(dtype):
    set_seed()
    BT, V, ignore_index = 128, 1024, -100
    base, target = _basic_inputs(BT, V)
    _scatter_ignored(target, frac=0.25, ignore_index=ignore_index)
    _assert_ce_parity(
        base,
        target,
        dtype,
        reduction="mean",
        ignore_index=ignore_index,
        lse_square_scale=1e-4,
        softcap=30.0,
        return_z_loss=True,
    )


@cuda_required
@hip_preferred
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_ce_forward_only_matches_triton(dtype):
    set_seed()
    base, target = _basic_inputs(64, 1024)
    _assert_ce_parity(base, target, dtype, reduction="mean", requires_grad=False, check_grad=False)


@cuda_required
@hip_preferred
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_ce_not_last_layer_grad_matches_triton(reduction, dtype):
    set_seed()
    base, target = _basic_inputs(64, 1024)
    _assert_ce_parity(base, target, dtype, reduction=reduction, grad_scale=2.0)


@cuda_required
@hip_preferred
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_ce_reduction_none_perrow_grad_matches_triton(dtype):
    set_seed()
    BT, V = 64, 1024
    base, target = _basic_inputs(BT, V)
    grad_vec = torch.rand(BT, device="cuda", dtype=torch.float32)
    _assert_ce_parity(base, target, dtype, reduction="none", grad_vec=grad_vec)


@cuda_required
@hip_preferred
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this device"),
        ),
        torch.float32,
    ],
    ids=["bf16", "fp32"],
)
def test_ce_noncontiguous_input_matches_triton(dtype):
    set_seed()
    BT, V = 64, 1024
    base = torch.randn(V, BT, device="cuda", dtype=torch.float32).t()
    assert not base.is_contiguous()
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    _assert_ce_parity(base, target, dtype, reduction="mean", requires_grad=False, check_grad=False)


@cuda_required
@hip_preferred
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ce_label_smoothing_matches_triton(reduction, dtype):
    set_seed()
    base, target = _basic_inputs(64, 1024)
    _assert_ce_parity(base, target, dtype, reduction=reduction, label_smoothing=0.1)


# =============================================================================
# B. Unsupported FlyDSL branches — must raise NotImplementedError (not silently diverge)
# =============================================================================
@cuda_required
@hip_preferred
def test_ce_rejects_unsupported_weight():
    set_seed()
    base, target = _basic_inputs(4, 64)
    x = base.clone().requires_grad_(True)
    w = torch.ones(64, device="cuda")
    with pytest.raises(NotImplementedError, match="class weights"):
        _apply(_flydsl_ce(), x, target, weight=w)


@cuda_required
@hip_preferred
def test_ce_rejects_unsupported_token_metrics():
    set_seed()
    base, target = _basic_inputs(4, 64)
    x = base.clone().requires_grad_(True)
    with pytest.raises(NotImplementedError):
        _apply(_flydsl_ce(), x, target, return_token_accuracy=True)


# =============================================================================
# C. Selection via LIGER_KERNEL_IMPL
# =============================================================================
@pytest.mark.skipif(
    os.environ.get("LIGER_KERNEL_IMPL", "").strip().lower() != "flydsl",
    reason="flydsl selection test requires LIGER_KERNEL_IMPL=flydsl",
)
def test_liger_kernel_impl_flydsl_selects_flydsl_ce():
    repo = Path(__file__).resolve().parents[2]
    pythonpath = os.pathsep.join(filter(None, [str(repo / "src"), os.environ.get("PYTHONPATH", "")]))
    env = {**os.environ, "LIGER_KERNEL_IMPL": "flydsl", "PYTHONPATH": pythonpath}
    script = textwrap.dedent(
        """
        from liger_kernel.ops import LigerCrossEntropyFunction
        mod = LigerCrossEntropyFunction.__module__
        assert mod.startswith("liger_kernel.ops.flydsl."), mod
        print(mod)
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().startswith("liger_kernel.ops.flydsl.")
