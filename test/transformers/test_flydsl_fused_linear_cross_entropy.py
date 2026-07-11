"""Parity checks for the optional FlyDSL fused-linear cross-entropy backend.

Scaffolding mirrors ``test_cutedsl_cross_entropy.py``; uses ``test.utils`` helpers.
"""

import os
import subprocess
import sys
import textwrap

from pathlib import Path

import pytest
import torch

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

_TOL = {
    torch.float32: (1e-5, 5e-4),
    torch.bfloat16: (5e-3, 5e-2),
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

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="flydsl FLCE requires a GPU")
hip_preferred = pytest.mark.skipif(
    torch.cuda.is_available() and not getattr(torch.version, "hip", None),
    reason="flydsl targets ROCm/AMD (torch.version.hip)",
)


# =============================================================================
# Imports (skip — don't fail — when the optional backend isn't installed)
# =============================================================================
def _flydsl_flce_mod():
    try:
        import liger_kernel.ops.flydsl.ops.fused_linear_cross_entropy as mod
    except ImportError as exc:
        pytest.skip(f"flydsl backend not importable: {exc}")
    return mod


def _flydsl_flce():
    return _flydsl_flce_mod().LigerFusedLinearCrossEntropyFunction


def _triton_flce():
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

    return LigerFusedLinearCrossEntropyFunction


# =============================================================================
# Helpers
# =============================================================================
def _apply(fn, _input, weight, target, bias=None, **opts):
    return fn.apply(
        _input,
        weight,
        target,
        bias,
        opts.get("ce_weight"),
        opts.get("ignore_index", -100),
        opts.get("lse_square_scale", 0.0),
        opts.get("label_smoothing", 0.0),
        opts.get("reduction", "mean"),
        opts.get("softcap"),
        opts.get("return_z_loss", False),
        opts.get("accum_dtype"),
        opts.get("use_token_scaling", False),
        opts.get("return_token_accuracy", False),
        opts.get("return_predicted_tokens", False),
    )


def _run_or_skip(thunk):
    try:
        return thunk()
    except NotImplementedError as exc:
        pytest.skip(f"flydsl FLCE branch not implemented yet: {exc}")


def _run(fn, base_h, weight, target, dtype, *, bias=None, requires_grad=True, grad_scale=None, **opts):
    x = base_h.to(dtype).clone().detach().requires_grad_(requires_grad)
    w = weight.to(dtype).clone().detach().requires_grad_(requires_grad)
    b = bias.to(dtype).clone().detach().requires_grad_(requires_grad) if bias is not None else None
    loss, z_loss, token_acc, pred = _apply(fn, x, w, target, b, **opts)
    gi = gw = gb = None
    if requires_grad:
        if opts.get("reduction") == "none":
            loss.backward(torch.ones_like(loss))
        elif grad_scale is not None:
            (loss * grad_scale).backward()
        else:
            loss.backward()
        gi = x.grad.detach().float()
        gw = w.grad.detach().float()
        gb = None if b is None else b.grad.detach().float()
    return (
        loss.detach().float(),
        None if z_loss is None else z_loss.detach().float(),
        None if token_acc is None else token_acc.detach().float(),
        None if pred is None else pred.detach(),
        gi,
        gw,
        gb,
    )


def _assert_close(out, ref, atol, rtol, what):
    if out is None and ref is None:
        return
    assert out is not None and ref is not None, (
        f"{what}: one side is None (out={out is not None}, ref={ref is not None})"
    )
    assert_verbose_allclose(out, ref, atol=atol, rtol=rtol, extra_info=f"[{what}]")


def _assert_flce_parity(base_h, weight, target, dtype, *, bias=None, check_grads=True, **opts):
    atol, rtol = _TOL[dtype]
    ref = _run(_triton_flce(), base_h, weight, target, dtype, bias=bias, **opts)
    out = _run_or_skip(lambda: _run(_flydsl_flce(), base_h, weight, target, dtype, bias=bias, **opts))
    _assert_close(out[0], ref[0], atol, rtol, "loss")
    if opts.get("return_z_loss"):
        _assert_close(out[1], ref[1], atol, rtol, "z_loss")
    if opts.get("return_token_accuracy"):
        _assert_close(out[2], ref[2], atol, rtol, "token_accuracy")
    if opts.get("return_predicted_tokens"):
        assert torch.equal(out[3], ref[3]), "predicted_tokens mismatch vs Triton"
    if check_grads and opts.get("requires_grad", True):
        _assert_close(out[4], ref[4], atol, rtol, "grad_input")
        _assert_close(out[5], ref[5], atol, rtol, "grad_weight")
        if bias is not None:
            _assert_close(out[6], ref[6], atol, rtol, "grad_bias")


def _basic_inputs(BT, H, V, *, bias=False):
    base = torch.randn(BT, H, device="cuda", dtype=torch.float32)
    weight = torch.randn(V, H, device="cuda", dtype=torch.float32)
    b = torch.randn(V, device="cuda", dtype=torch.float32) if bias else None
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    return base, weight, target, b


def _scatter_ignored(target, frac, ignore_index=-100):
    BT = target.shape[0]
    k = max(1, int(BT * frac))
    target[torch.randperm(BT, device=target.device)[:k]] = ignore_index
    return target


# =============================================================================
# A. Parity vs Triton
# =============================================================================
@cuda_required
@hip_preferred
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("bias", [False, True], ids=["no_bias", "bias"])
def test_flce_core_matches_triton(dtype, reduction, bias):
    set_seed()
    base, weight, target, b = _basic_inputs(128, 64, 512, bias=bias)
    _assert_flce_parity(base, weight, target, dtype, bias=b, reduction=reduction)


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
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_flce_ignore_index_matches_triton(reduction, dtype):
    set_seed()
    ignore_index = -100
    base, weight, target, _ = _basic_inputs(64, 32, 256)
    _scatter_ignored(target, frac=1 / 3, ignore_index=ignore_index)
    _assert_flce_parity(base, weight, target, dtype, reduction=reduction, ignore_index=ignore_index)


@cuda_required
@hip_preferred
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_flce_all_ignored_matches_triton(reduction):
    set_seed()
    ignore_index = -100
    base, weight, _, _ = _basic_inputs(16, 32, 128)
    target = torch.full((16,), ignore_index, device="cuda", dtype=torch.long)
    out = _run_or_skip(
        lambda: _run(
            _flydsl_flce(), base, weight, target, torch.float32, reduction=reduction, ignore_index=ignore_index
        )
    )
    ref = _run(_triton_flce(), base, weight, target, torch.float32, reduction=reduction, ignore_index=ignore_index)
    _assert_close(out[0], ref[0], *_TOL[torch.float32], "loss(all-ignored)")
    assert torch.equal(out[0], torch.zeros_like(out[0])), "all-ignored loss must be exactly 0"
    assert out[4] is not None and out[4].abs().max().item() == 0.0
    assert out[5] is not None and out[5].abs().max().item() == 0.0


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
@pytest.mark.parametrize("return_z_loss", [True, False], ids=["zret", "znoret"])
def test_flce_z_loss_matches_triton(return_z_loss, dtype):
    set_seed()
    base, weight, target, _ = _basic_inputs(64, 32, 256)
    _assert_flce_parity(base, weight, target, dtype, lse_square_scale=1e-4, return_z_loss=return_z_loss)


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
def test_flce_softcap_matches_triton(dtype):
    set_seed()
    base, weight, target, _ = _basic_inputs(64, 32, 256)
    _assert_flce_parity(base, weight, target, dtype, softcap=30.0)


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
def test_flce_label_smoothing_matches_triton(dtype):
    set_seed()
    base, weight, target, _ = _basic_inputs(64, 32, 256)
    _assert_flce_parity(base, weight, target, dtype, label_smoothing=0.1)


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
def test_flce_combined_features_matches_triton(dtype):
    set_seed()
    ignore_index = -100
    base, weight, target, _ = _basic_inputs(64, 32, 256)
    _scatter_ignored(target, frac=0.25, ignore_index=ignore_index)
    _assert_flce_parity(
        base,
        weight,
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
def test_flce_forward_only_matches_triton(dtype):
    set_seed()
    base, weight, target, _ = _basic_inputs(64, 32, 256)
    _assert_flce_parity(base, weight, target, dtype, requires_grad=False, check_grads=False)


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
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_flce_not_last_layer_grad_matches_triton(reduction, dtype):
    set_seed()
    base, weight, target, _ = _basic_inputs(64, 32, 256)
    _assert_flce_parity(base, weight, target, dtype, reduction=reduction, grad_scale=2.0)


# =============================================================================
# B. Chunking / memory invariants
# =============================================================================
@cuda_required
@hip_preferred
def test_flce_chunk_size_aspect_ratio():
    compute_flce_chunk_size = _flydsl_flce_mod().compute_flce_chunk_size
    BT, H, V = 4096, 128, 32000
    assert compute_flce_chunk_size(BT, H, V) == 32
    assert compute_flce_chunk_size(BT, H, V) * V < BT * V


@cuda_required
@hip_preferred
def test_flce_never_allocates_full_logits():
    set_seed()
    BT, H, V = 2048, 64, 8192
    chunk = _flydsl_flce_mod().compute_flce_chunk_size(BT, H, V)
    assert chunk < BT
    base, weight, target, _ = _basic_inputs(BT, H, V)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = _apply(_flydsl_flce(), base.requires_grad_(True), weight.requires_grad_(True), target)
    out[0].backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    full_logits_bytes = BT * V * 4
    delta = peak - before
    assert delta < full_logits_bytes, f"peak delta {delta} >= full logits {full_logits_bytes} (chunk={chunk})"


@pytest.mark.skipif(
    not hasattr(torch.nn.functional, "linear_cross_entropy"),
    reason="torch.nn.functional.linear_cross_entropy not available",
)
@cuda_required
@hip_preferred
def test_flce_vs_pytorch_linear_cross_entropy():
    from torch.nn.functional import linear_cross_entropy
    from torch.nn.modules.linear_cross_entropy_options import LinearCrossEntropyOptions

    set_seed()
    BT, H, V = 64, 32, 256
    base, weight, target, _ = _basic_inputs(BT, H, V)
    x = base.clone().requires_grad_(True)
    w = weight.clone().requires_grad_(True)
    opts = LinearCrossEntropyOptions(chunking_method="aspect_ratio")
    pt_loss = linear_cross_entropy(x, w, target, reduction="mean", ignore_index=-100, options=opts)
    fly_loss = _apply(
        _flydsl_flce(), x.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True), target
    )[0]
    _assert_close(fly_loss.float(), pt_loss.float(), 1e-4, 1e-4, "loss vs F.linear_cross_entropy")


# =============================================================================
# C. Unsupported FlyDSL branches
# =============================================================================
@cuda_required
@hip_preferred
def test_flce_rejects_unsupported_weight():
    set_seed()
    base, weight, target, _ = _basic_inputs(4, 16, 32)
    with pytest.raises(NotImplementedError):
        _apply(
            _flydsl_flce(),
            base.requires_grad_(True),
            weight.requires_grad_(True),
            target,
            ce_weight=torch.ones(32, device="cuda"),
        )


# =============================================================================
# D. Selection via LIGER_KERNEL_IMPL
# =============================================================================
@pytest.mark.skipif(
    os.environ.get("LIGER_KERNEL_IMPL", "").strip().lower() != "flydsl",
    reason="flydsl selection test requires LIGER_KERNEL_IMPL=flydsl",
)
def test_liger_kernel_impl_flydsl_selects_flce():
    repo = Path(__file__).resolve().parents[2]
    pythonpath = os.pathsep.join(filter(None, [str(repo / "src"), os.environ.get("PYTHONPATH", "")]))
    env = {**os.environ, "LIGER_KERNEL_IMPL": "flydsl", "PYTHONPATH": pythonpath}
    script = textwrap.dedent(
        """
        from liger_kernel.ops import LigerFusedLinearCrossEntropyFunction
        mod = LigerFusedLinearCrossEntropyFunction.__module__
        assert mod.startswith("liger_kernel.ops.flydsl."), mod
        print(mod)
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().startswith("liger_kernel.ops.flydsl.")
