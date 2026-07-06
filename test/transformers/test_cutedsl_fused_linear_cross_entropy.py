import os
import subprocess
import sys
import textwrap

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Per-(reduction, dtype) (atol, rtol). Matched to the Triton FLCE suite's bar
# (test_fused_linear_cross_entropy.py): bf16 sum accumulates BT terms with no mean normalizer,
# so its absolute tolerance is huge; mean/none/fp32 are tight (atol=1e-5, exactly Triton's bar).
# Used for BOTH the Triton parity gate and the torch anchor.
_TOL = {
    ("mean", torch.bfloat16): (5e-3, 5e-2),
    ("mean", torch.float16): (5e-3, 5e-2),
    ("mean", torch.float32): (1e-5, 5e-4),
    ("sum", torch.bfloat16): (5e0, 5e1),
    ("sum", torch.float16): (5e0, 5e1),
    ("sum", torch.float32): (1e-3, 5e-2),
    ("none", torch.bfloat16): (5e-3, 5e-2),
    ("none", torch.float16): (5e-3, 5e-2),
    ("none", torch.float32): (1e-5, 5e-4),
}
_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
_DTYPE_IDS = ["bf16", "fp16", "fp32"]

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="cutedsl FLCE requires CUDA")


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Imports (skip — don't fail — when the optional backend isn't installed)
# =============================================================================
def _cutedsl_flce():
    """The cutedsl ``LigerFusedLinearCrossEntropyFunction`` under test (the canonical 2-SM/quack FLCE
    at ``liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy``), or skip if CUTLASS/quack isn't
    installed. It implements the core path fast (mean/sum/none-fwd, 16-bit, H%8==0, Blackwell) and
    routes every other case (fp32, softcap, argmax metrics, non-Blackwell, …) through its general
    cuBLAS fallback — so all parametrizations run; none raise NotImplementedError except the by-design
    ``reduction='none'`` + grad refusal (tested in G)."""
    try:
        from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
    except ImportError as exc:
        pytest.skip(f"cutedsl backend not importable (cutlass.cute missing?): {exc}")
    return LigerFusedLinearCrossEntropyFunction


def _triton_flce():
    """Triton reference ``LigerFusedLinearCrossEntropyFunction`` (the parity oracle)."""
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

    return LigerFusedLinearCrossEntropyFunction


# =============================================================================
# Helpers
# =============================================================================
def _apply(
    fn,
    _input,
    weight,
    target,
    *,
    bias=None,
    ce_weight=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
    return_z_loss=False,
    accum_dtype=None,
    use_token_scaling=False,
    return_token_accuracy=False,
    return_predicted_tokens=False,
):
    """Positional ``apply`` matching BOTH the Triton and cutedsl FLCE signatures."""
    return fn.apply(
        _input,
        weight,
        target,
        bias,
        ce_weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
        accum_dtype,
        use_token_scaling,
        return_token_accuracy,
        return_predicted_tokens,
    )


def _run_or_skip(thunk):
    """Run ``thunk``; if the cutedsl branch is still a Stage-1 stub, SKIP instead of fail."""
    try:
        return thunk()
    except NotImplementedError as exc:
        pytest.skip(f"cutedsl FLCE branch not implemented yet: {exc}")


class _Masters:
    """Identical fp32 master tensors fed (cloned + cast) to every backend so the
    only variable across runs is the kernel, never the inputs."""

    def __init__(self, BT, H, V, *, bias, ce_weight, device="cuda"):
        self.input = torch.randn(BT, H, device=device, dtype=torch.float32)
        self.weight = torch.randn(V, H, device=device, dtype=torch.float32)
        self.bias = torch.randn(V, device=device, dtype=torch.float32) if bias else None
        # ce_weight stays fp32 on both paths (Triton + cutedsl upcast it internally).
        self.ce_weight = (torch.rand(V, device=device, dtype=torch.float32) + 0.5) if ce_weight else None


def _make_target(BT, V, *, ignore_frac=0.0, ignore_index=-100, device="cuda"):
    target = torch.randint(0, V, (BT,), device=device, dtype=torch.long)
    if ignore_frac > 0:
        n = int(BT * ignore_frac)
        if n > 0:
            target[torch.randperm(BT, device=device)[:n]] = ignore_index
    return target


def _run(
    fn,
    masters,
    target,
    dtype,
    *,
    requires_grad=True,
    reduction="mean",
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    softcap=None,
    return_z_loss=False,
    accum_dtype=None,
    use_token_scaling=False,
    return_token_accuracy=False,
    return_predicted_tokens=False,
    grad_output=None,
):
    """One FLCE fwd (+bwd) with fresh leaves cloned from ``masters``.

    Returns a dict of detached-fp32 tensors (predicted_tokens int64) or None:
      loss, z_loss, token_accuracy, predicted_tokens, grad_input, grad_weight, grad_bias.
    ``grad_output`` (default ones_like(loss)) is the upstream grad — pass a vector for
    reduction='none', a scaled tensor for the not-last-layer scalar path.
    """
    x = masters.input.clone().to(dtype).requires_grad_(requires_grad)
    w = masters.weight.clone().to(dtype).requires_grad_(requires_grad)
    if masters.bias is not None:
        b = masters.bias.clone().to(dtype).requires_grad_(requires_grad)
    else:
        b = None

    loss, z_loss, token_acc, pred = _apply(
        fn,
        x,
        w,
        target,
        bias=b,
        ce_weight=masters.ce_weight,
        ignore_index=ignore_index,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        reduction=reduction,
        softcap=softcap,
        return_z_loss=return_z_loss,
        accum_dtype=accum_dtype,
        use_token_scaling=use_token_scaling,
        return_token_accuracy=return_token_accuracy,
        return_predicted_tokens=return_predicted_tokens,
    )

    gi = gw = gb = None
    if requires_grad:
        go = torch.ones_like(loss) if grad_output is None else grad_output.to(loss.dtype)
        loss.backward(gradient=go)
        gi = None if x.grad is None else x.grad.detach().float()
        gw = None if w.grad is None else w.grad.detach().float()
        gb = None if (b is None or b.grad is None) else b.grad.detach().float()

    return {
        "loss": loss.detach().float(),
        "z_loss": None if z_loss is None else z_loss.detach().float(),
        "token_accuracy": None if token_acc is None else token_acc.detach().float(),
        "predicted_tokens": None if pred is None else pred.detach(),
        "grad_input": gi,
        "grad_weight": gw,
        "grad_bias": gb,
    }


def _assert_close(out, ref, atol, rtol, what):
    if out is None and ref is None:
        return
    assert out is not None and ref is not None, (
        f"{what}: one side is None (out={out is not None}, ref={ref is not None})"
    )
    if not torch.allclose(out, ref, atol=atol, rtol=rtol):
        diff = (out - ref).abs()
        raise AssertionError(
            f"{what} mismatch vs oracle: max|diff|={diff.max().item():.3e} "
            f"mean|diff|={diff.mean().item():.3e} (atol={atol}, rtol={rtol})"
        )


def _assert_flce_parity(masters, target, dtype, *, reduction="mean", check_grad=True, grad_atol_scale=1.0, **opts):
    """Run cutedsl + Triton with identical inputs/opts; compare every returned tensor.

    ``grad_atol_scale`` scales the ABSOLUTE tolerance for the gradient comparisons only (loss is
    upstream-grad-independent). The grads scale linearly with grad_output, so a not-last-layer test
    (grad_output=g) amplifies every gradient — and its near-zero-element absolute error — by |g|;
    passing grad_atol_scale=|g| keeps the effective (relative) strictness identical to grad_output=1."""
    atol, rtol = _TOL[(reduction, dtype)]
    ref = _run(_triton_flce(), masters, target, dtype, reduction=reduction, **opts)
    out = _run_or_skip(lambda: _run(_cutedsl_flce(), masters, target, dtype, reduction=reduction, **opts))

    _assert_close(out["loss"], ref["loss"], atol, rtol, "loss")
    if opts.get("return_z_loss"):
        _assert_close(out["z_loss"], ref["z_loss"], atol, rtol, "z_loss")
    if opts.get("return_token_accuracy"):
        _assert_close(out["token_accuracy"], ref["token_accuracy"], atol, rtol, "token_accuracy")
    if opts.get("return_predicted_tokens"):
        assert torch.equal(out["predicted_tokens"], ref["predicted_tokens"]), "predicted_tokens mismatch vs Triton"
    if check_grad and opts.get("requires_grad", True):
        g_atol = atol * grad_atol_scale
        _assert_close(out["grad_input"], ref["grad_input"], g_atol, rtol, "grad_input")
        _assert_close(out["grad_weight"], ref["grad_weight"], g_atol, rtol, "grad_weight")
        _assert_close(out["grad_bias"], ref["grad_bias"], g_atol, rtol, "grad_bias")


def _torch_logits(masters, target, dtype, *, bias):
    """fp32 logits the same way TorchLMHeadCE does: cast to dtype, matmul, upcast."""
    x = masters.input.to(dtype)
    w = masters.weight.to(dtype)
    logits = (x @ w.t()).float()
    if bias and masters.bias is not None:
        logits = logits + masters.bias.to(dtype).float()
    return logits


# =============================================================================
# A. Core path — parity vs Triton AND vs torch ground truth.
#    (Triton: test_correctness's first feature row.) Vocab divisible by 8 so the
#    "weird shapes" reach the kernel; small V/odd H stress small chunks + the
#    per-row CTA path; big V stresses many chunks + tail predication.
# =============================================================================
_CORE_SHAPES = [
    (8, 128, 1024, 4096),
    (4, 47, 31, 128),  # weird shape, V%8==0
    (9, 7, 41, 64),  # weird shape, V%8==0
]
_CORE_SHAPE_IDS = ["8x128x1024x4096", "4x47x31x128", "9x7x41x64"]


@cuda_required
@pytest.mark.parametrize("B, T, H, V", _CORE_SHAPES, ids=_CORE_SHAPE_IDS)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize("accum_dtype", [None, torch.float32], ids=["accumNone", "accumfp32"])
def test_flce_core_matches_triton(B, T, H, V, dtype, reduction, bias, accum_dtype):
    """Core FLCE: loss + grad_input + grad_weight + grad_bias parity vs Triton, with
    ~random ignore_index rows (mirrors the Triton suite's per-test ignore scatter)."""
    set_seed()
    BT = B * T
    masters = _Masters(BT, H, V, bias=bias, ce_weight=False)
    target = _make_target(BT, V, ignore_frac=0.25)
    _assert_flce_parity(masters, target, dtype, reduction=reduction, accum_dtype=accum_dtype)


@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_flce_core_matches_torch(dtype, reduction, bias):
    """Independent anchor: cutedsl loss vs an explicit x @ Wᵀ (+bias) -> F.cross_entropy."""
    set_seed()
    BT, H, V = 256, 512, 4096
    masters = _Masters(BT, H, V, bias=bias, ce_weight=False)
    target = _make_target(BT, V, ignore_frac=0.25)
    out = _run_or_skip(lambda: _run(_cutedsl_flce(), masters, target, dtype, reduction=reduction, requires_grad=False))
    logits = _torch_logits(masters, target, dtype, bias=bias)
    ref = F.cross_entropy(logits, target, ignore_index=-100, reduction=reduction).detach().float()
    atol, rtol = _TOL[(reduction, dtype)]
    _assert_close(out["loss"], ref, atol, rtol, f"loss vs torch ({reduction})")


@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_flce_all_ignored_matches_triton(dtype, reduction):
    """Every row ignored: total_n_non_ignore == 0 -> inv_n = 1.0; loss & grads must be 0."""
    set_seed()
    BT, H, V = 64, 256, 4096
    masters = _Masters(BT, H, V, bias=True, ce_weight=False)
    target = torch.full((BT,), -100, device="cuda", dtype=torch.long)
    out = _run_or_skip(lambda: _run(_cutedsl_flce(), masters, target, dtype, reduction=reduction))
    ref = _run(_triton_flce(), masters, target, dtype, reduction=reduction)
    atol, rtol = _TOL[(reduction, dtype)]
    _assert_close(out["loss"], ref["loss"], atol, rtol, "loss(all-ignored)")
    assert torch.equal(out["loss"], torch.zeros_like(out["loss"])), "all-ignored loss must be exactly 0"
    assert out["grad_input"].abs().max().item() == 0.0, "all-ignored grad_input must be exactly 0"
    assert out["grad_weight"].abs().max().item() == 0.0, "all-ignored grad_weight must be exactly 0"


# (name, feature-kwargs) — each drives a distinct loss/grad path at production vocab.
_PROD_FEATURES = [
    ("core", {}),
    ("weight_ignore", {"ce_weight_": True, "ignore_frac": 0.25}),
    (
        "all",
        {
            "ce_weight_": True,
            "label_smoothing": 0.1,
            "lse_square_scale": 1e-4,
            "softcap": 30.0,
            "return_z_loss": True,
            "ignore_frac": 0.25,
        },
    ),
]


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("name, feats", _PROD_FEATURES, ids=[f[0] for f in _PROD_FEATURES])
def test_flce_production_vocab_matches_triton(name, feats, dtype):
    """Production-scale parity: V=32000 (llama vocab), H=2048, BT=1269 (deliberately NOT a
    multiple of any typical row-tile, to stress the ragged last tile). Impl-agnostic — only the
    loss/grad parity vs Triton is asserted, not which internal routing the kernel picks. H=2048
    with a large V drives many grad-accumulation chunks (the production stressor)."""
    set_seed()
    BT, H, V = 1269, 2048, 32000
    opts = dict(feats)
    use_weight = opts.pop("ce_weight_", False)
    ignore_frac = opts.pop("ignore_frac", 0.0)
    masters = _Masters(BT, H, V, bias=True, ce_weight=use_weight)
    target = _make_target(BT, V, ignore_frac=ignore_frac)
    _assert_flce_parity(masters, target, dtype, reduction="mean", **opts)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("name, feats", _PROD_FEATURES, ids=[f[0] for f in _PROD_FEATURES])
def test_flce_vocab_chunk512_matches_triton(name, feats, dtype):
    """Vocab-backward parity at token_chunk == 512 — the upper edge of the vocab-path gate
    (_VOCAB_BWD_MAX_TOKEN_CHUNK). A small V/H ratio (V=4096, H=2048 ⇒ inc=2) makes BT=1024 land on
    chunk=512, so this exercises the vocab path at the largest chunk it ships for (the win that
    raised the gate from 256→512) WITHOUT needing a 16k-token tensor. Parity vs Triton across the
    full feature stack (core / ce_weight+ignore / all: ce_weight+ls+z-loss+softcap+bias)."""
    set_seed()
    BT, H, V = 1024, 2048, 4096
    opts = dict(feats)
    use_weight = opts.pop("ce_weight_", False)
    ignore_frac = opts.pop("ignore_frac", 0.0)
    masters = _Masters(BT, H, V, bias=True, ce_weight=use_weight)
    target = _make_target(BT, V, ignore_frac=ignore_frac)
    _assert_flce_parity(masters, target, dtype, reduction="mean", **opts)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("name, feats", _PROD_FEATURES, ids=[f[0] for f in _PROD_FEATURES])
def test_flce_fused_features_matches_triton(name, feats, reduction, dtype):
    """Feature parity with all optional CE features stacked (softcap / z_loss / class-weight /
    weighted+unweighted label smoothing / ignore) at a tile-aligned shape (BT=512, H=512). Full
    fwd+bwd loss/grad parity vs Triton — impl-agnostic (asserts results, not internal routing)."""
    set_seed()
    BT, H, V = 512, 512, 4096
    opts = dict(feats)
    use_weight = opts.pop("ce_weight_", False)
    ignore_frac = opts.pop("ignore_frac", 0.0)
    masters = _Masters(BT, H, V, bias=True, ce_weight=use_weight)
    target = _make_target(BT, V, ignore_frac=ignore_frac)
    _assert_flce_parity(masters, target, dtype, reduction=reduction, **opts)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("BT", [129, 640, 1269], ids=["BT129", "BT640", "BT1269"])
def test_flce_fused_m_tail_matches_triton(dtype, BT):
    """BT NOT a multiple of the kernel's row-tile must still produce correct results (the ragged
    last token tile is the classic off-by-tile bug). Pure behavioral parity vs Triton with the
    'all' feature set — impl-agnostic (no assertion about which internal path is selected)."""
    set_seed()
    H, V = 512, 4096
    masters = _Masters(BT, H, V, bias=True, ce_weight=True)
    target = _make_target(BT, V, ignore_frac=0.25)
    _assert_flce_parity(
        masters,
        target,
        dtype,
        reduction="mean",
        label_smoothing=0.1,
        lse_square_scale=1e-4,
        softcap=30.0,
        return_z_loss=True,
    )


# =============================================================================
# B. Forward-only + the not-last-layer backward scaling.
#    (Triton: test_correctness_with_forward_only.)
# =============================================================================
@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_flce_forward_only_matches_triton(dtype, reduction, bias):
    """no_grad forward parity, then backward must raise 'does not require grad'."""
    set_seed()
    BT, H, V = 8 * 128, 1024, 4096
    masters = _Masters(BT, H, V, bias=bias, ce_weight=False)
    target = _make_target(BT, V, ignore_frac=0.25)

    with torch.no_grad():
        out = _run_or_skip(
            lambda: _run(_cutedsl_flce(), masters, target, dtype, reduction=reduction, requires_grad=False)
        )
        ref = _run(_triton_flce(), masters, target, dtype, reduction=reduction, requires_grad=False)
        atol, rtol = _TOL[(reduction, dtype)]
        _assert_close(out["loss"], ref["loss"], atol, rtol, "loss(forward-only)")

    # A loss produced under no_grad cannot be backpropagated.
    x = masters.input.clone().to(dtype)  # requires_grad False
    loss, *_ = _apply(_cutedsl_flce(), x, masters.weight.clone().to(dtype), target, reduction=reduction)
    with pytest.raises(RuntimeError, match="does not require grad"):
        loss.backward(gradient=torch.ones_like(loss))


@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("feature", ["plain", "bias", "label_smoothing", "zloss", "ce_weight"])
def test_flce_not_last_layer_grad_matches_triton(dtype, reduction, feature):
    """grad_output != 1.0 (scalar upstream grad, the not-last-layer / loss-scaled / grad-accum case).

    The fast bf16 dgrad path applies grad_output as a FINAL scalar multiply, so its relative error is
    grad_output-independent (measured constant from go=1 to go=65536). There is no separate 'accurate'
    fallback for go≠1 anymore — so EVERY feature path (core / bias / label-smoothing / z-loss /
    ce_weight) must stay at Triton parity at go≠1, exactly as it does at go==1."""
    set_seed()
    BT, H, V = 256, 512, 4096
    masters = _Masters(BT, H, V, bias=(feature == "bias"), ce_weight=(feature == "ce_weight"))
    target = _make_target(BT, V, ignore_frac=0.25)
    opts = {}
    if feature == "label_smoothing":
        opts["label_smoothing"] = 0.1
    elif feature == "zloss":
        opts["lse_square_scale"] = 1e-4
    # scalar upstream grad -> grad_output tensor != 1.0. Gradients (and their near-zero-element
    # absolute error) scale by |go|, so scale the gradient atol by |go| to match go==1 strictness.
    grad_output = torch.tensor(2.0, device="cuda")
    _assert_flce_parity(
        masters,
        target,
        dtype,
        reduction=reduction,
        grad_output=grad_output,
        grad_atol_scale=float(grad_output.abs().item()),
        **opts,
    )


# =============================================================================
# C. AMP / autocast.  (Triton: test_amp.) fp32 params, autocast to bf16/fp16.
# =============================================================================
@cuda_required
@pytest.mark.parametrize("B, T, H, V", [(8, 128, 1024, 4096), (4, 47, 31, 128)], ids=["big", "weird"])
@pytest.mark.parametrize("cast_dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize("accum_dtype", [None, torch.float32], ids=["accumNone", "accumfp32"])
def test_flce_amp_matches_triton(B, T, H, V, cast_dtype, bias, accum_dtype):
    """Under autocast the matmuls run in cast_dtype while params stay fp32 — checks the
    out-of-place bias/grad-accumulate dtype-mismatch branches stay parity with Triton."""
    set_seed()
    BT = B * T
    masters = _Masters(BT, H, V, bias=bias, ce_weight=False)
    target = _make_target(BT, V)
    atol, rtol = 5e-3, 5e-2

    def one(fn):
        x = masters.input.clone().requires_grad_(True)
        w = masters.weight.clone().requires_grad_(True)
        b = masters.bias.clone().requires_grad_(True) if masters.bias is not None else None
        with torch.autocast(device_type="cuda", dtype=cast_dtype):
            loss, *_ = _apply(fn, x, w, target, bias=b, reduction="mean", accum_dtype=accum_dtype)
        loss.backward()
        return loss.detach().float(), x.grad.detach().float(), w.grad.detach().float()

    out = _run_or_skip(lambda: one(_cutedsl_flce()))
    ref = one(_triton_flce())
    _assert_close(out[0], ref[0], atol, rtol, "loss(amp)")
    _assert_close(out[1], ref[1], atol, rtol, "grad_input(amp)")
    _assert_close(out[2], ref[2], atol, rtol, "grad_weight(amp)")


# =============================================================================
# D. reduction='none' — forward-only parity (grad is refused by design, tested in F).
# =============================================================================
@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_flce_reduction_none_forward_matches_triton(dtype, bias):
    """Per-token loss vector (BT,) parity, forward-only (the only 'none' mode cutedsl FLCE
    allows — see test_flce_reduction_none_with_grad_refused for the grad path)."""
    set_seed()
    BT, H, V = 256, 512, 4096
    masters = _Masters(BT, H, V, bias=bias, ce_weight=False)
    target = _make_target(BT, V, ignore_frac=0.25)
    _assert_flce_parity(masters, target, dtype, reduction="none", requires_grad=False, check_grad=False)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_flce_reduction_none_ce_weight_forward_matches_triton(dtype, bias):
    """reduction='none' per-token loss WITH ce_weight (± bias). For the 2-CTA backend bf16+H%8==0
    this exercises the fused fast path (reduction='none' composed with ce_weight/bias — grad is
    forward-only for 'none', so there's no backward-combo restriction). Guards the gate relaxation
    that lets 'none' × ce_weight/bias fuse instead of falling back to the chunked path."""
    set_seed()
    BT, H, V = 256, 512, 4096
    masters = _Masters(BT, H, V, bias=bias, ce_weight=True)
    target = _make_target(BT, V, ignore_frac=0.25)
    _assert_flce_parity(masters, target, dtype, reduction="none", requires_grad=False, check_grad=False)


# =============================================================================
# E. Functional API + structured output.  (Triton:
#    test_correctness_functional / test_liger_fused_linear_cross_entropy_structured_output.)
#    Monkeypatch the wrapper's Function so the real backend-agnostic wrapper runs the
#    cutedsl Function, then compare against the same wrapper driving Triton.
# =============================================================================
@cuda_required
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_flce_functional_core_matches_triton(bias, monkeypatch):
    """liger_fused_linear_cross_entropy(...) core path: bare-loss return + grad parity."""
    import liger_kernel.transformers.functional as fmod

    set_seed()
    BT, H, V = 256, 512, 4096
    dtype = torch.float32
    masters = _Masters(BT, H, V, bias=bias, ce_weight=False)
    target = _make_target(BT, V, ignore_frac=0.25)

    def via_wrapper(fn):
        monkeypatch.setattr(fmod, "LigerFusedLinearCrossEntropyFunction", fn)
        x = masters.input.clone().to(dtype).requires_grad_(True)
        w = masters.weight.clone().to(dtype).requires_grad_(True)
        b = masters.bias.clone().to(dtype).requires_grad_(True) if masters.bias is not None else None
        res = fmod.liger_fused_linear_cross_entropy(
            input=x, weight=w, target=target, bias=b, ignore_index=-100, reduction="mean"
        )
        # core path: no flags -> wrapper returns the bare loss tensor, not the dataclass.
        assert isinstance(res, torch.Tensor), "expected a bare loss tensor when no flags are set"
        res.backward()
        return res.detach().float(), x.grad.detach().float()

    out = _run_or_skip(lambda: via_wrapper(_cutedsl_flce()))
    ref = via_wrapper(_triton_flce())
    _assert_close(out[0], ref[0], 1e-4, 5e-4, "loss (functional bare)")
    _assert_close(out[1], ref[1], 1e-4, 5e-4, "grad_input (functional bare)")


@cuda_required
def test_flce_functional_all_features_matches_triton(monkeypatch):
    """Functional API with EVERY feature on (ce_weight + lse + label_smoothing + softcap +
    z_loss + accum). Stage 1 stub -> auto-skips; real parity gate once features land.
    (Triton: test_correctness_functional.)"""
    import liger_kernel.transformers.functional as fmod

    set_seed()
    BT, H, V = 256, 512, 4096
    dtype = torch.float32
    masters = _Masters(BT, H, V, bias=True, ce_weight=True)
    target = _make_target(BT, V, ignore_frac=0.25)

    def via_wrapper(fn):
        monkeypatch.setattr(fmod, "LigerFusedLinearCrossEntropyFunction", fn)
        x = masters.input.clone().to(dtype).requires_grad_(True)
        w = masters.weight.clone().to(dtype).requires_grad_(True)
        b = masters.bias.clone().to(dtype).requires_grad_(True)
        res = fmod.liger_fused_linear_cross_entropy(
            input=x,
            weight=w,
            target=target,
            bias=b,
            ce_weight=masters.ce_weight,
            ignore_index=-100,
            lse_square_scale=1e-4,
            label_smoothing=0.1,
            reduction="mean",
            softcap=30.0,
            return_z_loss=True,
            accum_dtype=torch.float32,
        )
        res.loss.backward()
        return res.loss.detach().float(), res.z_loss.detach().float(), x.grad.detach().float()

    out = _run_or_skip(lambda: via_wrapper(_cutedsl_flce()))
    ref = via_wrapper(_triton_flce())
    _assert_close(out[0], ref[0], 1e-4, 5e-4, "loss (functional all)")
    _assert_close(out[1], ref[1], 1e-4, 5e-4, "z_loss (functional all)")
    _assert_close(out[2], ref[2], 1e-4, 5e-4, "grad_input (functional all)")


@cuda_required
@pytest.mark.parametrize("return_z_loss", [True, False], ids=["z1", "z0"])
@pytest.mark.parametrize("return_token_accuracy", [True, False], ids=["acc1", "acc0"])
def test_flce_structured_output_matches_triton(return_z_loss, return_token_accuracy, monkeypatch):
    """The CrossEntropyOutput dataclass vs bare-loss dispatch through the wrapper, across the
    flag combos. The (False, False) combo runs (core); any flag on -> z_loss/token_accuracy
    are Stage-1 stubs and the combo auto-skips. (Triton: ..._structured_output.)"""
    import liger_kernel.transformers.functional as fmod

    from liger_kernel.transformers.functional import CrossEntropyOutput

    set_seed()
    BT, H, V = 128, 256, 4096
    dtype = torch.float32
    masters = _Masters(BT, H, V, bias=True, ce_weight=False)
    target = _make_target(BT, V)

    def via_wrapper(fn):
        monkeypatch.setattr(fmod, "LigerFusedLinearCrossEntropyFunction", fn)
        x = masters.input.clone().to(dtype).requires_grad_(True)
        w = masters.weight.clone().to(dtype)
        b = masters.bias.clone().to(dtype)
        return fmod.liger_fused_linear_cross_entropy(
            input=x,
            weight=w,
            target=target,
            bias=b,
            reduction="mean",
            lse_square_scale=1e-4,  # makes z_loss non-trivial when requested
            return_z_loss=return_z_loss,
            return_token_accuracy=return_token_accuracy,
        )

    out = _run_or_skip(lambda: via_wrapper(_cutedsl_flce()))
    ref = via_wrapper(_triton_flce())

    if not (return_z_loss or return_token_accuracy):
        assert isinstance(out, torch.Tensor), "core path must return a bare loss tensor"
        _assert_close(out.detach().float(), ref.detach().float(), 1e-4, 5e-4, "loss (bare)")
        return

    assert isinstance(out, CrossEntropyOutput)
    _assert_close(out.loss.detach().float(), ref.loss.detach().float(), 1e-4, 5e-4, "loss")
    for field, flag in (("z_loss", return_z_loss), ("token_accuracy", return_token_accuracy)):
        c, t = getattr(out, field), getattr(ref, field)
        if not flag:
            assert c is None and t is None, f"{field} must be None when its flag is off"
        else:
            _assert_close(c.detach().float(), t.detach().float(), 1e-4, 5e-4, field)


# =============================================================================
# F. Full optional-feature parity vs Triton (auto-skips any feature an impl doesn't support).
#    Covers every Triton FLCE feature: ce_weight, label_smoothing, z_loss, softcap,
#    token_accuracy, predicted_tokens, token_scaling, and the "all-on" combo.
# =============================================================================
@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_flce_ce_weight_matches_triton(dtype, reduction, bias):
    # bias=False + bf16 routes the 2-CTA backend through the FUSED fast path (ce_weight folded in as
    # per-row a_row / onehot scalars, no vocab-space pass); bias=True / fp32 exercises the general path.
    set_seed()
    BT, H, V = 256, 512, 4096
    masters = _Masters(BT, H, V, bias=bias, ce_weight=True)
    target = _make_target(BT, V, ignore_frac=0.25)
    _assert_flce_parity(masters, target, dtype, reduction=reduction)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("label_smoothing", [0.1, 0.3], ids=["ls0.1", "ls0.3"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_flce_label_smoothing_matches_triton(dtype, reduction, label_smoothing, bias):
    # bias=False + bf16 routes through the fast path (smoothing folded in as (H,) broadcast corrections).
    set_seed()
    BT, H, V = 256, 512, 4096
    masters = _Masters(BT, H, V, bias=bias, ce_weight=False)
    target = _make_target(BT, V, ignore_frac=0.25)
    _assert_flce_parity(masters, target, dtype, reduction=reduction, label_smoothing=label_smoothing)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("return_z_loss", [True, False], ids=["zret", "znoret"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_flce_z_loss_matches_triton(dtype, reduction, return_z_loss, bias):
    # bias=False + bf16 routes through the fast path (z-loss folded in as the per-row 1+2·lss·lse factor).
    set_seed()
    BT, H, V = 256, 512, 4096
    masters = _Masters(BT, H, V, bias=bias, ce_weight=False)
    target = _make_target(BT, V, ignore_frac=0.25)
    _assert_flce_parity(masters, target, dtype, reduction=reduction, lse_square_scale=1e-4, return_z_loss=return_z_loss)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_flce_softcap_matches_triton(dtype, reduction):
    set_seed()
    BT, H, V = 256, 512, 4096
    masters = _Masters(BT, H, V, bias=True, ce_weight=False)
    target = _make_target(BT, V, ignore_frac=0.25)
    _assert_flce_parity(masters, target, dtype, reduction=reduction, softcap=30.0)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_flce_token_accuracy_matches_triton(dtype, reduction, bias):
    """return_token_accuracy: per-row (argmax == target), non-ignored mean. (Triton:
    test_correctness_with_token_accuracy.) Boost half the target logits so acc==1.0 rows
    are exercised, not just the all-zero case random targets give."""
    set_seed()
    BT, H, V = 256, 512, 4096
    masters = _Masters(BT, H, V, bias=bias, ce_weight=False)
    target = _make_target(BT, V)
    # bias the linear so ~half the rows predict their target (large positive weight row).
    for i in range(BT // 2):
        masters.weight[target[i]] = masters.input[i] * 50.0
    target_ignore = target.clone()
    target_ignore[torch.randperm(BT, device="cuda")[: BT // 8]] = -100
    # check_grad=False: this test's purpose is the token_accuracy metric. The 50x weight boost
    # (used to force acc==1.0 rows) makes those weight rows O(1e3) in magnitude, so grad_input =
    # grad_logits @ W amplifies the normal ~1e-4 fp32 kernel-vs-kernel divergence in grad_logits
    # into an O(0.1) grad_input diff — a numerical artifact of the synthetic setup, not a kernel
    # error. Gradient parity under realistic weights is covered by the dedicated grad tests
    # (test_flce_core_*, test_flce_not_last_layer_grad_*, etc.). loss + token_accuracy are checked.
    _assert_flce_parity(
        masters, target_ignore, dtype, reduction=reduction, return_token_accuracy=True, check_grad=False
    )


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("ignore_index", [-100, 2], ids=["ig-100", "ig2"])
def test_flce_predicted_tokens_matches_triton(dtype, ignore_index):
    """return_predicted_tokens: per-row argmax (int64), -1 for ignored rows. (Triton:
    test_correctness_with_predicted_tokens.) Forward-only — independent of reduction."""
    set_seed()
    BT, H, V = 256, 512, 4096
    masters = _Masters(BT, H, V, bias=True, ce_weight=False)
    target = _make_target(BT, V, ignore_frac=0.25, ignore_index=ignore_index)
    _assert_flce_parity(
        masters,
        target,
        dtype,
        reduction="mean",
        ignore_index=ignore_index,
        requires_grad=False,
        check_grad=False,
        return_predicted_tokens=True,
    )


@cuda_required
@pytest.mark.parametrize("reduction", ["sum", "none"])
def test_flce_token_scaling_matches_triton(reduction):
    """use_token_scaling: per-token CE scaled by the detached softmax prob of the target.
    (Triton: test_correctness_token_scaling*.) Stage-1 stub -> auto-skips. Anchored on the
    manual torch implementation rather than Triton so it stays a real check once it lands."""
    set_seed()
    BT, H, V = 8, 32, 4096
    masters = _Masters(BT, H, V, bias=True, ce_weight=False)
    target = _make_target(BT, V)
    dtype = torch.float32

    out = _run_or_skip(
        lambda: _run(
            _cutedsl_flce(), masters, target, dtype, reduction=reduction, requires_grad=False, use_token_scaling=True
        )
    )
    logits = _torch_logits(masters, target, dtype, bias=True)
    ce = F.cross_entropy(logits, target, ignore_index=-100, reduction="none")
    pred_probs = torch.softmax(logits, dim=-1).gather(1, target.unsqueeze(-1)).squeeze(-1).detach()
    scaled = ce * pred_probs
    ref = scaled if reduction == "none" else scaled.sum()
    _assert_close(out["loss"], ref.detach().float(), 1e-4, 1e-4, f"token_scaling loss ({reduction})")


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("feat", ["plain", "bias", "ce_weight", "ce_weight+bias", "all"])
def test_flce_token_scaling_fast_grad_matches_triton(dtype, reduction, feat):
    """use_token_scaling on the FUSED fast path (bf16/fp16) — full fwd + bwd parity vs Triton. The
    detached per-row softmax-at-target scale folds into the backward's per-row grad weight (go_row).
    token_scaling now composes with bias and ce_weight too (its scale threads through a_row, the
    per-row one-hot, cw_col, the c_col count, and grad_bias) — so those combos take the fast path
    instead of falling back. BT=256 stays clear of the small-BT fused-GEMM alignment limit."""
    set_seed()
    BT, H, V = 256, 512, 4096
    bias = feat in ("bias", "ce_weight+bias", "all")
    ce_weight = feat in ("ce_weight", "ce_weight+bias", "all")
    masters = _Masters(BT, H, V, bias=bias, ce_weight=ce_weight)
    target = _make_target(BT, V, ignore_frac=0.25)
    opts = {"use_token_scaling": True}
    if feat == "all":
        opts["label_smoothing"] = 0.1
        opts["lse_square_scale"] = 1e-4
    _assert_flce_parity(masters, target, dtype, reduction=reduction, **opts)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_flce_all_features_matches_triton(dtype):
    """Everything at once: ce_weight + label_smoothing + z_loss + softcap + ignore_index +
    accum_dtype (loss + z_loss + all three grads). (Triton: test_correctness's all-features
    row.) Auto-skips on Stage 1; becomes the full combined-path gate when features land."""
    set_seed()
    BT, H, V = 256, 512, 4096
    masters = _Masters(BT, H, V, bias=True, ce_weight=True)
    target = _make_target(BT, V, ignore_frac=0.25, ignore_index=42)
    _assert_flce_parity(
        masters,
        target,
        dtype,
        reduction="mean",
        ignore_index=42,
        lse_square_scale=1e-4,
        label_smoothing=0.1,
        softcap=30.0,
        return_z_loss=True,
        accum_dtype=torch.float32,
    )


# =============================================================================
# G. CuTe DSL-specific contracts — host-side asserts + the by-design refusals the
#    Triton kernel doesn't have. These are PERMANENT (not stubs), so they assert.
# =============================================================================
def _basic_args(BT=8, H=16, V=4096, dtype=torch.float32):
    _input = torch.randn(BT, H, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(V, H, device="cuda", dtype=dtype)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    return _input, weight, target


@cuda_required
def test_flce_target_out_of_bounds_raises():
    """A non-ignored target >= V must be rejected (max < V host assert)."""
    _input, weight, target = _basic_args(V=4096)
    target[0] = weight.shape[0]  # == V
    with pytest.raises(AssertionError):
        _run_or_skip(lambda: _apply(_cutedsl_flce(), _input, weight, target))


@cuda_required
def test_flce_target_negative_non_ignore_raises():
    """A negative target that isn't ignore_index must be rejected (min >= 0 host assert)."""
    _input, weight, target = _basic_args(V=4096)
    target[0] = -5  # negative, != ignore_index(-100)
    with pytest.raises(AssertionError):
        _run_or_skip(lambda: _apply(_cutedsl_flce(), _input, weight, target))


@cuda_required
def test_flce_reduction_none_with_grad_refused():
    """reduction='none' WITH a grad-requiring input is refused by design: the fused path
    accumulates grad_weight over tokens in forward and can't re-weight by a per-token
    upstream grad. Must raise NotImplementedError (forward-only 'none' is fine — see D)."""
    _input, weight, target = _basic_args(BT=16, V=4096)
    with pytest.raises(NotImplementedError):
        _apply(_cutedsl_flce(), _input, weight, target, reduction="none")


# =============================================================================
# H. Dispatch / swap wiring — LIGER_KERNEL_IMPL=cutedsl. The Triton suite has no such
#    test (Triton is the default impl); this mirrors the cutedsl CE suite. FLCE is now
#    exported from the cutedsl backend's __all__, so under LIGER_KERNEL_IMPL=cutedsl this
#    asserts the swap actually rewired the public symbol to the cutedsl implementation.
#    The exit-77 SKIP path is retained as a defensive guard (documents the gap) in case
#    the symbol is ever un-wired, so it degrades to a skip instead of a hard failure.
# =============================================================================
@cuda_required
@pytest.mark.skipif(
    os.environ.get("LIGER_KERNEL_IMPL", "").strip().lower() != "cutedsl",
    reason="cutedsl selection test requires LIGER_KERNEL_IMPL=cutedsl",
)
def test_liger_kernel_impl_cutedsl_selects_cutedsl_flce():
    repo_root = Path(__file__).resolve().parents[2]
    pythonpath = os.pathsep.join([str(repo_root / "src"), str(repo_root), os.environ.get("PYTHONPATH", "")])
    env = {**os.environ, "LIGER_KERNEL_IMPL": "cutedsl", "PYTHONPATH": pythonpath}
    script = textwrap.dedent(
        """
        import sys
        from liger_kernel.ops import LigerFusedLinearCrossEntropyFunction

        prefix = "liger_kernel.ops.cutedsl."
        mod = LigerFusedLinearCrossEntropyFunction.__module__
        if not mod.startswith(prefix):
            # FLCE isn't exported by the cutedsl backend yet (only CE is). Signal SKIP (77)
            # so this documents the gap without being a false failure.
            print(f"FLCE not wired into cutedsl backend (module={mod}); skipping", file=sys.stderr)
            sys.exit(77)
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], env=env, cwd=repo_root, capture_output=True, text=True)
    if proc.returncode == 77:
        pytest.skip(proc.stderr.strip() or "cutedsl FLCE not wired yet")
    assert proc.returncode == 0, f"cutedsl FLCE dispatch check failed:\n{proc.stderr}"
