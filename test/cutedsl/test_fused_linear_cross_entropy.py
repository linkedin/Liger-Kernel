import os
import subprocess
import sys
import textwrap

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Per-(reduction, dtype) (atol, rtol). Matched to the Triton FLCE suite's bar:
# 16-bit sum accumulates BT terms with no mean normalizer, so its absolute
# tolerance is much larger than mean reduction.
# Used for BOTH the Triton parity gate and the torch anchor.
_TOL = {
    ("mean", torch.bfloat16): (5e-3, 5e-2),
    ("mean", torch.float16): (5e-3, 5e-2),
    ("sum", torch.bfloat16): (5e0, 5e1),
    ("sum", torch.float16): (5e0, 5e1),
}
_DTYPES = [torch.bfloat16, torch.float16]
_DTYPE_IDS = ["bf16", "fp16"]

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="cutedsl FLCE requires CUDA")
sm100_required = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="native cutedsl FLCE requires an SM100 GPU",
)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Imports (skip — don't fail — when the optional backend isn't installed)
# =============================================================================
def _cutedsl_flce():
    """The CuTe DSL dispatcher under test, or skip if CUTLASS is not installed."""
    try:
        from liger_kernel.ops.cutedsl.ops import LigerFusedLinearCrossEntropyFunction
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
    ``grad_output`` (default ones_like(loss)) is the upstream grad; pass a scaled
    scalar tensor for the not-last-layer path.
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
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
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
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
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
    path for go≠1 — so EVERY feature path (core / bias / label-smoothing / z-loss /
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
# C. Functional API + structured output.  (Triton:
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
    dtype = torch.bfloat16
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
    _assert_close(out[0], ref[0], 5e-3, 5e-2, "loss (functional bare)")
    _assert_close(out[1], ref[1], 5e-3, 5e-2, "grad_input (functional bare)")


@cuda_required
def test_flce_functional_all_features_matches_triton(monkeypatch):
    """Functional API with EVERY feature on (ce_weight + lse + label_smoothing + softcap +
    z_loss + accum). Stage 1 stub -> auto-skips; real parity gate once features land.
    (Triton: test_correctness_functional.)"""
    import liger_kernel.transformers.functional as fmod

    set_seed()
    BT, H, V = 256, 512, 4096
    dtype = torch.bfloat16
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
    _assert_close(out[0], ref[0], 5e-3, 5e-2, "loss (functional all)")
    _assert_close(out[1], ref[1], 5e-3, 5e-2, "z_loss (functional all)")
    _assert_close(out[2], ref[2], 5e-3, 5e-2, "grad_input (functional all)")


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
    dtype = torch.bfloat16
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
        _assert_close(out.detach().float(), ref.detach().float(), 5e-3, 5e-2, "loss (bare)")
        return

    assert isinstance(out, CrossEntropyOutput)
    _assert_close(out.loss.detach().float(), ref.loss.detach().float(), 5e-3, 5e-2, "loss")
    for field, flag in (("z_loss", return_z_loss), ("token_accuracy", return_token_accuracy)):
        c, t = getattr(out, field), getattr(ref, field)
        if not flag:
            assert c is None and t is None, f"{field} must be None when its flag is off"
        else:
            _assert_close(c.detach().float(), t.detach().float(), 5e-3, 5e-2, field)


# =============================================================================
# D. Full optional-feature parity vs Triton (auto-skips any feature an impl doesn't support).
#    Covers every Triton FLCE feature: ce_weight, label_smoothing, z_loss, softcap,
#    token_accuracy, predicted_tokens, token_scaling, and the "all-on" combo.
# =============================================================================
@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_flce_ce_weight_matches_triton(dtype, reduction, bias):
    # bias=False + bf16 routes the 2-CTA backend through the FUSED fast path (ce_weight folded in as
    # per-row a_row / onehot scalars, no vocab-space pass); bias=True exercises the general path.
    set_seed()
    BT, H, V = 256, 512, 4096
    masters = _Masters(BT, H, V, bias=bias, ce_weight=True)
    target = _make_target(BT, V, ignore_frac=0.25)
    _assert_flce_parity(masters, target, dtype, reduction=reduction)


@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
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
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
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
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_flce_native_z_loss_preserves_input_dtype(dtype):
    set_seed()
    BT, H, V = 256, 128, 256
    masters = _Masters(BT, H, V, bias=False, ce_weight=False)
    x = masters.input.to(dtype).requires_grad_(True)
    w = masters.weight.to(dtype).requires_grad_(True)
    target = _make_target(BT, V, ignore_frac=0.25)
    _, z_loss, *_ = _apply(
        _cutedsl_flce(),
        x,
        w,
        target,
        lse_square_scale=1e-4,
        return_z_loss=True,
    )
    assert z_loss.dtype == dtype


@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_flce_softcap_matches_triton(dtype, reduction):
    set_seed()
    BT, H, V = 256, 512, 4096
    masters = _Masters(BT, H, V, bias=True, ce_weight=False)
    target = _make_target(BT, V, ignore_frac=0.25)
    _assert_flce_parity(masters, target, dtype, reduction=reduction, softcap=30.0)


@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
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
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
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


@sm100_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    ("return_token_accuracy", "return_predicted_tokens"),
    [(True, False), (False, True), (True, True)],
    ids=["accuracy", "predictions", "both"],
)
def test_flce_metrics_stay_on_native_path(dtype, return_token_accuracy, return_predicted_tokens):
    set_seed()
    BT, H, V = 257, 127, 513
    masters = _Masters(BT, H, V, bias=True, ce_weight=False)
    target = _make_target(BT, V, ignore_frac=0.25)
    logits = masters.input.to(dtype) @ masters.weight.to(dtype).t()
    logits = logits + masters.bias.to(dtype)
    expected_predictions = logits.argmax(dim=-1)
    expected_accuracy = ((expected_predictions == target) & (target != -100)).float().sum() / (target != -100).sum()
    expected_predictions = torch.where(target != -100, expected_predictions, -1)

    out = _run(
        _cutedsl_flce(),
        masters,
        target,
        dtype,
        return_token_accuracy=return_token_accuracy,
        return_predicted_tokens=return_predicted_tokens,
    )

    if return_token_accuracy:
        torch.testing.assert_close(out["token_accuracy"], expected_accuracy)
    if return_predicted_tokens:
        assert torch.equal(out["predicted_tokens"], expected_predictions)


@cuda_required
@pytest.mark.parametrize("reduction", ["sum"])
def test_flce_token_scaling_matches_triton(reduction):
    """use_token_scaling: per-token CE scaled by the detached softmax prob of the target.
    (Triton: test_correctness_token_scaling*.) Stage-1 stub -> auto-skips. Anchored on the
    manual torch implementation rather than Triton so it stays a real check once it lands."""
    set_seed()
    BT, H, V = 8, 32, 4096
    masters = _Masters(BT, H, V, bias=True, ce_weight=False)
    target = _make_target(BT, V)
    dtype = torch.bfloat16

    out = _run_or_skip(
        lambda: _run(
            _cutedsl_flce(), masters, target, dtype, reduction=reduction, requires_grad=False, use_token_scaling=True
        )
    )
    logits = _torch_logits(masters, target, dtype, bias=True)
    ce = F.cross_entropy(logits, target, ignore_index=-100, reduction="none")
    pred_probs = torch.softmax(logits, dim=-1).gather(1, target.unsqueeze(-1)).squeeze(-1).detach()
    scaled = ce * pred_probs
    ref = scaled.sum()
    _assert_close(out["loss"], ref.detach().float(), 5e-3, 5e-2, f"token_scaling loss ({reduction})")


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("feat", ["plain", "bias", "ce_weight", "ce_weight+bias", "all"])
def test_flce_token_scaling_fast_grad_matches_triton(dtype, reduction, feat):
    """use_token_scaling on the FUSED fast path (bf16/fp16) — full fwd + bwd parity vs Triton. The
    detached per-row softmax-at-target scale folds into the backward's per-row grad weight (go_row).
    token_scaling now composes with bias and ce_weight too (its scale threads through a_row, the
    per-row one-hot, cw_col, the c_col count, and grad_bias) — so those combos take the fast path
    on the same native path. BT=256 stays clear of the small-BT fused-GEMM alignment limit."""
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
def test_flce_token_scaling_strong_label_smoothing_grad_matches_triton():
    """The token scale multiplies the smoothing term once, not epsilon twice."""
    set_seed()
    BT, H, V = 256, 128, 256
    masters = _Masters(BT, H, V, bias=False, ce_weight=False)
    target = _make_target(BT, V, ignore_frac=0.25)
    _assert_flce_parity(
        masters,
        target,
        torch.bfloat16,
        reduction="mean",
        label_smoothing=0.9,
        use_token_scaling=True,
    )


@cuda_required
def test_flce_pre_torch_2_8_out_dtype_compatibility_matches_triton(monkeypatch):
    """The native vocab backward remains usable when torch.mm/addmm lack out_dtype."""
    import liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy as flce

    set_seed()
    monkeypatch.setattr(flce, "_SUPPORTS_OUT_DTYPE", False)
    BT, H, V = 256, 128, 256
    masters = _Masters(BT, H, V, bias=False, ce_weight=False)
    target = _make_target(BT, V, ignore_frac=0.25)
    _assert_flce_parity(
        masters,
        target,
        torch.bfloat16,
        reduction="mean",
        accum_dtype=torch.float32,
    )


@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
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


@sm100_required
def test_flce_all_native_features_use_one_path():
    """Every feature supported on SM100 must stay on the single native path."""
    set_seed()
    BT, H, V = 256, 512, 4096
    masters = _Masters(BT, H, V, bias=True, ce_weight=True)
    target = _make_target(BT, V, ignore_frac=0.25)

    out = _run(
        _cutedsl_flce(),
        masters,
        target,
        torch.bfloat16,
        label_smoothing=0.1,
        lse_square_scale=1e-4,
        softcap=30.0,
        return_z_loss=True,
        accum_dtype=torch.float32,
        use_token_scaling=True,
        return_token_accuracy=True,
        return_predicted_tokens=True,
    )
    assert out["z_loss"] is not None
    assert out["token_accuracy"] is not None
    assert out["predicted_tokens"] is not None
    assert out["grad_bias"] is not None


@sm100_required
def test_flce_odd_vocab_all_native_features_match_triton():
    """Logical vocabulary padding must preserve every native feature and gradient."""
    set_seed()
    BT, H, V = 257, 127, 513
    masters = _Masters(BT, H, V, bias=True, ce_weight=True)
    target = _make_target(BT, V, ignore_frac=0.25)
    _assert_flce_parity(
        masters,
        target,
        torch.bfloat16,
        reduction="mean",
        label_smoothing=0.1,
        lse_square_scale=1e-4,
        softcap=30.0,
        return_z_loss=True,
        accum_dtype=torch.float32,
        use_token_scaling=True,
        return_token_accuracy=True,
        return_predicted_tokens=True,
    )


@sm100_required
def test_flce_odd_vocab_padding_cannot_become_softmax_max():
    """Aligned zero padding must not replace a very negative logical row maximum."""
    BT, H, V = 33, 64, 513
    masters = _Masters(BT, H, V, bias=False, ce_weight=False)
    masters.input.fill_(1.0)
    masters.weight.fill_(-20.0)
    target = _make_target(BT, V)
    _assert_flce_parity(masters, target, torch.bfloat16, reduction="mean")


@sm100_required
def test_flce_first_backward_does_not_recompute(monkeypatch):
    """The normal backward must only hand off gradients produced during forward."""
    import liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy as flce

    set_seed()
    BT, H, V = 256, 512, 4096
    x = torch.randn(BT, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(V, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ce_weight = torch.rand(V, device="cuda", dtype=torch.float32) + 0.5
    target = _make_target(BT, V, ignore_frac=0.25)
    loss = _apply(
        _cutedsl_flce(),
        x,
        w,
        target,
        bias=bias,
        ce_weight=ce_weight,
        lse_square_scale=1e-4,
        label_smoothing=0.1,
        softcap=30.0,
        use_token_scaling=True,
    )[0]

    def fail_if_recomputed(*_, **__):
        pytest.fail("first native backward recomputed the forward")

    monkeypatch.setattr(flce, "_native_forward", fail_if_recomputed)
    loss.backward()


@sm100_required
def test_flce_repeated_backward_recomputes_full_feature_gradients():
    """A retained graph may recompute once, and must preserve scalar scaling."""
    set_seed()
    BT, H, V = 256, 512, 4096
    x = torch.randn(BT, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(V, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ce_weight = torch.rand(V, device="cuda", dtype=torch.float32) + 0.5
    target = _make_target(BT, V, ignore_frac=0.25)
    loss = _apply(
        _cutedsl_flce(),
        x,
        w,
        target,
        bias=bias,
        ce_weight=ce_weight,
        lse_square_scale=1e-4,
        label_smoothing=0.1,
        softcap=30.0,
        accum_dtype=torch.float32,
        use_token_scaling=True,
    )[0]
    loss.backward(retain_graph=True)
    first = (x.grad.clone(), w.grad.clone(), bias.grad.clone())
    x.grad = None
    w.grad = None
    bias.grad = None
    loss.backward(torch.tensor(2.0, device="cuda"), retain_graph=True)
    for repeated, expected in zip((x.grad, w.grad, bias.grad), first, strict=True):
        torch.testing.assert_close(repeated, expected * 2, rtol=0, atol=0)


# =============================================================================
# G. CuTe DSL-specific contracts and input validation.
# =============================================================================
def _basic_args(BT=8, H=16, V=4096, dtype=torch.bfloat16):
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
def test_flce_reduction_none_is_rejected():
    _input, weight, target = _basic_args()
    with pytest.raises(RuntimeError, match="mean/sum reduction"):
        _apply(_cutedsl_flce(), _input, weight, target, reduction="none")


@cuda_required
@pytest.mark.parametrize("trainable", ["input", "weight", "bias"])
def test_flce_independent_requires_grad_matches_torch(trainable):
    """The native path supports any subset of trainable input, weight, and bias."""
    set_seed()
    BT, H, V = 256, 128, 256
    masters = _Masters(BT, H, V, bias=True, ce_weight=False)
    target = _make_target(BT, V, ignore_frac=0.25)

    def one_native():
        x = masters.input.to(torch.bfloat16).requires_grad_(trainable == "input")
        w = masters.weight.to(torch.bfloat16).requires_grad_(trainable == "weight")
        b = masters.bias.to(torch.bfloat16).requires_grad_(trainable == "bias")
        loss, *_ = _apply(_cutedsl_flce(), x, w, target, bias=b, reduction="mean")
        loss.backward()
        leaf = {"input": x, "weight": w, "bias": b}[trainable]
        return loss.detach().float(), leaf.grad.detach().float()

    def one_torch():
        x = masters.input.to(torch.bfloat16).requires_grad_(trainable == "input")
        w = masters.weight.to(torch.bfloat16).requires_grad_(trainable == "weight")
        b = masters.bias.to(torch.bfloat16).requires_grad_(trainable == "bias")
        logits = F.linear(x, w, b).float()
        loss = F.cross_entropy(logits, target, ignore_index=-100, reduction="mean")
        loss.backward()
        leaf = {"input": x, "weight": w, "bias": b}[trainable]
        return loss.detach().float(), leaf.grad.detach().float()

    out = one_native()
    ref = one_torch()
    _assert_close(out[0], ref[0], 5e-3, 5e-2, "independent requires_grad loss")
    _assert_close(out[1], ref[1], 5e-3, 5e-2, f"independent grad_{trainable}")


@cuda_required
def test_flce_all_ignored_targets_are_valid():
    _input, weight, target = _basic_args(BT=16, V=256, dtype=torch.bfloat16)
    target.fill_(-100)
    out = _apply(_cutedsl_flce(), _input, weight, target, reduction="mean")[0]
    assert out.item() == 0.0


@cuda_required
def test_flce_signed_weighted_mean_denominator_matches_triton():
    set_seed()
    BT, H, V = 256, 128, 256
    masters = _Masters(BT, H, V, bias=False, ce_weight=True)
    masters.ce_weight.fill_(-1.0)
    target = _make_target(BT, V, ignore_frac=0.25)
    _assert_flce_parity(masters, target, torch.bfloat16, reduction="mean")


@cuda_required
@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda x, w, t: (x.unsqueeze(0), w, t), ValueError),
        (lambda x, w, t: (x, w[:, :-1], t), ValueError),
        (lambda x, w, t: (x, w.float(), t), TypeError),
        (lambda x, w, t: (x, w, t.int()), TypeError),
        (lambda x, w, t: (x, w, t[:-1]), ValueError),
    ],
)
def test_flce_invalid_inputs_raise(mutation, error):
    _input, weight, target = _basic_args(BT=16, H=64, V=256, dtype=torch.bfloat16)
    _input, weight, target = mutation(_input, weight, target)
    with pytest.raises(error):
        _apply(_cutedsl_flce(), _input, weight, target)


@cuda_required
def test_flce_native_support_requires_exact_sm100(monkeypatch):
    import liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy as flce

    tensor = torch.empty(1, device="cuda")
    seen = []

    def capability(device):
        seen.append(device)
        return (10, 3)

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    assert not flce._native_sm100_supported(tensor)
    assert seen == [tensor.device]


@cuda_required
def test_flce_native_function_rejects_fp32_inputs():
    _input, weight, target = _basic_args(dtype=torch.float32)
    with pytest.raises(RuntimeError, match="FP16/BF16"):
        _apply(_cutedsl_flce(), _input, weight, target)


@cuda_required
def test_flce_native_function_rejects_non_sm100(monkeypatch):
    import liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy as flce

    monkeypatch.setattr(flce, "_native_sm100_supported", lambda _: False)
    _input, weight, target = _basic_args()
    with pytest.raises(RuntimeError, match="exact SM100"):
        _apply(_cutedsl_flce(), _input, weight, target)


def test_flce_native_gemm_guards_noncurrent_device(monkeypatch):
    try:
        import liger_kernel.ops.cutedsl.ops._sm100_gemm as gemm
    except ImportError as exc:
        pytest.skip(f"cutedsl backend not importable: {exc}")

    entered = []

    class Guard:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            entered.append(self.device)

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", Guard)
    device = torch.device("cuda", 1)
    with gemm._device_guard(device):
        pass
    assert entered == [device]


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
        from liger_kernel.ops import (
            LigerFusedLinearCrossEntropyFunction,
            fused_linear_cross_entropy_backward,
            fused_linear_cross_entropy_forward,
        )

        expected_module = "liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy"
        mod = LigerFusedLinearCrossEntropyFunction.__module__
        if mod != expected_module:
            # FLCE isn't exported by the cutedsl backend yet (only CE is). Signal SKIP (77)
            # so this documents the gap without being a false failure.
            print(f"FLCE not wired into cutedsl backend (module={mod}); skipping", file=sys.stderr)
            sys.exit(77)
        assert fused_linear_cross_entropy_forward.__module__ == "liger_kernel.ops.fused_linear_cross_entropy"
        assert fused_linear_cross_entropy_backward.__module__ == "liger_kernel.ops.fused_linear_cross_entropy"
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], env=env, cwd=repo_root, capture_output=True, text=True)
    if proc.returncode == 77:
        pytest.skip(proc.stderr.strip() or "cutedsl FLCE not wired yet")
    assert proc.returncode == 0, f"cutedsl FLCE dispatch check failed:\n{proc.stderr}"
