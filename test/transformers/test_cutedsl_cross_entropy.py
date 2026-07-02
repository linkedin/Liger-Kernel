import os
import subprocess
import sys
import textwrap

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Per-dtype (atol, rtol) for the cutedsl-vs-Triton parity checks. Anchored to the Triton CE
# suite's main-correctness bar (test_cross_entropy.py: fp32=(1e-8,1e-6), bf16=(1e-8,5e-2)),
# loosened only where measured kernel-vs-kernel divergence requires it — this suite stresses
# the kernels harder than Triton's does (fp16, extreme peaked logits, aux metrics):
#   * bf16 -> Triton's rtol bar (5e-2), but rtol alone does NOT dominate on the extreme
#     peaked-softmax rows (logit=50) in the accuracy test: those produce near-zero grad
#     entries where bf16's ~2-digit mantissa lands a few ULPs from Triton (~8.7e-6 abs), and
#     rtol*|ref| collapses toward zero there. A 5e-5 atol floor covers it with ~5.7x headroom
#     and is still 100x tighter than the original 5e-3.
#   * fp32 -> loss is bit-identical; grad matches (1e-8,1e-6) except the extreme peaked-softmax
#     rows (logit=50) in the accuracy test, where base-2 ex2.approx divergence reaches ~1.1e-6
#     abs at sum/none scale. A 1e-5 atol floor covers that with ~10x headroom (still 10x tighter
#     than the original 1e-4).
#   * fp16 -> Triton has no tight fp16 bar (its only fp16 reference uses 1e-2). fp16's ~3-digit
#     mantissa rounds near-zero grad entries a few ULPs apart (~7.6e-6 abs); the 5e-3 atol floor
#     covers it and is still 2x tighter than Triton's fp16 reference.
_TOL = {
    torch.float32: (1e-5, 1e-5),
    torch.bfloat16: (5e-5, 5e-2),
    torch.float16: (5e-3, 5e-2),
}
_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
_DTYPE_IDS = ["bf16", "fp16", "fp32"]

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="cutedsl CE requires CUDA")


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Imports (skip — don't fail — when the optional backend isn't installed)
# =============================================================================
def _cutedsl_ce():
    """cutedsl ``LigerCrossEntropyFunction``, or skip if CUTLASS isn't installed."""
    try:
        from liger_kernel.ops.cutedsl.ops.cross_entropy import LigerCrossEntropyFunction
    except ImportError as exc:
        pytest.skip(f"cutedsl backend not importable (cutlass.cute missing?): {exc}")
    return LigerCrossEntropyFunction


def _cutedsl_ce_module():
    """The cutedsl CE *module* — white-box access to ``_compile_cache`` (for the warp-count test)."""
    try:
        import liger_kernel.ops.cutedsl.ops.cross_entropy as mod
    except ImportError as exc:
        pytest.skip(f"cutedsl backend not importable: {exc}")
    return mod


def _triton_ce():
    """Triton reference ``LigerCrossEntropyFunction`` (the parity oracle)."""
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
    """Positional ``apply`` matching BOTH the Triton and cutedsl CE signatures."""
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
    """Run ``thunk``; if the cutedsl branch is still a stub, SKIP instead of fail."""
    try:
        return thunk()
    except NotImplementedError as exc:
        pytest.skip(f"cutedsl CE branch not implemented yet: {exc}")


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
    """One CE fwd (+bwd); returns ``(loss, z_loss, grad, token_accuracy, predicted_tokens)``,
    each detached fp32 (predicted_tokens int64) or None.

    Backward protocol (drives the three branches in ``cross_entropy_backward``):
      * reduction in {mean, sum}, default      -> ``grad_output == 1.0`` fast path.
      * reduction in {mean, sum}, ``grad_scale`` -> scalar ``grad_output != 1`` (not-last-layer).
      * reduction == 'none'                     -> per-row vector ``grad_output``.
    """
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
    if not torch.allclose(out, ref, atol=atol, rtol=rtol):
        diff = (out - ref).abs()
        raise AssertionError(
            f"{what} mismatch vs Triton: max|diff|={diff.max().item():.3e} "
            f"mean|diff|={diff.mean().item():.3e} (atol={atol}, rtol={rtol})"
        )


def _assert_ce_parity(base_logits, target, dtype, *, check_grad=True, **opts):
    """Run cutedsl + Triton with identical inputs/opts and compare loss[/z_loss][/grad]."""
    atol, rtol = _TOL[dtype]
    ref = _run(_triton_ce(), base_logits, target, dtype, **opts)
    out = _run_or_skip(lambda: _run(_cutedsl_ce(), base_logits, target, dtype, **opts))
    _assert_close(out[0], ref[0], atol, rtol, "loss")
    if opts.get("return_z_loss"):
        _assert_close(out[1], ref[1], atol, rtol, "z_loss")
    if check_grad and opts.get("requires_grad", True):
        _assert_close(out[2], ref[2], atol, rtol, "grad")
    if opts.get("return_token_accuracy"):
        _assert_close(out[3], ref[3], atol, rtol, "token_accuracy")
    if opts.get("return_predicted_tokens"):
        assert torch.equal(out[4], ref[4]), "predicted_tokens mismatch vs Triton (exact int match expected)"


# =============================================================================
# A. Parity vs Triton — implemented branches (auto-skip while a feature is a stub)
# =============================================================================
@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("BT, V", [(4, 4096), (1024, 4096)], ids=["bt4", "bt1024"])
def test_ce_core_matches_triton(BT, V, reduction, dtype):
    """dtype x reduction x BT core matrix. BT=4 stresses the per-row-CTA small-BT path."""
    set_seed()
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    _assert_ce_parity(base, target, dtype, reduction=reduction)


@cuda_required
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ce_core_matches_torch_reference(reduction):
    """Independent anchor: cutedsl fp32 loss vs torch.nn.functional.cross_entropy."""
    set_seed()
    BT, V = 256, 4096
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    out = _run_or_skip(
        lambda: _run(_cutedsl_ce(), base, target, torch.float32, reduction=reduction, requires_grad=False)
    )
    out_loss = out[0]
    ref = F.cross_entropy(base, target, reduction=reduction).detach().float()
    _assert_close(out_loss, ref, 1e-3, 1e-3, f"loss vs torch ({reduction})")


@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ce_ignore_index_matches_triton(reduction, dtype):
    """~1/3 of rows ignored — exercises the per-row ignore mask + n_non_ignore scaling."""
    set_seed()
    BT, V, ignore_index = 512, 4096, -100
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    target[torch.randperm(BT, device="cuda")[: BT // 3]] = ignore_index
    _assert_ce_parity(base, target, dtype, reduction=reduction, ignore_index=ignore_index)


@cuda_required
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_ce_all_ignored_matches_triton(reduction):
    """Every row ignored: n_non_ignore == 0 -> inv_n = 1.0, loss & grad must be exactly 0."""
    set_seed()
    BT, V, ignore_index = 64, 4096, -100
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.full((BT,), ignore_index, device="cuda", dtype=torch.long)
    out = _run_or_skip(
        lambda: _run(_cutedsl_ce(), base, target, torch.float32, reduction=reduction, ignore_index=ignore_index)
    )
    ref = _run(_triton_ce(), base, target, torch.float32, reduction=reduction, ignore_index=ignore_index)
    _assert_close(out[0], ref[0], 1e-5, 1e-4, "loss(all-ignored)")
    assert torch.equal(out[0], torch.zeros_like(out[0])), "all-ignored loss must be exactly 0"
    assert out[2] is not None and out[2].abs().max().item() == 0.0, "all-ignored grad must be exactly 0"


@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("return_z_loss", [True, False], ids=["zret", "znoret"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ce_z_loss_matches_triton(reduction, return_z_loss, dtype):
    """lse_square_scale != 0 routes to the streaming kernel; compares loss AND z_loss."""
    set_seed()
    BT, V = 512, 4096
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    _assert_ce_parity(base, target, dtype, reduction=reduction, lse_square_scale=1e-4, return_z_loss=return_z_loss)


@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ce_softcap_matches_triton(reduction, dtype):
    """softcap = 30.0 -> softcap*tanh(x/softcap) applied to logits (streaming path)."""
    set_seed()
    BT, V = 512, 4096
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    _assert_ce_parity(base, target, dtype, reduction=reduction, softcap=30.0)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_ce_combined_features_matches_triton(dtype):
    """ignore_index + z_loss + softcap together (the 'all-on' combo)."""
    set_seed()
    BT, V, ignore_index = 512, 4096, -100
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    target[torch.randperm(BT, device="cuda")[: BT // 4]] = ignore_index
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
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_ce_forward_only_matches_triton(dtype):
    """requires_grad=False -> the has_grad=False compile path (no in-place grad write)."""
    set_seed()
    BT, V = 256, 4096
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    _assert_ce_parity(base, target, dtype, reduction="mean", requires_grad=False, check_grad=False)


@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_ce_not_last_layer_grad_matches_triton(reduction, dtype):
    """grad_output != 1.0 (scalar): exercises the `_input * grad_output` backward branch."""
    set_seed()
    BT, V = 256, 4096
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    _assert_ce_parity(base, target, dtype, reduction=reduction, grad_scale=2.0)


@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_ce_reduction_none_perrow_grad_matches_triton(dtype):
    """reduction='none' with a per-row grad_output vector: the ndim>0 backward branch."""
    set_seed()
    BT, V = 256, 4096
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    grad_vec = torch.rand(BT, device="cuda", dtype=torch.float32)  # SAME vector fed to both backends
    _assert_ce_parity(base, target, dtype, reduction="none", grad_vec=grad_vec)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_ce_noncontiguous_input_matches_triton(dtype):
    """Non-contiguous logits -> the host-side `.contiguous()` branch. Forward-only loss parity."""
    set_seed()
    BT, V = 128, 4096
    base = torch.randn(V, BT, device="cuda", dtype=torch.float32).t()  # (BT, V), stride(-1)=BT != 1
    assert not base.is_contiguous()
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    _assert_ce_parity(base, target, dtype, reduction="mean", requires_grad=False, check_grad=False)


# =============================================================================
# B. Warp-count selection (white-box on the compile cache): the streaming kernel bakes
#    num_warps per dtype, mirroring the Triton CE Blackwell convention — 8 warps for 2-byte
#    dtypes (instruction-issue-bound), 32 for fp32 (bandwidth-bound). num_warps is the last
#    element of the compile-cache key.
# =============================================================================
@cuda_required
@pytest.mark.parametrize(
    "dtype, expected_warps",
    [(torch.bfloat16, 8), (torch.float16, 8), (torch.float32, 32)],
    ids=["bf16", "fp16", "fp32"],
)
def test_num_warps_matches_dtype_convention(dtype, expected_warps):
    mod = _cutedsl_ce_module()
    base = torch.randn(64, 4096, device="cuda", dtype=torch.float32)
    target = torch.randint(0, 4096, (64,), device="cuda", dtype=torch.long)
    mod._compile_cache.clear()
    _run_or_skip(lambda: _run(_cutedsl_ce(), base, target, dtype, reduction="mean", requires_grad=False))
    warps = {k[-1] for k in mod._compile_cache}
    assert warps == {expected_warps}, f"expected num_warps={expected_warps} for {dtype}, got cache keys {warps}"


# =============================================================================
# C. Parity vs Triton — class weight, label_smoothing, and the argmax aux outputs.
#    (Each was a NotImplementedError scope guard until the feature landed; now a real
#    parity gate. If a feature is ever re-stubbed, its test auto-skips, not fails.)
# =============================================================================
def _basic_inputs(BT=64, V=4096):
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    return base, target


def _rand_weight(V):
    """Strictly-positive fp32 class-weight vector (cutedsl upcasts weight to fp32 internally)."""
    return torch.rand(V, device="cuda", dtype=torch.float32) + 0.5


def _scatter_ignored(target, frac, ignore_index=-100):
    BT = target.shape[0]
    target[torch.randperm(BT, device=target.device)[: int(BT * frac)]] = ignore_index


@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ce_class_weight_matches_triton(reduction, dtype):
    """Per-class weight: mean denominator becomes sum_non_ignore_weight; grad scales by weight_y."""
    set_seed()
    BT, V = 512, 4096
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    _scatter_ignored(target, 0.25)
    _assert_ce_parity(base, target, dtype, reduction=reduction, weight=_rand_weight(V))


@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("label_smoothing", [0.1, 0.3], ids=["ls0.1", "ls0.3"])
def test_ce_label_smoothing_matches_triton(label_smoothing, reduction, dtype):
    """label_smoothing (no weight): the scaled_x_sum fwd reduction + additive -eps grad term."""
    set_seed()
    BT, V = 512, 4096
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    _scatter_ignored(target, 0.25)
    _assert_ce_parity(base, target, dtype, reduction=reduction, label_smoothing=label_smoothing)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ce_weight_and_label_smoothing_matches_triton(reduction, dtype):
    """The coupled path: weighted scaled_x_sum + weighted dloss_smooth (per-column weight stream)."""
    set_seed()
    BT, V = 512, 4096
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    _scatter_ignored(target, 0.25)
    _assert_ce_parity(base, target, dtype, reduction=reduction, weight=_rand_weight(V), label_smoothing=0.1)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_ce_all_features_matches_triton(dtype):
    """Everything at once: weight + label_smoothing + z_loss + softcap + ignore_index (loss+grad+z)."""
    set_seed()
    BT, V = 512, 4096
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    _scatter_ignored(target, 0.25)
    _assert_ce_parity(
        base,
        target,
        dtype,
        reduction="mean",
        weight=_rand_weight(V),
        label_smoothing=0.1,
        lse_square_scale=1e-4,
        softcap=30.0,
        return_z_loss=True,
    )


@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ce_token_accuracy_matches_triton(reduction, dtype):
    """token_accuracy = per-row (argmax == target), reduced to the non-ignored mean for mean/sum.

    Force ~half the rows to predict correctly (boost the target logit) so the acc==1.0 path
    is exercised, not just the trivial all-zero case random targets would give.
    """
    set_seed()
    BT, V = 256, 4096
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    for i in range(BT // 2):
        base[i, target[i]] = 50.0
    _scatter_ignored(target, 0.125)
    _assert_ce_parity(base, target, dtype, reduction=reduction, return_token_accuracy=True)


@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_ce_predicted_tokens_matches_triton(dtype):
    """predicted_tokens = per-row argmax column (int64), -1 for ignored rows. Exact int parity."""
    set_seed()
    BT, V = 256, 4096
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    _scatter_ignored(target, 0.125)
    # predicted_tokens is independent of reduction; forward-only is enough.
    _assert_ce_parity(
        base, target, dtype, reduction="none", requires_grad=False, check_grad=False, return_predicted_tokens=True
    )


# =============================================================================
# D. Input validation — host-side asserts.
# =============================================================================
@cuda_required
def test_vocab_not_multiple_of_vec_raises():
    """fp32 needs V % 4 == 0 (128-bit vectorized loads); 4094 % 4 == 2 must be rejected."""
    base, target = _basic_inputs(BT=8, V=4094)
    with pytest.raises(AssertionError):
        _apply(_cutedsl_ce(), base, target)


@cuda_required
def test_target_out_of_bounds_raises():
    """A non-ignored target >= V must be rejected."""
    base, target = _basic_inputs(BT=8, V=4096)
    target[0] = base.shape[1]  # == V, out of [0, V)
    with pytest.raises(AssertionError):
        _apply(_cutedsl_ce(), base, target)


@cuda_required
def test_target_negative_non_ignore_raises():
    """A negative target that isn't ignore_index must be rejected (min >= 0 check)."""
    base, target = _basic_inputs(BT=8, V=4096)
    target[0] = -5  # negative but != ignore_index(-100)
    with pytest.raises(AssertionError):
        _apply(_cutedsl_ce(), base, target)


# =============================================================================
# E. Dispatch / swap wiring — the only thing the registry layer adds over the
#    parity tests above: does LIGER_KERNEL_IMPL=cutedsl actually rewire the public
#    LigerCrossEntropyFunction to the cutedsl implementation? (Triton is the default
#    impl, so it needs no such check.) Subprocess because the swap happens at import.
# =============================================================================
@cuda_required
@pytest.mark.skipif(
    os.environ.get("LIGER_KERNEL_IMPL", "").strip().lower() != "cutedsl",
    reason="cutedsl selection test requires LIGER_KERNEL_IMPL=cutedsl",
)
def test_liger_kernel_impl_cutedsl_selects_cutedsl_ce():
    repo_root = Path(__file__).resolve().parents[2]
    pythonpath = os.pathsep.join([str(repo_root / "src"), str(repo_root), os.environ.get("PYTHONPATH", "")])
    env = {**os.environ, "LIGER_KERNEL_IMPL": "cutedsl", "PYTHONPATH": pythonpath}
    script = textwrap.dedent(
        """
        from liger_kernel.transformers.cross_entropy import LigerCrossEntropyFunction

        prefix = "liger_kernel.ops.cutedsl."
        mod = LigerCrossEntropyFunction.__module__
        if not mod.startswith(prefix):
            raise AssertionError(f"Expected cutedsl LigerCrossEntropyFunction from {prefix}, got {mod}")
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, env=env, cwd=repo_root)


# =============================================================================
# F. Coverage parity with the Triton suite — production vocab/shape, scalar-grad
#    combined with other params, and the functional/structured-output wrapper.
#    The Triton suite parametrizes every correctness test at V=32000 (+ odd shapes);
#    the cutedsl kernel's tiling is V-dependent, so these close the gaps a V=4096-only
#    matrix leaves — most importantly the tail-predication path.
# =============================================================================
# (name, feature-kwargs) — each exercises a distinct loss/grad path at production vocab.
_BIG_FEATURES = [
    ("core", {}),
    ("weight_ignore", {"weight_": True, "ignore": True}),
    ("smoothing_ignore", {"label_smoothing": 0.1, "ignore": True}),
    (
        "all",
        {
            "weight_": True,
            "label_smoothing": 0.1,
            "lse_square_scale": 1e-4,
            "softcap": 30.0,
            "ignore": True,
            "return_z_loss": True,
        },
    ),
]


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("BT, V", [(1269, 32000), (256, 32000)], ids=["1269x32000", "256x32000"])
@pytest.mark.parametrize("name, feats", _BIG_FEATURES, ids=[f[0] for f in _BIG_FEATURES])
def test_ce_production_vocab_matches_triton(name, feats, BT, V, dtype):
    """V=32000 (llama vocab) + the odd (3,423,32000)->1269 flat shape.

    This is the only V here that leaves a PARTIAL last tile (num_vec % 256 != 0), so it is
    what actually exercises the tail-predication path — V=4096/36864 divide the 256-thread
    tile grid evenly and never predicate a thread off. Also production scale (many tiles).
    """
    set_seed()
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    if feats.get("ignore"):
        _scatter_ignored(target, 0.25)
    opts = {k: v for k, v in feats.items() if k in ("label_smoothing", "lse_square_scale", "softcap", "return_z_loss")}
    if feats.get("weight_"):
        opts["weight"] = _rand_weight(V)
    _assert_ce_parity(base, target, dtype, reduction="mean", **opts)


@cuda_required
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_ce_not_last_layer_with_other_params_matches_triton(reduction, dtype):
    """Scalar grad_output != 1 combined with z_loss + softcap + label_smoothing + ignore_index
    (Triton's not_last_layer_with_other_params): backward scaling must compose with every
    forward feature term, not just plain CE (which test_ce_not_last_layer_grad covers)."""
    set_seed()
    BT, V = 512, 4096
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)
    _scatter_ignored(target, 0.25)
    _assert_ce_parity(
        base,
        target,
        dtype,
        reduction=reduction,
        grad_scale=2.0,
        lse_square_scale=1e-4,
        softcap=30.0,
        label_smoothing=0.1,
        return_z_loss=True,
    )


@cuda_required
@pytest.mark.parametrize("return_z_loss", [True, False], ids=["z1", "z0"])
@pytest.mark.parametrize("return_token_accuracy", [True, False], ids=["acc1", "acc0"])
@pytest.mark.parametrize("return_predicted_tokens", [True, False], ids=["pred1", "pred0"])
def test_ce_functional_structured_output_matches_triton(
    return_z_loss, return_token_accuracy, return_predicted_tokens, monkeypatch
):
    """End-to-end through ``liger_cross_entropy(...)``: the cutedsl Function must drive the
    bare-loss path and the ``CrossEntropyOutput`` dataclass identically to Triton across all 8
    (z_loss, token_accuracy, predicted_tokens) flag combinations (None where a flag is off).
    Monkeypatches the wrapper's Function so the real backend-agnostic wrapper code runs.
    """
    import liger_kernel.transformers.functional as fmod

    set_seed()
    BT, V = 256, 4096
    base = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda", dtype=torch.long)

    def via_wrapper(fn):
        monkeypatch.setattr(fmod, "LigerCrossEntropyFunction", fn)
        return fmod.liger_cross_entropy(
            base.clone(),
            target,
            reduction="mean",
            lse_square_scale=1e-4,  # make z_loss non-trivial when requested
            return_z_loss=return_z_loss,
            return_token_accuracy=return_token_accuracy,
            return_predicted_tokens=return_predicted_tokens,
        )

    out_c = _run_or_skip(lambda: via_wrapper(_cutedsl_ce()))
    out_t = via_wrapper(_triton_ce())

    if not (return_z_loss or return_token_accuracy or return_predicted_tokens):
        # bare-loss path: liger_cross_entropy returns the loss tensor itself, not the dataclass.
        assert not hasattr(out_c, "loss"), "expected a bare loss tensor when no flags are set"
        _assert_close(out_c.detach().float(), out_t.detach().float(), 1e-4, 1e-3, "loss (bare)")
        return

    _assert_close(out_c.loss.detach().float(), out_t.loss.detach().float(), 1e-4, 1e-3, "loss")
    for field, flag in (
        ("z_loss", return_z_loss),
        ("token_accuracy", return_token_accuracy),
        ("predicted_tokens", return_predicted_tokens),
    ):
        c, t = getattr(out_c, field), getattr(out_t, field)
        if not flag:
            assert c is None and t is None, f"{field} must be None when its flag is off"
        elif field == "predicted_tokens":
            assert torch.equal(c, t), "predicted_tokens mismatch vs Triton"
        else:
            _assert_close(c.detach().float(), t.detach().float(), 1e-4, 1e-3, field)
