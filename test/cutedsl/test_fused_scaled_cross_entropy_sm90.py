"""Correctness tests for the Hopper fused scaled cross entropy CuTe DSL kernel."""

import os
import subprocess
import sys
import textwrap

from pathlib import Path

import pytest
import torch

from test.utils import assert_verbose_allclose
from test.utils import set_seed

# Every test in this module launches SM90 CuTe DSL kernels, so gate the whole
# file at collection time: on a CPU-only, non-NVIDIA or non-Hopper host nothing
# here is collectable.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="CuTe DSL fused scaled cross entropy requires a Hopper (SM90) GPU",
)


def _sm90_module():
    try:
        import liger_kernel.ops.cutedsl.ops.fused_scaled_cross_entropy_sm90 as module
    except ImportError as exc:
        pytest.skip(f"CuTe DSL backend is not installed: {exc}")
    return module


def _frontend_function():
    from liger_kernel.ops import LigerFusedLinearScaledCrossEntropyFunction

    return LigerFusedLinearScaledCrossEntropyFunction


def _frontend_module():
    import liger_kernel.ops.fused_linear_scaled_cross_entropy as module

    return module


def _reference_nll(x, weight, target, ignore_index, temperature=1.0):
    logits = (x.float() @ weight.float().t()) / temperature
    valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0)
    lse = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(1, safe_target[:, None]).squeeze(1)
    return torch.where(valid, lse - target_logits, torch.zeros_like(lse))


def _reference_nll_entropy(x, weight, target, ignore_index, temperature=1.0):
    logits = (x.float() @ weight.float().t()) / temperature
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0)
    nll = -log_probs.gather(1, safe_target[:, None]).squeeze(1)
    zeros = torch.zeros_like(nll)
    return torch.where(valid, nll, zeros), torch.where(valid, entropy, zeros)


def _verl_reference(x, weight, target, grad_nll, grad_entropy, temperature):
    """Dependency-free equivalent of Verl's experimental fused PPO function."""
    logits = ((x @ weight.t()) / temperature).float()
    probs = logits.softmax(dim=-1)
    log_probs = logits.log_softmax(dim=-1)
    token_log_probs = log_probs.gather(-1, target[:, None]).squeeze(-1)
    entropy = (torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)).to(x.dtype)

    one_hot = torch.zeros_like(logits).scatter_(-1, target[:, None], 1)
    dlogits = (-grad_nll).float()[:, None] * (one_hot - probs)
    dlogits += probs * (log_probs + entropy.float()[:, None]) * (-grad_entropy.float()[:, None])
    dlogits = dlogits.to(x.dtype) / temperature
    return -token_log_probs, entropy, dlogits @ weight, dlogits.t() @ x


def _make_inputs(bt, hidden_size, vocab_size, ignore_stride=None):
    x = torch.randn(bt, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.25
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16) / hidden_size**0.5
    target = torch.randint(vocab_size, (bt,), device="cuda", dtype=torch.long)
    if ignore_stride is not None:
        target[::ignore_stride] = -100
    return x, weight, target


def _run_liger(
    x,
    weight,
    target,
    grad_output,
    ignore_index=-100,
    temperature=1.0,
    m_tiles_per_cluster=1,
):
    function = _frontend_function()
    x = x.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    nll = function.apply(
        x,
        weight,
        target,
        temperature,
        ignore_index,
        m_tiles_per_cluster,
        False,
    )
    if grad_output is not None:
        nll.backward(grad_output)
        return nll.detach(), x.grad.detach(), weight.grad.detach()
    return nll.detach(), None, None


def _run_reference(x, weight, target, grad_output, ignore_index=-100, temperature=1.0):
    x = x.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    nll = _reference_nll(x, weight, target, ignore_index, temperature)
    if grad_output is not None:
        nll.backward(grad_output)
        return nll.detach(), x.grad.detach(), weight.grad.detach()
    return nll.detach(), None, None


@pytest.mark.parametrize("shape", [(256, 128, 512), (193, 70, 517)], ids=["aligned", "ragged"])
@pytest.mark.parametrize("m_tiles_per_cluster", [1, 2], ids=["mtiles1", "mtiles2"])
def test_sm90_per_token_nll_and_grads_match_torch(shape, m_tiles_per_cluster):
    set_seed(42)
    bt, hidden_size, vocab_size = shape
    x, weight, target = _make_inputs(bt, hidden_size, vocab_size, ignore_stride=7)
    grad_output = torch.rand(bt, device="cuda", dtype=torch.float32) + 0.25

    actual = _run_liger(
        x,
        weight,
        target,
        grad_output,
        m_tiles_per_cluster=m_tiles_per_cluster,
    )
    expected = _run_reference(x, weight, target, grad_output)

    assert actual[0].shape == (bt,)
    assert actual[0].dtype == torch.float32
    assert actual[1].shape == x.shape and actual[1].dtype == x.dtype
    assert actual[2].shape == weight.shape and actual[2].dtype == weight.dtype
    assert_verbose_allclose(actual[0].float(), expected[0].float(), atol=5e-2, rtol=5e-2)
    assert_verbose_allclose(actual[1].float(), expected[1].float(), atol=2e-2, rtol=1e-1)
    assert_verbose_allclose(actual[2].float(), expected[2].float(), atol=2e-2, rtol=1e-1)


@pytest.mark.parametrize("temperature", [0.7, 1.5])
def test_sm90_temperature_matches_torch(temperature):
    set_seed(43)
    x, weight, target = _make_inputs(193, 70, 517, ignore_stride=7)
    grad_output = torch.rand(193, device="cuda", dtype=torch.float32) + 0.25

    actual = _run_liger(x, weight, target, grad_output, temperature=temperature)
    expected = _run_reference(x, weight, target, grad_output, temperature=temperature)

    assert_verbose_allclose(actual[0].float(), expected[0].float(), atol=5e-2, rtol=5e-2)
    assert_verbose_allclose(actual[1].float(), expected[1].float(), atol=2e-2, rtol=1e-1)
    assert_verbose_allclose(actual[2].float(), expected[2].float(), atol=2e-2, rtol=1e-1)


@pytest.mark.parametrize("m_tiles_per_cluster", [1, 2])
def test_sm90_differentiable_entropy_matches_torch(m_tiles_per_cluster):
    set_seed(44)
    x, weight, target = _make_inputs(193, 70, 517, ignore_stride=7)
    temperature = 0.8
    grad_nll = torch.rand(193, device="cuda", dtype=torch.float32) + 0.25
    grad_entropy = torch.rand(193, device="cuda", dtype=torch.float32) - 0.5

    function = _frontend_function()
    xa = x.detach().clone().requires_grad_(True)
    wa = weight.detach().clone().requires_grad_(True)
    nll, entropy = function.apply(
        xa,
        wa,
        target,
        temperature,
        -100,
        m_tiles_per_cluster,
        True,
    )
    torch.autograd.backward((nll, entropy), (grad_nll, grad_entropy.to(entropy.dtype)))

    xr = x.detach().clone().requires_grad_(True)
    wr = weight.detach().clone().requires_grad_(True)
    ref_nll, ref_entropy = _reference_nll_entropy(xr, wr, target, -100, temperature)
    torch.autograd.backward((ref_nll, ref_entropy), (grad_nll, grad_entropy))

    assert entropy.dtype == torch.bfloat16
    assert entropy[target == -100].count_nonzero().item() == 0
    assert_verbose_allclose(nll.float(), ref_nll.float(), atol=5e-2, rtol=5e-2)
    assert_verbose_allclose(entropy.float(), ref_entropy.float(), atol=5e-2, rtol=5e-2)
    assert_verbose_allclose(xa.grad.float(), xr.grad.float(), atol=3e-2, rtol=1.5e-1)
    assert_verbose_allclose(wa.grad.float(), wr.grad.float(), atol=3e-2, rtol=1.5e-1)


def test_sm90_entropy_only_backward_matches_torch():
    set_seed(144)
    x, weight, target = _make_inputs(193, 70, 517, ignore_stride=7)
    grad_entropy = torch.rand(193, device="cuda", dtype=torch.float32) - 0.5

    function = _frontend_function()
    actual_x = x.detach().clone().requires_grad_(True)
    actual_weight = weight.detach().clone().requires_grad_(True)
    _, actual_entropy = function.apply(
        actual_x,
        actual_weight,
        target,
        0.8,
        -100,
        1,
        True,
    )
    actual_entropy.backward(grad_entropy.to(actual_entropy.dtype))

    expected_x = x.detach().clone().requires_grad_(True)
    expected_weight = weight.detach().clone().requires_grad_(True)
    _, expected_entropy = _reference_nll_entropy(expected_x, expected_weight, target, -100, 0.8)
    expected_entropy.backward(grad_entropy)

    assert_verbose_allclose(actual_x.grad.float(), expected_x.grad.float(), atol=3e-2, rtol=1.5e-1)
    assert_verbose_allclose(actual_weight.grad.float(), expected_weight.grad.float(), atol=3e-2, rtol=1.5e-1)


def test_sm90_matches_verl_fused_ppo_formulas():
    set_seed(45)
    x, weight, target = _make_inputs(193, 70, 517)
    temperature = 0.8
    grad_nll = torch.rand(193, device="cuda", dtype=torch.float32) + 0.25
    grad_entropy = torch.rand(193, device="cuda", dtype=torch.float32) - 0.5

    function = _frontend_function()
    xa = x.detach().clone().requires_grad_(True)
    wa = weight.detach().clone().requires_grad_(True)
    nll, entropy = function.apply(
        xa,
        wa,
        target,
        temperature,
        -100,
        1,
        True,
    )
    torch.autograd.backward((nll, entropy), (grad_nll, grad_entropy.to(entropy.dtype)))

    ref_nll, ref_entropy, ref_dx, ref_dw = _verl_reference(
        x,
        weight,
        target,
        grad_nll,
        grad_entropy,
        temperature,
    )
    assert_verbose_allclose(nll.float(), ref_nll.float(), atol=5e-2, rtol=5e-2)
    assert_verbose_allclose(entropy.float(), ref_entropy.float(), atol=5e-2, rtol=5e-2)
    assert_verbose_allclose(xa.grad.float(), ref_dx.float(), atol=3e-2, rtol=1.5e-1)
    assert_verbose_allclose(wa.grad.float(), ref_dw.float(), atol=3e-2, rtol=1.5e-1)


def test_sm90_frontend_matches_fused_linear_ppo_fallback():
    set_seed(46)
    x, weight, target = _make_inputs(521, 70, 517, ignore_stride=11)
    temperature = 0.8
    grad_nll = torch.rand(521, device="cuda", dtype=torch.float32) + 0.25
    grad_entropy = torch.rand(521, device="cuda", dtype=torch.float32) - 0.5
    frontend = _frontend_module()

    sm90_input = x.detach().clone().requires_grad_(True)
    sm90_weight = weight.detach().clone().requires_grad_(True)
    sm90_nll, sm90_entropy = frontend.LigerFusedLinearScaledCrossEntropyFunction.apply(
        sm90_input,
        sm90_weight,
        target,
        temperature,
        -100,
        1,
        True,
    )
    torch.autograd.backward((sm90_nll, sm90_entropy), (grad_nll, grad_entropy.to(sm90_entropy.dtype)))

    fallback_input = x.detach().clone().requires_grad_(True)
    fallback_weight = weight.detach().clone().requires_grad_(True)
    fallback_nll, fallback_entropy = frontend._FusedLinearPPOFallbackFunction.apply(
        fallback_input,
        fallback_weight,
        target,
        temperature,
        -100,
        True,
    )
    torch.autograd.backward(
        (fallback_nll, fallback_entropy),
        (grad_nll, grad_entropy.to(fallback_entropy.dtype)),
    )

    assert_verbose_allclose(sm90_nll.float(), fallback_nll.float(), atol=5e-2, rtol=5e-2)
    assert_verbose_allclose(sm90_entropy.float(), fallback_entropy.float(), atol=5e-2, rtol=5e-2)
    assert_verbose_allclose(sm90_input.grad.float(), fallback_input.grad.float(), atol=3e-2, rtol=1.5e-1)
    assert_verbose_allclose(sm90_weight.grad.float(), fallback_weight.grad.float(), atol=3e-2, rtol=1.5e-1)


def test_sm90_ignored_entropy_rows_stay_zero_with_large_logits():
    set_seed(47)
    x = (torch.randn(128, 64, device="cuda", dtype=torch.bfloat16) * 20).requires_grad_(True)
    weight = (torch.randn(256, 64, device="cuda", dtype=torch.bfloat16) * 20).requires_grad_(True)
    target = torch.full((128,), -100, device="cuda", dtype=torch.long)

    module = _sm90_module()
    nll, entropy = module.LigerFusedScaledCrossEntropySM90Function.apply(
        x,
        weight,
        target,
        0.1,
        -100,
        1,
        True,
    )
    torch.autograd.backward((nll, entropy), (torch.ones_like(nll), torch.ones_like(entropy)))

    assert nll.count_nonzero().item() == 0
    assert entropy.count_nonzero().item() == 0
    assert torch.isfinite(x.grad).all() and x.grad.count_nonzero().item() == 0
    assert torch.isfinite(weight.grad).all() and weight.grad.count_nonzero().item() == 0


def test_sm90_ignore_index_rows_are_zero():
    set_seed(0)
    bt, hidden_size, vocab_size = 192, 96, 257
    x, weight, target = _make_inputs(bt, hidden_size, vocab_size, ignore_stride=5)
    grad_output = torch.ones(bt, device="cuda", dtype=torch.float32)

    nll, dx, _ = _run_liger(x, weight, target, grad_output)
    ignored = target == -100

    assert nll[ignored].count_nonzero().item() == 0
    assert nll[~ignored].min().item() > 0.0
    assert dx[ignored].count_nonzero().item() == 0


def test_sm90_zero_grad_rows_produce_zero_dx():
    set_seed(1)
    bt, hidden_size, vocab_size = 192, 96, 257
    x, weight, target = _make_inputs(bt, hidden_size, vocab_size, ignore_stride=11)
    grad_output = torch.zeros(bt, device="cuda", dtype=torch.float32)
    grad_output[::5] = torch.rand((bt + 4) // 5, device="cuda", dtype=torch.float32) + 0.25

    actual = _run_liger(x, weight, target, grad_output)
    expected = _run_reference(x, weight, target, grad_output)

    assert actual[1][grad_output == 0].count_nonzero().item() == 0
    assert_verbose_allclose(actual[1].float(), expected[1].float(), atol=2e-2, rtol=1e-1)
    assert_verbose_allclose(actual[2].float(), expected[2].float(), atol=2e-2, rtol=1e-1)


def test_sm90_all_ignored_is_zero():
    set_seed(2)
    bt, hidden_size, vocab_size = 128, 64, 256
    x, weight, _ = _make_inputs(bt, hidden_size, vocab_size)
    target = torch.full((bt,), -100, device="cuda", dtype=torch.long)
    grad_output = torch.ones(bt, device="cuda", dtype=torch.float32)

    nll, dx, dw = _run_liger(x, weight, target, grad_output)

    assert nll.count_nonzero().item() == 0
    assert dx.count_nonzero().item() == 0
    assert dw.count_nonzero().item() == 0


def test_sm90_many_wave_dw_matches_torch_within_bf16_accumulation_error():
    """Multi-wave BF16 dW accumulation must track the FP32 torch reference."""
    set_seed(17)
    bt, hidden_size, vocab_size = 2048, 256, 512
    x, weight, target = _make_inputs(bt, hidden_size, vocab_size, ignore_stride=11)
    grad_output = torch.rand(bt, device="cuda", dtype=torch.float32) + 0.25

    waved = _run_liger(x, weight, target, grad_output)  # two 1024-token waves
    expected = _run_reference(x, weight, target, grad_output)[2].float()

    # Absolute tolerance is relaxed because every
    # wave's contribution is rounded to BF16 (~2^-8 relative) before the add.
    assert_verbose_allclose(waved[2].float(), expected, atol=5e-2, rtol=2e-1)

    scale = expected.abs().mean()
    wave_err = (waved[2].float() - expected).abs().mean() / scale
    assert wave_err < 2e-2, wave_err


def test_sm90_backward_uses_one_fused_kernel(monkeypatch):
    """Backward dispatches one fused kernel with the fixed wave cap."""
    module = _sm90_module()
    set_seed(12)
    bt, hidden_size, vocab_size = 512, 128, 256
    x, weight, target = _make_inputs(bt, hidden_size, vocab_size)
    x = x.requires_grad_(True)
    weight = weight.requires_grad_(True)
    grad_output = torch.rand(bt, device="cuda", dtype=torch.float32) + 0.25

    calls = []
    original = module.fused_backward_one_kernel

    def wrapper(*args, **kwargs):
        calls.append(kwargs["config"].m_tiles_per_wave)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "fused_backward_one_kernel", wrapper)

    module.LigerFusedScaledCrossEntropySM90Function.apply(x, weight, target).backward(grad_output)
    assert calls == [4]


@pytest.mark.parametrize("m_tiles_per_cluster", [1, 3], ids=["mtiles1", "mtiles3"])
@pytest.mark.parametrize("return_entropy", [False, True], ids=["nll", "entropy"])
def test_sm90_m_tiles_values_route_fragment_forward(monkeypatch, m_tiles_per_cluster, return_entropy):
    module = _sm90_module()
    calls = []

    def fake_fragment(x, weight, target, temperature, ignore_index, return_entropy, config):
        calls.append((temperature, ignore_index, return_entropy, config))
        zeros = torch.zeros(x.shape[0], device=x.device)
        entropy = zeros.to(torch.bfloat16) if return_entropy else None
        return zeros, entropy, zeros

    monkeypatch.setattr(module, "scaled_ce_forward_fragment", fake_fragment)

    x = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
    target = torch.zeros(4, device="cuda", dtype=torch.long)
    module.fused_scaled_cross_entropy_forward(
        x,
        weight,
        target,
        1.0,
        -100,
        m_tiles_per_cluster,
        return_entropy,
    )

    assert calls == [(1.0, -100, return_entropy, None)]


def test_sm90_long_sequence_forward_tuning_table():
    module = _sm90_module()

    assert module._select_forward_config(8192, 4096, 131072, False) is None
    assert module._select_forward_config(16384, 4096, 131072, False).split_n == 4
    assert module._select_forward_config(16384, 4096, 131072, True).split_n == 2
    assert module._select_forward_config(32768, 4096, 131072, False).split_n == 2
    assert module._select_forward_config(32768, 4096, 131072, True).split_n == 2
    assert module._select_forward_config(32768, 4096, 65536, False) is None


def test_sm90_rejects_scalar_and_mismatched_grad_output():
    set_seed(3)
    module = _sm90_module()
    bt, hidden_size, vocab_size = 64, 64, 128
    x, weight, target = _make_inputs(bt, hidden_size, vocab_size)

    _, _, lse, x_padded, weight_padded, hidden = module.fused_scaled_cross_entropy_forward(
        x, weight, target, 1.0, -100, 1
    )
    for bad_grad in (
        torch.tensor(1.0, device="cuda"),
        torch.ones(bt - 1, device="cuda"),
        torch.ones(bt, 1, device="cuda"),
    ):
        with pytest.raises(ValueError):
            module.fused_scaled_cross_entropy_backward(
                bad_grad, x_padded, weight_padded, target, lse, 1.0, -100, hidden
            )
    with pytest.raises(TypeError):
        module.fused_scaled_cross_entropy_backward(1.0, x_padded, weight_padded, target, lse, 1.0, -100, hidden)


def test_sm90_sum_reduction_composed_in_autograd():
    """``sum().backward()`` is the supported reduction path: autograd expands
    the scalar into the per-token vector of ones before the kernel sees it."""
    set_seed(4)
    bt, hidden_size, vocab_size = 128, 64, 256
    x, weight, target = _make_inputs(bt, hidden_size, vocab_size, ignore_stride=6)

    module = _sm90_module()
    xa = x.detach().clone().requires_grad_(True)
    wa = weight.detach().clone().requires_grad_(True)
    module.LigerFusedScaledCrossEntropySM90Function.apply(xa, wa, target).sum().backward()

    xr = x.detach().clone().requires_grad_(True)
    wr = weight.detach().clone().requires_grad_(True)
    _reference_nll(xr, wr, target, -100).sum().backward()

    assert_verbose_allclose(xa.grad.float(), xr.grad.float(), atol=2e-2, rtol=1e-1)
    assert_verbose_allclose(wa.grad.float(), wr.grad.float(), atol=2e-2, rtol=1e-1)


def test_sm90_rejects_bad_dtypes_shapes_and_options():
    module = _sm90_module()
    fn = module.LigerFusedScaledCrossEntropySM90Function
    x = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
    target = torch.zeros(4, device="cuda", dtype=torch.long)

    with pytest.raises(TypeError):
        fn.apply(x.to(torch.float16), weight.to(torch.float16), target)
    with pytest.raises(TypeError):
        fn.apply(x, weight, target.to(torch.int32))
    with pytest.raises(ValueError):
        fn.apply(x, weight, target[:2])
    with pytest.raises(ValueError):
        fn.apply(x, torch.randn(16, 8, device="cuda", dtype=torch.bfloat16), target)
    with pytest.raises(ValueError):
        fn.apply(x, weight, target, 0.0)
    with pytest.raises(TypeError):
        fn.apply(x, weight, target, "cold")
    with pytest.raises(ValueError):
        fn.apply(x, weight, target, float("inf"))
    with pytest.raises(ValueError):
        fn.apply(x, weight, target, 1.0, -100, 0)
    with pytest.raises(TypeError):
        fn.apply(x, weight, target, 1.0, -100, 1.5)
    with pytest.raises(TypeError):
        fn.apply(x, weight, target, 1.0, -100, 1, 2.0)


def test_sm90_rejects_out_of_range_targets():
    module = _sm90_module()
    fn = module.LigerFusedScaledCrossEntropySM90Function
    x = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
    target = torch.zeros(4, device="cuda", dtype=torch.long)
    target[1] = 16

    with pytest.raises(ValueError):
        fn.apply(x, weight, target)

    target[1] = -100
    fn.apply(x, weight, target)


def test_sm90_frontend_and_backend_exports_are_additive():
    _sm90_module()
    repo_root = Path(__file__).resolve().parents[2]
    pythonpath = os.pathsep.join([str(repo_root / "src"), str(repo_root), os.environ.get("PYTHONPATH", "")])
    env = {
        **os.environ,
        "CUTE_DSL_ARCH": "sm_90a",
        "LIGER_KERNEL_IMPL": "cutedsl",
        "PYTHONPATH": pythonpath,
    }
    script = textwrap.dedent(
        """
        import liger_kernel.ops.cutedsl.ops as cutedsl_ops
        from liger_kernel.ops import LigerFusedLinearScaledCrossEntropyFunction

        expected_frontend = "liger_kernel.ops.fused_linear_scaled_cross_entropy"
        actual_frontend = LigerFusedLinearScaledCrossEntropyFunction.__module__
        if actual_frontend != expected_frontend:
            raise AssertionError(f"Expected {expected_frontend}, got {actual_frontend}")
        expected_backend = "liger_kernel.ops.cutedsl.ops.fused_scaled_cross_entropy_sm90"
        actual_backend = cutedsl_ops.LigerFusedScaledCrossEntropySM90Function.__module__
        if actual_backend != expected_backend:
            raise AssertionError(f"Expected {expected_backend}, got {actual_backend}")
        for name in (
            "LigerFusedScaledCrossEntropySM90Function",
            "fused_scaled_cross_entropy_forward",
            "fused_scaled_cross_entropy_backward",
        ):
            if name not in cutedsl_ops.__all__:
                raise AssertionError(f"{name} missing from cutedsl ops __all__")
        if hasattr(cutedsl_ops, "LigerFusedLinearScaledCrossEntropyFunction"):
            raise AssertionError("root frontend must not be exported from cutedsl ops")
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, env=env, cwd=repo_root)
