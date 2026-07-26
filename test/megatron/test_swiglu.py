"""Unit tests for LigerMegatronSwiGLU.

Sections 1-5 deliberately do not import megatron-core; the wrapper's contract is verified
against a local reproduction of Megatron's own reference::

    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2

The parametrization style mirrors ``test/megatron/test_rms_norm.py``.

Section 6 is different, and deliberately so. Verifying only against a reimplementation
leaves one class of bug completely uncovered: anything where our *belief* about Megatron's
API is wrong. A signature change, a renamed module path, or a different dispatch condition
would leave every test above green while the integration silently breaks at runtime. Those
tests import the real megatron-core and skip cleanly when it is absent, so they cost
nothing in the default CI environment. Run them with::

    pip install megatron-core
    pytest test/megatron/test_swiglu.py -v
"""

import os

import pytest
import torch
import torch.nn.functional as F

from liger_kernel.megatron.swiglu import LigerMegatronSwiGLU
from liger_kernel.ops.swiglu import LigerFusedGateUpSiLUMulFunction
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.utils import infer_device
from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

device = infer_device()

set_seed(42)
torch.use_deterministic_algorithms(True)

if device == "cuda":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# ---------------------------------------------------------------------------
# References + helpers
# ---------------------------------------------------------------------------


def _megatron_swiglu_reference(y):
    """Byte-for-byte reproduction of megatron.core.fusions.fused_bias_swiglu.swiglu."""
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2


def _megatron_bias_swiglu_reference(y, bias):
    """Reproduction of Megatron's ``bias_swiglu`` — bias is added before the activation."""
    return _megatron_swiglu_reference(y + bias)


class _RecordingFallback:
    """Stand-in for Megatron's native ``bias_swiglu_impl``.

    Records every call so tests can assert *that* the fallback was taken and *with what*,
    and returns Megatron's reference result so numerical assertions still hold.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, input, bias, fp8_input_store=False, cpu_offload_input=False):
        self.calls.append(
            {
                "input": input,
                "bias": bias,
                "fp8_input_store": fp8_input_store,
                "cpu_offload_input": cpu_offload_input,
            }
        )
        if bias is not None:
            return _megatron_bias_swiglu_reference(input, bias)
        return _megatron_swiglu_reference(input)


# ---------------------------------------------------------------------------
# Forward + backward correctness
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize(
    "bs, sl, ffn",
    [
        (2, 128, 512),
        # Llama-7B-class intermediate size — the shape the kernel is tuned for.
        (2, 64, 11008),
        # weird shapes
        (5, 123, 123),
        # single row: exercises the degenerate n_rows == 1 grid.
        (1, 1, 64),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-2,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_correctness_3d(bs, sl, ffn, dtype, atol, rtol):
    """3D ``[seq, batch, 2*ffn]`` input — the shape Megatron's MLP actually passes."""
    _tensor = torch.randn(bs, sl, 2 * ffn, device=device, dtype=dtype)
    do = torch.randn(bs, sl, ffn, device=device, dtype=dtype)

    h1 = _tensor.clone().requires_grad_(True)
    h2 = _tensor.clone().requires_grad_(True)

    ref_o = _megatron_swiglu_reference(h1)
    ref_o.backward(do, retain_graph=True)

    liger_o = LigerMegatronSwiGLU()(h2)
    liger_o.backward(do, retain_graph=True)

    assert liger_o.shape == ref_o.shape
    assert_verbose_allclose(liger_o, ref_o, atol=atol, rtol=rtol)
    assert_verbose_allclose(h2.grad, h1.grad, atol=atol, rtol=rtol)


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize("tokens, ffn", [(256, 512), (128, 11008)])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-2,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_correctness_2d(tokens, ffn, dtype, atol, rtol):
    """2D ``[tokens, 2*ffn]`` input — Megatron's impl accepts both ranks."""
    _tensor = torch.randn(tokens, 2 * ffn, device=device, dtype=dtype)
    do = torch.randn(tokens, ffn, device=device, dtype=dtype)

    h1 = _tensor.clone().requires_grad_(True)
    h2 = _tensor.clone().requires_grad_(True)

    ref_o = _megatron_swiglu_reference(h1)
    ref_o.backward(do, retain_graph=True)

    liger_o = LigerMegatronSwiGLU()(h2)
    liger_o.backward(do, retain_graph=True)

    assert liger_o.shape == ref_o.shape
    assert_verbose_allclose(liger_o, ref_o, atol=atol, rtol=rtol)
    assert_verbose_allclose(h2.grad, h1.grad, atol=atol, rtol=rtol)


def test_input_is_not_mutated():
    """The fused gate-up backward must write to a fresh buffer, never into the saved input.

    Megatron keeps a reference to the fc1 output for activation recompute and CUDA-graph
    capture. The two-tensor ``LigerSiLUMulFunction`` writes gradients in place into the
    buffers it saved; the fused variant deliberately does not, mirroring
    ``LigerMegatronRMSNorm``'s ``in_place=False``.
    """
    x = torch.randn(4, 8, 64, device=device, dtype=torch.float32)
    h = x.clone().requires_grad_(True)
    before = h.detach().clone()

    out = LigerMegatronSwiGLU()(h)
    out.backward(torch.randn_like(out))

    assert torch.equal(h.detach(), before), "LigerMegatronSwiGLU mutated its input tensor"


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 0.0, 0.0),
        pytest.param(
            torch.bfloat16,
            0.0,
            0.0,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_fused_gate_up_matches_two_tensor_kernel(dtype, atol, rtol):
    """The fused kernel must be bit-identical to chunk + ``LigerSiLUMulFunction``.

    The fused path exists purely to remove the copies ``torch.chunk`` forces; it changes
    no arithmetic. Asserting exact equality (not allclose) pins that down, so a future
    edit to either kernel that silently diverges fails here.
    """
    tokens, ffn = 128, 512
    _tensor = torch.randn(tokens, 2 * ffn, device=device, dtype=dtype)
    do = torch.randn(tokens, ffn, device=device, dtype=dtype)

    h_fused = _tensor.clone().requires_grad_(True)
    fused_o = LigerFusedGateUpSiLUMulFunction.apply(h_fused)
    fused_o.backward(do)

    h_split = _tensor.clone().requires_grad_(True)
    gate, up = torch.chunk(h_split, 2, dim=-1)
    split_o = LigerSiLUMulFunction.apply(gate.contiguous(), up.contiguous())
    split_o.backward(do)

    assert torch.equal(fused_o, split_o)
    assert torch.equal(h_fused.grad, h_split.grad)


def test_in_place_backward_matches_out_of_place():
    """``in_place=True`` must produce identical gradients, only reusing the input buffer."""
    _tensor = torch.randn(64, 2 * 256, device=device, dtype=torch.float32)
    do = torch.randn(64, 256, device=device, dtype=torch.float32)

    h_out = _tensor.clone().requires_grad_(True)
    LigerMegatronSwiGLU()(h_out).backward(do)

    h_in = _tensor.clone().requires_grad_(True)
    LigerMegatronSwiGLU(in_place=True)(h_in).backward(do)

    assert torch.equal(h_in.grad, h_out.grad)


def test_in_place_backward_clobbers_the_saved_input():
    """Pin the documented cost of ``in_place=True``: the fc1 output is destroyed.

    This is not incidental -- it is the whole reason the flag is opt-in. If a future change
    makes it non-destructive the flag has become free and the default should be revisited.
    """
    x = torch.randn(8, 2 * 64, device=device, dtype=torch.float32)
    h = x.clone().requires_grad_(True)
    before = h.detach().clone()

    LigerMegatronSwiGLU(in_place=True)(h).backward(torch.randn(8, 64, device=device))

    assert not torch.equal(h.detach(), before)


def test_in_place_rejects_a_second_backward():
    """A repeated backward under ``in_place=True`` must raise, not return wrong gradients.

    Autograd cannot catch this on its own: its version counter is bumped by PyTorch
    in-place *ops*, and a Triton kernel writing through the data pointer does not touch it.
    Without an explicit guard the second backward recomputes silu from a buffer that now
    holds gradients and returns plausible-looking garbage (measured max abs error ~6.8 on a
    unit-normal input) with no error at all. This test pins the guard.
    """
    x = torch.randn(8, 2 * 64, device=device, dtype=torch.float32)
    do = torch.randn(8, 64, device=device)

    h = x.clone().requires_grad_(True)
    out = LigerMegatronSwiGLU(in_place=True)(h)
    out.backward(do, retain_graph=True)

    with pytest.raises(RuntimeError, match="in_place=True"):
        out.backward(do, retain_graph=True)


def test_default_path_supports_repeated_backward():
    """The out-of-place default must remain safe to backpropagate through twice."""
    x = torch.randn(8, 2 * 64, device=device, dtype=torch.float32)
    do = torch.randn(8, 64, device=device)

    h = x.clone().requires_grad_(True)
    out = LigerMegatronSwiGLU()(h)

    out.backward(do, retain_graph=True)
    first = h.grad.clone()
    h.grad = None
    out.backward(do, retain_graph=True)

    assert torch.equal(h.grad, first)


# ---------------------------------------------------------------------------
# Fallback behavior for configurations Liger cannot serve
# ---------------------------------------------------------------------------


def test_bias_falls_back_to_native():
    fallback = _RecordingFallback()
    module = LigerMegatronSwiGLU(fallback_impl=fallback)

    x = torch.randn(16, 64, device=device, dtype=torch.float32)
    bias = torch.randn(64, device=device, dtype=torch.float32)

    out = module(x, bias)

    assert len(fallback.calls) == 1
    assert fallback.calls[0]["bias"] is bias
    assert_verbose_allclose(out, _megatron_bias_swiglu_reference(x, bias), atol=1e-5, rtol=1e-5)


def test_fp8_input_store_falls_back_to_native():
    fallback = _RecordingFallback()
    module = LigerMegatronSwiGLU(fallback_impl=fallback)

    x = torch.randn(16, 64, device=device, dtype=torch.float32)
    module(x, None, True, False)

    assert len(fallback.calls) == 1
    assert fallback.calls[0]["fp8_input_store"] is True


def test_cpu_offload_input_falls_back_to_native():
    fallback = _RecordingFallback()
    module = LigerMegatronSwiGLU(fallback_impl=fallback)

    x = torch.randn(16, 64, device=device, dtype=torch.float32)
    module(x, None, False, True)

    assert len(fallback.calls) == 1
    assert fallback.calls[0]["cpu_offload_input"] is True


def test_supported_config_does_not_touch_fallback():
    """The common path — no bias, no FP8, no offload — must run Liger, not the fallback."""
    fallback = _RecordingFallback()
    module = LigerMegatronSwiGLU(fallback_impl=fallback)

    x = torch.randn(16, 64, device=device, dtype=torch.float32)
    out = module(x, None, False, False)

    assert fallback.calls == []
    assert_verbose_allclose(out, _megatron_swiglu_reference(x), atol=1e-5, rtol=1e-5)


def test_fallback_is_logged_once_per_reason(caplog):
    """A fallback taken every microbatch must not spam the log once per step."""
    module = LigerMegatronSwiGLU(fallback_impl=_RecordingFallback())
    x = torch.randn(16, 64, device=device, dtype=torch.float32)
    bias = torch.randn(64, device=device, dtype=torch.float32)

    with caplog.at_level("INFO", logger="liger_kernel.megatron.swiglu"):
        for _ in range(5):
            module(x, bias)

    fallback_records = [r for r in caplog.records if "falling back" in r.getMessage()]
    assert len(fallback_records) == 1


def test_unsupported_config_without_fallback_raises_actionable_error():
    """Mode 2 users who never wired a fallback must get a clear error, not silent
    numerical drift."""
    module = LigerMegatronSwiGLU()  # no fallback_impl
    x = torch.randn(16, 64, device=device, dtype=torch.float32)
    bias = torch.randn(64, device=device, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="apply_liger_kernel_to_megatron"):
        module(x, bias)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_4d_input():
    module = LigerMegatronSwiGLU()
    x = torch.randn(2, 2, 4, 64, device=device, dtype=torch.float32)

    with pytest.raises(ValueError, match="2D"):
        module(x)


def test_rejects_odd_last_dimension():
    """An odd trailing dim means the tensor isn't a concatenated gate/up pair."""
    module = LigerMegatronSwiGLU()
    x = torch.randn(16, 63, device=device, dtype=torch.float32)

    with pytest.raises(ValueError, match="even"):
        module(x)


def test_signature_matches_megatron_positional_order():
    """Megatron's MLP and SharedExpertMLP both call bias_swiglu_impl positionally, so
    argument order is part of the contract, not just argument names."""
    import inspect

    params = list(inspect.signature(LigerMegatronSwiGLU.forward).parameters)
    assert params == ["self", "input", "bias", "fp8_input_store", "cpu_offload_input"]


# ---------------------------------------------------------------------------
# 6. End-to-end validation against a real megatron-core install
# ---------------------------------------------------------------------------
#
# Everything above verifies the wrapper against our own reproduction of Megatron's
# formula. That cannot catch a wrong belief about Megatron's API. The tests below import
# the real package and pin: the true signature and positional order of
# ``bias_swiglu_impl``, numerical parity with Megatron's own TorchScript implementation,
# that the consumer module paths the monkey patch rebinds actually exist and are rebound,
# and that a real ``MLP`` produces identical output and gradients when patched.
#
# The import is conditional rather than a module-level ``pytest.importorskip`` so that the
# dependency-free tests above still run when megatron-core is absent.

try:
    import megatron.core  # noqa: F401

    _MEGATRON_AVAILABLE = True
except ImportError:
    _MEGATRON_AVAILABLE = False

requires_megatron = pytest.mark.skipif(
    not _MEGATRON_AVAILABLE
    or not (torch.cuda.is_available() or (getattr(torch, "xpu", None) and torch.xpu.is_available())),
    reason="requires megatron-core and an accelerator",
)

_PATCH_TARGETS = (
    "megatron.core.fusions.fused_bias_swiglu",
    "megatron.core.transformer.mlp",
    "megatron.core.transformer.moe.shared_experts",
)


@pytest.fixture
def restore_patch():
    """Snapshot and restore ``bias_swiglu_impl`` in every module the patch touches.

    Restoring only the consumer modules is not enough: the patch is idempotent on the
    *defining* module, so leaving that one patched makes a later ``apply_...`` call
    short-circuit and the next test observes a half-applied patch.
    """
    import importlib

    mods = [importlib.import_module(name) for name in _PATCH_TARGETS]
    saved = [(m, m.bias_swiglu_impl) for m in mods]
    try:
        yield {m.__name__: fn for m, fn in saved}
    finally:
        for m, fn in saved:
            m.bias_swiglu_impl = fn


@requires_megatron
def test_real_bias_swiglu_impl_signature_is_what_the_wrapper_assumes():
    """``LigerMegatronSwiGLU.forward`` mirrors this signature positionally.

    Both ``MLP.forward`` and ``SharedExpertMLP.forward`` call ``bias_swiglu_impl``
    positionally, so parameter *order* is part of the contract, not just parameter names.
    If Megatron reorders or inserts an argument this fails loudly instead of the patch
    silently passing ``fp8_input_store`` where ``bias`` was expected.
    """
    import inspect

    from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl

    real = list(inspect.signature(bias_swiglu_impl).parameters)
    ours = [p for p in inspect.signature(LigerMegatronSwiGLU.forward).parameters if p != "self"]

    assert real[: len(ours)] == ours, (
        f"megatron-core's bias_swiglu_impl signature {real} no longer matches the order "
        f"LigerMegatronSwiGLU.forward accepts {ours}."
    )


@requires_megatron
@pytest.mark.parametrize("dtype, atol, rtol", [(torch.float32, 1e-6, 1e-6), (torch.bfloat16, 1e-2, 1e-2)])
@pytest.mark.parametrize("shape", [(4, 2, 512), (2, 1, 11008)])
def test_matches_real_megatron_numerically(shape, dtype, atol, rtol):
    """Forward and backward parity against Megatron's actual TorchScript implementation."""
    from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl

    s, b, ffn = shape
    _tensor = torch.randn(s, b, 2 * ffn, device=device, dtype=dtype)
    do = torch.randn(s, b, ffn, device=device, dtype=dtype)

    h_ref = _tensor.clone().requires_grad_(True)
    ref = bias_swiglu_impl(h_ref, None, False, False)
    ref.backward(do)

    h_liger = _tensor.clone().requires_grad_(True)
    got = LigerMegatronSwiGLU()(h_liger, None, False, False)
    got.backward(do)

    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(h_liger.grad, h_ref.grad, atol=atol, rtol=rtol)


@requires_megatron
def test_patch_rebinds_the_real_consumer_modules(restore_patch):
    """The by-name imports in the real megatron-core must actually be rebound.

    ``megatron.core.transformer.mlp`` does ``from ...fused_bias_swiglu import
    bias_swiglu_impl`` at import time, so patching only the defining module leaves the
    consumer holding the original function object. This verifies both that those module
    paths still exist and that the rebinding reaches them.
    """
    import importlib

    from liger_kernel.megatron.monkey_patch import apply_liger_kernel_to_megatron

    apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=False, swiglu=True)

    for name in _PATCH_TARGETS:
        mod = importlib.import_module(name)
        patched = mod.bias_swiglu_impl
        original = restore_patch[name]
        assert patched is not original, f"{name}.bias_swiglu_impl was not rebound"
        # The patch installs a plain function wrapping a shared LigerMegatronSwiGLU and
        # tags it so re-application is idempotent. __wrapped__ must point back at
        # Megatron's real function so unsupported configs fall back rather than recurse.
        assert getattr(patched, "__liger_patched__", False), f"{name} rebound to an untagged object"
        assert patched.__wrapped__ is original

    # And it must actually compute Liger's result, not just look patched.
    mlp_mod = importlib.import_module("megatron.core.transformer.mlp")
    x = torch.randn(4, 2, 128, device=device, dtype=torch.float32)
    torch.testing.assert_close(
        mlp_mod.bias_swiglu_impl(x, None, False, False),
        LigerMegatronSwiGLU()(x, None, False, False),
        atol=0,
        rtol=0,
    )


@requires_megatron
def test_patch_rebinds_consumers_imported_after_the_first_call(restore_patch):
    """Re-applying the patch must still reach a consumer that reverted to the original.

    Real-world shape of this: a consumer module is imported lazily, *after*
    ``apply_liger_kernel_to_megatron`` already ran, so it captures Megatron's original at
    its own import time. A naive "already patched, return" short-circuit would leave that
    consumer permanently unpatched.
    """
    import importlib

    from liger_kernel.megatron.monkey_patch import apply_liger_kernel_to_megatron

    apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=False, swiglu=True)

    mlp_mod = importlib.import_module("megatron.core.transformer.mlp")
    mlp_mod.bias_swiglu_impl = restore_patch["megatron.core.transformer.mlp"]

    apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=False, swiglu=True)

    assert getattr(mlp_mod.bias_swiglu_impl, "__liger_patched__", False), (
        "a consumer that reverted to Megatron's original was not re-bound on re-apply"
    )


@pytest.fixture(scope="module")
def model_parallel():
    """Minimal single-rank model-parallel init so real Megatron layers can be built."""
    from megatron.core import parallel_state
    from megatron.core import tensor_parallel

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    # ColumnParallelLinear initializes its weight under the model-parallel RNG fork, which
    # does not exist until this is called. Without it, building any parallel layer raises
    # "cuda rng state model-parallel-rng is not added".
    tensor_parallel.model_parallel_cuda_manual_seed(0)
    torch.manual_seed(0)
    yield
    parallel_state.destroy_model_parallel()


def _build_mlp(hidden, ffn, dtype):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.mlp import MLP
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        ffn_hidden_size=ffn,
        num_attention_heads=4,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=True,
        add_bias_linear=False,
        bf16=(dtype is torch.bfloat16),
        params_dtype=dtype,
    )
    spec = get_gpt_layer_local_spec().submodules.mlp.submodules
    return MLP(config, spec).to(device=device, dtype=dtype)


@requires_megatron
def test_real_mlp_forward_backward_parity(model_parallel, restore_patch):
    """A real Megatron ``MLP`` must produce identical results with the patch applied.

    This is the test that exercises the actual dispatch path -- ``MLP.forward`` deciding to
    call ``bias_swiglu_impl`` based on ``config.bias_activation_fusion`` /
    ``gated_linear_unit`` / ``activation_func``, with the real ``linear_fc1`` output layout
    feeding it.

    One MLP instance is reused for both runs rather than building two and copying weights:
    the patch is a module-level symbol swap, so the same instance picks it up, and this
    sidesteps Megatron's ``_extra_state`` entries (which are None and break a naive
    ``state_dict`` round-trip).
    """
    import importlib

    from liger_kernel.megatron.monkey_patch import apply_liger_kernel_to_megatron

    hidden, ffn, dtype = 128, 512, torch.float32
    mlp_mod = importlib.import_module("megatron.core.transformer.mlp")
    original = restore_patch["megatron.core.transformer.mlp"]

    mlp = _build_mlp(hidden, ffn, dtype)
    x = torch.randn(8, 2, hidden, device=device, dtype=dtype)

    def run():
        for p_ in mlp.parameters():
            p_.grad = None
        h = x.clone().requires_grad_(True)
        out, _ = mlp(h)
        out.sum().backward()
        return out.detach().clone(), h.grad.clone(), [p_.grad.clone() for p_ in mlp.parameters()]

    ref_out, ref_dx, ref_dw = run()

    apply_liger_kernel_to_megatron(rms_norm=False, cross_entropy=False, swiglu=True)
    assert mlp_mod.bias_swiglu_impl is not original, "patch did not take effect"
    liger_out, liger_dx, liger_dw = run()

    torch.testing.assert_close(liger_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(liger_dx, ref_dx, atol=1e-5, rtol=1e-5)
    for got, want in zip(liger_dw, ref_dw):
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
