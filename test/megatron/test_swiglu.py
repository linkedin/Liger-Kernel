"""Unit tests for LigerMegatronSwiGLU.

These tests deliberately do not import megatron-core; the wrapper's contract is verified
against a local reproduction of Megatron's own reference::

    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2

The parametrization style mirrors ``test/megatron/test_rms_norm.py``.
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
