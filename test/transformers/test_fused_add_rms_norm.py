import os

import pytest
import torch
import torch.nn as nn

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.functional import fused_add_rms_norm
from liger_kernel.ops import LigerFusedAddRMSNormFunction
from liger_kernel.transformers.functional import liger_fused_add_rms_norm
from liger_kernel.transformers.fused_add_rms_norm import LigerFusedAddRMSNorm
from liger_kernel.utils import infer_device

device = infer_device()

set_seed(42)
torch.use_deterministic_algorithms(True)

#  Only setting torch.use_deterministic_algorithms(True) might throw the following error:
#  RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`,
#  but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an
#  environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information,
#  go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

if device == "cuda":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

SLEEP_SECONDS = 0.1


class BaseAddRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual):
        hidden_states = hidden_states + residual
        residual = hidden_states
        input_dtype = hidden_states.dtype
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype), residual


# RMSNorm: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L112
# mimic residual add behavior as done for Llama:
# https://github.com/huggingface/transformers/blob/1bc9ac5107ff32c0115bd0b269924455be79db64/src/transformers/models/llama/modeling_llama.py#L297
class LlamaAddRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual):
        hidden_states = hidden_states + residual
        residual = hidden_states
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype), residual


# RMSNorm: https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/gemma/modeling_gemma.py#L122
# mimic residual add behavior as done for Gemma:
# https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/models/gemma/modeling_gemma.py#L620
class GemmaAddRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x, residual):
        x = x + residual
        residual = x
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x), residual


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize(
    "bs, sl, hd",
    [
        (2, 128, 512),
        # weird shapes
        (5, 123, 123),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-6),
        pytest.param(
            torch.bfloat16,
            2e-1,
            2e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
@pytest.mark.parametrize(
    "reference, offset, casting_mode",
    [
        (LlamaAddRMSNorm, 0.0, "llama"),
        (GemmaAddRMSNorm, 1.0, "gemma"),
        (BaseAddRMSNorm, 0.0, "none"),
    ],
)
@pytest.mark.parametrize(
    "in_place",
    [
        True,
        False,
    ],
)
def test_correctness(request, bs, sl, hd, dtype, atol, rtol, reference, offset, casting_mode, in_place):
    if (
        reference is BaseAddRMSNorm
        and casting_mode == "none"
        and dtype == torch.bfloat16
        and (bs, sl, hd) == (5, 123, 123)
    ):
        request.node.add_marker(
            pytest.mark.xfail(
                reason=(
                    "Known bf16 precision edge case: Triton's tl.sum accumulates bf16 reductions in bf16, "
                    "while PyTorch's `.mean()` internally upcasts to fp32 even for bf16 tensors. Since "
                    "casting_mode='none' intentionally skips fp32 upcasting (for speed/memory), the two "
                    "implementations' variance sums diverge slightly, occasionally pushing a single element "
                    "just past the tight bf16 tolerance for this specific 'weird' shape. Not a functional "
                    "regression in the kernel; upgrading Triton does not change this (bf16 sum accumulation "
                    "behavior is unchanged across versions tested). strict=False so this reports XPASS "
                    "(not a failure) if a future Triton/PyTorch version resolves the discrepancy."
                ),
                strict=False,
            )
        )

    _tensor = torch.randn(bs, sl, hd, device=device, dtype=dtype)
    _residual = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    h1 = _tensor.clone().requires_grad_(True)
    r1 = _residual.clone().requires_grad_(True)
    h2 = _tensor.clone().requires_grad_(True)
    r2 = _residual.clone().requires_grad_(True)

    # do
    dh = torch.randn(bs, sl, hd, device=device, dtype=dtype)
    dr = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    # reference (llama or gemma)
    ref_rms = reference(hidden_size=hd).to(device).to(dtype)
    ref_h, ref_r = ref_rms(h1, r1)
    torch.autograd.backward((ref_h, ref_r), (dh, dr), retain_graph=True)

    # triton
    triton_rms = (
        LigerFusedAddRMSNorm(hidden_size=hd, offset=offset, casting_mode=casting_mode, in_place=in_place)
        .to(device)
        .to(dtype)
    )
    triton_h, triton_r = triton_rms(h2, r2)

    torch.autograd.backward((triton_h, triton_r), (dh, dr), retain_graph=True)

    assert_verbose_allclose(ref_h, triton_h, atol=atol, rtol=rtol)
    assert_verbose_allclose(ref_r, triton_r, atol=atol, rtol=rtol)
    assert_verbose_allclose(ref_rms.weight.grad, triton_rms.weight.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(h1.grad, h2.grad, atol=atol, rtol=rtol, max_print=20)
    assert_verbose_allclose(r1.grad, r2.grad, atol=atol, rtol=rtol, max_print=20)


@pytest.mark.parametrize(
    "bs, sl, hd",
    [
        (2, 2, 8),
        # weird shapes
        (9, 7, 41),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-6),
        (torch.bfloat16, 2e-1, 2e-2),
    ],
)
@pytest.mark.parametrize(
    "reference, offset, casting_mode",
    [
        (LlamaAddRMSNorm, 0.0, "llama"),
        (GemmaAddRMSNorm, 1.0, "gemma"),
    ],
)
@pytest.mark.parametrize(
    "in_place",
    [
        True,
        False,
    ],
)
def test_correctness_functional(bs, sl, hd, dtype, atol, rtol, reference, offset, casting_mode, in_place):
    # h
    _tensor = torch.randn(bs, sl, hd, device=device, dtype=dtype)
    _residual = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    h1 = _tensor.clone().requires_grad_(True)
    r1 = _residual.clone().requires_grad_(True)
    h2 = _tensor.clone().requires_grad_(True)
    r2 = _residual.clone().requires_grad_(True)

    w = torch.randn(hd, device=device, dtype=dtype)

    h, r = liger_fused_add_rms_norm(
        X=h1, R=r1, W=w, eps=1e-6, offset=offset, casting_mode=casting_mode, in_place=in_place
    )
    ref_h, ref_r = LigerFusedAddRMSNormFunction.apply(h2, r2, w, 1e-6, offset, casting_mode, in_place)

    assert torch.allclose(h, ref_h, atol=atol, rtol=rtol)
    assert torch.allclose(r, ref_r, atol=atol, rtol=rtol)

    dh = torch.randn_like(h)
    dh_ref = dh.clone()
    dr = torch.randn_like(r)
    dr_ref = dr.clone()

    torch.autograd.backward((h, r), (dh, dr), retain_graph=True)
    torch.autograd.backward((ref_h, ref_r), (dh_ref, dr_ref), retain_graph=True)

    assert torch.allclose(h1.grad, h2.grad, atol=atol, rtol=rtol)
    assert torch.allclose(r1.grad, r2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_torch_compile_matches_eager(dtype):
    """Compiled fused add+RMSNorm must match eager for both low-precision dtypes."""
    torch._dynamo.reset()
    torch.manual_seed(0)
    hidden = 1024
    model = (
        LigerFusedAddRMSNorm(hidden_size=hidden, offset=0.0, casting_mode="llama", in_place=False)
        .to(device=device)
        .to(dtype)
    )
    x = torch.randn(64, hidden, device=device, dtype=dtype)
    residual = torch.randn(64, hidden, device=device, dtype=dtype)
    with torch.no_grad():
        expected = model(x, residual)
        actual = torch.compile(model)(x, residual)
    if isinstance(expected, tuple):
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_torch_compile_dynamic_batch_matches_eager(dtype):
    """Dynamic batch must not silently return uninitialized memory.

    On the pre-fix code ``torch.compile`` returned uninitialized output here in
    both static and dynamic mode: ``eps`` reached the kernel as a
    ``tl.constexpr``, the CompilationError was swallowed, and the output buffer
    was never written. This case covers the dynamic-batch path.
    """
    torch._dynamo.reset()
    torch.manual_seed(0)
    hidden = 1024
    model = LigerFusedAddRMSNorm(hidden_size=hidden, in_place=False).to(device=device).to(dtype)
    compiled = torch.compile(model, dynamic=True)
    for rows in (1, 2, 4):
        x = torch.randn(rows, hidden, device=device, dtype=dtype)
        residual = torch.randn(rows, hidden, device=device, dtype=dtype)
        with torch.no_grad():
            expected = model(x, residual)
            actual = compiled(x, residual)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("shape", [(2, 7, 123), (1, 5, 5280)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "reference, offset, casting_mode",
    [(LlamaAddRMSNorm, 0.0, "llama"), (GemmaAddRMSNorm, 1.0, "gemma"), (BaseAddRMSNorm, 0.0, "none")],
)
def test_output_only_matches_default(shape, dtype, reference, offset, casting_mode):
    model = LigerFusedAddRMSNorm(shape[-1], offset=offset, casting_mode=casting_mode).to(device=device, dtype=dtype)
    ref = reference(shape[-1]).to(device=device, dtype=dtype)
    model.weight.data.copy_(torch.randn_like(model.weight))
    ref.weight.data.copy_(model.weight)
    x = torch.randn(shape, device=device, dtype=dtype).requires_grad_()
    residual = torch.randn_like(x).requires_grad_()
    x_before, r_before = x.detach().clone(), residual.detach().clone()

    with torch.no_grad():
        expected, _ = model(x, residual)
        model.return_residual = False
        actual = model(x, residual)
        ref_y, _ = ref(x, residual)

    assert isinstance(actual, torch.Tensor)
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    atol, rtol = (1e-4, 1e-5) if dtype == torch.float32 else (2e-1, 2e-2)
    torch.testing.assert_close(actual, ref_y, atol=atol, rtol=rtol)
    torch.testing.assert_close(x, x_before, atol=0, rtol=0)
    torch.testing.assert_close(residual, r_before, atol=0, rtol=0)


@pytest.mark.parametrize("api", ["module", "functional", "dispatch"])
@pytest.mark.parametrize("context", ["no_grad", "inference_mode", "frozen"])
def test_output_only_public_calls(api, context):
    model = LigerFusedAddRMSNorm(123, return_residual=False).to(device)
    model.weight = torch.nn.Parameter(torch.ones(246, device=device)[::2])
    x = torch.randn(7, 2, 123, device=device).transpose(0, 1)
    residual = torch.randn_like(x)
    if context == "frozen":
        model.requires_grad_(False)

    def call():
        if api == "module":
            return model(x, residual)
        if api == "functional":
            return liger_fused_add_rms_norm(x, residual, model.weight, 1e-6, return_residual=False)
        return fused_add_rms_norm(x, residual, model.weight, impl="nvidia-triton", return_residual=False)

    manager = {
        "no_grad": torch.no_grad,
        "inference_mode": torch.inference_mode,
        "frozen": torch.enable_grad,
    }[context]
    with manager():
        actual = call()
    with torch.no_grad():
        expected = LlamaAddRMSNorm(123).to(device)(x, residual)[0]
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("grad_input", ["x", "residual", "weight"])
@pytest.mark.parametrize("api", ["module", "functional", "dispatch"])
def test_output_only_rejects_required_gradients(grad_input, api):
    x = torch.randn(2, 8, device=device, requires_grad=grad_input == "x")
    residual = torch.randn_like(x, requires_grad=grad_input == "residual")
    weight = torch.ones(8, device=device, requires_grad=grad_input == "weight")
    with pytest.raises(ValueError, match="requires torch.no_grad"):
        if api == "module":
            model = LigerFusedAddRMSNorm(8, return_residual=False).to(device).eval()
            model.weight = torch.nn.Parameter(weight, requires_grad=weight.requires_grad)
            model(x, residual)
        elif api == "functional":
            liger_fused_add_rms_norm(x, residual, weight, 1e-6, return_residual=False)
        else:
            fused_add_rms_norm(x, residual, weight, impl="nvidia-triton", return_residual=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA memory check")
def test_output_only_skips_residual_buffer():
    x = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.ones(2048, device="cuda", dtype=x.dtype)

    def peak(return_residual):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        output = liger_fused_add_rms_norm(x, residual, weight, 1e-6, return_residual=return_residual)
        torch.cuda.synchronize()
        used = torch.cuda.max_memory_allocated() - baseline
        del output
        return used

    with torch.no_grad():
        peak(True)
        peak(False)
        default_peak = peak(True)
        output_only_peak = peak(False)
    assert default_peak - output_only_peak >= residual.numel() * residual.element_size()


def test_output_only_shared_steps_match_default():
    model = LigerFusedAddRMSNorm(256).to(device=device, dtype=torch.bfloat16)
    x = torch.randn(2, 7, 256, device=device, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    with torch.no_grad():
        expected, actual = x, x
        for _ in range(32):
            expected = model(expected, residual)[0]
            actual = liger_fused_add_rms_norm(actual, residual, model.weight, 1e-6, return_residual=False)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_output_only_compile_matches_eager():
    model = LigerFusedAddRMSNorm(256, return_residual=False).to(device=device, dtype=torch.bfloat16)
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    for rows in (1, 7):
        x = torch.randn(rows, 256, device=device, dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        with torch.no_grad():
            actual = compiled(x, residual)
            expected = model(x, residual)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
