import os

import pytest
import torch
import torch.nn as nn

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.ops import LigerModulatedRMSNormFunction
from liger_kernel.transformers.functional import liger_modulated_rms_norm
from liger_kernel.transformers.modulated_rms_norm import LigerModulatedRMSNorm
from liger_kernel.utils import infer_device

device = infer_device()

set_seed(42)
torch.use_deterministic_algorithms(True)

if device == "cuda":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _broadcast_modulation(modulation, hidden_states):
    if modulation.dim() == 1:
        return modulation
    if modulation.shape == hidden_states.shape:
        return modulation
    if hidden_states.dim() == 3 and modulation.dim() == 2 and modulation.shape[0] == hidden_states.shape[0]:
        return modulation[:, None, :]
    if hidden_states.dim() == 2 and modulation.dim() == 2 and hidden_states.shape[0] % modulation.shape[0] == 0:
        rows_per_modulation = hidden_states.shape[0] // modulation.shape[0]
        return modulation.repeat_interleave(rows_per_modulation, dim=0)
    raise AssertionError("Unsupported modulation shape for reference implementation.")


class ModulatedRMSNormReference(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, offset=0.0, casting_mode="llama", elementwise_affine=True):
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.eps = eps
        self.offset = offset
        self.casting_mode = casting_mode

    def _norm(self, hidden_states):
        input_dtype = hidden_states.dtype

        if self.casting_mode == "llama":
            normed = hidden_states.to(torch.float32)
            variance = normed.pow(2).mean(-1, keepdim=True)
            normed = normed * torch.rsqrt(variance + self.eps)
            normed = normed.to(input_dtype)
            if self.elementwise_affine:
                normed = normed * (self.offset + self.weight)
            return normed

        if self.casting_mode == "gemma":
            normed = hidden_states.to(torch.float32)
            variance = normed.pow(2).mean(-1, keepdim=True)
            normed = normed * torch.rsqrt(variance + self.eps)
            if self.elementwise_affine:
                normed = normed * (self.offset + self.weight.float())
            return normed.to(input_dtype)

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        normed = hidden_states * torch.rsqrt(variance + self.eps)
        if self.elementwise_affine:
            normed = normed * (self.offset + self.weight)
        return normed

    def forward(self, hidden_states, scale, shift=None):
        normed = self._norm(hidden_states)
        scale = _broadcast_modulation(scale, hidden_states)
        output = normed * (1 + scale)
        if shift is not None:
            shift = _broadcast_modulation(shift, hidden_states)
            output = output + shift
        return output


def _make_modulation(shape, hd, scale_mode, dtype):
    if scale_mode == "global":
        mod_shape = (hd,)
    elif scale_mode == "batch":
        bs = shape[0]
        mod_shape = (bs, hd)
    elif scale_mode == "row":
        mod_shape = shape
    else:
        raise ValueError(f"Unsupported scale mode: {scale_mode}")
    scale = torch.randn(mod_shape, device=device, dtype=dtype) * 0.1
    shift = torch.randn(mod_shape, device=device, dtype=dtype) * 0.1
    return scale, shift


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize(
    "bs, sl, hd",
    [
        (2, 16, 512),
        (5, 7, 123),
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
    "offset, casting_mode",
    [
        (0.0, "llama"),
        (1.0, "gemma"),
        pytest.param(
            0.0,
            "none",
            marks=pytest.mark.skipif(device == "npu", reason="Ascend NPU does not support this test"),
        ),
    ],
)
@pytest.mark.parametrize("scale_mode", ["global", "batch", "row"])
@pytest.mark.parametrize("has_shift", [True, False])
@pytest.mark.parametrize("elementwise_affine", [True, False])
def test_correctness(bs, sl, hd, dtype, atol, rtol, offset, casting_mode, scale_mode, has_shift, elementwise_affine):
    shape = (bs, sl, hd)
    tensor = torch.randn(shape, device=device, dtype=dtype)
    scale, shift = _make_modulation(shape, hd, scale_mode, dtype)
    shift = shift if has_shift else None

    h1 = tensor.clone().requires_grad_(True)
    h2 = tensor.clone().requires_grad_(True)
    scale1 = scale.clone().requires_grad_(True)
    scale2 = scale.clone().requires_grad_(True)
    shift1 = shift.clone().requires_grad_(True) if shift is not None else None
    shift2 = shift.clone().requires_grad_(True) if shift is not None else None

    grad = torch.randn(shape, device=device, dtype=dtype)

    ref = (
        ModulatedRMSNormReference(
            hidden_size=hd,
            offset=offset,
            casting_mode=casting_mode,
            elementwise_affine=elementwise_affine,
        )
        .to(device)
        .to(dtype)
    )
    triton_mod = (
        LigerModulatedRMSNorm(
            hidden_size=hd,
            offset=offset,
            casting_mode=casting_mode,
            in_place=False,
            elementwise_affine=elementwise_affine,
        )
        .to(device)
        .to(dtype)
    )

    if elementwise_affine:
        with torch.no_grad():
            triton_mod.weight.copy_(ref.weight)

    ref_out = ref(h1, scale1, shift1)
    triton_out = triton_mod(h2, scale2, shift2)

    ref_out.backward(grad, retain_graph=True)
    triton_out.backward(grad, retain_graph=True)

    assert_verbose_allclose(ref_out, triton_out, atol=atol, rtol=rtol)
    assert_verbose_allclose(h1.grad, h2.grad, atol=atol, rtol=rtol, max_print=20)
    assert_verbose_allclose(scale1.grad, scale2.grad, atol=atol, rtol=rtol, max_print=20)
    if has_shift:
        assert_verbose_allclose(shift1.grad, shift2.grad, atol=atol, rtol=rtol, max_print=20)
    if elementwise_affine:
        assert_verbose_allclose(ref.weight.grad, triton_mod.weight.grad, atol=atol, rtol=rtol)


@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
@pytest.mark.parametrize("scale_mode", ["global", "batch", "row"])
@pytest.mark.parametrize("has_shift", [True, False])
def test_mixed_dtype_modulation(scale_mode, has_shift):
    """
    X in bf16, scale/shift in fp32: a common DiT/AdaLN setup. Verifies the kernel
    handles mismatched modulation dtype (atomic_add path requires an implicit cast),
    preserves X.dtype on the output, and produces scale/shift gradients in their
    original (fp32) dtype.
    """
    bs, sl, hd = 2, 4, 64
    shape = (bs, sl, hd)
    x_dtype = torch.bfloat16
    mod_dtype = torch.float32

    tensor = torch.randn(shape, device=device, dtype=x_dtype)
    scale, shift = _make_modulation(shape, hd, scale_mode, mod_dtype)
    shift = shift if has_shift else None

    h_triton = tensor.clone().requires_grad_(True)
    scale_triton = scale.clone().requires_grad_(True)
    shift_triton = shift.clone().requires_grad_(True) if shift is not None else None

    grad = torch.randn(shape, device=device, dtype=x_dtype)

    triton_mod = LigerModulatedRMSNorm(hidden_size=hd, in_place=False).to(device).to(x_dtype)
    triton_out = triton_mod(h_triton, scale_triton, shift_triton)
    assert triton_out.dtype == x_dtype, "Liger must preserve X dtype on output"

    triton_out.backward(grad)
    assert scale_triton.grad.dtype == mod_dtype
    if has_shift:
        assert shift_triton.grad.dtype == mod_dtype

    # Reference: do the whole computation in fp32, then compare against bf16 triton out.
    h_ref = tensor.clone().float().requires_grad_(True)
    scale_ref = scale.clone().requires_grad_(True)
    shift_ref = shift.clone().requires_grad_(True) if shift is not None else None
    ref = ModulatedRMSNormReference(hidden_size=hd).to(device).to(torch.float32)
    with torch.no_grad():
        ref.weight.copy_(triton_mod.weight.float())
    ref_out_fp32 = ref(h_ref, scale_ref, shift_ref)
    ref_out_fp32.backward(grad.float())

    assert_verbose_allclose(ref_out_fp32.bfloat16(), triton_out, atol=2e-1, rtol=2e-2)
    assert_verbose_allclose(h_ref.grad.bfloat16(), h_triton.grad, atol=2e-1, rtol=2e-2)
    assert_verbose_allclose(scale_ref.grad, scale_triton.grad, atol=2e-1, rtol=2e-2)
    if has_shift:
        assert_verbose_allclose(shift_ref.grad, shift_triton.grad, atol=2e-1, rtol=2e-2)


@pytest.mark.parametrize("in_place", [True, False])
@pytest.mark.parametrize("has_shift", [True, False])
@pytest.mark.parametrize("elementwise_affine", [True, False])
def test_functional_correctness(in_place, has_shift, elementwise_affine):
    bs, sl, hd = 2, 3, 8
    dtype = torch.float32
    tensor = torch.randn(bs, sl, hd, device=device, dtype=dtype)
    scale = torch.randn(bs, hd, device=device, dtype=dtype) * 0.1
    shift = torch.randn(bs, hd, device=device, dtype=dtype) * 0.1 if has_shift else None
    weight = torch.randn(hd, device=device, dtype=dtype) if elementwise_affine else None

    h1 = tensor.clone().requires_grad_(True)
    h2 = tensor.clone().requires_grad_(True)
    scale1 = scale.clone().requires_grad_(True)
    scale2 = scale.clone().requires_grad_(True)
    shift1 = shift.clone().requires_grad_(True) if shift is not None else None
    shift2 = shift.clone().requires_grad_(True) if shift is not None else None
    w1 = weight.clone().requires_grad_(True) if weight is not None else None
    w2 = weight.clone().requires_grad_(True) if weight is not None else None

    y1 = liger_modulated_rms_norm(h1, w1, scale1, shift1, in_place=in_place)
    y2 = LigerModulatedRMSNormFunction.apply(h2, w2, scale2, shift2, 1e-6, 0.0, "llama", in_place)

    assert_verbose_allclose(y1, y2, atol=1e-4, rtol=1e-6)

    grad = torch.randn_like(y2)
    y1.backward(grad.clone())
    y2.backward(grad.clone())

    assert_verbose_allclose(h1.grad, h2.grad, atol=1e-4, rtol=1e-6)
    assert_verbose_allclose(scale1.grad, scale2.grad, atol=1e-4, rtol=1e-6)
    if has_shift:
        assert_verbose_allclose(shift1.grad, shift2.grad, atol=1e-4, rtol=1e-6)
    if elementwise_affine:
        assert_verbose_allclose(w1.grad, w2.grad, atol=1e-4, rtol=1e-6)
