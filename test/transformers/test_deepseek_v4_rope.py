import importlib

import pytest
import torch

from test.utils import assert_verbose_allclose
from test.utils import supports_bfloat16

from liger_kernel.ops import LigerDeepseekV4RopeFunction
from liger_kernel.transformers import apply_liger_kernel_to_deepseek_v4
from liger_kernel.transformers.deepseek_v4_rope import liger_deepseek_v4_rotary_pos_emb
from liger_kernel.transformers.functional import liger_deepseek_v4_rope
from liger_kernel.utils import infer_device

try:
    from transformers.models.deepseek_v4 import modeling_deepseek_v4
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb

    IS_DEEPSEEK_V4_AVAILABLE = True
except Exception:
    IS_DEEPSEEK_V4_AVAILABLE = False


device = infer_device()


def _make_input(batch_size, num_heads, seq_len, head_dim, unsqueeze_dim, contiguous, dtype):
    if unsqueeze_dim == 1:
        if contiguous:
            return torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        return torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype).transpose(1, 2)

    if contiguous:
        return torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    return torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype).transpose(1, 2)


def _make_trig(batch_size, seq_len, rope_dim, non_contiguous):
    angles = torch.randn(batch_size, seq_len, rope_dim // 2, device=device, dtype=torch.float32)
    if not non_contiguous:
        return angles.cos(), angles.sin()

    cos_storage = torch.empty(batch_size, seq_len, rope_dim, device=device, dtype=torch.float32)
    sin_storage = torch.empty_like(cos_storage)
    cos_storage[..., ::2] = angles.cos()
    sin_storage[..., ::2] = angles.sin()
    return cos_storage[..., ::2], sin_storage[..., ::2]


@pytest.mark.skipif(not IS_DEEPSEEK_V4_AVAILABLE, reason="DeepSeek-V4 is not available in transformers.")
@pytest.mark.parametrize(
    "batch_size,num_heads,seq_len,head_dim,rope_dim,unsqueeze_dim,contiguous,trig_batch_size,trig_contiguous,negative_sin",
    [
        (1, 5, 7, 96, 32, 1, False, 1, True, False),
        (2, 3, 11, 65, 16, 1, True, 1, False, True),
        (2, 4, 13, 80, 48, 1, False, 2, False, False),
        (1, 7, 9, 40, 40, 2, True, 1, True, True),
        (2, 2, 33, 512, 64, 2, False, 2, True, False),
    ],
)
@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (torch.float32, 1e-6, 1e-6),
        pytest.param(
            torch.bfloat16,
            2e-2,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (torch.float16, 2e-3, 1e-3),
    ],
)
def test_deepseek_v4_rope_forward_backward(
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    rope_dim,
    unsqueeze_dim,
    contiguous,
    trig_batch_size,
    trig_contiguous,
    negative_sin,
    dtype,
    atol,
    rtol,
):
    torch.manual_seed(42)
    x = _make_input(batch_size, num_heads, seq_len, head_dim, unsqueeze_dim, contiguous, dtype)
    cos, sin = _make_trig(trig_batch_size, seq_len, rope_dim, not trig_contiguous)
    if negative_sin:
        sin = -sin

    x_ref = x.detach().clone().requires_grad_(True)
    x_liger = x.detach().clone().requires_grad_(True)
    x_before = x_liger.detach().clone()
    output_ref = apply_rotary_pos_emb(x_ref, cos, sin, unsqueeze_dim)
    output_liger = liger_deepseek_v4_rotary_pos_emb(x_liger, cos, sin, unsqueeze_dim)

    assert output_liger.is_contiguous()
    assert torch.equal(x_liger, x_before)
    assert_verbose_allclose(output_ref, output_liger, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(output_ref)
    (grad_ref,) = torch.autograd.grad(output_ref, x_ref, grad_output)
    (grad_liger,) = torch.autograd.grad(output_liger, x_liger, grad_output)
    assert_verbose_allclose(grad_ref, grad_liger, atol=atol, rtol=rtol)


@pytest.mark.skipif(not IS_DEEPSEEK_V4_AVAILABLE, reason="DeepSeek-V4 is not available in transformers.")
def test_deepseek_v4_rope_functional_apis():
    x = torch.randn(1, 3, 5, 32, device=device, dtype=torch.float32)
    cos, sin = _make_trig(1, 5, 16, False)

    expected = LigerDeepseekV4RopeFunction.apply(x, cos, sin, 1)
    assert torch.equal(liger_deepseek_v4_rope(x, cos, sin), expected)
    assert torch.equal(liger_deepseek_v4_rotary_pos_emb(x, cos, sin), expected)


@pytest.mark.skipif(not IS_DEEPSEEK_V4_AVAILABLE, reason="DeepSeek-V4 is not available in transformers.")
@pytest.mark.parametrize(
    "x_dtype,cos_dtype,sin_dtype",
    [
        (torch.float64, torch.float32, torch.float32),
        (torch.float64, torch.float64, torch.float64),
        (torch.float32, torch.float64, torch.float64),
        (torch.float32, torch.float64, torch.float32),
    ],
)
def test_deepseek_v4_rope_fp64_semantics(x_dtype, cos_dtype, sin_dtype):
    torch.manual_seed(42)
    x = torch.randn(2, 5, 7, 65, device=device, dtype=x_dtype) * 1e8
    angles = torch.randn(2, 7, 8, device=device, dtype=torch.float64)
    cos = angles.cos().to(cos_dtype)
    sin = angles.sin().to(sin_dtype)

    x_ref = x.detach().clone().requires_grad_(True)
    x_liger = x.detach().clone().requires_grad_(True)
    output_ref = apply_rotary_pos_emb(x_ref, cos, sin)
    output_liger = liger_deepseek_v4_rotary_pos_emb(x_liger, cos, sin)

    nope_dim = x.shape[-1] - 2 * cos.shape[-1]
    assert torch.equal(output_ref[..., :nope_dim], output_liger[..., :nope_dim])
    assert_verbose_allclose(output_ref, output_liger, atol=16.0, rtol=2e-7)

    grad_output = torch.randn_like(output_ref)
    (grad_ref,) = torch.autograd.grad(output_ref, x_ref, grad_output)
    (grad_liger,) = torch.autograd.grad(output_liger, x_liger, grad_output)
    assert_verbose_allclose(grad_ref, grad_liger, atol=5e-7, rtol=2e-7)


@pytest.mark.skipif(not IS_DEEPSEEK_V4_AVAILABLE, reason="DeepSeek-V4 is not available in transformers.")
def test_deepseek_v4_rope_patch_and_revert():
    original_source = apply_rotary_pos_emb.__code__.co_code
    try:
        apply_liger_kernel_to_deepseek_v4(
            rope=True,
            cross_entropy=False,
            fused_linear_cross_entropy=False,
            rms_norm=False,
            swiglu=False,
        )
        assert modeling_deepseek_v4.apply_rotary_pos_emb is liger_deepseek_v4_rotary_pos_emb

        importlib.reload(modeling_deepseek_v4)
        assert modeling_deepseek_v4.apply_rotary_pos_emb is not liger_deepseek_v4_rotary_pos_emb
        assert modeling_deepseek_v4.apply_rotary_pos_emb.__code__.co_code == original_source
    finally:
        importlib.reload(modeling_deepseek_v4)


@pytest.mark.parametrize("unsqueeze_dim", [0, 3])
def test_deepseek_v4_rope_rejects_unsupported_unsqueeze_dim(unsqueeze_dim):
    x = torch.randn(1, 2, 3, 8, device=device)
    cos = torch.randn(1, 3, 2, device=device)
    sin = torch.randn_like(cos)

    with pytest.raises(ValueError, match="unsqueeze_dim must be 1 or 2"):
        liger_deepseek_v4_rotary_pos_emb(x, cos, sin, unsqueeze_dim)
