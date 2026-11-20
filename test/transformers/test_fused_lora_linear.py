import pytest
import torch
import torch.nn.functional as F

from test.utils import assert_verbose_allclose
from test.utils import infer_device
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.ops import LigerFusedLoRALinearFunction
from liger_kernel.transformers.functional import liger_fused_lora_linear
from liger_kernel.transformers.fused_lora_linear import LigerFusedLoRALinear

set_seed(42)
device = infer_device()


class TorchFusedLoRALinear(torch.nn.Module):
    """PEFT-style LoRA linear: frozen base + trainable low-rank update."""

    def __init__(self, in_features, out_features, r=16, lora_alpha=16, bias=False, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = lora_alpha / r
        self.base_layer = torch.nn.Linear(in_features, out_features, bias=bias, **factory_kwargs)
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)
        self.lora_A = torch.nn.Parameter(torch.empty(r, in_features, **factory_kwargs))
        self.lora_B = torch.nn.Parameter(torch.zeros(out_features, r, **factory_kwargs))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x):
        base = F.linear(x, self.base_layer.weight, self.base_layer.bias)
        return base + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


def _init_pair(in_features, out_features, r, lora_alpha, bias, dtype):
    torch_layer = TorchFusedLoRALinear(
        in_features, out_features, r=r, lora_alpha=lora_alpha, bias=bias, device=device, dtype=dtype
    )
    liger_layer = LigerFusedLoRALinear(
        in_features, out_features, r=r, lora_alpha=lora_alpha, bias=bias, device=device, dtype=dtype
    )
    with torch.no_grad():
        liger_layer.base_layer.weight.copy_(torch_layer.base_layer.weight)
        if bias:
            liger_layer.base_layer.bias.copy_(torch_layer.base_layer.bias)
        liger_layer.lora_A.copy_(torch_layer.lora_A)
        liger_layer.lora_B.copy_(torch_layer.lora_B)
    return torch_layer, liger_layer


@pytest.mark.parametrize(
    "B, T, K, N, r",
    [
        (2, 8, 256, 256, 8),
        (1, 1, 1023, 1024, 16),
        (3, 7, 512, 768, 13),  # non-power-of-2 rank -> BLOCK_R masking
        (2, 4, 2048, 1024, 64),
    ],
)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-4),
        pytest.param(
            torch.bfloat16,
            1e-1,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_liger_fused_lora_linear_correctness(B, T, K, N, r, bias, dtype, atol, rtol):
    torch_layer, liger_layer = _init_pair(K, N, r, 16, bias, dtype)

    x = torch.randn(B, T, K, device=device, dtype=dtype)
    x_ref = x.clone().requires_grad_(True)
    x_liger = x.clone().requires_grad_(True)

    y_ref = torch_layer(x_ref)
    y_liger = liger_layer(x_liger)
    assert_verbose_allclose(y_ref, y_liger, rtol=rtol, atol=atol, extra_info="[output]")

    grad_output = torch.randn_like(y_ref)
    torch.autograd.backward(y_ref, grad_output)
    torch.autograd.backward(y_liger, grad_output)

    assert_verbose_allclose(x_ref.grad, x_liger.grad, rtol=rtol, atol=atol, extra_info="[input.grad]")
    assert_verbose_allclose(
        torch_layer.lora_A.grad, liger_layer.lora_A.grad, rtol=rtol, atol=atol, extra_info="[lora_A.grad]"
    )
    assert_verbose_allclose(
        torch_layer.lora_B.grad, liger_layer.lora_B.grad, rtol=rtol, atol=atol, extra_info="[lora_B.grad]"
    )
    assert liger_layer.base_layer.weight.grad is None
    if bias:
        assert liger_layer.base_layer.bias.grad is None


@pytest.mark.parametrize("dtype, atol, rtol", [(torch.float32, 1e-4, 1e-4)])
def test_liger_fused_lora_linear_functional_grads(dtype, atol, rtol):
    """Exercise trainable W/bias (full fine-tuning path) and the functional API."""
    M, K, N, r = 64, 128, 96, 4
    x = torch.randn(M, K, device=device, dtype=dtype)
    W = torch.randn(N, K, device=device, dtype=dtype)
    A = torch.randn(r, K, device=device, dtype=dtype)
    B = torch.randn(N, r, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)
    grad_output = torch.randn(M, N, device=device, dtype=dtype)

    def reference(xx, Ww, Aa, Bb, bb):
        return F.linear(xx, Ww, bb) + F.linear(F.linear(xx, Aa), Bb) * (16 / r)

    args_ref = [t.clone().requires_grad_(True) for t in (x, W, A, B, bias)]
    args_liger = [t.clone().requires_grad_(True) for t in (x, W, A, B, bias)]

    y_ref = reference(*args_ref)
    y_liger = liger_fused_lora_linear(*args_liger, scale=16 / r)
    assert_verbose_allclose(y_ref, y_liger, rtol=rtol, atol=atol, extra_info="[functional output]")

    grads_ref = torch.autograd.grad(y_ref, args_ref, grad_output)
    grads_liger = torch.autograd.grad(y_liger, args_liger, grad_output)
    names = ["x", "W", "A", "B", "bias"]
    for name, gr, gl in zip(names, grads_ref, grads_liger):
        assert_verbose_allclose(gr, gl, rtol=rtol, atol=atol, extra_info=f"[functional {name}.grad]")

    # Direct Function.apply must match the functional wrapper.
    x_copy = args_liger[0].detach().clone().requires_grad_(True)
    y_apply = LigerFusedLoRALinearFunction.apply(
        x_copy,
        args_liger[1],
        args_liger[2],
        args_liger[3],
        args_liger[4],
        16 / r,
    )
    assert_verbose_allclose(y_liger, y_apply, rtol=rtol, atol=atol, extra_info="[functional vs apply]")


def test_liger_fused_lora_linear_non_contiguous_input():
    K, N, r, M = 512, 256, 8, 32
    x = torch.randn(K, M, device=device).t().requires_grad_(True)
    assert not x.is_contiguous()

    torch_layer, liger_layer = _init_pair(K, N, r, 16, False, torch.float32)
    y_ref = torch_layer(x)
    y_liger = liger_layer(x)
    assert_verbose_allclose(y_ref, y_liger, rtol=1e-4, atol=1e-4, extra_info="[non-contiguous output]")


def test_liger_fused_lora_linear_from_linear():
    K, N, r = 256, 128, 8
    linear = torch.nn.Linear(K, N, bias=True, device=device)
    liger_layer = LigerFusedLoRALinear.from_linear(linear, r=r, lora_alpha=16)
    torch_layer = TorchFusedLoRALinear(K, N, r=r, lora_alpha=16, bias=True, device=device)
    with torch.no_grad():
        torch_layer.base_layer.weight.copy_(linear.weight)
        torch_layer.base_layer.bias.copy_(linear.bias)
        torch_layer.lora_A.copy_(liger_layer.lora_A)
        torch_layer.lora_B.copy_(liger_layer.lora_B)

    x = torch.randn(16, K, device=device)
    assert_verbose_allclose(torch_layer(x), liger_layer(x), rtol=1e-4, atol=1e-4, extra_info="[from_linear]")


def test_liger_fused_lora_linear_training_parity():
    """Short training run must match the reference trajectory (convergence smoke test)."""
    K, N, r = 128, 64, 4
    torch_layer, liger_layer = _init_pair(K, N, r, 16, True, torch.float32)
    opt_ref = torch.optim.SGD([torch_layer.lora_A, torch_layer.lora_B], lr=0.1)
    opt_liger = torch.optim.SGD([liger_layer.lora_A, liger_layer.lora_B], lr=0.1)

    torch.manual_seed(7)
    for step in range(10):
        x = torch.randn(32, K, device=device)

        loss_ref = torch_layer(x).square().mean()
        loss_liger = liger_layer(x).square().mean()
        assert_verbose_allclose(loss_ref, loss_liger, rtol=1e-4, atol=1e-5, extra_info=f"[loss step {step}]")

        opt_ref.zero_grad()
        opt_liger.zero_grad()
        loss_ref.backward()
        loss_liger.backward()
        opt_ref.step()
        opt_liger.step()

    assert_verbose_allclose(
        torch_layer.lora_A, liger_layer.lora_A, rtol=1e-4, atol=1e-5, extra_info="[lora_A after training]"
    )
    assert_verbose_allclose(
        torch_layer.lora_B, liger_layer.lora_B, rtol=1e-4, atol=1e-5, extra_info="[lora_B after training]"
    )
