from types import SimpleNamespace

import pytest
import torch

from liger_kernel.transformers import swiglu
from liger_kernel.transformers.monkey_patch import _patch_swiglu_module
from liger_kernel.transformers.swiglu import LigerExperts
from liger_kernel.transformers.swiglu import LigerLfm2MoeExperts


@pytest.mark.parametrize(
    ("tokens", "rocm", "expected"),
    [
        (256, True, "liger"),
        (512, True, "grouped_mm"),
        (512, False, "liger"),
    ],
)
def test_lfm2_moe_rocm_shape_dispatch(monkeypatch, tokens, rocm, expected):
    from transformers.integrations import moe

    config = SimpleNamespace(
        hidden_size=8,
        moe_intermediate_size=4,
        num_experts=4,
    )
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeExperts

    experts = Lfm2MoeExperts(config)
    _patch_swiglu_module(experts, LigerLfm2MoeExperts)
    hidden_states = torch.randn(tokens, config.hidden_size)
    top_k_index = torch.zeros(tokens, 2, dtype=torch.long)
    top_k_weights = torch.full((tokens, 2), 0.5)
    calls = []

    monkeypatch.setattr(swiglu, "is_hip", lambda: rocm)
    monkeypatch.setattr(torch.nn.functional, "grouped_mm", object(), raising=False)

    def fake_grouped_mm(module, hidden, indices, weights):
        assert module.has_gate
        assert not module.has_bias
        assert not module.is_transposed
        torch.testing.assert_close(
            module._apply_gate(torch.ones(tokens, 8)),
            torch.nn.functional.silu(torch.ones(tokens, 4)),
        )
        calls.append("grouped_mm")
        return hidden

    def fake_liger(module, hidden, indices, weights):
        calls.append("liger")
        return hidden

    monkeypatch.setattr(moe, "grouped_mm_experts_forward", fake_grouped_mm)
    monkeypatch.setattr(LigerExperts, "forward", fake_liger)

    output = experts(hidden_states, top_k_index, top_k_weights)

    assert calls == [expected]
    torch.testing.assert_close(output, hidden_states)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm required")
def test_lfm2_moe_grouped_mm_dispatch_forward_backward(monkeypatch):
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeExperts

    grouped_mm_available = hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm")
    if not grouped_mm_available:
        pytest.skip("PyTorch grouped MM is unavailable")

    config = SimpleNamespace(hidden_size=64, moe_intermediate_size=32, num_experts=4)
    reference = Lfm2MoeExperts(config).to(device="cuda", dtype=torch.bfloat16)
    actual = Lfm2MoeExperts(config).to(device="cuda", dtype=torch.bfloat16)
    actual.load_state_dict(reference.state_dict())
    _patch_swiglu_module(actual, LigerLfm2MoeExperts)
    monkeypatch.setattr(swiglu, "is_hip", lambda: True)

    torch.manual_seed(42)
    hidden_reference = torch.randn(512, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    hidden_actual = hidden_reference.detach().clone().requires_grad_(True)
    indices = torch.randint(0, config.num_experts, (512, 2), device="cuda")
    weights_reference = (
        torch.softmax(torch.randn(512, 2, device="cuda"), dim=-1).to(torch.bfloat16).requires_grad_(True)
    )
    weights_actual = weights_reference.detach().clone().requires_grad_(True)

    output_reference = reference(hidden_reference, indices, weights_reference)
    output_actual = actual(hidden_actual, indices, weights_actual)
    torch.testing.assert_close(output_actual, output_reference, atol=0.08, rtol=0.02)

    output_reference.float().sum().backward()
    output_actual.float().sum().backward()
    torch.testing.assert_close(hidden_actual.grad, hidden_reference.grad, atol=0.08, rtol=0.02)
    torch.testing.assert_close(weights_actual.grad, weights_reference.grad, atol=0.08, rtol=0.02)
    for actual_parameter, reference_parameter in zip(actual.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad, atol=0.5, rtol=0.05)
