import pytest
import torch

from liger_kernel.ops import LigerLfm2MoeRouterFunction
from liger_kernel.utils import infer_device

device = infer_device()


def _reference(router_logits, expert_bias, top_k, norm_topk_prob, routed_scaling_factor):
    probabilities = router_logits.sigmoid()
    scores = probabilities if expert_bias is None else probabilities + expert_bias
    selected_experts = torch.topk(scores, k=top_k, dim=-1).indices
    routing_weights = torch.gather(probabilities, 1, selected_experts)
    if norm_topk_prob:
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
    return selected_experts, routing_weights * routed_scaling_factor


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_expert_bias", [False, True])
@pytest.mark.parametrize("norm_topk_prob", [False, True])
@pytest.mark.parametrize(("num_experts", "top_k"), [(8, 2), (48, 4)])
def test_lfm2_moe_router_forward_backward(dtype, use_expert_bias, norm_topk_prob, num_experts, top_k):
    torch.manual_seed(42)
    router_logits = torch.randn(37, num_experts, device=device, dtype=dtype) * 0.5
    expert_bias = torch.randn(num_experts, device=device, dtype=torch.float32) * 0.01 if use_expert_bias else None
    grad = torch.randn(37, top_k, device=device, dtype=dtype)
    routed_scaling_factor = 1.7

    logits_ref = router_logits.detach().clone().requires_grad_(True)
    indices_ref, weights_ref = _reference(
        logits_ref,
        expert_bias,
        top_k,
        norm_topk_prob,
        routed_scaling_factor,
    )
    weights_ref.backward(grad)

    logits_liger = router_logits.detach().clone().requires_grad_(True)
    indices_liger, weights_liger = LigerLfm2MoeRouterFunction.apply(
        logits_liger,
        expert_bias,
        top_k,
        norm_topk_prob,
        routed_scaling_factor,
    )
    weights_liger.backward(grad)

    torch.testing.assert_close(indices_liger.long(), indices_ref)
    atol = 1e-6 if dtype == torch.float32 else 2e-2
    rtol = 1e-6 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(weights_liger, weights_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(logits_liger.grad, logits_ref.grad, atol=atol, rtol=rtol)
