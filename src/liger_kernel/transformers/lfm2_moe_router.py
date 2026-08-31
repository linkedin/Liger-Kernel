import torch

from liger_kernel.ops import LigerLfm2MoeRouterFunction


def liger_lfm2_moe_route_tokens_to_experts(self, router_logits):
    """Route LFM2-MoE tokens without materializing full sigmoid routing weights."""
    expert_bias = self.expert_bias if self.use_expert_bias else None
    return LigerLfm2MoeRouterFunction.apply(
        router_logits,
        expert_bias,
        self.top_k,
        self.norm_topk_prob,
        self.routed_scaling_factor,
    )


def liger_lfm2_moe_router_forward(self, hidden_states, expert_bias=None):
    """Route tokens for the Transformers layout with a dedicated TopK router module."""
    router_logits = torch.nn.functional.linear(hidden_states, self.weight)
    selected_experts, routing_weights = LigerLfm2MoeRouterFunction.apply(
        router_logits,
        expert_bias,
        self.top_k,
        self.norm_topk_prob,
        self.routed_scaling_factor,
    )
    return router_logits, routing_weights, selected_experts
