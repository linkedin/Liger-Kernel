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
