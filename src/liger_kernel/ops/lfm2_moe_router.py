import torch
import triton
import triton.language as tl


@triton.jit
def _lfm2_moe_router_forward(
    router_logits,
    expert_bias,
    selected_experts,
    routing_weights,
    n_tokens,
    n_experts,
    stride_token,
    routed_scaling_factor: tl.constexpr,
    norm_topk_prob: tl.constexpr,
    has_expert_bias: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
):
    token = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_EXPERTS)
    expert_mask = expert_offsets < n_experts
    logits = tl.load(router_logits + token * stride_token + expert_offsets, mask=expert_mask, other=-float("inf"))
    # Match torch.sigmoid's output dtype before top-k; BF16 rounding can
    # otherwise change expert selection for nearly equal router scores.
    probabilities = tl.sigmoid(logits.to(tl.float32)).to(logits.dtype)
    scores = probabilities
    if has_expert_bias:
        scores = probabilities.to(tl.float32) + tl.load(expert_bias + expert_offsets, mask=expert_mask, other=0.0).to(
            tl.float32
        )

    k_offsets = tl.arange(0, BLOCK_K)
    topk_probabilities = tl.zeros((BLOCK_K,), dtype=tl.float32)
    topk_indices = tl.zeros((BLOCK_K,), dtype=tl.int32)
    for k in range(TOP_K):
        expert = tl.argmax(scores, axis=0, tie_break_left=True)
        probability = tl.sum(tl.where(expert_offsets == expert, probabilities, 0.0), axis=0)
        topk_probabilities = tl.where(k_offsets == k, probability, topk_probabilities)
        topk_indices = tl.where(k_offsets == k, expert, topk_indices)
        scores = tl.where(expert_offsets == expert, -float("inf"), scores)

    if norm_topk_prob:
        denominator = tl.sum(tl.where(k_offsets < TOP_K, topk_probabilities, 0.0), axis=0) + 1e-6
        topk_probabilities /= denominator
    topk_probabilities *= routed_scaling_factor

    output_mask = k_offsets < TOP_K
    output_offsets = token * TOP_K + k_offsets
    tl.store(selected_experts + output_offsets, topk_indices, mask=output_mask)
    tl.store(routing_weights + output_offsets, topk_probabilities, mask=output_mask)


@triton.jit
def _lfm2_moe_router_backward(
    grad_routing_weights,
    router_logits,
    selected_experts,
    grad_router_logits,
    n_tokens,
    n_experts,
    stride_token,
    routed_scaling_factor: tl.constexpr,
    norm_topk_prob: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
):
    token = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_EXPERTS)
    expert_mask = expert_offsets < n_experts
    k_offsets = tl.arange(0, BLOCK_K)
    mask = k_offsets < TOP_K
    offsets = token * TOP_K + k_offsets
    experts = tl.load(selected_experts + offsets, mask=mask, other=0).to(tl.int32)
    grad_weights = tl.load(grad_routing_weights + offsets, mask=mask, other=0.0).to(tl.float32)
    probabilities = tl.sigmoid(
        tl.load(router_logits + token * stride_token + experts, mask=mask, other=0.0).to(tl.float32)
    )

    if norm_topk_prob:
        denominator = tl.sum(probabilities, axis=0) + 1e-6
        weighted_grad_sum = tl.sum(grad_weights * probabilities, axis=0)
        grad_probabilities = routed_scaling_factor * (
            grad_weights / denominator - weighted_grad_sum / (denominator * denominator)
        )
    else:
        grad_probabilities = routed_scaling_factor * grad_weights

    grad_logits = grad_probabilities * probabilities * (1.0 - probabilities)
    dense_grad_logits = tl.sum(
        tl.where(
            expert_offsets[:, None] == experts[None, :],
            grad_logits[None, :],
            0.0,
        ),
        axis=1,
    )
    tl.store(
        grad_router_logits + token * stride_token + expert_offsets,
        dense_grad_logits,
        mask=expert_mask,
    )


class LigerLfm2MoeRouterFunction(torch.autograd.Function):
    """Fused sigmoid, biased top-k selection, and routing-weight normalization for LFM2-MoE."""

    @staticmethod
    def forward(ctx, router_logits, expert_bias, top_k, norm_topk_prob, routed_scaling_factor):
        if router_logits.ndim != 2:
            raise ValueError("router_logits must have shape [tokens, experts]")
        n_tokens, n_experts = router_logits.shape
        if not 0 < top_k <= n_experts:
            raise ValueError("top_k must be between one and the number of experts")

        selected_experts = torch.empty((n_tokens, top_k), dtype=torch.int32, device=router_logits.device)
        routing_weights = torch.empty((n_tokens, top_k), dtype=router_logits.dtype, device=router_logits.device)
        saved_bias = (
            expert_bias if expert_bias is not None else torch.empty(0, dtype=torch.float32, device=router_logits.device)
        )
        block_k = triton.next_power_of_2(top_k)
        _lfm2_moe_router_forward[(n_tokens,)](
            router_logits,
            saved_bias,
            selected_experts,
            routing_weights,
            n_tokens,
            n_experts,
            router_logits.stride(0),
            routed_scaling_factor=routed_scaling_factor,
            norm_topk_prob=norm_topk_prob,
            has_expert_bias=expert_bias is not None,
            TOP_K=top_k,
            BLOCK_K=block_k,
            BLOCK_EXPERTS=triton.next_power_of_2(n_experts),
        )
        ctx.save_for_backward(router_logits, selected_experts)
        ctx.n_experts = n_experts
        ctx.stride_token = router_logits.stride(0)
        ctx.top_k = top_k
        ctx.norm_topk_prob = norm_topk_prob
        ctx.routed_scaling_factor = routed_scaling_factor
        ctx.mark_non_differentiable(selected_experts)
        return selected_experts, routing_weights

    @staticmethod
    def backward(ctx, grad_selected_experts, grad_routing_weights):
        router_logits, selected_experts = ctx.saved_tensors
        n_tokens = selected_experts.shape[0]
        grad_router_logits = torch.empty(
            (n_tokens, ctx.n_experts), dtype=router_logits.dtype, device=router_logits.device
        )
        _lfm2_moe_router_backward[(n_tokens,)](
            grad_routing_weights,
            router_logits,
            selected_experts,
            grad_router_logits,
            n_tokens,
            ctx.n_experts,
            ctx.stride_token,
            routed_scaling_factor=ctx.routed_scaling_factor,
            norm_topk_prob=ctx.norm_topk_prob,
            TOP_K=ctx.top_k,
            BLOCK_K=triton.next_power_of_2(ctx.top_k),
            BLOCK_EXPERTS=triton.next_power_of_2(ctx.n_experts),
        )
        return grad_router_logits, None, None, None, None
