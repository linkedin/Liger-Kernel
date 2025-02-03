import pytest
import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOFunction
from liger_kernel.utils import infer_device
from test.utils import assert_verbose_allclose
from test.utils import set_seed

device = infer_device()

# set random seed globally
set_seed()


class TorchLMHeadGRPO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        beta: float = 0.1,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.beta = beta

    def forward(
        self,
        x,
        attention_mask,
        rewards,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
    ):
        # Forward pass through linear layer
        batch_size, seq_len, hidden_size = x.shape
        input_reshaped = x.view(-1, hidden_size)
        logits = (input_reshaped @ self.lin.weight.t()).view(batch_size, seq_len, -1)
        if self.lin.bias is not None:
            logits = logits + self.lin.bias

        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Get chosen token probabilities
        chosen_tokens = log_probs.argmax(dim=-1)
        chosen_token_logprobs = log_probs.gather(dim=-1, index=chosen_tokens.unsqueeze(-1)).squeeze(-1)

        # Get reference model probabilities
        if ref_input is not None and ref_weight is not None:
            with torch.no_grad():
                ref_input_reshaped = ref_input.view(-1, ref_input.size(-1))
                ref_logits = (ref_input_reshaped @ ref_weight.t()).view(batch_size, seq_len, -1)
                if ref_bias is not None:
                    ref_logits = ref_logits + ref_bias
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                ref_token_logprobs = ref_log_probs.gather(dim=-1, index=chosen_tokens.unsqueeze(-1)).squeeze(-1)
        else:
            ref_token_logprobs = chosen_token_logprobs.detach()

        # Compute advantages (exactly as in GRPOTrainer)
        mean_grouped_rewards = rewards.mean()
        std_grouped_rewards = rewards.std()
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-8)

        # Compute policy gradient loss with importance sampling ratio
        ratio = torch.exp(chosen_token_logprobs - chosen_token_logprobs.detach())
        policy_loss = -ratio * advantages.unsqueeze(1)

        # Compute KL penalty
        kl_div = (
            torch.exp(ref_token_logprobs - chosen_token_logprobs) - (ref_token_logprobs - chosen_token_logprobs) - 1.0
        )

        # Combine losses
        per_token_loss = policy_loss + self.beta * kl_div

        # Apply masking and normalize
        masked_loss = per_token_loss * attention_mask
        seq_lengths = attention_mask.sum(dim=1, keepdim=True)
        seq_lengths = torch.clamp(seq_lengths, min=1.0)
        loss = (masked_loss.sum(dim=1) / seq_lengths.squeeze(-1)).mean()

        # Compute metrics
        metrics = (
            chosen_token_logprobs.mean(),
            chosen_token_logprobs.std(),
            logits.mean(),
            (kl_div * attention_mask).sum(1).mean() / attention_mask.sum(1).mean(),
        )

        return loss, metrics


class LigerLMHeadGRPO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        beta: float = 0.1,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.grpo_loss = LigerFusedLinearGRPOFunction.apply
        self.beta = beta

    def forward(
        self,
        x,
        attention_mask,
        rewards,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
    ):
        # Pass only the arguments defined in LigerFusedLinearGRPOFunction.forward()
        return self.grpo_loss(
            x,  # _input
            self.lin.weight,  # weight
            attention_mask,  # attention_mask
            rewards,  # rewards
            self.lin.bias,  # bias
            ref_input,  # ref_input
            ref_weight,  # ref_weight
            ref_bias,  # ref_bias
            self.beta,  # beta
            True,  # compiled
            ref_input is not None,  # use_ref_model
        )


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (8, 128, 1024, 4096),
        (3, 47, 31, 123),  # random shape
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-2, 5e-2),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("beta", [0.1, 0.2])
def test_correctness(
    B,
    T,
    H,
    V,
    scalar,
    dtype,
    atol,
    rtol,
    bias,
    beta,
):
    torch_lm_head_grpo = TorchLMHeadGRPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        beta=beta,
    )
    liger_lm_head_grpo = LigerLMHeadGRPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        beta=beta,
    )

    # Initialize weights
    torch_lm_head_grpo.lin.weight.data = liger_lm_head_grpo.lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )
    if bias:
        torch_lm_head_grpo.lin.bias.data = liger_lm_head_grpo.lin.bias.data = torch.randn(V, device=device, dtype=dtype)

    # Create inputs
    _input = torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)

    # Create attention mask with random padding
    attention_mask = torch.ones(B, T, device=device)
    num_elements_to_mask = torch.randint(1, B * T // 2, (1,)).item()
    mask_indices = torch.randperm(B * T)[:num_elements_to_mask]
    attention_mask.view(-1)[mask_indices] = 0

    # Create rewards with random values
    rewards = torch.randn(B, device=device, dtype=dtype)

    # Create reference inputs (optional)
    ref_input = torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    ref_weight = torch.randn(V, H, device=device, dtype=dtype)
    ref_bias = torch.randn(V, device=device, dtype=dtype) if bias else None

    # Forward pass with reference model
    loss1, aux1 = torch_lm_head_grpo(
        input1, attention_mask, rewards, ref_input=ref_input, ref_weight=ref_weight, ref_bias=ref_bias
    )
    loss2, aux2 = liger_lm_head_grpo(
        input2, attention_mask, rewards, ref_input=ref_input, ref_weight=ref_weight, ref_bias=ref_bias
    )

    # Check losses match
    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    # Check metrics match
    assert len(aux1) == len(aux2)
    for metric1, metric2 in zip(aux1, aux2):
        assert_verbose_allclose(metric1, metric2, atol=atol, rtol=rtol)

    # Backward pass
    loss1.backward()
    loss2.backward()

    # Check gradients match
    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(
        torch_lm_head_grpo.lin.weight.grad,
        liger_lm_head_grpo.lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    if bias:
        assert_verbose_allclose(
            torch_lm_head_grpo.lin.bias.grad,
            liger_lm_head_grpo.lin.bias.grad,
            atol=atol,
            rtol=rtol,
        )

    # Test without reference model
    loss1, aux1 = torch_lm_head_grpo(input1, attention_mask, rewards)
    loss2, aux2 = liger_lm_head_grpo(input2, attention_mask, rewards)

    # Check losses match (without reference model)
    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    # Check metrics match (without reference model)
    assert len(aux1) == len(aux2)
    for metric1, metric2 in zip(aux1, aux2):
        assert_verbose_allclose(metric1, metric2, atol=atol, rtol=rtol)
