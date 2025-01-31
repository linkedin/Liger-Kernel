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
        # Get policy logits and log probs
        batch_size, seq_len, hidden_size = x.shape
        input_reshaped = x.view(-1, hidden_size)
        logits = (input_reshaped @ self.lin.weight.t()).view(batch_size, seq_len, -1)
        if self.lin.bias is not None:
            logits = logits + self.lin.bias
        log_probs = F.log_softmax(logits, dim=-1)

        # Get sequence-level log probs by taking max over vocab and summing over sequence
        seq_log_probs = log_probs.max(dim=-1).values
        policy_seq_logps = (seq_log_probs * attention_mask).sum(dim=-1)

        # Get reference model log probs if provided
        if ref_input is not None and ref_weight is not None:
            with torch.no_grad():
                ref_input_reshaped = ref_input.view(-1, ref_input.size(-1))
                ref_logits = (ref_input_reshaped @ ref_weight.t()).view(batch_size, seq_len, -1)
                if ref_bias is not None:
                    ref_logits = ref_logits + ref_bias
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                ref_seq_log_probs = ref_log_probs.max(dim=-1).values
                ref_seq_logps = (ref_seq_log_probs * attention_mask).sum(dim=-1)
        else:
            ref_seq_logps = policy_seq_logps.detach()

        # Compute advantages
        advantages = rewards - rewards.mean()
        if advantages.std() > 0:
            advantages = advantages / advantages.std()

        # Policy gradient loss
        policy_loss = -(advantages * policy_seq_logps)

        # KL penalty
        kl_div = policy_seq_logps - ref_seq_logps

        # Total loss
        loss = policy_loss + self.beta * kl_div

        # Return metrics for logging
        metrics = (
            policy_seq_logps.mean(),  # policy log probs mean
            policy_seq_logps.std(),  # policy log probs std
            logits.mean(),  # policy logits mean
            kl_div.mean(),  # KL divergence mean
        )

        return loss.mean(), metrics


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

    def forward(
        self,
        x,
        attention_mask,
        rewards,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
    ):
        return self.grpo_loss(
            x,
            self.lin.weight,
            rewards,
            attention_mask,
            self.lin.bias,
            loss_fn=LigerFusedLinearGRPOFunction.preference_loss_fn,
            chunk_size=1,
            beta=self.beta,
            compiled=True,
            use_ref_model=ref_input is not None,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
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

    # Create rewards
    rewards = torch.randn(B, device=device, dtype=dtype)

    # Forward pass
    loss1, aux1 = torch_lm_head_grpo(input1, attention_mask, rewards)
    loss2, aux2 = liger_lm_head_grpo(input2, attention_mask, rewards)

    # Check losses match
    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)
    assert len(aux1) == len(aux2)

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
