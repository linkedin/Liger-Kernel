import pytest
import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
from liger_kernel.chunked_loss.functional import liger_fused_linear_grpo
from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOFunction
from liger_kernel.transformers.grpo_loss import _reduce_grpo_loss
from liger_kernel.transformers.grpo_loss import triton_grpo_loss
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
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.2,
        temperature: float = 1.0,
        use_ref_model: bool = True,
        loss_type: str = "bnpo",
        max_completion_length: int | None = None,
        importance_sampling_level: str = "token",
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.beta = beta
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.temperature = temperature
        self.use_ref_model = use_ref_model
        self.loss_type = loss_type
        self.max_completion_length = max_completion_length
        self.importance_sampling_level = importance_sampling_level
        if self.loss_type == "dr_grpo":
            assert self.max_completion_length is not None, "max_completion_length must be provided for dr_grpo"

    @staticmethod
    def compute_per_token_components(
        per_token_logps,
        attention_mask,
        advantages,
        old_per_token_logps,
        ref_per_token_logps,
        epsilon_low,
        epsilon_high,
        beta,
        importance_sampling_level,
    ):
        attention_mask = attention_mask.to(per_token_logps.dtype)
        old_per_token_logps = (
            old_per_token_logps.float() if old_per_token_logps is not None else per_token_logps.detach()
        )
        log_ratio = per_token_logps - old_per_token_logps

        if importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * attention_mask).sum(-1) / attention_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
        expanded_advantages = advantages.unsqueeze(1)
        per_token_loss1 = coef_1 * expanded_advantages
        per_token_loss2 = coef_2 * expanded_advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        kl_div = None
        if beta != 0.0:
            ref_per_token_logps = ref_per_token_logps.float()
            kl_div = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1.0
            per_token_loss = per_token_loss + beta * kl_div

        if importance_sampling_level == "token":
            is_clipped = ((coef_1 < 1 - epsilon_low) & (expanded_advantages < 0)) | (
                (coef_1 > 1 + epsilon_high) & (expanded_advantages > 0)
            )
        else:  # sequence level
            # For sequence level, coef_1 is shape (B, 1), advantages is shape (B,)
            seq_advantages = advantages
            is_clipped = ((coef_1.squeeze(-1) < 1 - epsilon_low) & (seq_advantages < 0)) | (
                (coef_1.squeeze(-1) > 1 + epsilon_high) & (seq_advantages > 0)
            )
            is_clipped = is_clipped.unsqueeze(1).expand_as(attention_mask)
        return per_token_loss, kl_div, is_clipped

    def forward(
        self,
        x,  # Shape: [batch_size, seq_len, hidden_size]
        selected_token_ids,  # Shape: [batch_size, seq_len]
        attention_mask,  # Shape: [batch_size, seq_len]
        advantages,  # Shape: [batch_size,]
        ref_per_token_logps=None,  # Shape: [batch_size, seq_len]
        old_per_token_logps=None,
        ref_input=None,  # Shape: [batch_size, seq_len, hidden_size]
    ):
        logits = x @ self.lin.weight.t()
        if self.lin.bias is not None:
            logits = logits + self.lin.bias.float()
        if self.temperature != 1.0:
            logits = logits / self.temperature
        # Get log probabilities
        log_probs = F.log_softmax(logits.float(), dim=-1)

        # Get chosen token probabilities
        per_token_logps = log_probs.gather(dim=-1, index=selected_token_ids.unsqueeze(-1)).squeeze(-1)

        # Get reference model probabilities,
        if ref_per_token_logps is None:
            if self.use_ref_model:
                with torch.no_grad():
                    ref_logits = ref_input @ self.ref_lin.weight.t()
                    if self.ref_lin.bias is not None:
                        ref_logits = ref_logits + self.ref_lin.bias.float()
                    if self.temperature != 1.0:
                        ref_logits = ref_logits / self.temperature
                    ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
                    ref_per_token_logps = ref_log_probs.gather(dim=-1, index=selected_token_ids.unsqueeze(-1)).squeeze(
                        -1
                    )
            else:
                ref_per_token_logps = per_token_logps.detach()

        per_token_loss, kl_div, is_clipped = self.compute_per_token_components(
            per_token_logps,
            attention_mask,
            advantages,
            old_per_token_logps,
            ref_per_token_logps,
            self.epsilon_low,
            self.epsilon_high,
            self.beta,
            self.importance_sampling_level,
        )

        # Apply masking and calculate loss based on loss_type
        if self.loss_type == "grpo":
            loss = ((per_token_loss * attention_mask).sum(-1) / torch.clamp(attention_mask.sum(-1), min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * attention_mask).sum() / torch.clamp(attention_mask.sum(), min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * attention_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        elif self.loss_type == "dapo":
            normalizer = attention_mask.sum().clamp(min=1.0)
            loss = (per_token_loss * attention_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Compute metrics
        metrics = []
        if self.beta != 0.0:
            metrics.append(((kl_div * attention_mask).sum() / torch.clamp(attention_mask.sum(), min=1.0)))
        metrics.append((is_clipped.float() * attention_mask).sum() / torch.clamp(attention_mask.sum(), min=1.0))
        return loss, metrics


class LigerLMHeadGRPO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        beta: float = 0.1,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.2,
        temperature: float = 1.0,
        use_ref_model: bool = True,
        loss_type: str = "bnpo",
        max_completion_length: int | None = None,
        importance_sampling_level: str = "token",
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.grpo_loss = LigerFusedLinearGRPOLoss(
            beta=beta,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            temperature=temperature,
            use_ref_model=use_ref_model,
            compiled=True,
            loss_type=loss_type,
            max_completion_length=max_completion_length,
            importance_sampling_level=importance_sampling_level,
        )

    def forward(
        self,
        x,
        selected_token_ids,
        attention_mask,
        advantages,
        ref_per_token_logps=None,
        old_per_token_logps=None,
        ref_input=None,
    ):
        # Pass only the arguments defined in LigerFusedLinearGRPOFunction.forward()
        return self.grpo_loss(
            x,  # _input
            self.lin.weight,  # weight
            selected_token_ids,  # selected_token_ids
            attention_mask,  # attention_mask
            advantages,  # advantages
            self.lin.bias,  # bias
            ref_per_token_logps,  # ref_per_token_logps
            old_per_token_logps,  # old_per_token_logps
            ref_input,  # ref_input
            self.ref_lin.weight,  # ref_weight
            self.ref_lin.bias,  # ref_bias
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
        (1.0, torch.bfloat16, 5e-2, 5e-1),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "beta, epsilon_low, epsilon_high, temperature",
    [
        # Standard settings
        (0.1, 0.2, 0.2, 1.0),
        (0.0, 0.1, 0.1, 2.0),
    ],
)
@pytest.mark.parametrize(
    "use_ref_model, use_ref_per_token_logps, old_per_token_logps",
    [
        (True, True, True),
        (True, False, False),
        (False, False, True),
    ],
)
@pytest.mark.parametrize("loss_type", ["bnpo", "grpo", "dr_grpo", "dapo"])
@pytest.mark.parametrize("importance_sampling_level", ["token", "sequence"])
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
    epsilon_low,
    epsilon_high,
    temperature,
    use_ref_per_token_logps,
    use_ref_model,
    old_per_token_logps,
    loss_type,
    importance_sampling_level,
):
    # Reset torch compiler cache for each parameter of the test case
    torch.compiler.reset()
    max_completion_length = T if loss_type == "dr_grpo" else None

    torch_lm_head_grpo = TorchLMHeadGRPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        beta=beta,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        temperature=temperature,
        use_ref_model=use_ref_model,
        loss_type=loss_type,
        max_completion_length=max_completion_length,
        importance_sampling_level=importance_sampling_level,
    )
    liger_lm_head_grpo = LigerLMHeadGRPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        beta=beta,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        temperature=temperature,
        use_ref_model=use_ref_model,
        loss_type=loss_type,
        max_completion_length=max_completion_length,
        importance_sampling_level=importance_sampling_level,
    )

    # Initialize weights
    torch_lm_head_grpo.lin.weight.data = liger_lm_head_grpo.lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )
    if bias:
        torch_lm_head_grpo.lin.bias.data = liger_lm_head_grpo.lin.bias.data = torch.randn(V, device=device, dtype=dtype)

    # set ref weights to be close to the original weights
    torch_lm_head_grpo.ref_lin.weight.data = liger_lm_head_grpo.ref_lin.weight.data = (
        torch_lm_head_grpo.lin.weight.data + torch.randn(V, H, device=device, dtype=dtype) * 0.01
    )
    if bias:
        torch_lm_head_grpo.ref_lin.bias.data = liger_lm_head_grpo.ref_lin.bias.data = (
            torch_lm_head_grpo.lin.bias.data + torch.randn(V, device=device, dtype=dtype) * 0.01
        )

    # Create inputs with shape [B, T, H]
    _input = torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)

    # Create selected token ids with shape [B, T]
    selected_token_ids = torch.randint(0, V, (B, T), device=device)

    # Compute per-token logps
    with torch.no_grad():
        logits = _input @ torch_lm_head_grpo.lin.weight.t()
        if torch_lm_head_grpo.lin.bias is not None:
            logits = logits + torch_lm_head_grpo.lin.bias
        logits = logits / temperature
        logps = F.log_softmax(logits.float(), dim=-1)
        per_token_logps = logps.gather(dim=-1, index=selected_token_ids.unsqueeze(-1)).squeeze(-1)

    # Create attention mask with random padding [B, T]
    attention_mask = torch.ones(B, T, device=device)
    num_elements_to_mask = torch.randint(1, B * T // 2, (1,)).item()
    mask_indices = torch.randperm(B * T)[:num_elements_to_mask]
    attention_mask.view(-1)[mask_indices] = 0

    # Create advantages with shape [B]
    advantages = torch.rand(B, device=device, dtype=dtype)

    ref_per_token_logps = None
    ref_input = None
    if use_ref_model and use_ref_per_token_logps:
        # Create reference log probs with shape [B, T]
        ref_per_token_logps = per_token_logps.detach() + torch.randn(B, T, device=device) * 0.01
    elif use_ref_model:
        # Create reference inputs (optional) with shape [B, T, H] if ref_log_probs is None
        ref_input = _input.detach() + torch.randn(B, T, H, device=device, dtype=dtype) * 0.01

    if old_per_token_logps:
        old_per_token_logps = per_token_logps.detach() + torch.randn(B, T, device=device) * 0.01
    else:
        old_per_token_logps = None

    # Forward pass with reference model
    loss1, aux1 = torch_lm_head_grpo(
        input1,
        selected_token_ids,
        attention_mask,
        advantages,
        ref_per_token_logps=ref_per_token_logps,
        old_per_token_logps=old_per_token_logps,
        ref_input=ref_input,
    )
    loss2, aux2 = liger_lm_head_grpo(
        input2,
        selected_token_ids,
        attention_mask,
        advantages,
        ref_per_token_logps=ref_per_token_logps,
        old_per_token_logps=old_per_token_logps,
        ref_input=ref_input,
    )
    # Check losses match
    assert not torch.isnan(loss1)
    assert not torch.isnan(loss2)
    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    # Check metrics match
    assert len(aux1) == len(aux2)
    # aggregated metrics are unstable for bfloat16
    for metric1, metric2 in zip(aux1, aux2):
        assert_verbose_allclose(metric1, metric2, atol=atol, rtol=rtol)

    # Backward pass
    loss1.backward()
    loss2.backward()

    # Check gradients match for loss_type
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
        (1.0, torch.bfloat16, 5e-2, 5e-1),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
def test_functional_correctness(
    B,
    T,
    H,
    V,
    scalar,
    dtype,
    atol,
    rtol,
    bias,
):
    # Reset torch compiler cache for each parameter of the test case
    torch.compiler.reset()
    max_completion_length = T
    _input = torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)

    _weight = torch.randn(V, H, device=device, dtype=dtype) * scalar
    weight1 = _weight.detach().clone().requires_grad_(True)
    weight2 = _weight.detach().clone().requires_grad_(True)

    selected_token_ids = torch.randint(0, V, (B, T), device=device)

    attention_mask = torch.ones(B, T, device=device)

    advantages = torch.rand(B, device=device, dtype=dtype)

    if bias:
        _bias = torch.randn(V, device=device, dtype=dtype) * scalar
        bias1 = _bias.detach().clone().requires_grad_(True)
        bias2 = _bias.detach().clone().requires_grad_(True)
    else:
        bias1 = None
        bias2 = None

    ref_input = torch.randn(B, T, H, device=device, dtype=dtype) * scalar

    _ref_weight = _weight.detach() + torch.randn(V, H, device=device, dtype=dtype) * 0.01
    ref_weight1 = _ref_weight.detach().clone().requires_grad_(True)
    ref_weight2 = _ref_weight.detach().clone().requires_grad_(True)

    if bias:
        _ref_bias = _bias.detach() + torch.randn(V, device=device, dtype=dtype) * 0.01
        ref_bias1 = _ref_bias.detach().clone().requires_grad_(True)
        ref_bias2 = _ref_bias.detach().clone().requires_grad_(True)
    else:
        ref_bias1 = None
        ref_bias2 = None

    old_per_token_logps = None
    ref_per_token_logps = None

    loss1, aux1 = liger_fused_linear_grpo(
        input1,
        weight1,
        selected_token_ids,
        attention_mask,
        advantages,
        bias1,
        ref_per_token_logps,
        old_per_token_logps,
        ref_input,
        ref_weight1,
        ref_bias1,
        0.04,
        0.2,
        0.2,
        "bnpo",
        max_completion_length,
        "token",
        1.0,
        False,
        True,
        1,
    )

    loss2, aux2 = LigerFusedLinearGRPOFunction.apply(
        input2,
        weight2,
        selected_token_ids,
        attention_mask,
        advantages,
        bias2,
        ref_per_token_logps,
        old_per_token_logps,
        ref_input,
        ref_weight2,
        ref_bias2,
        0.04,
        0.2,
        0.2,
        "bnpo",
        max_completion_length,
        "token",
        1.0,
        False,
        True,
        1,
    )

    assert not torch.isnan(loss1)
    assert not torch.isnan(loss2)
    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    # Check metrics match
    assert len(aux1) == len(aux2)
    # aggregated metrics are unstable for bfloat16
    for metric1, metric2 in zip(aux1, aux2):
        assert_verbose_allclose(metric1, metric2, atol=atol, rtol=rtol)


@pytest.mark.parametrize("loss_type", ["grpo", "bnpo", "dr_grpo", "dapo"])
def test_reduce_grpo_loss_matches_reference(loss_type):
    torch.manual_seed(0)
    per_token_loss = torch.randn(3, 5)
    mask = torch.randint(0, 2, (3, 5), device=per_token_loss.device, dtype=torch.long)
    mask[:, 0] = 1  # ensure at least one valid token per sequence
    max_completion_length = 5 if loss_type == "dr_grpo" else None

    reduced = _reduce_grpo_loss(per_token_loss, mask, loss_type, max_completion_length)

    mask_f = mask.to(per_token_loss.dtype)
    if loss_type == "grpo":
        expected = ((per_token_loss * mask_f).sum(-1) / mask_f.sum(-1).clamp(min=1.0)).mean()
    elif loss_type == "bnpo":
        expected = (per_token_loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)
    elif loss_type == "dr_grpo":
        expected = (per_token_loss * mask_f).sum() / (per_token_loss.size(0) * max_completion_length)
    else:  # dapo
        expected = (per_token_loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)

    assert_verbose_allclose(reduced, expected)


def test_reduce_grpo_loss_requires_max_completion_length():
    per_token_loss = torch.randn(2, 3)
    mask = torch.ones_like(per_token_loss, dtype=torch.long)
    with pytest.raises(ValueError):
        _reduce_grpo_loss(per_token_loss, mask, "dr_grpo", max_completion_length=None)


@pytest.mark.parametrize("loss_type,beta", [("bnpo", 0.0), ("dapo", 0.04)])
def test_triton_grpo_loss_matches_reference(loss_type, beta):
    pytest.importorskip("triton")
    device = infer_device()

    B, T, V = 2, 4, 16
    logits = torch.randn(B, T + 1, V, device=device, dtype=torch.float32).contiguous()
    completion_ids = torch.randint(0, V, (B, T), device=device)
    completion_mask = torch.randint(0, 2, (B, T), device=device, dtype=torch.long)
    completion_mask[:, 0] = 1  # ensure each sequence has at least one valid token
    advantages = torch.randn(B, device=device, dtype=torch.float32)
    old_logp = torch.randn(B, T, device=device, dtype=torch.float32)
    ref_logp = torch.randn(B, T, device=device, dtype=torch.float32) if beta != 0.0 else None

    per_token_loss, per_token_kl, is_clipped = triton_grpo_loss(
        logits=logits,
        old_logp=old_logp,
        ref_logp=ref_logp,
        completion_ids=completion_ids,
        advantages=advantages,
        completion_mask=completion_mask,
        temperature=1.0,
        beta=beta,
        eps_low=0.2,
        eps_high=0.2,
        inplace=False,
        loss_type=loss_type,
        max_completion_length=T,
        reduce=False,
    )

    logits_main = logits[:, :-1, :]
    log_probs = torch.log_softmax(logits_main, dim=-1)
    per_token_logps = log_probs.gather(dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)
    ref_tokens = ref_logp if ref_logp is not None else per_token_logps.detach()
    reference_loss, reference_kl, reference_is_clipped = TorchLMHeadGRPO.compute_per_token_components(
        per_token_logps,
        completion_mask.float(),
        advantages,
        old_logp,
        ref_tokens,
        0.2,
        0.2,
        beta,
        "token",
    )

    mask = completion_mask.float()
    mask_bool = mask.bool()
    assert_verbose_allclose(per_token_loss, reference_loss * mask)
    assert torch.equal(is_clipped.bool()[mask_bool], reference_is_clipped[mask_bool])
    if beta != 0.0:
        assert_verbose_allclose(per_token_kl, reference_kl * mask)
    else:
        assert per_token_kl is None

    reduced_loss, metrics = triton_grpo_loss(
        logits=logits,
        old_logp=old_logp,
        ref_logp=ref_logp,
        completion_ids=completion_ids,
        advantages=advantages,
        completion_mask=completion_mask,
        temperature=1.0,
        beta=beta,
        eps_low=0.2,
        eps_high=0.2,
        inplace=False,
        loss_type=loss_type,
        max_completion_length=T,
        reduce=True,
    )
    expected_loss = _reduce_grpo_loss(reference_loss, completion_mask, loss_type, T)
    assert_verbose_allclose(reduced_loss, expected_loss)
    if beta != 0.0:
        assert_verbose_allclose(metrics[0], _masked_mean(reference_kl, completion_mask))
        clip_metric = metrics[1]
    else:
        clip_metric = metrics[0]
    assert_verbose_allclose(clip_metric, _masked_mean(reference_is_clipped.float(), completion_mask))


def _reference_per_token_loss(
    logits,
    completion_ids,
    completion_mask,
    advantages,
    old_logp,
    ref_logp,
    beta,
    eps_low,
    eps_high,
    temperature=1.0,
):
    logits = logits[:, :-1, :] / temperature
    log_probs = torch.log_softmax(logits, dim=-1)
    per_token_logps = log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
    old = old_logp if old_logp is not None else per_token_logps.detach()
    coef_1 = torch.exp(per_token_logps - old)
    coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)
    per_token_loss1 = coef_1 * advantages.unsqueeze(1)
    per_token_loss2 = coef_2 * advantages.unsqueeze(1)
    per_token_loss = -torch.minimum(per_token_loss1, per_token_loss2)
    is_clipped = per_token_loss1 < per_token_loss2
    mask = completion_mask.to(torch.bool)
    per_token_loss = per_token_loss.masked_fill(~mask, 0.0)
    is_clipped = is_clipped & mask
    if beta != 0.0:
        kl = torch.exp(ref_logp - per_token_logps) - (ref_logp - per_token_logps) - 1.0
        kl = kl.masked_fill(~mask, 0.0)
        per_token_loss = per_token_loss + beta * kl
    else:
        kl = None
    return {
        "per_token_loss": per_token_loss,
        "kl": kl,
        "is_clipped": is_clipped,
    }


def _masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=1.0)
