import pytest
import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
from liger_kernel.chunked_loss.functional import liger_fused_linear_grpo
from liger_kernel.chunked_loss.fused_linear_ppo import LigerFusedLinearPPOBase
from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOFunction
from liger_kernel.transformers.grpo_loss import _reduce_grpo_loss
from liger_kernel.transformers.grpo_loss import triton_grpo_loss
from liger_kernel.utils import infer_device
from test.utils import assert_verbose_allclose
from test.utils import set_seed

device = infer_device()

# set random seed globally
set_seed()


def sapo_loss_fn(importance_ratio: torch.Tensor, temperature: float) -> torch.Tensor:
    """SAPO (Soft Adaptive Policy Optimization) loss function for torch reference.

    Reference: https://huggingface.co/papers/2511.20347
    TRL implementation: https://github.com/huggingface/trl/blob/1bd2a52ec2d8344050af736d60cdc735181ae4b8/trl/trainer/grpo_trainer.py#L1913
    """
    if temperature <= 0:
        raise ValueError("sapo_temperature must be > 0.")
    sigmoid_input = temperature * (importance_ratio - 1)
    sigmoid_smoothed_loss = torch.sigmoid(sigmoid_input)
    return sigmoid_smoothed_loss * 4 / temperature


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
        sapo_temperature_pos: float = 1.0,
        sapo_temperature_neg: float = 1.05,
        delta: float | None = None,
        use_bias_correction_kl: bool = False,
        vespo_k_pos: float = 2.0,
        vespo_lambda_pos: float = 3.0,
        vespo_k_neg: float = 3.0,
        vespo_lambda_neg: float = 2.0,
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
        self.sapo_temperature_pos = sapo_temperature_pos
        self.sapo_temperature_neg = sapo_temperature_neg
        self.delta = delta
        self.use_bias_correction_kl = use_bias_correction_kl
        self.vespo_k_pos = vespo_k_pos
        self.vespo_lambda_pos = vespo_lambda_pos
        self.vespo_k_neg = vespo_k_neg
        self.vespo_lambda_neg = vespo_lambda_neg
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
        loss_type: str = "grpo",
        sapo_temperature_pos: float = 1.0,
        sapo_temperature_neg: float = 1.05,
        vllm_is_ratio=None,
        delta=None,
        use_bias_correction_kl=False,
        vespo_k_pos: float = 2.0,
        vespo_lambda_pos: float = 3.0,
        vespo_k_neg: float = 3.0,
        vespo_lambda_neg: float = 2.0,
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
        expanded_advantages = advantages.unsqueeze(1)

        if loss_type == "sapo":
            # SAPO: Soft Adaptive Policy Optimization
            # Uses sigmoid-based soft gating instead of hard clipping
            # Reference: https://github.com/huggingface/trl/blob/1bd2a52ec2d8344050af736d60cdc735181ae4b8/trl/trainer/grpo_trainer.py#L2037-L2046
            per_token_loss = torch.empty_like(coef_1)
            advantages_expanded = expanded_advantages.expand_as(coef_1)
            positive_advantages_mask = advantages_expanded > 0

            per_token_loss[positive_advantages_mask] = sapo_loss_fn(
                coef_1[positive_advantages_mask], sapo_temperature_pos
            )
            per_token_loss[~positive_advantages_mask] = sapo_loss_fn(
                coef_1[~positive_advantages_mask], sapo_temperature_neg
            )
            per_token_loss = -per_token_loss * advantages_expanded
            # SAPO doesn't use clipping metrics
            is_lower_clipped = torch.zeros_like(coef_1, dtype=torch.bool)
            is_upper_clipped = torch.zeros_like(coef_1, dtype=torch.bool)
        elif loss_type == "vespo":
            # VESPO: Value-Enhanced Sequence-level Policy Optimization.
            # phi_seq is detached, acts as a gradient-scaling coefficient on per_token_logps.
            from liger_kernel.chunked_loss.grpo_loss import get_gamma_weights

            phi_seq = get_gamma_weights(
                advantages=advantages,
                log_ratio_per_token=log_ratio,
                mask=attention_mask,
                importance_sampling_ratio=vllm_is_ratio,
                k_pos=vespo_k_pos,
                lambda_pos=vespo_lambda_pos,
                k_neg=vespo_k_neg,
                lambda_neg=vespo_lambda_neg,
            )
            per_token_loss = -phi_seq * expanded_advantages * per_token_logps
            is_lower_clipped = torch.zeros_like(coef_1, dtype=torch.bool)
            is_upper_clipped = torch.zeros_like(coef_1, dtype=torch.bool)
        elif loss_type == "cispo":
            # CISPO: clip and detach the importance weights
            upper_bound = epsilon_high
            lower_bound = None
            coef_2 = torch.clamp(coef_1, lower_bound, upper_bound).detach()
            is_lower_clipped = torch.zeros_like(coef_1, dtype=torch.bool)
            is_upper_clipped = coef_1 > upper_bound
            # CISPO: clip and detach the importance weights, multiply by log probs
            # Reference: https://github.com/huggingface/trl/blob/035c3ff151b953ca72cdfe0ee966bc1469a26fde/trl/trainer/grpo_trainer.py#L2030
            per_token_loss = -coef_2 * expanded_advantages * per_token_logps
        else:
            upper_bound = 1 + epsilon_high
            lower_bound = 1 - epsilon_low
            coef_2 = torch.clamp(coef_1, lower_bound, upper_bound)
            is_lower_clipped = coef_1 < lower_bound
            is_upper_clipped = coef_1 > upper_bound
            if delta is not None:
                coef_1 = torch.clamp(coef_1, max=delta)
            per_token_loss1 = coef_1 * expanded_advantages
            per_token_loss2 = coef_2 * expanded_advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Apply vLLM importance sampling correction BEFORE KL penalty.
        # VESPO folds this into phi_seq in log space, so we skip it here.
        if vllm_is_ratio is not None and loss_type != "vespo":
            per_token_loss = per_token_loss * vllm_is_ratio

        kl_div = None
        if beta != 0.0:
            ref_per_token_logps = ref_per_token_logps.float()
            kl_div = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1.0
            if use_bias_correction_kl:
                # TRL: per_token_kl *= coef_1 with coef_1 reflecting importance_sampling_level.
                kl_div = kl_div * torch.exp(log_importance_weights)
            per_token_loss = per_token_loss + beta * kl_div

        # Adjust clipping metric calculation based on importance sampling level
        if importance_sampling_level == "token":
            is_clipped = (is_lower_clipped & (expanded_advantages < 0)) | (is_upper_clipped & (expanded_advantages > 0))
        else:  # sequence level
            # For sequence level, coef_1 is shape (B, 1), advantages is shape (B,)
            is_clipped = (is_lower_clipped & (expanded_advantages < 0)) | (is_upper_clipped & (expanded_advantages > 0))
            is_clipped = is_clipped.expand_as(attention_mask)
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
        vllm_is_ratio=None,  # Shape: [batch_size, seq_len] or None
    ):
        logits = x @ self.lin.weight.t()
        if self.lin.bias is not None:
            logits = logits + self.lin.bias
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
            self.loss_type,
            self.sapo_temperature_pos,
            self.sapo_temperature_neg,
            vllm_is_ratio=vllm_is_ratio,
            delta=self.delta,
            use_bias_correction_kl=self.use_bias_correction_kl,
            vespo_k_pos=self.vespo_k_pos,
            vespo_lambda_pos=self.vespo_lambda_pos,
            vespo_k_neg=self.vespo_k_neg,
            vespo_lambda_neg=self.vespo_lambda_neg,
        )

        # Apply masking and calculate loss based on loss_type
        if self.loss_type == "grpo" or self.loss_type == "sapo":
            # SAPO uses same normalization as GRPO (per-sequence)
            loss = ((per_token_loss * attention_mask).sum(-1) / torch.clamp(attention_mask.sum(-1), min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * attention_mask).sum() / torch.clamp(attention_mask.sum(), min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * attention_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        elif self.loss_type == "dapo":
            normalizer = LigerFusedLinearPPOBase._compute_dapo_normalizer(attention_mask)
            loss = (per_token_loss * attention_mask).sum() / normalizer
        elif self.loss_type == "cispo":
            normalizer = attention_mask.sum().clamp(min=1.0)
            loss = (per_token_loss * attention_mask).sum() / normalizer
        elif self.loss_type == "vespo":
            normalizer = LigerFusedLinearPPOBase._compute_dapo_normalizer(attention_mask)
            loss = (per_token_loss * attention_mask).sum() / normalizer
        elif self.loss_type == "luspo":
            loss = (per_token_loss * attention_mask.sum(-1, keepdim=True)).mean()
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
        sapo_temperature_pos: float = 1.0,
        sapo_temperature_neg: float = 1.05,
        delta: float | None = None,
        use_bias_correction_kl: bool = False,
        vespo_k_pos: float = 2.0,
        vespo_lambda_pos: float = 3.0,
        vespo_k_neg: float = 3.0,
        vespo_lambda_neg: float = 2.0,
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
            sapo_temperature_pos=sapo_temperature_pos,
            sapo_temperature_neg=sapo_temperature_neg,
            delta=delta,
            use_bias_correction_kl=use_bias_correction_kl,
            vespo_k_pos=vespo_k_pos,
            vespo_lambda_pos=vespo_lambda_pos,
            vespo_k_neg=vespo_k_neg,
            vespo_lambda_neg=vespo_lambda_neg,
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
        vllm_is_ratio=None,
        num_items_in_batch=None,
    ):
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
            vllm_is_ratio=vllm_is_ratio,
            num_items_in_batch=num_items_in_batch,
        )


@pytest.mark.parametrize("dtype, atol, rtol", [(torch.float32, 1e-5, 1e-5), (torch.bfloat16, 1e-1, 1e-1)])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (3, 17, 31, 123),  # small: no chunking exercised
        (1, 4096, 256, 5000),  # large: exercises both sequence and vocab chunking
    ],
)
def test_selective_chunk_forward_matches_reference(B, T, H, V, dtype, atol, rtol, bias):
    set_seed()
    x = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(V, H, device=device, dtype=dtype, requires_grad=True)
    bias_tensor = torch.randn(V, device=device, dtype=dtype, requires_grad=True) if bias else None
    selected_token_ids = torch.randint(0, V, (B, T), device=device)

    out = LigerFusedLinearPPOBase.chunk_forward(x, weight, selected_token_ids, bias=bias_tensor, temperature=0.9)

    logits = x @ weight.t()
    if bias_tensor is not None:
        logits = logits + bias_tensor
    ref = torch.log_softmax((logits / 0.9).float(), dim=-1).gather(-1, selected_token_ids.unsqueeze(-1)).squeeze(-1)

    assert_verbose_allclose(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("loss_type", ["dapo", "grpo"])
@pytest.mark.parametrize("compiled", [True, False])
def test_correctness_large_seq_exercises_chunking(loss_type, compiled):
    """Test with N > seq_chunk_size and V > vocab_chunk_size to exercise both chunking loops."""
    set_seed()
    torch.compiler.reset()
    B, T, H, V = 1, 4096, 256, 5000
    dtype = torch.float32

    torch_lm = TorchLMHeadGRPO(H=H, V=V, dtype=dtype, beta=0.04, loss_type=loss_type, use_ref_model=False)
    liger_lm = LigerLMHeadGRPO(H=H, V=V, dtype=dtype, beta=0.04, loss_type=loss_type, use_ref_model=False)

    torch_lm.lin.weight.data = liger_lm.lin.weight.data = torch.randn(V, H, device=device, dtype=dtype)

    _input = torch.randn(B, T, H, device=device, dtype=dtype)
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)
    selected_token_ids = torch.randint(0, V, (B, T), device=device)
    attention_mask = torch.ones(B, T, device=device)
    attention_mask[:, -64:] = 0
    advantages = torch.randn(B, device=device, dtype=dtype)

    loss1, _ = torch_lm(input1, selected_token_ids, attention_mask, advantages)
    loss2, _ = liger_lm(input2, selected_token_ids, attention_mask, advantages)

    assert_verbose_allclose(loss1, loss2, atol=2e-5, rtol=1e-3)

    loss1.backward()
    loss2.backward()
    assert_verbose_allclose(input1.grad, input2.grad, atol=2e-5, rtol=1e-3)
    assert_verbose_allclose(torch_lm.lin.weight.grad, liger_lm.lin.weight.grad, atol=2e-5, rtol=1e-3)


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
        (1.0, torch.bfloat16, 1e-1, 5e-1),
        (1.0, torch.float32, 2e-5, 1e-3),
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
@pytest.mark.parametrize("loss_type", ["bnpo", "grpo", "dr_grpo", "dapo", "cispo", "sapo", "luspo", "vespo"])
@pytest.mark.parametrize("importance_sampling_level", ["token", "sequence"])
@pytest.mark.parametrize("delta", [None, 2.0])
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
    delta,
):
    if importance_sampling_level == "sequence" and loss_type in ("cispo", "sapo", "vespo"):
        pytest.skip(f"Sequence-level importance sampling is not supported for loss_type='{loss_type}'")
    if importance_sampling_level == "token" and loss_type == "luspo":
        pytest.skip("Token-level importance sampling is not supported for loss_type='luspo'")
    if delta is not None and loss_type in ("cispo", "sapo", "vespo"):
        pytest.skip(f"delta is not supported for loss_type='{loss_type}'")
    # LUSPO amplifies per-token rounding by O(seq_len) because the loss scales by
    # attention_mask.sum(-1). VESPO's phi = exp(log_phi) similarly amplifies small
    # log_ratio deltas from chunked per_token_logps. Combined with torch.compile cache
    # pollution across the ~1000 tests in this file, both produce sporadic mismatches
    # on H100 (and occasionally on bf16 3090 Ti) even though they pass in isolation.
    if loss_type == "luspo" and V >= 4096 and device == "cuda" and torch.cuda.get_device_capability()[0] >= 9:
        pytest.skip("luspo at large V flakes on H100+ due to torch.compile cache pollution; passes in isolation")
    if loss_type == "vespo" and dtype == torch.bfloat16:
        pytest.skip(
            "vespo bf16 is numerically unstable: exp(log_phi) amplifies bf16 rounding in chunked per_token_logps"
        )
    if loss_type == "vespo" and V >= 4096:
        pytest.skip(
            "vespo at large V is numerically unstable due to exp(log_phi) amplification of chunked logprob noise"
        )

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
        delta=delta,
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
        delta=delta,
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
        logps = F.log_softmax(logits, dim=-1)
        per_token_logps = logps.gather(dim=-1, index=selected_token_ids.unsqueeze(-1)).squeeze(-1)

    # Create attention mask with random padding [B, T]
    attention_mask = torch.ones(B, T, device=device)
    num_elements_to_mask = torch.randint(1, B * T // 2, (1,)).item()
    mask_indices = torch.randperm(B * T)[:num_elements_to_mask]
    attention_mask.view(-1)[mask_indices] = 0

    # Create advantages with shape [B] and ensure mixed signs for SAPO
    advantages = torch.randn(B, device=device, dtype=dtype)
    advantages[0] = -advantages[0].abs()
    if B > 1:
        advantages[1] = advantages[1].abs()

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


@pytest.mark.parametrize("loss_type", ["grpo", "dapo"])
@pytest.mark.parametrize("importance_sampling_level", ["token", "sequence"])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 5e-4),
    ],
)
def test_correctness_with_bias_correction_kl(loss_type, importance_sampling_level, dtype, atol, rtol):
    """Test use_bias_correction_kl (importance-sampling-corrected KL from DeepSeek-V3.2).

    Covers both ``importance_sampling_level`` values: TRL multiplies ``per_token_kl``
    by ``coef_1`` whose shape mirrors the importance-sampling level (token: (B, T);
    sequence: (B, 1)). Liger must do the same — historically it always recomputed a
    token-level ratio, which silently miscomputed the bias-corrected KL when
    sequence-level importance sampling was selected.
    """
    set_seed()
    B, T, H, V = 3, 47, 31, 123
    beta = 0.1  # Must be non-zero for KL to matter
    torch.compiler.reset()

    torch_lm_head_grpo = TorchLMHeadGRPO(
        H=H,
        V=V,
        dtype=dtype,
        beta=beta,
        loss_type=loss_type,
        importance_sampling_level=importance_sampling_level,
        use_bias_correction_kl=True,
    )
    liger_lm_head_grpo = LigerLMHeadGRPO(
        H=H,
        V=V,
        dtype=dtype,
        beta=beta,
        loss_type=loss_type,
        importance_sampling_level=importance_sampling_level,
        use_bias_correction_kl=True,
    )

    torch_lm_head_grpo.lin.weight.data = liger_lm_head_grpo.lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )
    torch_lm_head_grpo.ref_lin.weight.data = liger_lm_head_grpo.ref_lin.weight.data = (
        torch_lm_head_grpo.lin.weight.data + torch.randn(V, H, device=device, dtype=dtype) * 0.01
    )

    _input = torch.randn(B, T, H, device=device, dtype=dtype)
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)

    selected_token_ids = torch.randint(0, V, (B, T), device=device)
    attention_mask = torch.ones(B, T, device=device, dtype=dtype)
    attention_mask[:, -10:] = 0
    advantages = torch.randn(B, device=device, dtype=torch.float32)
    old_per_token_logps = torch.randn(B, T, device=device, dtype=torch.float32)

    loss1, metrics1 = torch_lm_head_grpo(
        input1,
        selected_token_ids,
        attention_mask,
        advantages,
        old_per_token_logps=old_per_token_logps,
        ref_input=input1.detach(),
    )
    loss2, metrics2 = liger_lm_head_grpo(
        input2,
        selected_token_ids,
        attention_mask,
        advantages,
        old_per_token_logps=old_per_token_logps,
        ref_input=input2.detach(),
    )

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)
    loss1.backward()
    loss2.backward()
    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(
        torch_lm_head_grpo.lin.weight.grad,
        liger_lm_head_grpo.lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("loss_type", ["bnpo", "grpo", "dapo", "cispo", "sapo", "luspo", "vespo"])
@pytest.mark.parametrize("beta", [0.0, 0.1])
def test_correctness_with_vllm_is_ratio(loss_type, beta):
    """Test vllm_is_ratio correctness against torch reference, and 1D/2D shape equivalence."""
    if loss_type == "luspo":
        pytest.skip("Token-level importance sampling is not supported for loss_type='luspo'")
    torch.compiler.reset()
    B, T, H, V = 4, 32, 64, 128
    dtype = torch.float32
    atol, rtol = 1e-5, 5e-4

    _weight = torch.randn(V, H, device=device, dtype=dtype)
    _input = torch.randn(B, T, H, device=device, dtype=dtype)
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)

    selected_token_ids = torch.randint(0, V, (B, T), device=device)
    attention_mask = torch.ones(B, T, device=device)
    attention_mask[:, -5:] = 0
    advantages = torch.randn(B, device=device, dtype=dtype)
    advantages[0] = -advantages[0].abs()  # ensure mixed signs for SAPO

    vllm_is_ratio = torch.rand(B, T, device=device, dtype=torch.float32) * 0.999 + 0.001

    torch_lm = TorchLMHeadGRPO(H=H, V=V, dtype=dtype, beta=beta, loss_type=loss_type, use_ref_model=False)
    liger_lm = LigerLMHeadGRPO(H=H, V=V, dtype=dtype, beta=beta, loss_type=loss_type, use_ref_model=False)
    torch_lm.lin.weight.data = liger_lm.lin.weight.data = _weight.clone()

    loss1, aux1 = torch_lm(input1, selected_token_ids, attention_mask, advantages, vllm_is_ratio=vllm_is_ratio)
    loss2, aux2 = liger_lm(input2, selected_token_ids, attention_mask, advantages, vllm_is_ratio=vllm_is_ratio)

    assert not torch.isnan(loss1)
    assert not torch.isnan(loss2)
    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)
    for m1, m2 in zip(aux1, aux2):
        assert_verbose_allclose(m1, m2, atol=atol, rtol=rtol)

    loss1.backward()
    loss2.backward()
    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(torch_lm.lin.weight.grad, liger_lm.lin.weight.grad, atol=atol, rtol=rtol)

    # Verify 1D (B,) gives same result as (B, 1)
    uniform_val = 0.42
    input3 = _input.detach().clone().requires_grad_(True)
    input4 = _input.detach().clone().requires_grad_(True)
    liger3 = LigerLMHeadGRPO(H=H, V=V, dtype=dtype, beta=beta, loss_type=loss_type, use_ref_model=False)
    liger4 = LigerLMHeadGRPO(H=H, V=V, dtype=dtype, beta=beta, loss_type=loss_type, use_ref_model=False)
    liger3.lin.weight.data = liger4.lin.weight.data = _weight.clone()

    loss3, _ = liger3(
        input3,
        selected_token_ids,
        attention_mask,
        advantages,
        vllm_is_ratio=torch.full((B,), uniform_val, device=device),
    )
    loss4, _ = liger4(
        input4,
        selected_token_ids,
        attention_mask,
        advantages,
        vllm_is_ratio=torch.full((B, 1), uniform_val, device=device),
    )
    assert_verbose_allclose(loss3, loss4, atol=1e-5, rtol=1e-5)
    loss3.backward()
    loss4.backward()
    assert_verbose_allclose(input3.grad, input4.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("loss_type", ["dapo", "cispo", "vespo"])
def test_num_items_in_batch_normalizer(loss_type):
    """``num_items_in_batch`` overrides the dapo/cispo/vespo normalizer.

    TRL's ``compute_loss`` for these loss types divides by ``num_items_in_batch /
    num_processes`` — the total active tokens across the entire generation batch
    (all gradient-accumulation micro-batches × all processes). The Liger default
    falls back to the current micro-batch's mask, which biases per-token weights
    by micro-batch size when grad-accum micro-batches have unequal lengths.

    This test verifies, in single-process world:
    1. Passing ``num_items_in_batch=mask.sum()`` matches the default normalizer.
    2. Doubling ``num_items_in_batch`` halves both loss and input gradients
       (linear in the normalizer, no other dependence).
    """
    set_seed()
    torch.compiler.reset()
    B, T, H, V = 3, 47, 31, 123
    dtype = torch.float32

    _weight = torch.randn(V, H, device=device, dtype=dtype)
    _input = torch.randn(B, T, H, device=device, dtype=dtype)
    selected_token_ids = torch.randint(0, V, (B, T), device=device)
    attention_mask = torch.ones(B, T, device=device)
    attention_mask[:, -5:] = 0
    advantages = torch.randn(B, device=device, dtype=dtype)
    advantages[0] = -advantages[0].abs()
    advantages[1] = advantages[1].abs()

    mask_sum = attention_mask.sum().item()

    def _run(num_items_in_batch):
        liger = LigerLMHeadGRPO(H=H, V=V, dtype=dtype, beta=0.0, loss_type=loss_type, use_ref_model=False)
        liger.lin.weight.data = _weight.clone()
        inp = _input.detach().clone().requires_grad_(True)
        loss, _ = liger(
            inp,
            selected_token_ids,
            attention_mask,
            advantages,
            num_items_in_batch=num_items_in_batch,
        )
        loss.backward()
        return loss.detach(), inp.grad.detach().clone()

    loss_default, grad_default = _run(num_items_in_batch=None)
    loss_match, grad_match = _run(num_items_in_batch=mask_sum)
    loss_double, grad_double = _run(num_items_in_batch=mask_sum * 2)

    assert_verbose_allclose(loss_default, loss_match, atol=1e-5, rtol=1e-5)
    assert_verbose_allclose(grad_default, grad_match, atol=1e-5, rtol=1e-5)

    assert_verbose_allclose(loss_double * 2, loss_default, atol=1e-5, rtol=1e-5)
    assert_verbose_allclose(grad_double * 2, grad_default, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("loss_type", ["dapo", "cispo", "vespo"])
def test_num_items_in_batch_matches_trl_formula(loss_type):
    """Liger with ``num_items_in_batch=N`` matches TRL's ``sum / (N / num_processes)``.

    Reproduces TRL's exact formula in single-process world (num_processes=1):
    ``loss = (per_token_loss * mask).sum() / num_items_in_batch``.
    """
    set_seed()
    torch.compiler.reset()
    B, T, H, V = 3, 47, 31, 123
    dtype = torch.float32

    _weight = torch.randn(V, H, device=device, dtype=dtype)
    _input = torch.randn(B, T, H, device=device, dtype=dtype)
    selected_token_ids = torch.randint(0, V, (B, T), device=device)
    attention_mask = torch.ones(B, T, device=device)
    attention_mask[:, -5:] = 0
    advantages = torch.randn(B, device=device, dtype=dtype)
    advantages[0] = -advantages[0].abs()
    advantages[1] = advantages[1].abs()

    # Pick num_items_in_batch != mask.sum() to exercise the new path.
    num_items_in_batch = float(attention_mask.sum().item()) * 1.7

    # Liger: pass num_items_in_batch through the new param.
    liger = LigerLMHeadGRPO(H=H, V=V, dtype=dtype, beta=0.0, loss_type=loss_type, use_ref_model=False)
    liger.lin.weight.data = _weight.clone()
    input1 = _input.detach().clone().requires_grad_(True)
    loss_liger, _ = liger(
        input1,
        selected_token_ids,
        attention_mask,
        advantages,
        num_items_in_batch=num_items_in_batch,
    )

    # Torch reference: use num_items_in_batch directly as the normalizer.
    # We monkey-patch TorchLMHeadGRPO's branch by overriding the loss with TRL's exact formula.
    torch_lm = TorchLMHeadGRPO(H=H, V=V, dtype=dtype, beta=0.0, loss_type=loss_type, use_ref_model=False)
    torch_lm.lin.weight.data = _weight.clone()
    input2 = _input.detach().clone().requires_grad_(True)

    logits = input2 @ torch_lm.lin.weight.t()
    log_probs = F.log_softmax(logits.float(), dim=-1)
    per_token_logps = log_probs.gather(dim=-1, index=selected_token_ids.unsqueeze(-1)).squeeze(-1)
    per_token_loss, _, _ = TorchLMHeadGRPO.compute_per_token_components(
        per_token_logps,
        attention_mask,
        advantages,
        old_per_token_logps=None,
        ref_per_token_logps=None,
        epsilon_low=0.2,
        epsilon_high=0.2,
        beta=0.0,
        importance_sampling_level="token",
        loss_type=loss_type,
    )
    loss_ref = (per_token_loss * attention_mask).sum() / num_items_in_batch

    assert_verbose_allclose(loss_liger, loss_ref, atol=1e-5, rtol=1e-4)

    loss_liger.backward()
    loss_ref.backward()
    assert_verbose_allclose(input1.grad, input2.grad, atol=1e-5, rtol=1e-4)


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


@pytest.mark.parametrize("loss_type", ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"])
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
    elif loss_type == "luspo":
        expected = (per_token_loss * mask_f.sum(-1, keepdim=True)).mean()
    else:  # dapo/cispo
        expected = (per_token_loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)

    assert_verbose_allclose(reduced, expected)


def test_reduce_grpo_loss_requires_max_completion_length():
    per_token_loss = torch.randn(2, 3)
    mask = torch.ones_like(per_token_loss, dtype=torch.long)
    reduced = _reduce_grpo_loss(per_token_loss, mask, "dr_grpo", max_completion_length=None)
    expected = (per_token_loss * mask).sum() / (per_token_loss.size(0) * per_token_loss.size(1))
    assert_verbose_allclose(reduced, expected)


@pytest.mark.parametrize("loss_type", ["cispo", "sapo"])
def test_sequence_level_rejects_unsupported_loss_types(loss_type):
    """Sequence-level importance sampling should raise ValueError for cispo and sapo."""
    B, T, H, V = 2, 8, 16, 32
    dtype = torch.float32

    liger_lm = LigerLMHeadGRPO(
        H=H,
        V=V,
        dtype=dtype,
        beta=0.0,
        loss_type=loss_type,
        use_ref_model=False,
        importance_sampling_level="sequence",
    )

    _input = torch.randn(B, T, H, device=device, dtype=dtype).requires_grad_(True)
    selected_token_ids = torch.randint(0, V, (B, T), device=device)
    attention_mask = torch.ones(B, T, device=device)
    advantages = torch.randn(B, device=device)

    with pytest.raises(ValueError, match="Sequence-level importance sampling is not supported"):
        liger_lm(_input, selected_token_ids, attention_mask, advantages)


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
    delta=None,
    use_bias_correction_kl=False,
):
    logits = logits[:, :-1, :] / temperature
    log_probs = torch.log_softmax(logits, dim=-1)
    per_token_logps = log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
    old = old_logp if old_logp is not None else per_token_logps.detach()
    coef_1 = torch.exp(per_token_logps - old)
    coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)
    if delta is not None:
        coef_1 = torch.clamp(coef_1, max=delta)
    per_token_loss1 = coef_1 * advantages.unsqueeze(1)
    per_token_loss2 = coef_2 * advantages.unsqueeze(1)
    per_token_loss = -torch.minimum(per_token_loss1, per_token_loss2)
    is_clipped = per_token_loss1 < per_token_loss2
    mask = completion_mask.to(torch.bool)
    per_token_loss = per_token_loss.masked_fill(~mask, 0.0)
    is_clipped = is_clipped & mask
    if beta != 0.0:
        kl = torch.exp(ref_logp - per_token_logps) - (ref_logp - per_token_logps) - 1.0
        if use_bias_correction_kl:
            kl = kl * torch.exp(per_token_logps - old)
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
