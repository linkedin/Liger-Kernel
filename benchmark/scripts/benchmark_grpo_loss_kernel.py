"""
Benchmark for GRPO loss kernel (ops layer).

This benchmark tests the performance of the low-level GRPO loss kernel
(liger_kernel.ops.grpo_loss.GrpoLossFunction) against a pure PyTorch baseline.

Unlike benchmark_grpo_loss.py which tests the chunked_loss layer (fused linear + loss),
this benchmark focuses on the kernel-level implementation without the linear layer fusion.

Usage:
    python benchmark_grpo_loss_kernel.py [--overwrite]
"""
import os
import sys

import torch
import torch.nn as nn
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


#############################################################################
# Module wrappers for GRPO loss kernel
#############################################################################


class TorchGRPOLoss(nn.Module):
    """Torch baseline module for GRPO loss kernel."""

    def __init__(self, temperature=1.0, beta=0.0, eps_low=0.2, eps_high=0.2):
        super().__init__()
        self.temperature = temperature
        self.beta = beta
        self.eps_low = eps_low
        self.eps_high = eps_high

    def forward(self, logits, old_logp, ref_logp, completion_ids, advantages, completion_mask):
        B, L_ADD_1, N = logits.shape
        L = L_ADD_1 - 1

        # Compute log probabilities
        logits_for_loss = logits[:, :-1, :] / self.temperature
        log_probs = torch.nn.functional.log_softmax(logits_for_loss, dim=-1)

        # Gather log probs for selected tokens
        completion_ids_expanded = completion_ids.unsqueeze(-1)
        logp = log_probs.gather(dim=-1, index=completion_ids_expanded).squeeze(-1)

        # Compute importance ratio
        if old_logp is None:
            old_logp_val = logp
        else:
            old_logp_val = old_logp

        coef_1 = torch.exp(logp - old_logp_val)

        # Compute clipped coefficient
        coef_2 = torch.clamp(coef_1, 1 - self.eps_low, 1 + self.eps_high)

        # Expand advantages to per-token
        advantages_expanded = advantages.unsqueeze(-1).expand(-1, L)

        # Compute per-token loss
        per_token_loss1 = coef_1 * advantages_expanded
        per_token_loss2 = coef_2 * advantages_expanded
        per_token_loss = -torch.minimum(per_token_loss1, per_token_loss2)

        # Add KL penalty if beta > 0
        if self.beta != 0.0 and ref_logp is not None:
            kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1
            per_token_loss += self.beta * kl

        # Apply mask and reduce
        if completion_mask is not None:
            mask = completion_mask.float()
        else:
            mask = torch.ones(B, L, device=logits.device)

        # Reduce loss (GRPO uses per-sequence mean)
        loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()

        return loss


class LigerGRPOLoss(nn.Module):
    """Liger module wrapper for GRPO loss kernel."""

    def __init__(self, temperature=1.0, beta=0.0, eps_low=0.2, eps_high=0.2, importance_sampling_level="token"):
        super().__init__()
        self.temperature = temperature
        self.beta = beta
        self.eps_low = eps_low
        self.eps_high = eps_high
        self.importance_sampling_level = importance_sampling_level

    def forward(self, logits, old_logp, ref_logp, completion_ids, advantages, completion_mask):
        from liger_kernel.ops import GrpoLossFunction

        return GrpoLossFunction.apply(
            logits,
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask,
            self.temperature,
            self.beta,
            self.eps_low,
            self.eps_high,
            False,  # inplace
            "grpo",  # loss_type
            None,  # max_completion_length
            True,  # reduce
            self.importance_sampling_level,
            1.0,  # sapo_temperature_pos
            1.05,  # sapo_temperature_neg
            None,  # vllm_is_ratio
            None,  # delta
            False,  # use_bias_correction_kl
        )[0]


#############################################################################
# Test the memory consumption of the GRPO loss kernel
#############################################################################


def bench_memory_grpo_loss_kernel(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    B = input.x
    T = input.extra_benchmark_config["T"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    importance_sampling_level = input.extra_benchmark_config["importance_sampling_level"]
    provider = input.kernel_provider

    temperature = 1.0
    beta = input.extra_benchmark_config.get("beta", 0.0)
    eps_low = 0.2
    eps_high = 0.2

    # Instantiate modules
    torch_grpo = TorchGRPOLoss(temperature=temperature, beta=beta, eps_low=eps_low, eps_high=eps_high).to(device)
    liger_grpo = LigerGRPOLoss(
        temperature=temperature,
        beta=beta,
        eps_low=eps_low,
        eps_high=eps_high,
        importance_sampling_level=importance_sampling_level,
    ).to(device)

    # Create inputs
    logits = torch.randn(B, T + 1, V, requires_grad=True, dtype=dtype, device=device)
    completion_ids = torch.randint(0, V, (B, T), dtype=torch.long, device=device)
    advantages = torch.randn(B, dtype=dtype, device=device)
    completion_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    old_logp = None  # On-policy case
    ref_logp = torch.randn(B, T, dtype=torch.float32, device=device) if beta > 0 else None

    def fwd():
        if provider == "liger":
            return liger_grpo(logits, old_logp, ref_logp, completion_ids, advantages, completion_mask)
        elif provider == "torch":
            return torch_grpo(logits, old_logp, ref_logp, completion_ids, advantages, completion_mask)

    def full():
        y = fwd()
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


#############################################################################
# Test the speed of the GRPO loss kernel
#############################################################################


def bench_speed_grpo_loss_kernel(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    B = input.x
    T = input.extra_benchmark_config["T"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    importance_sampling_level = input.extra_benchmark_config["importance_sampling_level"]
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    temperature = 1.0
    beta = input.extra_benchmark_config.get("beta", 0.0)
    eps_low = 0.2
    eps_high = 0.2

    # Instantiate modules
    torch_grpo = TorchGRPOLoss(temperature=temperature, beta=beta, eps_low=eps_low, eps_high=eps_high).to(device)
    liger_grpo = LigerGRPOLoss(
        temperature=temperature,
        beta=beta,
        eps_low=eps_low,
        eps_high=eps_high,
        importance_sampling_level=importance_sampling_level,
    ).to(device)

    # Create inputs
    logits = torch.randn(B, T + 1, V, requires_grad=True, dtype=dtype, device=device)
    completion_ids = torch.randint(0, V, (B, T), dtype=torch.long, device=device)
    advantages = torch.randn(B, dtype=dtype, device=device)
    completion_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    old_logp = None  # On-policy case
    ref_logp = torch.randn(B, T, dtype=torch.float32, device=device) if beta > 0 else None

    def fwd():
        if provider == "liger":
            return liger_grpo(logits, old_logp, ref_logp, completion_ids, advantages, completion_mask)
        elif provider == "torch":
            return torch_grpo(logits, old_logp, ref_logp, completion_ids, advantages, completion_mask)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            fwd,
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        y = fwd()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(retain_graph=True),
            grad_to_none=[logits],
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            rep=100,
            quantiles=QUANTILES,
        )
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    # Benchmark token-level importance sampling (original GRPO)
    token_configs = {
        "kernel_name": "grpo_loss_kernel_token",
        "x_name": "B",
        "x_label": "Batch Size",
        "x_values": [2 ** i for i in range(1, 5)],
        "kernel_providers": ["liger", "torch"],
        "extra_benchmark_configs": [
            {
                "T": 512,
                "V": 32000,
                "beta": 0.1,
                "dtype": torch.bfloat16,
                "importance_sampling_level": "token",
            }
        ],
        "overwrite": args.overwrite,
    }

    # Benchmark sequence-level importance sampling (GSPO)
    sequence_configs = {
        "kernel_name": "grpo_loss_kernel_sequence",
        "x_name": "B",
        "x_label": "Batch Size",
        "x_values": [2 ** i for i in range(1, 5)],
        "kernel_providers": ["liger", "torch"],
        "extra_benchmark_configs": [
            {
                "T": 512,
                "V": 32000,
                "beta": 0.1,
                "dtype": torch.bfloat16,
                "importance_sampling_level": "sequence",
            }
        ],
        "overwrite": args.overwrite,
    }

    # Run benchmarks for token-level (GRPO)
    print("Benchmarking GRPO (token-level importance sampling)...")
    run_benchmarks(
        bench_test_fn=bench_speed_grpo_loss_kernel,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **token_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_grpo_loss_kernel,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **token_configs,
    )

    # Run benchmarks for sequence-level (GSPO)
    print("Benchmarking GSPO (sequence-level importance sampling)...")
    run_benchmarks(
        bench_test_fn=bench_speed_grpo_loss_kernel,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **sequence_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_grpo_loss_kernel,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **sequence_configs,
    )
