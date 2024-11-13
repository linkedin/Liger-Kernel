import torch
import torch.nn.functional as F
import triton
from utils import (
    QUANTILES,
    SingleBenchmarkRunInput,
    SingleBenchmarkRunOutput,
    _test_memory,
    parse_benchmark_script_args,
    run_benchmarks,
)
from typing import Tuple

from liger_kernel.chunked_loss.orpo_loss import LigerFusedLinearORPOFunction


def odds_ratio_loss(
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        beta: float = 0.1,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        """Compute ORPO's odds ratio (OR) loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the ORPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
            The log odds ratio of the chosen responses over the rejected responses ratio for logging purposes.
            The `log(sigmoid(log_odds_chosen))` for logging purposes.
        """

        # Derived from Eqs. (4) and (7) from https://huggingface.co/papers/2403.07691 by using log identities and exp(log(P(y|x)) = P(y|x)
        log_odds = (policy_chosen_logps - policy_rejected_logps) - (
            torch.log1p(-torch.exp(policy_chosen_logps))
            - torch.log1p(-torch.exp(policy_rejected_logps))
        )
        ratio = F.logsigmoid(log_odds)
        losses = beta * ratio

        return losses


class TorchLMHeadORPO(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param H: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param reduction: reduction method
    """

    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        self.lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype
        )
        self.ce_loss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )
        self.odds_ratio_loss = odds_ratio_loss

    def forward(self, x, y):
        logits = self.lin(x)
        ce_loss = self.ce_loss(logits[:logits.shape[0] // 2].view(-1, logits.shape[-1]), y[:y.shape[0] // 2].view(-1))
        all_logprobs = F.log_softmax(logits, dim=-1).mean(dim=-1)
        chosen_logprobs = all_logprobs[:all_logprobs.shape[0] // 2]
        rejected_logprobs = all_logprobs[all_logprobs.shape[0] // 2:]
        or_loss = self.odds_ratio_loss(chosen_logprobs, rejected_logprobs)
        return ce_loss - or_loss.mean()


class LigerLMHeadORPO(torch.nn.Module):
    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        self.lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype
        )
        self.orpo_loss = LigerFusedLinearORPOFunction.apply

    def forward(self, x, y):
        return self.orpo_loss(x, self.lin.weight, y)


#############################################################################
# Test the memory consumption of the linear fused cross entropy loss
#############################################################################


def bench_memory_fused_linear_orpo_loss(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    B = input.x
    T = input.extra_benchmark_config["T"]
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    provider = input.kernel_provider

    device = "cuda"
    torch_lm_head_orpo = TorchLMHeadORPO(H=H, V=V, dtype=dtype).to(device)
    liger_lm_head_orpo = LigerLMHeadORPO(H=H, V=V, dtype=dtype).to(device)

    _input = torch.randn(B, T, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(V, (B, T), dtype=torch.long, device=device)

    def fwd():
        if provider == "liger":
            return liger_lm_head_orpo(_input, target)
        elif provider == "huggingface":
            return torch_lm_head_orpo(_input, target)

    def full():
        y = fwd()
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


# #############################################################################
# # Test the speed of the fused linear cross entropy loss
# #############################################################################


def bench_speed_fused_linear_orpo_loss(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    B = input.x
    T = input.extra_benchmark_config["T"]
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    device = "cuda"

    torch_lm_head_orpo = TorchLMHeadORPO(H=H, V=V, dtype=dtype).to(device)
    liger_lm_head_orpo = LigerLMHeadORPO(H=H, V=V, dtype=dtype).to(device)

    _input = torch.randn(B, T, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(V, (B, T), dtype=torch.long, device=device)

    def fwd():
        if provider == "liger":
            return liger_lm_head_orpo(_input, target)
        elif provider == "huggingface":
            return torch_lm_head_orpo(_input, target)

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
            grad_to_none=[_input],
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

    common_configs = {
        "kernel_name": "fused_linear_orpo_loss",
        "x_name": "B",
        "x_label": "B",
        "x_values": [2**i for i in range(1, 6)],
        "kernel_providers": ["liger", "huggingface"],
        "extra_benchmark_configs": [
            {"T": 4096, "H": 4096, "V": 128256, "mode": "forward", "dtype": torch.bfloat16}
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_fused_linear_orpo_loss,
        kernel_operation_modes=["forward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs
    )
    run_benchmarks(
        bench_test_fn=bench_memory_fused_linear_orpo_loss,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs
    )
