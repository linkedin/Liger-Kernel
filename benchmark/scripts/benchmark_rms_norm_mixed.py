import math

import torch
import torch.nn as nn
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.utils import infer_device

device = infer_device()


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16.0, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.requires_grad_(False)  # base weight frozen (LoRA)
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.scaling = alpha / r
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Init with small random values so grads flow through both A and B
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

    def forward(self, x):
        base = x @ self.weight.t()
        lora = (x @ self.lora_A.t()) @ self.lora_B.t()
        out = base + lora * self.scaling
        if self.bias is not None:
            out = out + self.bias
        return out


class MixedBlock(nn.Module):
    def __init__(self, norm_cls, hidden_size, eps, lora_r, lora_alpha):
        super().__init__()
        self.norm = norm_cls(hidden_size=hidden_size, eps=eps)
        self.proj = LoRALinear(hidden_size, hidden_size, r=lora_r, alpha=lora_alpha)

    def forward(self, x):
        return self.proj(self.norm(x))


def _build_block(provider, hidden_size, eps, dtype, lora_r, lora_alpha, freeze_norm_weight):
    norm_cls = LigerRMSNorm if provider == "liger" else LlamaRMSNorm
    block = MixedBlock(norm_cls, hidden_size=hidden_size, eps=eps, lora_r=lora_r, lora_alpha=lora_alpha)
    block = block.to(device=device, dtype=dtype)
    if freeze_norm_weight:
        block.norm.weight.requires_grad_(False)
    return block


def _grad_to_none_tensors(module, x):
    tensors = [x]
    for p in module.parameters():
        if p.requires_grad:
            tensors.append(p)
    return tensors


def bench_speed_rms_norm_mixed(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    N = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    extra = input.extra_benchmark_config
    M = extra["M"]
    eps = extra["eps"]
    dtype = extra["dtype"]
    lora_r = extra["lora_r"]
    lora_alpha = extra["lora_alpha"]
    freeze_norm_weight = extra.get("freeze_norm_weight", True)

    x_shape = (M, N)

    block = _build_block(provider, N, eps, dtype, lora_r, lora_alpha, freeze_norm_weight)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def y_fwd():
        return block(x)

    grad_to_none = _grad_to_none_tensors(block, x)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            y_fwd,
            grad_to_none=grad_to_none,
            rep=500,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        y = y_fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            grad_to_none=grad_to_none,
            rep=500,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y = y_fwd()
            y.backward(dy, retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            grad_to_none=grad_to_none,
            rep=500,
            quantiles=QUANTILES,
        )

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_rms_norm_mixed(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    N = input.x
    provider = input.kernel_provider

    extra = input.extra_benchmark_config
    M = extra["M"]
    eps = extra["eps"]
    dtype = extra["dtype"]
    lora_r = extra["lora_r"]
    lora_alpha = extra["lora_alpha"]
    freeze_norm_weight = extra.get("freeze_norm_weight", True)

    x_shape = (M, N)

    block = _build_block(provider, N, eps, dtype, lora_r, lora_alpha, freeze_norm_weight)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def y_fwd():
        return block(x)

    def full():
        y = y_fwd()
        y.backward(dy, retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)

    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "rms_norm_mixed",
        "x_name": "H",
        "x_label": "hidden size",
        "x_values": [2**i for i in range(10, 16)],
        "kernel_providers": ["liger", "huggingface"],
        "extra_benchmark_configs": [
            {
                "M": 2048,
                "dtype": torch.bfloat16,
                "eps": 1e-6,
                "lora_r": 8,
                "lora_alpha": 16.0,
                "freeze_norm_weight": True,
            }
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_rms_norm_mixed,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_rms_norm_mixed,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
