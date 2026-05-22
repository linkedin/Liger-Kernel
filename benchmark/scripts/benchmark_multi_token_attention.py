import math
import os
import sys

import torch
import triton

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import compute_model_config_sweep_config
from benchmark_model_configs import compute_seq_len_sweep_config
from benchmark_model_configs import get_benchmark_model_config
from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.multi_token_attention import LigerMultiTokenAttention
from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class TorchMultiTokenAttention(torch.nn.Module):
    def __init__(self, C_in, C_out, K, groups, bias, dtype, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(C_out, C_in // groups, K, K, dtype=dtype, device=device))
        self.bias = torch.nn.Parameter(torch.empty(C_out, dtype=dtype, device=device)) if bias else None
        self.K = K
        self.groups = groups

    def forward(self, scores):
        B, C_in, L, _ = scores.shape
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=scores.device)).view(1, 1, L, L)
        inf = torch.tensor(-1e9, device=scores.device, dtype=scores.dtype)
        zero = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        s_inf = scores.masked_fill(~mask, inf)
        probs = torch.nn.functional.softmax(s_inf, dim=-1)
        out_c = torch.nn.functional.conv2d(
            probs, self.weight, self.bias, stride=1, padding=self.K // 2, groups=self.groups
        )
        return out_c.masked_fill(~mask, zero)


def _setup_multi_token_attention(input: SingleBenchmarkRunInput):
    """Create input tensors and multi-token attention from benchmark config."""
    cfg = input.extra_benchmark_config
    C_in = cfg["C_in"]
    C_out = cfg["C_out"]
    K = cfg["K"]
    groups = cfg["groups"]
    bias = cfg["bias"]
    dtype = cfg["dtype"]
    B = cfg.get("B", 2)
    L = cfg.get("L", input.x)

    liger_attn = (
        LigerMultiTokenAttention(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=K,
            stride=1,
            padding=K // 2,
            dilation=1,
            groups=groups,
            bias=bias,
        )
        .to(device)
        .to(dtype)
    )

    torch_attn = TorchMultiTokenAttention(
        C_in=C_in, C_out=C_out, K=K, groups=groups, bias=bias, dtype=dtype, device=device
    )

    with torch.no_grad():
        torch_attn.weight.copy_(liger_attn.weight)
        if bias:
            torch_attn.bias.copy_(liger_attn.bias)

    x = torch.randn(B, C_in, L, L, dtype=dtype, device=device, requires_grad=True)
    dy = torch.randn_like(x)

    if input.kernel_provider == "liger":
        fwd_fn = lambda: liger_attn(x)
    elif input.kernel_provider == "torch":
        fwd_fn = lambda: torch_attn(x)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for multi-token attention")

    # Warmup
    _ = fwd_fn()
    _.backward(dy, retain_graph=True)

    return x, dy, fwd_fn


def bench_speed_multi_token_attention(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, dy, fwd = _setup_multi_token_attention(input)
    mode = input.kernel_operation_mode

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, grad_to_none=[x], rep=100, quantiles=QUANTILES)
    elif mode == "backward":
        y = fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            grad_to_none=[x],
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward(dy, retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, grad_to_none=[x], rep=100, quantiles=QUANTILES)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_multi_token_attention(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, dy, fwd_fn = _setup_multi_token_attention(input)

    def full():
        y = fwd_fn()
        y.backward(dy, retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)

    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


def _resolve_model_config_multi_token_attention(input: SingleBenchmarkRunInput):
    cfg = input.extra_benchmark_config
    model_info = cfg["model_configs"][input.x]
    return _setup_multi_token_attention(
        SingleBenchmarkRunInput(
            x=input.x,
            kernel_provider=input.kernel_provider,
            extra_benchmark_config={
                "C_in": cfg["C_in"],
                "C_out": cfg["C_out"],
                "K": cfg["K"],
                "groups": cfg["groups"],
                "bias": cfg["bias"],
                "dtype": model_info["dtype"],
                "B": cfg["B"],
                "L": cfg["L"],
            },
        )
    )


def bench_speed_multi_token_attention_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, dy, fwd_fn = _resolve_model_config_multi_token_attention(input)
    mode = input.kernel_operation_mode

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd_fn, grad_to_none=[x], rep=100, quantiles=QUANTILES)
    elif mode == "backward":
        y = fwd_fn()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            grad_to_none=[x],
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y = fwd_fn()
            y.backward(dy, retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, grad_to_none=[x], rep=100, quantiles=QUANTILES)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_multi_token_attention_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, dy, fwd_fn = _resolve_model_config_multi_token_attention(input)

    def full():
        y = fwd_fn()
        y.backward(dy, retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        all_model_configs = list(MODEL_REGISTRY.values())
        B = 2

        def _probe_factory(model_cfg, probe_bt):
            def _probe():
                # Memory scales as O(L^2) due to (B, C_in, L, L) shape
                L = int((probe_bt // B) ** 0.5)
                probe_input = SingleBenchmarkRunInput(
                    x=0,
                    kernel_provider="torch",
                    extra_benchmark_config={
                        "C_in": 4,
                        "C_out": 4,
                        "K": 3,
                        "groups": 1,
                        "bias": True,
                        "dtype": model_cfg.dtype,
                        "B": B,
                        "L": L,
                    },
                )
                _, _, fwd_fn = _setup_multi_token_attention(probe_input)
                return fwd_fn()

            return _probe

        sweep = compute_model_config_sweep_config(all_model_configs, probe_fn_factory=_probe_factory, bt=args.bt)
        # sweep.seq_len assumes linear scaling, but memory scales as O(L^2)
        # So we need to take sqrt to get the actual safe L value
        safe_L = int((sweep.seq_len) ** 0.5)
        model_configs_info = {cfg.name: {"dtype": cfg.dtype} for cfg in sweep.model_configs}

        common_configs = {
            "kernel_name": "multi_token_attention",
            "x_name": "model_config",
            "x_label": "model configuration",
            "x_values": [cfg.name for cfg in sweep.model_configs],
            "kernel_providers": ["liger", "torch"],
            "extra_benchmark_configs": [
                {
                    "model_configs": model_configs_info,
                    "C_in": 4,
                    "C_out": 4,
                    "K": 3,
                    "groups": 1,
                    "bias": True,
                    "B": sweep.batch_size,
                    "L": safe_L,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_multi_token_attention_model_config,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_multi_token_attention_model_config,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
    else:
        model = get_benchmark_model_config(args.model)
        B = 2
        probe_L = 256

        def _probe():
            probe_input = SingleBenchmarkRunInput(
                x=0,
                kernel_provider="torch",
                extra_benchmark_config={
                    "C_in": 4,
                    "C_out": 4,
                    "K": 3,
                    "groups": 1,
                    "bias": True,
                    "dtype": model.dtype,
                    "B": B,
                    "L": probe_L,
                },
            )
            _, _, fwd_fn = _setup_multi_token_attention(probe_input)
            return fwd_fn()

        # Memory scales as O(L^2)
        config = compute_seq_len_sweep_config(
            model,
            probe_fn=_probe,
            probe_seq_len=probe_L,
            probe_batch_size=B,
            scaling_method="quadratic",
        )

        common_configs = {
            "kernel_name": "multi_token_attention",
            "x_name": "L",
            "x_label": "sequence length",
            "x_values": [2**i for i in range(5, int(math.log2(max(32, config.seq_len))) + 1)],
            "kernel_providers": ["liger", "torch"],
            "extra_benchmark_configs": [
                {"C_in": 4, "C_out": 4, "K": 3, "groups": 1, "bias": True, "dtype": model.dtype, "B": B}
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_multi_token_attention,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_multi_token_attention,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
