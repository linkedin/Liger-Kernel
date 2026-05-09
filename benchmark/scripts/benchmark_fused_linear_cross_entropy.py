import math

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

from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from liger_kernel.utils import infer_device

device = infer_device()


class TorchLMHeadCE(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param H: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param reduction: reduction method
    """

    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")

    def forward(self, x, y):
        logits = self.lin(x)
        return self.ce_loss(logits, y)


class LigerLMHeadCE(torch.nn.Module):
    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100, accum_dtype=None):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ce_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean", accum_dtype=accum_dtype
        )

    def forward(self, x, y):
        return self.ce_loss(self.lin.weight, x, y)


def _setup_fused_linear_cross_entropy(input: SingleBenchmarkRunInput):
    """Create input tensor, target, and fused linear CE from benchmark config."""
    cfg = input.extra_benchmark_config
    H = cfg["hidden_size"]
    V = cfg["vocab_size"]
    dtype = cfg["dtype"]
    BT = input.x

    _input = torch.randn(BT, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(V, (BT, 1), dtype=torch.long, device=device).squeeze(1)

    if input.kernel_provider == "liger":
        lm_head_ce = LigerLMHeadCE(H=H, V=V, dtype=dtype).to(device)
    elif input.kernel_provider == "liger-fp32-accum":
        lm_head_ce = LigerLMHeadCE(H=H, V=V, dtype=dtype, accum_dtype=torch.float32).to(device)
    else:
        lm_head_ce = TorchLMHeadCE(H=H, V=V, dtype=dtype).to(device)
    return _input, target, lm_head_ce


def bench_speed_fused_linear_cross_entropy(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    _input, target, lm_head_ce = _setup_fused_linear_cross_entropy(input)
    mode = input.kernel_operation_mode

    def fwd():
        return lm_head_ce(_input, target)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            fwd,
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "no-grad-forward":
        with torch.no_grad():
            ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, rep=100, quantiles=QUANTILES)
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

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, rep=100, quantiles=QUANTILES)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_fused_linear_cross_entropy(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    _input, target, lm_head_ce = _setup_fused_linear_cross_entropy(input)

    def full():
        y = lm_head_ce(_input, target)
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)

    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


def _resolve_model_config_fused_linear_cross_entropy(input: SingleBenchmarkRunInput):
    """Resolve model-config-sweep input into standard setup args."""
    cfg = input.extra_benchmark_config
    model_info = cfg["model_configs"][input.x]
    return _setup_fused_linear_cross_entropy(
        SingleBenchmarkRunInput(
            x=cfg["BT"],
            kernel_provider=input.kernel_provider,
            extra_benchmark_config={
                "hidden_size": model_info["hidden_size"],
                "vocab_size": model_info["vocab_size"],
                "dtype": model_info["dtype"],
            },
        )
    )


def bench_speed_fused_linear_cross_entropy_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    _input, target, lm_head_ce = _resolve_model_config_fused_linear_cross_entropy(input)
    mode = input.kernel_operation_mode

    def fwd():
        return lm_head_ce(_input, target)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            fwd,
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "no-grad-forward":
        with torch.no_grad():
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
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_fused_linear_cross_entropy_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    _input, target, lm_head_ce = _resolve_model_config_fused_linear_cross_entropy(input)

    def full():
        y = lm_head_ce(_input, target)
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        all_model_configs = list(MODEL_REGISTRY.values())

        def _probe_factory(model_cfg, probe_bt):
            def _probe():
                probe_input = SingleBenchmarkRunInput(
                    x=probe_bt,
                    kernel_provider="huggingface",
                    extra_benchmark_config={
                        "hidden_size": model_cfg.hidden_size,
                        "vocab_size": model_cfg.vocab_size,
                        "dtype": model_cfg.dtype,
                    },
                )
                _input, target, lm_head_ce = _setup_fused_linear_cross_entropy(probe_input)
                return lm_head_ce(_input, target)

            return _probe

        sweep = compute_model_config_sweep_config(all_model_configs, probe_fn_factory=_probe_factory, bt=args.bt)

        model_configs_info = {
            cfg.name: {
                "hidden_size": cfg.hidden_size,
                "vocab_size": cfg.vocab_size,
                "dtype": cfg.dtype,
            }
            for cfg in sweep.model_configs
        }

        common_configs = {
            "kernel_name": "fused_linear_cross_entropy",
            "x_name": "model_config",
            "x_label": "model configuration",
            "x_values": [cfg.name for cfg in sweep.model_configs],
            "kernel_providers": ["liger", "liger-fp32-accum", "huggingface"],
            "extra_benchmark_configs": [
                {
                    "model_configs": model_configs_info,
                    "BT": sweep.bt,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_fused_linear_cross_entropy_model_config,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_fused_linear_cross_entropy_model_config,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
    else:
        model = get_benchmark_model_config(args.model)
        probe_bt = 1024

        def _probe():
            probe_input = SingleBenchmarkRunInput(
                x=probe_bt,
                kernel_provider="huggingface",
                extra_benchmark_config={
                    "hidden_size": model.hidden_size,
                    "vocab_size": model.vocab_size,
                    "dtype": model.dtype,
                },
            )
            _input, target, lm_head_ce = _setup_fused_linear_cross_entropy(probe_input)
            return lm_head_ce(_input, target)

        config = compute_seq_len_sweep_config(model, probe_fn=_probe, probe_seq_len=probe_bt)

        common_configs = {
            "kernel_name": "fused_linear_cross_entropy",
            "x_name": "BT",
            "x_label": "B * T",
            "x_values": [2**i for i in range(10, int(math.log2(config.batch_size * config.seq_len)) + 1)],
            "kernel_providers": ["liger", "liger-fp32-accum", "huggingface"],
            "extra_benchmark_configs": [
                {
                    "hidden_size": model.hidden_size,
                    "vocab_size": model.vocab_size,
                    "dtype": model.dtype,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_fused_linear_cross_entropy,
            kernel_operation_modes=["forward", "backward", "full", "no-grad-forward"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_fused_linear_cross_entropy,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
