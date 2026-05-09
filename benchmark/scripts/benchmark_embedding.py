import math
import os
import sys

import torch
import triton

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import compute_model_config_sweep_config
from benchmark_model_configs import compute_seq_len_sweep_config
from benchmark_model_configs import get_benchmark_model_config
from torch.nn import Embedding
from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.experimental.embedding import LigerEmbedding
from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# NOTE: For torch compile, we will just use default inductor settings. No further customization
# is needed.


def _setup_embedding(input: SingleBenchmarkRunInput):
    """Create input tensors and embedding module from benchmark config."""
    cfg = input.extra_benchmark_config
    V = cfg.get("vocab_size", input.x)
    D = cfg["hidden_size"]
    dtype = cfg["dtype"]
    BT = cfg.get("BT", input.x)
    T = cfg.get("T", 512)
    B = max(1, BT // T) if "BT" not in cfg else BT // T

    # If BT is the x value, compute B from BT and T
    if "BT" not in cfg:
        B = max(1, input.x // T)
        BT = B * T

    input_ids = torch.randint(0, V, (B, T), device=device)

    if input.kernel_provider == "liger":
        emb = LigerEmbedding(V, D).to(device).to(dtype)
    elif input.kernel_provider == "torch_compile":
        emb = torch.compile(Embedding(V, D).to(device).to(dtype))
    elif input.kernel_provider == "huggingface":
        emb = Embedding(V, D).to(device).to(dtype)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for embedding")

    fwd_fn = lambda: emb(input_ids)
    return input_ids, fwd_fn


def bench_speed_embedding(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    input_ids, fwd = _setup_embedding(input)
    mode = input.kernel_operation_mode

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, quantiles=QUANTILES, rep=100)
    elif mode == "backward":
        output = fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: output.backward(torch.randn_like(output), retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=[input_ids],
            rep=100,
        )
    elif mode == "full":

        def full():
            output = fwd()
            output.backward(torch.randn_like(output))

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, quantiles=QUANTILES, rep=100)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_embedding(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    input_ids, fwd_fn = _setup_embedding(input)

    def full():
        output = fwd_fn()
        output.backward(torch.randn_like(output))

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


def _resolve_model_config_embedding(input: SingleBenchmarkRunInput):
    cfg = input.extra_benchmark_config
    model_info = cfg["model_configs"][input.x]
    return _setup_embedding(
        SingleBenchmarkRunInput(
            x=input.x,
            kernel_provider=input.kernel_provider,
            extra_benchmark_config={
                "vocab_size": model_info["vocab_size"],
                "hidden_size": model_info["hidden_size"],
                "dtype": model_info["dtype"],
                "BT": cfg["BT"],
                "T": cfg["T"],
            },
        )
    )


def bench_speed_embedding_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    input_ids, fwd_fn = _resolve_model_config_embedding(input)
    mode = input.kernel_operation_mode

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd_fn, quantiles=QUANTILES, rep=100)
    elif mode == "backward":
        output = fwd_fn()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: output.backward(torch.randn_like(output), retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=[input_ids],
            rep=100,
        )
    elif mode == "full":

        def full():
            output = fwd_fn()
            output.backward(torch.randn_like(output))

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, quantiles=QUANTILES, rep=100)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_embedding_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    input_ids, fwd_fn = _resolve_model_config_embedding(input)

    def full():
        output = fwd_fn()
        output.backward(torch.randn_like(output))

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
                T = max(1, probe_bt // B)
                probe_input = SingleBenchmarkRunInput(
                    x=0,
                    kernel_provider="huggingface",
                    extra_benchmark_config={
                        "vocab_size": model_cfg.vocab_size,
                        "hidden_size": model_cfg.hidden_size,
                        "dtype": model_cfg.dtype,
                        "BT": probe_bt,
                        "T": T,
                    },
                )
                _, fwd_fn = _setup_embedding(probe_input)
                return fwd_fn()

            return _probe

        sweep = compute_model_config_sweep_config(all_model_configs, probe_fn_factory=_probe_factory, bt=args.bt)
        model_configs_info = {
            cfg.name: {"vocab_size": cfg.vocab_size, "hidden_size": cfg.hidden_size, "dtype": cfg.dtype}
            for cfg in sweep.model_configs
        }

        common_configs = {
            "kernel_name": "embedding",
            "x_name": "model_config",
            "x_label": "model configuration",
            "x_values": [cfg.name for cfg in sweep.model_configs],
            "kernel_providers": ["liger", "huggingface", "torch_compile"],
            "extra_benchmark_configs": [{"model_configs": model_configs_info, "BT": sweep.bt, "T": sweep.seq_len}],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_embedding_model_config,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_embedding_model_config,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
    else:
        model = get_benchmark_model_config(args.model)
        T = 512
        probe_bt = 2048

        def _probe():
            B = probe_bt // T
            probe_input = SingleBenchmarkRunInput(
                x=0,
                kernel_provider="huggingface",
                extra_benchmark_config={
                    "vocab_size": model.vocab_size,
                    "hidden_size": model.hidden_size,
                    "dtype": model.dtype,
                    "BT": B * T,
                    "T": T,
                },
            )
            _, fwd_fn = _setup_embedding(probe_input)
            return fwd_fn()

        config = compute_seq_len_sweep_config(model, probe_fn=_probe, probe_seq_len=probe_bt)

        common_configs = {
            "kernel_name": "embedding",
            "x_name": "BT",
            "x_label": "B x T",
            "x_values": [2**i for i in range(10, int(math.log2(max(1024, config.batch_size * config.seq_len))) + 1)],
            "kernel_providers": ["liger", "huggingface", "torch_compile"],
            "extra_benchmark_configs": [
                {"vocab_size": model.vocab_size, "hidden_size": model.hidden_size, "dtype": model.dtype, "T": T}
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_embedding,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_embedding,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
