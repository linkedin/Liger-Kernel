import torch
import triton

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

# NOTE: For torch compile, we will just use default inductor settings. No further customization
# is needed.


def bench_speed_embedding(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    V = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    B = input.extra_benchmark_config["B"]
    T = input.extra_benchmark_config["T"]
    D = input.extra_benchmark_config["D"]
    dtype = input.extra_benchmark_config["dtype"]

    torch_emb = Embedding(V, D).to(device).to(dtype)
    liger_emb = LigerEmbedding(V, D).to(device).to(dtype)
    torch_compile_emb = torch.compile(torch_emb)

    input_ids = torch.randint(0, V, (B, T), device=device)

    def fwd():
        if provider == "liger":
            return liger_emb(input_ids)
        elif provider == "torch_compile":
            return torch_compile_emb(input_ids)
        else:
            return torch_emb(input_ids)

    def full():
        output = fwd()
        output.backward(torch.randn_like(output))

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, quantiles=QUANTILES, rep=100)
    elif mode == "full":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, quantiles=QUANTILES, rep=100)
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_embedding(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    V = input.x
    provider = input.kernel_provider

    B = input.extra_benchmark_config["B"]
    T = input.extra_benchmark_config["T"]
    D = input.extra_benchmark_config["D"]
    dtype = input.extra_benchmark_config["dtype"]

    torch_emb = Embedding(V, D).to(device).to(dtype)
    liger_emb = LigerEmbedding(V, D).to(device).to(dtype)
    torch_compile_emb = torch.compile(torch_emb)

    input_ids = torch.randint(0, V, (B, T), device=device)

    def fwd():
        if provider == "liger":
            return liger_emb(input_ids)
        elif provider == "torch_compile":
            return torch_compile_emb(input_ids)
        else:
            return torch_emb(input_ids)

    def full():
        output = fwd()
        output.backward(torch.randn_like(output))

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "embedding",
        "x_name": "V",
        "x_label": "embedding dimension",
        "x_values": [2**i for i in range(10, 18)],
        "kernel_providers": ["liger", "huggingface", "torch_compile"],
        "extra_benchmark_configs": [
            # BERT
            {"B": 32, "T": 512, "D": 768, "dtype": torch.float32},
            # Llama
            {"B": 8, "T": 2048, "D": 4096, "dtype": torch.float32},
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_embedding,
        kernel_operation_modes=["forward", "full"],
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
