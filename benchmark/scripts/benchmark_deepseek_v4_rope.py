import argparse
import functools

import torch
import triton

from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb
from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import run_benchmarks

from liger_kernel.transformers.deepseek_v4_rope import liger_deepseek_v4_rotary_pos_emb
from liger_kernel.utils import infer_device

device = infer_device()


@functools.lru_cache(maxsize=1)
def _compiled_rope():
    return torch.compile(apply_rotary_pos_emb, fullgraph=True)


def _make_input(seq_len, num_heads, head_dim, rope_dim, dtype):
    storage = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    x = storage.transpose(1, 2).detach().requires_grad_(True)
    angles = torch.randn(1, seq_len, rope_dim // 2, device=device, dtype=torch.float32)
    return x, angles.cos(), angles.sin()


def _provider_fn(provider):
    if provider == "huggingface":
        return apply_rotary_pos_emb
    if provider == "torch_compile":
        return _compiled_rope()
    if provider == "liger":
        return liger_deepseek_v4_rotary_pos_emb
    raise ValueError(f"Unsupported provider: {provider}")


def _setup_single(input: SingleBenchmarkRunInput):
    cfg = input.extra_benchmark_config
    x, cos, sin = _make_input(
        input.x,
        cfg["num_heads"],
        cfg["head_dim"],
        cfg["rope_dim"],
        cfg["dtype"],
    )
    grad_output = torch.randn_like(x)
    rope_fn = _provider_fn(input.kernel_provider)
    return (x,), (grad_output,), lambda: rope_fn(x, cos, sin)


def _setup_aggregate(input: SingleBenchmarkRunInput):
    cfg = input.extra_benchmark_config
    q, cos, sin = _make_input(input.x, cfg["q_heads"], cfg["head_dim"], cfg["rope_dim"], cfg["dtype"])
    kv, _, _ = _make_input(input.x, cfg["kv_heads"], cfg["head_dim"], cfg["rope_dim"], cfg["dtype"])
    output, _, _ = _make_input(
        input.x,
        cfg["output_heads"],
        cfg["head_dim"],
        cfg["rope_dim"],
        cfg["dtype"],
    )
    grad_outputs = (torch.randn_like(q), torch.randn_like(kv), torch.randn_like(output))
    rope_fn = _provider_fn(input.kernel_provider)

    def forward():
        return (
            rope_fn(q, cos, sin),
            rope_fn(kv, cos, sin),
            rope_fn(output, cos, -sin),
        )

    return (q, kv, output), grad_outputs, forward


def _operation(inputs, grad_outputs, forward, mode):
    if mode == "forward":
        return forward
    if mode == "backward":
        outputs = forward()
        return lambda: torch.autograd.grad(outputs, inputs, grad_outputs, retain_graph=True)
    if mode == "full":
        return lambda: torch.autograd.grad(forward(), inputs, grad_outputs)
    raise ValueError(f"Unsupported mode: {mode}")


def _speed_benchmark(setup_fn, warmup, repetitions):
    def benchmark(input: SingleBenchmarkRunInput):
        inputs, grad_outputs, forward = setup_fn(input)
        fn = _operation(inputs, grad_outputs, forward, input.kernel_operation_mode)
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=repetitions,
            quantiles=QUANTILES,
        )
        return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)

    return benchmark


def _memory_benchmark(setup_fn, repetitions):
    def benchmark(input: SingleBenchmarkRunInput):
        inputs, grad_outputs, forward = setup_fn(input)
        fn = _operation(inputs, grad_outputs, forward, input.kernel_operation_mode)
        mem_50, mem_20, mem_80 = _test_memory(fn, _iter=repetitions, quantiles=QUANTILES)
        return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)

    return benchmark


def _run_workload(
    kernel_name,
    setup_fn,
    config,
    warmup,
    repetitions,
    memory_repetitions,
    overwrite,
):
    common = {
        "kernel_name": kernel_name,
        "x_name": "T",
        "x_label": "sequence length",
        "x_values": [512, 2048, 4096],
        "kernel_providers": ["huggingface", "torch_compile", "liger"],
        "kernel_operation_modes": ["forward", "backward", "full"],
        "extra_benchmark_configs": [config],
        "overwrite": overwrite,
    }
    run_benchmarks(
        bench_test_fn=_speed_benchmark(setup_fn, warmup, repetitions),
        metric_name="speed",
        metric_unit="ms",
        **common,
    )
    run_benchmarks(
        bench_test_fn=_memory_benchmark(setup_fn, memory_repetitions),
        metric_name="memory",
        metric_unit="MB",
        **common,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-V4 partial interleaved RoPE.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--single-warmup", type=int, default=25)
    parser.add_argument("--single-repetitions", type=int, default=100)
    parser.add_argument("--aggregate-warmup", type=int, default=10)
    parser.add_argument("--aggregate-repetitions", type=int, default=40)
    parser.add_argument("--memory-repetitions", type=int, default=5)
    args = parser.parse_args()

    if device != "cuda":
        raise RuntimeError("DeepSeek-V4 RoPE benchmarks require an NVIDIA GPU.")

    _run_workload(
        kernel_name="deepseek_v4_rope",
        setup_fn=_setup_single,
        config={
            "bsz": 1,
            "num_heads": 64,
            "head_dim": 512,
            "rope_dim": 64,
            "dtype": torch.bfloat16,
            "layout": "transposed_bhsd",
            "warmup": args.single_warmup,
            "repetitions": args.single_repetitions,
        },
        warmup=args.single_warmup,
        repetitions=args.single_repetitions,
        memory_repetitions=args.memory_repetitions,
        overwrite=args.overwrite,
    )
    _run_workload(
        kernel_name="deepseek_v4_rope_aggregate",
        setup_fn=_setup_aggregate,
        config={
            "bsz": 1,
            "q_heads": 64,
            "kv_heads": 1,
            "output_heads": 64,
            "head_dim": 512,
            "rope_dim": 64,
            "dtype": torch.bfloat16,
            "layout": "transposed_bhsd",
            "warmup": args.aggregate_warmup,
            "repetitions": args.aggregate_repetitions,
        },
        warmup=args.aggregate_warmup,
        repetitions=args.aggregate_repetitions,
        memory_repetitions=args.memory_repetitions,
        overwrite=args.overwrite,
    )
