"""Compare the current and fused-linear CuTe DSL SwiGLU forward paths."""

import argparse
import gc
import statistics
import time

import torch
import torch.nn.functional as F

from liger_kernel.ops.cutedsl.ops.swiglu import fused_swiglu
from liger_kernel.ops.cutedsl.ops.swiglu import pack_swiglu_weights
from liger_kernel.ops.cutedsl.ops.swiglu import swiglu_forward

_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def _time_once(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _benchmark_pair(elementwise, fused, warmup, samples, cooldown_ms):
    for _ in range(warmup):
        elementwise()
        fused()
    torch.cuda.synchronize()

    timings = {"cutedsl-elementwise": [], "cutedsl-fused": []}
    for sample in range(samples):
        providers = (
            (("cutedsl-elementwise", elementwise), ("cutedsl-fused", fused))
            if sample % 2 == 0
            else (("cutedsl-fused", fused), ("cutedsl-elementwise", elementwise))
        )
        for name, fn in providers:
            timings[name].append(_time_once(fn))
        if cooldown_ms:
            time.sleep(cooldown_ms / 1000)
    return {name: statistics.median(values) for name, values in timings.items()}


def _measure_peak_activation_bytes(fn):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    del output
    return peak


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument(
        "--cooldown-ms",
        type=float,
        default=20.0,
        help="GPU idle time between interleaved sample pairs to limit clock droop.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("The fused-linear provider requires an exact SM100 GPU.")
    if args.hidden_size % 64:
        raise ValueError("--hidden-size must be divisible by 64.")
    if min(args.tokens) <= 0:
        raise ValueError("--tokens values must be positive.")

    dtype = _DTYPES[args.dtype]
    device = torch.device("cuda")
    torch.manual_seed(0)
    weight_scale = args.hidden_size**-0.5
    gate_weight = torch.randn(
        args.intermediate_size,
        args.hidden_size,
        device=device,
        dtype=dtype,
    ).mul_(weight_scale)
    up_weight = torch.randn_like(gate_weight).mul_(weight_scale)
    packed_weight, output_features = pack_swiglu_weights(gate_weight, up_weight)

    print(
        "| tokens | two GEMMs + SwiGLU (ms) | fused SwiGLU (ms) | speedup | "
        "baseline peak (MiB) | fused peak (MiB) | memory ratio |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    with torch.inference_mode():
        for tokens in args.tokens:
            x = torch.randn(tokens, args.hidden_size, device=device, dtype=dtype)

            def elementwise():
                gate = F.linear(x, gate_weight)
                up = F.linear(x, up_weight)
                return swiglu_forward(gate, up)[2]

            def fused():
                return fused_swiglu(x, packed_weight, output_features)

            torch.testing.assert_close(
                fused().float(),
                elementwise().float(),
                atol=0.05,
                rtol=0.03,
            )
            timings = _benchmark_pair(
                elementwise,
                fused,
                args.warmup,
                args.samples,
                args.cooldown_ms,
            )
            elementwise_ms = timings["cutedsl-elementwise"]
            fused_ms = timings["cutedsl-fused"]
            elementwise_peak = _measure_peak_activation_bytes(elementwise)
            fused_peak = _measure_peak_activation_bytes(fused)
            print(
                f"| {tokens} | {elementwise_ms:.6f} | {fused_ms:.6f} | "
                f"{elementwise_ms / fused_ms:.3f}x | {elementwise_peak / 2**20:.0f} | "
                f"{fused_peak / 2**20:.0f} | {elementwise_peak / fused_peak:.2f}x |"
            )
    print(
        "\nWeight packing is excluded from latency. Peak memory is incremental activation "
        "allocation above persistent inputs and weights; packed and unpacked weights have equal storage."
    )


if __name__ == "__main__":
    main()
