"""Compare the current and fused-linear CuTe DSL SwiGLU forward paths."""

import argparse
import gc
import json
import statistics
import subprocess
import sys
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


def _make_weights(hidden_size, intermediate_size, dtype, device):
    weight_scale = hidden_size**-0.5
    gate_weight = torch.randn(
        intermediate_size,
        hidden_size,
        device=device,
        dtype=dtype,
    ).mul_(weight_scale)
    up_weight = torch.randn_like(gate_weight).mul_(weight_scale)
    return gate_weight, up_weight


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
    parser.add_argument(
        "--memory-provider",
        choices=("baseline", "fused"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _run_memory_provider(args, dtype, device):
    torch.manual_seed(0)
    gate_weight, up_weight = _make_weights(
        args.hidden_size,
        args.intermediate_size,
        dtype,
        device,
    )
    if args.memory_provider == "fused":
        packed_weight, output_features = pack_swiglu_weights(gate_weight, up_weight)
        del gate_weight, up_weight
        gc.collect()
        torch.cuda.empty_cache()

    peaks = {}
    with torch.inference_mode():
        for tokens in args.tokens:
            x = torch.randn(tokens, args.hidden_size, device=device, dtype=dtype)
            if args.memory_provider == "baseline":

                def provider(current_x):
                    gate = F.linear(current_x, gate_weight)
                    up = F.linear(current_x, up_weight)
                    return swiglu_forward(gate, up)[2]

            else:

                def provider(current_x):
                    return fused_swiglu(current_x, packed_weight, output_features)

            warm = provider(x)
            torch.cuda.synchronize()
            del warm
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            output = provider(x)
            torch.cuda.synchronize()
            peaks[tokens] = torch.cuda.max_memory_allocated()
            del output, x
            gc.collect()
            torch.cuda.empty_cache()
    print(f"PEAK_BYTES={json.dumps(peaks, sort_keys=True)}")


def _measure_isolated_peaks(args, provider):
    command = [
        sys.executable,
        __file__,
        "--tokens",
        *(str(tokens) for tokens in args.tokens),
        "--hidden-size",
        str(args.hidden_size),
        "--intermediate-size",
        str(args.intermediate_size),
        "--dtype",
        args.dtype,
        "--memory-provider",
        provider,
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    line = next(line for line in reversed(result.stdout.splitlines()) if line.startswith("PEAK_BYTES="))
    return {int(tokens): peak for tokens, peak in json.loads(line.removeprefix("PEAK_BYTES=")).items()}


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
    if args.memory_provider is not None:
        _run_memory_provider(args, dtype, device)
        return

    baseline_peaks = _measure_isolated_peaks(args, "baseline")
    fused_peaks = _measure_isolated_peaks(args, "fused")

    torch.manual_seed(0)
    gate_weight, up_weight = _make_weights(
        args.hidden_size,
        args.intermediate_size,
        dtype,
        device,
    )
    packed_weight, output_features = pack_swiglu_weights(gate_weight, up_weight)

    print(
        "| tokens | two GEMMs + SwiGLU (ms) | fused SwiGLU (ms) | speedup | "
        "baseline total peak (MiB) | fused total peak (MiB) | memory ratio |"
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
            elementwise_peak = baseline_peaks[tokens]
            fused_peak = fused_peaks[tokens]
            print(
                f"| {tokens} | {elementwise_ms:.6f} | {fused_ms:.6f} | "
                f"{elementwise_ms / fused_ms:.3f}x | {elementwise_peak / 2**20:.0f} | "
                f"{fused_peak / 2**20:.0f} | {elementwise_peak / fused_peak:.2f}x |"
            )
    print(
        "\nWeight packing is excluded from latency. Memory is total peak PyTorch CUDA allocation "
        "from isolated provider processes, including input, one weight representation, and output."
    )


if __name__ == "__main__":
    main()
