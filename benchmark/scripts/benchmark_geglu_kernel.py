"""Paired SM103 benchmark for the legacy and column-tiled GeGLU kernels."""

import argparse
import json
import statistics
import sys

from contextlib import contextmanager

import torch
import triton

import liger_kernel.ops.geglu as geglu_ops

from liger_kernel.utils import infer_device_arch

QUANTILES = (0.5, 0.2, 0.8)
PROVIDERS = ("legacy", "dispatched")
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=_positive_int, nargs="+", required=True)
    parser.add_argument("--cols", type=_positive_int, nargs="+", required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), required=True)
    parser.add_argument("--warmup", type=_positive_int, default=100, help="do_bench warmup time in milliseconds")
    parser.add_argument("--rep", type=_positive_int, default=500, help="do_bench measurement time in milliseconds")
    parser.add_argument("--rounds", type=_positive_int, default=3, help="number of alternating provider pairs")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@contextmanager
def _force_provider(provider):
    original = geglu_ops.infer_device_arch
    forced_arch = "hopper" if provider == "legacy" else "blackwell_ultra"
    geglu_ops.infer_device_arch = lambda: forced_arch
    try:
        yield
    finally:
        geglu_ops.infer_device_arch = original


def _bench(fn, warmup, rep):
    median, p20, p80 = triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        quantiles=QUANTILES,
    )
    return {"p20_ms": p20, "median_ms": median, "p80_ms": p80}


def _bench_provider(provider, a, b, dc, warmup, rep):
    a_backward = a.clone()
    b_backward = b.clone()
    with _force_provider(provider):
        geglu_ops.geglu_forward(a, b)
        geglu_ops.geglu_backward(a_backward, b_backward, dc)
        torch.cuda.synchronize()

        a_backward.copy_(a)
        b_backward.copy_(b)
        forward = _bench(lambda: geglu_ops.geglu_forward(a, b), warmup, rep)
        backward = _bench(lambda: geglu_ops.geglu_backward(a_backward, b_backward, dc), warmup, rep)

    full = {key: forward[key] + backward[key] for key in forward}
    return {"forward": forward, "backward": backward, "full": full}


def _aggregate(samples):
    return {
        operation: {
            quantile: statistics.median(sample[operation][quantile] for sample in samples)
            for quantile in ("p20_ms", "median_ms", "p80_ms")
        }
        for operation in ("forward", "backward", "full")
    }


def _speedups(legacy, dispatched):
    return {
        operation: legacy[operation]["median_ms"] / dispatched[operation]["median_ms"]
        for operation in ("forward", "backward", "full")
    }


def _print_summary(rows, cols, dtype, legacy, dispatched, speedups):
    print(f"rows={rows} cols={cols} dtype={dtype}", file=sys.stderr)
    for operation in ("forward", "backward", "full"):
        print(
            f"  {operation}: legacy={legacy[operation]['median_ms']:.6f} ms "
            f"dispatched={dispatched[operation]['median_ms']:.6f} ms "
            f"speedup={speedups[operation]:.3f}x",
            file=sys.stderr,
        )


def main():
    args = _parse_args()
    torch.cuda.set_device(0)
    arch = infer_device_arch()
    if arch != "blackwell_ultra":
        raise SystemExit(f"This benchmark requires SM103 (blackwell_ultra), detected {arch}")

    dtype = DTYPES[args.dtype]
    environment = {
        "kind": "environment",
        "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "dtype": args.dtype,
        "rows": args.rows,
        "cols": args.cols,
        "warmup_ms": args.warmup,
        "rep_ms": args.rep,
        "rounds": args.rounds,
        "seed": args.seed,
    }
    print(json.dumps(environment, sort_keys=True), flush=True)

    for rows in args.rows:
        for cols in args.cols:
            torch.manual_seed(args.seed)
            a = torch.randn((rows, cols), device="cuda", dtype=dtype)
            b = torch.randn_like(a)
            dc = torch.randn_like(a)
            samples = {provider: [] for provider in PROVIDERS}

            for round_index in range(args.rounds):
                provider_order = PROVIDERS if round_index % 2 == 0 else tuple(reversed(PROVIDERS))
                for provider in provider_order:
                    measurements = _bench_provider(provider, a, b, dc, args.warmup, args.rep)
                    samples[provider].append(measurements)
                    print(
                        json.dumps(
                            {
                                "kind": "sample",
                                "provider": provider,
                                "round": round_index,
                                "rows": rows,
                                "cols": cols,
                                "dtype": args.dtype,
                                **measurements,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

            legacy = _aggregate(samples["legacy"])
            dispatched = _aggregate(samples["dispatched"])
            speedups = _speedups(legacy, dispatched)
            print(
                json.dumps(
                    {
                        "kind": "summary",
                        "rows": rows,
                        "cols": cols,
                        "dtype": args.dtype,
                        "legacy": legacy,
                        "dispatched": dispatched,
                        "speedup": speedups,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            _print_summary(rows, cols, args.dtype, legacy, dispatched, speedups)


if __name__ == "__main__":
    main()
