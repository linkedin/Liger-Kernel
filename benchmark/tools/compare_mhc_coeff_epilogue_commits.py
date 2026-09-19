import argparse
import inspect
import json
import math
import os
import statistics
import subprocess
import sys

from pathlib import Path


def _csv_values(value, convert):
    try:
        values = [convert(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def _parse_args():
    parser = argparse.ArgumentParser(description="Compare exact-source mHC coefficient backward implementations.")
    parser.add_argument("--baseline-worktree", type=Path)
    parser.add_argument("--candidate-worktree", type=Path)
    parser.add_argument("--dtype", type=lambda value: _csv_values(value, str), default=["bfloat16"])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--tokens", type=lambda value: _csv_values(value, int), default=[128, 512, 2048])
    parser.add_argument("--streams", type=int, default=4, help="Hyper-connections (HC).")
    parser.add_argument("--channels", type=lambda value: _csv_values(value, int), default=[1024, 2048, 4096, 8192])
    parser.add_argument("--tmax", type=int, default=20)
    parser.add_argument("--modes", type=lambda value: _csv_values(value, str), default=["backward", "full"])
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--cache-mb", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output", type=Path, default=Path("benchmark_mhc_coeff_epilogue.jsonl"))

    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_provider", choices=("baseline", "candidate"), help=argparse.SUPPRESS)
    parser.add_argument("--_worktree", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_round", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    supported_dtypes = {"float16", "bfloat16", "float32"}
    if not set(args.dtype) <= supported_dtypes:
        parser.error(f"--dtype must contain only {sorted(supported_dtypes)}")
    if not set(args.modes) <= {"backward", "full"}:
        parser.error("--modes must contain only backward and full")
    positive_values = {
        "batch": args.batch,
        "streams": args.streams,
        "tmax": args.tmax,
        "rounds": args.rounds,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "cache-mb": args.cache_mb,
        "tokens": min(args.tokens),
        "channels": min(args.channels),
    }
    for name, value in positive_values.items():
        if value <= 0:
            parser.error(f"--{name} must be positive")

    if args._worker:
        if args._provider is None or args._worktree is None or args._round is None:
            parser.error("internal worker mode requires provider, worktree, and round")
    elif args.baseline_worktree is None or args.candidate_worktree is None:
        parser.error("--baseline-worktree and --candidate-worktree are required")
    return args


def _quantiles(values):
    ordered = sorted(values)

    def percentile(fraction):
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "p20_ms": percentile(0.2),
        "median_ms": statistics.median(ordered),
        "p80_ms": percentile(0.8),
    }


def _git(worktree, *args):
    return subprocess.run(
        ["git", "-C", str(worktree), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _worktree_metadata(worktree):
    worktree = worktree.resolve()
    if not worktree.is_dir():
        raise RuntimeError(f"Worktree does not exist: {worktree}")
    status = _git(worktree, "status", "--porcelain")
    if status:
        raise RuntimeError(f"Worktree is not clean: {worktree}\n{status}")
    return {
        "path": str(worktree),
        "commit": _git(worktree, "rev-parse", "HEAD"),
        "branch": _git(worktree, "branch", "--show-current") or None,
    }


def _worker_command(args, provider, worktree, round_index):
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--_provider",
        provider,
        "--_worktree",
        str(worktree),
        "--_round",
        str(round_index),
        "--dtype",
        ",".join(args.dtype),
        "--batch",
        str(args.batch),
        "--tokens",
        ",".join(map(str, args.tokens)),
        "--streams",
        str(args.streams),
        "--channels",
        ",".join(map(str, args.channels)),
        "--tmax",
        str(args.tmax),
        "--modes",
        ",".join(args.modes),
        "--rounds",
        str(args.rounds),
        "--warmup",
        str(args.warmup),
        "--repetitions",
        str(args.repetitions),
        "--cache-mb",
        str(args.cache_mb),
        "--seed",
        str(args.seed),
    ]


def _run_coordinator(args):
    worktrees = {
        "baseline": _worktree_metadata(args.baseline_worktree),
        "candidate": _worktree_metadata(args.candidate_worktree),
    }
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:

        def emit(payload):
            line = json.dumps(payload, sort_keys=True)
            print(line)
            print(line, file=output, flush=True)

        emit(
            {
                "kind": "run",
                "worktrees": worktrees,
                "rounds": args.rounds,
                "warmup": args.warmup,
                "repetitions": args.repetitions,
            }
        )
        for round_index in range(args.rounds):
            providers = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
            for provider in providers:
                worktree = Path(worktrees[provider]["path"])
                print(f"round={round_index} provider={provider} worktree={worktree}", file=sys.stderr, flush=True)
                environment = os.environ.copy()
                environment["PYTHONPATH"] = os.pathsep.join((str(worktree / "src"), str(worktree)))
                result = subprocess.run(
                    _worker_command(args, provider, worktree, round_index),
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                if result.returncode:
                    sys.stderr.write(result.stderr)
                    raise RuntimeError(f"{provider} worker failed with exit code {result.returncode}")
                for line in result.stdout.splitlines():
                    payload = json.loads(line)
                    emit(payload)
                    if payload.get("kind") == "round":
                        records.append(payload)

        keys = sorted({(record["dtype"], record["T"], record["C"], record["mode"]) for record in records})
        print("dtype      B  HC     T      C  mode      baseline_ms candidate_ms speedup", file=sys.stderr)
        for dtype, tokens, channels, mode in keys:
            selected = [
                record
                for record in records
                if (record["dtype"], record["T"], record["C"], record["mode"]) == (dtype, tokens, channels, mode)
            ]
            summaries = {}
            for provider in ("baseline", "candidate"):
                medians = [record["latency"]["median_ms"] for record in selected if record["provider"] == provider]
                if len(medians) != args.rounds:
                    raise RuntimeError(
                        f"Expected {args.rounds} rounds for {dtype}/T{tokens}/C{channels}/{mode}/{provider}, "
                        f"got {len(medians)}"
                    )
                summaries[provider] = _quantiles(medians)
                emit(
                    {
                        "kind": "summary",
                        "provider": provider,
                        "dtype": dtype,
                        "B": args.batch,
                        "T": tokens,
                        "HC": args.streams,
                        "C": channels,
                        "tmax": args.tmax,
                        "mode": mode,
                        "round_median_quantiles": summaries[provider],
                    }
                )
            speedup = summaries["baseline"]["median_ms"] / summaries["candidate"]["median_ms"]
            emit(
                {
                    "kind": "comparison",
                    "dtype": dtype,
                    "B": args.batch,
                    "T": tokens,
                    "HC": args.streams,
                    "C": channels,
                    "tmax": args.tmax,
                    "mode": mode,
                    "baseline_ms": summaries["baseline"]["median_ms"],
                    "candidate_ms": summaries["candidate"]["median_ms"],
                    "speedup": speedup,
                }
            )
            print(
                f"{dtype:10} {args.batch:2} {args.streams:3} {tokens:5} {channels:6} {mode:8} "
                f"{summaries['baseline']['median_ms']:11.6f} {summaries['candidate']['median_ms']:12.6f} "
                f"{speedup:7.3f}x",
                file=sys.stderr,
            )
    print(f"JSONL: {args.output.resolve()}", file=sys.stderr)


def _import_exact_source(worktree):
    worktree = worktree.resolve()
    sys.path.insert(0, str(worktree / "src"))
    sys.path.insert(1, str(worktree))
    import torch
    import triton

    from liger_kernel.transformers.functional import LigerMHCCoeffsFunction
    from liger_kernel.transformers.functional import liger_mhc_coeffs

    paths = [
        Path(liger_mhc_coeffs.__code__.co_filename).resolve(),
        Path(inspect.getfile(LigerMHCCoeffsFunction)).resolve(),
    ]
    for path in paths:
        if worktree not in path.parents:
            raise RuntimeError(f"Imported {path}, expected source under {worktree}")
    return torch, triton, liger_mhc_coeffs, paths


def _run_worker(args):
    torch, triton, liger_mhc_coeffs, imported_paths = _import_exact_source(args._worktree)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    props = torch.cuda.get_device_properties(0)
    print(
        json.dumps(
            {
                "kind": "worker_environment",
                "provider": args._provider,
                "round": args._round,
                "worktree": str(args._worktree.resolve()),
                "imported": [str(path) for path in imported_paths],
                "gpu": props.name,
                "capability": torch.cuda.get_device_capability(0),
                "sm_count": props.multi_processor_count,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "triton": triton.__version__,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    cache = torch.empty(args.cache_mb * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    def launch_quantiles(fn):
        fn()
        torch.cuda.synchronize()
        for _ in range(args.warmup):
            cache.zero_()
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.repetitions)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.repetitions)]
        for start, end in zip(starts, ends):
            cache.zero_()
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        return _quantiles([start.elapsed_time(end) for start, end in zip(starts, ends)])

    def run_shape(dtype_name, tokens, channels):
        dtype = dtype_map[dtype_name]
        generator = torch.Generator(device="cuda")
        generator.manual_seed(args.seed + tokens + channels + 17 * args.streams + 101 * args.tmax)
        k = args.streams * channels
        m = args.streams * args.streams + 2 * args.streams
        x = torch.randn(
            args.batch,
            tokens,
            args.streams,
            channels,
            generator=generator,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )
        phi = (torch.randn(k, m, generator=generator, device="cuda", dtype=dtype) * 0.02).requires_grad_(True)
        b = (torch.randn(m, generator=generator, device="cuda", dtype=torch.float32) * 0.01).requires_grad_(True)
        alpha_pre = torch.tensor(0.9, device="cuda", dtype=torch.float32, requires_grad=True)
        alpha_post = torch.tensor(1.1, device="cuda", dtype=torch.float32, requires_grad=True)
        alpha_res = torch.tensor(0.8, device="cuda", dtype=torch.float32, requires_grad=True)
        inputs = (x, phi, b, alpha_pre, alpha_post, alpha_res)

        def forward():
            return liger_mhc_coeffs(
                *inputs,
                allow_fp32=dtype == torch.float32,
                tmax=args.tmax,
                rms_eps=1e-6,
                pre_eps=1e-4,
                sinkhorn_eps=1e-6,
                post_mult=2.0,
            )

        outputs = forward()
        grad_outputs = tuple(
            torch.randn(output.shape, generator=generator, device="cuda", dtype=output.dtype) * 0.7 + 0.1
            for output in outputs
        )

        def backward():
            return torch.autograd.grad(outputs, inputs, grad_outputs, retain_graph=True)

        def full():
            return torch.autograd.grad(forward(), inputs, grad_outputs)

        functions = {"backward": backward, "full": full}
        mode_order = args.modes if (tokens + channels + args._round) % 2 == 0 else list(reversed(args.modes))
        for mode in mode_order:
            print(
                json.dumps(
                    {
                        "kind": "round",
                        "provider": args._provider,
                        "round": args._round,
                        "dtype": dtype_name,
                        "B": args.batch,
                        "T": tokens,
                        "HC": args.streams,
                        "C": channels,
                        "tmax": args.tmax,
                        "mode": mode,
                        "warmup": args.warmup,
                        "repetitions": args.repetitions,
                        "latency": launch_quantiles(functions[mode]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    for dtype_name in args.dtype:
        for tokens in args.tokens:
            for channels in args.channels:
                run_shape(dtype_name, tokens, channels)
                torch.cuda.empty_cache()


def main():
    args = _parse_args()
    if args._worker:
        _run_worker(args)
    else:
        _run_coordinator(args)


if __name__ == "__main__":
    main()
