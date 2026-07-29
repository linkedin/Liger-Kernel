"""Run a kernel benchmark twice (Triton + CuTile) and merge results into one CSV.

Workflow:
    python scripts/run_cutile_compare.py --kernel geglu [benchmark args...]

This driver spawns the per-kernel benchmark script in two subprocesses with
different env vars, so all three series (liger_triton / liger_cutile /
huggingface or torch) land in `benchmark/data/all_benchmark_data_cutile.csv`
under distinct `kernel_provider` values, ready for direct plotting via:

    python ../benchmarks_visualizer.py \
        --kernel-name <name> --metric-name speed \
        --data-file data/all_benchmark_data_cutile.csv
"""

import argparse
import os
import subprocess
import sys

CUTILE_ENABLED_KERNELS = [
    "cross_entropy",
    "fused_linear_jsd",
    "geglu",
    "jsd",
    "layer_norm",
]


def main():
    parser = argparse.ArgumentParser(
        description="Compare Triton vs CuTile Liger kernels in one CSV.",
        # Unknown args are forwarded to the underlying benchmark script.
    )
    parser.add_argument(
        "--kernel",
        required=True,
        choices=CUTILE_ENABLED_KERNELS,
        help="Kernel to compare. Must have a cuTile backend.",
    )
    args, passthrough = parser.parse_known_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bench_script = os.path.join(script_dir, f"benchmark_{args.kernel}.py")
    if not os.path.isfile(bench_script):
        print(f"error: benchmark script not found: {bench_script}", file=sys.stderr)
        sys.exit(1)

    # Both runs target the same _cutile.csv; provider_tag disambiguates the
    # "liger" rows so they don't overwrite each other on the dedup key.
    runs = [
        ("triton baseline", {"LIGER_KERNEL_IMPL": "", "LIGER_BENCH_PROVIDER_TAG": "liger_triton"}),
        ("cutile", {"LIGER_KERNEL_IMPL": "cutile", "LIGER_BENCH_PROVIDER_TAG": "liger_cutile"}),
    ]

    for label, run_env in runs:
        print(f"\n========== {args.kernel}: {label} ==========\n", flush=True)
        env = {**os.environ, "LIGER_BENCH_TARGET": "cutile", **run_env}
        result = subprocess.run(
            [sys.executable, bench_script, *passthrough],
            env=env,
            cwd=script_dir,
        )
        if result.returncode != 0:
            print(f"error: {label} run failed with exit code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
