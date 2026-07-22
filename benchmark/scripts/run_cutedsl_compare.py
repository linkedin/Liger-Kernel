"""Run a kernel benchmark with Triton and CuTe DSL backends into one CSV.

Usage (from ``benchmark/scripts``)::

    python run_cutedsl_compare.py --kernel cross_entropy [benchmark args...]
    python run_cutedsl_compare.py --kernel rms_norm --source h100 [benchmark args...]

Backend selection is process-global and happens at import time, so this driver
spawns the underlying benchmark script twice:

  * baseline: ``LIGER_KERNEL_IMPL`` unset
  * CuTe DSL: ``LIGER_KERNEL_IMPL=cutedsl``

``LIGER_BENCH_PROVIDER_TAG`` keeps the ``liger_triton`` and ``liger_cutedsl``
rows distinct. Runs are sequential because the CSV writer performs an unlocked
read-modify-write.

For backward compatibility, cross-entropy without ``--source`` writes
``all_benchmark_data_cutedsl.csv``. Other runs use a kernel-specific target,
such as ``all_benchmark_data_cutedsl_rms_norm_h100.csv``.
"""

import argparse
import os
import subprocess
import sys

CUTEDSL_ENABLED_KERNELS = ["cross_entropy", "rms_norm"]


def main():
    parser = argparse.ArgumentParser(
        description="Compare Triton and CuTe DSL Liger kernels in one CSV.",
        # Unknown args are forwarded to the underlying benchmark script.
    )
    parser.add_argument(
        "--kernel",
        default="rms_norm",
        choices=CUTEDSL_ENABLED_KERNELS,
        help="Kernel to compare. Must have a CuTe DSL backend.",
    )
    parser.add_argument(
        "--source",
        default="",
        help="Optional label (for example, 'h100' or 'b200') appended to the CSV target name.",
    )
    args, passthrough = parser.parse_known_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bench_script = os.path.join(script_dir, f"benchmark_{args.kernel}.py")
    if not os.path.isfile(bench_script):
        print(f"error: benchmark script not found: {bench_script}", file=sys.stderr)
        sys.exit(1)

    source = args.source.strip().lower()
    if args.kernel == "cross_entropy" and not source:
        target = "cutedsl"
    else:
        target = f"cutedsl_{args.kernel}" + (f"_{source}" if source else "")

    runs = [
        ("triton baseline", {"LIGER_KERNEL_IMPL": "", "LIGER_BENCH_PROVIDER_TAG": "liger_triton"}),
        ("cutedsl", {"LIGER_KERNEL_IMPL": "cutedsl", "LIGER_BENCH_PROVIDER_TAG": "liger_cutedsl"}),
    ]

    for label, run_env in runs:
        print(f"\n========== {args.kernel} [{source or 'default'}]: {label} ==========\n", flush=True)
        env = {**os.environ, "LIGER_BENCH_TARGET": target, **run_env}
        result = subprocess.run(
            [sys.executable, bench_script, *passthrough],
            env=env,
            cwd=script_dir,
        )
        if result.returncode != 0:
            print(f"error: {label} run failed with exit code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)

    print(f"\nWrote merged results to benchmark/data/all_benchmark_data_{target}.csv", flush=True)


if __name__ == "__main__":
    main()
