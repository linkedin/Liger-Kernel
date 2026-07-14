"""Run an RMSNorm benchmark three ways (HuggingFace / Triton / CuTe DSL) into one CSV.

Usage (from ``benchmark/scripts``)::

    python run_cutedsl_compare.py --kernel rms_norm --source h100 [benchmark args...]
    python run_cutedsl_compare.py --kernel rms_norm --source b200 [benchmark args...]

Why a driver: selecting a Liger backend is a process-global, import-time decision
(``LIGER_KERNEL_IMPL`` is read when ``liger_kernel.ops`` is first imported). To get
both the Triton and the CuTe DSL kernels in one comparison without leaking backend
registration state between them, we spawn the per-kernel benchmark script twice in
separate subprocesses:

  * baseline : LIGER_KERNEL_IMPL unset  -> provider rows "huggingface" + "liger_triton"
  * cutedsl  : LIGER_KERNEL_IMPL=cutedsl -> provider rows "huggingface" + "liger_cutedsl"

Both runs write to ``benchmark/data/all_benchmark_data_cutedsl_rms_norm[_<source>].csv``
(``LIGER_BENCH_TARGET``); the ``liger`` provider is renamed per run via
``LIGER_BENCH_PROVIDER_TAG`` so the three series coexist without colliding on the
CSV dedup key. The runs are executed sequentially because the CSV writer does an
unlocked read-modify-write.

``--source`` keeps H100 and B200 results in *separate* files on a shared checkout,
and deliberately avoids the ``all_benchmark_data_cutedsl_h100.csv`` /
``all_benchmark_data_cutedsl_b200.csv`` names so genuine CuTe DSL results are never
merged into any pre-existing prototype CSV.

Plot with, e.g.::

    python ../benchmarks_visualizer.py --kernel-name rms_norm --metric-name speed \\
        --data-file data/all_benchmark_data_cutedsl_rms_norm_h100.csv
"""

import argparse
import os
import subprocess
import sys

# Kernels that (a) have a CuTe DSL backend and (b) are in scope for this PR.
CUTEDSL_ENABLED_KERNELS = ["rms_norm"]


def main():
    parser = argparse.ArgumentParser(
        description="Compare HuggingFace vs Triton vs CuTe DSL Liger RMSNorm in one CSV.",
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
        help="Optional label (e.g. 'h100' or 'b200') to keep results from different "
        "GPUs in separate CSV files. Appended to the CSV target name.",
    )
    args, passthrough = parser.parse_known_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bench_script = os.path.join(script_dir, f"benchmark_{args.kernel}.py")
    if not os.path.isfile(bench_script):
        print(f"error: benchmark script not found: {bench_script}", file=sys.stderr)
        sys.exit(1)

    # Distinct from the prototype all_benchmark_data_cutedsl_{h100,b200}.csv files,
    # so genuine CuTe DSL rows are never merged into stale cuTile results.
    source = args.source.strip().lower()
    target = f"cutedsl_{args.kernel}" + (f"_{source}" if source else "")

    # Sequential runs (unlocked CSV read-modify-write). provider_tag disambiguates the
    # "liger" rows so the Triton and CuTe DSL series don't overwrite each other.
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
