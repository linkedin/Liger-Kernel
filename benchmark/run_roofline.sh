#!/usr/bin/env bash
# Usage: NCU=/path/to/ncu bash benchmark/run_roofline.sh [outdir]
set -euo pipefail
NCU=${NCU:-ncu}
OUT=${1:-roofline_out}
M=gpu__time_duration.sum,dram__bytes_read.sum
mkdir -p "$OUT"
run() {  # name, kernel-regex
  # caches flushed before each launch; clocks left unlocked, as in training (a base-clock lock slows these kernels a lot)
  "$NCU" --csv --print-units base --metrics $M --cache-control all --clock-control none \
    --kernel-name "regex:$2" --launch-skip 10 --launch-count 20 \
    python3 benchmark/benchmark_roofline.py --kernel "$1" > "$OUT/$1.csv"
}
run copy vectorized_elementwise_kernel
run mul vectorized_elementwise_kernel
run rmsnorm _rms_norm_forward_kernel
run swiglu _swiglu_forward_kernel
python3 benchmark/roofline_report.py "$OUT"
