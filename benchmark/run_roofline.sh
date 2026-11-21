#!/usr/bin/env bash
# Usage: NCU=/path/to/ncu bash benchmark/run_roofline.sh [outdir]
set -euo pipefail
NCU=${NCU:-ncu}
OUT=${1:-roofline_out}
M=dram__bytes.sum.per_second,dram__bytes.sum,gpu__time_duration.sum,lts__t_sector_hit_rate.pct
mkdir -p "$OUT"
run() {  # name, kernel-regex
  "$NCU" --csv --print-units base --metrics $M \
    --kernel-name "regex:$2" --launch-skip 10 --launch-count 20 \
    python3 benchmark/benchmark_roofline.py --kernel "$1" > "$OUT/$1.csv"
}
run copy vectorized_elementwise_kernel
run rmsnorm _rms_norm_forward_kernel
run swiglu _swiglu_forward_kernel
python3 benchmark/roofline_report.py "$OUT"
