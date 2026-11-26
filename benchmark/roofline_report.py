"""
Turn ncu CSVs (from benchmark_roofline.py) into a roofline-style table.
Usage: python3 benchmark/roofline_report.py roofline_out
"""

import csv
import statistics as st
import sys

from pathlib import Path

# Must match benchmark_roofline.py
T, H, INTER, B = 4096, 4096, 14336, 2  # tokens, hidden, intermediate, bytes/elem (fp16)
MODEL_BYTES = {  # simple model: compulsory (reads, writes) only
    "copy": (T * H * B, T * H * B),  # read x, write out
    "mul": (2 * T * INTER * B, T * INTER * B),  # read a, b, write c
    "rmsnorm": (T * H * B, T * H * B),  # read x, write y (weight/rstd ignored)
    "swiglu": (2 * T * INTER * B, T * INTER * B),  # read a, b, write c
}
# Each Liger kernel is compared with a plain PyTorch op that reads and writes the same tensors
BASELINE = {"rmsnorm": "copy", "swiglu": "mul"}


def load(path):
    with open(path) as f:
        lines = [line for line in f if line.startswith('"')]
    rows = list(csv.reader(lines))
    idx = {n: i for i, n in enumerate(rows[0])}
    data = {}
    for r in rows[1:]:
        v = float(r[idx["Metric Value"]].replace(",", ""))
        data.setdefault(r[idx["Metric Name"]], []).append(v)
    return {k: st.mean(v) for k, v in data.items()}


def main(d):
    res = {k: load(Path(d) / f"{k}.csv") for k in MODEL_BYTES}

    def gbps(k):  # modeled bytes / kernel time; bytes per ns == GB/s
        return sum(MODEL_BYTES[k]) / res[k]["gpu__time_duration.sum"]

    out = [["Kernel", "Baseline", "Time (us)", "GB/s", "Baseline GB/s", "% of baseline", "DRAM reads / model"]]
    for k, base in BASELINE.items():
        out.append(
            [
                k,
                base,
                f"{res[k]['gpu__time_duration.sum'] / 1e3:.2f}",
                f"{gbps(k):.1f}",
                f"{gbps(base):.1f}",
                f"{100 * gbps(k) / gbps(base):.1f}",
                f"{res[k]['dram__bytes_read.sum'] / MODEL_BYTES[k][0]:.2f}",
            ]
        )
    for r in out:
        print(" | ".join(f"{c:>18}" for c in r))
    with open(Path(d) / "roofline_table.csv", "w", newline="") as f:
        csv.writer(f).writerows(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "roofline_out")
