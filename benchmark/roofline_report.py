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
MODEL_BYTES = {  # simple model: compulsory reads + writes only
    "copy": 2 * T * INTER * B,  # read x, write out
    "rmsnorm": 2 * T * H * B,  # read x, write y (weight/rstd ignored)
    "swiglu": 3 * T * INTER * B,  # read a, b, write c
}


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
    res = {k: load(Path(d) / f"{k}.csv") for k in ["copy", "rmsnorm", "swiglu"]}
    peak = res["copy"]["dram__bytes.sum.per_second"]
    out = [
        [
            "Kernel",
            "Achieved GB/s",
            "% of copy peak",
            "L2 hit %",
            "DRAM MB (measured)",
            "DRAM MB (model)",
            "Measured/Model",
        ]
    ]
    for k, m in res.items():
        bw = m["dram__bytes.sum.per_second"]
        mb = m["dram__bytes.sum"]
        out.append(
            [
                k,
                f"{bw / 1e9:.1f}",
                f"{100 * bw / peak:.1f}",
                f"{m['lts__t_sector_hit_rate.pct']:.1f}",
                f"{mb / 1e6:.1f}",
                f"{MODEL_BYTES[k] / 1e6:.1f}",
                f"{mb / MODEL_BYTES[k]:.2f}",
            ]
        )
    for r in out:
        print(" | ".join(f"{c:>18}" for c in r))
    with open(Path(d) / "roofline_table.csv", "w", newline="") as f:
        csv.writer(f).writerows(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "roofline_out")
