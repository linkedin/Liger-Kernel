#!/usr/bin/env python3
"""Plot GRPO baseline vs vocab-chunked diffs from benchmark CSVs."""

import argparse
import csv
import json

from collections import defaultdict
from typing import Dict
from typing import Tuple

import matplotlib.pyplot as plt


def _load_rows(path: str) -> Dict[Tuple[str, str, str, float, str, str], float]:
    rows = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["kernel_provider"] != "liger":
                continue
            metric_name = row["metric_name"]
            if metric_name not in {"speed", "memory"}:
                continue
            try:
                extra_cfg = json.loads(row["extra_benchmark_config_str"])
            except json.JSONDecodeError:
                extra_cfg = {}
            importance_sampling_level = extra_cfg.get("importance_sampling_level")
            if importance_sampling_level not in {"token", "sequence"}:
                continue
            key = (
                row["kernel_name"],
                row["kernel_operation_mode"] or "",
                metric_name,
                float(row["x_value"]),
                importance_sampling_level,
                row["gpu_name"],
            )
            rows[key] = float(row["y_value_50"])
    return rows


def _collect_diffs(baseline_rows, new_rows, metric_name, operation_mode):
    diffs = defaultdict(dict)
    for key, base_val in baseline_rows.items():
        kernel_name, mode, metric, x_value, level, gpu_name = key
        if metric != metric_name or mode != operation_mode:
            continue
        new_val = new_rows.get(key)
        if new_val is None:
            continue
        pct_change = (new_val - base_val) / base_val * 100.0
        diffs[level][x_value] = pct_change
    return diffs


def main():
    parser = argparse.ArgumentParser(description="Plot GRPO benchmark diffs.")
    parser.add_argument(
        "--baseline",
        default="/mnt/home/kashif/Liger-Kernel-baseline/benchmark/data/grpo_baseline.csv",
        help="Baseline CSV path.",
    )
    parser.add_argument(
        "--new",
        default="benchmark/data/grpo_vocab_chunk.csv",
        help="New CSV path.",
    )
    parser.add_argument(
        "--output",
        default="benchmark/grpo_diff.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--mode",
        default="full",
        choices=["forward", "full", "backward"],
        help="Operation mode to compare for speed.",
    )
    args = parser.parse_args()

    baseline_rows = _load_rows(args.baseline)
    new_rows = _load_rows(args.new)

    speed_diffs = _collect_diffs(baseline_rows, new_rows, "speed", args.mode)
    memory_diffs = _collect_diffs(baseline_rows, new_rows, "memory", "full")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")
    fig.suptitle("GRPO Liger Diff: New vs Baseline (percent change)", fontsize=14, fontweight="bold")

    for col, level in enumerate(["token", "sequence"]):
        xs = sorted(set(speed_diffs.get(level, {}).keys()))
        ys = [speed_diffs.get(level, {}).get(x, 0.0) for x in xs]
        ax = axes[0][col]
        ax.bar(xs, ys, color="#2a9d8f")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Speed ({args.mode}) - {level}")
        ax.set_xlabel("B")
        if col == 0:
            ax.set_ylabel("% change (ms)")

        xs = sorted(set(memory_diffs.get(level, {}).keys()))
        ys = [memory_diffs.get(level, {}).get(x, 0.0) for x in xs]
        ax = axes[1][col]
        ax.bar(xs, ys, color="#e76f51")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Memory (full) - {level}")
        ax.set_xlabel("B")
        if col == 0:
            ax.set_ylabel("% change (MB)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
