#!/usr/bin/env python3
"""Plot GRPO memory usage comparisons."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add both benchmark and scripts directories to path
benchmark_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(benchmark_dir, "scripts")
sys.path.insert(0, benchmark_dir)
sys.path.insert(0, scripts_dir)

from benchmark_grpo_loss import bench_memory_grpo_loss
from utils import SingleBenchmarkRunInput

print("Running memory benchmarks for GRPO...")

# Configuration for memory benchmarks
batch_sizes = [2, 4, 8]
configs = {"torch": [], "liger": [], "triton": []}

# Run memory benchmarks for different providers
for B in batch_sizes:
    print(f"\nBatch size {B}...")

    # Liger (fused) memory
    liger_input = SingleBenchmarkRunInput(
        x=B,
        kernel_provider="liger",
        kernel_operation_mode="full",
        extra_benchmark_config={
            "T": 512,
            "H": 2048,
            "V": 8192,
            "importance_sampling_level": "token",
            "loss_type": "bnpo",
            "max_completion_length": None,
            "dtype": torch.bfloat16,
            "beta": 0.04,
            "epsilon_low": 0.2,
            "epsilon_high": 0.2,
            "temperature": 1.0,
            "compiled": False,
        },
    )
    result = bench_memory_grpo_loss(liger_input)
    configs["liger"].append(result.y_50)
    print(f"  Liger: {result.y_50:.2f} MB")

    # PyTorch memory
    torch_input = SingleBenchmarkRunInput(
        x=B,
        kernel_provider="torch",
        kernel_operation_mode="full",
        extra_benchmark_config={
            "T": 512,
            "H": 2048,
            "V": 8192,
            "importance_sampling_level": "token",
            "loss_type": "bnpo",
            "max_completion_length": None,
            "dtype": torch.bfloat16,
            "beta": 0.04,
            "epsilon_low": 0.2,
            "epsilon_high": 0.2,
            "temperature": 1.0,
            "compiled": False,
        },
    )
    result = bench_memory_grpo_loss(torch_input)
    configs["torch"].append(result.y_50)
    print(f"  PyTorch: {result.y_50:.2f} MB")

    # Triton memory
    triton_input = SingleBenchmarkRunInput(
        x=B,
        kernel_provider="triton",
        kernel_operation_mode="full",
        extra_benchmark_config={
            "T": 512,
            "V": 8192,
            "importance_sampling_level": "token",
            "loss_type": "bnpo",
            "max_completion_length": None,
            "dtype": torch.float32,
            "beta": 0.04,
            "epsilon_low": 0.2,
            "epsilon_high": 0.2,
            "temperature": 1.0,
        },
    )
    result = bench_memory_grpo_loss(triton_input)
    configs["triton"].append(result.y_50)
    print(f"  Triton: {result.y_50:.2f} MB")

print("\n" + "=" * 60)
print("Creating visualizations...")

# Create memory usage comparison plot
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(batch_sizes))
width = 0.25

bars1 = ax.bar(x - width, configs["torch"], width, label="PyTorch", color="#ff7f0e", alpha=0.8)
bars2 = ax.bar(x, configs["liger"], width, label="Liger (Fused)", color="#2ca02c", alpha=0.8)
bars3 = ax.bar(x + width, configs["triton"], width, label="Triton", color="#1f77b4", alpha=0.8)

ax.set_xlabel("Batch Size", fontsize=12)
ax.set_ylabel("Memory Usage (MB)", fontsize=12)
ax.set_title("GRPO Memory Usage Comparison\n(T=512, H=2048, V=8192, loss_type=bnpo)", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(batch_sizes)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.0f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("grpo_memory_comparison.png", dpi=150, bbox_inches="tight")
print("✓ Plot saved to grpo_memory_comparison.png")
plt.close()

# Create memory efficiency plot
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate memory efficiency (PyTorch baseline / implementation)
torch_baseline = np.array(configs["torch"])
liger_efficiency = torch_baseline / np.array(configs["liger"])
triton_efficiency = torch_baseline / np.array(configs["triton"])

bars1 = ax.bar(x - width / 2, liger_efficiency, width, label="Liger vs PyTorch", color="#2ca02c", alpha=0.8)
bars2 = ax.bar(x + width / 2, triton_efficiency, width, label="Triton vs PyTorch", color="#1f77b4", alpha=0.8)

ax.axhline(y=1.0, color="r", linestyle="--", label="PyTorch Baseline", alpha=0.7, linewidth=2)
ax.set_xlabel("Batch Size", fontsize=12)
ax.set_ylabel("Memory Efficiency (×)", fontsize=12)
ax.set_title("GRPO Memory Efficiency vs PyTorch\n(Higher is Better)", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(batch_sizes)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}×", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("grpo_memory_efficiency.png", dpi=150, bbox_inches="tight")
print("✓ Plot saved to grpo_memory_efficiency.png")

print("\nSummary:")
print(f"{'Provider':<12} {'B=2 (MB)':<12} {'B=4 (MB)':<12} {'B=8 (MB)':<12}")
print("-" * 50)
for provider, values in configs.items():
    print(f"{provider:<12} {values[0]:<12.2f} {values[1]:<12.2f} {values[2]:<12.2f}")

print("\nMemory Savings vs PyTorch:")
print(f"{'Provider':<12} {'B=2':<12} {'B=4':<12} {'B=8':<12}")
print("-" * 50)
for provider in ["liger", "triton"]:
    savings = [(configs["torch"][i] - configs[provider][i]) / configs["torch"][i] * 100 for i in range(3)]
    print(f"{provider:<12} {savings[0]:>10.1f}% {savings[1]:>10.1f}% {savings[2]:>10.1f}%")
