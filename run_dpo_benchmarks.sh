#!/bin/bash

echo "🚀 Running DPO Memory Benchmarks"
echo "=================================="

# Create output directory
mkdir -p benchmark_results

# Quick test first
echo "📋 Running quick test..."
python test_dpo_benchmark.py

echo ""
echo "📊 Running detailed benchmarks..."

# Test different configurations
configurations=(
    "--loss triton --loss_type sigmoid --global_bs 64 --mbs 8 --mmbs 4 --seq_len 512"
    "--loss liger --loss_type sigmoid --global_bs 64 --mbs 8 --mmbs 4 --seq_len 512"
    "--loss triton --loss_type apo_zero --global_bs 64 --mbs 8 --mmbs 4 --seq_len 512"
    "--loss liger --loss_type apo_zero --global_bs 64 --mbs 8 --mmbs 4 --seq_len 512"
    "--loss triton --loss_type sigmoid --global_bs 64 --mbs 8 --mmbs 4 --seq_len 1024"
    "--loss liger --loss_type sigmoid --global_bs 64 --mbs 8 --mmbs 4 --seq_len 1024"
    "--loss triton --loss_type sigmoid --global_bs 64 --mbs 8 --mmbs 4 --seq_len 512 --use_ref_model"
    "--loss liger --loss_type sigmoid --global_bs 64 --mbs 8 --mmbs 4 --seq_len 512 --use_ref_model"
)

for config in "${configurations[@]}"; do
    echo "Running: python benchmark_dpo_memory.py $config"
    if python benchmark_dpo_memory.py $config; then
        echo "✅ Success"
    else
        echo "❌ Failed with exit code $?"
    fi
    echo "---"
done

echo ""
echo "📈 Benchmark Summary"
echo "===================="
echo "Results saved to dpo_benchmark_infos.jsonl"

# Show summary
if [ -f "dpo_benchmark_infos.jsonl" ]; then
    echo ""
    echo "Recent results:"
    tail -n 8 dpo_benchmark_infos.jsonl | jq -r '"\(.loss_impl) \(.loss_type) seq_len=\(.seq_len) ref=\(.use_ref_model): \(.["memory_reserved(GB)"])GB memory, \(.["samples/s"]) samples/s"'
fi

echo "✨ All benchmarks complete!" 