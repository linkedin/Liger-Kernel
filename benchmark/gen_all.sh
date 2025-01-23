#!/bin/bash

rm -rf data/*
rm -rf visualizations/*

python scripts/benchmark_rms_norm.py
python benchmarks_visualizer.py --kernel-name rms_norm --metric-name speed --kernel-operation-mode full
python benchmarks_visualizer.py --kernel-name rms_norm --metric-name memory --kernel-operation-mode full

python scripts/benchmark_layer_norm.py
python benchmarks_visualizer.py --kernel-name layer_norm --metric-name speed --kernel-operation-mode full
python benchmarks_visualizer.py --kernel-name layer_norm --metric-name memory --kernel-operation-mode full

python scripts/benchmark_rope.py
python benchmarks_visualizer.py --kernel-name rope --metric-name speed --kernel-operation-mode full
python benchmarks_visualizer.py --kernel-name rope --metric-name memory --kernel-operation-mode full

python scripts/benchmark_cross_entropy.py
python benchmarks_visualizer.py --kernel-name cross_entropy --metric-name speed --kernel-operation-mode full
python benchmarks_visualizer.py --kernel-name cross_entropy --metric-name memory --kernel-operation-mode full

python scripts/benchmark_swiglu.py
python benchmarks_visualizer.py --kernel-name swiglu --metric-name speed --kernel-operation-mode full
python benchmarks_visualizer.py --kernel-name swiglu --metric-name memory --kernel-operation-mode full

python scripts/benchmark_geglu.py
python benchmarks_visualizer.py --kernel-name geglu --metric-name speed --kernel-operation-mode full
python benchmarks_visualizer.py --kernel-name geglu --metric-name memory --kernel-operation-mode full
