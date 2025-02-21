## Benchmarking Liger Kernels

Follow these steps to benchmark and visualize kernel performance:

1. Create a benchmark script
   - Add your script under `benchmark/scripts/`
   - Name it according to the kernel (e.g., `benchmark_<kernel_name>.py`)

2. Run the benchmark
   - Results will be saved to `benchmark/data/all_benchmark_data.csv`
   
   Example: Benchmarking KTO Loss
   ```bash
   cd benchmark
   python scripts/benchmark_kto_loss.py
   ```

3. Visualize results
   - Use the visualization script with appropriate parameters
   
   Example: Visualizing KTO Loss benchmark results
   ```bash
   python benchmarks_visualizer.py \
       --kernel-name kto_loss \
       --metric-name memory \
       --kernel-operation-mode full
   ```

4. View results
   - Generated plots will be saved in `benchmark/visualizations/`