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
   - Use the visualization script with optional modes:

     * To target specific mode(s), pass `--kernel-operation-mode` one or more values.
     * If you omit `--kernel-operation-mode`, the script will:
       - For `speed` metrics: generate plots for all available modes (forward/backward/full).
       - For `memory` metrics: generate only the `full` plot.

   Examples:
   1. Specific modes (speed):
   ```bash
   python benchmarks_visualizer.py \
       --kernel-name kto_loss \
       --metric-name speed \
       --kernel-operation-mode forward backward
   ```
   2. All modes (speed):
   ```bash
   python benchmarks_visualizer.py \
       --kernel-name kto_loss \
       --metric-name speed
   ```
   3. Memory (always full):
   ```bash
   python benchmarks_visualizer.py \
       --kernel-name kto_loss \
       --metric-name memory
   ```

4. View results
   - Generated plots will be saved in `benchmark/visualizations/`