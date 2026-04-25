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
   python scripts/benchmark_kto_loss.py --sweep-mode model_config [--model llama_3_8b]
   python scripts/benchmark_kto_loss.py [--sweep-mode token_length] [--bt 2048]
   ```

3. Visualize results
   - Use the visualization script with optional modes:

     * `--sweep-mode`: Select which sweep data to plot.
       - `token_length` (default): plots where x-axis is sequence length.
       - `model_config`: plots where x-axis is model configuration.
     * To target specific operation mode(s), pass `--kernel-operation-mode` one or more values.
     * If you omit `--kernel-operation-mode`, the script will:
       - For `speed` metrics: generate plots for all available modes (forward/backward/full).
       - For `memory` metrics: generate only the `full` plot.

   Examples:
   1. Token-length sweep, specific modes (speed):
   ```bash
   python benchmarks_visualizer.py \
       --kernel-name kto_loss \
       --metric-name speed \
       --kernel-operation-mode forward backward
   ```
   2. Token-length sweep, all modes (speed):
   ```bash
   python benchmarks_visualizer.py \
       --kernel-name kto_loss \
       --metric-name speed
   ```
   3. Model-config sweep (speed):
   ```bash
   python benchmarks_visualizer.py \
       --kernel-name geglu \
       --metric-name speed \
       --sweep-mode model_config
   ```
   4. Memory (always full):
   ```bash
   python benchmarks_visualizer.py \
       --kernel-name kto_loss \
       --metric-name memory
   ```

4. View results
   - Generated plots will be saved in `benchmark/visualizations/`
   - Filenames include the sweep mode when specified (e.g. `geglu_speed_full_model_config.png`)