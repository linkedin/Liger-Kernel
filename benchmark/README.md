## Benchmarking Liger Kernels

### Benchmark Framework Overview

The benchmarking system is designed to provide a **consistent, low-boilerplate way** to evaluate kernel performance across:

* Different **model configurations** (e.g., LLaMA and Qwen variants)
* Different **sequence lengths / Batch size * token length**
* Multiple **kernel providers** (e.g., `liger`, `huggingface`)

#### Core Concepts

1. `setup_fn`

    Defines how to **construct inputs and modules** for a single forward pass.

    * Input: `SingleBenchmarkRunInput`
    * Output: tuple of tensors / modules


    ```python
    def _setup_fn(input: SingleBenchmarkRunInput) -> Tuple[Any, ...]:
        x = ...
        layer = ...
        return x, layer
    ```

2. Benchmark Function Builders

    Reusable helpers to generate benchmark functions handle:

    * forward / backward / full modes
    * timing and memory measurement

    ```python
    build_speed_bench_fn(setup_fn)
    build_memory_bench_fn(setup_fn)
    ```

3. Sweep Builders

    (a) `build_model_config_sweep`

    * Sweeps across **model configurations**(e.g. hidden size, dtype, vocab size)
    * Keeps total tokens (`B * T`) approximately constant
    * Automatically derives a suitable `(B, T)` that will not cause OOM under the given token budget
    * `probe_dim` must align with how `input.x` is interpreted in `setup_fn`

    ```python
    common_configs = build_model_config_sweep(
        kernel_name=...,
        all_model_configs=...,
        setup_fn=...,
        model_keys=[...],
        probe_dim: Literal["T", "B", "BT"] = "T"
    )
    ```

    (b) `build_token_length_sweep`

    * Sweeps along a **chosen scaling dimension**:

    * `"T"` → sequence length
    * `"B"` → batch size
    * `"BT"` → total tokens
    * Uses a **single fixed model configuration**
    * Maintains a consistent memory model via bytes-per-token estimation
    * `scale_dim` must align with how `input.x` is interpreted in `setup_fn`

    ```python
    common_configs = build_token_length_sweep(
        kernel_name=...,
        probe_x=...,
        model=...,
        setup_fn=...,
        model_keys=[...],
        scale_dim: Literal["T", "B", "BT"] = "T",
    )
    ```

4. `model_keys` and `extra_configs`

    * `model_keys`: attributes pulled from `ModelConfig`

    * e.g. `["hidden_size", "dtype"]`

    * `extra_configs`: static overrides

    * e.g. `{"eps": 1e-6}`

    These form `extra_benchmark_config`, passed into `setup_fn`.


### Benchmark workflow:

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