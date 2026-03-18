# Guideline for Adding Benchmark Scripts

This document describes how to add new benchmark scripts to Liger-Kernel in line with the shared framework.

## 1. Where and how to add a script

- **Location**: `benchmark/scripts/`
- **Naming**: `benchmark_<kernel_name>.py` (e.g. `benchmark_geglu.py`, `benchmark_swiglu.py`)

## 2. Use shared infrastructure

Do **not** hardcode batch size, sequence length, or model dimensions. Use:

| Need | Use |
|------|-----|
| Model dimensions (hidden_size, vocab_size, etc.) | `benchmark_model_configs.py`: `ModelConfig`, `get_benchmark_model_config()` |
| Safe sweep config (seq_len or hidden_size) | `compute_seq_len_sweep_config()` (returns `SeqLenSweepConfig`) or `compute_hidden_size_sweep_config()` (returns `HiddenSizeSweepConfig`), with optional `estimate_kernel_peak_memory()` |
| Speed / memory measurement | `utils.py`: `run_speed_benchmark()`, `run_memory_benchmark()` |
| CLI (overwrite, model choice) | `utils.py`: `parse_benchmark_script_args()` (includes `--model`) |
| Running the grid and writing CSV | `utils.py`: `run_benchmarks()` |

## 3. Script structure (three parts)

### 3.1 Setup factory

Define a single **setup function** that builds inputs and the layer (or callable) from `SingleBenchmarkRunInput`, so both speed and memory benchmarks reuse the same setup.

- **Signature**: `_setup_<kernel>(input: SingleBenchmarkRunInput) -> (tensors, layer_or_fn)`
- **Input**: `input.x` is the varying dimension (e.g. sequence length); `input.extra_benchmark_config` holds `bsz`, `hidden_size`, `dtype`, etc.; `input.kernel_provider` identifies the implementation variant (e.g. `"liger"`, `"huggingface"`, `"torch"`; values are kernel-specific).
- **Return**: Whatever the benchmark helpers need (e.g. `(x, layer)` for a single-tensor forward like GEGLU).

Example (conceptually):

```python
def _setup_geglu(input: SingleBenchmarkRunInput):
    cfg = input.extra_benchmark_config
    # Build config, create x tensor, instantiate LigerGEGLUMLP or LlamaMLP by provider
    return x, layer
```

### 3.2 Speed and memory benchmark functions

Each takes `SingleBenchmarkRunInput` and returns `SingleBenchmarkRunOutput` by calling the shared helpers.

- **Speed**: `run_speed_benchmark(fwd_fn, mode, input_tensors, rep=...)`
- **Memory**: `run_memory_benchmark(fwd_fn, mode)`
- **Modes**: Use `["full", "forward", "backward"]` for both speed and memory for consistency.

Example:

```python
def bench_speed_geglu(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _setup_geglu(input)
    return run_speed_benchmark(lambda: layer(x), input.kernel_operation_mode, [x])

def bench_memory_geglu(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _setup_geglu(input)
    return run_memory_benchmark(lambda: layer(x), input.kernel_operation_mode)
```

For **scalar output** (e.g. loss) or **multiple outputs** (e.g. RoPE), use the appropriate helpers from `utils.py` if available (e.g. loss or multi-output variants), or implement custom measurement and still use the same setup factory and `run_benchmarks()`.

### 3.3 `__main__`: model config, shape computation, run

1. Parse args: `args = parse_benchmark_script_args()` and resolve `model = get_benchmark_model_config(args.model)`.
2. (Recommended) Measure peak memory with a small probe using the **highest-memory baseline** implementation (e.g. `"huggingface"` or `"torch"`):
   - Define a `_probe()` function that creates tensors/layers, runs a forward pass, and returns the output tensor. `_probe()` owns setup; `estimate_kernel_peak_memory` handles memory-stat reset before the call, runs `.backward()`, and performs cleanup (gc + cache clear) afterward.
   - Call `peak_bytes = estimate_kernel_peak_memory(probe_fn=_probe)`.
3. Compute sweep config (device memory is obtained internally by both helpers):
   - **Sequence-length sweep** (e.g. GEGLU, SwiGLU): convert peak bytes to per-token (`kernel_bpt = peak_bytes // probe_seq_len`), then `config = compute_seq_len_sweep_config(model, kernel_bytes_per_token=kernel_bpt)`. The returned `SeqLenSweepConfig` has `batch_size` and `seq_len`.
   - **Hidden-size sweep** (e.g. DyT): pass total peak bytes directly: `config = compute_hidden_size_sweep_config(model, kernel_peak_bytes=peak_bytes, bt=BT)`. The returned `HiddenSizeSweepConfig` has `bt` and `max_hidden_size`.
4. Build `x_values` from `config.seq_len` (seq_len sweep) or `config.max_hidden_size` (hidden_size sweep).
5. Build `extra_benchmark_configs` from `model` and config:
   - Seq_len sweep: e.g. `bsz=config.batch_size`, `hidden_size=model.hidden_size`, `dtype=model.dtype`.
   - Hidden_size sweep: e.g. `BT=config.bt`, `dtype=model.dtype`.
6. Call `run_benchmarks(..., kernel_operation_modes=["full", "forward", "backward"], ...)` for both speed and memory.

## 4. CLI

Scripts should support:

- `--overwrite`: overwrite existing rows in the benchmark CSV.
- `--model`: model profile name from `MODEL_REGISTRY` (e.g. `llama_2_7b`, `llama_3_8b`). Default when not set is `DEFAULT_MODEL_CONFIG` (e.g. `llama_3_8b`).

These are provided by `parse_benchmark_script_args()` in `utils.py`.

## 5. Reference scripts

- **Element-wise (single tensor in/out, seq_len sweep)**: `benchmark_geglu.py`, `benchmark_swiglu.py` — `compute_seq_len_sweep_config()`.
- **Element-wise (single tensor in/out, hidden_size sweep)**: `benchmark_dyt.py` — `compute_hidden_size_sweep_config()`.

## 6. Checklist for a new script

- [ ] Script under `benchmark/scripts/` named `benchmark_<kernel>.py`.
- [ ] Single `_setup_<kernel>(SingleBenchmarkRunInput)` used by both speed and memory.
- [ ] Speed/memory implemented via `run_speed_benchmark` / `run_memory_benchmark` (or the correct variant for loss / multi-output).
- [ ] `kernel_operation_modes=["full", "forward", "backward"]` for both speed and memory.
- [ ] No hardcoded batch size or sequence length; use `compute_seq_len_sweep_config()` or `compute_hidden_size_sweep_config()` (and optionally `estimate_kernel_peak_memory()`).
- [ ] Model dimensions and dtype from `ModelConfig` / `get_benchmark_model_config()` / `args.model`.
- [ ] CLI via `parse_benchmark_script_args()` (so `--model` and `--overwrite` work).
- [ ] Results written through `run_benchmarks()` so data goes to the shared CSV.
