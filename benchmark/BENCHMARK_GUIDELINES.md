# Guideline for Adding Benchmark Scripts

## 1. Where to add a script

- **Location**: `benchmark/scripts/`
- **Naming**: `benchmark_<kernel_name>.py` (e.g. `benchmark_geglu.py`, `benchmark_dyt.py`)

## 2. Shared infrastructure

Do **not** hardcode batch size, sequence length, or model dimensions. All benchmark scripts share the following:

| Need | Use |
|------|-----|
| Model dimensions (hidden_size, vocab_size, etc.) | `benchmark_model_configs.py`: `ModelConfig`, `MODEL_REGISTRY`, `get_benchmark_model_config()` |
| Memory probing | `benchmark_model_configs.py`: `estimate_kernel_peak_memory()` |
| Safe sweep configs | `compute_seq_len_sweep_config()`, `compute_hidden_size_sweep_config()`, `compute_model_config_sweep_config()` |
| Speed / memory measurement | `utils.py`: `run_speed_benchmark()`, `run_memory_benchmark()` |
| Running the grid and writing CSV | `utils.py`: `run_benchmarks()` |
| CLI arguments | `utils.py`: `parse_benchmark_script_args()` — provides `--model`, `--overwrite`, `--sweep-mode`, `--bt` |

### 2.1 Setup factory

Define a single **setup function** that builds inputs and the layer from `SingleBenchmarkRunInput`, so both speed and memory benchmarks reuse the same setup.

- **Signature**: `_setup_<kernel>(input: SingleBenchmarkRunInput) -> (tensors, layer_or_fn)`
- **Input**: `input.x` is the varying dimension (e.g. seq_len or hidden_size); `input.extra_benchmark_config` holds fixed params like `bsz`, `hidden_size`, `dtype`; `input.kernel_provider` identifies the implementation variant (`"liger"`, `"huggingface"`, `"torch"`, etc.).

```python
def _setup_geglu(input: SingleBenchmarkRunInput):
    cfg = input.extra_benchmark_config
    # Build model config, create x tensor, instantiate layer by provider
    return x, layer
```

### 2.2 Speed and memory benchmark functions

Each takes `SingleBenchmarkRunInput` and returns `SingleBenchmarkRunOutput`:

```python
def bench_speed_geglu(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _setup_geglu(input)
    return run_speed_benchmark(lambda: layer(x), input.kernel_operation_mode, [x])

def bench_memory_geglu(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _setup_geglu(input)
    return run_memory_benchmark(lambda: layer(x), input.kernel_operation_mode)
```

- Use `kernel_operation_modes=["full", "forward", "backward"]` for both speed and memory.
- For **scalar output** (e.g. loss) or **multiple outputs** (e.g. RoPE), implement custom measurement logic but still use the same setup factory and `run_benchmarks()`.

### 2.3 Memory probing

Most scripts should probe peak memory before computing sweep configs:

1. Define a `_probe()` that creates tensors/layers at a small scale and returns the output tensor.
2. Call `peak_bytes = estimate_kernel_peak_memory(probe_fn=_probe)`.
3. Use `peak_bytes` to derive safe sweep parameters (see sections 3 and 4).

Use the **highest-memory baseline** implementation for probing (e.g. `"huggingface"` or `"torch"`) to get a safe upper bound.

## 3. D1 — Non-model dimension sweep

Sweep non-model dimensions (e.g. sequence length, BT) with a **fixed model config**. Use `--model` to select which model.

### 3.1 How to implement

In `__main__`, the `token_length` sweep mode (default) follows this pattern:

1. Parse args and resolve model: `args = parse_benchmark_script_args()`, `model = get_benchmark_model_config(args.model)`.
2. Probe and compute sweep config:
   - **seq_len sweep** (GEGLU, SwiGLU, etc.): `kernel_bpt = peak_bytes // probe_seq_len`, then `config = compute_seq_len_sweep_config(model, kernel_bytes_per_token=kernel_bpt)`. Returns `SeqLenSweepConfig` with `batch_size` and `seq_len`.
   - **BT sweep** (other ops): use `BT` directly as a fixed dimension if no sweep is needed.
3. Build `x_values` from `config.seq_len` (e.g. `[2**i for i in range(10, log2(config.seq_len) + 1)]`).
4. Build `extra_benchmark_configs` with fixed model dimensions: `bsz=config.batch_size`, `hidden_size=model.hidden_size`, `dtype=model.dtype`, etc.
5. Call `run_benchmarks(...)` for both speed and memory.

### 3.2 How to run

```bash
# Default model (llama_3_8b)
python benchmark_geglu.py

# Specific model
python benchmark_geglu.py --model llama_2_7b

# Overwrite existing CSV entries
python benchmark_geglu.py --model llama_3_8b --overwrite
```

### 3.3 Reference scripts

- **seq_len sweep**: `benchmark_geglu.py`, `benchmark_swiglu.py` — `compute_seq_len_sweep_config()`

## 4. D2 — Model dimension sweep

Sweep model-related dimensions (e.g. hidden_size, or discrete model configs from `MODEL_REGISTRY`) with a **fixed token count**. Use `--bt` to set the token count.

D2 has two variants:

### 4.1 Continuous sweep (e.g. hidden_size)

Sweep a single model parameter (like hidden_size) in a continuous range with fixed BT.

**How to implement:**

1. Probe: measure peak memory at `(BT, model.hidden_size)`.
2. `config = compute_hidden_size_sweep_config(model, kernel_peak_bytes=peak_bytes, bt=BT)`. Returns `HiddenSizeSweepConfig` with `bt` and `max_hidden_size`.
3. Build `x_values` from `config.max_hidden_size` (e.g. `[1024 * i for i in range(1, 17) if 1024 * i <= config.max_hidden_size]`).
4. Build `extra_benchmark_configs` with `BT=config.bt`, `dtype=model.dtype`, etc.
5. Call `run_benchmarks(...)`.

**Reference**: `benchmark_dyt.py` — hidden_size sweep with `compute_hidden_size_sweep_config()`.

### 4.2 Discrete model-config sweep

Sweep across all `MODEL_REGISTRY` entries as discrete data points. Activated by `--sweep-mode model_config`.

**How to implement:**

1. Add a `_resolve_model_config_<kernel>` helper that maps `input.x` (model index) to a standard `SingleBenchmarkRunInput`:

```python
def _resolve_model_config_geglu(input: SingleBenchmarkRunInput):
    """Resolve model-config-sweep input into standard setup args."""
    cfg = input.extra_benchmark_config
    model_info = cfg["model_configs"][int(input.x)]
    return _setup_geglu(SingleBenchmarkRunInput(
        x=cfg["seq_len"],
        kernel_provider=input.kernel_provider,
        extra_benchmark_config={
            "bsz": cfg["bsz"],
            "hidden_size": model_info["hidden_size"],
            "intermediate_size": model_info["intermediate_size"],
            "hidden_act": cfg["hidden_act"],
            "dtype": model_info["dtype"],
        },
    ))
```

2. Add `bench_speed_<kernel>_model_config` and `bench_memory_<kernel>_model_config`:

```python
def bench_speed_geglu_model_config(input):
    x, layer = _resolve_model_config_geglu(input)
    return run_speed_benchmark(lambda: layer(x), input.kernel_operation_mode, [x])
```

3. In `__main__`, gate on `args.sweep_mode == "model_config"`:
   - Build `_probe_factory(model_cfg, probe_seq_len)` that returns a probe callable.
   - Call `sweep = compute_model_config_sweep_config(all_model_configs, probe_fn_factory=..., bt=args.bt)`.
   - Build `model_configs_info` (list of dicts with each model's dimensions) and pass in `extra_benchmark_configs`.
   - `x_values = list(range(len(sweep.model_configs)))` (model indices).
   - Call `run_benchmarks(bench_test_fn=bench_speed_<kernel>_model_config, ...)`.

**Reference**: `benchmark_geglu.py`, `benchmark_swiglu.py`, `benchmark_dyt.py` — all support `--sweep-mode model_config`.

### 4.3 How to run

```bash
# Discrete model-config sweep with default bt=2048
python benchmark_geglu.py --sweep-mode model_config

# With custom bt
python benchmark_geglu.py --sweep-mode model_config --bt 4096
```

## 5. Checklist

- [ ] Script under `benchmark/scripts/` named `benchmark_<kernel>.py`.
- [ ] Single `_setup_<kernel>(SingleBenchmarkRunInput)` used by both speed and memory.
- [ ] Speed/memory via `run_speed_benchmark` / `run_memory_benchmark` (or custom variant for loss/multi-output).
- [ ] `kernel_operation_modes=["full", "forward", "backward"]` for both speed and memory.
- [ ] No hardcoded batch size or sequence length; sweep configs from `compute_*_sweep_config()` + `estimate_kernel_peak_memory()`.
- [ ] Model dimensions and dtype from `ModelConfig` / `get_benchmark_model_config()` / `args.model`.
- [ ] CLI via `parse_benchmark_script_args()` (so `--model`, `--overwrite`, `--sweep-mode`, `--bt` all work).
- [ ] Results written through `run_benchmarks()` to the shared CSV.
- [ ] Model-config sweep: `_resolve_model_config_<kernel>`, `bench_speed_<kernel>_model_config`, `bench_memory_<kernel>_model_config`, and `__main__` model-config code path.
