# Benchmark Template

## File: `benchmark/scripts/benchmark_{kernel}.py`

```python
import os
import sys

import torch

from benchmark_model_configs import compute_hidden_size_sweep_config
from benchmark_model_configs import estimate_kernel_peak_memory
from benchmark_model_configs import get_benchmark_model_config
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import parse_benchmark_script_args
from utils import run_benchmarks
from utils import run_memory_benchmark
from utils import run_speed_benchmark

from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def _setup_{kernel}(input: SingleBenchmarkRunInput):
    """Create input tensor and layer from benchmark config."""
    # Import both Liger and PyTorch reference implementations
    from test.transformers.test_{kernel} import Liger{Kernel}
    from test.transformers.test_{kernel} import Torch{Kernel}

    cfg = input.extra_benchmark_config
    hidden_size = input.x  # Or whatever the sweep variable is

    x = torch.randn(cfg["BT"], hidden_size, device=device, dtype=cfg["dtype"], requires_grad=True)

    if input.kernel_provider == "liger":
        layer = Liger{Kernel}(hidden_size=hidden_size).to(device)
    elif input.kernel_provider == "torch":
        layer = Torch{Kernel}(hidden_size=hidden_size).to(device)
    elif input.kernel_provider == "torch_compile":
        layer = torch.compile(Torch{Kernel}(hidden_size=hidden_size).to(device))
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for {kernel}")

    return x, layer


def bench_speed_{kernel}(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _setup_{kernel}(input)
    return run_speed_benchmark(lambda: layer(x), input.kernel_operation_mode, [x])


def bench_memory_{kernel}(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _setup_{kernel}(input)
    return run_memory_benchmark(lambda: layer(x), input.kernel_operation_mode)


BT = 4096  # Fixed batch*seq for sweeping hidden_size


if __name__ == "__main__":
    args = parse_benchmark_script_args()
    model = get_benchmark_model_config(args.model)

    def _probe():
        probe_input = SingleBenchmarkRunInput(
            x=model.hidden_size,
            kernel_provider="torch",
            extra_benchmark_config={"BT": BT, "dtype": model.dtype},
        )
        x, layer = _setup_{kernel}(probe_input)
        return layer(x)

    peak_bytes = estimate_kernel_peak_memory(probe_fn=_probe)
    sweep_config = compute_hidden_size_sweep_config(model, peak_bytes, bt=BT)
    x_values = [1024 * i for i in range(1, 17) if 1024 * i <= sweep_config.max_hidden_size] or [model.hidden_size]

    common_configs = {
        "kernel_name": "{kernel}",
        "x_name": "hidden_size",
        "x_label": "hidden_size",
        "x_values": x_values,
        "kernel_providers": ["liger", "torch", "torch_compile"],
        "extra_benchmark_configs": [{"BT": sweep_config.bt, "dtype": model.dtype}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_{kernel},
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_{kernel},
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
```

### Key Rules

1. **Import both implementations from test file** — avoids duplicating reference code
2. **Support "liger", "torch", and "torch_compile" providers**
3. **Use `run_speed_benchmark` and `run_memory_benchmark`** from utils — don't write custom measurement
4. **Use `get_benchmark_model_config`** and sweep utilities for standard model sizes
5. **Use `estimate_kernel_peak_memory`** to auto-compute safe x_values range
6. **Run both speed AND memory** for all three modes (forward, backward, full)
7. **Use `BT = 4096`** as default batch*seq when sweeping hidden_size
8. **Adapt sweep variable** — some kernels sweep `seq_len` or `vocab_size` instead of `hidden_size`. Use the appropriate `compute_*_sweep_config` utility.
