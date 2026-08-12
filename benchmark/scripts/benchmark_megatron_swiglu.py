"""Benchmark Liger's Megatron-LM SwiGLU wrapper.

Compares four providers on the fused gate-up activation call shape
``[seq, batch, 2 * ffn_local]`` -- exactly the tensor Megatron's ``linear_fc1`` hands to
``bias_swiglu_impl``:

  - **liger**: ``LigerMegatronSwiGLU`` -- Liger's fused gate-up Triton kernel in the
    Megatron-shaped wrapper. This is the default configuration.
  - **liger_in_place**: the same, with ``in_place=True``, which writes the backward
    gradient into the fc1 output buffer rather than allocating a new one. Same speed,
    one fewer activation-sized allocation; opt-in because it destroys that buffer.
  - **megatron**: Megatron's ``bias_swiglu_impl`` (``bias_activation_fusion=True``), a
    chain of ``@jit_fuser`` TorchScript helpers. This is the symbol Liger displaces.
  - **torch**: eager ``F.silu(y_1) * y_2`` over ``torch.chunk(y, 2, -1)`` -- the
    unfused reference, and structurally what Megatron runs when
    ``bias_activation_fusion=False`` (there it's a closure inside ``MLP.forward``).

Why there is no ``--tp-size`` flag (unlike the Megatron CE benchmark): SwiGLU is
elementwise and token-local. It issues **no collectives**, and tensor parallelism affects
it only by shrinking the per-rank column count to ``ffn_hidden_size / tp``. Sweeping
``ffn_local`` on a single GPU therefore already covers every TP configuration -- TP=8 at
``ffn=28672`` is the same kernel work as the ``ffn_local=3584`` point on this curve. Peak
memory is likewise per-rank and scales as 1/TP.

The x-axis spans the Blackwell tiling threshold: ``liger_kernel.ops.swiglu`` switches to a
column-tiled 2D grid when ``next_pow2(n_cols) >= 16384`` on Blackwell, so the two largest
points exercise that path on B200 and the one-row path everywhere else.

Requires a Liger-supported accelerator (CUDA / ROCm). With megatron-core not installed the
``megatron`` provider is silently dropped and the run proceeds with ``liger`` + ``torch``.

Output goes to the shared ``benchmark/data/all_benchmark_data.csv`` -- rows are tagged with
``kernel_name="megatron_swiglu"`` and the standard visualizer renders them via:

    python benchmark/benchmarks_visualizer.py \\
        --kernel-name megatron_swiglu --metric-name speed
    python benchmark/benchmarks_visualizer.py \\
        --kernel-name megatron_swiglu --metric-name memory
"""

import torch
import torch.nn.functional as F
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.megatron import LigerMegatronSwiGLU
from liger_kernel.utils import infer_device

device = infer_device()

try:
    from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl

    _MEGATRON_AVAILABLE = True
except ImportError:
    bias_swiglu_impl = None
    _MEGATRON_AVAILABLE = False


def _torch_swiglu(y):
    """Eager reference — identical math to Megatron's ``swiglu``, minus the JIT fusion."""
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2


def _make_fwd(provider: str):
    if provider == "liger":
        module = LigerMegatronSwiGLU()
        return lambda y: module(y, None, False, False)
    if provider == "liger_in_place":
        module = LigerMegatronSwiGLU(in_place=True)
        return lambda y: module(y, None, False, False)
    if provider == "torch":
        return _torch_swiglu
    if provider == "megatron":
        if not _MEGATRON_AVAILABLE:
            raise RuntimeError("megatron-core not installed; cannot benchmark 'megatron' provider")
        return lambda y: bias_swiglu_impl(y, None, False, False)
    raise ValueError(f"unknown provider: {provider!r}")


def _make_input(s: int, b: int, ffn_local: int, requires_grad: bool = True) -> torch.Tensor:
    # 2 * ffn_local: Megatron's linear_fc1 emits gate and up concatenated on the last dim.
    return torch.randn(s, b, 2 * ffn_local, device=device, dtype=torch.bfloat16, requires_grad=requires_grad)


def bench_speed_megatron_swiglu(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    ffn_local = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    s = input.extra_benchmark_config["S"]
    b = input.extra_benchmark_config["B"]

    fwd_fn = _make_fwd(provider)
    x = _make_input(s, b, ffn_local)

    def fwd():
        return fwd_fn(x)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, rep=100, quantiles=QUANTILES)
    elif mode == "backward":
        # Rerun fwd each iteration: Liger's in-place backward consumes the saved buffers,
        # so a retained graph would corrupt on the second pass. Subtract the "forward"
        # row to get backward-only timing.
        def _fwd_bwd():
            if x.grad is not None:
                x.grad = None
            out = fwd()
            out.sum().backward()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(_fwd_bwd, rep=100, quantiles=QUANTILES)
    elif mode == "full":

        def full():
            if x.grad is not None:
                x.grad = None
            y = fwd()
            y.sum().backward()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, rep=100, quantiles=QUANTILES)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_megatron_swiglu(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    ffn_local = input.x
    provider = input.kernel_provider
    s = input.extra_benchmark_config["S"]
    b = input.extra_benchmark_config["B"]

    fwd_fn = _make_fwd(provider)
    x = _make_input(s, b, ffn_local)

    def full():
        if x.grad is not None:
            x.grad = None
        y = fwd_fn(x)
        y.sum().backward()

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    providers = ["liger", "liger_in_place", "torch"]
    if _MEGATRON_AVAILABLE:
        providers.append("megatron")

    common_configs = {
        "kernel_name": "megatron_swiglu",
        "x_name": "ffn_local",
        "x_label": "per-rank FFN hidden size",
        # 1024 → 32768. Llama-7B is 11008 and Llama-70B is 28672, so this brackets
        # production sizes; the top two points cross the Blackwell tiling threshold.
        "x_values": [2**i for i in range(10, 16)],
        "kernel_providers": providers,
        # Megatron's standard training shape, matching the megatron CE benchmark.
        "extra_benchmark_configs": [{"S": 2048, "B": 4}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_megatron_swiglu,
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_megatron_swiglu,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
