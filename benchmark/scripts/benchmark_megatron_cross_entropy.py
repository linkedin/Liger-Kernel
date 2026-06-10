"""Benchmark Liger's Megatron-LM cross-entropy wrapper.

Compares four providers on the per-token CE call shape ``[seq, batch, vocab]``:

  - **torch**: vanilla ``F.cross_entropy``
  - **megatron**: Megatron's *fused* ``fused_vocab_parallel_cross_entropy`` path
    (``cross_entropy_loss_fusion=True``, JIT-fused via TorchScript)
  - **megatron-unfused**: Megatron's *unfused* ``vocab_parallel_cross_entropy``
    path (``cross_entropy_loss_fusion=False``, eager Python; the path users on
    ``label_smoothing`` typically end up on)
  - **liger**: ``LigerMegatronCrossEntropy`` — Liger's Triton CE wrapped in the
    Megatron fused signature. Same kernel regardless of which Megatron symbol
    it was patched onto, so we only benchmark it once.

Requires a Liger-supported accelerator (CUDA / ROCm). With megatron-core not
installed, both megatron providers are silently skipped.

Output goes to the shared ``benchmark/data/all_benchmark_data.csv`` like every
other Liger benchmark — rows are tagged with ``kernel_name="megatron_cross_entropy"``
and the standard visualizer renders them via:

    python benchmark/benchmarks_visualizer.py \\
        --kernel-name megatron_cross_entropy --metric-name speed
    python benchmark/benchmarks_visualizer.py \\
        --kernel-name megatron_cross_entropy --metric-name memory
"""

import os

import torch
import torch.nn.functional as F
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.megatron import LigerMegatronCrossEntropy
from liger_kernel.utils import infer_device

device = infer_device()

try:
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
    from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

    _MEGATRON_AVAILABLE = True
except ImportError:
    fused_vocab_parallel_cross_entropy = None
    vocab_parallel_cross_entropy = None
    _MEGATRON_AVAILABLE = False


def _make_inputs(s: int, b: int, v: int, requires_grad: bool = True):
    logits = torch.randn(s, b, v, device=device, dtype=torch.bfloat16, requires_grad=requires_grad)
    target = torch.randint(0, v, (s, b), device=device, dtype=torch.long)
    return logits, target


def _pytorch_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    s, b, v = logits.shape
    return F.cross_entropy(
        logits.reshape(-1, v).float(),
        target.reshape(-1),
        reduction="none",
    ).reshape(s, b)


def _ensure_single_rank_tp_group():
    """Initialize torch.distributed (single-rank) and return a usable TP group.

    For a single-process benchmark we use the world group of size 1; the
    internal all-reduce becomes a no-op.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(backend="nccl")
    return dist.group.WORLD


def _select_fwd(provider: str):
    if provider == "liger":
        ce = LigerMegatronCrossEntropy(reduction="none")
        return lambda logits, target: ce(logits, target)
    if provider == "torch":
        return _pytorch_cross_entropy
    if provider == "megatron":
        if not _MEGATRON_AVAILABLE:
            raise RuntimeError("megatron-core not installed; cannot benchmark 'megatron' provider")
        tp_group = _ensure_single_rank_tp_group()

        def _megatron_fused_call(logits, target):
            return fused_vocab_parallel_cross_entropy(logits, target, tp_group)

        return _megatron_fused_call
    if provider == "megatron-unfused":
        if not _MEGATRON_AVAILABLE:
            raise RuntimeError(
                "megatron-core not installed; cannot benchmark 'megatron-unfused' provider"
            )
        tp_group = _ensure_single_rank_tp_group()

        def _megatron_unfused_call(logits, target):
            # Unfused signature: (logits, target, label_smoothing=0.0, tp_group=None)
            return vocab_parallel_cross_entropy(logits, target, 0.0, tp_group)

        return _megatron_unfused_call
    raise ValueError(f"unknown provider: {provider!r}")


def bench_speed_megatron_cross_entropy(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    v = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    s = input.extra_benchmark_config["S"]
    b = input.extra_benchmark_config["B"]

    logits, target = _make_inputs(s, b, v)
    fwd_fn = _select_fwd(provider)

    def fwd():
        return fwd_fn(logits, target)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, rep=100, quantiles=QUANTILES)
    elif mode == "backward":
        # Megatron's fused CE writes gradients in-place into saved tensors during backward,
        # which breaks the standard retain_graph=True / repeated-backward pattern do_bench
        # uses elsewhere. Run a fresh fwd+bwd each iteration so each backward sees an
        # unmodified autograd graph. Measurement therefore includes forward time —
        # subtract the "forward" measurement to derive backward-only timing.
        def _fwd_bwd():
            if logits.grad is not None:
                logits.grad = None
            out = fwd_fn(logits, target)
            out.sum().backward()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(_fwd_bwd, rep=100, quantiles=QUANTILES)
    elif mode == "full":

        def full():
            y = fwd()
            y.sum().backward()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, rep=100, quantiles=QUANTILES)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_megatron_cross_entropy(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    v = input.x
    provider = input.kernel_provider
    s = input.extra_benchmark_config["S"]
    b = input.extra_benchmark_config["B"]

    logits, target = _make_inputs(s, b, v)
    fwd_fn = _select_fwd(provider)

    def full():
        y = fwd_fn(logits, target)
        y.sum().backward()

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    providers = ["liger", "torch"]
    if _MEGATRON_AVAILABLE:
        providers.append("megatron")
        providers.append("megatron-unfused")

    common_configs = {
        "kernel_name": "megatron_cross_entropy",
        "x_name": "V",
        "x_label": "vocab size",
        "x_values": [2**i for i in range(12, 18)],
        "kernel_providers": providers,
        "extra_benchmark_configs": [{"S": 2048, "B": 4}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_megatron_cross_entropy,
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_megatron_cross_entropy,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
