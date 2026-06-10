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

Output:
  - CSV: ``benchmark/data/all_benchmark_data_megatron.csv``
    (separate per-component file, mirroring the recent ``all_benchmark_data_cutile.csv``
    precedent — keeps the PR diff scannable)
  - Plots (best-effort): ``benchmark/visualizations/megatron_cross_entropy_*.png``
    rendered when matplotlib is available; skipped silently otherwise.
"""

import functools
import os

import torch
import torch.nn.functional as F
import triton

import utils as benchmark_utils
from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

_CSV_FILENAME = "all_benchmark_data_megatron.csv"

# Redirect CSV output to a per-component file (parallel to all_benchmark_data_cutile.csv).
# We can't reuse the LIGER_KERNEL_IMPL knob because it also drives kernel-backend
# selection in liger_kernel.ops — overloading it would force us onto a megatron-named
# backend that doesn't exist. Patching update_benchmark_data_csv is surgical and only
# affects this benchmark process.
_original_update_benchmark_data_csv = benchmark_utils.update_benchmark_data_csv


@functools.wraps(_original_update_benchmark_data_csv)
def _patched_update_benchmark_data_csv(*args, **kwargs):
    kwargs["filename"] = _CSV_FILENAME
    return _original_update_benchmark_data_csv(*args, **kwargs)


benchmark_utils.update_benchmark_data_csv = _patched_update_benchmark_data_csv

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

try:
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt  # noqa: E402

    _HAVE_MATPLOTLIB = True
except ImportError:
    plt = None
    _HAVE_MATPLOTLIB = False


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


# ---------------------------------------------------------------------------
# Plot generation (best-effort).
# ---------------------------------------------------------------------------


def _generate_plots(out_dir: str) -> None:
    """Generate one PNG per (metric, mode) combination from the CSV we just wrote.

    Silently skipped when matplotlib is unavailable. Reads the CSV rather than
    re-running benchmarks so the plots use the same numbers that landed on disk.
    """
    if not _HAVE_MATPLOTLIB:
        print("[plots] matplotlib not available; skipping plot generation.")
        return

    import csv
    from pathlib import Path

    csv_path = Path(os.path.join(os.path.dirname(__file__), "..", "data", "all_benchmark_data_megatron.csv"))
    if not csv_path.exists():
        print(f"[plots] CSV not found at {csv_path}; skipping plot generation.")
        return

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # CSV layout is denormalized: one row per (provider, mode, metric, x_value).
    # We need to aggregate rows back into (provider → x_values, y50_list, y20_list, y80_list)
    # series before plotting.
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("kernel_name") == "megatron_cross_entropy":
                rows.append(row)

    if not rows:
        print("[plots] no megatron_cross_entropy rows in the CSV; skipping plots.")
        return

    series: dict = {}  # (metric_name, mode, provider) → {xs, y50, y20, y80, meta}
    for row in rows:
        key = (row["metric_name"], row["kernel_operation_mode"], row["kernel_provider"])
        entry = series.setdefault(
            key,
            {
                "xs": [], "y50": [], "y20": [], "y80": [],
                "x_label": row.get("x_label", "x"),
                "metric_unit": row.get("metric_unit", ""),
            },
        )
        try:
            entry["xs"].append(float(row["x_value"]))
            entry["y50"].append(float(row["y_value_50"]))
            entry["y20"].append(float(row["y_value_20"]))
            entry["y80"].append(float(row["y_value_80"]))
        except (KeyError, ValueError):
            # If the CSV schema differs from what we expect, surface it loudly but skip
            # plotting rather than crashing the whole benchmark.
            print(f"[plots] WARNING: skipping malformed row: {row}")
            continue

    if not series:
        print("[plots] no usable series in the CSV; skipping plots.")
        return

    # Group series by (metric_name, mode) so each plot collects all providers for that slice.
    plots: dict = {}
    for (metric_name, mode, provider), entry in series.items():
        plots.setdefault((metric_name, mode), []).append((provider, entry))

    plot_paths = []
    for (metric_name, mode), provider_entries in plots.items():
        mode_label = mode if mode not in (None, "", "None") else "full"
        fig, ax = plt.subplots(figsize=(8, 5))
        x_label = "x"
        metric_unit = ""
        for provider, entry in sorted(provider_entries, key=lambda pe: pe[0]):
            # Sort points by x so the line plot is monotone.
            order = sorted(range(len(entry["xs"])), key=lambda i: entry["xs"][i])
            xs = [entry["xs"][i] for i in order]
            y50 = [entry["y50"][i] for i in order]
            y20 = [entry["y20"][i] for i in order]
            y80 = [entry["y80"][i] for i in order]
            # Capture the line's color so fill_between's band uses the same hue across
            # matplotlib versions that don't share auto-cycles between the two calls.
            (line,) = ax.plot(xs, y50, marker="o", label=provider)
            ax.fill_between(xs, y20, y80, alpha=0.2, color=line.get_color())
            x_label = entry["x_label"]
            metric_unit = entry["metric_unit"]

        ax.set_xscale("log", base=2)
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"{metric_name} ({metric_unit})")
        ax.set_title(f"Megatron CE — {metric_name}, mode={mode_label}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        out_path = out_dir_path / f"megatron_cross_entropy_{metric_name}_{mode_label}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        plot_paths.append(str(out_path))

    print(f"[plots] wrote {len(plot_paths)} plot(s):")
    for p in plot_paths:
        print(f"  - {p}")


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

    _generate_plots(out_dir=os.path.join(os.path.dirname(__file__), "..", "visualizations"))
