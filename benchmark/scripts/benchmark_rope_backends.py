"""RoPE benchmark comparing all three Liger backends in one run.

Sibling of ``benchmark_rope.py``. The original compares a single ``liger`` provider
(whichever backend ``LIGER_KERNEL_IMPL`` selects) against HuggingFace. This version
imports the Triton, cuTile and CuTe DSL RoPE autograd Functions *directly* -- like
the unit tests do -- so all three land in one CSV in a single process, without
juggling ``LIGER_KERNEL_IMPL`` across runs.

Providers emitted: ``huggingface``, ``triton``, ``cutile``, ``cutedsl`` (a backend
is skipped automatically if its optional SDK is not importable).

Timing modes (``--timing``), always in **bf16**:

* ``eager`` -- ``triton.testing.do_bench`` around one call (metric ``speed``). This is
  the realistic *host-inclusive* per-call cost. RoPE is a tiny, memory-bound op, so a
  single eager launch is dominated by Python + kernel-launch overhead (for the DSL
  backends, DLPack marshalling). In eager the three backends are often indistinguishable
  because we are measuring launch overhead, not the kernel.
* ``graph`` -- CUDA-graph capture + replay (metric ``speed_graph``). Amortises host
  launch to ~0 and exposes the *device-only* kernel time -- the fair kernel-quality
  number, and what a graph-captured / GPU-bound trainer actually sees. Also reports
  memory (eager only).

Sweep dimensions (``--sweep-dim``):

* ``head_dim`` (default) -- x-axis head_dim in {64, 96, 128, 256}, one extra config per
  sequence length. head_dim selects each kernel's internal path (cuTile ALIGNED 4-D vs
  gather/scatter; CuTe DSL TMA vs token).
* ``seq`` -- x-axis sequence length in {1024, 2048, 4096, 8192}, one extra config per
  head_dim.

Usage:
    LIGER_BENCH_TARGET=rope_headdim python benchmark_rope_backends.py --overwrite
    LIGER_BENCH_TARGET=rope_headdim python benchmark_rope_backends.py --timing graph --overwrite
    LIGER_BENCH_TARGET=rope_seq     python benchmark_rope_backends.py --sweep-dim seq --overwrite

``LIGER_BENCH_TARGET`` routes output to benchmark/data/all_benchmark_data_<target>.csv.
"""

import argparse
import os
import sys

import torch

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import build_memory_bench_fn
from utils import build_speed_bench_fn
from utils import run_benchmarks

from liger_kernel.utils import infer_device
from liger_kernel.utils import transformers_version_dispatch

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# Import each backend's autograd Function directly. Importing the kernel modules
# (not liger_kernel.ops) sidesteps the LIGER_KERNEL_IMPL replacement machinery, so
# Triton / cuTile / CuTe DSL can be compared side by side in ONE process. Optional
# backends are skipped if their SDK (cuda-tile / nvidia-cutlass-dsl) is missing.
from liger_kernel.ops.rope import LigerRopeFunction as _TritonRope  # noqa: E402

PROVIDER_FNS = {"triton": _TritonRope}

try:
    from liger_kernel.ops.cutile.ops.rope import LigerRopeFunction as _CuTileRope

    PROVIDER_FNS["cutile"] = _CuTileRope
except Exception as e:  # pragma: no cover - optional backend
    print(f"[benchmark_rope_backends] cuTile backend unavailable, skipping: {e}")

try:
    from liger_kernel.ops.cutedsl.ops.rope import LigerRopeCuteDSLFunction as _CuTeDSLRope

    PROVIDER_FNS["cutedsl"] = _CuTeDSLRope
except Exception as e:  # pragma: no cover - optional backend
    print(f"[benchmark_rope_backends] CuTe DSL backend unavailable, skipping: {e}")


# Sweep space (bf16 only).
HEAD_DIMS = [64, 96, 128, 256]
SEQ_LENS = [1024, 2048, 4096, 8192]
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8

# Production RoPE shapes (num_attention_heads, num_key_value_heads, head_dim) for a
# seq-length sweep. Deliberately mixes shapes that hit CuTile's ALIGNED fast path
# (pow2 head counts) with ones that fall to its gather/scatter path (non-pow2 head
# counts) -- both occur in mainstream models. CuTeDSL uses its TMA path for all
# (head_dim=128 is 128-bit vectorizable).
PRODUCTION_MODELS = [
    {"model": "qwen3_30b_a3b", "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128},
    {"model": "mixtral_8x7b", "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128},
    {"model": "mixtral_8x22b", "num_attention_heads": 48, "num_key_value_heads": 8, "head_dim": 128},
]


def _resolve_dims(input: SingleBenchmarkRunInput):
    """Return (seq_len, head_dim, num_q, num_kv, dtype) from the swept x and config."""
    cfg = input.extra_benchmark_config
    num_q = cfg["num_attention_heads"]
    num_kv = cfg["num_key_value_heads"]
    dtype = cfg["dtype"]
    if "seq_len" in cfg:  # head_dim sweep: x is head_dim
        seq_len, head_dim = cfg["seq_len"], input.x
    else:  # seq sweep: x is seq_len
        seq_len, head_dim = input.x, cfg["head_dim"]
    return seq_len, head_dim, num_q, num_kv, dtype


def _build_qk(seq_len, head_dim, num_q, num_kv, dtype, requires_grad):
    """q/k in projection layout, returned as (bsz, n_head, seq, hd) transpose-views."""
    q = torch.randn((1, seq_len, num_q, head_dim), device=device, requires_grad=requires_grad, dtype=dtype).transpose(
        1, 2
    )
    k = torch.randn((1, seq_len, num_kv, head_dim), device=device, requires_grad=requires_grad, dtype=dtype).transpose(
        1, 2
    )
    return q, k


def _build_cos_sin(seq_len, head_dim, num_kv, k):
    rotary_emb = transformers_version_dispatch(
        "4.48.0",
        LlamaRotaryEmbedding,
        LlamaRotaryEmbedding,
        before_kwargs={"dim": head_dim, "device": device},
        after_kwargs={"config": LlamaConfig(num_kv_heads=num_kv, head_dim=head_dim), "device": device},
    )
    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    return rotary_emb(k, pos_ids)


def setup_rope(input: SingleBenchmarkRunInput):
    """Eager setup: build inputs and return (q, fwd_closure) for the do_bench harness."""
    seq_len, head_dim, num_q, num_kv, dtype = _resolve_dims(input)
    q, k = _build_qk(seq_len, head_dim, num_q, num_kv, dtype, requires_grad=True)
    cos, sin = _build_cos_sin(seq_len, head_dim, num_kv, k)

    provider = input.kernel_provider
    if provider == "huggingface":
        fwd_fn = lambda: apply_rotary_pos_emb(q, k, cos, sin)
    elif provider in PROVIDER_FNS:
        rope_fn = PROVIDER_FNS[provider]
        fwd_fn = lambda: rope_fn.apply(q, k, cos, sin)
    else:
        raise ValueError(f"Invalid provider: {provider} for RoPE embedding")

    return q, lambda _: fwd_fn()[0]


# ---------------------------------------------------------------------------
# CUDA-graph (device-only) timing
# ---------------------------------------------------------------------------
def _percentiles(times):
    a = sorted(times)
    n = len(a)

    def p(q):
        return a[min(n - 1, max(0, int(round(q * (n - 1)))))]

    return p(0.2), p(0.5), p(0.8)


def _time_cuda_graph(fn, iters: int = 50, replays: int = 30):
    """Capture ``iters`` back-to-back calls into a CUDA graph, replay, return (p20,p50,p80) ms.

    Warm up on a side stream first (JIT compile / populate caches), then capture. Timing a
    single Python launch would leave the GPU idle while the host marshals args, charging that
    host cost to the kernel; capturing many calls and replaying amortises host launch to ~0
    and measures device time only.
    """
    with torch.no_grad():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn()
        torch.cuda.synchronize()

        per_call = []
        for _ in range(replays):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            g.replay()
            end.record()
            torch.cuda.synchronize()
            per_call.append(start.elapsed_time(end) / iters)
    return _percentiles(per_call)


def _make_backward_closure(provider, q, k, cos, sin, seq_len, head_dim, num_q, num_kv, dtype):
    """Functional device-only backward closure (SDPA-representative transpose-view grad).

    Autograd's ``.grad`` cannot be captured in a CUDA graph, so we time each backend's
    *functional* ``rope_backward`` directly. HuggingFace has no standalone backward kernel
    -> returns None.
    """
    dq = torch.randn((1, seq_len, num_q, head_dim), device=device, dtype=dtype).transpose(1, 2)
    dk = torch.randn((1, seq_len, num_kv, head_dim), device=device, dtype=dtype).transpose(1, 2)
    if provider == "triton":
        from liger_kernel.ops.rope import rope_backward

        return lambda: rope_backward(dq, dk, cos, sin)
    if provider == "cutedsl":
        from liger_kernel.ops.cutedsl.ops.rope import rope_backward

        return lambda: rope_backward(dq, dk, cos, sin)
    if provider == "cutile":
        from liger_kernel.ops.cutile.ops.rope import rope_backward
        from liger_kernel.ops.cutile.ops.rope import rope_forward

        _, _, scos, ssin, cos_bs, aligned, tqh, tkh, thd, odt = rope_forward(q.clone(), k.clone(), cos, sin)
        return lambda: rope_backward(
            dq, dk, scos, ssin, cos_bs, aligned, tqh, tkh, thd, odt, 1, seq_len, num_q, num_kv, head_dim
        )
    return None


def build_graph_bench_fn():
    """Bench fn for run_benchmarks that measures device-only time via CUDA graphs."""

    def bench(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
        seq_len, head_dim, num_q, num_kv, dtype = _resolve_dims(input)
        provider = input.kernel_provider
        mode = input.kernel_operation_mode

        if mode == "forward":
            q, k = _build_qk(seq_len, head_dim, num_q, num_kv, dtype, requires_grad=False)
            cos, sin = _build_cos_sin(seq_len, head_dim, num_kv, k)
            if provider == "huggingface":
                fn = lambda: apply_rotary_pos_emb(q, k, cos, sin)
            else:
                rope_fn = PROVIDER_FNS[provider]
                fn = lambda: rope_fn.apply(q, k, cos, sin)
            p20, p50, p80 = _time_cuda_graph(fn)
        elif mode == "backward":
            q, k = _build_qk(seq_len, head_dim, num_q, num_kv, dtype, requires_grad=False)
            cos, sin = _build_cos_sin(seq_len, head_dim, num_kv, k)
            bwd = _make_backward_closure(provider, q, k, cos, sin, seq_len, head_dim, num_q, num_kv, dtype)
            if bwd is None:
                return SingleBenchmarkRunOutput(y_20=float("nan"), y_50=float("nan"), y_80=float("nan"))
            p20, p50, p80 = _time_cuda_graph(bwd)
        elif mode == "full":
            # Device full-pass = forward kernel + functional backward kernel chained in one
            # graph. HF has no capturable backward (autograd) -> N/A.
            q, k = _build_qk(seq_len, head_dim, num_q, num_kv, dtype, requires_grad=False)
            cos, sin = _build_cos_sin(seq_len, head_dim, num_kv, k)
            bwd = _make_backward_closure(provider, q, k, cos, sin, seq_len, head_dim, num_q, num_kv, dtype)
            if provider == "huggingface" or bwd is None:
                return SingleBenchmarkRunOutput(y_20=float("nan"), y_50=float("nan"), y_80=float("nan"))
            rope_fn = PROVIDER_FNS[provider]
            fwd = lambda: rope_fn.apply(q, k, cos, sin)

            def full():
                fwd()
                bwd()

            p20, p50, p80 = _time_cuda_graph(full)
        else:
            raise ValueError(f"graph timing supports forward/backward/full, got {mode}")

        return SingleBenchmarkRunOutput(y_20=p20, y_50=p50, y_80=p80)

    return bench


def parse_args():
    parser = argparse.ArgumentParser(description="RoPE 3-backend comparison benchmark (bf16)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing rows in the CSV.")
    parser.add_argument(
        "--sweep-dim",
        choices=["head_dim", "seq", "models"],
        default="head_dim",
        help="x-axis to sweep. 'head_dim' fixes seq_len per extra config; 'seq' fixes head_dim; "
        "'models' sweeps seq_len for each production model shape (Qwen3-30B-A3B, Mixtral 8x7B/8x22B).",
    )
    parser.add_argument(
        "--timing",
        choices=["eager", "graph", "both"],
        default="both",
        help="'eager' = host-inclusive do_bench (metric speed + memory); "
        "'graph' = device-only CUDA-graph replay (metric speed_graph); 'both' (default).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.sweep_dim == "head_dim":
        x_name, x_label, x_values = "D", "head dim", HEAD_DIMS
        extra_benchmark_configs = [
            {
                "num_attention_heads": NUM_Q_HEADS,
                "num_key_value_heads": NUM_KV_HEADS,
                "dtype": torch.bfloat16,
                "seq_len": s,
            }
            for s in (2048, 8192)
        ]
    elif args.sweep_dim == "models":
        # x-axis = sequence length; one extra config per production model shape.
        x_name, x_label, x_values = "T", "sequence length", SEQ_LENS
        extra_benchmark_configs = [{**m, "dtype": torch.bfloat16} for m in PRODUCTION_MODELS]
    else:
        x_name, x_label, x_values = "T", "sequence length", SEQ_LENS
        extra_benchmark_configs = [
            {
                "num_attention_heads": NUM_Q_HEADS,
                "num_key_value_heads": NUM_KV_HEADS,
                "dtype": torch.bfloat16,
                "head_dim": d,
            }
            for d in (64, 128)
        ]

    all_providers = ["huggingface"] + list(PROVIDER_FNS.keys())
    base = {
        "kernel_name": "rope",
        "x_name": x_name,
        "x_label": x_label,
        "x_values": x_values,
        "extra_benchmark_configs": extra_benchmark_configs,
        "overwrite": args.overwrite,
    }

    if args.timing in ("eager", "both"):
        run_benchmarks(
            bench_test_fn=build_speed_bench_fn(setup_rope),
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            kernel_providers=all_providers,
            **base,
        )
        run_benchmarks(
            bench_test_fn=build_memory_bench_fn(setup_rope),
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            kernel_providers=all_providers,
            **base,
        )

    if args.timing in ("graph", "both"):
        graph_bench = build_graph_bench_fn()
        # Forward: all providers (HF forward is a capturable sequence of torch ops).
        run_benchmarks(
            bench_test_fn=graph_bench,
            kernel_operation_modes=["forward"],
            metric_name="speed_graph",
            metric_unit="ms",
            kernel_providers=all_providers,
            **base,
        )
        # Backward + full: Liger backends only (HF has no capturable functional backward).
        run_benchmarks(
            bench_test_fn=graph_bench,
            kernel_operation_modes=["backward", "full"],
            metric_name="speed_graph",
            metric_unit="ms",
            kernel_providers=list(PROVIDER_FNS.keys()),
            **base,
        )
