"""SwiGLU backend comparison — Triton vs cuTile vs CuTe DSL (+ unfused reference).

Sibling of ``benchmark_swiglu.py`` (which benchmarks the whole MLP module for a single
``liger`` provider). This one isolates the **Level-1 elementwise SwiGLU kernel**
``c = silu(a * gate) * b`` (a=gate, b=up) and compares all backends in one process:

  * ``huggingface`` -- the unfused reference ``F.silu(a) * b`` (two torch ops).
  * ``triton``      -- liger_kernel.ops.swiglu
  * ``cutile``      -- liger_kernel.ops.cutile.ops.swiglu
  * ``cutedsl``     -- liger_kernel.ops.cutedsl.ops.swiglu

Not covered: the fused-linear / fused-MLP variants (they fold in the gate/up GEMMs and
are a separate study).

Why custom timing (not the shared do_bench harness): the Liger SwiGLU backward writes
da/db **in place** into the saved input buffers (a memory optimization). Driving that
through autograd's ``.backward()`` on reused leaf tensors triggers an in-place-on-view
error. So we time the **functional** forward/backward closures directly -- identical
device work, no autograd bookkeeping -- for both eager and CUDA-graph modes. This also
makes eager and graph measure exactly the same closures, differing only in whether host
launch overhead is amortized.

Metrics written to the CSV:
  * ``speed``       -- eager median time (host-inclusive), modes forward/backward/full
  * ``speed_graph`` -- CUDA-graph device-only time, modes forward/backward/full
  * ``memory``      -- peak MB for a full (forward + backward) pass

Sweep (``--sweep-dim``), bf16:
  * ``n_cols`` (default) -- x = intermediate/FFN size over production widths; drives each
                            backend's tile-alignment behaviour. One config per token count.
  * ``tokens``           -- x = rows (batch*seq). One config per n_cols.

Usage:
    LIGER_BENCH_TARGET=swiglu_ncols  python benchmark_swiglu_backends.py --overwrite
    LIGER_BENCH_TARGET=swiglu_tokens python benchmark_swiglu_backends.py --sweep-dim tokens --overwrite
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import triton

from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import run_benchmarks

from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

QUANTILES = [0.5, 0.2, 0.8]

# Import each backend's autograd Function directly (bypasses LIGER_KERNEL_IMPL routing).
from liger_kernel.ops.swiglu import LigerSiLUMulFunction as _TritonSwiGLU  # noqa: E402

PROVIDER_FNS = {"triton": _TritonSwiGLU}

try:
    from liger_kernel.ops.cutile.ops.swiglu import LigerSiLUMulFunction as _CuTileSwiGLU

    PROVIDER_FNS["cutile"] = _CuTileSwiGLU
except Exception as e:  # pragma: no cover - optional backend
    print(f"[benchmark_swiglu_backends] cuTile backend unavailable, skipping: {e}")

try:
    from liger_kernel.ops.cutedsl.ops.swiglu import LigerSiLUMulCuteDSLFunction as _CuTeDSLSwiGLU

    PROVIDER_FNS["cutedsl"] = _CuTeDSLSwiGLU
except Exception as e:  # pragma: no cover - optional backend
    print(f"[benchmark_swiglu_backends] CuTe DSL backend unavailable, skipping: {e}")


# Sweep space (bf16). n_cols = intermediate/FFN size; mostly non-pow2 in real models.
N_COLS = [8192, 11008, 13824, 14336, 18944]  # gemma-pow2, llama2, qwen2.5-14b, llama3/mixtral, qwen2.5-7b
TOKENS = [1024, 2048, 4096, 8192]
DEFAULT_TOKENS = 4096
DEFAULT_NCOLS = 14336


def _resolve_dims(input: SingleBenchmarkRunInput):
    cfg = input.extra_benchmark_config
    if "n_cols" in cfg:  # tokens sweep: x is tokens
        return input.x, cfg["n_cols"], cfg["dtype"]
    return cfg["tokens"], input.x, cfg["dtype"]  # n_cols sweep: x is n_cols


def _make_ab(tokens, n_cols, dtype, requires_grad=False):
    a = torch.randn(tokens, n_cols, device=device, dtype=dtype, requires_grad=requires_grad)
    b = torch.randn(tokens, n_cols, device=device, dtype=dtype, requires_grad=requires_grad)
    return a, b


# ---------------------------------------------------------------------------
# Functional closures (same device work for eager and graph)
# ---------------------------------------------------------------------------
def _forward_closure(provider, a, b):
    if provider == "huggingface":
        return lambda: F.silu(a) * b
    fn = PROVIDER_FNS[provider]
    return lambda: fn.apply(a, b)


def _backward_closure(provider, a, b, dc):
    """Functional device-only backward (recompute-from-inputs, writes da/db in place).

    HuggingFace has no standalone backward kernel -> returns None.
    """
    if provider == "triton":
        from liger_kernel.ops.swiglu import swiglu_backward

        return lambda: swiglu_backward(a, b, dc, 1.0)
    if provider == "cutedsl":
        from liger_kernel.ops.cutedsl.ops.swiglu import swiglu_backward

        return lambda: swiglu_backward(a, b, dc, 1.0)
    if provider == "cutile":
        # cuTile exposes no public functional backward; replicate the class's launch.
        import cuda.tile as ct

        from liger_kernel.ops.cutile.ops.swiglu import MAX_FUSED_SIZE_BWD
        from liger_kernel.ops.cutile.ops.swiglu import _calculate_block_size
        from liger_kernel.ops.cutile.ops.swiglu import _swiglu_bwd_ct
        from liger_kernel.ops.cutile.ops.swiglu import _swiglu_bwd_ct_aligned

        n_cols = a.shape[-1]
        block = _calculate_block_size(n_cols, MAX_FUSED_SIZE_BWD)
        kernel = _swiglu_bwd_ct_aligned if n_cols % block == 0 else _swiglu_bwd_ct
        rows = a.shape[0]

        def cutile_bwd():
            ct.launch(
                torch.cuda.current_stream(),
                (rows, 1, 1),
                kernel,
                (dc, a, b, int(n_cols), int(block), 1.0),
            )

        return cutile_bwd
    return None


# ---------------------------------------------------------------------------
# Timers
# ---------------------------------------------------------------------------
def _time_eager(fn):
    ms_50, ms_20, ms_80 = triton.testing.do_bench(fn, rep=100, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def _percentiles(times):
    a = sorted(times)
    n = len(a)

    def p(q):
        return a[min(n - 1, max(0, int(round(q * (n - 1)))))]

    return p(0.2), p(0.5), p(0.8)


def _time_graph(fn, iters: int = 50, replays: int = 30):
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
    p20, p50, p80 = _percentiles(per_call)
    # Guard against an EMPTY capture: if a kernel launches on a stream other than the
    # capture stream (e.g. CuTe DSL SwiGLU, which unlike CuTe DSL RoPE does not thread
    # torch's current stream into its launch), nothing gets recorded and replay is a
    # ~0us no-op. Every real SwiGLU kernel here is >>1us, so treat sub-us as not-captured.
    if p50 * 1e3 < 1.0:  # p50 is ms; 1e3 -> us
        return _NAN
    return SingleBenchmarkRunOutput(y_20=p20, y_50=p50, y_80=p80)


_NAN = SingleBenchmarkRunOutput(y_20=float("nan"), y_50=float("nan"), y_80=float("nan"))


def _build_closure(provider, mode, tokens, n_cols, dtype, allow_autograd=False):
    """Return a device closure for (provider, mode), or None if N/A.

    ``allow_autograd`` enables an autograd-based path for HuggingFace backward/full
    (it has no standalone backward kernel; its backward is autograd differentiating
    ``F.silu(a)*b``). Only valid in eager -- autograd cannot be CUDA-graph captured,
    so the graph path leaves ``allow_autograd=False`` and HF bwd/full stay N/A.
    """
    if provider == "huggingface" and mode in ("backward", "full") and allow_autograd:
        a, b = _make_ab(tokens, n_cols, dtype, requires_grad=True)
        dc = torch.randn(tokens, n_cols, device=device, dtype=dtype)
        if mode == "full":

            def hf_full():
                a.grad = None
                b.grad = None
                c = F.silu(a) * b
                c.backward(dc)

            return hf_full

        # backward: build the forward graph once, time the autograd backward.
        c = F.silu(a) * b

        def hf_bwd():
            a.grad = None
            b.grad = None
            c.backward(dc, retain_graph=True)

        return hf_bwd

    a, b = _make_ab(tokens, n_cols, dtype)
    if mode == "forward":
        return _forward_closure(provider, a, b)
    dc = torch.randn(tokens, n_cols, device=device, dtype=dtype)
    if mode == "backward":
        return _backward_closure(provider, a, b, dc)
    if mode == "full":
        if provider == "huggingface":
            return None
        fwd = _forward_closure(provider, a, b)
        bwd = _backward_closure(provider, a, b, dc)

        def full():
            fwd()
            bwd()

        return full
    raise ValueError(mode)


def build_speed_bench_fn(timing):
    timer = _time_eager if timing == "eager" else _time_graph
    allow_autograd = timing == "eager"  # HF bwd/full need autograd, which graphs can't capture

    def bench(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
        tokens, n_cols, dtype = _resolve_dims(input)
        fn = _build_closure(input.kernel_provider, input.kernel_operation_mode, tokens, n_cols, dtype, allow_autograd)
        if fn is None:
            return _NAN
        return timer(fn)

    return bench


def bench_memory(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    """Peak MB for a full (forward + backward) pass."""
    tokens, n_cols, dtype = _resolve_dims(input)
    provider = input.kernel_provider
    dc = torch.randn(tokens, n_cols, device=device, dtype=dtype)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    if provider == "huggingface":
        # unfused reference: forward + autograd backward (no standalone kernel).
        a, b = _make_ab(tokens, n_cols, dtype, requires_grad=True)
        c = F.silu(a) * b
        c.backward(dc)
    else:
        a, b = _make_ab(tokens, n_cols, dtype)
        fwd = _forward_closure(provider, a, b)
        bwd = _backward_closure(provider, a, b, dc)
        c = fwd()
        bwd()
        del c
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return SingleBenchmarkRunOutput(y_20=peak_mb, y_50=peak_mb, y_80=peak_mb)


def parse_args():
    parser = argparse.ArgumentParser(description="SwiGLU backend comparison benchmark (bf16)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sweep-dim", choices=["n_cols", "tokens"], default="n_cols")
    parser.add_argument("--timing", choices=["eager", "graph", "both"], default="both")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.sweep_dim == "n_cols":
        x_name, x_label, x_values = "H", "intermediate size (n_cols)", N_COLS
        extra_benchmark_configs = [{"tokens": t, "dtype": torch.bfloat16} for t in (2048, DEFAULT_TOKENS)]
    else:
        x_name, x_label, x_values = "T", "tokens (rows)", TOKENS
        # one "good" (14336, ÷2048 -> cuTile fast path) and one "bad" (13824, not ÷2048
        # -> cuTile masked path) FFN width, to see whether the cliff is token-dependent.
        extra_benchmark_configs = [{"n_cols": c, "dtype": torch.bfloat16} for c in (14336, 13824)]

    all_providers = ["huggingface"] + list(PROVIDER_FNS.keys())
    base = {
        "kernel_name": "swiglu",
        "x_name": x_name,
        "x_label": x_label,
        "x_values": x_values,
        "extra_benchmark_configs": extra_benchmark_configs,
        "overwrite": args.overwrite,
    }

    if args.timing in ("eager", "both"):
        run_benchmarks(
            bench_test_fn=build_speed_bench_fn("eager"),
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            kernel_providers=all_providers,
            **base,
        )
        run_benchmarks(
            bench_test_fn=bench_memory,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            kernel_providers=all_providers,
            **base,
        )

    if args.timing in ("graph", "both"):
        run_benchmarks(
            bench_test_fn=build_speed_bench_fn("graph"),
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed_graph",
            metric_unit="ms",
            kernel_providers=all_providers,
            **base,
        )
