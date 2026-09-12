"""Profile the fused QK-Norm + RoPE kernel against the unfused PyTorch baseline.

The baseline is the *exact* sequence the ``qwen3_attention_forward`` monkeypatch
replaces (see ``modeling_qwen3.Qwen3Attention.forward``):

    q = q_norm(q.view(B, T, n_qh, hd)).transpose(1, 2)   # per-head RMSNorm + transpose
    k = k_norm(k.view(B, T, n_kh, hd)).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)           # RoPE

vs. the single fused Triton kernel:

    q, k = liger_qk_norm_rope(q, k, wq, wk, cos, sin, eps)

We report forward / backward / full latency (via ``triton.testing.do_bench``),
peak activation memory, and a ``torch.profiler`` CUDA-kernel breakdown so the
"before vs after" kernel behaviour is visible.

Shapes are the real Qwen3 dense attention configs (head_dim=128, GQA, eps=1e-6).

Usage::

    python benchmark/scripts/profile_qk_norm_rope.py                 # all configs, bf16
    python benchmark/scripts/profile_qk_norm_rope.py --dtype float32
    python benchmark/scripts/profile_qk_norm_rope.py --seq-len 8192 --bsz 1
    python benchmark/scripts/profile_qk_norm_rope.py --trace         # dump chrome traces
"""

import argparse
import gc

import torch
import triton

from torch.profiler import ProfilerActivity
from torch.profiler import profile

from liger_kernel.transformers.qk_norm_rope import liger_qk_norm_rope
from liger_kernel.utils import infer_device

device = infer_device()

# (name, n_q_head, n_kv_head, head_dim)  --  real Qwen3 dense configs
QWEN3_CONFIGS = [
    ("qwen3_0.6b", 16, 8, 128),
    ("qwen3_4b/8b", 32, 8, 128),
    ("qwen3_14b", 40, 8, 128),
    ("qwen3_32b", 64, 8, 128),
]

EPS = 1e-6  # Qwen3 rms_norm_eps


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rms_norm_ref(x, weight, eps):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


def baseline_forward(q, k, wq, wk, cos, sin, eps):
    """Unfused reference == what the monkeypatch replaces (q_norm/k_norm + RoPE)."""
    qn = rms_norm_ref(q, wq, eps).transpose(1, 2)
    kn = rms_norm_ref(k, wk, eps).transpose(1, 2)
    return apply_rotary_pos_emb(qn, kn, cos, sin)


def fused_forward(q, k, wq, wk, cos, sin, eps):
    return liger_qk_norm_rope(q, k, wq, wk, cos, sin, eps)


def make_cos_sin(seq_len, head_dim, dtype):
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (1000000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().unsqueeze(0).to(dtype), emb.sin().unsqueeze(0).to(dtype)


def make_inputs(bsz, seq_len, n_qh, n_kh, hd, dtype, requires_grad):
    q = torch.randn(bsz, seq_len, n_qh, hd, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(bsz, seq_len, n_kh, hd, device=device, dtype=dtype, requires_grad=requires_grad)
    wq = (1.0 + 0.1 * torch.randn(hd, device=device, dtype=dtype)).requires_grad_(requires_grad)
    wk = (1.0 + 0.1 * torch.randn(hd, device=device, dtype=dtype)).requires_grad_(requires_grad)
    cos, sin = make_cos_sin(seq_len, hd, dtype)
    return q, k, wq, wk, cos, sin


def _bench(fn):
    """Median / p20 / p80 ms via triton.testing.do_bench."""
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8], grad_to_none=None)
    return ms


def bench_mode(fwd, inputs, mode):
    q, k, wq, wk, cos, sin = inputs

    if mode == "forward":
        return _bench(lambda: fwd(q, k, wq, wk, cos, sin, EPS))

    # produce a fixed upstream grad
    oq, ok = fwd(q, k, wq, wk, cos, sin, EPS)
    gq = torch.randn_like(oq)
    gk = torch.randn_like(ok)

    if mode == "backward":

        def run():
            for t in (q, k, wq, wk):
                if t.grad is not None:
                    t.grad = None
            torch.autograd.backward((oq, ok), (gq, gk), retain_graph=True)

        return _bench(run)

    if mode == "full":

        def run():
            for t in (q, k, wq, wk):
                if t.grad is not None:
                    t.grad = None
            o1, o2 = fwd(q, k, wq, wk, cos, sin, EPS)
            torch.autograd.backward((o1, o2), (gq, gk))

        return _bench(run)

    raise ValueError(mode)


def peak_memory_mb(fwd, inputs):
    q, k, wq, wk, cos, sin = inputs
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    o1, o2 = fwd(q, k, wq, wk, cos, sin, EPS)
    torch.autograd.backward((o1, o2), (torch.randn_like(o1), torch.randn_like(o2)))
    return torch.cuda.max_memory_allocated() / (1024**2)


def run_latency(bsz, seq_len, dtype):
    print(f"\n{'=' * 96}")
    print(f"Latency (ms, median)  |  bsz={bsz} seq_len={seq_len} dtype={dtype}  device={torch.cuda.get_device_name()}")
    print(f"{'=' * 96}")
    header = (
        f"{'config':<12} {'n_qh/n_kh/hd':<14} "
        f"{'fwd base':>9} {'fwd fus':>9} {'fwd x':>6}  "
        f"{'full base':>10} {'full fus':>10} {'full x':>7}  "
        f"{'mem base':>9} {'mem fus':>9} {'mem save':>9}"
    )
    print(header)
    print("-" * len(header))

    for name, n_qh, n_kh, hd in QWEN3_CONFIGS:
        inputs = make_inputs(bsz, seq_len, n_qh, n_kh, hd, dtype, requires_grad=True)

        fwd_b = bench_mode(baseline_forward, inputs, "forward")
        fwd_f = bench_mode(fused_forward, inputs, "forward")
        full_b = bench_mode(baseline_forward, inputs, "full")
        full_f = bench_mode(fused_forward, inputs, "full")
        mem_b = peak_memory_mb(baseline_forward, inputs)
        mem_f = peak_memory_mb(fused_forward, inputs)

        print(
            f"{name:<12} {f'{n_qh}/{n_kh}/{hd}':<14} "
            f"{fwd_b:>9.4f} {fwd_f:>9.4f} {fwd_b / fwd_f:>5.2f}x  "
            f"{full_b:>10.4f} {full_f:>10.4f} {full_b / full_f:>6.2f}x  "
            f"{mem_b:>8.1f}M {mem_f:>8.1f}M {(1 - mem_f / mem_b) * 100:>7.1f}%"
        )

        del inputs
        gc.collect()
        torch.cuda.empty_cache()


def run_profiler(bsz, seq_len, dtype, trace):
    """torch.profiler kernel breakdown for one representative config (8B)."""
    name, n_qh, n_kh, hd = QWEN3_CONFIGS[1]
    print(f"\n{'=' * 96}")
    print(f"torch.profiler CUDA kernels  |  {name} bsz={bsz} seq_len={seq_len} dtype={dtype}")
    print(f"{'=' * 96}")

    for tag, fwd in (("BEFORE (unfused baseline)", baseline_forward), ("AFTER (fused kernel)", fused_forward)):
        inputs = make_inputs(bsz, seq_len, n_qh, n_kh, hd, dtype, requires_grad=True)
        q, k, wq, wk, cos, sin = inputs

        # warmup
        for _ in range(5):
            o1, o2 = fwd(q, k, wq, wk, cos, sin, EPS)
            torch.autograd.backward((o1, o2), (torch.randn_like(o1), torch.randn_like(o2)))
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                o1, o2 = fwd(q, k, wq, wk, cos, sin, EPS)
                torch.autograd.backward((o1, o2), (torch.randn_like(o1), torch.randn_like(o2)))
            torch.cuda.synchronize()

        print(f"\n--- {tag} ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))

        if trace:
            fname = f"trace_qk_norm_rope_{'baseline' if fwd is baseline_forward else 'fused'}.json"
            prof.export_chrome_trace(fname)
            print(f"chrome trace -> {fname}")

        del inputs
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--profiler", action="store_true", help="run torch.profiler kernel breakdown")
    parser.add_argument("--trace", action="store_true", help="export chrome traces (implies --profiler)")
    parser.add_argument("--no-latency", action="store_true")
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for profiling.")

    if not args.no_latency:
        run_latency(args.bsz, args.seq_len, dtype)

    if args.profiler or args.trace:
        run_profiler(args.bsz, args.seq_len, dtype, args.trace)


if __name__ == "__main__":
    main()
