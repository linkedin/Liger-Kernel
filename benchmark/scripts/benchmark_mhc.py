import argparse
import time

import torch

from utils import mhc_coeffs_ref

from liger_kernel.ops.mhc import liger_mhc_coeffs
from liger_kernel.ops.mhc import liger_mhc_post_res
from liger_kernel.ops.mhc import liger_mhc_pre


def _time_loop(fn, iters=200, warmup=50) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) * 1e3 / iters


def _peak_bytes(fn) -> int:
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def _fmt_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 2):.1f} MB"


def bench_pair(name, fn_fast, fn_ref, iters=200, warmup=50):
    t_fast = _time_loop(fn_fast, iters=iters, warmup=warmup)
    t_ref = _time_loop(fn_ref, iters=iters, warmup=warmup)
    peak_fast = _peak_bytes(fn_fast)
    peak_ref = _peak_bytes(fn_ref)
    speedup = t_ref / t_fast if t_fast > 0 else float("inf")
    print(
        f"{name:20s}: fast {t_fast:.3f} ms | ref {t_ref:.3f} ms | {speedup:.2f}x | "
        f"peak { _fmt_mb(peak_fast) } vs { _fmt_mb(peak_ref) }"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--HC", type=int, default=4)
    ap.add_argument("--C", type=int, default=8192)
    ap.add_argument("--tmax", type=int, default=20)
    args = ap.parse_args()

    device = "cuda"
    B, T, HC, C = args.B, args.T, args.HC, args.C
    K = HC * C
    M = HC * HC + 2 * HC

    x = torch.randn(B, T, HC, C, device=device, dtype=torch.bfloat16)
    phi = torch.randn(K, M, device=device, dtype=x.dtype) * 0.02
    b = torch.zeros(M, device=device, dtype=torch.float32)
    alpha_pre = torch.tensor(1.0, device=device, dtype=torch.float32)
    alpha_post = torch.tensor(1.0, device=device, dtype=torch.float32)
    alpha_res = torch.tensor(1.0, device=device, dtype=torch.float32)
    coeffs_cfg = dict(tmax=args.tmax, rms_eps=1e-6, pre_eps=0.0, sinkhorn_eps=1e-6, post_mult=2.0)

    def _zero_grads(tensors):
        for t in tensors:
            if t.grad is not None:
                t.grad = None

    def run_coeffs():
        return liger_mhc_coeffs(x, phi, b, alpha_pre, alpha_post, alpha_res, **coeffs_cfg)

    def run_coeffs_ref():
        return mhc_coeffs_ref(x, phi, b, alpha_pre, alpha_post, alpha_res, **coeffs_cfg)

    h_pre, h_post, h_res = run_coeffs()
    f_out = torch.randn(B, T, C, device=device, dtype=torch.bfloat16)

    def run_pre():
        liger_mhc_pre(x, h_pre)

    def run_pre_ref():
        (x.float() * h_pre.unsqueeze(-1)).sum(dim=-2)

    def run_post_res():
        liger_mhc_post_res(x, f_out, h_post, h_res)

    def run_post_res_ref():
        torch.einsum("...oi,...ic->...oc", h_res, x.float()) + h_post.unsqueeze(-1) * f_out.float().unsqueeze(-2)

    # Forward-only benchmarks
    bench_pair("mhc_coeffs fwd", run_coeffs, run_coeffs_ref)
    bench_pair("mhc_pre fwd", run_pre, run_pre_ref)
    bench_pair("mhc_post_res fwd", run_post_res, run_post_res_ref)

    # Backward benchmarks (includes forward+backward)
    def make_coeffs_bw(fn_coeffs):
        x_bw = x.detach().clone().requires_grad_(True)
        phi_bw = phi.detach().clone().requires_grad_(True)
        b_bw = b.detach().clone().requires_grad_(True)
        ap_bw = alpha_pre.detach().clone().requires_grad_(True)
        apo_bw = alpha_post.detach().clone().requires_grad_(True)
        ar_bw = alpha_res.detach().clone().requires_grad_(True)

        def _run():
            h_pre_b, h_post_b, h_res_b = fn_coeffs(x_bw, phi_bw, b_bw, ap_bw, apo_bw, ar_bw)
            loss = h_pre_b.square().mean() + h_post_b.square().mean() + h_res_b.square().mean()
            loss.backward()
            _zero_grads([x_bw, phi_bw, b_bw, ap_bw, apo_bw, ar_bw])

        return _run

    def make_pre_bw(fn_pre):
        x_bw = x.detach().clone().requires_grad_(True)
        h_bw = h_pre.detach().clone().requires_grad_(True)

        def _run():
            out = fn_pre(x_bw, h_bw)
            loss = out.square().mean()
            loss.backward()
            _zero_grads([x_bw, h_bw])

        return _run

    def make_post_res_bw(fn_post_res):
        x_bw = x.detach().clone().requires_grad_(True)
        f_bw = f_out.detach().clone().requires_grad_(True)
        hpost_bw = h_post.detach().clone().requires_grad_(True)
        hres_bw = h_res.detach().clone().requires_grad_(True)

        def _run():
            out = fn_post_res(x_bw, f_bw, hpost_bw, hres_bw)
            loss = out.square().mean()
            loss.backward()
            _zero_grads([x_bw, f_bw, hpost_bw, hres_bw])

        return _run

    bench_pair(
        "mhc_coeffs bwd",
        make_coeffs_bw(lambda a, b_, c, d, e, f: liger_mhc_coeffs(a, b_, c, d, e, f, **coeffs_cfg)),
        make_coeffs_bw(lambda a, b_, c, d, e, f: mhc_coeffs_ref(a, b_, c, d, e, f, **coeffs_cfg)),
    )
    bench_pair(
        "mhc_pre bwd",
        make_pre_bw(lambda a, b_: liger_mhc_pre(a, b_)),
        make_pre_bw(lambda a, b_: (a.float() * b_.unsqueeze(-1)).sum(dim=-2)),
    )
    bench_pair(
        "mhc_post_res bwd",
        make_post_res_bw(lambda a, b_, c, d: liger_mhc_post_res(a, b_, c, d)),
        make_post_res_bw(
            lambda a, b_, c, d: torch.einsum("...oi,...ic->...oc", d, a.float())
            + c.unsqueeze(-1) * b_.float().unsqueeze(-2)
        ),
    )


if __name__ == "__main__":
    main()
