import argparse
import os
import sys

import torch
import triton

from utils import QUANTILES
from utils import _test_memory

from liger_kernel.ops.mhc import liger_mhc_coeffs
from liger_kernel.ops.mhc import liger_mhc_post_res
from liger_kernel.ops.mhc import liger_mhc_pre
from liger_kernel.utils import infer_device

device = infer_device()


def _fmt_ms(ms: float) -> str:
    return f"{ms:.3f} ms"


def _fmt_mb(num_mb: float) -> str:
    return f"{num_mb:.1f} MB"


def _bench_time(fn, *, grad_to_none=None, rep=200):
    ms_50, ms_20, ms_80 = triton.testing.do_bench(fn, quantiles=QUANTILES, grad_to_none=grad_to_none, rep=rep)
    return ms_50, ms_20, ms_80


def _bench_memory(fn):
    mem_50, mem_20, mem_80 = _test_memory(fn, quantiles=QUANTILES)
    return mem_50, mem_20, mem_80


def bench_pair(name, fn_fast, fn_ref, *, grad_to_none_fast=None, grad_to_none_ref=None, rep=200):
    t50_f, t20_f, t80_f = _bench_time(fn_fast, grad_to_none=grad_to_none_fast, rep=rep)
    t50_r, t20_r, t80_r = _bench_time(fn_ref, grad_to_none=grad_to_none_ref, rep=rep)
    m50_f, m20_f, m80_f = _bench_memory(fn_fast)
    m50_r, m20_r, m80_r = _bench_memory(fn_ref)

    speedup = t50_r / t50_f if t50_f > 0 else float("inf")
    print(
        f"{name:20s}: "
        f"fast {_fmt_ms(t50_f)} (p20 {_fmt_ms(t20_f)}, p80 {_fmt_ms(t80_f)}) | "
        f"ref {_fmt_ms(t50_r)} (p20 {_fmt_ms(t20_r)}, p80 {_fmt_ms(t80_r)}) | "
        f"{speedup:.2f}x | "
        f"peak {_fmt_mb(m50_f)} (p20 {_fmt_mb(m20_f)}, p80 {_fmt_mb(m80_f)}) vs "
        f"{_fmt_mb(m50_r)} (p20 {_fmt_mb(m20_r)}, p80 {_fmt_mb(m80_r)})"
    )


def main():
    if device != "cuda":
        raise RuntimeError("CUDA device required for mHC benchmarks")

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from test.transformers.test_mhc import mhc_coeffs_ref

    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--HC", type=int, default=4)
    ap.add_argument("--C", type=int, default=4096)
    ap.add_argument("--tmax", type=int, default=20)
    args = ap.parse_args()

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

    bench_pair("mhc_coeffs fwd", run_coeffs, run_coeffs_ref)
    bench_pair("mhc_pre fwd", run_pre, run_pre_ref)
    bench_pair("mhc_post_res fwd", run_post_res, run_post_res_ref)

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

        return _run, [x_bw, phi_bw, b_bw, ap_bw, apo_bw, ar_bw]

    def make_pre_bw(fn_pre):
        x_bw = x.detach().clone().requires_grad_(True)
        h_bw = h_pre.detach().clone().requires_grad_(True)

        def _run():
            out = fn_pre(x_bw, h_bw)
            loss = out.square().mean()
            loss.backward()

        return _run, [x_bw, h_bw]

    def make_post_res_bw(fn_post_res):
        x_bw = x.detach().clone().requires_grad_(True)
        f_bw = f_out.detach().clone().requires_grad_(True)
        hpost_bw = h_post.detach().clone().requires_grad_(True)
        hres_bw = h_res.detach().clone().requires_grad_(True)

        def _run():
            out = fn_post_res(x_bw, f_bw, hpost_bw, hres_bw)
            loss = out.square().mean()
            loss.backward()

        return _run, [x_bw, f_bw, hpost_bw, hres_bw]

    coeffs_bw_fast, coeffs_bw_grads_fast = make_coeffs_bw(
        lambda a, b_, c, d, e, f: liger_mhc_coeffs(a, b_, c, d, e, f, **coeffs_cfg)
    )
    coeffs_bw_ref, coeffs_bw_grads_ref = make_coeffs_bw(
        lambda a, b_, c, d, e, f: mhc_coeffs_ref(a, b_, c, d, e, f, **coeffs_cfg)
    )
    pre_bw_fast, pre_bw_grads_fast = make_pre_bw(lambda a, b_: liger_mhc_pre(a, b_))
    pre_bw_ref, pre_bw_grads_ref = make_pre_bw(lambda a, b_: (a.float() * b_.unsqueeze(-1)).sum(dim=-2))
    post_bw_fast, post_bw_grads_fast = make_post_res_bw(lambda a, b_, c, d: liger_mhc_post_res(a, b_, c, d))
    post_bw_ref, post_bw_grads_ref = make_post_res_bw(
        lambda a, b_, c, d: torch.einsum("...oi,...ic->...oc", d, a.float())
        + c.unsqueeze(-1) * b_.float().unsqueeze(-2)
    )

    bench_pair(
        "mhc_coeffs bwd",
        coeffs_bw_fast,
        coeffs_bw_ref,
        grad_to_none_fast=coeffs_bw_grads_fast,
        grad_to_none_ref=coeffs_bw_grads_ref,
    )
    bench_pair(
        "mhc_pre bwd",
        pre_bw_fast,
        pre_bw_ref,
        grad_to_none_fast=pre_bw_grads_fast,
        grad_to_none_ref=pre_bw_grads_ref,
    )
    bench_pair(
        "mhc_post_res bwd",
        post_bw_fast,
        post_bw_ref,
        grad_to_none_fast=post_bw_grads_fast,
        grad_to_none_ref=post_bw_grads_ref,
    )


if __name__ == "__main__":
    main()
