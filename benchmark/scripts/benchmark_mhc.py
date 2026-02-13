import os
import sys

import torch
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.functional import liger_mhc_coeffs
from liger_kernel.transformers.functional import liger_mhc_post_res
from liger_kernel.transformers.functional import liger_mhc_pre
from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def bench_speed_mhc(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    from test.transformers.test_mhc import mhc_coeffs_ref

    T = input.x
    B = input.extra_benchmark_config["B"]
    HC = input.extra_benchmark_config["HC"]
    C = input.extra_benchmark_config["C"]
    sub_kernel = input.extra_benchmark_config["sub_kernel"]
    tmax = input.extra_benchmark_config["tmax"]
    rms_eps = input.extra_benchmark_config["rms_eps"]
    pre_eps = input.extra_benchmark_config["pre_eps"]
    sinkhorn_eps = input.extra_benchmark_config["sinkhorn_eps"]
    post_mult = input.extra_benchmark_config["post_mult"]
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    coeffs_cfg = dict(tmax=tmax, rms_eps=rms_eps, pre_eps=pre_eps, sinkhorn_eps=sinkhorn_eps, post_mult=post_mult)
    need_grad = mode in ("backward", "full")

    x = torch.randn(B, T, HC, C, device=device, dtype=torch.bfloat16, requires_grad=need_grad)
    K, M = HC * C, HC * HC + 2 * HC
    phi = (torch.randn(K, M, device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(need_grad)
    b_param = torch.zeros(M, device=device, dtype=torch.float32, requires_grad=need_grad)
    alpha_pre = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=need_grad)
    alpha_post = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=need_grad)
    alpha_res = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=need_grad)

    grad_to_none = [x, phi, b_param, alpha_pre, alpha_post, alpha_res] if need_grad else None

    if sub_kernel == "coeffs":

        def fwd():
            if provider == "liger":
                return liger_mhc_coeffs(x, phi, b_param, alpha_pre, alpha_post, alpha_res, **coeffs_cfg)
            return mhc_coeffs_ref(x, phi, b_param, alpha_pre, alpha_post, alpha_res, **coeffs_cfg)

        def fwd_loss():
            h_pre, h_post, h_res = fwd()
            return h_pre.square().mean() + h_post.square().mean() + h_res.square().mean()

    elif sub_kernel == "pre":
        with torch.no_grad():
            h_pre_c, _, _ = liger_mhc_coeffs(
                x.detach(),
                phi.detach(),
                b_param.detach(),
                alpha_pre.detach(),
                alpha_post.detach(),
                alpha_res.detach(),
                **coeffs_cfg,
            )
        h_pre_c.requires_grad_(need_grad)
        grad_to_none = [x, h_pre_c] if need_grad else None

        def fwd():
            if provider == "liger":
                return liger_mhc_pre(x, h_pre_c)
            return (x.float() * h_pre_c.unsqueeze(-1)).sum(dim=-2)

        def fwd_loss():
            return fwd().square().mean()

    elif sub_kernel == "post_res":
        with torch.no_grad():
            _, h_post_c, h_res_c = liger_mhc_coeffs(
                x.detach(),
                phi.detach(),
                b_param.detach(),
                alpha_pre.detach(),
                alpha_post.detach(),
                alpha_res.detach(),
                **coeffs_cfg,
            )
        h_post_c.requires_grad_(need_grad)
        h_res_c.requires_grad_(need_grad)
        f_out = torch.randn(B, T, C, device=device, dtype=torch.bfloat16, requires_grad=need_grad)
        grad_to_none = [x, f_out, h_post_c, h_res_c] if need_grad else None

        def fwd():
            if provider == "liger":
                return liger_mhc_post_res(x, f_out, h_post_c, h_res_c)
            return torch.einsum("...oi,...ic->...oc", h_res_c, x.float()) + h_post_c.unsqueeze(
                -1
            ) * f_out.float().unsqueeze(-2)

        def fwd_loss():
            return fwd().square().mean()

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, rep=100, quantiles=QUANTILES)
    elif mode == "backward":
        y = fwd_loss()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(retain_graph=True),
            grad_to_none=grad_to_none,
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y = fwd_loss()
            y.backward()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, grad_to_none=grad_to_none, rep=100, quantiles=QUANTILES)

    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_mhc(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    from test.transformers.test_mhc import mhc_coeffs_ref

    T = input.x
    B = input.extra_benchmark_config["B"]
    HC = input.extra_benchmark_config["HC"]
    C = input.extra_benchmark_config["C"]
    sub_kernel = input.extra_benchmark_config["sub_kernel"]
    tmax = input.extra_benchmark_config["tmax"]
    rms_eps = input.extra_benchmark_config["rms_eps"]
    pre_eps = input.extra_benchmark_config["pre_eps"]
    sinkhorn_eps = input.extra_benchmark_config["sinkhorn_eps"]
    post_mult = input.extra_benchmark_config["post_mult"]
    provider = input.kernel_provider

    coeffs_cfg = dict(tmax=tmax, rms_eps=rms_eps, pre_eps=pre_eps, sinkhorn_eps=sinkhorn_eps, post_mult=post_mult)

    x = torch.randn(B, T, HC, C, device=device, dtype=torch.bfloat16, requires_grad=True)
    K, M = HC * C, HC * HC + 2 * HC
    phi = (torch.randn(K, M, device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    b_param = torch.zeros(M, device=device, dtype=torch.float32, requires_grad=True)
    alpha_pre = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)
    alpha_post = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)
    alpha_res = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)

    if sub_kernel == "coeffs":

        def full():
            if provider == "liger":
                hp, hpo, hr = liger_mhc_coeffs(x, phi, b_param, alpha_pre, alpha_post, alpha_res, **coeffs_cfg)
            else:
                hp, hpo, hr = mhc_coeffs_ref(x, phi, b_param, alpha_pre, alpha_post, alpha_res, **coeffs_cfg)
            (hp.square().mean() + hpo.square().mean() + hr.square().mean()).backward()

    elif sub_kernel == "pre":
        with torch.no_grad():
            h_pre_c, _, _ = liger_mhc_coeffs(
                x.detach(),
                phi.detach(),
                b_param.detach(),
                alpha_pre.detach(),
                alpha_post.detach(),
                alpha_res.detach(),
                **coeffs_cfg,
            )
        h_pre_c.requires_grad_(True)

        def full():
            if provider == "liger":
                out = liger_mhc_pre(x, h_pre_c)
            else:
                out = (x.float() * h_pre_c.unsqueeze(-1)).sum(dim=-2)
            out.square().mean().backward()

    elif sub_kernel == "post_res":
        with torch.no_grad():
            _, h_post_c, h_res_c = liger_mhc_coeffs(
                x.detach(),
                phi.detach(),
                b_param.detach(),
                alpha_pre.detach(),
                alpha_post.detach(),
                alpha_res.detach(),
                **coeffs_cfg,
            )
        h_post_c.requires_grad_(True)
        h_res_c.requires_grad_(True)
        f_out = torch.randn(B, T, C, device=device, dtype=torch.bfloat16, requires_grad=True)

        def full():
            if provider == "liger":
                out = liger_mhc_post_res(x, f_out, h_post_c, h_res_c)
            else:
                out = torch.einsum("...oi,...ic->...oc", h_res_c, x.float()) + h_post_c.unsqueeze(
                    -1
                ) * f_out.float().unsqueeze(-2)
            out.square().mean().backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    for sub_kernel in ["coeffs", "pre", "post_res"]:
        common_configs = {
            "kernel_name": f"mhc_{sub_kernel}",
            "x_name": "T",
            "x_label": "Sequence Length (T)",
            "x_values": [2**i for i in range(7, 12)],
            "kernel_providers": ["liger", "torch"],
            "extra_benchmark_configs": [
                {
                    "B": 4,
                    "HC": 4,
                    "C": 4096,
                    "tmax": 20,
                    "rms_eps": 1e-6,
                    "pre_eps": 0.0,
                    "sinkhorn_eps": 1e-6,
                    "post_mult": 2.0,
                    "sub_kernel": sub_kernel,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_mhc,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )

        run_benchmarks(
            bench_test_fn=bench_memory_mhc,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
