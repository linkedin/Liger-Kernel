import math
import os
import sys

import torch
import triton

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import compute_model_config_sweep_config
from benchmark_model_configs import compute_seq_len_sweep_config
from benchmark_model_configs import get_benchmark_model_config
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


def _setup_mhc(input: SingleBenchmarkRunInput):
    """Create input tensors and MHC kernel from benchmark config."""
    from test.transformers.test_mhc import mhc_coeffs_ref

    cfg = input.extra_benchmark_config
    T = cfg.get("T", input.x)
    B = cfg["B"]
    HC = cfg["HC"]
    C = cfg["C"]
    sub_kernel = cfg["sub_kernel"]
    tmax = cfg["tmax"]
    rms_eps = cfg["rms_eps"]
    pre_eps = cfg["pre_eps"]
    sinkhorn_eps = cfg["sinkhorn_eps"]
    post_mult = cfg["post_mult"]
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
            if input.kernel_provider == "liger":
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
            if input.kernel_provider == "liger":
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
            if input.kernel_provider == "liger":
                return liger_mhc_post_res(x, f_out, h_post_c, h_res_c)
            return torch.einsum("...oi,...ic->...oc", h_res_c, x.float()) + h_post_c.unsqueeze(
                -1
            ) * f_out.float().unsqueeze(-2)

        def fwd_loss():
            return fwd().square().mean()

    return grad_to_none, fwd, fwd_loss


def bench_speed_mhc(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    grad_to_none, fwd, fwd_loss = _setup_mhc(input)
    mode = input.kernel_operation_mode

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
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_mhc(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    grad_to_none, fwd, fwd_loss = _setup_mhc(input)

    def full():
        y = fwd_loss()
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


def _resolve_model_config_mhc(input: SingleBenchmarkRunInput):
    cfg = input.extra_benchmark_config
    model_info = cfg["model_configs"][input.x]
    return _setup_mhc(
        SingleBenchmarkRunInput(
            x=input.x,
            kernel_provider=input.kernel_provider,
            kernel_operation_mode=input.kernel_operation_mode,
            extra_benchmark_config={
                "B": cfg["B"],
                "HC": cfg["HC"],
                "C": model_info["hidden_size"],
                "T": cfg["T"],
                "sub_kernel": cfg["sub_kernel"],
                "tmax": cfg["tmax"],
                "rms_eps": cfg["rms_eps"],
                "pre_eps": cfg["pre_eps"],
                "sinkhorn_eps": cfg["sinkhorn_eps"],
                "post_mult": cfg["post_mult"],
            },
        )
    )


def bench_speed_mhc_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    grad_to_none, fwd, fwd_loss = _resolve_model_config_mhc(input)
    mode = input.kernel_operation_mode

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
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_mhc_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    grad_to_none, fwd, fwd_loss = _resolve_model_config_mhc(input)

    def full():
        y = fwd_loss()
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    mhc_defaults = {"tmax": 20, "rms_eps": 1e-6, "pre_eps": 0.0, "sinkhorn_eps": 1e-6, "post_mult": 2.0}

    for sub_kernel in ["coeffs", "pre", "post_res"]:
        if args.sweep_mode == "model_config":
            all_model_configs = list(MODEL_REGISTRY.values())
            B = 4
            HC = 4

            def _probe_factory(model_cfg, probe_bt, _sk=sub_kernel):
                def _probe():
                    T = max(1, probe_bt // B)
                    probe_input = SingleBenchmarkRunInput(
                        x=0,
                        kernel_provider="torch",
                        kernel_operation_mode="full",
                        extra_benchmark_config={
                            "B": B,
                            "HC": HC,
                            "C": model_cfg.hidden_size,
                            "T": T,
                            "sub_kernel": _sk,
                            **mhc_defaults,
                        },
                    )
                    _, _, fwd_loss = _setup_mhc(probe_input)
                    return fwd_loss()

                return _probe

            sweep = compute_model_config_sweep_config(all_model_configs, probe_fn_factory=_probe_factory, bt=args.bt)
            model_configs_info = {
                cfg.name: {"hidden_size": cfg.hidden_size, "dtype": cfg.dtype} for cfg in sweep.model_configs
            }

            common_configs = {
                "kernel_name": f"mhc_{sub_kernel}",
                "x_name": "model_config",
                "x_label": "model configuration",
                "x_values": [cfg.name for cfg in sweep.model_configs],
                "kernel_providers": ["liger", "torch"],
                "extra_benchmark_configs": [
                    {
                        "model_configs": model_configs_info,
                        "B": sweep.batch_size,
                        "HC": HC,
                        "T": sweep.seq_len,
                        "sub_kernel": sub_kernel,
                        **mhc_defaults,
                    }
                ],
                "overwrite": args.overwrite,
            }

            run_benchmarks(
                bench_test_fn=bench_speed_mhc_model_config,
                kernel_operation_modes=["forward", "backward", "full"],
                metric_name="speed",
                metric_unit="ms",
                **common_configs,
            )
            run_benchmarks(
                bench_test_fn=bench_memory_mhc_model_config,
                kernel_operation_modes=["full"],
                metric_name="memory",
                metric_unit="MB",
                **common_configs,
            )
        else:
            model = get_benchmark_model_config(args.model)
            B = 4
            HC = 4
            probe_T = 256

            def _probe(_sk=sub_kernel):
                probe_input = SingleBenchmarkRunInput(
                    x=0,
                    kernel_provider="torch",
                    kernel_operation_mode="full",
                    extra_benchmark_config={
                        "B": B,
                        "HC": HC,
                        "C": model.hidden_size,
                        "T": probe_T,
                        "sub_kernel": _sk,
                        **mhc_defaults,
                    },
                )
                _, _, fwd_loss = _setup_mhc(probe_input)
                return fwd_loss()

            config = compute_seq_len_sweep_config(model, probe_fn=_probe, probe_seq_len=probe_T)

            common_configs = {
                "kernel_name": f"mhc_{sub_kernel}",
                "x_name": "T",
                "x_label": "Sequence Length (T)",
                "x_values": [2**i for i in range(7, int(math.log2(max(128, config.seq_len))) + 1)],
                "kernel_providers": ["liger", "torch"],
                "extra_benchmark_configs": [
                    {
                        "B": B,
                        "HC": HC,
                        "C": model.hidden_size,
                        "sub_kernel": sub_kernel,
                        **mhc_defaults,
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
