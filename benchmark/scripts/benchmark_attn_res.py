import torch
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.ops.attn_res import LigerAttnResFunction
from liger_kernel.utils import infer_device

device = infer_device()


def pytorch_attn_res(V, w_query, w_norm, eps=1e-6):
    """
    Reference PyTorch implementation.
    V: [N, B, T, D], w_query: [D], w_norm: [D]
    """
    N, B, T, D = V.shape
    V_f32 = V.float()
    rms = torch.sqrt(V_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    K = (V_f32 / rms).to(V.dtype) * w_norm

    scores = torch.einsum("d, n b t d -> n b t", w_query.float(), K.float())
    alpha = scores.softmax(dim=0)

    h = torch.einsum("n b t, n b t d -> b t d", alpha, V.float()).to(V.dtype)
    return h


def bench_speed_attn_res(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    D = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    extra = input.extra_benchmark_config
    N = extra["N"]
    M = extra["M"]
    dtype = extra["dtype"]
    eps = extra["eps"]

    B, T = 4, M // 4  # split M tokens into batch and sequence
    V = torch.randn(N, B, T, D, device=device, dtype=dtype)
    w_query = torch.randn(D, device=device, dtype=dtype) * 0.02
    w_norm = torch.ones(D, device=device, dtype=dtype)
    dy = torch.randn(B, T, D, device=device, dtype=dtype)
    V.requires_grad_(True)

    def fwd_pytorch():
        return pytorch_attn_res(V, w_query, w_norm, eps)

    def fwd_liger():
        return LigerAttnResFunction.apply(V, w_query, w_norm, eps)

    def y_fwd():
        if provider == "liger":
            return fwd_liger()
        if provider == "pytorch":
            return fwd_pytorch()

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            y_fwd, grad_to_none=[V], rep=500, quantiles=QUANTILES,
        )
    elif mode == "backward":
        y = y_fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            grad_to_none=[V], rep=500, quantiles=QUANTILES,
        )
    elif mode == "full":
        def full():
            y = y_fwd()
            y.backward(dy, retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full, grad_to_none=[V], rep=500, quantiles=QUANTILES,
        )

    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_attn_res(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    D = input.x
    provider = input.kernel_provider

    extra = input.extra_benchmark_config
    N = extra["N"]
    M = extra["M"]
    dtype = extra["dtype"]
    eps = extra["eps"]

    B, T = 4, M // 4
    V = torch.randn(N, B, T, D, device=device, dtype=dtype)
    w_query = torch.randn(D, device=device, dtype=dtype) * 0.02
    w_norm = torch.ones(D, device=device, dtype=dtype)
    dy = torch.randn(B, T, D, device=device, dtype=dtype)
    V.requires_grad_(True)

    def fwd_pytorch():
        return pytorch_attn_res(V, w_query, w_norm, eps)

    def fwd_liger():
        return LigerAttnResFunction.apply(V, w_query, w_norm, eps)

    def full():
        if provider == "liger":
            y = fwd_liger()
        else:
            y = fwd_pytorch()
        y.backward(dy, retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "attn_res",
        "x_name": "D",
        "x_label": "hidden size",
        "x_values": [2**i for i in range(10, 15)],
        "kernel_providers": ["liger", "pytorch"],
        "extra_benchmark_configs": [
            {"N": 8, "M": 2048, "dtype": torch.float16, "eps": 1e-6},
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_attn_res,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_attn_res,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
