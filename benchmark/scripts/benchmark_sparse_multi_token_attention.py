import torch
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.multi_token_attention import LigerMultiTokenAttention
from liger_kernel.utils import infer_device

device = infer_device()


class TorchSparseMultiTokenAttention(torch.nn.Module):
    def __init__(self, C_in, C_out, K, groups, bias, dtype, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(C_out, C_in // groups, K, K, dtype=dtype, device=device))
        self.bias = torch.nn.Parameter(torch.empty(C_out, dtype=dtype, device=device)) if bias else None
        self.K = K
        self.groups = groups
        self.dtype = dtype
        self.compute_dtype = torch.float32

    def forward(self, scores):
        B, C_in, L, _ = scores.shape
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=scores.device)).view(1, 1, L, L)
        inf = torch.tensor(-1e9, device=scores.device, dtype=self.compute_dtype)
        zero = torch.tensor(0.0, device=scores.device, dtype=self.compute_dtype)

        s_compute = scores.to(self.compute_dtype)
        s_inf = s_compute.masked_fill(~mask, inf)

        dim = -1
        z = s_inf

        z_sorted, _ = torch.sort(z, dim=dim, descending=True)

        cum_sum = torch.cumsum(z_sorted, dim=dim)

        k_indices = torch.arange(1, L + 1, device=z.device, dtype=z.dtype).view(1, 1, 1, L)

        is_positive = z_sorted > -1e8
        condition = (1 + k_indices * z_sorted > cum_sum) & is_positive
        k_sparsemax = torch.sum(condition, dim=dim, keepdim=True)

        k_sparsemax_safe = torch.max(k_sparsemax, torch.ones_like(k_sparsemax))

        cum_sum_k = torch.gather(cum_sum, dim=dim, index=k_sparsemax_safe.long() - 1)

        tau = (cum_sum_k - 1) / k_sparsemax_safe.to(z.dtype)
        tau = torch.where(k_sparsemax == 0, torch.full_like(tau, float("inf")), tau)

        probs = torch.clamp(z - tau, min=0)

        weight_compute = self.weight.to(self.compute_dtype)
        bias_compute = self.bias.to(self.compute_dtype) if self.bias is not None else None

        out_c = torch.nn.functional.conv2d(
            probs, weight_compute, bias_compute, stride=1, padding=self.K // 2, groups=self.groups
        )
        return out_c.masked_fill(~mask, zero).to(scores.dtype)


def bench_speed_sparse_multi_token_attention(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    L = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    extra_benchmark_config = input.extra_benchmark_config
    B = extra_benchmark_config["B"]
    C_in = extra_benchmark_config["C_in"]
    C_out = extra_benchmark_config["C_out"]
    K = extra_benchmark_config["K"]
    groups = extra_benchmark_config["groups"]
    bias = extra_benchmark_config["bias"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (B, C_in, L, L)

    liger_attn = (
        LigerMultiTokenAttention(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=K,
            stride=1,
            padding=K // 2,
            dilation=1,
            groups=groups,
            bias=bias,
            sparse=True,
        )
        .to(device)
        .to(dtype)
    )

    torch_attn = TorchSparseMultiTokenAttention(
        C_in=C_in, C_out=C_out, K=K, groups=groups, bias=bias, dtype=dtype, device=device
    )

    with torch.no_grad():
        torch.nn.init.kaiming_uniform_(liger_attn.weight, a=5**0.5)
        if bias:
            torch.nn.init.zeros_(liger_attn.bias)
        torch_attn.weight.copy_(liger_attn.weight)
        if bias:
            torch_attn.bias.copy_(liger_attn.bias)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def fwd():
        if provider == "liger":
            return liger_attn(x)
        elif provider == "torch":
            return torch_attn(x)

    print(f"Starting Warmup for input size: {x_shape}")
    _ = fwd()
    if mode in ("backward", "full"):
        y = _
        y.backward(dy, retain_graph=True)
    print("Done Warmup")

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, grad_to_none=[x], rep=100, quantiles=QUANTILES)
    elif mode == "backward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            grad_to_none=[x],
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward(dy, retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, grad_to_none=[x], rep=100, quantiles=QUANTILES)

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_sparse_multi_token_attention(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    L = input.x
    provider = input.kernel_provider

    extra_benchmark_config = input.extra_benchmark_config
    B = extra_benchmark_config["B"]
    C_in = extra_benchmark_config["C_in"]
    C_out = extra_benchmark_config["C_out"]
    K = extra_benchmark_config["K"]
    groups = extra_benchmark_config["groups"]
    bias = extra_benchmark_config["bias"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (B, C_in, L, L)

    liger_attn = (
        LigerMultiTokenAttention(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=K,
            stride=1,
            padding=K // 2,
            dilation=1,
            groups=groups,
            bias=bias,
            sparse=True,
        )
        .to(device)
        .to(dtype)
    )

    torch_attn = TorchSparseMultiTokenAttention(
        C_in=C_in, C_out=C_out, K=K, groups=groups, bias=bias, dtype=dtype, device=device
    )

    with torch.no_grad():
        torch.nn.init.kaiming_uniform_(liger_attn.weight, a=5**0.5)
        if bias:
            torch.nn.init.zeros_(liger_attn.bias)
        torch_attn.weight.copy_(liger_attn.weight)
        if bias:
            torch_attn.bias.copy_(liger_attn.bias)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def fwd():
        if provider == "liger":
            return liger_attn(x)
        elif provider == "torch":
            return torch_attn(x)

    def full():
        y = fwd()
        y.backward(dy, retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)

    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "sparse_multi_token_attention",
        "x_name": "L",
        "x_label": "sequence length",
        "x_values": [2**i for i in range(5, 10)],
        "kernel_providers": ["liger", "torch"],
        "extra_benchmark_configs": [
            {
                "B": 2,
                "C_in": 4,
                "C_out": 4,
                "K": 3,
                "groups": 1,
                "bias": True,
                "dtype": torch.float32,
            }
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_sparse_multi_token_attention,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_sparse_multi_token_attention,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
