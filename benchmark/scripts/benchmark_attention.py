import torch
import triton
from transformers.models.llama.modeling_llama import repeat_kv

from utils import (
    QUANTILES,
    SingleBenchmarkRunInput,
    SingleBenchmarkRunOutput,
    _test_memory,
    parse_benchmark_script_args,
    run_benchmarks,
)
from liger_kernel.ops.flash_attention import flash_attn_func


#############################################################################
# Test the memory consumption of the attention layer
#############################################################################


def bench_memory_attention(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    seqlen = input.x
    batch_size = input.extra_benchmark_config["batch_size"]
    nheads_q = input.extra_benchmark_config["nheads_q"]
    nheads_kv = input.extra_benchmark_config["nheads_kv"]
    hidden_size = input.extra_benchmark_config["hidden_size"]
    dtype = input.extra_benchmark_config["dtype"]
    provider = input.kernel_provider
    device = "cuda"

    head_dim = hidden_size // nheads_q
    q = torch.normal(
        0, 0.5, (batch_size, seqlen, nheads_q, head_dim), dtype=dtype, device=device
    ).requires_grad_()
    k = torch.normal(
        0, 0.5, (batch_size, seqlen, nheads_kv, head_dim), dtype=dtype, device=device
    ).requires_grad_()
    v = torch.normal(
        0, 0.5, (batch_size, seqlen, nheads_kv, head_dim), dtype=dtype, device=device
    ).requires_grad_()
    do = torch.randn_like(q)

    if provider == "torch":
        q, k, v, do = [x.transpose(1, 2).contiguous() for x in [q, k, v, do]]

    def fwd():
        if provider == "liger":
            return flash_attn_func(q, k, v)
        if provider == "torch":
            if nheads_q == nheads_kv:
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)
            else:
                ngroups = nheads_q // nheads_kv
                return torch.nn.functional.scaled_dot_product_attention(
                    q, repeat_kv(k, ngroups), repeat_kv(v, ngroups)
                )

    def full():
        y = fwd()
        y.backward(do)

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


# #############################################################################
# # Test the speed of the fused linear cross entropy loss
# #############################################################################


def bench_speed_attention(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    seqlen = input.x
    batch_size = input.extra_benchmark_config["batch_size"]
    nheads_q = input.extra_benchmark_config["nheads_q"]
    nheads_kv = input.extra_benchmark_config["nheads_kv"]
    hidden_size = input.extra_benchmark_config["hidden_size"]
    dtype = input.extra_benchmark_config["dtype"]
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    device = "cuda"

    head_dim = hidden_size // nheads_q
    q = torch.normal(
        0, 0.5, (batch_size, seqlen, nheads_q, head_dim), dtype=dtype, device=device
    ).requires_grad_()
    k = torch.normal(
        0, 0.5, (batch_size, seqlen, nheads_kv, head_dim), dtype=dtype, device=device
    ).requires_grad_()
    v = torch.normal(
        0, 0.5, (batch_size, seqlen, nheads_kv, head_dim), dtype=dtype, device=device
    ).requires_grad_()
    do = torch.randn_like(q)

    if provider == "torch":
        q, k, v, do = [x.transpose(1, 2).contiguous() for x in [q, k, v, do]]

    def fwd():
        if provider == "liger":
            return flash_attn_func(q, k, v)
        if provider == "torch":
            if nheads_q == nheads_kv:
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)
            else:
                ngroups = nheads_q // nheads_kv
                return torch.nn.functional.scaled_dot_product_attention(
                    q, repeat_kv(k, ngroups), repeat_kv(v, ngroups)
                )

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            fwd,
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        y = fwd()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(do, retain_graph=True),
            grad_to_none=[q, k, v],
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward(do)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            rep=100,
            quantiles=QUANTILES,
        )
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "attention",
        "x_name": "seqlen",
        "x_label": "Sequence length",
        "x_values": [2**i for i in range(5, 15)],
        "kernel_providers": ["liger", "torch"],
        "extra_benchmark_configs": [
            {
                "batch_size": 4,
                "nheads_q": 32,
                "nheads_kv": 8,
                "hidden_size": 4096,
                "dtype": torch.float16,
            }
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_attention,
        kernel_operation_modes=["forward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs
    )
    run_benchmarks(
        bench_test_fn=bench_memory_attention,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs
    )
