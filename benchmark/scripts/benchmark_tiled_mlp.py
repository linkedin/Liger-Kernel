import math

import torch
import torch.nn as nn
import triton

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP
from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.transformers.tiled_mlp import LigerTiledGEGLUMLP
from liger_kernel.transformers.tiled_mlp import LigerTiledSwiGLUMLP
from liger_kernel.utils import infer_device

device = infer_device()


# DeepSpeed TiledMLP implementation
# Based on: https://github.com/deepspeedai/DeepSpeed/blob/v0.18.2/deepspeed/runtime/sequence_parallel/ulysses_sp.py#L838
class DeepSpeedTiledMLP(torch.autograd.Function):
    """
    DeepSpeed's TiledMLP implementation for fair comparison.
    This is the actual DeepSpeed algorithm that performs tiled MLP computation
    to massively reduce memory usage with very long sequence lengths.

    This module re-computes forward in the backward, so forward occurs twice per iteration.
    """

    @staticmethod
    def forward(ctx, fn, self, x, shards, compute_params) -> torch.Tensor:
        ctx.fn = fn
        ctx.self = self
        ctx.shards = shards
        ctx.compute_params = [p for p in compute_params if p.requires_grad] if compute_params else []
        ctx.save_for_backward(x)

        # x.shape could be [bs, seqlen, hidden_size] or [seqlen, hidden_size] (moe experts)
        x_shards = list(torch.chunk(x, chunks=shards, dim=-2))
        with torch.no_grad():
            output_shards = [fn(self, x_shard) for x_shard in x_shards]
        output_unsharded = torch.cat(output_shards, dim=-2)

        return output_unsharded

    @staticmethod
    def backward(ctx, *grads):
        fn = ctx.fn
        (x,) = ctx.saved_tensors
        self = ctx.self
        shards = ctx.shards
        compute_params = ctx.compute_params

        x_requires_grad = x.requires_grad
        x = x.detach()
        # detach() unsets x.requires_grad, so restore it
        x.requires_grad_(x_requires_grad)

        # x.shape could be [bs, seqlen, hidden_size] or [seqlen, hidden_size] (moe experts)
        hidden_size = x.shape[-1]
        x_shape_orig = x.shape

        # flatten bs+seqlen to avoid having stride issues when narrowing into seqlen w/ bs>1
        x = x.view(-1, hidden_size)
        incoming_grad = grads[0].view(-1, hidden_size)
        x_grad = torch.zeros_like(x)

        x_shards = list(torch.chunk(x, chunks=shards, dim=0))

        for i, x_shard in enumerate(x_shards):
            # Tell deepspeed not to add a new grad to its ipg bucket until the last shard is run
            # XXX: DDP, FSDP will need something similar to make it work
            if compute_params:
                if i + 1 < shards:
                    for param in compute_params:
                        if hasattr(param, "ds_grad_is_ready"):
                            param.ds_grad_is_ready = False
                else:
                    # last shard, can add the grad
                    for param in compute_params:
                        if hasattr(param, "ds_grad_is_ready"):
                            param.ds_grad_is_ready = True

            x_shard.requires_grad_(x_requires_grad)

            # if seqlen is not exactly divisible by shards the last step will be shorter than shard_step
            shard_step = x_shards[i].shape[0]
            shard_offset = i * x_shards[0].shape[0]

            x_shard.grad = x_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)
            incoming_grad_shard = incoming_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)
            with torch.enable_grad():
                output = fn(self, x_shard)
            torch.autograd.backward(output, incoming_grad_shard)

        # unflatten
        x_grad = x_grad.view(x_shape_orig)

        return (None, None, x_grad, None, None)


# DeepSpeed TiledMLP wrapper to match our interface
class DeepSpeedTiledMLPWrapper(nn.Module):
    """
    Wrapper for DeepSpeed's TiledMLP to match the interface used in benchmarks.
    Uses the DeepSpeed TiledMLP algorithm for memory-efficient MLP computation.
    """

    def __init__(self, config, num_shards=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_shards = num_shards

        self.mlp = LlamaMLP(config=config)

    def forward(self, x):
        # Calculate num_shards if not provided
        num_shards = self.num_shards
        if num_shards is None:
            hidden_size = x.shape[-1]
            seqlen = x.shape[-2]
            num_shards = math.ceil(seqlen / hidden_size)
        num_shards = max(1, num_shards)

        # Collect compute parameters for DeepSpeed ZeRO compatibility
        compute_params = [
            self.mlp.down_proj.weight,
            self.mlp.gate_proj.weight,
            self.mlp.up_proj.weight,
        ]

        # Define the MLP forward function for DeepSpeed TiledMLP
        def mlp_forward(mlp_module, x_input):
            return mlp_module.down_proj(mlp_module.act_fn(mlp_module.gate_proj(x_input)) * mlp_module.up_proj(x_input))

        # Use DeepSpeed's TiledMLP implementation
        return DeepSpeedTiledMLP.apply(
            mlp_forward,
            self.mlp,
            x,
            num_shards,
            compute_params,
        )


def bench_speed_tiled_mlp(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    seq_len = input.x
    bsz = input.extra_benchmark_config["bsz"]
    hidden_size = input.extra_benchmark_config["hidden_size"]
    intermediate_size = input.extra_benchmark_config["intermediate_size"]
    hidden_act = input.extra_benchmark_config["hidden_act"]
    dtype = input.extra_benchmark_config["dtype"]
    num_shards = input.extra_benchmark_config.get("num_shards", None)
    activation_type = input.extra_benchmark_config["activation_type"]
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    llama_config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
    )

    x_shape = (bsz, seq_len, hidden_size)

    # initialize input
    x = torch.randn(*x_shape, device=device, dtype=dtype, requires_grad=True)

    if activation_type == "geglu":
        if provider == "huggingface":
            layer = LlamaMLP(config=llama_config).to(device).to(dtype)
        elif provider == "liger":
            layer = LigerGEGLUMLP(config=llama_config).to(device).to(dtype)
        elif provider == "liger_tiled":
            layer = LigerTiledGEGLUMLP(config=llama_config, num_shards=num_shards).to(device).to(dtype)
        elif provider == "deepspeed_tiled":
            layer = DeepSpeedTiledMLPWrapper(config=llama_config, num_shards=num_shards).to(device).to(dtype)
        else:
            raise ValueError(f"Invalid provider: {provider} for GEGLU")
    elif activation_type == "swiglu":
        if provider == "huggingface":
            layer = LlamaMLP(config=llama_config).to(device).to(dtype)
        elif provider == "liger":
            layer = LigerSwiGLUMLP(config=llama_config).to(device).to(dtype)
        elif provider == "liger_tiled":
            layer = LigerTiledSwiGLUMLP(config=llama_config, num_shards=num_shards).to(device).to(dtype)
        elif provider == "deepspeed_tiled":
            layer = DeepSpeedTiledMLPWrapper(config=llama_config, num_shards=num_shards).to(device).to(dtype)
        else:
            raise ValueError(f"Invalid provider: {provider} for SwiGLU")
    else:
        raise ValueError(f"Invalid activation_type: {activation_type}")

    def fwd():
        return layer(x)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            fwd,
            grad_to_none=[x],
            rep=10,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        do = torch.randn_like(x)
        y = fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(do, retain_graph=True),
            grad_to_none=[x],
            rep=10,
            quantiles=QUANTILES,
        )
    else:

        def full():
            y = fwd()
            y.backward(torch.randn_like(y), retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            grad_to_none=[x],
            rep=10,
            quantiles=QUANTILES,
        )

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_tiled_mlp(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    seq_len = input.x
    bsz = input.extra_benchmark_config["bsz"]
    hidden_size = input.extra_benchmark_config["hidden_size"]
    intermediate_size = input.extra_benchmark_config["intermediate_size"]
    hidden_act = input.extra_benchmark_config["hidden_act"]
    dtype = input.extra_benchmark_config["dtype"]
    num_shards = input.extra_benchmark_config.get("num_shards", None)
    activation_type = input.extra_benchmark_config["activation_type"]
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    llama_config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
    )

    x_shape = (bsz, seq_len, hidden_size)
    # initialize input
    x = torch.randn(*x_shape, device=device, dtype=dtype, requires_grad=True)

    if activation_type == "geglu":
        if provider == "huggingface":
            layer = LlamaMLP(config=llama_config).to(device).to(dtype)
        elif provider == "liger":
            layer = LigerGEGLUMLP(config=llama_config).to(device).to(dtype)
        elif provider == "liger_tiled":
            layer = LigerTiledGEGLUMLP(config=llama_config, num_shards=num_shards).to(device).to(dtype)
        elif provider == "deepspeed_tiled":
            layer = DeepSpeedTiledMLPWrapper(config=llama_config, num_shards=num_shards).to(device).to(dtype)
        else:
            raise ValueError(f"Invalid provider: {provider} for GEGLU")
    elif activation_type == "swiglu":
        if provider == "huggingface":
            layer = LlamaMLP(config=llama_config).to(device).to(dtype)
        elif provider == "liger":
            layer = LigerSwiGLUMLP(config=llama_config).to(device).to(dtype)
        elif provider == "liger_tiled":
            layer = LigerTiledSwiGLUMLP(config=llama_config, num_shards=num_shards).to(device).to(dtype)
        elif provider == "deepspeed_tiled":
            layer = DeepSpeedTiledMLPWrapper(config=llama_config, num_shards=num_shards).to(device).to(dtype)
        else:
            raise ValueError(f"Invalid provider: {provider} for SwiGLU")
    else:
        raise ValueError(f"Invalid activation_type: {activation_type}")

    def fwd():
        return layer(x)

    def full():
        y = fwd()
        y.backward(torch.randn_like(y), retain_graph=True)

    if mode == "forward":
        mem_50, mem_20, mem_80 = _test_memory(
            fwd,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        do = torch.randn_like(x)
        y = fwd()
        mem_50, mem_20, mem_80 = _test_memory(
            lambda: y.backward(do, retain_graph=True),
            quantiles=QUANTILES,
        )
    else:
        mem_50, mem_20, mem_80 = _test_memory(
            full,
            quantiles=QUANTILES,
        )

    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    # Benchmark GEGLU variants
    kernel_providers_geglu = ["huggingface", "liger", "liger_tiled", "deepspeed_tiled"]

    common_configs_geglu = {
        "kernel_name": "tiled_geglu",
        "x_name": "T",
        "x_label": "sequence length",
        "x_values": [2**i for i in range(10, 15)],  # 1024 to 16384
        "kernel_providers": kernel_providers_geglu,
        "extra_benchmark_configs": [
            {
                "bsz": 2,
                "hidden_size": 2048,
                "intermediate_size": 4096,
                "hidden_act": "gelu_pytorch_tanh",
                "activation_type": "geglu",
                "num_shards": 4,
                "dtype": torch.bfloat16,
            }
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_tiled_mlp,
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs_geglu,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_tiled_mlp,
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs_geglu,
    )

    # Benchmark SwiGLU variants
    kernel_providers_swiglu = ["huggingface", "liger", "liger_tiled", "deepspeed_tiled"]

    common_configs_swiglu = {
        "kernel_name": "tiled_swiglu",
        "x_name": "T",
        "x_label": "sequence length",
        "x_values": [2**i for i in range(10, 15)],  # 1024 to 16384
        "kernel_providers": kernel_providers_swiglu,
        "extra_benchmark_configs": [
            {
                "bsz": 2,
                "hidden_size": 2048,
                "intermediate_size": 4096,
                "hidden_act": "silu",
                "activation_type": "swiglu",
                "num_shards": 4,
                "dtype": torch.bfloat16,
            }
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_tiled_mlp,
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs_swiglu,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_tiled_mlp,
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs_swiglu,
    )
