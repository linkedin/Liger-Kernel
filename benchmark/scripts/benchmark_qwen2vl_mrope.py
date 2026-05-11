import math
import os
import sys

import torch
import triton

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import compute_model_config_sweep_config
from benchmark_model_configs import compute_seq_len_sweep_config
from benchmark_model_configs import get_benchmark_model_config
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLTextConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLRotaryEmbedding
from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb
from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def _setup_qwen2vl_mrope(input: SingleBenchmarkRunInput):
    """Create input tensors and Qwen2VL M-RoPE embedding from benchmark config."""
    cfg = input.extra_benchmark_config
    num_q_heads = cfg["num_q_heads"]
    num_kv_heads = cfg["num_kv_heads"]
    dtype = cfg["dtype"]
    hidden_size = cfg.get("hidden_size", input.x)
    seq_len = cfg.get("seq_len", input.x)

    head_dim = hidden_size // num_q_heads
    mrope_section_hw = head_dim * 3 // 16
    mrope_section = [
        head_dim // 2 - 2 * mrope_section_hw,
        mrope_section_hw,
        mrope_section_hw,
    ]
    config = Qwen2VLTextConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        rope_theta=1000000.0,
        mrope_section=mrope_section,
    )
    rotary_emb = Qwen2VLRotaryEmbedding(config, device=device)
    q = torch.randn(
        (1, seq_len, num_q_heads, head_dim),
        device=device,
        requires_grad=True,
        dtype=dtype,
    ).transpose(1, 2)
    k = torch.randn(
        (1, seq_len, num_kv_heads, head_dim),
        device=device,
        requires_grad=True,
        dtype=dtype,
    ).transpose(1, 2)
    dq, dk = (
        torch.randn_like(q, device=device, dtype=dtype),
        torch.randn_like(k, device=device, dtype=dtype),
    )
    pos_ids = torch.arange(seq_len * 3, device=device, dtype=torch.long).view(3, 1, -1)
    cos, sin = rotary_emb(k, pos_ids)

    if input.kernel_provider == "liger":
        fwd_fn = lambda: liger_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
    elif input.kernel_provider == "huggingface":
        fwd_fn = lambda: apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for M-RoPE embedding")

    return q, k, dq, dk, fwd_fn


def bench_speed_qwen2vl_mrope(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    q, k, dq, dk, fwd = _setup_qwen2vl_mrope(input)
    mode = input.kernel_operation_mode

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            fwd,
            grad_to_none=[q, k],
            rep=400,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        q_out, k_out = fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: torch.autograd.grad((q_out, k_out), (q, k), (dq, dk), allow_unused=True, retain_graph=True),
            grad_to_none=[q, k],
            rep=400,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            q_out, k_out = fwd()
            torch.autograd.grad((q_out, k_out), (q, k), (dq, dk), allow_unused=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            grad_to_none=[q, k],
            rep=400,
            quantiles=QUANTILES,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_qwen2vl_mrope(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    q, k, dq, dk, fwd_fn = _setup_qwen2vl_mrope(input)

    def full():
        q_out, k_out = fwd_fn()
        torch.autograd.grad((q_out, k_out), (q, k), (dq, dk), allow_unused=True, retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(
        full,
        quantiles=QUANTILES,
    )
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


def _resolve_model_config_qwen2vl_mrope(input: SingleBenchmarkRunInput):
    cfg = input.extra_benchmark_config
    model_info = cfg["model_configs"][input.x]
    return _setup_qwen2vl_mrope(
        SingleBenchmarkRunInput(
            x=input.x,
            kernel_provider=input.kernel_provider,
            extra_benchmark_config={
                "hidden_size": model_info["hidden_size"],
                "num_q_heads": model_info["num_q_heads"],
                "num_kv_heads": model_info["num_kv_heads"],
                "dtype": model_info["dtype"],
                "seq_len": cfg["seq_len"],
            },
        )
    )


def bench_speed_qwen2vl_mrope_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    q, k, dq, dk, fwd_fn = _resolve_model_config_qwen2vl_mrope(input)
    mode = input.kernel_operation_mode

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd_fn, grad_to_none=[q, k], rep=400, quantiles=QUANTILES)
    elif mode == "backward":
        q_out, k_out = fwd_fn()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: torch.autograd.grad((q_out, k_out), (q, k), (dq, dk), allow_unused=True, retain_graph=True),
            grad_to_none=[q, k],
            rep=400,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            q_out, k_out = fwd_fn()
            torch.autograd.grad((q_out, k_out), (q, k), (dq, dk), allow_unused=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, grad_to_none=[q, k], rep=400, quantiles=QUANTILES)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_qwen2vl_mrope_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    q, k, dq, dk, fwd_fn = _resolve_model_config_qwen2vl_mrope(input)

    def full():
        q_out, k_out = fwd_fn()
        torch.autograd.grad((q_out, k_out), (q, k), (dq, dk), allow_unused=True, retain_graph=True)

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

    if args.sweep_mode == "model_config":
        all_model_configs = list(MODEL_REGISTRY.values())

        def _probe_factory(model_cfg, probe_bt):
            def _probe():
                probe_input = SingleBenchmarkRunInput(
                    x=0,
                    kernel_provider="huggingface",
                    extra_benchmark_config={
                        "hidden_size": model_cfg.hidden_size,
                        "num_q_heads": model_cfg.num_attention_heads,
                        "num_kv_heads": model_cfg.num_key_value_heads,
                        "dtype": model_cfg.dtype,
                        "seq_len": probe_bt,
                    },
                )
                _, _, _, _, fwd_fn = _setup_qwen2vl_mrope(probe_input)
                return fwd_fn()[0]

            return _probe

        sweep = compute_model_config_sweep_config(all_model_configs, probe_fn_factory=_probe_factory, bt=args.bt)
        model_configs_info = {
            cfg.name: {
                "hidden_size": cfg.hidden_size,
                "num_q_heads": cfg.num_attention_heads,
                "num_kv_heads": cfg.num_key_value_heads,
                "dtype": cfg.dtype,
            }
            for cfg in sweep.model_configs
        }

        common_configs = {
            "kernel_name": "qwen2vl_mrope",
            "x_name": "model_config",
            "x_label": "model configuration",
            "x_values": [cfg.name for cfg in sweep.model_configs],
            "kernel_providers": ["liger", "huggingface"],
            "extra_benchmark_configs": [{"model_configs": model_configs_info, "seq_len": sweep.seq_len}],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_qwen2vl_mrope_model_config,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_qwen2vl_mrope_model_config,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
    else:
        model = get_benchmark_model_config(args.model)
        probe_seq_len = 2048

        def _probe():
            probe_input = SingleBenchmarkRunInput(
                x=0,
                kernel_provider="huggingface",
                extra_benchmark_config={
                    "hidden_size": model.hidden_size,
                    "num_q_heads": model.num_attention_heads,
                    "num_kv_heads": model.num_key_value_heads,
                    "dtype": model.dtype,
                    "seq_len": probe_seq_len,
                },
            )
            _, _, _, _, fwd_fn = _setup_qwen2vl_mrope(probe_input)
            return fwd_fn()[0]

        config = compute_seq_len_sweep_config(model, probe_fn=_probe, probe_seq_len=probe_seq_len)

        common_configs = {
            "kernel_name": "qwen2vl_mrope",
            "x_name": "T",
            "x_label": "sequence length",
            "x_values": [2**i for i in range(10, int(math.log2(max(1024, config.seq_len))) + 1)],
            "kernel_providers": ["liger", "huggingface"],
            "extra_benchmark_configs": [
                {
                    "hidden_size": model.hidden_size,
                    "num_q_heads": model.num_attention_heads,
                    "num_kv_heads": model.num_key_value_heads,
                    "dtype": model.dtype,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_qwen2vl_mrope,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_qwen2vl_mrope,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
