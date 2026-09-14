"""Standardized Qwen4Exp operator benchmarks."""

import gc
import os
import sys

from types import MethodType

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import QWEN4_EXP
from benchmark_model_configs import build_model_config_sweep
from benchmark_model_configs import build_token_length_sweep
from benchmark_model_configs import get_benchmark_model_config
from test.transformers.test_qwen4_exp import qwen4_exp_eos_aware_ngram_hash_ref
from test.transformers.test_qwen4_exp import qwen4_exp_gr_write_ref
from test.transformers.test_qwen4_exp import qwen4_exp_group_rms_norm_ref
from test.transformers.test_qwen4_exp import qwen4_exp_hyper_connection_pre_ref
from test.transformers.test_qwen4_exp import qwen4_exp_sum_three_grads
from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import parse_benchmark_script_args
from utils import run_benchmarks
from utils import run_memory_benchmark
from utils import run_speed_benchmark

from liger_kernel.ops.qwen4_exp import LigerGroupRMSNormFusedFunction
from liger_kernel.ops.qwen4_exp import LigerGroupRMSNormWrite4Function
from liger_kernel.transformers.functional import liger_qwen4_exp_gr_write
from liger_kernel.transformers.functional import liger_qwen4_exp_hyper_connection_pre
from liger_kernel.transformers.functional import liger_qwen4_exp_ngram_hash
from liger_kernel.transformers.monkey_patch import _patch_rms_norm_module
from liger_kernel.transformers.qwen4_exp import liger_qwen4_exp_gated_residual_forward
from liger_kernel.utils import infer_device

device = infer_device()

_MODEL_KEYS = ["name"]
_GROUPED_RMS_CASES = ("group_rms_multi_consumer", "group_rms_write4")
_ALL_CASES = ("gr_write", "hyper_pre", "gated_residual", "ngram_hash", *_GROUPED_RMS_CASES)


def _qwen4_value(model, name):
    value = getattr(model, name)
    return getattr(QWEN4_EXP, name) if value is None else value


def _resolve_model_and_seq_len(input: SingleBenchmarkRunInput):
    cfg = input.extra_benchmark_config
    if isinstance(input.x, str):
        return MODEL_REGISTRY[input.x], cfg["seq_len"]
    return MODEL_REGISTRY[cfg["name"]], int(input.x)


def _setup_qwen4_exp(input: SingleBenchmarkRunInput):
    """Create one Qwen4 benchmark case from a standardized model config."""
    torch.manual_seed(42)
    model, seq_len = _resolve_model_and_seq_len(input)
    cfg = input.extra_benchmark_config
    # Shared sweep probing runs before the safe benchmark batch size is attached.
    batch_size = cfg.get("bsz", 1)
    provider = input.kernel_provider
    sub_kernel = cfg["sub_kernel"]
    if sub_kernel in _GROUPED_RMS_CASES and provider not in ("liger_rms_norm", "liger"):
        raise ValueError(f"{sub_kernel} providers must be 'liger_rms_norm' or 'liger', got {provider!r}.")
    hidden_size = model.hidden_size
    hc_count = _qwen4_value(model, "hc_count")
    dtype = model.dtype
    backward_fn = None

    if sub_kernel == "gr_write":
        block_output = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=True)
        residual = torch.randn(
            batch_size, seq_len, hc_count * hidden_size, device=device, dtype=dtype, requires_grad=True
        )
        write_logits = torch.randn(batch_size, seq_len, hc_count, device=device, dtype=dtype, requires_grad=True)
        fn = liger_qwen4_exp_gr_write if provider == "liger" else qwen4_exp_gr_write_ref
        fwd_fn = lambda: fn(block_output, residual, write_logits)
        grad_tensors = [block_output, residual, write_logits]
    elif sub_kernel == "hyper_pre":
        shape = (batch_size, seq_len, hc_count * hidden_size)
        mix_logits = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        normalized_input = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        fn = liger_qwen4_exp_hyper_connection_pre if provider == "liger" else qwen4_exp_hyper_connection_pre_ref
        fwd_fn = lambda: fn(mix_logits, normalized_input, hc_count)
        grad_tensors = [mix_logits, normalized_input]
    elif sub_kernel == "gated_residual":
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

        shape = (batch_size, seq_len, hc_count * hidden_size)
        config = Qwen4ExpTextConfig(
            hidden_size=hidden_size,
            hc_count=hc_count,
            rms_norm_eps=model.rms_norm_eps,
        )
        module = Qwen4ExpTextGatedResidual(config, use_combine=True).to(device, dtype)
        hyper_input = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        # Isolate the complete GatedResidual read/write overhead around an arbitrary
        # attention/MLP result without including the intervening block itself.
        block_output = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=True)
        grad_mixed = torch.randn_like(block_output)
        grad_written = torch.randn_like(hyper_input)
        grad_tensors = [hyper_input, block_output, *module.parameters()]

        if provider == "liger":
            _patch_rms_norm_module(module.hc_norm, offset=1.0, casting_mode="gemma", in_place=False)
            module.forward = MethodType(liger_qwen4_exp_gated_residual_forward, module)

            def fwd_fn():
                mixed_input, residual, write_logits = module(hyper_input, return_write_logits=True)
                return mixed_input, liger_qwen4_exp_gr_write(block_output, residual, write_logits)

        else:

            def fwd_fn():
                mixed_input, residual, injection_weights = module(hyper_input)
                injection = block_output.unsqueeze(-2) * injection_weights.unsqueeze(-1)
                return mixed_input, residual + injection.flatten(-2)

        def backward_fn(outputs, retain_graph):
            torch.autograd.backward(outputs, (grad_mixed, grad_written), retain_graph=retain_graph)
    elif sub_kernel == "ngram_hash":
        from transformers.models.qwen4_exp.modeling_qwen4_exp import _build_layer_multipliers
        from transformers.models.qwen4_exp.modeling_qwen4_exp import _find_nth_prime_after

        ngram_size = _qwen4_value(model, "ngram_size")
        heads_per_ngram = _qwen4_value(model, "heads_per_ngram")
        n_heads = (ngram_size - 1) * heads_per_ngram
        eos_token_id = 2
        input_ids = torch.randint(model.vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
        previous_context = torch.randint(
            model.vocab_size,
            (batch_size, ngram_size - 1),
            device=device,
            dtype=torch.long,
        )
        multipliers = _build_layer_multipliers(
            model.vocab_size,
            ngram_size,
            ple_layer_index=0,
            seed=_qwen4_value(model, "ngram_seed"),
        ).to(device)
        vocab_size_base = _qwen4_value(model, "ngram_vocab_size_base")
        vocab_sizes = torch.tensor(
            [_find_nth_prime_after(vocab_size_base - 1, head_idx + 1) for head_idx in range(n_heads)],
            device=device,
            dtype=torch.long,
        )
        offsets = torch.cat([vocab_sizes.new_zeros(1), vocab_sizes.cumsum(0)[:-1]])
        fn = liger_qwen4_exp_ngram_hash if provider == "liger" else qwen4_exp_eos_aware_ngram_hash_ref
        fwd_fn = lambda: fn(previous_context, input_ids, multipliers, vocab_sizes, offsets, eos_token_id)
        grad_tensors = []
    elif sub_kernel == "group_rms_multi_consumer":
        shape = (batch_size * seq_len, hc_count * hidden_size)
        hyper_input = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        rms_weight = (torch.randn(shape[-1], device=device, dtype=dtype) * 0.02).requires_grad_(True)
        grads = [torch.randn_like(hyper_input) for _ in range(3)]
        grad_tensors = [hyper_input, rms_weight]
        if provider == "liger":
            fwd_fn = lambda: LigerGroupRMSNormFusedFunction.apply(
                hyper_input, rms_weight, model.rms_norm_eps, 1.0, "gemma", hc_count
            )

            def backward_fn(outputs, retain_graph):
                torch.autograd.backward(outputs, grads, retain_graph=retain_graph)

        else:
            fwd_fn = lambda: (
                qwen4_exp_group_rms_norm_ref(
                    hyper_input,
                    rms_weight,
                    model.rms_norm_eps,
                    1.0,
                    "gemma",
                    hc_count,
                ),
            )

            def backward_fn(outputs, retain_graph):
                outputs[0].backward(qwen4_exp_sum_three_grads(*grads), retain_graph=retain_graph)

    elif sub_kernel == "group_rms_write4":
        shape = (batch_size * seq_len, hc_count * hidden_size)
        hyper_input = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        rms_weight = (torch.randn(shape[-1], device=device, dtype=dtype) * 0.02).requires_grad_(True)
        write_weight = (torch.randn((hc_count, shape[-1]), device=device, dtype=dtype) * 0.02).requires_grad_(True)
        grads = [
            torch.randn_like(hyper_input),
            torch.randn_like(hyper_input),
            torch.randn((shape[0], hc_count), device=device, dtype=dtype),
        ]
        grad_tensors = [hyper_input, rms_weight, write_weight]
        if provider == "liger":
            fwd_fn = lambda: LigerGroupRMSNormWrite4Function.apply(
                hyper_input,
                rms_weight,
                write_weight,
                model.rms_norm_eps,
                1.0,
                "gemma",
                hc_count,
            )
        else:

            def fwd_fn():
                normalized = qwen4_exp_group_rms_norm_ref(
                    hyper_input, rms_weight, model.rms_norm_eps, 1.0, "gemma", hc_count
                )
                write_logits = torch.mm(normalized, write_weight.transpose(0, 1)) / hc_count
                return normalized, normalized.view(normalized.shape), write_logits

        def backward_fn(outputs, retain_graph):
            torch.autograd.backward(outputs, grads, retain_graph=retain_graph)

    else:
        raise ValueError(f"Unknown Qwen4Exp sub-kernel: {sub_kernel}")

    return fwd_fn, grad_tensors, backward_fn


def _probe_from_setup(fwd_fn, _grad_tensors, _backward_fn):
    output = fwd_fn()
    if isinstance(output, tuple):
        output = output[0]
    if output.is_floating_point():
        return output
    # estimate_kernel_peak_memory always performs backward. The hash output is
    # intentionally non-differentiable, so return a scalar leaf after executing
    # the real forward; peak accounting has already observed the hash allocation.
    return torch.zeros((), device=output.device, requires_grad=True)


def bench_speed_qwen4_exp(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    fwd_fn, grad_tensors, backward_fn = _setup_qwen4_exp(input)
    if backward_fn is not None:
        return _run_multi_output_speed_benchmark(
            fwd_fn,
            backward_fn,
            input.kernel_operation_mode,
            grad_tensors,
        )
    return run_speed_benchmark(fwd_fn, input.kernel_operation_mode, grad_tensors)


def bench_memory_qwen4_exp(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    fwd_fn, grad_tensors, backward_fn = _setup_qwen4_exp(input)
    if backward_fn is not None:
        return _run_multi_output_memory_benchmark(
            fwd_fn,
            backward_fn,
            input.kernel_operation_mode,
            grad_tensors,
            input.extra_benchmark_config.get("memory_kind", "peak"),
        )
    return run_memory_benchmark(fwd_fn, input.kernel_operation_mode)


def _run_multi_output_speed_benchmark(fwd_fn, backward_fn, mode, grad_tensors):
    import triton

    if mode == "forward":
        operation = fwd_fn
    elif mode == "backward":
        outputs = fwd_fn()
        operation = lambda: backward_fn(outputs, True)
    elif mode == "full":

        def operation():
            outputs = fwd_fn()
            backward_fn(outputs, False)

    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'forward', 'backward', or 'full'.")

    ms_50, ms_20, ms_80 = triton.testing.do_bench(
        operation,
        grad_to_none=grad_tensors,
        rep=100,
        quantiles=QUANTILES,
    )
    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def _run_multi_output_memory_benchmark(fwd_fn, backward_fn, mode, grad_tensors, memory_kind):
    if memory_kind not in ("peak", "transient"):
        raise ValueError(f"Unsupported memory kind: {memory_kind}")
    torch_device_module = getattr(torch, device)
    outputs = fwd_fn() if mode == "backward" else None
    samples = []
    for _ in range(10):
        for tensor in grad_tensors:
            tensor.grad = None
        gc.collect()
        torch_device_module.empty_cache()
        torch_device_module.synchronize()
        baseline = torch_device_module.memory_allocated()
        torch_device_module.reset_peak_memory_stats()
        if mode == "forward":
            result = fwd_fn()
        elif mode == "backward":
            backward_fn(outputs, True)
            result = None
        elif mode == "full":
            result = fwd_fn()
            backward_fn(result, False)
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'forward', 'backward', or 'full'.")
        torch_device_module.synchronize()
        peak = torch_device_module.max_memory_allocated()
        samples.append((peak if memory_kind == "peak" else peak - baseline) / 2**20)
        del result
    values = torch.tensor(samples, dtype=torch.float32)
    median, lower, upper = torch.quantile(values, torch.tensor(QUANTILES)).tolist()
    return SingleBenchmarkRunOutput(y_20=lower, y_50=median, y_80=upper)


def _build_common_config(args, sub_kernel):
    extra_configs = {"sub_kernel": sub_kernel}
    probe_provider = "liger_rms_norm" if sub_kernel in _GROUPED_RMS_CASES else "torch"
    if args.sweep_mode == "model_config":
        return build_model_config_sweep(
            kernel_name=f"qwen4_exp_{sub_kernel}",
            all_model_configs=list(MODEL_REGISTRY.values()),
            setup_fn=_setup_qwen4_exp,
            model_keys=_MODEL_KEYS,
            probe_provider=probe_provider,
            extra_configs=extra_configs,
            probe_dim="T",
            forward_fn=_probe_from_setup,
            bt=args.bt,
            overwrite=args.overwrite,
        )

    model = get_benchmark_model_config(args.model or QWEN4_EXP.name)
    return build_token_length_sweep(
        kernel_name=f"qwen4_exp_{sub_kernel}",
        probe_x=min(args.bt, model.max_position_embeddings),
        model=model,
        setup_fn=_setup_qwen4_exp,
        model_keys=_MODEL_KEYS,
        extra_configs=extra_configs,
        scale_dim="T",
        forward_fn=_probe_from_setup,
        probe_provider=probe_provider,
        x_label="sequence length",
        overwrite=args.overwrite,
    )


def main():
    args = parse_benchmark_script_args()
    for sub_kernel in _ALL_CASES:
        common_configs = _build_common_config(args, sub_kernel)
        baseline_provider = "liger_rms_norm" if sub_kernel in _GROUPED_RMS_CASES else "torch"
        common_configs["kernel_providers"] = [baseline_provider, "liger"]
        operation_modes = ["forward"] if sub_kernel == "ngram_hash" else ["forward", "backward", "full"]
        run_benchmarks(
            bench_test_fn=bench_speed_qwen4_exp,
            kernel_operation_modes=operation_modes,
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        memory_kinds = ("peak", "transient") if sub_kernel in _GROUPED_RMS_CASES else (None,)
        for memory_kind in memory_kinds:
            if memory_kind is None:
                memory_configs = common_configs
                metric_name = "memory"
            else:
                memory_configs = {
                    **common_configs,
                    "extra_benchmark_configs": [
                        {**config, "memory_kind": memory_kind} for config in common_configs["extra_benchmark_configs"]
                    ],
                }
                metric_name = f"{memory_kind}_memory"
            run_benchmarks(
                bench_test_fn=bench_memory_qwen4_exp,
                kernel_operation_modes=operation_modes,
                metric_name=metric_name,
                metric_unit="MB",
                **memory_configs,
            )


if __name__ == "__main__":
    main()
