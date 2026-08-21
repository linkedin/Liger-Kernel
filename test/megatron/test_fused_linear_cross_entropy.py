from __future__ import annotations

import tempfile

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from liger_kernel.megatron import LigerMegatronFusedLinearCrossEntropy
from liger_kernel.megatron import liger_megatron_fused_linear_cross_entropy_output_processor
from liger_kernel.ops.megatron_fused_linear_cross_entropy import liger_megatron_fused_linear_cross_entropy


def _reference_loss(hidden, weight, target, bias=None, ignore_index=-100):
    logits = hidden.float() @ weight.float().t()
    if bias is not None:
        logits = logits + bias.float()
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).reshape(target.shape)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron FLCE requires CUDA")
@pytest.mark.parametrize("shape", [(2, 3, 8, 16), (3, 2, 17, 32)])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_megatron_flce_tp1_matches_pytorch(shape, with_bias, dtype):
    s, b, h, v = shape
    torch.manual_seed(42)
    hidden_base = torch.randn(s, b, h, device="cuda", dtype=dtype)
    weight_base = torch.randn(v, h, device="cuda", dtype=dtype) * 0.02
    bias_base = torch.randn(v, device="cuda", dtype=dtype) * 0.02 if with_bias else None
    target = torch.randint(0, v, (s, b), device="cuda")
    target.reshape(-1)[0] = -100
    upstream = torch.randn(s, b, device="cuda")

    hidden_ref = hidden_base.clone().requires_grad_(True)
    weight_ref = weight_base.clone().requires_grad_(True)
    bias_ref = bias_base.clone().requires_grad_(True) if bias_base is not None else None
    hidden_liger = hidden_base.clone().requires_grad_(True)
    weight_liger = weight_base.clone().requires_grad_(True)
    bias_liger = bias_base.clone().requires_grad_(True) if bias_base is not None else None

    reference = _reference_loss(hidden_ref, weight_ref, target, bias_ref)
    actual = liger_megatron_fused_linear_cross_entropy(
        hidden_liger,
        weight_liger,
        target,
        bias=bias_liger,
    )

    torch.testing.assert_close(actual, reference, atol=5e-3, rtol=5e-2)
    reference.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(hidden_liger.grad, hidden_ref.grad, atol=5e-3, rtol=5e-2)
    torch.testing.assert_close(weight_liger.grad, weight_ref.grad, atol=5e-3, rtol=5e-2)
    if with_bias:
        torch.testing.assert_close(bias_liger.grad, bias_ref.grad, atol=5e-3, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron FLCE requires CUDA")
def test_megatron_flce_module_contract():
    module = LigerMegatronFusedLinearCrossEntropy(ignore_index=-1)
    hidden = torch.randn(2, 3, 8, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16) * 0.02
    target = torch.randint(0, 16, (2, 3), device="cuda")
    target[0, 0] = -1

    actual = module(hidden, weight, target)
    reference = _reference_loss(hidden, weight, target, ignore_index=-1)
    torch.testing.assert_close(actual, reference, atol=5e-3, rtol=5e-2)
    assert "ignore_index=-1" in repr(module)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron FLCE requires CUDA")
def test_megatron_flce_output_processor_matches_materialized_path():
    class ColumnParallelLinear:
        def __init__(self, weight, bias):
            self.weight = weight
            self.bias = bias
            self.tp_group = None
            self.gather_output = False
            self.sequence_parallel = False
            self.gradient_accumulation_fusion = False
            self.disable_grad_reduce = False
            self.explicit_expert_comm = False
            self.skip_bias_add = False

    torch.manual_seed(43)
    hidden = torch.randn(3, 2, 8, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16) * 0.02
    bias = torch.randn(16, device="cuda", dtype=torch.bfloat16) * 0.02
    labels = torch.randint(16, (2, 3), device="cuda")
    output_layer = ColumnParallelLinear(weight, bias)
    config = SimpleNamespace(
        defer_embedding_wgrad_compute=False,
        mtp_num_layers=None,
        use_mup=False,
    )

    actual = liger_megatron_fused_linear_cross_entropy_output_processor(
        hidden_states=hidden,
        output_layer=output_layer,
        output_weight=None,
        labels=labels,
        runtime_gather_output=None,
        config=config,
    )
    reference = _reference_loss(hidden, weight, labels.t().contiguous(), bias).t().contiguous()
    torch.testing.assert_close(actual, reference, atol=5e-3, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron FLCE requires CUDA")
def test_megatron_flce_output_processor_rejects_gathered_logits():
    output_layer = SimpleNamespace(
        gather_output=True,
        sequence_parallel=False,
        gradient_accumulation_fusion=False,
        disable_grad_reduce=False,
        explicit_expert_comm=False,
        skip_bias_add=False,
    )
    config = SimpleNamespace(
        defer_embedding_wgrad_compute=False,
        mtp_num_layers=None,
        use_mup=False,
    )

    with pytest.raises(RuntimeError, match="not Megatron's native ColumnParallelLinear.*gathers TP logits"):
        liger_megatron_fused_linear_cross_entropy_output_processor(
            hidden_states=torch.empty(1, 1, 8, device="cuda", dtype=torch.bfloat16),
            output_layer=output_layer,
            output_weight=None,
            labels=torch.zeros(1, 1, device="cuda", dtype=torch.long),
            runtime_gather_output=None,
            config=config,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron FLCE requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_megatron_flce_triton_split_k_dx_matches_pytorch(dtype):
    torch.manual_seed(44)
    hidden_base = torch.randn(8, 4, 64, device="cuda", dtype=dtype)
    weight_base = torch.randn(8192, 64, device="cuda", dtype=dtype) * 0.02
    target = torch.randint(0, weight_base.shape[0], hidden_base.shape[:-1], device="cuda")
    upstream = torch.randn_like(target, dtype=torch.float32)

    hidden_ref = hidden_base.clone().requires_grad_(True)
    weight_ref = weight_base.clone().requires_grad_(True)
    hidden_triton = hidden_base.clone().requires_grad_(True)
    weight_triton = weight_base.clone().requires_grad_(True)

    reference = _reference_loss(hidden_ref, weight_ref, target)
    actual = liger_megatron_fused_linear_cross_entropy(hidden_triton, weight_triton, target)
    torch.testing.assert_close(actual, reference, atol=5e-3, rtol=5e-2)

    reference.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(hidden_triton.grad, hidden_ref.grad, atol=5e-3, rtol=5e-2)
    torch.testing.assert_close(weight_triton.grad, weight_ref.grad, atol=5e-3, rtol=5e-2)


def test_megatron_flce_rejects_cpu_inputs():
    hidden = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    weight = torch.randn(16, 8, dtype=torch.bfloat16)
    target = torch.randint(0, 16, (2, 3))

    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        liger_megatron_fused_linear_cross_entropy(hidden, weight, target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron FLCE requires CUDA")
def test_megatron_flce_rejects_non_long_targets():
    hidden = torch.randn(2, 3, 8, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16)
    target = torch.zeros(2, 3, device="cuda", dtype=torch.int32)

    with pytest.raises(TypeError, match="target must have dtype torch.long"):
        liger_megatron_fused_linear_cross_entropy(hidden, weight, target)


def _tp_worker(rank, world_size, file_name, dtype):
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{file_name}",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    tp_group = dist.group.WORLD
    s, b, h, v_global = 3, 2, 17, 32
    v_local = v_global // world_size

    torch.manual_seed(123)
    hidden_base = torch.randn(s, b, h, device=device, dtype=dtype)
    weight_global = torch.randn(v_global, h, device=device, dtype=dtype) * 0.02
    bias_global = torch.randn(v_global, device=device, dtype=dtype) * 0.02
    target = torch.randint(0, v_global, (s, b), device=device)
    upstream = torch.randn(s, b, device=device)
    target.reshape(-1)[0] = -100
    for tensor in (hidden_base, weight_global, bias_global, target, upstream):
        dist.broadcast(tensor, src=0, group=tp_group)

    start = rank * v_local
    end = start + v_local
    hidden_liger = hidden_base.clone().requires_grad_(True)
    weight_local = weight_global[start:end].clone().requires_grad_(True)
    bias_local = bias_global[start:end].clone().requires_grad_(True)

    actual = liger_megatron_fused_linear_cross_entropy(
        hidden_liger,
        weight_local,
        target,
        bias=bias_local,
        tp_group=tp_group,
    )

    hidden_ref = hidden_base.clone().requires_grad_(True)
    weight_ref = weight_global.clone().requires_grad_(True)
    bias_ref = bias_global.clone().requires_grad_(True)
    reference = _reference_loss(hidden_ref, weight_ref, target, bias_ref)

    torch.testing.assert_close(actual, reference, atol=5e-3, rtol=5e-2)
    actual.backward(upstream)
    reference.backward(upstream)
    torch.testing.assert_close(hidden_liger.grad, hidden_ref.grad, atol=5e-3, rtol=5e-2)
    torch.testing.assert_close(weight_local.grad, weight_ref.grad[start:end], atol=5e-3, rtol=5e-2)
    torch.testing.assert_close(bias_local.grad, bias_ref.grad[start:end], atol=5e-3, rtol=5e-2)
    dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires at least two CUDA GPUs",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_megatron_flce_tp2_matches_global_reference(dtype):
    with tempfile.NamedTemporaryFile() as rendezvous:
        mp.spawn(
            _tp_worker,
            args=(2, rendezvous.name, dtype),
            nprocs=2,
            join=True,
        )
