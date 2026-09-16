import copy
import importlib.util

from types import MethodType

import pytest
import torch
import torch.nn as nn

from test.utils import assert_verbose_allclose
from test.utils import infer_device
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.ops.qwen4_exp import LigerGroupRMSNormFusedFunction
from liger_kernel.ops.qwen4_exp import LigerGroupRMSNormWrite4Function
from liger_kernel.ops.qwen4_exp import LigerQwen4ExpGRWriteFunction
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.utils import is_hip
from liger_kernel.transformers.functional import liger_qwen4_exp_gr_write
from liger_kernel.transformers.functional import liger_qwen4_exp_hyper_connection_pre
from liger_kernel.transformers.functional import liger_qwen4_exp_ngram_hash
from liger_kernel.transformers.monkey_patch import _patch_rms_norm_module
from liger_kernel.transformers.qwen4_exp import liger_qwen4_exp_gated_residual_forward
from liger_kernel.transformers.qwen4_exp import liger_qwen4_exp_ngram_embedding_forward
from liger_kernel.transformers.qwen4_exp import patch_qwen4_exp_text_module_for_hyper_connection
from liger_kernel.transformers.qwen4_exp import patch_qwen4_exp_text_module_for_ngram
from liger_kernel.transformers.qwen4_exp import patch_qwen4_exp_text_module_for_swiglu
from liger_kernel.transformers.rms_norm import LigerRMSNorm

device = infer_device()

requires_qwen4_exp = pytest.mark.skipif(
    importlib.util.find_spec("transformers.models.qwen4_exp") is None,
    reason="Qwen4Exp is unavailable in this Transformers version",
)

pytestmark = pytest.mark.skipif(
    device != "cuda" or is_hip(), reason="Qwen4Exp Triton kernels require an NVIDIA CUDA GPU"
)


def qwen4_exp_ngram_hash_ref(shifted_ids, multipliers, vocab_sizes, offsets):
    ngram_size = shifted_ids.shape[-1]
    n_heads = vocab_sizes.numel()
    heads_per_ngram = n_heads // (ngram_size - 1)
    blocks = []
    for current_ngram_size in range(2, ngram_size + 1):
        mixed_ids = shifted_ids[..., 0] * multipliers[0]
        for position in range(1, current_ngram_size):
            mixed_ids = torch.bitwise_xor(mixed_ids, shifted_ids[..., position] * multipliers[position])
        start = (current_ngram_size - 2) * heads_per_ngram
        end = start + heads_per_ngram
        blocks.append(mixed_ids.unsqueeze(-1).remainder(vocab_sizes[start:end]) + offsets[start:end])
    return torch.cat(blocks, dim=-1)


def qwen4_exp_eos_aware_ngram_hash_ref(previous_context, input_ids, multipliers, vocab_sizes, offsets, eos_token_id):
    token_history = torch.cat([previous_context, input_ids], dim=-1)
    positions = torch.arange(token_history.shape[1], device=token_history.device, dtype=torch.long)
    eos_positions = torch.where(token_history == eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [eos_positions.new_full((token_history.shape[0], 1), -1), previous_eos_inclusive[:, :-1]], dim=1
    )
    position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
    shifted_tokens = []
    for shift in range(previous_context.shape[1] + 1):
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(token_history.shape[0], -1)
        shifted = token_history.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        shifted_tokens.append(torch.where(valid, shifted, eos_token_id))
    shifted_tokens = torch.stack(shifted_tokens, dim=-1)[:, -input_ids.shape[1] :]
    return qwen4_exp_ngram_hash_ref(shifted_tokens, multipliers, vocab_sizes, offsets)


def qwen4_exp_hyper_connection_pre_ref(mix_logits, normalized_input, n_groups):
    hidden_size = normalized_input.shape[-1] // n_groups
    mix = torch.sigmoid(mix_logits).unflatten(-1, (n_groups, hidden_size))
    values = normalized_input.unflatten(-1, (n_groups, hidden_size))
    return (mix * values).mean(dim=-2)


def qwen4_exp_gr_write_ref(block_output, residual, write_logits):
    write_scale = 2 * torch.sigmoid(write_logits)
    injection = block_output.unsqueeze(-2) * write_scale.unsqueeze(-1)
    return residual + injection.flatten(-2)


def qwen4_exp_group_rms_norm_ref(hidden_states, weight, eps, offset, casting_mode, n_groups):
    """Simpler exact baseline using the shared generic grouped-RMS primitive."""
    return LigerRMSNormFunction.apply(
        hidden_states,
        weight,
        eps,
        offset,
        casting_mode,
        False,
        None,
        n_groups,
    )


def qwen4_exp_sum_three_grads(grad0, grad1, grad2):
    """Materialize the explicit left-associated low-precision reference sum."""
    return (grad0 + grad1) + grad2


def _patch_qwen4_exp_candidate(model):
    """Patch one already-constructed candidate without mutating HF classes globally."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextExperts
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextMLP
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm

    for module in model.modules():
        if isinstance(module, Qwen4ExpTextRMSNorm):
            _patch_rms_norm_module(module, offset=1.0, casting_mode="gemma", in_place=False)
        else:
            patch_qwen4_exp_text_module_for_swiglu(module, Qwen4ExpTextMLP, Qwen4ExpTextExperts)
        patch_qwen4_exp_text_module_for_ngram(module, Qwen4ExpTextNGramEmbedding)
        patch_qwen4_exp_text_module_for_hyper_connection(
            module,
            Qwen4ExpTextGatedResidual,
            Qwen4ExpTextDecoderLayer,
        )


def _qwen4_exp_hybrid_config():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    return Qwen4ExpTextConfig(
        vocab_size=101,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts_per_tok=1,
        num_experts=2,
        hc_count=4,
        hc_lowrank=4,
        indexer_n_heads=1,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=4,
        indexer_compress_ratio=2,
        ple_layer_ids=[1],
        ple_embed_dim=8,
        ple_conv_kernel_size=2,
        ngram_size=2,
        heads_per_ngram=1,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=32,
        eos_token_id=2,
        pad_token_id=0,
        layer_types=["linear_attention", "qwen_sparse_attention"],
        attn_implementation="eager",
        use_cache=True,
        tie_word_embeddings=False,
    )


class _NGramIdRecorder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.ids = []

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, ngram_ids):
        self.ids.append(ngram_ids.detach().clone())
        return self.embedding(ngram_ids)


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((128, 4, 1024), torch.float32),
        pytest.param(
            (2048, 4, 1024),
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
            id="profile-bf16",
        ),
        pytest.param(
            (512, 4, 2560),
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
            id="production-bf16",
        ),
    ],
)
def test_qwen4_exp_group_rms_norm_add3_backward(shape, dtype):
    set_seed(42)
    rows, n_groups, group_size = shape
    dim = n_groups * group_size
    x = torch.randn((rows, dim), device=device, dtype=dtype, requires_grad=True)
    weight = (torch.randn(dim, device=device, dtype=dtype) * 0.02).requires_grad_(True)
    grads = [torch.randn_like(x) for _ in range(3)]

    outputs = LigerGroupRMSNormFusedFunction.apply(x, weight, 1e-6, 1.0, "gemma", n_groups)
    assert outputs[0].data_ptr() == outputs[1].data_ptr() == outputs[2].data_ptr()
    torch.autograd.backward(outputs, grads)
    actual_dx = x.grad.detach().clone()
    actual_dw = weight.grad.detach().clone()

    ref_x = x.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    reference = LigerRMSNormFunction.apply(ref_x, ref_weight, 1e-6, 1.0, "gemma", False, None, n_groups)
    reference.backward(qwen4_exp_sum_three_grads(*grads))

    assert_verbose_allclose(outputs[0], reference, atol=0.0, rtol=0.0)
    assert_verbose_allclose(actual_dx, ref_x.grad, atol=0.0, rtol=0.0)
    assert_verbose_allclose(actual_dw, ref_weight.grad, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "rows, used_consumers",
    [
        pytest.param(32, (1,), id="only-middle"),
        pytest.param(32, (1, 2), id="missing-first"),
        pytest.param(512, (0, 2), id="missing-middle-512"),
        pytest.param(2048, (0, 2), id="missing-middle-2048"),
        pytest.param(32, (0, 1), id="missing-last"),
    ],
)
def test_qwen4_exp_group_rms_norm_sparse_consumers_match_explicit_zeros(monkeypatch, rows, used_consumers):
    if not supports_bfloat16():
        pytest.skip("bfloat16 not supported on this GPU")
    import liger_kernel.ops.rms_norm as rms_norm_ops

    set_seed(42)
    n_groups, group_size = 4, 2048
    dim = n_groups * group_size
    x = torch.randn((rows, dim), device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = (torch.randn(dim, device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    grads = [torch.randn_like(x) for _ in range(3)]

    calls = 0
    original_add2 = rms_norm_ops.rms_group_norm_backward_add2

    def record_add2(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_add2(*args, **kwargs)

    monkeypatch.setattr(rms_norm_ops, "rms_group_norm_backward_add2", record_add2)
    outputs = LigerGroupRMSNormFusedFunction.apply(x, weight, 1e-6, 1.0, "gemma", n_groups)
    torch.autograd.backward(
        tuple(outputs[index] for index in used_consumers),
        tuple(grads[index] for index in used_consumers),
    )
    actual_dx = x.grad.detach().clone()
    actual_dw = weight.grad.detach().clone()
    assert calls == (1 if len(used_consumers) == 2 else 0)

    ref_x = x.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    reference = LigerGroupRMSNormFusedFunction.apply(ref_x, ref_weight, 1e-6, 1.0, "gemma", n_groups)
    explicit_grads = [
        grad if index in used_consumers else torch.zeros_like(grads[0]) for index, grad in enumerate(grads)
    ]
    torch.autograd.backward(reference, tuple(explicit_grads))

    assert outputs[0].data_ptr() == outputs[1].data_ptr() == outputs[2].data_ptr()
    assert_verbose_allclose(outputs[0], reference[0], atol=0.0, rtol=0.0)
    assert_verbose_allclose(actual_dx, ref_x.grad, atol=0.0, rtol=0.0)
    assert_verbose_allclose(actual_dw, ref_weight.grad, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "shape",
    [
        (2048, 4, 1024),
        (512, 4, 2560),
        (2048, 4, 2560),
    ],
)
def test_qwen4_exp_group_rms_norm_write4_backward(shape):
    if not supports_bfloat16():
        pytest.skip("bfloat16 not supported on this GPU")
    set_seed(42)
    rows, n_groups, group_size = shape
    dim = n_groups * group_size
    x = torch.randn((rows, dim), device=device, dtype=torch.bfloat16, requires_grad=True)
    rms_weight = (torch.randn(dim, device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    write_weight = (torch.randn((n_groups, dim), device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    grad_down = torch.randn_like(x)
    grad_pre = torch.randn_like(x)
    grad_write = torch.randn((rows, n_groups), device=device, dtype=torch.bfloat16)

    norm_down, norm_pre, write_logits = LigerGroupRMSNormWrite4Function.apply(
        x, rms_weight, write_weight, 1e-6, 1.0, "gemma", n_groups
    )
    torch.autograd.backward((norm_down, norm_pre, write_logits), (grad_down, grad_pre, grad_write))
    actual_grads = (x.grad.detach().clone(), rms_weight.grad.detach().clone(), write_weight.grad.detach().clone())

    ref_x = x.detach().clone().requires_grad_(True)
    ref_rms_weight = rms_weight.detach().clone().requires_grad_(True)
    ref_write_weight = write_weight.detach().clone().requires_grad_(True)
    ref_down, ref_write, ref_pre = LigerGroupRMSNormFusedFunction.apply(
        ref_x, ref_rms_weight, 1e-6, 1.0, "gemma", n_groups
    )
    ref_write_logits = torch.mm(ref_write.view(rows, dim), ref_write_weight.transpose(0, 1)) / n_groups
    torch.autograd.backward((ref_down, ref_pre, ref_write_logits), (grad_down, grad_pre, grad_write))

    assert norm_down.data_ptr() == norm_pre.data_ptr()
    assert_verbose_allclose(norm_down, ref_down, atol=0.0, rtol=0.0)
    assert_verbose_allclose(write_logits, ref_write_logits, atol=0.0, rtol=0.0)
    for actual, expected in zip(actual_grads, (ref_x.grad, ref_rms_weight.grad, ref_write_weight.grad)):
        assert_verbose_allclose(actual, expected, atol=4e-3, rtol=0.0)


@pytest.mark.parametrize("rows", [512, 2048])
def test_qwen4_exp_group_rms_norm_benchmark_reference_parity(rows):
    if not supports_bfloat16():
        pytest.skip("bfloat16 not supported on this GPU")
    set_seed(42)
    n_groups, hidden_size = 4, 2048
    dim = n_groups * hidden_size
    x = torch.randn((rows, dim), device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = (torch.randn(dim, device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    grads = [torch.randn_like(x) for _ in range(3)]

    outputs = LigerGroupRMSNormFusedFunction.apply(x, weight, 1e-6, 1.0, "gemma", n_groups)
    torch.autograd.backward(outputs, grads)

    ref_x = x.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    reference = qwen4_exp_group_rms_norm_ref(ref_x, ref_weight, 1e-6, 1.0, "gemma", n_groups)
    reference.backward(qwen4_exp_sum_three_grads(*grads))

    assert_verbose_allclose(outputs[0], reference, atol=0.0, rtol=0.0)
    assert_verbose_allclose(x.grad, ref_x.grad, atol=0.0, rtol=0.0)
    assert_verbose_allclose(weight.grad, ref_weight.grad, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("rows", [512, 2048])
def test_qwen4_exp_group_rms_norm_write4_benchmark_reference_parity(rows):
    if not supports_bfloat16():
        pytest.skip("bfloat16 not supported on this GPU")
    set_seed(42)
    n_groups, hidden_size = 4, 2048
    dim = n_groups * hidden_size
    x = torch.randn((rows, dim), device=device, dtype=torch.bfloat16, requires_grad=True)
    rms_weight = (torch.randn(dim, device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    write_weight = (torch.randn((n_groups, dim), device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    grads = [
        torch.randn_like(x),
        torch.randn_like(x),
        torch.randn((rows, n_groups), device=device, dtype=torch.bfloat16),
    ]

    outputs = LigerGroupRMSNormWrite4Function.apply(x, rms_weight, write_weight, 1e-6, 1.0, "gemma", n_groups)
    torch.autograd.backward(outputs, grads)

    ref_x = x.detach().clone().requires_grad_(True)
    ref_rms_weight = rms_weight.detach().clone().requires_grad_(True)
    ref_write_weight = write_weight.detach().clone().requires_grad_(True)
    ref_down, ref_for_write, ref_pre = LigerGroupRMSNormFusedFunction.apply(
        ref_x, ref_rms_weight, 1e-6, 1.0, "gemma", n_groups
    )
    ref_write_logits = torch.mm(ref_for_write, ref_write_weight.transpose(0, 1)) / n_groups
    torch.autograd.backward((ref_down, ref_pre, ref_write_logits), grads)

    assert_verbose_allclose(outputs[0], ref_down, atol=0.0, rtol=0.0)
    assert_verbose_allclose(outputs[2], ref_write_logits, atol=0.0, rtol=0.0)
    for actual, expected in zip(
        (x.grad, rms_weight.grad, write_weight.grad),
        (ref_x.grad, ref_rms_weight.grad, ref_write_weight.grad),
    ):
        assert_verbose_allclose(actual, expected, atol=4e-3, rtol=0.0)


@pytest.mark.parametrize(
    "used_consumers",
    [
        pytest.param((0,), id="only-down"),
        pytest.param((1,), id="only-pre"),
        pytest.param((2,), id="only-write"),
        pytest.param((0, 1), id="missing-write"),
        pytest.param((0, 2), id="missing-pre"),
        pytest.param((1, 2), id="missing-down"),
    ],
)
def test_qwen4_exp_group_rms_norm_write4_sparse_consumers(monkeypatch, used_consumers):
    if not supports_bfloat16():
        pytest.skip("bfloat16 not supported on this GPU")
    import liger_kernel.ops.qwen4_exp as qwen4_exp_ops

    fused_calls = 0
    original_fused_backward = qwen4_exp_ops._qwen4_exp_group_rms_norm_backward_write4

    def record_fused_backward(*args, **kwargs):
        nonlocal fused_calls
        fused_calls += 1
        return original_fused_backward(*args, **kwargs)

    monkeypatch.setattr(qwen4_exp_ops, "_qwen4_exp_group_rms_norm_backward_write4", record_fused_backward)
    set_seed(42)
    rows, n_groups, group_size = 32, 4, 64
    dim = n_groups * group_size
    inputs = [
        torch.randn((rows, dim), device=device, dtype=torch.bfloat16, requires_grad=True),
        (torch.randn(dim, device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True),
        (torch.randn((n_groups, dim), device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True),
    ]
    grads = [
        torch.randn_like(inputs[0]),
        torch.randn_like(inputs[0]),
        torch.randn((rows, n_groups), device=device, dtype=torch.bfloat16),
    ]
    outputs = LigerGroupRMSNormWrite4Function.apply(*inputs, 1e-6, 1.0, "gemma", n_groups)
    torch.autograd.backward(
        tuple(outputs[index] for index in used_consumers),
        tuple(grads[index] for index in used_consumers),
    )
    actual = [None if tensor.grad is None else tensor.grad.detach().clone() for tensor in inputs]
    assert fused_calls == 0

    reference_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    ref_down, ref_write, ref_pre = LigerGroupRMSNormFusedFunction.apply(
        reference_inputs[0], reference_inputs[1], 1e-6, 1.0, "gemma", n_groups
    )
    ref_write_logits = torch.mm(ref_write, reference_inputs[2].transpose(0, 1)) / n_groups
    reference_outputs = (ref_down, ref_pre, ref_write_logits)
    torch.autograd.backward(
        tuple(reference_outputs[index] for index in used_consumers),
        tuple(grads[index] for index in used_consumers),
    )
    for value, expected in zip(actual, reference_inputs):
        if expected.grad is None:
            assert value is None
        else:
            assert_verbose_allclose(value, expected.grad, atol=4e-3, rtol=0.0)


@pytest.mark.parametrize(
    "used_consumers", [(3,), (0, 3), (1, 3), (2, 3), (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1, 2), (0, 1, 2, 3)]
)
@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_qwen4_exp_write4_residual_consumers(used_consumers):
    """An unused residual stays undefined; residual-only use leaves Parameter grads undefined."""
    set_seed(43)
    inputs = [
        torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
        for shape in ((32, 256), (256,), (4, 256))
    ]
    reference_inputs = [value.detach().clone().requires_grad_() for value in inputs]
    actual = LigerGroupRMSNormWrite4Function.apply(*inputs, 1e-6, 1.0, "gemma", 4, True)
    reference = (*LigerGroupRMSNormWrite4Function.apply(*reference_inputs, 1e-6, 1.0, "gemma", 4), reference_inputs[0])
    gradients = [torch.randn_like(value) for value in reference]
    saved_gradients = [value.clone() for value in gradients]
    for values in (actual, reference):
        torch.autograd.backward(tuple(values[i] for i in used_consumers), tuple(gradients[i] for i in used_consumers))
    for value, expected in zip(actual, reference):
        assert torch.equal(value, expected)
    for value, expected in zip(inputs, reference_inputs):
        assert (value.grad is None) == (expected.grad is None)
        if value.grad is not None:
            assert torch.equal(value.grad, expected.grad)
    for value, expected in zip(gradients, saved_gradients):
        assert torch.equal(value, expected)


def _qwen4_exp_residual_join_boundary(module, x, block, fuse_residual):
    if fuse_residual:
        mixed, residual, logits = liger_qwen4_exp_gated_residual_forward(module, x, return_write_logits=True)
    else:
        # The pre-fusion production boundary: Write4 has three outputs and x bypasses it.
        down, pre, logits = LigerGroupRMSNormWrite4Function.apply(
            x,
            module.hc_norm.weight,
            module.block_inject_weight.weight,
            module.hc_norm.variance_epsilon,
            1.0,
            "gemma",
            4,
        )
        hidden = torch.nn.functional.silu(module.input_mix_weight_down(down) / 4)
        mixed = liger_qwen4_exp_hyper_connection_pre(module.input_mix_weight_up(hidden), pre, 4)
        residual = x
    output = LigerQwen4ExpGRWriteFunction.apply(block, residual, logits)
    return mixed, output


@requires_qwen4_exp
@pytest.mark.parametrize("rows,noncontiguous", [(512, False), (2048, False), (32, True)])
@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_qwen4_exp_residual_join_full_boundary(rows, noncontiguous):
    """Protect original residual storage, exact full-boundary gradients, hooks and accumulation."""
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    set_seed(44)
    config = Qwen4ExpTextConfig(hidden_size=2560, hc_count=4, hc_lowrank=320)
    module = Qwen4ExpTextGatedResidual(config).to(device, torch.bfloat16)
    _patch_rms_norm_module(module.hc_norm, offset=1.0, casting_mode="gemma", in_place=False)
    reference_module = copy.deepcopy(module)
    x = torch.randn(1, rows, 10240 * (2 if noncontiguous else 1), device=device, dtype=torch.bfloat16)
    if noncontiguous:
        x = x[..., 1::2]
    x.requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    block = torch.randn(1, rows, 2560, device=device, dtype=torch.bfloat16, requires_grad=True)
    reference_block = block.detach().clone().requires_grad_()
    _, residual, _ = liger_qwen4_exp_gated_residual_forward(module, x, return_write_logits=True)
    assert residual.data_ptr() == x.data_ptr()
    assert residual.stride() == x.stride()
    assert residual.storage_offset() == x.storage_offset()
    assert torch.equal(residual, x)
    # Check immutability without a protective copy on the production shapes, and exercise
    # ensure_contiguous in both custom backwards on the strided-input case.
    gm = torch.randn(1, rows, 5120, device=device, dtype=torch.bfloat16)[..., ::2]
    go = torch.randn(1, rows, 20480, device=device, dtype=torch.bfloat16)[..., ::2]
    if not noncontiguous:
        gm, go = gm.contiguous(), go.contiguous()
    saved_gm, saved_go = gm.clone(), go.clone()
    hooks, ref_hooks = [], []
    x.register_hook(lambda grad: hooks.append(grad.clone()))
    reference_x.register_hook(lambda grad: ref_hooks.append(grad.clone()))
    parameter_hooks, ref_parameter_hooks = {}, {}
    for target, captured in ((module, parameter_hooks), (reference_module, ref_parameter_hooks)):
        for name, parameter in target.named_parameters():
            parameter.register_hook(
                lambda grad, name=name, captured=captured: captured.setdefault(name, []).append(grad.clone())
            )
    optimizers = [
        torch.optim.AdamW(target.parameters(), lr=1e-4, foreach=True) for target in (module, reference_module)
    ]
    for _ in range(2):
        actual = _qwen4_exp_residual_join_boundary(module, x, block, True)
        reference = _qwen4_exp_residual_join_boundary(reference_module, reference_x, reference_block, False)
        for value, expected in zip(actual, reference):
            assert torch.equal(value, expected)
        for repeat in range(2):
            torch.autograd.backward(actual, (gm, go), retain_graph=repeat == 0)
            torch.autograd.backward(reference, (gm, go), retain_graph=repeat == 0)
            assert torch.equal(x.grad, reference_x.grad)
            assert torch.equal(block.grad, reference_block.grad)
            for parameter, expected in zip(module.parameters(), reference_module.parameters()):
                assert torch.equal(parameter.grad, expected.grad)
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for parameter, expected in zip(module.parameters(), reference_module.parameters()):
            assert torch.equal(parameter, expected)
    assert torch.equal(gm, saved_gm) and torch.equal(go, saved_go)
    assert len(hooks) == len(ref_hooks) == 4
    for value, expected in zip(hooks, ref_hooks):
        assert torch.equal(value, expected)
    for name in parameter_hooks:
        assert len(parameter_hooks[name]) == len(ref_parameter_hooks[name]) == 4
        for value, expected in zip(parameter_hooks[name], ref_parameter_hooks[name]):
            assert torch.equal(value, expected)


@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_qwen4_exp_residual_join_rounds_rms_gradient_before_add():
    """Cancellation must happen after BF16 rounding, not against the FP32 RMS result."""
    set_seed(45)
    x = torch.randn(32, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, device=device, dtype=torch.bfloat16, requires_grad=True)
    write_weight = torch.randn(4, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    outputs = LigerGroupRMSNormWrite4Function.apply(x, weight, write_weight, 1e-6, 1.0, "gemma", 4)
    gradients = tuple(torch.randn_like(value) for value in outputs)
    expected = torch.autograd.grad(outputs, (x, weight, write_weight), gradients)
    residual_gradient = -expected[0]
    fused = LigerGroupRMSNormWrite4Function.apply(x, weight, write_weight, 1e-6, 1.0, "gemma", 4, True)
    actual = torch.autograd.grad(fused, (x, weight, write_weight), (*gradients, residual_gradient))
    assert torch.count_nonzero(expected[0]) > 0
    assert torch.count_nonzero(actual[0]) == 0
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])


@requires_qwen4_exp
def test_qwen4_exp_residual_join_module_hooks():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    set_seed(47)
    module = Qwen4ExpTextGatedResidual(Qwen4ExpTextConfig(hidden_size=64, hc_count=4, hc_lowrank=16)).to(
        device, torch.bfloat16
    )
    _patch_rms_norm_module(module.hc_norm, offset=1.0, casting_mode="gemma", in_place=False)
    module.forward = MethodType(liger_qwen4_exp_gated_residual_forward, module)
    x = torch.randn(1, 32, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    block = torch.randn(1, 32, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    reference_module = copy.deepcopy(module)
    reference_x, reference_block = [value.detach().clone().requires_grad_() for value in (x, block)]
    forward_calls, backward_grads = [], []
    module.register_forward_hook(lambda mod, inputs, outputs: forward_calls.append(outputs[1].data_ptr()))
    module.register_full_backward_hook(
        lambda mod, grad_input, grad_output: backward_grads.append(grad_input[0].clone())
    )
    mixed, residual, logits = module(x, return_write_logits=True)
    actual = (mixed, LigerQwen4ExpGRWriteFunction.apply(block, residual, logits))
    reference = _qwen4_exp_residual_join_boundary(reference_module, reference_x, reference_block, False)
    gradients = tuple(torch.randn_like(value) for value in actual)
    torch.autograd.backward(actual, gradients)
    torch.autograd.backward(reference, gradients)
    assert forward_calls == [x.data_ptr()]
    assert len(backward_grads) == 1
    assert torch.equal(backward_grads[0], reference_x.grad)
    for value, expected in zip(
        (x, block, *module.parameters()), (reference_x, reference_block, *reference_module.parameters())
    ):
        assert torch.equal(value.grad, expected.grad)


@requires_qwen4_exp
@pytest.mark.parametrize("execution", ["compile", "checkpoint", "checkpoint-reentrant"])
def test_qwen4_exp_residual_join_execution_modes(execution):
    from torch.utils.checkpoint import checkpoint
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    set_seed(46)
    config = Qwen4ExpTextConfig(hidden_size=64, hc_count=4, hc_lowrank=16)
    module = Qwen4ExpTextGatedResidual(config).to(device, torch.bfloat16)
    _patch_rms_norm_module(module.hc_norm, offset=1.0, casting_mode="gemma", in_place=False)
    reference_module = copy.deepcopy(module)
    x = torch.randn(1, 32, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    block = torch.randn(1, 32, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    reference_x, reference_block = [value.detach().clone().requires_grad_() for value in (x, block)]

    def candidate(a, b):
        return _qwen4_exp_residual_join_boundary(module, a, b, True)

    def baseline(a, b):
        return _qwen4_exp_residual_join_boundary(reference_module, a, b, False)

    if execution == "compile":
        torch.compiler.reset()
        candidate = torch.compile(candidate, fullgraph=True)
        baseline = torch.compile(baseline, fullgraph=True)
        actual, reference = candidate(x, block), baseline(reference_x, reference_block)
    else:
        reentrant = execution == "checkpoint-reentrant"
        actual = checkpoint(candidate, x, block, use_reentrant=reentrant)
        reference = checkpoint(baseline, reference_x, reference_block, use_reentrant=reentrant)
    gradients = tuple(torch.randn_like(value) for value in actual)
    torch.autograd.backward(actual, gradients)
    torch.autograd.backward(reference, gradients)
    for value, expected in zip(actual, reference):
        assert torch.equal(value, expected)
    for value, expected in zip(
        (x, block, *module.parameters()), (reference_x, reference_block, *reference_module.parameters())
    ):
        assert torch.equal(value.grad, expected.grad)
    if execution == "compile":
        torch.compiler.reset()


@requires_qwen4_exp
def test_qwen4_exp_hyper_connection_native_rms_norm_epsilon_forward_backward():
    """The fused HyperConnection must honor runtime updates to native ``hc_norm.eps``."""
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    set_seed(42)
    config = Qwen4ExpTextConfig(hidden_size=32, hc_count=4, hc_lowrank=8, rms_norm_eps=1e-5)
    reference_module = Qwen4ExpTextGatedResidual(config).to(device)
    liger_module = copy.deepcopy(reference_module)
    _patch_rms_norm_module(liger_module.hc_norm, offset=1.0, casting_mode="gemma", in_place=False)
    reference_module.hc_norm.eps = 2e-4
    liger_module.hc_norm.eps = 2e-4
    assert reference_module.hc_norm.eps == liger_module.hc_norm.eps == 2e-4
    # Instance patching adds the internal compatibility field. Keeping it deliberately stale
    # proves the public HF ``eps`` attribute is the runtime source of truth.
    assert liger_module.hc_norm.variance_epsilon == config.rms_norm_eps

    # Small magnitudes make the configured epsilon observable and ensure that a
    # silent fallback to 1e-6 fails this regression test by a wide margin.
    reference_input = (torch.randn(2, 7, 4 * config.hidden_size, device=device) * 3e-3).requires_grad_(True)
    liger_input = reference_input.detach().clone().requires_grad_(True)
    reference = reference_module(reference_input)
    output = liger_qwen4_exp_gated_residual_forward(liger_module, liger_input)
    grad_mixed = torch.randn_like(reference[0])
    grad_injection = torch.randn_like(reference[2])
    torch.autograd.backward((reference[0], reference[2]), (grad_mixed, grad_injection))
    torch.autograd.backward((output[0], output[2]), (grad_mixed, grad_injection))

    assert_verbose_allclose(output[0], reference[0], atol=2e-5, rtol=2e-5)
    assert_verbose_allclose(output[2], reference[2], atol=2e-5, rtol=2e-5)
    assert_verbose_allclose(liger_input.grad, reference_input.grad, atol=2e-5, rtol=2e-5)
    for (actual_name, actual), (expected_name, expected) in zip(
        liger_module.named_parameters(), reference_module.named_parameters()
    ):
        assert actual_name == expected_name
        assert_verbose_allclose(actual.grad, expected.grad, atol=2e-5, rtol=2e-5)


@requires_qwen4_exp
def test_qwen4_exp_hyper_connection_keeps_native_rms_norm_active_when_liger_rms_is_disabled():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    set_seed(42)
    config = Qwen4ExpTextConfig(hidden_size=32, hc_count=4, hc_lowrank=8, rms_norm_eps=1e-5)
    reference_module = Qwen4ExpTextGatedResidual(config).to(device)
    liger_module = copy.deepcopy(reference_module)
    # Incidental compatibility attributes must not opt a native HF norm into the fused grouped path.
    for name, value in (("casting_mode", "gemma"), ("in_place", False), ("offset", 1.0), ("row_mode", None)):
        setattr(liger_module.hc_norm, name, value)
    assert not getattr(liger_module.hc_norm, "_liger_rms_norm_patched", False)

    norm_calls = 0
    native_norm_forward = liger_module.hc_norm.forward

    def record_native_norm(_self, hidden_states):
        nonlocal norm_calls
        norm_calls += 1
        return native_norm_forward(hidden_states)

    liger_module.hc_norm.forward = MethodType(record_native_norm, liger_module.hc_norm)
    liger_module.forward = MethodType(liger_qwen4_exp_gated_residual_forward, liger_module)

    reference_input = torch.randn(2, 7, 4 * config.hidden_size, device=device, requires_grad=True)
    liger_input = reference_input.detach().clone().requires_grad_(True)
    reference = reference_module(reference_input)
    output = liger_module(liger_input)
    assert norm_calls == 1

    grad_mixed = torch.randn_like(reference[0])
    grad_injection = torch.randn_like(reference[2])
    torch.autograd.backward((reference[0], reference[2]), (grad_mixed, grad_injection))
    torch.autograd.backward((output[0], output[2]), (grad_mixed, grad_injection))

    assert_verbose_allclose(output[0], reference[0], atol=2e-5, rtol=2e-5)
    assert_verbose_allclose(output[1], reference[1], atol=0.0, rtol=0.0)
    assert_verbose_allclose(output[2], reference[2], atol=2e-5, rtol=2e-5)
    assert_verbose_allclose(liger_input.grad, reference_input.grad, atol=2e-5, rtol=2e-5)
    reference_parameters = dict(reference_module.named_parameters())
    for name, parameter in liger_module.named_parameters():
        assert_verbose_allclose(parameter.grad, reference_parameters[name].grad, atol=2e-5, rtol=2e-5)


@requires_qwen4_exp
@pytest.mark.parametrize(
    "shape, group_size",
    [
        ((2, 3, 32), None),
        ((2, 3, 128), 32),
        ((2, 3, 96), 32),
        ((2, 3, 4, 32), None),
        ((7, 32), None),
    ],
    ids=["normal", "grouped-4", "grouped-3", "qsa-query-4d", "pooled-key-2d"],
)
@pytest.mark.parametrize(
    "dtype, atol",
    [
        (torch.float32, 2e-5),
        pytest.param(
            torch.bfloat16,
            2e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_qwen4_exp_rms_norm_shape_api_native_liger_forward_backward(shape, group_size, dtype, atol):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm

    from liger_kernel.transformers.rms_norm import LigerRMSNormForQwen4Exp

    set_seed(42)
    dim = shape[-1]
    native_module = Qwen4ExpTextRMSNorm(dim, group_size=group_size, eps=1e-5).to(device, dtype)
    liger_module = LigerRMSNormForQwen4Exp(dim, group_size=group_size, eps=1e-5).to(device, dtype)
    with torch.no_grad():
        native_module.weight.normal_(mean=0.0, std=0.02)
        liger_module.weight.copy_(native_module.weight)
    assert native_module.group_size == liger_module.group_size == group_size
    assert native_module.eps == liger_module.eps == liger_module.variance_epsilon == 1e-5
    native_module.eps = 2e-4
    liger_module.eps = 2e-4
    assert native_module.eps == liger_module.eps == liger_module.variance_epsilon == 2e-4
    native_input = torch.randn(shape, device=device, dtype=dtype).requires_grad_(True)
    liger_input = native_input.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(native_input)
    native = native_module(native_input)
    output = liger_module(liger_input)
    native.backward(grad_output)
    output.backward(grad_output)
    assert torch.isfinite(output).all()
    assert torch.isfinite(liger_input.grad).all()
    assert torch.isfinite(liger_module.weight.grad).all()
    assert_verbose_allclose(output, native, atol=atol, rtol=atol)
    assert_verbose_allclose(liger_input.grad, native_input.grad, atol=atol, rtol=atol)
    assert_verbose_allclose(liger_module.weight.grad, native_module.weight.grad, atol=atol, rtol=atol)


@requires_qwen4_exp
@pytest.mark.parametrize("use_combine", [True, False], ids=["decoder-combine", "final-mixer"])
@pytest.mark.parametrize(
    "dtype, atol",
    [
        (torch.float32, 2e-6),
        pytest.param(
            torch.bfloat16,
            8e-3,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_qwen4_exp_gated_residual_native_liger_forward_backward(use_combine, dtype, atol):
    """Compare genuine native HF GatedResidual against the instance-patched Liger path."""
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    set_seed(42)
    config = Qwen4ExpTextConfig(hidden_size=32, hc_count=4, hc_lowrank=8, rms_norm_eps=1e-5)
    reference_module = Qwen4ExpTextGatedResidual(config, use_combine=use_combine).to(device, dtype)
    liger_module = copy.deepcopy(reference_module)
    _patch_rms_norm_module(liger_module.hc_norm, offset=1.0, casting_mode="gemma", in_place=False)
    liger_module.forward = MethodType(liger_qwen4_exp_gated_residual_forward, liger_module)
    assert reference_module.forward.__func__ is Qwen4ExpTextGatedResidual.forward
    assert liger_module.forward.__func__ is liger_qwen4_exp_gated_residual_forward

    reference_input = torch.randn(
        2,
        7,
        config.hc_count * config.hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    liger_input = reference_input.detach().clone().requires_grad_(True)
    reference = reference_module(reference_input)
    output = liger_module(liger_input, return_write_logits=use_combine)

    if use_combine:
        reference_mixed, _, reference_injection = reference
        mixed, _, write_logits = output
        if dtype == torch.bfloat16:
            assert type(write_logits.grad_fn).__name__ == "LigerGroupRMSNormWrite4FunctionBackward"
        else:
            assert type(write_logits.grad_fn).__name__ != "LigerGroupRMSNormWrite4FunctionBackward"
        injection = 2 * torch.sigmoid(write_logits)
        grad_mixed = torch.randn_like(reference_mixed)
        grad_injection = torch.randn_like(reference_injection)
        torch.autograd.backward((reference_mixed, reference_injection), (grad_mixed, grad_injection))
        torch.autograd.backward((mixed, injection), (grad_mixed, grad_injection))
        forward_pairs = (("mixed", mixed, reference_mixed), ("injection", injection, reference_injection))
    else:
        grad_mixed = torch.randn_like(reference)
        reference.backward(grad_mixed)
        output.backward(grad_mixed)
        forward_pairs = (("mixed", output, reference),)

    compared = [*forward_pairs, ("input.grad", liger_input.grad, reference_input.grad)]
    reference_parameters = dict(reference_module.named_parameters())
    compared.extend(
        (f"{name}.grad", parameter.grad, reference_parameters[name].grad)
        for name, parameter in liger_module.named_parameters()
    )
    for name, actual, expected in compared:
        max_abs = (actual.float() - expected.float()).abs().max().item()
        cosine = torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
        print(f"{name}: max_abs={max_abs:.8g}, cosine={cosine.item():.8g}")
        assert max_abs <= atol
        assert cosine >= (0.9998 if dtype == torch.bfloat16 else 0.99999)


@requires_qwen4_exp
def test_qwen4_exp_gated_residual_raw_write_logits_reject_unsupported_placement():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    from liger_kernel.transformers.qwen4_exp import patch_qwen4_exp_text_module_for_hyper_connection

    config = Qwen4ExpTextConfig(hidden_size=8, hc_count=4, hc_lowrank=4, rms_norm_eps=1e-5)
    module = Qwen4ExpTextGatedResidual(config, use_combine=True)
    patch_qwen4_exp_text_module_for_hyper_connection(
        module,
        Qwen4ExpTextGatedResidual,
        Qwen4ExpTextDecoderLayer,
    )
    hyper_input = torch.randn(2, 3, config.hc_count * config.hidden_size)

    with pytest.raises(RuntimeError, match="decoder must fall back before requesting raw write logits"):
        module(hyper_input, return_write_logits=True)


@requires_qwen4_exp
@pytest.mark.parametrize("hc_count", [3, 4], ids=["native-count", "grouped-fusion"])
def test_qwen4_exp_gated_residual_raw_write_logits_requires_combine_mode(hc_count):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    config = Qwen4ExpTextConfig(hidden_size=8, hc_count=hc_count, hc_lowrank=4, rms_norm_eps=1e-5)
    module = Qwen4ExpTextGatedResidual(config, use_combine=False).to(device)
    _patch_rms_norm_module(module.hc_norm, offset=1.0, casting_mode="gemma", in_place=False)
    module.forward = MethodType(liger_qwen4_exp_gated_residual_forward, module)
    hyper_input = torch.randn(2, 3, hc_count * config.hidden_size, device=device)

    with pytest.raises(RuntimeError, match="return_write_logits=True requires block_inject_weight"):
        module(hyper_input, return_write_logits=True)


@requires_qwen4_exp
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Qwen4Exp Accelerate offload regression requires NVIDIA CUDA",
)
@pytest.mark.parametrize("patch_first", [True, False], ids=["liger-then-accelerate", "accelerate-then-liger"])
def test_qwen4_exp_gated_residual_preserves_accelerate_offload_wrapper(patch_first):
    accelerate_hooks = pytest.importorskip("accelerate.hooks")
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    set_seed(42)
    config = Qwen4ExpTextConfig(hidden_size=16, hc_count=4, hc_lowrank=4, rms_norm_eps=1e-5)
    reference = Qwen4ExpTextGatedResidual(config, use_combine=True).to(device, torch.bfloat16)
    module = copy.deepcopy(reference)
    _patch_rms_norm_module(module.hc_norm, offset=1.0, casting_mode="gemma", in_place=False)
    weights_map = {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}

    if patch_first:
        patch_qwen4_exp_text_module_for_hyper_connection(
            module,
            Qwen4ExpTextGatedResidual,
            Qwen4ExpTextDecoderLayer,
        )
    accelerate_hooks.attach_align_device_hook(
        module,
        execution_device=torch.device("cuda", torch.cuda.current_device()),
        offload=True,
        weights_map=weights_map,
        preload_module_classes=[type(module).__name__],
    )
    accelerate_forward = module.forward
    if not patch_first:
        patch_qwen4_exp_text_module_for_hyper_connection(
            module,
            Qwen4ExpTextGatedResidual,
            Qwen4ExpTextDecoderLayer,
        )
        assert module.forward is accelerate_forward

    assert getattr(module._old_forward, "__func__", module._old_forward) is liger_qwen4_exp_gated_residual_forward
    assert {parameter.device.type for parameter in module.parameters()} == {"meta"}

    for _ in range(2):
        hyper_input = torch.randn(
            2,
            5,
            config.hc_count * config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        expected_mixed, expected_residual, expected_injection = reference(hyper_input)
        mixed, residual, write_logits = module(hyper_input, return_write_logits=True)
        assert_verbose_allclose(mixed, expected_mixed, atol=8e-3, rtol=2e-3)
        assert torch.equal(residual, expected_residual)
        assert_verbose_allclose(2 * torch.sigmoid(write_logits), expected_injection, atol=8e-3, rtol=2e-3)
        assert {parameter.device.type for parameter in module.parameters()} == {"meta"}


@requires_qwen4_exp
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Qwen4Exp Accelerate offload regression requires NVIDIA CUDA",
)
@pytest.mark.parametrize("module_kind", ["mlp", "experts"])
@pytest.mark.parametrize("patch_first", [True, False], ids=["liger-then-accelerate", "accelerate-then-liger"])
def test_qwen4_exp_swiglu_preserves_accelerate_offload_wrapper(module_kind, patch_first):
    accelerate_hooks = pytest.importorskip("accelerate.hooks")
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextExperts
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextMLP

    set_seed(42)
    config = Qwen4ExpTextConfig(
        hidden_size=32,
        intermediate_size=16,
        moe_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=1,
        hidden_act="silu",
        experts_implementation="batched_mm" if module_kind == "experts" else "eager",
    )
    module_class = Qwen4ExpTextMLP if module_kind == "mlp" else Qwen4ExpTextExperts
    reference = module_class(config).to(device, torch.bfloat16)
    module = copy.deepcopy(reference)
    weights_map = {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}

    if patch_first:
        patch_qwen4_exp_text_module_for_swiglu(module, Qwen4ExpTextMLP, Qwen4ExpTextExperts)
    accelerate_hooks.attach_align_device_hook(
        module,
        execution_device=torch.device("cuda", torch.cuda.current_device()),
        offload=True,
        weights_map=weights_map,
        preload_module_classes=[type(module).__name__],
    )
    accelerate_forward = module.forward
    if not patch_first:
        patch_qwen4_exp_text_module_for_swiglu(module, Qwen4ExpTextMLP, Qwen4ExpTextExperts)
        assert module.forward is accelerate_forward

    assert {parameter.device.type for parameter in module.parameters()} == {"meta"}
    for _ in range(2):
        hidden_states = torch.randn(5, config.hidden_size, device=device, dtype=torch.bfloat16)
        if module_kind == "mlp":
            expected = reference(hidden_states)
            actual = module(hidden_states)
        else:
            top_k_index = torch.tensor([[0], [1], [2], [3], [0]], device=device)
            top_k_weights = torch.ones(5, 1, device=device, dtype=torch.bfloat16)
            expected = reference(hidden_states, top_k_index, top_k_weights)
            actual = module(hidden_states, top_k_index, top_k_weights)
        assert_verbose_allclose(actual, expected, atol=8e-3, rtol=2e-3)
        assert {parameter.device.type for parameter in module.parameters()} == {"meta"}


@requires_qwen4_exp
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Qwen4Exp Accelerate offload regression requires NVIDIA CUDA",
)
@pytest.mark.parametrize("offloaded_child", ["attn_hyper_connection", "mlp_hyper_connection"])
@pytest.mark.parametrize("patch_first", [True, False], ids=["liger-then-accelerate", "accelerate-then-liger"])
def test_qwen4_exp_decoder_accelerate_offload_keeps_whole_decoder_fallback(monkeypatch, offloaded_child, patch_first):
    accelerate_hooks = pytest.importorskip("accelerate.hooks")
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    class NativeAttention(torch.nn.Module):
        def forward(self, hidden_states, position_embeddings, **kwargs):
            return hidden_states * 0.5, None

    def make_decoder(config):
        decoder = object.__new__(Qwen4ExpTextDecoderLayer)
        torch.nn.Module.__init__(decoder)
        decoder.ple = None
        decoder.attn_hyper_connection = Qwen4ExpTextGatedResidual(config, use_combine=True)
        decoder.mlp_hyper_connection = Qwen4ExpTextGatedResidual(config, use_combine=True)
        decoder.layer_type = "full_attention"
        decoder.self_attn = NativeAttention()
        decoder.mlp = torch.nn.Identity()
        return decoder.to(device, torch.bfloat16)

    set_seed(42)
    config = Qwen4ExpTextConfig(hidden_size=16, hc_count=4, hc_lowrank=4, rms_norm_eps=1e-5)
    reference = make_decoder(config)
    decoder = copy.deepcopy(reference)
    for current_decoder in (reference, decoder):
        for hyper_connection in (current_decoder.attn_hyper_connection, current_decoder.mlp_hyper_connection):
            _patch_rms_norm_module(hyper_connection.hc_norm, offset=1.0, casting_mode="gemma", in_place=False)
    for name in ("attn_hyper_connection", "mlp_hyper_connection"):
        if name == offloaded_child:
            continue
        hyper_connection = getattr(reference, name)
        patch_qwen4_exp_text_module_for_hyper_connection(
            hyper_connection,
            Qwen4ExpTextGatedResidual,
            Qwen4ExpTextDecoderLayer,
        )

    module_to_offload = getattr(decoder, offloaded_child)
    weights_map = {name: tensor.detach().cpu().clone() for name, tensor in module_to_offload.state_dict().items()}
    modules_to_patch = (decoder.attn_hyper_connection, decoder.mlp_hyper_connection, decoder)
    if patch_first:
        for module in modules_to_patch:
            patch_qwen4_exp_text_module_for_hyper_connection(
                module,
                Qwen4ExpTextGatedResidual,
                Qwen4ExpTextDecoderLayer,
            )
    accelerate_hooks.attach_align_device_hook(
        module_to_offload,
        execution_device=torch.device("cuda", torch.cuda.current_device()),
        offload=True,
        weights_map=weights_map,
    )
    if not patch_first:
        for module in modules_to_patch:
            patch_qwen4_exp_text_module_for_hyper_connection(
                module,
                Qwen4ExpTextGatedResidual,
                Qwen4ExpTextDecoderLayer,
            )

    native_decoder_calls = 0
    native_decoder_forward = decoder.__dict__["_liger_qwen4_exp_native_decoder_forward"]

    def record_native_decoder_forward(module, *args, **kwargs):
        nonlocal native_decoder_calls
        native_decoder_calls += 1
        return native_decoder_forward(module, *args, **kwargs)

    decoder.__dict__["_liger_qwen4_exp_native_decoder_forward"] = record_native_decoder_forward
    monkeypatch.setattr(
        LigerQwen4ExpGRWriteFunction,
        "apply",
        lambda *args, **kwargs: pytest.fail("Liger GRWrite ran despite whole-decoder fallback"),
    )
    position_embeddings = (torch.empty(0, device=device), torch.empty(0, device=device))
    for _ in range(2):
        hidden_states = torch.randn(
            2,
            5,
            config.hc_count * config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        expected = reference(hidden_states, position_embeddings)
        output = decoder(hidden_states, position_embeddings)
        assert_verbose_allclose(output, expected, atol=8e-3, rtol=2e-3)
        assert {parameter.device.type for parameter in module_to_offload.parameters()} == {"meta"}
    assert native_decoder_calls == 2


@requires_qwen4_exp
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="mixed-placement Qwen4Exp HyperConnection regression requires NVIDIA CUDA",
)
@pytest.mark.parametrize("offloaded_child", ["attn_hyper_connection", "mlp_hyper_connection"])
def test_qwen4_exp_decoder_mixed_hyper_connection_placement_falls_back_before_partial_execution(
    monkeypatch, offloaded_child
):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    from liger_kernel.transformers.qwen4_exp import patch_qwen4_exp_text_module_for_hyper_connection

    class CountingPLE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden_states, *args, **kwargs):
            self.calls += 1
            return torch.zeros_like(hidden_states)

    class NativeAttention(torch.nn.Module):
        def forward(self, hidden_states, position_embeddings, **kwargs):
            return hidden_states * 0.5, None

    config = Qwen4ExpTextConfig(hidden_size=8, hc_count=4, hc_lowrank=4, rms_norm_eps=1e-5)
    decoder = object.__new__(Qwen4ExpTextDecoderLayer)
    torch.nn.Module.__init__(decoder)
    decoder.ple = CountingPLE()
    decoder.attn_hyper_connection = Qwen4ExpTextGatedResidual(config, use_combine=True).to(device)
    decoder.mlp_hyper_connection = Qwen4ExpTextGatedResidual(config, use_combine=True).to(device)
    decoder.layer_type = "full_attention"
    decoder.self_attn = NativeAttention()
    decoder.mlp = torch.nn.Identity()
    for module in (decoder.attn_hyper_connection, decoder.mlp_hyper_connection, decoder):
        patch_qwen4_exp_text_module_for_hyper_connection(
            module,
            Qwen4ExpTextGatedResidual,
            Qwen4ExpTextDecoderLayer,
        )

    offloaded_module = getattr(decoder, offloaded_child).cpu()
    runtime_device = torch.device(device)
    offloaded_module.register_forward_pre_hook(lambda _module, args: (args[0].cpu(),))
    offloaded_module.register_forward_hook(
        lambda _module, _args, output: tuple(
            value.to(runtime_device) if isinstance(value, torch.Tensor) else value for value in output
        )
    )
    hidden_states = torch.randn(2, 3, config.hc_count * config.hidden_size, device=runtime_device)
    position_embeddings = (torch.empty(0, device=runtime_device), torch.empty(0, device=runtime_device))
    native_decoder_forward = decoder.__dict__["_liger_qwen4_exp_native_decoder_forward"]
    expected = native_decoder_forward(decoder, hidden_states, position_embeddings)
    decoder.ple.calls = 0

    monkeypatch.setattr(
        LigerQwen4ExpGRWriteFunction,
        "apply",
        lambda *args, **kwargs: pytest.fail("Liger GRWrite ran after mixed-placement preflight failed"),
    )
    output = decoder(hidden_states, position_embeddings)

    assert decoder.ple.calls == 1
    assert torch.equal(output, expected)


@requires_qwen4_exp
@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
@pytest.mark.parametrize("hc_count", [2, 3, 4, 8])
def test_qwen4_exp_gated_residual_hc_count_native_liger_parity(hc_count):
    """Only hc_count=4 uses Write4; every supported count remains HF-compatible."""
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    set_seed(42)
    config = Qwen4ExpTextConfig(hidden_size=24, hc_count=hc_count, hc_lowrank=8, rms_norm_eps=1e-5)
    native_module = Qwen4ExpTextGatedResidual(config, use_combine=True).to(device, torch.bfloat16)
    liger_module = copy.deepcopy(native_module)
    _patch_rms_norm_module(liger_module.hc_norm, offset=1.0, casting_mode="gemma", in_place=False)
    liger_module.forward = MethodType(liger_qwen4_exp_gated_residual_forward, liger_module)
    native_input = torch.randn(2, 7, hc_count * config.hidden_size, device=device, dtype=torch.bfloat16).requires_grad_(
        True
    )
    liger_input = native_input.detach().clone().requires_grad_(True)
    native_mixed, _, native_injection = native_module(native_input)
    liger_mixed, _, write_logits = liger_module(liger_input, return_write_logits=True)
    uses_write4 = type(write_logits.grad_fn).__name__ == "LigerGroupRMSNormWrite4FunctionBackward"
    assert uses_write4 == (hc_count == 4)
    if hc_count != 4:
        assert type(liger_mixed.grad_fn) is type(native_mixed.grad_fn)
    liger_injection = 2 * torch.sigmoid(write_logits)
    grad_mixed = torch.randn_like(native_mixed)
    grad_injection = torch.randn_like(native_injection)
    torch.autograd.backward((native_mixed, native_injection), (grad_mixed, grad_injection))
    torch.autograd.backward((liger_mixed, liger_injection), (grad_mixed, grad_injection))

    for actual, expected in (
        (liger_mixed, native_mixed),
        (liger_injection, native_injection),
        (liger_input.grad, native_input.grad),
    ):
        assert_verbose_allclose(actual, expected, atol=8e-3, rtol=2e-3)
    native_parameters = dict(native_module.named_parameters())
    for name, parameter in liger_module.named_parameters():
        assert_verbose_allclose(parameter.grad, native_parameters[name].grad, atol=8e-3, rtol=2e-3)


@requires_qwen4_exp
@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_qwen4_exp_hyper_connection_write4_gr_write_cuda_graph_forward_backward():
    """Capture the production BF16 Write4-to-GRWrite HyperConnection path."""
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    set_seed(42)
    config = Qwen4ExpTextConfig(hidden_size=64, hc_count=4, hc_lowrank=16, rms_norm_eps=1e-5)
    base_module = Qwen4ExpTextGatedResidual(config).to(device, torch.bfloat16)
    _patch_rms_norm_module(base_module.hc_norm, offset=1.0, casting_mode="gemma", in_place=False)
    grad_mixed = torch.randn(2, 8, config.hidden_size, device=device, dtype=torch.bfloat16)
    grad_output = torch.randn(2, 8, config.hc_count * config.hidden_size, device=device, dtype=torch.bfloat16)

    def forward_backward(module, hyper_input, block_output):
        mixed_input, residual, write_logits = liger_qwen4_exp_gated_residual_forward(
            module, hyper_input, return_write_logits=True
        )
        output = LigerQwen4ExpGRWriteFunction.apply(block_output, residual, write_logits)
        torch.autograd.backward((mixed_input, output), (grad_mixed, grad_output))
        return mixed_input, write_logits, output

    # Warm Triton, cuBLAS, and autograd allocations on a side stream before capture.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        warmup_module = copy.deepcopy(base_module)
        warmup_input = torch.randn(
            2,
            8,
            config.hc_count * config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        warmup_block = torch.randn(2, 8, config.hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
        forward_backward(warmup_module, warmup_input, warmup_block)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph_module = copy.deepcopy(base_module)
    static_input = torch.randn(
        2,
        8,
        config.hc_count * config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    static_block = torch.randn(2, 8, config.hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_mixed, captured_write_logits, captured_output = forward_backward(
            graph_module, static_input, static_block
        )

    assert type(captured_write_logits.grad_fn).__name__ == "LigerGroupRMSNormWrite4FunctionBackward"
    assert type(captured_output.grad_fn).__name__ == "LigerQwen4ExpGRWriteFunctionBackward"
    replay_input = torch.randn_like(static_input)
    replay_block = torch.randn_like(static_block)
    reference_module = copy.deepcopy(base_module)
    reference_input = replay_input.detach().clone().requires_grad_(True)
    reference_block = replay_block.detach().clone().requires_grad_(True)
    reference_mixed, reference_write_logits, reference_output = forward_backward(
        reference_module, reference_input, reference_block
    )

    with torch.no_grad():
        static_input.copy_(replay_input)
        static_block.copy_(replay_block)
    static_input.grad.zero_()
    static_block.grad.zero_()
    for parameter in graph_module.parameters():
        parameter.grad.zero_()
    graph.replay()
    torch.cuda.synchronize()

    assert_verbose_allclose(captured_mixed, reference_mixed, atol=0.0, rtol=0.0)
    assert_verbose_allclose(captured_write_logits, reference_write_logits, atol=0.0, rtol=0.0)
    assert_verbose_allclose(captured_output, reference_output, atol=0.0, rtol=0.0)
    assert_verbose_allclose(static_input.grad, reference_input.grad, atol=0.0, rtol=0.0)
    assert_verbose_allclose(static_block.grad, reference_block.grad, atol=0.0, rtol=0.0)
    for (actual_name, actual), (expected_name, expected) in zip(
        graph_module.named_parameters(), reference_module.named_parameters()
    ):
        assert actual_name == expected_name
        assert_verbose_allclose(actual.grad, expected.grad, atol=0.0, rtol=0.0)


@pytest.fixture
def qwen4_exp_ple_compile_case():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer

    set_seed(42)
    config = Qwen4ExpTextConfig(
        hidden_size=32,
        hc_count=4,
        ple_embed_dim=32,
        ple_conv_kernel_size=2,
        ple_layer_ids=[1],
        ngram_size=2,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=128,
        eos_token_id=2,
        rms_norm_eps=1e-5,
    )
    module = Qwen4ExpTextPLELayer(config, layer_idx=0, ple_layer_index=0).to(device, torch.bfloat16)
    with torch.no_grad():
        module.conv1d.weight.normal_(mean=0.0, std=0.05)
    assert torch.count_nonzero(module.conv1d.weight).item() == module.conv1d.weight.numel()
    input_ids = torch.tensor([[11, 12, 2, 21, 22], [31, 2, 41, 42, 43]], device=device)
    hidden = torch.randn(2, 5, 4 * config.hidden_size, device=device, dtype=torch.bfloat16).requires_grad_(True)
    torch.compiler.reset()
    try:
        yield module, input_ids, hidden, torch.randn_like(hidden)
    finally:
        torch.compiler.reset()


def _patch_qwen4_exp_ple(module):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm

    for child in module.modules():
        patch_qwen4_exp_text_module_for_ngram(child, Qwen4ExpTextNGramEmbedding)
        if isinstance(child, Qwen4ExpTextRMSNorm):
            _patch_rms_norm_module(child, offset=1.0, casting_mode="gemma", in_place=False)
            assert child.forward.__func__ is LigerRMSNorm.forward
    assert module.forward.__func__ is Qwen4ExpTextPLELayer.forward


@requires_qwen4_exp
@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_qwen4_exp_liger_ngram_rms_ple_compile_fullgraph_eager_numerics_forward_backward(qwen4_exp_ple_compile_case):
    """Compare native and Liger PLE under eager and eager-numerics fullgraph compilation."""
    native_module, input_ids, native_input, grad_output = qwen4_exp_ple_compile_case
    reference_module = copy.deepcopy(native_module)
    native_compiled_module = copy.deepcopy(native_module)
    compiled_module = copy.deepcopy(native_module)
    for candidate in (reference_module, compiled_module):
        _patch_qwen4_exp_ple(candidate)

    reference_input = native_input.detach().clone().requires_grad_(True)
    native_compiled_input = native_input.detach().clone().requires_grad_(True)
    compiled_input = reference_input.detach().clone().requires_grad_(True)
    native = native_module(native_input, input_ids, past_key_values=None)
    native.backward(grad_output)
    reference = reference_module(reference_input, input_ids, past_key_values=None)
    reference.backward(grad_output)

    torch.compiler.reset()
    # Inductor normally fuses away BF16 downcast/upcast barriers between pointwise
    # operations. Preserve eager numerics for PLE's reduction-sensitive score path.
    with torch._inductor.config.patch(emulate_precision_casts=True):
        native_compiled = torch.compile(native_compiled_module, fullgraph=True, mode="reduce-overhead")
        native_compiled_output = native_compiled(native_compiled_input, input_ids, past_key_values=None)
        native_compiled_output.backward(grad_output)
        torch.compiler.reset()
        compiled = torch.compile(compiled_module, fullgraph=True, mode="reduce-overhead")
        output = compiled(compiled_input, input_ids, past_key_values=None)
        output.backward(grad_output)

    assert_verbose_allclose(reference, native, atol=0.0, rtol=0.0)
    assert_verbose_allclose(reference_input.grad, native_input.grad, atol=0.0, rtol=0.0)
    for (reference_name, reference_parameter), (native_name, native_parameter) in zip(
        reference_module.named_parameters(), native_module.named_parameters()
    ):
        assert reference_name == native_name
        assert_verbose_allclose(reference_parameter.grad, native_parameter.grad, atol=0.0, rtol=0.0)
    assert_verbose_allclose(output, native_compiled_output, atol=0.0, rtol=0.0)
    assert_verbose_allclose(compiled_input.grad, native_compiled_input.grad, atol=0.0, rtol=0.0)
    for (compiled_name, compiled_parameter), (native_name, native_parameter) in zip(
        compiled_module.named_parameters(), native_compiled_module.named_parameters()
    ):
        assert compiled_name == native_name
        assert_verbose_allclose(compiled_parameter.grad, native_parameter.grad, atol=0.0, rtol=0.0)

    def assert_compile_parity(actual, expected, *, max_abs=None):
        actual_float = actual.detach().float().flatten()
        expected_float = expected.detach().float().flatten()
        cosine = torch.nn.functional.cosine_similarity(actual_float, expected_float, dim=0)
        norm_ratio = actual_float.norm() / expected_float.norm()
        observed_max_abs = (actual_float - expected_float).abs().max().item()
        assert cosine >= 0.9998
        assert 0.98 <= norm_ratio <= 1.02
        if max_abs is not None:
            assert observed_max_abs <= max_abs
        return observed_max_abs

    # Inductor may select different vendor GEMM/convolution plans than eager;
    # require tightly aligned BF16 results without incorrectly demanding bit equality.
    output_max_abs = assert_compile_parity(output, reference, max_abs=3.125e-2)
    input_grad_max_abs = assert_compile_parity(compiled_input.grad, reference_input.grad, max_abs=3.125e-2)
    parameter_grad_max_abs = []
    for (actual_name, actual), (expected_name, expected) in zip(
        compiled_module.named_parameters(), reference_module.named_parameters()
    ):
        assert actual_name == expected_name
        parameter_grad_max_abs.append((actual_name, assert_compile_parity(actual.grad, expected.grad)))
    worst_parameter_name, worst_parameter_grad_max_abs = max(parameter_grad_max_abs, key=lambda item: item[1])
    print(
        f"compile max_abs: output={output_max_abs:.8g}, input_grad={input_grad_max_abs:.8g}, "
        f"parameter_grad={worst_parameter_grad_max_abs:.8g} ({worst_parameter_name})"
    )
    torch.compiler.reset()


@requires_qwen4_exp
@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_qwen4_exp_liger_ngram_rms_ple_default_compile_native_parity_forward_backward(qwen4_exp_ple_compile_case):
    """Compare native and Liger PLE under the default Inductor BF16 numerics."""
    native_module, input_ids, native_input, grad_output = qwen4_exp_ple_compile_case
    liger_module = copy.deepcopy(native_module)
    _patch_qwen4_exp_ple(liger_module)

    liger_input = native_input.detach().clone().requires_grad_(True)

    torch.compiler.reset()
    native_compiled = torch.compile(native_module, fullgraph=True, mode="reduce-overhead")
    native_output = native_compiled(native_input, input_ids, past_key_values=None)
    native_output.backward(grad_output)
    torch.compiler.reset()
    liger_compiled = torch.compile(liger_module, fullgraph=True, mode="reduce-overhead")
    liger_output = liger_compiled(liger_input, input_ids, past_key_values=None)
    liger_output.backward(grad_output)

    def assert_bf16_compile_parity(actual, expected, *, max_abs=5e-2):
        actual_float = actual.detach().float().flatten()
        expected_float = expected.detach().float().flatten()
        cosine = torch.nn.functional.cosine_similarity(actual_float, expected_float, dim=0)
        norm_ratio = actual_float.norm() / expected_float.norm().clamp_min(1e-12)
        assert cosine >= 0.9995
        assert 0.98 <= norm_ratio <= 1.02
        if max_abs is not None:
            assert (actual_float - expected_float).abs().max() <= max_abs

    # Default Inductor is free to fuse BF16 casts and select different convolution/GEMM
    # plans. A 0.05 absolute guard on activations, plus cosine and norm guards, accepts expected
    # compounded BF16 differences without allowing a material directional gradient drift.
    # Parameter gradients span different scales, so they use the scale-independent cosine and
    # norm-ratio gates rather than one absolute bound shared across every parameter tensor.
    assert_bf16_compile_parity(liger_output, native_output)
    assert_bf16_compile_parity(liger_input.grad, native_input.grad)
    native_parameters = dict(native_module.named_parameters())
    for name, parameter in liger_module.named_parameters():
        assert parameter.grad is not None
        assert native_parameters[name].grad is not None
        assert_bf16_compile_parity(parameter.grad, native_parameters[name].grad, max_abs=None)
    torch.compiler.reset()


@pytest.mark.parametrize(
    "shape, ngram_size, heads_per_ngram",
    [((2, 17), 2, 4), ((3, 11), 4, 3), ((2, 65), 4, 3)],
)
def test_qwen4_exp_ngram_hash(shape, ngram_size, heads_per_ngram):
    set_seed(42)
    n_heads = (ngram_size - 1) * heads_per_ngram
    input_ids = torch.randint(0, 248320, shape, device=device, dtype=torch.long)
    previous_context = torch.randint(0, 248320, (shape[0], ngram_size - 1), device=device, dtype=torch.long)
    eos_token_id = 2
    input_ids[:, ::5] = eos_token_id
    previous_context[:, 0] = eos_token_id
    multipliers = torch.tensor(
        [922337203685477 // (index + 1) | 1 for index in range(ngram_size)], device=device, dtype=torch.long
    )
    vocab_sizes = torch.arange(1009, 1009 + n_heads, device=device, dtype=torch.long)
    offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), vocab_sizes.cumsum(0)[:-1]])

    output = liger_qwen4_exp_ngram_hash(
        previous_context,
        input_ids,
        multipliers,
        vocab_sizes,
        offsets,
        eos_token_id,
    )
    reference = qwen4_exp_eos_aware_ngram_hash_ref(
        previous_context,
        input_ids,
        multipliers,
        vocab_sizes,
        offsets,
        eos_token_id,
    )
    assert output.dtype == torch.long
    assert torch.equal(output, reference)


def test_qwen4_exp_ngram_hash_rejects_rocm_explicitly(monkeypatch):
    import liger_kernel.ops.qwen4_exp as qwen4_exp_ops

    monkeypatch.setattr(qwen4_exp_ops, "is_hip", lambda: True)
    token_ids = torch.tensor([[1, 2]], device=device, dtype=torch.long)
    metadata = torch.tensor([3, 5], device=device, dtype=torch.long)
    with pytest.raises(ValueError, match="ROCm is not supported"):
        qwen4_exp_ops.qwen4_exp_ngram_hash(
            token_ids[:, :1],
            token_ids,
            metadata,
            metadata,
            metadata,
            eos_token_id=2,
        )


def test_qwen4_exp_ngram_hash_default_fullgraph_and_cuda_graph():
    set_seed(42)
    shape = (2, 17)
    ngram_size = 4
    n_heads = 9
    input_ids = torch.randint(0, 248320, shape, device=device, dtype=torch.long)
    previous_context = torch.randint(0, 248320, (shape[0], ngram_size - 1), device=device, dtype=torch.long)
    input_ids[:, ::5] = 2
    previous_context[:, 0] = 2
    multipliers = torch.tensor(
        [922337203685477 // (index + 1) | 1 for index in range(ngram_size)], device=device, dtype=torch.long
    )
    vocab_sizes = torch.arange(1009, 1009 + n_heads, device=device, dtype=torch.long)
    offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), vocab_sizes.cumsum(0)[:-1]])
    args = (previous_context, input_ids, multipliers, vocab_sizes, offsets, 2)
    expected = qwen4_exp_eos_aware_ngram_hash_ref(*args)

    torch.compiler.reset()
    compiled = torch.compile(liger_qwen4_exp_ngram_hash, fullgraph=True)
    assert torch.equal(compiled(*args), expected)
    torch.compiler.reset()

    static_args = tuple(value.clone() if isinstance(value, torch.Tensor) else value for value in args)
    liger_qwen4_exp_ngram_hash(*static_args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = liger_qwen4_exp_ngram_hash(*static_args)

    replay_previous = torch.randint_like(previous_context, low=0, high=248320)
    replay_input = torch.randint_like(input_ids, low=0, high=248320)
    replay_previous[:, 1] = 2
    replay_input[:, ::4] = 2
    replay_expected = qwen4_exp_eos_aware_ngram_hash_ref(
        replay_previous,
        replay_input,
        multipliers,
        vocab_sizes,
        offsets,
        2,
    )
    static_args[0].copy_(replay_previous)
    static_args[1].copy_(replay_input)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, replay_expected)


@requires_qwen4_exp
def test_qwen4_exp_ngram_embedding_eos_aware_forward():
    try:
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding
    except ImportError:
        pytest.skip("qwen4_exp module not available")

    config = Qwen4ExpTextConfig(
        vocab_size=101,
        hidden_size=32,
        hc_count=4,
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=128,
        eos_token_id=2,
    )
    module = Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, layer_idx=0, ple_layer_index=0).to(device)
    input_ids = torch.tensor([[11, 12, 2, 21, 22, 23], [31, 2, 41, 42, 2, 51]], device=device)

    token_history = torch.cat(
        [input_ids.new_full((input_ids.shape[0], module.context_len), module.eos_token_id), input_ids], dim=-1
    )
    shifted_tokens = torch.stack(
        [module._shift_right_ignore_eos(token_history, shift) for shift in range(module.ngram_size)], dim=-1
    )[:, -input_ids.shape[1] :]
    reference_ids = qwen4_exp_ngram_hash_ref(
        shifted_tokens,
        module.layer_multipliers,
        module.ngram_heads_vocab_sizes,
        module.ngram_heads_offsets,
    )
    reference = module.ngram_embedding(reference_ids).flatten(-2)

    output = liger_qwen4_exp_ngram_embedding_forward(module, input_ids, past_key_values=None)
    assert_verbose_allclose(output, reference, atol=0.0, rtol=0.0)


@requires_qwen4_exp
def test_qwen4_exp_ngram_embedding_incremental_cache_parity():
    try:
        from transformers.cache_utils import DynamicCache
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding
    except ImportError:
        pytest.skip("qwen4_exp module not available")

    config = Qwen4ExpTextConfig(
        vocab_size=101,
        hidden_size=32,
        num_hidden_layers=2,
        hc_count=4,
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=128,
        eos_token_id=2,
        layer_types=["linear_attention", "full_attention"],
    )
    reference_module = Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, layer_idx=0, ple_layer_index=0).to(
        device
    )
    liger_module = copy.deepcopy(reference_module)
    reference_cache = DynamicCache(config=config)
    liger_cache = DynamicCache(config=config)
    chunks = [
        torch.tensor([[11], [31]], device=device),
        torch.tensor([[12, 2, 21], [2, 41, 42]], device=device),
        torch.tensor([[22], [2]], device=device),
    ]

    for input_ids in chunks:
        reference = reference_module(input_ids, reference_cache)
        output = liger_qwen4_exp_ngram_embedding_forward(liger_module, input_ids, liger_cache)
        assert_verbose_allclose(output, reference, atol=0.0, rtol=0.0)
        assert torch.equal(reference_cache.layers[0].conv_states[2], liger_cache.layers[0].conv_states[2])


@requires_qwen4_exp
def test_qwen4_exp_ngram_exact_eos_ids_and_chunk_partition_invariance():
    """A B EOS C keeps history for EOS itself and resets history before C."""
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

    config = Qwen4ExpTextConfig(
        vocab_size=101,
        hidden_size=32,
        num_hidden_layers=2,
        hc_count=4,
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=128,
        eos_token_id=2,
        layer_types=["linear_attention", "qwen_sparse_attention"],
    )
    base_module = Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, layer_idx=0, ple_layer_index=0).to(device)
    input_ids = torch.tensor([[11, 12, 2, 21]], device=device)
    expected_shifted = torch.tensor(
        [[[11, 2, 2], [12, 11, 2], [2, 12, 11], [21, 2, 2]]],
        device=device,
    )
    expected_ids = qwen4_exp_ngram_hash_ref(
        expected_shifted,
        base_module.layer_multipliers,
        base_module.ngram_heads_vocab_sizes,
        base_module.ngram_heads_offsets,
    )

    baseline_ids = None
    baseline_history = None
    for partition in ([4], [2, 2], [1, 1, 1, 1]):
        reference_module = copy.deepcopy(base_module)
        liger_module = copy.deepcopy(base_module)
        reference_module.ngram_embedding = _NGramIdRecorder(reference_module.ngram_embedding)
        liger_module.ngram_embedding = _NGramIdRecorder(liger_module.ngram_embedding)
        liger_module.forward = MethodType(liger_qwen4_exp_ngram_embedding_forward, liger_module)
        assert reference_module.forward.__func__ is Qwen4ExpTextNGramEmbedding.forward
        assert liger_module.forward.__func__ is liger_qwen4_exp_ngram_embedding_forward
        reference_cache = DynamicCache(config=config)
        liger_cache = DynamicCache(config=config)

        start = 0
        for chunk_size in partition:
            end = start + chunk_size
            reference_module(input_ids[:, start:end], reference_cache)
            liger_module(input_ids[:, start:end], liger_cache)
            start = end

        reference_ids = torch.cat(reference_module.ngram_embedding.ids, dim=1)
        liger_ids = torch.cat(liger_module.ngram_embedding.ids, dim=1)
        reference_history = reference_cache.layers[0].conv_states[2]
        liger_history = liger_cache.layers[0].conv_states[2]
        assert torch.equal(reference_ids, expected_ids)
        assert torch.equal(liger_ids, expected_ids)
        assert torch.equal(liger_history, reference_history)
        if baseline_ids is None:
            baseline_ids = reference_ids
            baseline_history = reference_history
        else:
            assert torch.equal(reference_ids, baseline_ids)
            assert torch.equal(reference_history, baseline_history)

    assert torch.equal(baseline_history, torch.tensor([[2, 21]], device=device))


@requires_qwen4_exp
def test_qwen4_exp_multiple_ple_layers_use_independent_native_hash_namespaces():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

    config = _qwen4_exp_hybrid_config()
    config.num_hidden_layers = 5
    config.layer_types = ["linear_attention"] * 5
    config.ple_layer_ids = [1, 5]
    first = Qwen4ExpTextDecoderLayer(config, layer_idx=0).ple.ple_embedding.to(device)
    fifth = Qwen4ExpTextDecoderLayer(config, layer_idx=4).ple.ple_embedding.to(device)
    assert first.ple_layer_index == 0
    assert fifth.ple_layer_index == 1
    assert not torch.equal(first.layer_multipliers, fifth.layer_multipliers)
    assert not torch.equal(first.ngram_heads_vocab_sizes, fifth.ngram_heads_vocab_sizes)

    input_ids = torch.tensor([[11, 12, 2, 21]], device=device)
    observed_ids = []
    for native_module in (first, fifth):
        liger_module = copy.deepcopy(native_module)
        native_module.ngram_embedding = _NGramIdRecorder(native_module.ngram_embedding)
        liger_module.ngram_embedding = _NGramIdRecorder(liger_module.ngram_embedding)
        liger_module.forward = MethodType(liger_qwen4_exp_ngram_embedding_forward, liger_module)
        native_module(input_ids, past_key_values=None)
        liger_module(input_ids, past_key_values=None)
        native_ids = native_module.ngram_embedding.ids[0]
        liger_ids = liger_module.ngram_embedding.ids[0]
        assert torch.equal(liger_ids, native_ids)
        observed_ids.append(native_ids)
    assert not torch.equal(observed_ids[0], observed_ids[1])
    assert all(isinstance(module, Qwen4ExpTextNGramEmbedding) for module in (first, fifth))


@requires_qwen4_exp
def test_qwen4_exp_ngram_hash_consumes_runtime_buffers(monkeypatch):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

    config = Qwen4ExpTextConfig(
        vocab_size=101,
        hidden_size=32,
        num_hidden_layers=1,
        hc_count=4,
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=128,
        eos_token_id=2,
        layer_types=["linear_attention"],
    )
    native_module = Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, 0, 0).to(device)
    original_multipliers = native_module.layer_multipliers.clone()
    original_vocab_sizes = native_module.ngram_heads_vocab_sizes.clone()
    original_offsets = native_module.ngram_heads_offsets.clone()
    native_module.layer_multipliers.copy_(torch.tensor([17, 29, 43], device=device))
    native_module.ngram_heads_vocab_sizes.copy_(torch.tensor([13, 17, 19, 23], device=device))
    native_module.ngram_heads_offsets.copy_(torch.tensor([0, 13, 30, 49], device=device))
    assert not torch.equal(native_module.layer_multipliers, original_multipliers)
    assert not torch.equal(native_module.ngram_heads_vocab_sizes, original_vocab_sizes)
    assert not torch.equal(native_module.ngram_heads_offsets, original_offsets)

    liger_module = copy.deepcopy(native_module)
    native_module.ngram_embedding = _NGramIdRecorder(native_module.ngram_embedding)
    liger_module.ngram_embedding = _NGramIdRecorder(liger_module.ngram_embedding)
    liger_module.forward = MethodType(liger_qwen4_exp_ngram_embedding_forward, liger_module)
    import liger_kernel.transformers.qwen4_exp as qwen4_exp_transformers

    hash_calls = 0
    original_hash = qwen4_exp_transformers.qwen4_exp_ngram_hash

    def record_hash(*args, **kwargs):
        nonlocal hash_calls
        hash_calls += 1
        return original_hash(*args, **kwargs)

    monkeypatch.setattr(qwen4_exp_transformers, "qwen4_exp_ngram_hash", record_hash)
    input_ids = torch.tensor([[11, 12, 2, 21]], device=device)
    native_module(input_ids, None)
    liger_module(input_ids, None)
    previous_context = input_ids.new_full((1, config.ngram_size - 1), config.eos_token_id)
    expected_ids = qwen4_exp_eos_aware_ngram_hash_ref(
        previous_context,
        input_ids,
        native_module.layer_multipliers,
        native_module.ngram_heads_vocab_sizes,
        native_module.ngram_heads_offsets,
        config.eos_token_id,
    )
    assert hash_calls == 1
    assert torch.equal(native_module.ngram_embedding.ids[0], expected_ids)
    assert torch.equal(liger_module.ngram_embedding.ids[0], expected_ids)


@pytest.mark.parametrize(
    "mismatched_buffer",
    ["layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"],
)
def test_qwen4_exp_ngram_mismatched_cuda_metadata_falls_back_before_cache_mutation(monkeypatch, mismatched_buffer):
    import liger_kernel.transformers.qwen4_exp as qwen4_exp_transformers

    class DeviceOnlyTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, tensor_device):
            return torch.Tensor._make_wrapper_subclass(
                cls,
                (1,),
                dtype=torch.long,
                device=tensor_device,
                requires_grad=False,
            )

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            raise AssertionError("device-only tensor must not participate in an operation")

    class CacheMustRemainUntouched:
        native_calls = 0

        def has_previous_state(self, *args, **kwargs):
            raise AssertionError("Liger inspected the cache before native fallback")

    class NGramModule:
        def native_forward(self, input_ids, past_key_values):
            past_key_values.native_calls += 1
            return "native-fallback"

        _liger_qwen4_exp_native_ngram_forward = native_forward

    module = NGramModule()
    module.layer_multipliers = DeviceOnlyTensor("cuda:0")
    module.ngram_heads_vocab_sizes = DeviceOnlyTensor("cuda:0")
    module.ngram_heads_offsets = DeviceOnlyTensor("cuda:0")
    setattr(module, mismatched_buffer, DeviceOnlyTensor("cuda:1"))
    input_ids = DeviceOnlyTensor("cuda:0")
    cache = CacheMustRemainUntouched()
    monkeypatch.setattr(qwen4_exp_transformers, "is_hip", lambda: False)
    monkeypatch.setattr(
        qwen4_exp_transformers,
        "qwen4_exp_ngram_hash",
        lambda *args, **kwargs: pytest.fail("Liger hash ran for mismatched CUDA metadata"),
    )

    output = liger_qwen4_exp_ngram_embedding_forward(module, input_ids, cache)

    assert output == "native-fallback"
    assert cache.native_calls == 1


@requires_qwen4_exp
def test_qwen4_exp_ngram_incremental_cache_cpu_fallback_updates_once(monkeypatch):
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

    import liger_kernel.transformers.qwen4_exp as qwen4_exp_transformers

    from liger_kernel.transformers.qwen4_exp import patch_qwen4_exp_text_module_for_ngram

    config = Qwen4ExpTextConfig(
        vocab_size=101,
        hidden_size=32,
        num_hidden_layers=2,
        hc_count=4,
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=128,
        eos_token_id=2,
        layer_types=["linear_attention", "full_attention"],
    )
    reference_module = Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, layer_idx=0, ple_layer_index=0)
    patched_module = copy.deepcopy(reference_module)
    patch_qwen4_exp_text_module_for_ngram(patched_module, Qwen4ExpTextNGramEmbedding)
    reference_cache = DynamicCache(config=config)
    patched_cache = DynamicCache(config=config)
    update_calls = 0
    native_update = patched_cache.update_conv_state

    def record_update(*args, **kwargs):
        nonlocal update_calls
        update_calls += 1
        return native_update(*args, **kwargs)

    monkeypatch.setattr(patched_cache, "update_conv_state", record_update)
    monkeypatch.setattr(
        qwen4_exp_transformers,
        "qwen4_exp_ngram_hash",
        lambda *args, **kwargs: pytest.fail("Liger hash ran for CPU fallback"),
    )
    chunks = [torch.tensor([[11]]), torch.tensor([[12, 2, 21]]), torch.tensor([[22]])]

    for input_ids in chunks:
        reference = reference_module(input_ids, reference_cache)
        output = patched_module(input_ids, patched_cache)
        assert torch.equal(output, reference)
        assert torch.equal(reference_cache.layers[0].conv_states[2], patched_cache.layers[0].conv_states[2])

    assert update_calls == len(chunks)


@requires_qwen4_exp
def test_qwen4_exp_ngram_embedding_cpu_offload_native_liger_forward_backward():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

    config = Qwen4ExpTextConfig(
        vocab_size=101,
        hidden_size=32,
        num_hidden_layers=1,
        hc_count=4,
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ngram_size=2,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=128,
        eos_token_id=2,
        layer_types=["linear_attention"],
    )
    native_module = Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, 0, 0).to(device)
    liger_module = copy.deepcopy(native_module)
    native_module.ngram_embedding.to("cpu")
    liger_module.ngram_embedding.to("cpu")
    liger_module.forward = MethodType(liger_qwen4_exp_ngram_embedding_forward, liger_module)
    native_lookup_devices = []
    liger_lookup_devices = []
    native_hook = native_module.ngram_embedding.register_forward_pre_hook(
        lambda _module, args: native_lookup_devices.append(args[0].device)
    )
    liger_hook = liger_module.ngram_embedding.register_forward_pre_hook(
        lambda _module, args: liger_lookup_devices.append(args[0].device)
    )
    assert tuple(dict(native_module.named_parameters())) == ("ngram_embedding.weight",)
    assert tuple(dict(liger_module.named_parameters())) == ("ngram_embedding.weight",)
    assert native_module.ngram_embedding.weight.device.type == "cpu"
    assert liger_module.ngram_embedding.weight.device.type == "cpu"

    input_ids = torch.tensor([[11, 12, 2, 21]], device=device)
    native = native_module(input_ids, None)
    output = liger_module(input_ids, None)
    native_hook.remove()
    liger_hook.remove()
    assert native_lookup_devices == liger_lookup_devices == [torch.device("cpu")]
    assert native.device == input_ids.device
    assert output.device == input_ids.device
    assert_verbose_allclose(output, native, atol=0.0, rtol=0.0)
    grad_output = torch.randn_like(native)
    native.backward(grad_output)
    output.backward(grad_output)
    assert native_module.ngram_embedding.weight.grad.device.type == "cpu"
    assert liger_module.ngram_embedding.weight.grad.device.type == "cpu"
    assert_verbose_allclose(
        liger_module.ngram_embedding.weight.grad,
        native_module.ngram_embedding.weight.grad,
        atol=0.0,
        rtol=0.0,
    )


@requires_qwen4_exp
def test_qwen4_exp_ple_signed_sqrt_edges_and_mask_native_liger_forward_backward():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm

    set_seed(42)
    config = _qwen4_exp_hybrid_config()
    config.hidden_size = 8
    config.hc_count = 2
    config.ple_embed_dim = 8
    config.ple_conv_kernel_size = 2
    config.ngram_size = 2
    config.heads_per_ngram = 2
    native_module = Qwen4ExpTextPLELayer(config, layer_idx=0, ple_layer_index=0).to(device, torch.float32)
    with torch.no_grad():
        native_module.conv1d.weight.normal_(mean=0.0, std=0.05)
    assert torch.count_nonzero(native_module.conv1d.weight).item() == native_module.conv1d.weight.numel()
    input_ids = torch.tensor([[11, 12, 13, 14, 15, 16]], device=device)
    with torch.no_grad():
        embeddings = native_module.ple_embedding(input_ids, None)
        key_normed = native_module.norm_key(native_module.key_proj(embeddings))
    scales = torch.tensor([0.0, 1e-5, -1e-5, 1.0, -1.0, 1.0], device=device).view(1, -1, 1)
    native_input = (key_normed * scales).detach().requires_grad_(True)
    with torch.no_grad():
        query_normed = native_module.norm_query(native_input).unflatten(-1, (config.hc_count, config.hidden_size))
        key_groups = key_normed.unflatten(-1, (config.hc_count, config.hidden_size))
        scores = (key_groups * query_normed).sum(dim=-1) / config.hidden_size**0.5
    assert torch.equal(scores[:, 0], torch.zeros_like(scores[:, 0]))
    assert (scores[:, 1] > 0).all() and (scores[:, 2] < 0).all()
    assert (scores[:, 3] > 0).all() and (scores[:, 4] < 0).all()
    assert scores[:, 1].abs().max() < scores[:, 3].abs().min()
    assert scores[:, 2].abs().max() < scores[:, 4].abs().min()

    liger_module = copy.deepcopy(native_module)
    for module in liger_module.modules():
        patch_qwen4_exp_text_module_for_ngram(module, Qwen4ExpTextNGramEmbedding)
        if isinstance(module, Qwen4ExpTextRMSNorm):
            _patch_rms_norm_module(module, offset=1.0, casting_mode="gemma", in_place=False)
    liger_input = native_input.detach().clone().requires_grad_(True)
    conv_mask = torch.tensor([[1, 1, 1, 1, 1, 0]], device=device)
    native = native_module(native_input, input_ids, None, conv_mask=conv_mask)
    output = liger_module(liger_input, input_ids, None, conv_mask=conv_mask)
    grad_output = torch.randn_like(native)
    native.backward(grad_output)
    output.backward(grad_output)
    assert_verbose_allclose(output, native, atol=2e-5, rtol=2e-5)
    assert_verbose_allclose(liger_input.grad, native_input.grad, atol=2e-5, rtol=2e-5)
    native_parameters = dict(native_module.named_parameters())
    for name, parameter in liger_module.named_parameters():
        assert_verbose_allclose(parameter.grad, native_parameters[name].grad, atol=2e-5, rtol=2e-5)


@requires_qwen4_exp
@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
@pytest.mark.parametrize(
    "mode, partition, cache_kind",
    [
        ("no-cache", [4], None),
        ("dynamic-2-chunk", [2, 2], "dynamic"),
        ("dynamic-token", [1, 1, 1, 1], "dynamic"),
        ("static-token", [1, 1, 1, 1], "static"),
    ],
)
def test_qwen4_exp_full_model_native_liger_cache_decode_parity(mode, partition, cache_kind):
    """Validate the tiny hybrid QSA + GDN model, including every Qwen4Exp cache state."""
    from transformers.cache_utils import DynamicCache
    from transformers.cache_utils import StaticCache
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextAttention
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextQSAIndexer
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm

    set_seed(42)
    config = _qwen4_exp_hybrid_config()
    reference_model = Qwen4ExpForCausalLM(config).to(device, torch.bfloat16).eval()
    with torch.no_grad():
        for module in reference_model.modules():
            if isinstance(module, Qwen4ExpTextPLELayer):
                module.conv1d.weight.normal_(mean=0.0, std=0.05)
    ple_layers = [module for module in reference_model.modules() if isinstance(module, Qwen4ExpTextPLELayer)]
    assert ple_layers
    for ple in ple_layers:
        assert torch.count_nonzero(ple.conv1d.weight).item() == ple.conv1d.weight.numel()

    qsa_layers = [layer for layer in reference_model.model.layers if layer.layer_type == "qwen_sparse_attention"]
    assert qsa_layers
    assert all(isinstance(layer.self_attn, Qwen4ExpTextAttention) for layer in qsa_layers)
    assert all(isinstance(layer.self_attn.indexer, Qwen4ExpTextQSAIndexer) for layer in qsa_layers)

    liger_model = copy.deepcopy(reference_model)
    _patch_qwen4_exp_candidate(liger_model)
    reference_gr = next(module for module in reference_model.modules() if isinstance(module, Qwen4ExpTextGatedResidual))
    liger_gr = next(module for module in liger_model.modules() if isinstance(module, Qwen4ExpTextGatedResidual))
    reference_ple = next(module for module in reference_model.modules() if isinstance(module, Qwen4ExpTextPLELayer))
    liger_ple = next(module for module in liger_model.modules() if isinstance(module, Qwen4ExpTextPLELayer))
    reference_norm = next(module for module in reference_model.modules() if isinstance(module, Qwen4ExpTextRMSNorm))
    liger_norm = next(module for module in liger_model.modules() if isinstance(module, Qwen4ExpTextRMSNorm))
    assert reference_gr.forward.__func__ is Qwen4ExpTextGatedResidual.forward
    assert liger_gr.forward.__func__ is liger_qwen4_exp_gated_residual_forward
    assert reference_ple.forward.__func__ is Qwen4ExpTextPLELayer.forward
    assert liger_ple.forward.__func__ is Qwen4ExpTextPLELayer.forward
    assert reference_norm.forward.__func__ is Qwen4ExpTextRMSNorm.forward
    assert liger_norm.forward.__func__ is LigerRMSNorm.forward
    liger_qsa_layers = [layer for layer in liger_model.model.layers if layer.layer_type == "qwen_sparse_attention"]
    assert all(isinstance(layer.self_attn, Qwen4ExpTextAttention) for layer in liger_qsa_layers)
    assert all(isinstance(layer.self_attn.indexer, Qwen4ExpTextQSAIndexer) for layer in liger_qsa_layers)

    # Row 0 exercises the exact EOS boundary; row 1 is right-padded.
    input_ids = torch.tensor([[11, 12, 2, 21], [31, 32, 33, 0]], device=device)
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], device=device)

    def run(model, chunks, kind):
        if kind == "dynamic":
            cache = DynamicCache(config=config)
        elif kind == "static":
            cache = StaticCache(config=config, max_cache_len=input_ids.shape[1])
        else:
            cache = None
        logits = []
        hidden = []
        start = 0
        with torch.inference_mode():
            for chunk_size in chunks:
                end = start + chunk_size
                result = model(
                    input_ids=input_ids[:, start:end],
                    attention_mask=attention_mask[:, :end],
                    past_key_values=cache,
                    use_cache=cache is not None,
                    output_hidden_states=True,
                )
                logits.append(result.logits)
                hidden.append(result.hidden_states[-1])
                start = end
        return torch.cat(logits, dim=1), torch.cat(hidden, dim=1), cache

    native_full_logits, native_full_hidden, _ = run(reference_model, [input_ids.shape[1]], None)
    native_logits, native_hidden, native_cache = run(reference_model, partition, cache_kind)
    liger_logits, liger_hidden, liger_cache = run(liger_model, partition, cache_kind)

    metrics = {}

    def compare_float(name, actual, expected, atol, token_mask=None):
        assert torch.isfinite(actual).all(), f"{name} actual contains NaN/Inf"
        assert torch.isfinite(expected).all(), f"{name} expected contains NaN/Inf"
        if token_mask is not None:
            expanded_mask = token_mask.unsqueeze(-1).expand_as(actual)
            actual = actual.masked_select(expanded_mask)
            expected = expected.masked_select(expanded_mask)
        max_abs = (actual.float() - expected.float()).abs().max().item()
        metrics[name] = max_abs
        assert_verbose_allclose(actual, expected, atol=atol, rtol=2e-3)

    valid_tokens = attention_mask.bool()
    compare_float("native-mode-vs-full-logits", native_logits, native_full_logits, 2e-3, valid_tokens)
    compare_float("native-mode-vs-full-hidden", native_hidden, native_full_hidden, 2e-3, valid_tokens)
    compare_float("liger-vs-native-logits", liger_logits, native_logits, 2e-3)
    compare_float("liger-vs-native-hidden", liger_hidden, native_hidden, 2e-2)

    if native_cache is not None:
        assert type(liger_cache) is type(native_cache)
        assert torch.equal(
            torch.as_tensor(liger_cache.get_seq_length()), torch.as_tensor(native_cache.get_seq_length())
        )
        assert torch.equal(liger_cache.position_ids, native_cache.position_ids)
        native_linear, native_qsa = native_cache.layers
        liger_linear, liger_qsa = liger_cache.layers
        for state_name in ("conv_states", "recurrent_states"):
            native_states = getattr(native_linear, state_name)
            liger_states = getattr(liger_linear, state_name)
            assert native_states.keys() == liger_states.keys()
            for state_idx in native_states:
                if native_states[state_idx] is None:
                    assert liger_states[state_idx] is None
                    continue
                compare_float(
                    f"linear.{state_name}[{state_idx}]",
                    liger_states[state_idx],
                    native_states[state_idx],
                    2e-2,
                )
        assert all(native_linear.conv_states[state_idx] is not None for state_idx in (0, 1, 2))
        assert native_linear.recurrent_states[0] is not None
        for state_name in ("is_conv_states_initialized", "is_recurrent_states_initialized", "has_previous_state"):
            assert getattr(liger_linear, state_name) == getattr(native_linear, state_name)
        assert liger_linear.conv_kernel_size == native_linear.conv_kernel_size
        assert liger_linear.record_past == native_linear.record_past
        for state_name in ("keys", "values", "indexer_keys"):
            compare_float(
                f"qsa.{state_name}",
                getattr(liger_qsa, state_name),
                getattr(native_qsa, state_name),
                2e-2,
            )
        assert liger_qsa.is_initialized == native_qsa.is_initialized
        assert liger_qsa.is_indexer_initialized == native_qsa.is_indexer_initialized
        for state_name in ("cumulative_length", "indexer_cumulative_length"):
            if hasattr(native_qsa, state_name):
                assert torch.equal(getattr(liger_qsa, state_name), getattr(native_qsa, state_name))
    print(f"{mode}: " + ", ".join(f"{name}={value:.8g}" for name, value in metrics.items()))


@requires_qwen4_exp
@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_qwen4_exp_packed_cu_seq_lens_native_liger_forward_backward_and_isolation():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel

    set_seed(42)
    config = _qwen4_exp_hybrid_config()
    config.num_hidden_layers = 1
    config.layer_types = ["linear_attention"]
    config.ple_layer_ids = []
    native_packed = Qwen4ExpTextModel(config).to(device, torch.bfloat16).train()
    native_separate = copy.deepcopy(native_packed)
    liger_packed = copy.deepcopy(native_packed)
    _patch_qwen4_exp_candidate(liger_packed)

    observed_cu_seq_lens = []

    def record_cu_seq_lens(module):
        original_forward = module.forward

        def forward(_module, *args, **kwargs):
            cu_seq_lens = kwargs.get("cu_seq_lens_q")
            observed_cu_seq_lens.append(None if cu_seq_lens is None else cu_seq_lens.detach().clone())
            return original_forward(*args, **kwargs)

        module.forward = MethodType(forward, module)

    record_cu_seq_lens(native_packed.layers[0].linear_attn)
    record_cu_seq_lens(liger_packed.layers[0].linear_attn)
    cu_seq_lens = torch.tensor([0, 3, 6], device=device, dtype=torch.int32)
    packed_ids = torch.tensor([[11, 12, 13, 21, 22, 23]], device=device)
    separate_ids = packed_ids.view(2, 3)
    packed_positions = torch.tensor([[0, 1, 2, 0, 1, 2]], device=device)
    separate_positions = torch.tensor([[0, 1, 2], [0, 1, 2]], device=device)
    native_output = native_packed(
        input_ids=packed_ids,
        position_ids=packed_positions,
        use_cache=False,
        cu_seq_lens_q=cu_seq_lens,
    ).last_hidden_state
    liger_output = liger_packed(
        input_ids=packed_ids,
        position_ids=packed_positions,
        use_cache=False,
        cu_seq_lens_q=cu_seq_lens,
    ).last_hidden_state
    separate_output = native_separate(
        input_ids=separate_ids,
        position_ids=separate_positions,
        use_cache=False,
    ).last_hidden_state
    assert len(observed_cu_seq_lens) == 2
    assert all(torch.equal(observed, cu_seq_lens) for observed in observed_cu_seq_lens)
    # Treating the packed samples as a real batch must reproduce the packed execution. This
    # validates the GDN reset boundary without requiring bitwise identity from later BF16 MoE GEMMs.
    assert_verbose_allclose(native_output.view(2, 3, -1), separate_output, atol=2e-2, rtol=2e-3)
    assert_verbose_allclose(liger_output, native_output, atol=2e-2, rtol=2e-3)

    grad_output = torch.randn_like(native_output)
    native_output.backward(grad_output)
    liger_output.backward(grad_output)
    native_parameters = dict(native_packed.named_parameters())
    for name, parameter in liger_packed.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert_verbose_allclose(
            parameter.grad,
            native_parameters[name].grad,
            atol=2e-2,
            rtol=1e-2,
            extra_info=f"gradient mismatch for {name}\n",
        )


@requires_qwen4_exp
@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_qwen4_exp_output_contract_native_liger_values_and_hidden_state_immutability():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    from liger_kernel.transformers.model.qwen4_exp import lce_forward

    set_seed(42)
    config = _qwen4_exp_hybrid_config()
    native_model = Qwen4ExpForCausalLM(config).to(device, torch.bfloat16).eval()
    liger_model = copy.deepcopy(native_model)
    _patch_qwen4_exp_candidate(liger_model)
    liger_model.forward = MethodType(lce_forward, liger_model)
    input_ids = torch.tensor([[11, 12, 2, 21]], device=device)
    kwargs = dict(
        input_ids=input_ids,
        use_cache=False,
        output_hidden_states=True,
        output_router_logits=True,
        output_attentions=True,
    )
    with torch.inference_mode():
        native = native_model(**kwargs)
        output = liger_model(**kwargs)
    native_hidden_snapshots = tuple(hidden.clone() for hidden in native.hidden_states)
    liger_hidden_snapshots = tuple(hidden.clone() for hidden in output.hidden_states)
    assert len(native.hidden_states) == len(output.hidden_states) == config.num_hidden_layers + 1
    assert len(native.router_logits) == len(output.router_logits) == config.num_hidden_layers
    assert len(native.attentions) == len(output.attentions) == 1
    for actual, expected in zip(output.hidden_states, native.hidden_states):
        assert_verbose_allclose(actual, expected, atol=2e-2, rtol=2e-3)
    for actual, expected in zip(output.router_logits, native.router_logits):
        assert_verbose_allclose(actual, expected, atol=2e-2, rtol=2e-3)
    for actual, expected in zip(output.attentions, native.attentions):
        assert actual is not None and expected is not None
        assert_verbose_allclose(actual, expected, atol=2e-2, rtol=2e-3)

    # The output-capturing machinery intentionally transforms the decoder's hyper-stream output
    # into public hidden states. Verify the returned tensors themselves remain stable after reuse.
    changed_kwargs = {**kwargs, "input_ids": torch.tensor([[31, 32, 2, 41]], device=device)}
    with torch.inference_mode():
        native_model(**changed_kwargs)
        liger_model(**changed_kwargs)
    for captured, snapshot in zip(native.hidden_states, native_hidden_snapshots):
        assert torch.equal(captured, snapshot)
    for captured, snapshot in zip(output.hidden_states, liger_hidden_snapshots):
        assert torch.equal(captured, snapshot)


@requires_qwen4_exp
@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_qwen4_exp_gradient_checkpointing_native_liger_forward_backward():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer

    set_seed(42)
    config = _qwen4_exp_hybrid_config()
    config.num_hidden_layers = 1
    config.layer_types = ["linear_attention"]
    config.ple_layer_ids = [1]
    native_model = Qwen4ExpForCausalLM(config).to(device, torch.bfloat16).train()
    with torch.no_grad():
        for module in native_model.modules():
            if isinstance(module, Qwen4ExpTextPLELayer):
                module.conv1d.weight.normal_(mean=0.0, std=0.05)
    liger_model = copy.deepcopy(native_model)
    _patch_qwen4_exp_candidate(liger_model)
    native_model.gradient_checkpointing_enable()
    liger_model.gradient_checkpointing_enable()
    assert native_model.model.gradient_checkpointing
    assert liger_model.model.gradient_checkpointing
    input_ids = torch.tensor([[11, 12, 2, 21]], device=device)
    native = native_model(input_ids=input_ids, labels=input_ids, use_cache=False)
    output = liger_model(input_ids=input_ids, labels=input_ids, use_cache=False)
    native.loss.backward()
    output.loss.backward()
    assert torch.isfinite(native.loss)
    assert torch.isfinite(output.loss)
    assert_verbose_allclose(output.loss, native.loss, atol=2e-2, rtol=2e-3)
    assert_verbose_allclose(output.logits, native.logits, atol=2e-2, rtol=2e-3)
    native_parameters = dict(native_model.named_parameters())
    for name, parameter in liger_model.named_parameters():
        if native_parameters[name].grad is None:
            assert parameter.grad is None
            continue
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert_verbose_allclose(parameter.grad, native_parameters[name].grad, atol=2e-2, rtol=2e-3)


def test_qwen4_exp_lce_skip_logits_requires_labels():
    from liger_kernel.transformers.model.qwen4_exp import lce_forward

    with pytest.raises(ValueError, match="labels and shift_labels are None"):
        lce_forward(object(), skip_logits=True)


@requires_qwen4_exp
@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_qwen4_exp_lce_router_aux_loss_native_liger_training_parity():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    from liger_kernel.transformers.model.qwen4_exp import lce_forward

    set_seed(42)
    config = _qwen4_exp_hybrid_config()
    config.num_hidden_layers = 1
    config.layer_types = ["linear_attention"]
    config.ple_layer_ids = []
    config.router_aux_loss_coef = 0.25
    native_model = Qwen4ExpForCausalLM(config).to(device, torch.bfloat16).train()
    liger_model = copy.deepcopy(native_model)
    _patch_qwen4_exp_candidate(liger_model)
    liger_model.forward = MethodType(lce_forward, liger_model)

    input_ids = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]], device=device)
    labels = input_ids.clone()
    with torch.no_grad():
        native_base_loss = native_model(
            input_ids=input_ids,
            labels=labels,
            output_router_logits=False,
            use_cache=False,
        ).loss
        liger_base = liger_model(
            input_ids=input_ids,
            labels=labels,
            output_router_logits=False,
            skip_logits=True,
            use_cache=False,
        )

    native_output = native_model(
        input_ids=input_ids,
        labels=labels,
        output_router_logits=True,
        use_cache=False,
    )
    liger_output = liger_model(
        input_ids=input_ids,
        labels=labels,
        output_router_logits=True,
        skip_logits=True,
        use_cache=False,
    )

    assert liger_base.logits is None
    assert liger_output.logits is None
    assert native_output.aux_loss is not None
    assert liger_output.aux_loss is not None
    assert_verbose_allclose(liger_base.loss, native_base_loss, atol=2e-2, rtol=2e-3)
    assert_verbose_allclose(liger_output.aux_loss, native_output.aux_loss, atol=2e-2, rtol=2e-3)
    assert_verbose_allclose(liger_output.loss, native_output.loss, atol=2e-2, rtol=2e-3)
    assert_verbose_allclose(
        native_output.loss,
        native_base_loss + config.router_aux_loss_coef * native_output.aux_loss,
        atol=2e-6,
        rtol=2e-6,
    )
    assert_verbose_allclose(
        liger_output.loss,
        liger_base.loss + config.router_aux_loss_coef * liger_output.aux_loss,
        atol=2e-6,
        rtol=2e-6,
    )

    native_output.loss.backward()
    liger_output.loss.backward()
    native_parameters = dict(native_model.named_parameters())
    for name, parameter in liger_model.named_parameters():
        native_grad = native_parameters[name].grad
        if native_grad is None:
            assert parameter.grad is None
            continue
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert_verbose_allclose(
            parameter.grad,
            native_grad,
            atol=2e-2,
            rtol=2e-3,
            extra_info=f"gradient mismatch for {name}\n",
        )


@requires_qwen4_exp
@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_qwen4_exp_lce_shift_labels_and_embedding_tying_native_liger(tie_word_embeddings):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    from liger_kernel.transformers.model.qwen4_exp import lce_forward

    set_seed(42)
    config = _qwen4_exp_hybrid_config()
    config.num_hidden_layers = 1
    config.layer_types = ["linear_attention"]
    config.ple_layer_ids = []
    config.tie_word_embeddings = tie_word_embeddings
    native_model = Qwen4ExpForCausalLM(config).to(device).train()
    liger_model = copy.deepcopy(native_model)
    _patch_qwen4_exp_candidate(liger_model)
    liger_model.forward = MethodType(lce_forward, liger_model)

    assert (native_model.model.embed_tokens.weight is native_model.lm_head.weight) is tie_word_embeddings
    assert (liger_model.model.embed_tokens.weight is liger_model.lm_head.weight) is tie_word_embeddings
    assert native_model.state_dict().keys() == liger_model.state_dict().keys()

    input_ids = torch.tensor([[11, 12, 13, 14]], device=device)
    labels = torch.zeros_like(input_ids)
    shift_labels = torch.tensor([[12, 13, 14, -100]], device=device)
    native_output = native_model(
        input_ids=input_ids,
        labels=labels,
        shift_labels=shift_labels,
        use_cache=False,
    )
    expected_shift_loss = torch.nn.functional.cross_entropy(
        native_output.logits.float().view(-1, config.vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    assert_verbose_allclose(native_output.loss, expected_shift_loss, atol=2e-6, rtol=2e-6)
    liger_output = liger_model(
        input_ids=input_ids,
        labels=labels,
        shift_labels=shift_labels,
        skip_logits=True,
        use_cache=False,
    )
    assert liger_output.logits is None
    assert_verbose_allclose(liger_output.loss, native_output.loss, atol=2e-4, rtol=2e-4)

    native_output.loss.backward()
    liger_output.loss.backward()
    assert_verbose_allclose(
        liger_model.model.embed_tokens.weight.grad,
        native_model.model.embed_tokens.weight.grad,
        atol=2e-4,
        rtol=2e-4,
    )
    assert_verbose_allclose(
        liger_model.lm_head.weight.grad,
        native_model.lm_head.weight.grad,
        atol=2e-4,
        rtol=2e-4,
    )


@pytest.mark.parametrize("shape, n_groups", [((2, 13, 256), 4), ((3, 5, 154), 2)])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        torch.float32,
    ],
)
def test_qwen4_exp_hyper_connection_pre_forward_backward(shape, n_groups, dtype):
    set_seed(42)
    mix_logits = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    normalized_input = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    grad_output = torch.randn((*shape[:-1], shape[-1] // n_groups), device=device, dtype=dtype)

    output = liger_qwen4_exp_hyper_connection_pre(mix_logits, normalized_input, n_groups)
    output.backward(grad_output)
    actual_grads = [tensor.grad.detach().float().clone() for tensor in (mix_logits, normalized_input)]

    ref_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (mix_logits, normalized_input)]
    reference = qwen4_exp_hyper_connection_pre_ref(*ref_inputs, n_groups)
    reference.backward(grad_output)
    reference_grads = [tensor.grad.detach().float() for tensor in ref_inputs]

    tolerance = (8e-3, 2e-3) if dtype == torch.bfloat16 else (2e-5, 2e-5)
    assert_verbose_allclose(output.float(), reference.float(), atol=tolerance[0], rtol=tolerance[1])
    for actual, expected in zip(actual_grads, reference_grads):
        assert_verbose_allclose(actual, expected, atol=tolerance[0], rtol=tolerance[1])


@pytest.mark.parametrize(
    "shape, n_groups",
    [
        ((2, 13, 64), 4),
        ((3, 5, 77), 2),
        ((2, 11, 65), 3),
        ((2, 7, 63), 8),
        pytest.param((1, 2048, 2048), 4, id="production-shape"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        torch.float32,
    ],
)
def test_qwen4_exp_gr_write_forward_backward(shape, n_groups, dtype):
    set_seed(42)
    block_output = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    residual = torch.randn(*shape[:-1], n_groups * shape[-1], device=device, dtype=dtype, requires_grad=True)
    write_logits = torch.randn(*shape[:-1], n_groups, device=device, dtype=dtype, requires_grad=True)
    grad_output = torch.randn_like(residual)

    output = liger_qwen4_exp_gr_write(block_output, residual, write_logits)
    output.backward(grad_output)
    actual_grads = [tensor.grad.detach().float().clone() for tensor in (block_output, residual, write_logits)]

    ref_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (block_output, residual, write_logits)]
    reference = qwen4_exp_gr_write_ref(*ref_inputs)
    reference.backward(grad_output)
    reference_grads = [tensor.grad.detach().float() for tensor in ref_inputs]

    if dtype == torch.bfloat16:
        tolerances = ((2e-2, 2e-2), (1e-3, 1e-3), (0.0, 0.0), (2e-2, 2e-2))
    else:
        tolerances = ((2e-5, 2e-5),) * 4
    for actual, expected, (atol, rtol) in zip(
        (output.float(), *actual_grads), (reference.float(), *reference_grads), tolerances
    ):
        assert_verbose_allclose(actual, expected, atol=atol, rtol=rtol)
