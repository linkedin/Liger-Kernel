import copy
import inspect

from types import MethodType

import pytest
import torch

from test.utils import assert_verbose_allclose
from test.utils import supports_bfloat16
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import MoeModelOutputWithPast

try:
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
except ModuleNotFoundError as exc:
    if exc.name != "transformers.models.qwen4_exp":
        raise
    pytest.skip("Qwen4Exp is unavailable in this Transformers version", allow_module_level=True)

from liger_kernel.ops.qwen4_exp import LigerGroupRMSNormFusedFunction
from liger_kernel.ops.qwen4_exp import LigerGroupRMSNormWrite4Function
from liger_kernel.transformers.model.output_classes import LigerMoeCausalLMOutputWithPast
from liger_kernel.transformers.model.qwen4_exp import _can_use_fused_lce_lm_head
from liger_kernel.transformers.model.qwen4_exp import lce_forward
from liger_kernel.transformers.monkey_patch import _patch_rms_norm_module
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen4_exp
from liger_kernel.transformers.qwen4_exp import liger_qwen4_exp_gated_residual_forward
from liger_kernel.transformers.qwen4_exp import liger_qwen4_exp_ngram_embedding_forward
from liger_kernel.transformers.qwen4_exp import patch_qwen4_exp_text_module_for_ngram
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rms_norm import _liger_rms_norm_supports_grouped
from liger_kernel.utils import infer_device

device = infer_device()
pytestmark = pytest.mark.usefixtures("qwen4_exp_globals")
requires_nvidia = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Qwen4Exp integration regression requires NVIDIA CUDA",
)


def _tiny_config(*, return_dict=True, ple=False):
    return Qwen4ExpTextConfig(
        return_dict=return_dict,
        dtype=torch.bfloat16,
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
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
        indexer_budget=2,
        indexer_compress_ratio=1,
        ple_layer_ids=[1] if ple else [],
        ple_embed_dim=16,
        ngram_size=2,
        heads_per_ngram=1,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=32,
        eos_token_id=2,
        layer_types=["linear_attention"],
    )


class RecordingTextModel(torch.nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.calls = []

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        self.calls.append(kwargs.copy())
        source = inputs_embeds if inputs_embeds is not None else input_ids
        batch_size, sequence_length = source.shape[:2]
        hidden = torch.arange(
            batch_size * sequence_length * self.hidden_size,
            device=source.device,
            dtype=torch.float32,
        ).reshape(batch_size, sequence_length, self.hidden_size)
        hidden = hidden.div(97).to(torch.float32)
        if inputs_embeds is not None:
            hidden = hidden + inputs_embeds
        router_logits = (
            torch.arange(
                batch_size * sequence_length * self.num_experts,
                device=source.device,
                dtype=torch.float32,
            ).reshape(batch_size * sequence_length, self.num_experts)
            / 17,
        )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden,
            past_key_values=(torch.tensor(11, device=source.device),),
            hidden_states=(hidden + 1,),
            attentions=(hidden[..., :1] + 2,),
            router_logits=router_logits,
        )


class RecordingLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)
        self.sequence_lengths = []

    def forward(self, hidden_states):
        self.sequence_lengths.append(hidden_states.shape[-2])
        return super().forward(hidden_states)


def _stub_causal_lm(config):
    model = Qwen4ExpForCausalLM(config)
    model.model = RecordingTextModel(config.hidden_size, config.num_experts)
    model.lm_head = RecordingLinear(config.hidden_size, config.vocab_size)
    model.eval()
    return model


def _assert_output_values_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert_verbose_allclose(actual, expected, atol=0.0, rtol=0.0)
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            _assert_output_values_equal(actual_value, expected_value)
    else:
        assert actual == expected


@pytest.mark.parametrize("entrypoint", ["from_config", "from_pretrained"])
def test_qwen4_exp_auto_liger_loads_composite_config_text_model(monkeypatch, tmp_path, entrypoint):
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

    from liger_kernel.transformers import AutoLigerKernelForCausalLM
    from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
    from liger_kernel.transformers.rms_norm import LigerRMSNormForQwen4Exp

    config = Qwen4ExpConfig(text_config=_tiny_config().to_dict())
    native = AutoModelForCausalLM.from_config(config)
    assert isinstance(native, Qwen4ExpForCausalLM)
    if entrypoint == "from_pretrained":
        native.save_pretrained(tmp_path)
        config.save_pretrained(tmp_path)
        model = AutoLigerKernelForCausalLM.from_pretrained(tmp_path, engram=False, hyper_connection=False)
        for name, weight in native.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], weight, atol=0, rtol=0)
    else:
        model = AutoLigerKernelForCausalLM.from_config(config, engram=False, hyper_connection=False)
    assert isinstance(model, Qwen4ExpForCausalLM)
    assert model.config.model_type == "qwen4_exp_text"
    assert model.forward.__func__ is lce_forward
    assert isinstance(model.model.layers[0].attn_hyper_connection.hc_norm, LigerRMSNormForQwen4Exp)
    # Causal-LM config resolution must not advertise support for multimodal instances.
    assert "qwen4_exp" not in MODEL_TYPE_TO_APPLY_LIGER_FN


@requires_nvidia
@pytest.mark.parametrize(
    "return_write_logits",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
    ids=["grouped", "write4"],
)
@pytest.mark.parametrize("hook_type", ["forward_pre", "forward", "full_backward_pre", "full_backward"])
@pytest.mark.parametrize("global_hook", [False, True], ids=["local", "global"])
def test_qwen4_exp_grouped_fusion_preserves_norm_hooks(monkeypatch, return_write_logits, hook_type, global_hook):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    torch.manual_seed(73)
    dtype = torch.bfloat16 if return_write_logits else torch.float32
    native = Qwen4ExpTextGatedResidual(_tiny_config()).to(device, dtype)
    candidate = copy.deepcopy(native)
    _patch_rms_norm_module(candidate.hc_norm, offset=1.0, casting_mode="gemma", in_place=False)
    candidate.forward = MethodType(liger_qwen4_exp_gated_residual_forward, candidate)
    norms = (native.hc_norm, candidate.hc_norm)
    calls = [0, 0]

    def hook(module, *args):
        if module not in norms:
            return None
        calls[norms.index(module)] += 1
        if hook_type == "forward":
            return args[1] * 2
        # Forward-pre hooks transform inputs; backward hooks transform gradients.
        return tuple(value * 0.5 if value is not None else None for value in args[0])

    def reject_fusion(*args):
        pytest.fail("Norm hooks must execute through the module call, outside grouped fusion")

    monkeypatch.setattr(LigerGroupRMSNormFusedFunction, "apply", reject_fusion)
    monkeypatch.setattr(LigerGroupRMSNormWrite4Function, "apply", reject_fusion)
    if global_hook:
        handles = [getattr(torch.nn.modules.module, f"register_module_{hook_type}_hook")(hook)]
    else:
        handles = [getattr(norm, f"register_{hook_type}_hook")(hook) for norm in norms]
    try:
        x = torch.randn(2, 5, 64, device=device, dtype=dtype, requires_grad=True)
        candidate_x = x.detach().clone().requires_grad_(True)
        expected = native(x)
        actual = candidate(candidate_x, return_write_logits=return_write_logits)
        if return_write_logits:
            actual = (*actual[:2], 2 * torch.sigmoid(actual[2]))
        grads = tuple(torch.randn_like(value) for value in expected)
        torch.autograd.backward(expected, grads)
        torch.autograd.backward(actual, grads)
        assert calls == [1, 1]
        tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-5
        for output, reference in zip(actual, expected):
            torch.testing.assert_close(output, reference, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(candidate_x.grad, x.grad, atol=tolerance, rtol=tolerance)
        for parameter, reference in zip(candidate.parameters(), native.parameters()):
            torch.testing.assert_close(parameter.grad, reference.grad, atol=tolerance, rtol=tolerance)
    finally:
        for handle in handles:
            handle.remove()


def test_qwen4_exp_lce_signature_preserves_upstream_positional_prefix():
    from transformers.models.qwen4_exp import modeling_qwen4_exp

    native_parameters = list(inspect.signature(modeling_qwen4_exp.Qwen4ExpForCausalLM.forward).parameters.values())
    liger_parameters = list(inspect.signature(lce_forward).parameters.values())
    expected_names = [
        "self",
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "inputs_embeds",
        "labels",
        "use_cache",
        "output_router_logits",
        "logits_to_keep",
    ]
    assert [parameter.name for parameter in native_parameters[:10]] == expected_names
    assert [parameter.name for parameter in liger_parameters[:10]] == expected_names
    for native_parameter, liger_parameter in zip(native_parameters[:10], liger_parameters[:10]):
        assert native_parameter.kind is liger_parameter.kind
        assert native_parameter.default == liger_parameter.default
    assert inspect.signature(lce_forward).parameters["cache_position"].kind is inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(lce_forward).parameters["skip_logits"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("patch_globally", [False, True], ids=["instance", "global-before-construction"])
def test_qwen4_exp_lce_positional_binding_matches_native(monkeypatch, patch_globally):
    from transformers.models.qwen4_exp import modeling_qwen4_exp

    native_forward = modeling_qwen4_exp.Qwen4ExpForCausalLM.forward
    config = _tiny_config()
    reference = _stub_causal_lm(config)
    if patch_globally:
        apply_liger_kernel_to_qwen4_exp(
            fused_linear_cross_entropy=True,
            rms_norm=False,
            swiglu=False,
            engram=False,
            hyper_connection=False,
        )
        candidate = _stub_causal_lm(config)
    else:
        candidate = _stub_causal_lm(config)
        apply_liger_kernel_to_qwen4_exp(
            fused_linear_cross_entropy=True,
            rms_norm=False,
            swiglu=False,
            engram=False,
            hyper_connection=False,
            model=candidate,
        )
    candidate.lm_head.load_state_dict(reference.lm_head.state_dict())
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(4).unsqueeze(0)
    cache_position = torch.arange(4)
    positional_args = (input_ids, attention_mask, position_ids, None, None, None, False, False, 2)

    expected = native_forward(reference, *positional_args, cache_position=cache_position)
    actual = candidate(*positional_args, cache_position=cache_position)

    assert reference.model.calls[-1]["output_router_logits"] is False
    assert candidate.model.calls[-1]["output_router_logits"] is False
    assert torch.equal(reference.model.calls[-1]["cache_position"], cache_position)
    assert torch.equal(candidate.model.calls[-1]["cache_position"], cache_position)
    assert reference.lm_head.sequence_lengths[-1] == candidate.lm_head.sequence_lengths[-1] == 2
    _assert_output_values_equal(actual.logits, expected.logits)


@pytest.mark.parametrize(
    "config_return_dict, explicit_return_dict",
    [(True, None), (False, None), (True, True), (True, False)],
    ids=["config-true", "config-false", "explicit-true", "explicit-false"],
)
def test_qwen4_exp_lce_return_dict_matches_native(config_return_dict, explicit_return_dict):
    config = _tiny_config(return_dict=config_return_dict)
    reference = _stub_causal_lm(config)
    candidate = _stub_causal_lm(copy.deepcopy(config))
    candidate.load_state_dict(reference.state_dict())
    candidate.forward = MethodType(lce_forward, candidate)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[2, 3, 4, 5]])
    kwargs = {
        "labels": labels,
        "output_router_logits": True,
        "output_hidden_states": True,
        "output_attentions": True,
        "use_cache": True,
    }
    if explicit_return_dict is not None:
        kwargs["return_dict"] = explicit_return_dict

    expected = reference(input_ids, **kwargs)
    actual = candidate(input_ids, skip_logits=False, **kwargs)

    expected_return_dict = config_return_dict if explicit_return_dict is None else explicit_return_dict
    if expected_return_dict:
        assert type(expected).__name__ == "MoeCausalLMOutputWithPast"
        assert isinstance(actual, type(expected))
        assert isinstance(actual, LigerMoeCausalLMOutputWithPast)
        for field in ("loss", "aux_loss", "logits", "past_key_values", "hidden_states", "attentions", "router_logits"):
            _assert_output_values_equal(getattr(actual, field), getattr(expected, field))
    else:
        assert type(actual) is type(expected) is tuple
        _assert_output_values_equal(actual, expected)


@requires_nvidia
def test_qwen4_exp_lce_forces_real_text_model_structured_output_when_config_returns_tuple():
    reference_config = _tiny_config(return_dict=True)
    reference_config.use_cache = False
    candidate_config = copy.deepcopy(reference_config)
    candidate_config.return_dict = False
    reference = Qwen4ExpForCausalLM(reference_config).to("cuda").eval()
    candidate = Qwen4ExpForCausalLM(candidate_config).to("cuda").eval()
    candidate.load_state_dict(reference.state_dict())
    candidate.forward = MethodType(lce_forward, candidate)
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    labels = torch.tensor([[2, 3, 4, 5]], device="cuda")

    with torch.no_grad():
        expected = reference(
            input_ids,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
            return_dict=False,
        )
        actual = candidate(
            input_ids,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
            skip_logits=False,
        )

    assert type(actual) is type(expected) is tuple
    _assert_output_values_equal(actual, expected)


@requires_nvidia
@pytest.mark.parametrize("return_dict", [True, False], ids=["structured", "tuple"])
def test_qwen4_exp_lce_preserves_fused_token_metrics(return_dict):
    config = _tiny_config(return_dict=return_dict)
    model = _stub_causal_lm(config).to("cuda")
    model.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False, device="cuda")
    model.train()
    model.forward = MethodType(lce_forward, model)
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    labels = torch.tensor([[2, 3, 4, 5]], device="cuda")

    output = model(
        input_ids,
        labels=labels,
        skip_logits=True,
        return_token_accuracy=True,
        return_predicted_tokens=True,
    )

    if return_dict:
        assert output.logits is None
        assert output.token_accuracy is not None
        assert output.predicted_tokens is not None
    else:
        assert type(output) is tuple
        assert output[-2].ndim == 0
        assert output[-1].numel() == labels.numel()


@requires_nvidia
@pytest.mark.parametrize("patch_first", [True, False], ids=["liger-then-accelerate", "accelerate-then-liger"])
def test_qwen4_exp_ngram_preserves_accelerate_offload_wrapper(patch_first):
    accelerate_hooks = pytest.importorskip("accelerate.hooks")
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

    config = _tiny_config(ple=True)
    reference = Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, 0, 0).to("cuda")
    candidate = copy.deepcopy(reference)
    weights_map = {name: tensor.detach().cpu().clone() for name, tensor in candidate.state_dict().items()}
    if patch_first:
        patch_qwen4_exp_text_module_for_ngram(candidate, Qwen4ExpTextNGramEmbedding)
    accelerate_hooks.attach_align_device_hook(
        candidate,
        execution_device=torch.device("cuda", torch.cuda.current_device()),
        offload=True,
        weights_map=weights_map,
        preload_module_classes=[type(candidate).__name__],
    )
    accelerate_forward = candidate.forward
    if not patch_first:
        patch_qwen4_exp_text_module_for_ngram(candidate, Qwen4ExpTextNGramEmbedding)
        assert candidate.forward is accelerate_forward

    assert (
        getattr(candidate._old_forward, "__func__", candidate._old_forward) is liger_qwen4_exp_ngram_embedding_forward
    )
    assert candidate.__dict__["_liger_qwen4_exp_native_ngram_forward"] is not accelerate_forward
    assert {parameter.device.type for parameter in candidate.parameters()} == {"meta"}
    reference_cache = DynamicCache(config=config)
    candidate_cache = DynamicCache(config=config)
    update_calls = 0
    update_conv_state = candidate_cache.update_conv_state

    def counted_update(*args, **kwargs):
        nonlocal update_calls
        update_calls += 1
        return update_conv_state(*args, **kwargs)

    candidate_cache.update_conv_state = counted_update
    for input_ids in (torch.tensor([[11, 12]], device="cuda"), torch.tensor([[2, 21]], device="cuda")):
        expected = reference(input_ids, reference_cache)
        actual = candidate(input_ids, candidate_cache)
        assert_verbose_allclose(actual, expected, atol=0.0, rtol=0.0)
        assert torch.equal(reference_cache.layers[0].conv_states[2], candidate_cache.layers[0].conv_states[2])
        assert {parameter.device.type for parameter in candidate.parameters()} == {"meta"}
    assert update_calls == 2


@requires_nvidia
@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
@pytest.mark.parametrize("group_size", [None, 8], ids=["ordinary", "grouped"])
@pytest.mark.parametrize("patch_first", [True, False], ids=["liger-then-accelerate", "accelerate-then-liger"])
def test_qwen4_exp_rms_norm_preserves_accelerate_offload_wrapper(group_size, patch_first):
    accelerate_hooks = pytest.importorskip("accelerate.hooks")
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm

    reference = Qwen4ExpTextRMSNorm(16, group_size=group_size, eps=1e-5).to("cuda", torch.bfloat16)
    candidate = copy.deepcopy(reference)
    weights_map = {name: tensor.detach().cpu().clone() for name, tensor in candidate.state_dict().items()}
    if patch_first:
        _patch_rms_norm_module(candidate, offset=1.0, casting_mode="gemma", in_place=False)
    accelerate_hooks.attach_align_device_hook(
        candidate,
        execution_device=torch.device("cuda", torch.cuda.current_device()),
        offload=True,
        weights_map=weights_map,
        preload_module_classes=[type(candidate).__name__],
    )
    accelerate_forward = candidate.forward
    if not patch_first:
        _patch_rms_norm_module(candidate, offset=1.0, casting_mode="gemma", in_place=False)
        assert candidate.forward is accelerate_forward

    assert getattr(candidate._old_forward, "__func__", candidate._old_forward) is LigerRMSNorm.forward
    assert candidate._liger_rms_norm_patched is True
    assert candidate._liger_rms_norm_supports_grouped is _liger_rms_norm_supports_grouped()
    assert {parameter.device.type for parameter in candidate.parameters()} == {"meta"}
    for _ in range(2):
        hidden_states = torch.randn(2, 3, 16, device="cuda", dtype=torch.bfloat16)
        expected = reference(hidden_states)
        actual = candidate(hidden_states)
        assert_verbose_allclose(actual, expected, atol=8e-3, rtol=2e-3)
        assert {parameter.device.type for parameter in candidate.parameters()} == {"meta"}


@requires_nvidia
@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_qwen4_exp_grouped_rms_norm_native_policy_preserves_accelerate_wrapper(monkeypatch):
    accelerate_hooks = pytest.importorskip("accelerate.hooks")
    from transformers.models.qwen4_exp import modeling_qwen4_exp

    import liger_kernel.transformers.monkey_patch as monkey_patch_module

    native_forward = modeling_qwen4_exp.Qwen4ExpTextRMSNorm.forward
    config = _tiny_config(ple=True)
    model = modeling_qwen4_exp.Qwen4ExpForCausalLM(config).to("cuda", torch.bfloat16)
    grouped = next(module for module in model.modules() if getattr(module, "group_size", None) is not None)
    reference = copy.deepcopy(grouped)
    weights_map = {name: tensor.detach().cpu().clone() for name, tensor in grouped.state_dict().items()}
    accelerate_hooks.attach_align_device_hook(
        grouped,
        execution_device=torch.device("cuda", torch.cuda.current_device()),
        offload=True,
        weights_map=weights_map,
        preload_module_classes=[type(grouped).__name__],
    )
    accelerate_forward = grouped.forward
    monkeypatch.setattr(monkey_patch_module, "_liger_rms_norm_supports_grouped", lambda: False)

    apply_liger_kernel_to_qwen4_exp(
        fused_linear_cross_entropy=False,
        swiglu=False,
        engram=False,
        hyper_connection=False,
        model=model,
    )

    assert grouped.forward is accelerate_forward
    assert getattr(grouped._old_forward, "__func__", grouped._old_forward) is native_forward
    assert not getattr(grouped, "_liger_rms_norm_patched", False)
    assert {parameter.device.type for parameter in grouped.parameters()} == {"meta"}
    for _ in range(2):
        hidden_states = torch.randn(2, 3, grouped.weight.numel(), device="cuda", dtype=torch.bfloat16)
        expected = reference(hidden_states)
        actual = grouped(hidden_states)
        assert_verbose_allclose(actual, expected, atol=0.0, rtol=0.0)
        assert {parameter.device.type for parameter in grouped.parameters()} == {"meta"}


@pytest.mark.parametrize("patch_first", [True, False], ids=["liger-then-accelerate", "accelerate-then-liger"])
def test_qwen4_exp_top_level_lce_preserves_accelerate_wrapper(patch_first):
    accelerate_hooks = pytest.importorskip("accelerate.hooks")
    config = _tiny_config()
    reference = _stub_causal_lm(config)
    candidate = _stub_causal_lm(copy.deepcopy(config))
    candidate.load_state_dict(reference.state_dict())
    if patch_first:
        apply_liger_kernel_to_qwen4_exp(
            fused_linear_cross_entropy=True,
            rms_norm=False,
            swiglu=False,
            engram=False,
            hyper_connection=False,
            model=candidate,
        )
    accelerate_hooks.add_hook_to_module(candidate, accelerate_hooks.ModelHook())
    accelerate_forward = candidate.forward
    if not patch_first:
        apply_liger_kernel_to_qwen4_exp(
            fused_linear_cross_entropy=True,
            rms_norm=False,
            swiglu=False,
            engram=False,
            hyper_connection=False,
            model=candidate,
        )
        assert candidate.forward is accelerate_forward

    assert getattr(candidate._old_forward, "__func__", candidate._old_forward) is lce_forward
    for _ in range(2):
        input_ids = torch.tensor([[1, 2, 3, 4]])
        expected = reference(input_ids, logits_to_keep=2)
        actual = candidate(input_ids, logits_to_keep=2)
        _assert_output_values_equal(actual.logits, expected.logits)


@requires_nvidia
def test_qwen4_exp_fused_lce_falls_back_for_accelerate_offloaded_lm_head():
    accelerate_hooks = pytest.importorskip("accelerate.hooks")
    config = _tiny_config()
    reference = _stub_causal_lm(config).to("cuda")
    candidate = _stub_causal_lm(copy.deepcopy(config)).to("cuda")
    reference.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False, device="cuda")
    candidate.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False, device="cuda")
    candidate.load_state_dict(reference.state_dict())
    weights_map = {name: tensor.detach().cpu().clone() for name, tensor in candidate.lm_head.state_dict().items()}
    accelerate_hooks.attach_align_device_hook(
        candidate.lm_head,
        execution_device=torch.device("cuda", torch.cuda.current_device()),
        offload=True,
        weights_map=weights_map,
        preload_module_classes=[type(candidate.lm_head).__name__],
    )
    candidate.forward = MethodType(lce_forward, candidate)
    assert not _can_use_fused_lce_lm_head(candidate.lm_head, torch.empty(1, device="cuda"))
    assert {parameter.device.type for parameter in candidate.lm_head.parameters()} == {"meta"}
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    labels = torch.tensor([[2, 3, 4, 5]], device="cuda")
    for _ in range(2):
        expected = reference(input_ids, labels=labels)
        actual = candidate(input_ids, labels=labels, skip_logits=True)
        assert actual.logits is not None
        assert_verbose_allclose(actual.logits, expected.logits, atol=0.0, rtol=0.0)
        assert_verbose_allclose(actual.loss, expected.loss, atol=0.0, rtol=0.0)
        assert {parameter.device.type for parameter in candidate.lm_head.parameters()} == {"meta"}


def test_qwen4_exp_fused_lce_lm_head_eligibility_is_conservative():
    class LinearSubclass(torch.nn.Linear):
        pass

    class IdentityParametrization(torch.nn.Module):
        def forward(self, weight):
            return weight

    hidden_states = torch.randn(2, 3, 8)
    plain = torch.nn.Linear(8, 16, bias=False)
    tied = torch.nn.Linear(8, 16, bias=False)
    embedding = torch.nn.Embedding(16, 8)
    tied.weight = embedding.weight
    custom = LinearSubclass(8, 16, bias=False)
    parametrized = torch.nn.Linear(8, 16, bias=False)
    torch.nn.utils.parametrize.register_parametrization(parametrized, "weight", IdentityParametrization())
    hooked = torch.nn.Linear(8, 16, bias=False)
    hook = hooked.register_forward_hook(lambda _module, _args, output: output)
    instance_overridden = torch.nn.Linear(8, 16, bias=False)
    instance_overridden.forward = MethodType(
        lambda self, hidden: torch.nn.functional.linear(hidden, self.weight), instance_overridden
    )
    biased = torch.nn.Linear(8, 16, bias=True)

    assert _can_use_fused_lce_lm_head(plain, hidden_states)
    assert _can_use_fused_lce_lm_head(tied, hidden_states)
    assert not _can_use_fused_lce_lm_head(custom, hidden_states)
    assert not _can_use_fused_lce_lm_head(parametrized, hidden_states)
    assert not _can_use_fused_lce_lm_head(hooked, hidden_states)
    assert not _can_use_fused_lce_lm_head(instance_overridden, hidden_states)
    assert not _can_use_fused_lce_lm_head(biased, hidden_states)
    hook.remove()


@requires_nvidia
@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_qwen4_exp_write4_rejects_custom_linear_semantics(monkeypatch):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    class OffsetLinear(torch.nn.Linear):
        def forward(self, hidden_states):
            return super().forward(hidden_states) + 0.25

    config = _tiny_config()
    reference = Qwen4ExpTextGatedResidual(config, use_combine=True).to("cuda", torch.bfloat16)
    custom = OffsetLinear(
        reference.block_inject_weight.in_features,
        reference.block_inject_weight.out_features,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
    )
    custom.load_state_dict(reference.block_inject_weight.state_dict())
    reference.block_inject_weight = custom
    candidate = copy.deepcopy(reference)
    _patch_rms_norm_module(candidate.hc_norm, offset=1.0, casting_mode="gemma", in_place=False)
    candidate.forward = MethodType(liger_qwen4_exp_gated_residual_forward, candidate)
    monkeypatch.setattr(
        LigerGroupRMSNormWrite4Function,
        "apply",
        lambda *args, **kwargs: pytest.fail("Write4 direct-weight path ran for a custom Linear subclass"),
    )
    hyper_input = torch.randn(2, 3, config.hc_count * config.hidden_size, device="cuda", dtype=torch.bfloat16)
    expected_mixed, expected_residual, expected_injection = reference(hyper_input)
    actual_mixed, actual_residual, actual_write_logits = candidate(hyper_input, return_write_logits=True)
    assert_verbose_allclose(actual_mixed, expected_mixed, atol=8e-3, rtol=2e-3)
    assert torch.equal(actual_residual, expected_residual)
    assert_verbose_allclose(2 * torch.sigmoid(actual_write_logits), expected_injection, atol=8e-3, rtol=2e-3)


@requires_nvidia
@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_qwen4_exp_grouped_multi_gradient_backward_compiles_with_python_offset():
    x = torch.randn(2, 3, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gradients = tuple(torch.randn_like(x) for _ in range(3))

    def forward(current_x, current_weight):
        outputs = LigerGroupRMSNormFusedFunction.apply(current_x, current_weight, 1e-6, 1.0, "gemma", 4)
        return sum((output * gradient).sum() for output, gradient in zip(outputs, gradients))

    torch.compiler.reset()
    compiled = torch.compile(forward, fullgraph=True)
    loss = compiled(x, weight)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(weight.grad).all()
    torch.compiler.reset()


def test_qwen4_exp_global_and_instance_rms_norm_lifecycle_is_idempotent(monkeypatch):
    from transformers.models.qwen4_exp import modeling_qwen4_exp

    native_rms_norm_class = modeling_qwen4_exp.Qwen4ExpTextRMSNorm
    config = _tiny_config(ple=True)
    instance_then_global = modeling_qwen4_exp.Qwen4ExpForCausalLM(config)
    global_then_instance = modeling_qwen4_exp.Qwen4ExpForCausalLM(copy.deepcopy(config))
    expected_norms = [module for module in global_then_instance.modules() if isinstance(module, native_rms_norm_class)]
    expected_total = len(expected_norms)
    expected_grouped = sum(getattr(module, "group_size", None) is not None for module in expected_norms)
    expected_ordinary = expected_total - expected_grouped

    apply_liger_kernel_to_qwen4_exp(
        fused_linear_cross_entropy=False,
        swiglu=False,
        engram=False,
        hyper_connection=False,
        model=instance_then_global,
    )
    apply_liger_kernel_to_qwen4_exp(
        fused_linear_cross_entropy=False,
        swiglu=False,
        engram=False,
        hyper_connection=False,
    )
    apply_liger_kernel_to_qwen4_exp(
        fused_linear_cross_entropy=False,
        swiglu=False,
        engram=False,
        hyper_connection=False,
        model=global_then_instance,
    )
    constructed_after_global = modeling_qwen4_exp.Qwen4ExpForCausalLM(copy.deepcopy(config))
    apply_liger_kernel_to_qwen4_exp(
        fused_linear_cross_entropy=False,
        swiglu=False,
        engram=False,
        hyper_connection=False,
    )
    apply_liger_kernel_to_qwen4_exp(
        fused_linear_cross_entropy=False,
        swiglu=False,
        engram=False,
        hyper_connection=False,
        model=global_then_instance,
    )

    supports_grouped = _liger_rms_norm_supports_grouped()
    for model in (instance_then_global, global_then_instance, constructed_after_global):
        norms = [
            module
            for module in model.modules()
            if isinstance(module, native_rms_norm_class) or type(module).__name__ == "LigerRMSNormForQwen4Exp"
        ]
        ordinary = [module for module in norms if getattr(module, "group_size", None) is None]
        grouped = [module for module in norms if getattr(module, "group_size", None) is not None]
        assert len(norms) == expected_total
        assert len(ordinary) == expected_ordinary
        assert len(grouped) == expected_grouped
        assert all(module._liger_rms_norm_patched is True for module in ordinary)
        assert all(module.forward.__func__ is LigerRMSNorm.forward for module in ordinary)
        if supports_grouped:
            assert all(module._liger_rms_norm_patched is True for module in grouped)
            assert all(module.forward.__func__ is LigerRMSNorm.forward for module in grouped)
        else:
            assert all(not getattr(module, "_liger_rms_norm_patched", False) for module in grouped)


@requires_nvidia
def test_qwen4_exp_ngram_child_offload_materializes_in_child_hook():
    """Child-offloaded n-gram embedding must keep ids on CUDA and restore offload state."""
    accelerate_hooks = pytest.importorskip("accelerate.hooks")
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

    config = _tiny_config(ple=True)
    reference = Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, 0, 0).to("cuda")
    candidate = copy.deepcopy(reference)
    patch_qwen4_exp_text_module_for_ngram(candidate, Qwen4ExpTextNGramEmbedding)
    child_weights_map = {
        name: tensor.detach().cpu().clone() for name, tensor in candidate.ngram_embedding.state_dict().items()
    }
    accelerate_hooks.attach_align_device_hook(
        candidate.ngram_embedding,
        execution_device=torch.device("cuda", torch.cuda.current_device()),
        offload=True,
        weights_map=child_weights_map,
    )

    assert not hasattr(candidate, "_hf_hook")
    assert hasattr(candidate.ngram_embedding, "_hf_hook")
    assert candidate.ngram_embedding.weight.device.type == "meta"

    seen_weight_devices = []
    seen_input_devices = []

    def record_pre(module, args):
        seen_weight_devices.append(module.weight.device.type)
        seen_input_devices.append(args[0].device.type)

    pre_handle = candidate.ngram_embedding.register_forward_pre_hook(record_pre)
    try:
        reference_cache = DynamicCache(config=config)
        candidate_cache = DynamicCache(config=config)
        for input_ids in (torch.tensor([[11, 12]], device="cuda"), torch.tensor([[2, 21]], device="cuda")):
            assert candidate.ngram_embedding.weight.device.type == "meta"
            expected = reference(input_ids, reference_cache)
            actual = candidate(input_ids, candidate_cache)
            assert_verbose_allclose(actual, expected, atol=0.0, rtol=0.0)
            assert torch.equal(reference_cache.layers[0].conv_states[2], candidate_cache.layers[0].conv_states[2])
            assert candidate.ngram_embedding.weight.device.type == "meta"
    finally:
        pre_handle.remove()

    assert seen_weight_devices
    assert all(seen == "meta" for seen in seen_weight_devices)
    assert all(device == "cuda" for device in seen_input_devices)


@requires_nvidia
@pytest.mark.parametrize("global_hook", [False, True], ids=["local", "global"])
@pytest.mark.parametrize("hook_type", ["forward", "full_backward_pre", "full_backward"])
def test_qwen4_exp_decoder_preserves_hyper_connection_hook_contract(monkeypatch, global_hook, hook_type):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    from liger_kernel.ops.qwen4_exp import LigerQwen4ExpGRWriteFunction
    from liger_kernel.transformers.qwen4_exp import _uses_liger_hyper_connection
    from liger_kernel.transformers.qwen4_exp import patch_qwen4_exp_text_module_for_hyper_connection

    class Attention(torch.nn.Module):
        def forward(self, hidden_states, position_embeddings, **kwargs):
            return hidden_states * 0.5, None

    config = _tiny_config()
    reference = object.__new__(Qwen4ExpTextDecoderLayer)
    torch.nn.Module.__init__(reference)
    reference.ple = None
    reference.layer_type = "full_attention"
    reference.self_attn = Attention()
    reference.mlp = torch.nn.Identity()
    reference.attn_hyper_connection = Qwen4ExpTextGatedResidual(config)
    reference.mlp_hyper_connection = Qwen4ExpTextGatedResidual(config)
    reference.cuda()
    candidate = copy.deepcopy(reference)
    for module in candidate.modules():
        patch_qwen4_exp_text_module_for_hyper_connection(module, Qwen4ExpTextGatedResidual, Qwen4ExpTextDecoderLayer)
    x = torch.randn(2, 3, config.hc_count * config.hidden_size, device="cuda", requires_grad=True)
    candidate_x = x.detach().clone().requires_grad_(True)
    assert _uses_liger_hyper_connection(candidate.attn_hyper_connection, x)
    targets = [model.attn_hyper_connection for model in (reference, candidate)]
    calls = [0, 0]

    def hook(module, *args):
        if module not in targets:
            return None
        calls[targets.index(module)] += 1
        if hook_type == "forward":
            mixed, residual, injection_weights = args[1]
            return mixed, residual, torch.zeros_like(injection_weights)
        return tuple(value * 0.5 if value is not None else None for value in args[0])

    def reject_raw_logits(*args, **kwargs):
        pytest.fail("Decoder used raw write logits despite HyperConnection hooks")

    monkeypatch.setattr(LigerQwen4ExpGRWriteFunction, "apply", reject_raw_logits)
    handles = (
        [getattr(torch.nn.modules.module, f"register_module_{hook_type}_hook")(hook)]
        if global_hook
        else [getattr(module, f"register_{hook_type}_hook")(hook) for module in targets]
    )
    try:
        position_embeddings = (torch.empty(0, device="cuda"),) * 2
        expected = reference(x, position_embeddings)
        actual = candidate(candidate_x, position_embeddings)
        grad = torch.randn_like(expected)
        expected.backward(grad)
        actual.backward(grad)
        assert calls == [1, 1]
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(candidate_x.grad, x.grad, atol=2e-5, rtol=2e-5)
        for parameter, native_parameter in zip(candidate.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, native_parameter.grad, atol=2e-5, rtol=2e-5)
    finally:
        for handle in handles:
            handle.remove()
    assert _uses_liger_hyper_connection(candidate.attn_hyper_connection, x)


@requires_nvidia
@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
@pytest.mark.parametrize("hook_type", ["forward_pre", "forward", "full_backward_pre", "full_backward"])
def test_qwen4_exp_write4_declines_global_module_hooks(monkeypatch, hook_type):
    """Global hooks must disable the Write4 direct-weight path and run through the module call."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    from liger_kernel.transformers.qwen4_exp import _can_use_write4_linear

    torch.manual_seed(73)
    config = _tiny_config()
    native = Qwen4ExpTextGatedResidual(config, use_combine=True).to(device, torch.bfloat16)
    candidate = copy.deepcopy(native)
    _patch_rms_norm_module(candidate.hc_norm, offset=1.0, casting_mode="gemma", in_place=False)
    candidate.forward = MethodType(liger_qwen4_exp_gated_residual_forward, candidate)
    targets = (native.block_inject_weight, candidate.block_inject_weight)
    calls = [0, 0]

    def hook(module, *args):
        if module not in targets:
            return None
        calls[targets.index(module)] += 1
        if hook_type == "forward":
            return args[1] * 2
        return tuple(value * 0.5 if value is not None else None for value in args[0])

    def reject_write4(*args, **kwargs):
        pytest.fail("Write4 direct-weight path ran despite a global module hook")

    monkeypatch.setattr(LigerGroupRMSNormWrite4Function, "apply", reject_write4)
    assert _can_use_write4_linear(candidate.block_inject_weight)
    handle = getattr(torch.nn.modules.module, f"register_module_{hook_type}_hook")(hook)
    try:
        assert not _can_use_write4_linear(candidate.block_inject_weight)
        assert not _can_use_write4_linear(native.block_inject_weight)
        x = torch.randn(2, 3, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        candidate_x = x.detach().clone().requires_grad_(True)
        expected = native(x)
        actual = candidate(candidate_x, return_write_logits=True)
        actual = (*actual[:2], 2 * torch.sigmoid(actual[2]))
        grads = tuple(torch.randn_like(value) for value in expected)
        torch.autograd.backward(expected, grads)
        torch.autograd.backward(actual, grads)
        assert calls == [1, 1]
        tolerance = 2e-2
        for output, reference in zip(actual, expected):
            torch.testing.assert_close(output, reference, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(candidate_x.grad, x.grad, atol=tolerance, rtol=tolerance)
        for parameter, reference in zip(candidate.parameters(), native.parameters()):
            torch.testing.assert_close(parameter.grad, reference.grad, atol=tolerance, rtol=tolerance)
    finally:
        handle.remove()

    assert _can_use_write4_linear(candidate.block_inject_weight)


@pytest.mark.parametrize("hook_type", ["forward_pre", "forward", "full_backward_pre", "full_backward"])
def test_qwen4_exp_fused_lce_declines_global_module_hooks(hook_type):
    """Global hooks must disable the fused-LCE direct-weight path and run through lm_head."""
    targets = []
    calls = []

    def hook(module, *args):
        if module not in targets:
            return None
        calls[targets.index(module)] += 1
        if hook_type == "forward":
            return args[1] * 2
        return tuple(value * 0.5 if value is not None else None for value in args[0])

    config = _tiny_config()
    reference = _stub_causal_lm(config)
    candidate = _stub_causal_lm(copy.deepcopy(config))
    reference.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    candidate.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    candidate.lm_head.load_state_dict(reference.lm_head.state_dict())
    candidate.forward = MethodType(lce_forward, candidate)
    targets.extend((reference.lm_head, candidate.lm_head))
    calls.extend((0, 0))
    hidden_states = torch.randn(2, 3, config.hidden_size)
    assert _can_use_fused_lce_lm_head(candidate.lm_head, hidden_states)

    handle = getattr(torch.nn.modules.module, f"register_module_{hook_type}_hook")(hook)
    try:
        assert not _can_use_fused_lce_lm_head(candidate.lm_head, hidden_states)
        assert not _can_use_fused_lce_lm_head(reference.lm_head, hidden_states)
        labels = torch.tensor([[2, 3, 4, 5]])
        reference_input = torch.randn(1, 4, config.hidden_size, requires_grad=True)
        candidate_input = reference_input.detach().clone().requires_grad_(True)
        expected = reference(inputs_embeds=reference_input, labels=labels)
        actual = candidate(inputs_embeds=candidate_input, labels=labels, skip_logits=True)
        assert actual.logits is not None
        _assert_output_values_equal(actual.logits, expected.logits)
        _assert_output_values_equal(actual.loss, expected.loss)
        expected.loss.backward()
        actual.loss.backward()
        torch.testing.assert_close(candidate_input.grad, reference_input.grad, atol=0.0, rtol=0.0)
        torch.testing.assert_close(candidate.lm_head.weight.grad, reference.lm_head.weight.grad, atol=0.0, rtol=0.0)
        assert calls == [1, 1]
    finally:
        handle.remove()

    assert _can_use_fused_lce_lm_head(candidate.lm_head, hidden_states)
