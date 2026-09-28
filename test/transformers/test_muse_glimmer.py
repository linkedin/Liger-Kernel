"""Numerical and contract tests for the Muse Glimmer Liger integration.

These cover what `test_monkey_patch.py::test_apply_liger_kernel_to_instance_for_muse_glimmer`
and the convergence suite do not:

  * The convergence suite patches at *module* level (it calls the apply fn without
    `model=`). TRL and `HFTrainer(use_liger_kernel=True)` go through
    `_apply_liger_kernel_to_instance(model=...)` instead, which runs a different code
    path -- three bespoke RMSNorm helpers plus the vision LayerNorm patch. Nothing
    validated that path numerically.
  * `return_token_accuracy=True -> outputs.token_accuracy` is the subject of the feature
    request these patches implement, and had no model-level coverage at all.

Requires a GPU: the Liger modules dispatch to Triton kernels.
"""

import copy

import pytest
import torch

from test.utils import assert_verbose_allclose
from test.utils import supports_bfloat16

from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance


def is_muse_glimmer_available():
    try:
        import transformers.models.muse_glimmer  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = [
    pytest.mark.skipif(not is_muse_glimmer_available(), reason="muse_glimmer module not available"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Liger kernels require a GPU"),
]


def _mini_config():
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerConfig
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerTextConfig
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerVisionConfig

    # Text parameters must go in `text_config`; top-level fields are ignored.
    config = MuseGlimmerConfig(
        text_config=MuseGlimmerTextConfig(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            hidden_activation="silu",
            max_position_embeddings=128,
            sliding_window=16,
            rms_norm_eps=1e-5,
            post_norm_eps=1e-8,
            attention_dropout=0.0,
        ),
        vision_config=MuseGlimmerVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            patch_size=14,
            pos_emb_height=4,
            pos_emb_width=4,
            max_position_embeddings=16,
            merge_size=2,
        ),
    )
    config._attn_implementation = "sdpa"
    return config


def _build_pair(dtype):
    """An unpatched model and an instance-patched deepcopy of it, sharing weights."""
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerForConditionalGeneration

    torch.manual_seed(42)
    model_hf = MuseGlimmerForConditionalGeneration(_mini_config()).to("cuda").to(dtype)
    model_liger = copy.deepcopy(model_hf)
    _apply_liger_kernel_to_instance(model=model_liger)
    return model_hf, model_liger


def _assert_instance_patch_applied(model_liger):
    """Guard against a silently no-op patch, which would make every parity assert vacuous."""
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLPForMuseGlimmer

    text_model = model_liger.model.language_model
    assert text_model.norm._get_name() == LigerRMSNorm.__name__
    for layer in text_model.layers:
        assert layer.mlp._get_name() == LigerSwiGLUMLPForMuseGlimmer.__name__
        assert layer.input_layernorm._get_name() == LigerRMSNorm.__name__
        assert layer.input_layernorm.offset == 1.0
        assert not layer.input_layernorm.in_place
        assert getattr(layer.self_attn.qk_norm, "weight", None) is None

    assert getattr(model_liger.model.perception_emb_norm, "weight", None) is None
    embed_norm = text_model.embed_tokens.embed_norm
    assert getattr(embed_norm, "weight", None) is None


def _text_batch(batch_size=2, seq_len=16, vocab_size=512):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
    labels = input_ids.clone()
    labels[:, :2] = -100  # exercise the ignore_index path
    return input_ids, labels


def _expected_token_accuracy(logits, labels):
    """Reference accuracy over the same shifted, non-ignored positions Liger uses."""
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    mask = shift_labels != -100
    correct = shift_logits.argmax(dim=-1)[mask] == shift_labels[mask]
    return correct.float().mean()


# ---------------------------------------------------------------------------
# Gap A -- instance-patch numerical parity (the path TRL actually takes)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-4),
        pytest.param(
            torch.bfloat16,
            1e-2,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_muse_glimmer_instance_patch_numerical_parity(dtype, atol, rtol):
    model_hf, model_liger = _build_pair(dtype)
    _assert_instance_patch_applied(model_liger)

    model_hf.train()
    model_liger.train()

    input_ids, labels = _text_batch()

    out_hf = model_hf(input_ids=input_ids, labels=labels)
    out_liger = model_liger(input_ids=input_ids, labels=labels)

    # Training + labels means the patched forward should default to skip_logits.
    assert out_hf.logits is not None
    assert out_liger.logits is None

    assert_verbose_allclose(out_hf.loss, out_liger.loss, atol=atol, rtol=rtol, extra_info="[Loss]")

    out_hf.loss.backward()
    out_liger.loss.backward()

    # lm_head.weight is tied to embed_tokens.weight, so this also covers the fused
    # linear cross entropy weight gradient.
    assert_verbose_allclose(
        model_hf.get_input_embeddings().weight.grad,
        model_liger.get_input_embeddings().weight.grad,
        atol=atol,
        rtol=rtol,
        extra_info="[Embedding grad]",
    )

    hf_layer = model_hf.model.language_model.layers[0]
    liger_layer = model_liger.model.language_model.layers[0]
    assert_verbose_allclose(
        hf_layer.mlp.down_proj.weight.grad,
        liger_layer.mlp.down_proj.weight.grad,
        atol=atol,
        rtol=rtol,
        extra_info="[SwiGLU down_proj grad]",
    )
    assert_verbose_allclose(
        hf_layer.input_layernorm.weight.grad,
        liger_layer.input_layernorm.weight.grad,
        atol=atol,
        rtol=rtol,
        extra_info="[TextCentered RMSNorm grad]",
    )


def test_muse_glimmer_instance_patch_parity_without_skip_logits():
    """The non-fused branch reapplies output_multiplier and the tanh softcap by hand."""
    dtype, atol, rtol = torch.float32, 1e-4, 1e-4
    model_hf, model_liger = _build_pair(dtype)
    _assert_instance_patch_applied(model_liger)

    model_hf.eval()
    model_liger.eval()

    input_ids, labels = _text_batch()

    with torch.no_grad():
        out_hf = model_hf(input_ids=input_ids, labels=labels)
        out_liger = model_liger(input_ids=input_ids, labels=labels, skip_logits=False)

    assert out_liger.logits is not None
    assert_verbose_allclose(out_hf.logits, out_liger.logits, atol=atol, rtol=rtol, extra_info="[Logits]")
    assert_verbose_allclose(out_hf.loss, out_liger.loss, atol=atol, rtol=rtol, extra_info="[Loss]")


def test_muse_glimmer_vision_layer_norm_patch_parity():
    """Vision LayerNorm is only patched on the instance path, so nothing else covers it."""
    from liger_kernel.transformers.layer_norm import LigerLayerNorm

    dtype, atol, rtol = torch.float32, 1e-4, 1e-4
    model_hf, model_liger = _build_pair(dtype)

    vision_hf = model_hf.model.vision_tower
    vision_liger = model_liger.model.vision_tower
    assert vision_liger.ln_pre._get_name() == LigerLayerNorm.__name__
    assert vision_liger.layers[0].norm1._get_name() == LigerLayerNorm.__name__

    x = torch.randn(4, vision_hf.config.hidden_size, device="cuda", dtype=dtype)
    for name, mod_hf, mod_liger in [
        ("ln_pre", vision_hf.ln_pre, vision_liger.ln_pre),
        ("ln_post", vision_hf.ln_post, vision_liger.ln_post),
        ("norm1", vision_hf.layers[0].norm1, vision_liger.layers[0].norm1),
        ("norm2", vision_hf.layers[0].norm2, vision_liger.layers[0].norm2),
    ]:
        assert_verbose_allclose(mod_hf(x), mod_liger(x), atol=atol, rtol=rtol, extra_info=f"[vision {name}]")


# ---------------------------------------------------------------------------
# Gap B -- the token_accuracy contract TRL relies on
# ---------------------------------------------------------------------------
def test_muse_glimmer_returns_token_accuracy():
    model_hf, model_liger = _build_pair(torch.float32)
    model_hf.train()
    model_liger.train()

    input_ids, labels = _text_batch()

    out_liger = model_liger(input_ids=input_ids, labels=labels, return_token_accuracy=True)

    assert out_liger.token_accuracy is not None, (
        "return_token_accuracy=True must populate outputs.token_accuracy -- this is what "
        "TRL reads, and it warns and drops the metric when it is missing."
    )
    assert out_liger.logits is None, "token accuracy must not come at the cost of materializing logits"
    assert 0.0 <= float(out_liger.token_accuracy) <= 1.0

    # output_multiplier > 0 and tanh are monotonic, so argmax over HF's softcapped
    # logits matches the argmax Liger takes over the raw ones.
    with torch.no_grad():
        out_hf = model_hf(input_ids=input_ids, labels=labels)
    expected = _expected_token_accuracy(out_hf.logits, labels)
    assert_verbose_allclose(
        expected, out_liger.token_accuracy.float(), atol=1e-4, rtol=1e-4, extra_info="[token_accuracy]"
    )


def test_muse_glimmer_returns_predicted_tokens():
    model_hf, model_liger = _build_pair(torch.float32)
    model_hf.train()
    model_liger.train()

    input_ids, labels = _text_batch()
    batch_size, seq_len = input_ids.shape

    out_liger = model_liger(input_ids=input_ids, labels=labels, return_predicted_tokens=True)

    assert out_liger.predicted_tokens is not None
    assert out_liger.predicted_tokens.dtype == torch.int64
    assert out_liger.predicted_tokens.numel() == batch_size * seq_len

    with torch.no_grad():
        out_hf = model_hf(input_ids=input_ids, labels=labels)

    # Liger pads labels then shifts, so position i predicts labels[i+1] and the final
    # position is always ignored; ignored positions are left as -1.
    predicted = out_liger.predicted_tokens.view(batch_size, seq_len)
    shift_labels = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    mask = shift_labels != -100
    expected = out_hf.logits.argmax(dim=-1)
    assert torch.equal(predicted[mask], expected[mask])
    assert (predicted[~mask] == -1).all()


def test_muse_glimmer_token_accuracy_absent_when_not_requested():
    _, model_liger = _build_pair(torch.float32)
    model_liger.train()

    input_ids, labels = _text_batch()
    out = model_liger(input_ids=input_ids, labels=labels)

    # The field must exist on the output class (TRL uses getattr), but stay None so the
    # accuracy kernel is not paid for when nobody asked for it.
    assert hasattr(out, "token_accuracy")
    assert out.token_accuracy is None
    assert out.predicted_tokens is None
