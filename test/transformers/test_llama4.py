"""Contract tests for the Llama4 Liger forward.

`test_monkey_patch.py` checks that `lce_forward` gets installed; it does not check what
the installed forward does with `skip_logits`. These two cases stay on the non-fused
branch, so they need no GPU.
"""

import pytest
import torch
import transformers

from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_llama4


def is_llama4_available():
    try:
        import transformers.models.llama4  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not is_llama4_available(), reason="llama4 module not available")


def _build_patched_model():
    from transformers.models.llama4.modeling_llama4 import Llama4ForCausalLM

    config = transformers.models.llama4.configuration_llama4.Llama4TextConfig(
        dtype=torch.float32,
        rms_norm_eps=1e-5,
        hidden_size=32,
        intermediate_size=64,
        hidden_act="silu",
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        moe_layers=[],
    )
    model = Llama4ForCausalLM(config)
    # Only the forward is swapped: the norm/MLP/rope kernels are Triton and would need a GPU.
    apply_liger_kernel_to_llama4(model=model, rope=False, rms_norm=False, swiglu=False)
    return model


def test_llama4_skip_logits_false_materializes_logits_in_training():
    model = _build_patched_model()
    model.train()

    input_ids = torch.randint(0, 64, (2, 8))

    output = model(input_ids=input_ids, labels=input_ids, skip_logits=False)

    assert output.logits is not None
    assert output.loss is not None


def test_llama4_skip_logits_true_without_labels_raises():
    model = _build_patched_model()
    model.eval()

    input_ids = torch.randint(0, 64, (2, 8))

    with pytest.raises(ValueError, match="skip_logits is True"):
        model(input_ids=input_ids, skip_logits=True)
