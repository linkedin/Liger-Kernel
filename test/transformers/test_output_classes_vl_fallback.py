"""Regression tests for the vision-language output-class fallbacks.

transformers 5.23 removed the per-model ``*CausalLMOutputWithPast`` classes that
the Qwen-VL / GLM4V-MoE models used to return.  ``output_classes.py`` guarded
each ``Liger*`` subclass behind ``if _X is not None``, so on 5.23 the names were
never defined, and the forward modules that import them unconditionally failed
with ``ImportError`` at import time (#1638).

These tests are CPU-only: they build the output dataclasses directly and never
launch a kernel.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from liger_kernel.transformers.model import output_classes as oc

# (upstream base attribute, Liger subclass name, is_moe)
VL_OUTPUTS = [
    ("_Glm4vMoeCausalLMOutputWithPast", "LigerGlm4vMoeCausalLMOutputWithPast", True),
    ("_Qwen2VLCausalLMOutputWithPast", "LigerQwen2VLCausalLMOutputWithPast", False),
    ("_Qwen2_5_VLCausalLMOutputWithPast", "LigerQwen2_5_VLCausalLMOutputWithPast", False),
    ("_Qwen3VLCausalLMOutputWithPast", "LigerQwen3VLCausalLMOutputWithPast", False),
    ("_Qwen3VLMoeCausalLMOutputWithPast", "LigerQwen3VLMoeCausalLMOutputWithPast", True),
    ("_Qwen3_5CausalLMOutputWithPast", "LigerQwen3_5CausalLMOutputWithPast", False),
    ("_Qwen3_5MoeCausalLMOutputWithPast", "LigerQwen3_5MoeCausalLMOutputWithPast", True),
]

IDS = [liger for _, liger, _ in VL_OUTPUTS]

# The forward modules that import those names unconditionally.
VL_FORWARD_MODULES = [
    "liger_kernel.transformers.model.qwen2_vl",
    "liger_kernel.transformers.model.qwen2_5_vl",
    "liger_kernel.transformers.model.qwen3_vl",
    "liger_kernel.transformers.model.qwen3_vl_moe",
    "liger_kernel.transformers.model.qwen3_5",
    "liger_kernel.transformers.model.qwen3_5_moe",
    "liger_kernel.transformers.model.glm4v_moe",
]


@pytest.mark.parametrize(("upstream", "liger", "is_moe"), VL_OUTPUTS, ids=IDS)
def test_liger_output_class_is_always_defined(upstream, liger, is_moe):
    """The Liger name must exist even when transformers drops the upstream class."""
    assert hasattr(oc, liger), (
        f"{liger} is imported unconditionally by its forward module, so it must be defined even when {upstream} is None"
    )


@pytest.mark.parametrize(("upstream", "liger", "is_moe"), VL_OUTPUTS, ids=IDS)
def test_liger_output_class_picks_the_right_base(upstream, liger, is_moe):
    """Use the upstream class when it exists, otherwise the matching fallback."""
    base = getattr(oc, upstream)
    cls = getattr(oc, liger)
    if base is not None:
        assert issubclass(cls, base)
    elif is_moe:
        assert issubclass(cls, oc._LigerVLMoeOutputWithPast)
    else:
        assert issubclass(cls, oc._LigerVLOutputWithPast)


@pytest.mark.parametrize(("upstream", "liger", "is_moe"), VL_OUTPUTS, ids=IDS)
def test_liger_output_class_accepts_the_kwargs_the_forward_passes(upstream, liger, is_moe):
    """The VL forwards always pass ``rope_deltas``; MoE ones also pass ``aux_loss``."""
    cls = getattr(oc, liger)
    kwargs = {
        "loss": torch.tensor(0.5),
        "logits": torch.zeros(1, 4),
        "rope_deltas": torch.zeros(2, dtype=torch.long),
        "token_accuracy": 1.0,
        "predicted_tokens": 2,
    }
    if is_moe:
        kwargs["aux_loss"] = torch.tensor(0.1)
        kwargs["router_logits"] = (torch.zeros(1, 4),)

    out = cls(**kwargs)

    assert out.rope_deltas is not None
    assert out.token_accuracy == 1.0
    assert out.predicted_tokens == 2
    if is_moe:
        assert out.aux_loss is not None


@pytest.mark.parametrize("module", VL_FORWARD_MODULES, ids=[m.rsplit(".", 1)[-1] for m in VL_FORWARD_MODULES])
def test_forward_module_imports(module):
    """The regression from #1638: this import raised ImportError on transformers 5.23."""
    importlib.import_module(module)


def test_dense_fallback_base_defaults():
    out = oc._LigerVLOutputWithPast()
    assert out.rope_deltas is None
    assert out.logits is None


def test_moe_fallback_base_defaults():
    out = oc._LigerVLMoeOutputWithPast()
    assert out.rope_deltas is None
    assert out.aux_loss is None
    assert out.router_logits is None
