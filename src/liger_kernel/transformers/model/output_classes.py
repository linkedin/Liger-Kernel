"""
Custom output classes for Liger-Kernel that extend transformers' ModelOutput classes
with optional token accuracy field.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# The following model-specific outputs are optional and depend on the installed
# transformers version. Guard their imports so our module remains importable
# even when those models are not available in the environment.
# Newer transformers versions use standard causal-LM outputs for several VL
# models. Keep the legacy bases when available and retain Liger's rope_deltas field.
try:
    from transformers.models.gemma3.modeling_gemma3 import Gemma3CausalLMOutputWithPast as _Gemma3CausalLMOutputWithPast
except Exception:
    _Gemma3CausalLMOutputWithPast = None

try:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4CausalLMOutputWithPast as _Gemma4CausalLMOutputWithPast
except Exception:
    _Gemma4CausalLMOutputWithPast = None

try:
    from transformers.models.glm4v_moe import modeling_glm4v_moe

    _Glm4vMoeCausalLMOutputWithPast = getattr(
        modeling_glm4v_moe, "Glm4vMoeCausalLMOutputWithPast", MoeCausalLMOutputWithPast
    )
except ImportError:
    _Glm4vMoeCausalLMOutputWithPast = None

try:
    from transformers.models.internvl.modeling_internvl import (
        InternVLCausalLMOutputWithPast as _InternVLCausalLMOutputWithPast,
    )
except Exception:
    _InternVLCausalLMOutputWithPast = None

try:
    from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast as _LlavaCausalLMOutputWithPast
except Exception:
    _LlavaCausalLMOutputWithPast = None

try:
    from transformers.models.muse_glimmer.modeling_muse_glimmer import (
        MuseGlimmerCausalLMOutputWithPast as _MuseGlimmerCausalLMOutputWithPast,
    )
except Exception:
    _MuseGlimmerCausalLMOutputWithPast = None

try:
    from transformers.models.paligemma.modeling_paligemma import (
        PaliGemmaCausalLMOutputWithPast as _PaliGemmaCausalLMOutputWithPast,
    )
except Exception:
    _PaliGemmaCausalLMOutputWithPast = None

try:
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

    _Qwen2_5_VLCausalLMOutputWithPast = getattr(
        modeling_qwen2_5_vl, "Qwen2_5_VLCausalLMOutputWithPast", CausalLMOutputWithPast
    )
except ImportError:
    _Qwen2_5_VLCausalLMOutputWithPast = None

try:
    from transformers.models.qwen2_vl import modeling_qwen2_vl

    _Qwen2VLCausalLMOutputWithPast = getattr(modeling_qwen2_vl, "Qwen2VLCausalLMOutputWithPast", CausalLMOutputWithPast)
except ImportError:
    _Qwen2VLCausalLMOutputWithPast = None

try:
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    _Qwen3VLCausalLMOutputWithPast = getattr(modeling_qwen3_vl, "Qwen3VLCausalLMOutputWithPast", CausalLMOutputWithPast)
except ImportError:
    _Qwen3VLCausalLMOutputWithPast = None

try:
    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe

    _Qwen3VLMoeCausalLMOutputWithPast = getattr(
        modeling_qwen3_vl_moe, "Qwen3VLMoeCausalLMOutputWithPast", MoeCausalLMOutputWithPast
    )
except ImportError:
    _Qwen3VLMoeCausalLMOutputWithPast = None

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5CausalLMOutputWithPast as _Qwen3_5CausalLMOutputWithPast,
    )
except Exception:
    _Qwen3_5CausalLMOutputWithPast = None

try:
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe

    _Qwen3_5MoeCausalLMOutputWithPast = getattr(
        modeling_qwen3_5_moe, "Qwen3_5MoeCausalLMOutputWithPast", MoeCausalLMOutputWithPast
    )
except ImportError:
    _Qwen3_5MoeCausalLMOutputWithPast = None


@dataclass
class LigerCausalLMOutputWithPast(CausalLMOutputWithPast):
    token_accuracy: Optional[torch.FloatTensor] = None
    predicted_tokens: Optional[torch.LongTensor] = None


@dataclass
class LigerMoeCausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    token_accuracy: Optional[torch.FloatTensor] = None
    predicted_tokens: Optional[torch.LongTensor] = None


if _Gemma3CausalLMOutputWithPast is not None:

    @dataclass
    class LigerGemma3CausalLMOutputWithPast(_Gemma3CausalLMOutputWithPast):
        token_accuracy: Optional[torch.FloatTensor] = None
        predicted_tokens: Optional[torch.LongTensor] = None


if _Gemma4CausalLMOutputWithPast is not None:

    @dataclass
    class LigerGemma4CausalLMOutputWithPast(_Gemma4CausalLMOutputWithPast):
        token_accuracy: Optional[torch.FloatTensor] = None
        predicted_tokens: Optional[torch.LongTensor] = None


if _Glm4vMoeCausalLMOutputWithPast is not None:

    @dataclass
    class LigerGlm4vMoeCausalLMOutputWithPast(_Glm4vMoeCausalLMOutputWithPast):
        rope_deltas: Optional[torch.LongTensor] = None
        token_accuracy: Optional[torch.FloatTensor] = None
        predicted_tokens: Optional[torch.LongTensor] = None


if _LlavaCausalLMOutputWithPast is not None:

    @dataclass
    class LigerLlavaCausalLMOutputWithPast(_LlavaCausalLMOutputWithPast):
        token_accuracy: Optional[torch.FloatTensor] = None
        predicted_tokens: Optional[torch.LongTensor] = None


if _InternVLCausalLMOutputWithPast is not None:

    @dataclass
    class LigerInternVLCausalLMOutputWithPast(_InternVLCausalLMOutputWithPast):
        token_accuracy: Optional[torch.FloatTensor] = None
        predicted_tokens: Optional[torch.LongTensor] = None


if _MuseGlimmerCausalLMOutputWithPast is not None:

    @dataclass
    class LigerMuseGlimmerCausalLMOutputWithPast(_MuseGlimmerCausalLMOutputWithPast):
        token_accuracy: Optional[torch.FloatTensor] = None
        predicted_tokens: Optional[torch.LongTensor] = None


if _PaliGemmaCausalLMOutputWithPast is not None:

    @dataclass
    class LigerPaliGemmaCausalLMOutputWithPast(_PaliGemmaCausalLMOutputWithPast):
        token_accuracy: Optional[torch.FloatTensor] = None
        predicted_tokens: Optional[torch.LongTensor] = None


if _Qwen2_5_VLCausalLMOutputWithPast is not None:

    @dataclass
    class LigerQwen2_5_VLCausalLMOutputWithPast(_Qwen2_5_VLCausalLMOutputWithPast):
        rope_deltas: Optional[torch.LongTensor] = None
        token_accuracy: Optional[torch.FloatTensor] = None
        predicted_tokens: Optional[torch.LongTensor] = None


if _Qwen2VLCausalLMOutputWithPast is not None:

    @dataclass
    class LigerQwen2VLCausalLMOutputWithPast(_Qwen2VLCausalLMOutputWithPast):
        rope_deltas: Optional[torch.LongTensor] = None
        token_accuracy: Optional[torch.FloatTensor] = None
        predicted_tokens: Optional[torch.LongTensor] = None


if _Qwen3VLCausalLMOutputWithPast is not None:

    @dataclass
    class LigerQwen3VLCausalLMOutputWithPast(_Qwen3VLCausalLMOutputWithPast):
        rope_deltas: Optional[torch.LongTensor] = None
        token_accuracy: Optional[torch.FloatTensor] = None
        predicted_tokens: Optional[torch.LongTensor] = None


if _Qwen3VLMoeCausalLMOutputWithPast is not None:

    @dataclass
    class LigerQwen3VLMoeCausalLMOutputWithPast(_Qwen3VLMoeCausalLMOutputWithPast):
        rope_deltas: Optional[torch.LongTensor] = None
        token_accuracy: Optional[torch.FloatTensor] = None
        predicted_tokens: Optional[torch.LongTensor] = None


if _Qwen3_5CausalLMOutputWithPast is not None:

    @dataclass
    class LigerQwen3_5CausalLMOutputWithPast(_Qwen3_5CausalLMOutputWithPast):
        token_accuracy: Optional[torch.FloatTensor] = None
        predicted_tokens: Optional[torch.LongTensor] = None


if _Qwen3_5MoeCausalLMOutputWithPast is not None:

    @dataclass
    class LigerQwen3_5MoeCausalLMOutputWithPast(_Qwen3_5MoeCausalLMOutputWithPast):
        rope_deltas: Optional[torch.LongTensor] = None
        token_accuracy: Optional[torch.FloatTensor] = None
        predicted_tokens: Optional[torch.LongTensor] = None
