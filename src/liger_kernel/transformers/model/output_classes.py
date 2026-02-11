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
try:
    from transformers.models.gemma3.modeling_gemma3 import Gemma3CausalLMOutputWithPast as _Gemma3CausalLMOutputWithPast
except Exception:
    _Gemma3CausalLMOutputWithPast = None

try:
    from transformers.models.glm4v_moe.modeling_glm4v_moe import (
        Glm4vMoeCausalLMOutputWithPast as _Glm4vMoeCausalLMOutputWithPast,
    )
except Exception:
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
    from transformers.models.paligemma.modeling_paligemma import (
        PaliGemmaCausalLMOutputWithPast as _PaliGemmaCausalLMOutputWithPast,
    )
except Exception:
    _PaliGemmaCausalLMOutputWithPast = None

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLCausalLMOutputWithPast as _Qwen2_5_VLCausalLMOutputWithPast,
    )
except Exception:
    _Qwen2_5_VLCausalLMOutputWithPast = None

try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VLCausalLMOutputWithPast as _Qwen2VLCausalLMOutputWithPast,
    )
except Exception:
    _Qwen2VLCausalLMOutputWithPast = None

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLCausalLMOutputWithPast as _Qwen3VLCausalLMOutputWithPast,
    )
except Exception:
    _Qwen3VLCausalLMOutputWithPast = None

try:
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeCausalLMOutputWithPast as _Qwen3VLMoeCausalLMOutputWithPast,
    )
except Exception:
    _Qwen3VLMoeCausalLMOutputWithPast = None


@dataclass
class LigerCausalLMOutputWithPast(CausalLMOutputWithPast):
    token_accuracy: Optional[torch.FloatTensor] = None


@dataclass
class LigerMoeCausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    token_accuracy: Optional[torch.FloatTensor] = None


if _Gemma3CausalLMOutputWithPast is not None:

    @dataclass
    class LigerGemma3CausalLMOutputWithPast(_Gemma3CausalLMOutputWithPast):
        token_accuracy: Optional[torch.FloatTensor] = None


if _Glm4vMoeCausalLMOutputWithPast is not None:

    @dataclass
    class LigerGlm4vMoeCausalLMOutputWithPast(_Glm4vMoeCausalLMOutputWithPast):
        token_accuracy: Optional[torch.FloatTensor] = None


if _LlavaCausalLMOutputWithPast is not None:

    @dataclass
    class LigerLlavaCausalLMOutputWithPast(_LlavaCausalLMOutputWithPast):
        token_accuracy: Optional[torch.FloatTensor] = None


if _InternVLCausalLMOutputWithPast is not None:

    @dataclass
    class LigerInternVLCausalLMOutputWithPast(_InternVLCausalLMOutputWithPast):
        token_accuracy: Optional[torch.FloatTensor] = None


if _PaliGemmaCausalLMOutputWithPast is not None:

    @dataclass
    class LigerPaliGemmaCausalLMOutputWithPast(_PaliGemmaCausalLMOutputWithPast):
        token_accuracy: Optional[torch.FloatTensor] = None


if _Qwen2_5_VLCausalLMOutputWithPast is not None:

    @dataclass
    class LigerQwen2_5_VLCausalLMOutputWithPast(_Qwen2_5_VLCausalLMOutputWithPast):
        token_accuracy: Optional[torch.FloatTensor] = None


if _Qwen2VLCausalLMOutputWithPast is not None:

    @dataclass
    class LigerQwen2VLCausalLMOutputWithPast(_Qwen2VLCausalLMOutputWithPast):
        token_accuracy: Optional[torch.FloatTensor] = None


if _Qwen3VLCausalLMOutputWithPast is not None:

    @dataclass
    class LigerQwen3VLCausalLMOutputWithPast(_Qwen3VLCausalLMOutputWithPast):
        token_accuracy: Optional[torch.FloatTensor] = None


if _Qwen3VLMoeCausalLMOutputWithPast is not None:

    @dataclass
    class LigerQwen3VLMoeCausalLMOutputWithPast(_Qwen3VLMoeCausalLMOutputWithPast):
        token_accuracy: Optional[torch.FloatTensor] = None
