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
    from transformers.models.gemma4.modeling_gemma4 import Gemma4CausalLMOutputWithPast as _Gemma4CausalLMOutputWithPast
except Exception:
    _Gemma4CausalLMOutputWithPast = None

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

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5CausalLMOutputWithPast as _Qwen3_5CausalLMOutputWithPast,
    )
except Exception:
    _Qwen3_5CausalLMOutputWithPast = None

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeCausalLMOutputWithPast as _Qwen3_5MoeCausalLMOutputWithPast,
    )
except Exception:
    _Qwen3_5MoeCausalLMOutputWithPast = None


@dataclass
class LigerCausalLMOutputWithPast(CausalLMOutputWithPast):
    token_accuracy: Optional[torch.FloatTensor] = None
    predicted_tokens: Optional[torch.LongTensor] = None


@dataclass
class LigerMoeCausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    token_accuracy: Optional[torch.FloatTensor] = None
    predicted_tokens: Optional[torch.LongTensor] = None


@dataclass
class _LigerVLOutputWithPast(CausalLMOutputWithPast):
    """Fallback base for vision-language outputs.

    transformers 5.23 dropped the per-model ``*CausalLMOutputWithPast`` classes,
    so the guarded imports above resolve to ``None`` and the ``Liger*`` subclass
    was never defined. The ``Liger*`` names are imported unconditionally by the
    VL forward modules, which then failed with ``ImportError`` at import time.
    The patched forwards still carry ``rope_deltas``, so the fallback base
    declares it.
    """

    rope_deltas: Optional[torch.LongTensor] = None


@dataclass
class _LigerVLMoeOutputWithPast(MoeCausalLMOutputWithPast):
    """Fallback base for vision-language *MoE* outputs.

    Same reason as ``_LigerVLOutputWithPast``. ``aux_loss`` and ``router_logits``
    are inherited from ``MoeCausalLMOutputWithPast``; only ``rope_deltas``, which
    the patched VL forwards always pass, has to be declared here.
    """

    rope_deltas: Optional[torch.LongTensor] = None


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


_Glm4vMoeOutputBase = (
    _Glm4vMoeCausalLMOutputWithPast if _Glm4vMoeCausalLMOutputWithPast is not None else _LigerVLMoeOutputWithPast
)


@dataclass
class LigerGlm4vMoeCausalLMOutputWithPast(_Glm4vMoeOutputBase):
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


_Qwen2_5_VLOutputBase = (
    _Qwen2_5_VLCausalLMOutputWithPast if _Qwen2_5_VLCausalLMOutputWithPast is not None else _LigerVLOutputWithPast
)


@dataclass
class LigerQwen2_5_VLCausalLMOutputWithPast(_Qwen2_5_VLOutputBase):
    token_accuracy: Optional[torch.FloatTensor] = None
    predicted_tokens: Optional[torch.LongTensor] = None


_Qwen2VLOutputBase = (
    _Qwen2VLCausalLMOutputWithPast if _Qwen2VLCausalLMOutputWithPast is not None else _LigerVLOutputWithPast
)


@dataclass
class LigerQwen2VLCausalLMOutputWithPast(_Qwen2VLOutputBase):
    token_accuracy: Optional[torch.FloatTensor] = None
    predicted_tokens: Optional[torch.LongTensor] = None


_Qwen3VLOutputBase = (
    _Qwen3VLCausalLMOutputWithPast if _Qwen3VLCausalLMOutputWithPast is not None else _LigerVLOutputWithPast
)


@dataclass
class LigerQwen3VLCausalLMOutputWithPast(_Qwen3VLOutputBase):
    token_accuracy: Optional[torch.FloatTensor] = None
    predicted_tokens: Optional[torch.LongTensor] = None


_Qwen3VLMoeOutputBase = (
    _Qwen3VLMoeCausalLMOutputWithPast if _Qwen3VLMoeCausalLMOutputWithPast is not None else _LigerVLMoeOutputWithPast
)


@dataclass
class LigerQwen3VLMoeCausalLMOutputWithPast(_Qwen3VLMoeOutputBase):
    token_accuracy: Optional[torch.FloatTensor] = None
    predicted_tokens: Optional[torch.LongTensor] = None


_Qwen3_5OutputBase = (
    _Qwen3_5CausalLMOutputWithPast if _Qwen3_5CausalLMOutputWithPast is not None else _LigerVLOutputWithPast
)


@dataclass
class LigerQwen3_5CausalLMOutputWithPast(_Qwen3_5OutputBase):
    token_accuracy: Optional[torch.FloatTensor] = None
    predicted_tokens: Optional[torch.LongTensor] = None


_Qwen3_5MoeOutputBase = (
    _Qwen3_5MoeCausalLMOutputWithPast if _Qwen3_5MoeCausalLMOutputWithPast is not None else _LigerVLMoeOutputWithPast
)


@dataclass
class LigerQwen3_5MoeCausalLMOutputWithPast(_Qwen3_5MoeOutputBase):
    token_accuracy: Optional[torch.FloatTensor] = None
    predicted_tokens: Optional[torch.LongTensor] = None
