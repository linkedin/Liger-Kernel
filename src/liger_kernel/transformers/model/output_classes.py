"""
Custom output classes for Liger-Kernel that extend transformers' ModelOutput classes
with optional token accuracy field.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.models.gemma3.modeling_gemma3 import Gemma3CausalLMOutputWithPast
from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeCausalLMOutputWithPast
from transformers.models.internvl.modeling_internvl import InternVLCausalLMOutputWithPast
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast
from transformers.models.paligemma.modeling_paligemma import PaliGemmaCausalLMOutputWithPast
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeCausalLMOutputWithPast


@dataclass
class LigerCausalLMOutputWithPast(CausalLMOutputWithPast):
    token_accuracy: Optional[torch.FloatTensor] = None


@dataclass
class LigerMoeCausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    token_accuracy: Optional[torch.FloatTensor] = None


@dataclass
class LigerGemma3CausalLMOutputWithPast(Gemma3CausalLMOutputWithPast):
    token_accuracy: Optional[torch.FloatTensor] = None


@dataclass
class LigerGlm4vMoeCausalLMOutputWithPast(Glm4vMoeCausalLMOutputWithPast):
    token_accuracy: Optional[torch.FloatTensor] = None


@dataclass
class LigerLlavaCausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    token_accuracy: Optional[torch.FloatTensor] = None


@dataclass
class LigerInternVLCausalLMOutputWithPast(InternVLCausalLMOutputWithPast):
    token_accuracy: Optional[torch.FloatTensor] = None


@dataclass
class LigerPaliGemmaCausalLMOutputWithPast(PaliGemmaCausalLMOutputWithPast):
    token_accuracy: Optional[torch.FloatTensor] = None


@dataclass
class LigerQwen2_5_VLCausalLMOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    token_accuracy: Optional[torch.FloatTensor] = None


@dataclass
class LigerQwen2VLCausalLMOutputWithPast(Qwen2VLCausalLMOutputWithPast):
    token_accuracy: Optional[torch.FloatTensor] = None


@dataclass
class LigerQwen3VLCausalLMOutputWithPast(Qwen3VLCausalLMOutputWithPast):
    token_accuracy: Optional[torch.FloatTensor] = None


@dataclass
class LigerQwen3VLMoeCausalLMOutputWithPast(Qwen3VLMoeCausalLMOutputWithPast):
    token_accuracy: Optional[torch.FloatTensor] = None
