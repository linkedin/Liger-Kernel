import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Ensure deterministic behavior with CuBLAS

import pytest
import torch

from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers.models.gemma import GemmaConfig
from transformers.models.gemma import GemmaForCausalLM
from transformers.models.gemma2 import Gemma2Config
from transformers.models.gemma2 import Gemma2ForCausalLM
from transformers.models.llama import LlamaConfig
from transformers.models.llama import LlamaForCausalLM
from transformers.models.mistral import MistralConfig
from transformers.models.mistral import MistralForCausalLM
from transformers.models.mixtral import MixtralConfig
from transformers.models.mixtral import MixtralForCausalLM
from transformers.models.phi3 import Phi3Config
from transformers.models.phi3 import Phi3ForCausalLM
from transformers.models.qwen2 import Qwen2Config
from transformers.models.qwen2 import Qwen2ForCausalLM

from liger_kernel.transformers import apply_liger_kernel_to_falcon_h1
from liger_kernel.transformers import apply_liger_kernel_to_gemma
from liger_kernel.transformers import apply_liger_kernel_to_gemma2
from liger_kernel.transformers import apply_liger_kernel_to_gemma3_text
from liger_kernel.transformers import apply_liger_kernel_to_glm4
from liger_kernel.transformers import apply_liger_kernel_to_glm4v
from liger_kernel.transformers import apply_liger_kernel_to_glm4v_moe
from liger_kernel.transformers import apply_liger_kernel_to_granite
from liger_kernel.transformers import apply_liger_kernel_to_hunyuan_v1_dense
from liger_kernel.transformers import apply_liger_kernel_to_hunyuan_v1_moe
from liger_kernel.transformers import apply_liger_kernel_to_internvl
from liger_kernel.transformers import apply_liger_kernel_to_llama
from liger_kernel.transformers import apply_liger_kernel_to_llama4
from liger_kernel.transformers import apply_liger_kernel_to_llava
from liger_kernel.transformers import apply_liger_kernel_to_mistral
from liger_kernel.transformers import apply_liger_kernel_to_mixtral
from liger_kernel.transformers import apply_liger_kernel_to_mllama
from liger_kernel.transformers import apply_liger_kernel_to_olmo2
from liger_kernel.transformers import apply_liger_kernel_to_olmo3
from liger_kernel.transformers import apply_liger_kernel_to_phi3
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
from liger_kernel.transformers import apply_liger_kernel_to_qwen3
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_next
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl_moe
from liger_kernel.transformers import apply_liger_kernel_to_smollm3
from test.utils import DEFAULT_DATASET_PATH
from test.utils import MiniModelConfig
from test.utils import assert_verbose_allclose
from test.utils import get_logprobs
from test.utils import get_topk
from test.utils import require_deterministic
from test.utils import revert_liger_kernel_to_falcon_h1
from test.utils import revert_liger_kernel_to_gemma
from test.utils import revert_liger_kernel_to_gemma2
from test.utils import revert_liger_kernel_to_gemma3_text
from test.utils import revert_liger_kernel_to_glm4
from test.utils import revert_liger_kernel_to_glm4v
from test.utils import revert_liger_kernel_to_glm4v_moe
from test.utils import revert_liger_kernel_to_granite
from test.utils import revert_liger_kernel_to_hunyuan_v1
from test.utils import revert_liger_kernel_to_hunyuan_v1_moe
from test.utils import revert_liger_kernel_to_internvl
from test.utils import revert_liger_kernel_to_llama
from test.utils import revert_liger_kernel_to_llama4
from test.utils import revert_liger_kernel_to_llava
from test.utils import revert_liger_kernel_to_mistral
from test.utils import revert_liger_kernel_to_mixtral
from test.utils import revert_liger_kernel_to_mllama
from test.utils import revert_liger_kernel_to_olmo2
from test.utils import revert_liger_kernel_to_olmo3
from test.utils import revert_liger_kernel_to_phi3
from test.utils import revert_liger_kernel_to_qwen2
from test.utils import revert_liger_kernel_to_qwen2_5_vl
from test.utils import revert_liger_kernel_to_qwen2_vl
from test.utils import revert_liger_kernel_to_qwen3
from test.utils import revert_liger_kernel_to_qwen3_moe
from test.utils import revert_liger_kernel_to_qwen3_next
from test.utils import revert_liger_kernel_to_qwen3_vl
from test.utils import revert_liger_kernel_to_qwen3_vl_moe
from test.utils import revert_liger_kernel_to_smollm3
from test.utils import set_seed
from test.utils import simple_collate_fn

try:
    from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
    from transformers.models.llama4.modeling_llama4 import Llama4ForCausalLM

    LLAMA4_AVAILABLE = True
except ImportError:
    LLAMA4_AVAILABLE = False

try:
    # Mllama is only available in transformers>=4.45.0
    from transformers.models.mllama.configuration_mllama import MllamaTextConfig
    from transformers.models.mllama.modeling_mllama import MllamaForCausalLM

    MLLAMA_AVAILABLE = True
except ImportError:
    MLLAMA_AVAILABLE = False

try:
    # Qwen2-VL is only available in transformers>4.52.4
    import transformers

    from packaging import version
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

    QWEN2_VL_AVAILABLE = version.parse(transformers.__version__) >= version.parse("4.52.4")
except ImportError:
    QWEN2_VL_AVAILABLE = False

try:
    # Qwen2.5-VL is only available in transformers>4.52.4
    import transformers

    from packaging import version
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

    QWEN2_5_VL_AVAILABLE = version.parse(transformers.__version__) >= version.parse("4.52.4")
except ImportError:
    QWEN2_5_VL_AVAILABLE = False


try:
    import transformers

    from packaging import version
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    QWEN3_VL_AVAILABLE = version.parse(transformers.__version__) >= version.parse("4.57.0")
except ImportError:
    QWEN3_VL_AVAILABLE = False


try:
    import transformers

    from packaging import version
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeConfig
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeTextConfig
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeVisionConfig
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

    QWEN3_VL_MOE_AVAILABLE = version.parse(transformers.__version__) >= version.parse("4.57.0")
except ImportError:
    QWEN3_VL_MOE_AVAILABLE = False

try:
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False

try:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    QWEN3_VL_AVAILABLE = True
except ImportError:
    QWEN3_VL_AVAILABLE = False

try:
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeConfig
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

    QWEN3_VL_MOE_AVAILABLE = True
except ImportError:
    QWEN3_VL_MOE_AVAILABLE = False

try:
    from transformers.models.granite import GraniteConfig
    from transformers.models.granite import GraniteForCausalLM

    GRANITE_AVAILABLE = True
except ImportError:
    GRANITE_AVAILABLE = False

try:
    from transformers import CLIPVisionConfig
    from transformers.models.llava.configuration_llava import LlavaConfig
    from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration

    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False

try:
    # OLMO2 is only available in transformers>=4.47.0
    from transformers.models.olmo2.configuration_olmo2 import Olmo2Config
    from transformers.models.olmo2.modeling_olmo2 import Olmo2ForCausalLM

    OLMO2_AVAILABLE = True
except ImportError:
    OLMO2_AVAILABLE = False

try:
    # OLMO3 is only available in transformers>=4.57.0
    from transformers.models.olmo3.configuration_olmo3 import Olmo3Config
    from transformers.models.olmo3.modeling_olmo3 import Olmo3ForCausalLM

    OLMO3_AVAILABLE = True
except ImportError:
    OLMO3_AVAILABLE = False

try:
    # Glm4 is only available in transformers>=4.51.3
    from transformers.models.glm4.configuration_glm4 import Glm4Config
    from transformers.models.glm4.modeling_glm4 import Glm4ForCausalLM

    GLM4_AVAILABLE = True
except ImportError:
    GLM4_AVAILABLE = False

try:
    # Glm4v is only available in transformers>=4.51.3
    from transformers.models.glm4v.configuration_glm4v import Glm4vConfig
    from transformers.models.glm4v.modeling_glm4v import Glm4vForConditionalGeneration

    GLM4V_AVAILABLE = True
except ImportError:
    GLM4V_AVAILABLE = False

try:
    # Glm4v_moe is only available in transformers>=4.51.3
    from transformers.models.glm4v_moe.configuration_glm4v_moe import Glm4vMoeConfig
    from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeForConditionalGeneration

    GLM4V_MOE_AVAILABLE = True
except ImportError:
    GLM4V_MOE_AVAILABLE = False

try:
    from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

    GEMMA3_AVAILABLE = True
except ImportError:
    GEMMA3_AVAILABLE = False

try:
    # Smollm3 is only available in transformers>=4.53.0
    from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config
    from transformers.models.smollm3.modeling_smollm3 import SmolLM3ForCausalLM

    SMOLLM3_AVAILABLE = True
except ImportError:
    SMOLLM3_AVAILABLE = False

try:
    # InternVL is only available in transformers>=4.52.1
    from transformers.models.internvl.configuration_internvl import InternVLConfig
    from transformers.models.internvl.modeling_internvl import InternVLForConditionalGeneration

    INTERNVL_AVAILABLE = True
except ImportError:
    INTERNVL_AVAILABLE = False

try:
    # FalconH1 is only available in transformers>=4.53.0
    from transformers.models.falcon_h1.configuration_falcon_h1 import FalconH1Config
    from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1ForCausalLM

    FALCONH1_AVAILABLE = True
except ImportError:
    FALCONH1_AVAILABLE = False

try:
    # Qwen3Next is only available in transformers>=4.57.0
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM

    QWEN3NEXT_AVAILABLE = True
except ImportError:
    QWEN3NEXT_AVAILABLE = False

try:
    from transformers.models.hunyuan_v1_dense.configuration_hunyuan_v1_dense import HunYuanDenseV1Config
    from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import HunYuanDenseV1ForCausalLM
    from transformers.models.hunyuan_v1_moe.configuration_hunyuan_v1_moe import HunYuanMoEV1Config
    from transformers.models.hunyuan_v1_moe.modeling_hunyuan_v1_moe import HunYuanMoEV1ForCausalLM

    HUNYUAN_V1_AVAILABLE = True
except ImportError:
    HUNYUAN_V1_AVAILABLE = False

from liger_kernel.utils import infer_device

device = infer_device()

MINI_MODEL_SETUPS = {
    "mini_llama3": MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_llama,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_llama,
        model_class=LlamaForCausalLM,
        mini_model_config=LlamaConfig(
            attention_bias=False,
            attention_dropout=0.0,
            # Special token ids/vocab size to match Mistral-7B tokenizer used to create the tokenized dataset
            # https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json
            bos_token_id=1,  # 128000
            eos_token_id=2,  # 128001
            hidden_act="silu",
            hidden_size=1024,  # 4096
            initializer_range=0.02,
            intermediate_size=2048,  # 14336
            max_position_embeddings=8192,
            num_attention_heads=8,  # 32
            num_hidden_layers=4,  # 32
            num_key_value_heads=2,  # 8
            pretraining_tp=1,
            rms_norm_eps=1e-5,
            rope_scaling=None,
            rope_theta=500000.0,
            tie_word_embeddings=False,
            use_cache=True,
            vocab_size=32000,  # 128256,
            # At rope backward
            # Eager produces incontiguous dq and dk
            # SDPA produces contiguous dq and incontiguous dk
            # Flash_attn produces contiguous dq and dk
            attn_implementation="sdpa",  # default value, pytorch native attention
        ),
    ),
    "mini_qwen2": MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_qwen2,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_qwen2,
        model_class=Qwen2ForCausalLM,
        mini_model_config=Qwen2Config(
            attention_dropout=0.0,
            bos_token_id=1,  # 151643
            eos_token_id=2,  # 151643
            hidden_act="silu",
            hidden_size=896,
            initializer_range=0.02,
            intermediate_size=4864,
            max_position_embeddings=32768,  # 131072
            num_attention_heads=8,
            num_hidden_layers=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            sliding_window=131072,
            tie_word_embeddings=True,
            use_cache=True,
            vocab_size=32000,  # 151936
            # At rope backward
            # Eager produces incontiguous dq and dk
            # SDPA produces contiguous dq and incontiguous dk
            # Flash_attn produces contiguous dq and dk
            attn_implementation="sdpa",  # default value, pytorch native attention
        ),
    ),
    "mini_phi3": MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_phi3,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_phi3,
        model_class=Phi3ForCausalLM,
        mini_model_config=Phi3Config(
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,  # 32000
            hidden_act="silu",
            hidden_size=896,  # 3072
            initializer_range=0.02,
            intermediate_size=4864,  # 8192
            max_position_embeddings=4096,
            num_attention_heads=8,  # 32
            num_hidden_layers=4,  # 32
            num_key_value_heads=None,  # defaults to num_attention_heads
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            sliding_window=None,
            tie_word_embeddings=False,
            use_cache=True,
            vocab_size=32064,
            attn_implementation="eager",
        ),
    ),
    "mini_mistral": MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_mistral,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_mistral,
        model_class=MistralForCausalLM,
        mini_model_config=MistralConfig(
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=1024,
            initializer_range=0.02,
            intermediate_size=2048,
            max_position_embeddings=32768,
            num_attention_heads=8,
            num_hidden_layers=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            sliding_window=4096,
            tie_word_embeddings=False,
            use_cache=True,
            vocab_size=32000,
            attn_implementation="sdpa",
        ),
    ),
    "mini_mixtral": MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_mixtral,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_mixtral,
        model_class=MixtralForCausalLM,
        mini_model_config=MixtralConfig(
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,  # 4096
            initializer_range=0.02,
            intermediate_size=2048,  # 14336
            max_position_embeddings=32768,  # 32768
            num_attention_heads=8,  # 32
            num_hidden_layers=4,  # 32
            num_key_value_heads=2,  # 8
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            sliding_window=4096,
            tie_word_embeddings=False,
            use_cache=True,
            vocab_size=32000,
            attn_implementation="sdpa",
        ),
    ),
    "mini_gemma1": MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_gemma,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_gemma,
        model_class=GemmaForCausalLM,
        mini_model_config=GemmaConfig(
            vocab_size=32000,  # 256000
            hidden_size=1024,  # 3072
            intermediate_size=2048,  # 24576
            num_hidden_layers=4,  # 28
            num_attention_heads=4,  # 16
            num_key_value_heads=4,  # 16
            head_dim=256,
            # gemma1 model config uses `hidden_act` and point it to gelu,
            # https://huggingface.co/google/gemma-7b/blob/main/config.json#L10
            # but in reality it's ignored and HuggingFace will use tanh approximation:
            # https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/gemma/modeling_gemma.py#L175
            hidden_act="gelu",
            max_position_embeddings=8192,
            initializer_range=0.02,
            rms_norm_eps=1e-06,
            use_cache=True,
            pad_token_id=0,
            # Special token ids/vocab size to match Mistral-7B tokenizer used to create the tokenized dataset
            # https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json
            bos_token_id=1,  # 128000
            eos_token_id=2,  # 128001
            tie_word_embeddings=True,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
        ),
    ),
    "mini_gemma1.1": MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_gemma,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_gemma,
        model_class=GemmaForCausalLM,
        mini_model_config=GemmaConfig(
            vocab_size=32000,  # 256000
            hidden_size=1024,  # 3072
            intermediate_size=2048,  # 24576
            num_hidden_layers=4,  # 28
            num_attention_heads=4,  # 16
            num_key_value_heads=4,  # 16
            head_dim=256,
            hidden_activation="gelu_pytorch_tanh",
            max_position_embeddings=8192,
            initializer_range=0.02,
            rms_norm_eps=1e-06,
            use_cache=True,
            pad_token_id=0,
            # Special token ids/vocab size to match Mistral-7B tokenizer used to create the tokenized dataset
            # https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json
            bos_token_id=1,  # 128000
            eos_token_id=2,  # 128001
            tie_word_embeddings=True,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
        ),
    ),
    "mini_gemma2": MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_gemma2,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_gemma2,
        model_class=Gemma2ForCausalLM,
        mini_model_config=Gemma2Config(
            vocab_size=32000,  # 256000
            hidden_size=1024,  # 3072
            intermediate_size=2048,  # 24576
            num_hidden_layers=4,  # 28
            num_attention_heads=4,  # 16
            num_key_value_heads=4,  # 16
            head_dim=256,
            hidden_activation="gelu_pytorch_tanh",
            max_position_embeddings=8192,
            initializer_range=0.02,
            rms_norm_eps=1e-06,
            use_cache=True,
            pad_token_id=0,
            # Special token ids/vocab size to match Mistral-7B tokenizer used to create the tokenized dataset
            # https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json
            bos_token_id=1,  # 128000
            eos_token_id=2,  # 128001
            tie_word_embeddings=True,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
            attn_implementation="eager",
        ),
    ),
}
if LLAMA4_AVAILABLE:
    MINI_MODEL_SETUPS["mini_llama4"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_llama4,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_llama4,
        model_class=Llama4ForCausalLM,
        mini_model_config=Llama4TextConfig(
            bos_token_id=1,  # None
            eos_token_id=2,  # 151329, 151336, 151338
            pad_token_id=2,  # 151329
            partial_rotary_factor=1.0,
            cross_attention_layers=None,
            dropout=0,
            hidden_act="silu",
            hidden_size=1024,  # 6144
            initializer_range=0.02,
            intermediate_size=2048,  # 14336
            max_position_embeddings=4096,  # 32768
            num_attention_heads=8,  # 48
            num_hidden_layers=4,  # 61
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            rope_scaling=None,
            rope_theta=10000.0,
            tie_word_embeddings=False,
            use_cache=True,
            vocab_size=32000,  # 151552
            attention_bias=True,
            attn_implementation="sdpa",  # default value, pytorch native attention
        ),
    )

if QWEN3_AVAILABLE:
    MINI_MODEL_SETUPS["mini_qwen3"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_qwen3,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_qwen3,
        model_class=Qwen3ForCausalLM,
        mini_model_config=Qwen3Config(
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=896,
            initializer_range=0.02,
            intermediate_size=4864,
            max_position_embeddings=32768,
            num_attention_heads=8,
            num_hidden_layers=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            sliding_window=131072,
            tie_word_embeddings=True,
            use_cache=True,
            vocab_size=32000,
            attn_implementation="sdpa",
        ),
    )

    MINI_MODEL_SETUPS["mini_qwen3_moe"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_qwen3_moe,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_qwen3_moe,
        model_class=Qwen3MoeForCausalLM,
        mini_model_config=Qwen3MoeConfig(
            vocab_size=32000,  # 151936
            hidden_size=896,
            intermediate_size=4864,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            hidden_act="silu",
            max_position_embeddings=32768,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            use_sliding_window=False,
            sliding_window=4096,
            max_window_layers=28,
            attention_dropout=0.0,
            decoder_sparse_step=1,
            moe_intermediate_size=768,
            num_experts_per_tok=2,
            num_experts=8,
            norm_topk_prob=False,
            output_router_logits=False,
            router_aux_loss_coef=0.001,
            mlp_only_layers=None,
        ),
    )

if GEMMA3_AVAILABLE:
    MINI_MODEL_SETUPS["mini_gemma3_text"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_gemma3_text,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_gemma3_text,
        model_class=Gemma3ForCausalLM,
        mini_model_config=Gemma3TextConfig(
            vocab_size=32000,  # 262144
            hidden_size=1024,  # 1152
            intermediate_size=2048,  # 6912
            num_hidden_layers=4,  # 26
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=256,
            hidden_activation="gelu_pytorch_tanh",
            max_position_embeddings=8192,  # 32768
            initializer_range=0.02,
            rms_norm_eps=1e-06,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=1,
            tie_word_embeddings=True,
            rope_theta=10000.0,  # 1000000
            attention_bias=False,
            attention_dropout=0.0,
            attn_implementation="eager",
        ),
    )

if MLLAMA_AVAILABLE:
    MINI_MODEL_SETUPS["mini_mllama"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_mllama,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_mllama,
        model_class=MllamaForCausalLM,
        mini_model_config=MllamaTextConfig(
            bos_token_id=1,  # 128000
            eos_token_id=2,  # 128001
            pad_token_id=2,
            cross_attention_layers=None,
            dropout=0,
            hidden_act="silu",
            hidden_size=1024,  # 4096
            initializer_range=0.02,
            intermediate_size=2048,  # 14336
            max_position_embeddings=131_072,
            num_attention_heads=8,  # 32
            num_hidden_layers=4,  # 40
            num_key_value_heads=2,  # 8
            rms_norm_eps=1e-5,
            rope_scaling=dict(
                factor=8.0,
                high_freq_factor=4.0,
                low_freq_factor=1.0,
                original_max_position_embeddings=8192,
                rope_type="llama3",
            ),
            rope_theta=500_000,
            tie_word_embeddings=False,
            use_cache=True,
            vocab_size=32000,  # 128256,
            attn_implementation="sdpa",  # default value, pytorch native attention
        ),
    )

if QWEN2_VL_AVAILABLE:
    MINI_MODEL_SETUPS["mini_qwen2_vl"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_qwen2_vl,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_qwen2_vl,
        model_class=Qwen2VLForConditionalGeneration,
        mini_model_config=Qwen2VLConfig(
            attention_dropout=0.0,
            # bos and eos set to match the Mistral-7B tokenizer used to create the test dataset
            # https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json
            bos_token_id=1,  # 151643
            eos_token_id=2,  # 151645
            vision_start_token_id=32765,  # vocab_size - 5
            vision_end_token_id=32766,  # vocab_size - 4
            vision_token_id=32767,  # vocab_size - 3
            image_token_id=32768,  # vocab_size - 2
            video_token_id=32769,  # vocab_size - 1
            hidden_act="silu",
            hidden_size=1536,  # 8192
            initializer_range=0.02,
            intermediate_size=4864,  # 29568
            max_position_embeddings=32768,
            max_window_layers=4,  # 80
            num_attention_heads=12,  # 64
            num_hidden_layers=4,  # 80
            num_key_value_heads=2,  # 8
            rms_norm_eps=1e-6,  # 1e-5
            rope_theta=1000000.0,
            rope_scaling=dict(
                type="mrope",
                mrope_section=[16, 24, 24],  # (temporal, height, width)
            ),
            sliding_window=4096,
            tie_word_embeddings=False,
            use_cache=True,
            vocab_size=32768,  # 152064  # >32k, Mistral-7B tokenizer vocab size
            use_sliding_window=False,
            vision_config={
                "depth": 4,  # 32
                "embed_dim": 1280,
                "mlp_ratio": 4,
                "num_heads": 16,
                "in_chans": 3,
                "hidden_size": 128,  # 1536
                "patch_size": 14,
                "spatial_merge_size": 2,
                "spatial_patch_size": 14,
                "temporal_patch_size": 2,
            },
            attn_implementation="sdpa",
        ),
    )

if QWEN2_5_VL_AVAILABLE:
    MINI_MODEL_SETUPS["mini_qwen2_5_vl"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_qwen2_5_vl,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_qwen2_5_vl,
        model_class=Qwen2_5_VLForConditionalGeneration,
        mini_model_config=Qwen2_5_VLConfig(
            attention_dropout=0.0,
            # bos and eos set to match the Mistral-7B tokenizer used to create the test dataset
            # https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json
            bos_token_id=1,  # 151643
            eos_token_id=2,  # 151645
            vision_start_token_id=32765,  # vocab_size - 5
            vision_end_token_id=32766,  # vocab_size - 4
            vision_token_id=32767,  # vocab_size - 3
            image_token_id=32768,  # vocab_size - 2
            video_token_id=32769,  # vocab_size - 1
            hidden_act="silu",
            hidden_size=1536,  # 8192
            initializer_range=0.02,
            intermediate_size=4864,  # 29568
            max_position_embeddings=32768,
            max_window_layers=4,  # 80
            num_attention_heads=12,  # 64
            num_hidden_layers=4,  # 80
            num_key_value_heads=2,  # 8
            rms_norm_eps=1e-6,  # 1e-5
            rope_theta=1000000.0,
            rope_scaling=dict(
                type="mrope",
                mrope_section=[16, 24, 24],  # (temporal, height, width)
            ),
            sliding_window=4096,
            tie_word_embeddings=False,
            use_cache=True,
            vocab_size=32768,  # 152064  # >32k, Mistral-7B tokenizer vocab size
            use_sliding_window=False,
            vision_config={
                "depth": 4,  # 32
                "hidden_act": "silu",
                "hidden_size": 128,  # 1280
                "intermediate_size": 256,  # 3420
                "num_heads": 16,
                "in_chans": 3,
                "out_hidden_size": 128,  # 3584
                "patch_size": 14,
                "spatial_merge_size": 2,
                "spatial_patch_size": 14,
                "window_size": 112,
                "fullatt_block_indexes": [7, 15, 23, 31],
                "tokens_per_second": 2,
                "temporal_patch_size": 2,
            },
            attn_implementation="sdpa",
        ),
    )

if QWEN3_VL_AVAILABLE:
    MINI_MODEL_SETUPS["mini_qwen3_vl"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_qwen3_vl,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_qwen3_vl,
        model_class=Qwen3VLForConditionalGeneration,
        mini_model_config=Qwen3VLConfig(
            tie_word_embeddings=False,
            image_token_id=31997,
            video_token_id=31998,
            vision_start_token_id=31995,
            vision_end_token_id=31996,
            text_config=dict(
                attention_dropout=0.0,
                attn_implementation="sdpa",
                bos_token_id=1,
                eos_token_id=2,
                head_dim=112,
                hidden_act="silu",
                hidden_size=896,
                initializer_range=0.02,
                intermediate_size=4864,
                max_position_embeddings=32768,
                num_attention_heads=8,
                num_hidden_layers=4,
                num_key_value_heads=2,
                pad_token_id=2,
                rms_norm_eps=1e-6,
                rope_theta=1000000.0,
                rope_scaling=dict(
                    type="mrope",
                    mrope_section=[16, 24, 24],
                ),
                sliding_window=131072,
                tie_word_embeddings=False,
                use_cache=True,
                vocab_size=32000,
            ),
            vision_config=dict(
                depth=4,
                hidden_size=128,
                initializer_range=0.02,
                intermediate_size=256,
                num_heads=8,
                in_channels=3,
                patch_size=14,
                spatial_merge_size=2,
                temporal_patch_size=2,
                out_hidden_size=896,
                num_position_embeddings=576,
                deepstack_visual_indexes=[1, 2, 3],
            ),
        ),
    )

if QWEN3_VL_MOE_AVAILABLE:
    MINI_MODEL_SETUPS["mini_qwen3_vl_moe"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_qwen3_vl_moe,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_qwen3_vl_moe,
        model_class=Qwen3VLMoeForConditionalGeneration,
        mini_model_config=Qwen3VLMoeConfig(
            tie_word_embeddings=False,
            image_token_id=31997,
            video_token_id=31998,
            vision_start_token_id=31995,
            vision_end_token_id=31996,
            text_config=Qwen3VLMoeTextConfig(
                attention_dropout=0.0,
                attention_bias=False,
                attn_implementation="sdpa",
                bos_token_id=1,
                eos_token_id=2,
                head_dim=112,
                hidden_act="silu",
                hidden_size=896,
                initializer_range=0.02,
                intermediate_size=4864,
                max_position_embeddings=32768,
                num_attention_heads=8,
                num_hidden_layers=4,
                num_key_value_heads=2,
                pad_token_id=2,
                rms_norm_eps=1e-6,
                rope_theta=1000000.0,
                rope_scaling=dict(
                    type="mrope",
                    mrope_section=[16, 24, 24],
                ),
                sliding_window=131072,
                tie_word_embeddings=False,
                use_cache=True,
                vocab_size=32000,
                decoder_sparse_step=1,
                moe_intermediate_size=3072,
                num_experts_per_tok=2,
                num_experts=4,
                mlp_only_layers=[],
            ).to_dict(),
            vision_config=Qwen3VLMoeVisionConfig(
                depth=4,
                hidden_size=128,
                initializer_range=0.02,
                intermediate_size=256,
                num_heads=8,
                in_channels=3,
                patch_size=14,
                spatial_merge_size=2,
                temporal_patch_size=2,
                out_hidden_size=896,
                num_position_embeddings=576,
                deepstack_visual_indexes=[1, 2, 3],
            ).to_dict(),
        ),
    )

if GRANITE_AVAILABLE:
    MINI_MODEL_SETUPS["mini_granite3"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_granite,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_granite,
        model_class=GraniteForCausalLM,
        mini_model_config=GraniteConfig(
            attention_bias=False,
            attention_dropout=0.0,
            # Special token ids/vocab size to match Mistral-7B tokenizer used to create the tokenized dataset
            # https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json
            bos_token_id=1,  # 128000
            eos_token_id=2,  # 128001
            hidden_act="silu",
            hidden_size=1024,  # 4096
            initializer_range=0.02,
            intermediate_size=2048,  # 14336
            max_position_embeddings=8192,
            num_attention_heads=8,  # 32
            num_hidden_layers=4,  # 32
            num_key_value_heads=2,  # 8
            pretraining_tp=1,
            rms_norm_eps=1e-5,
            rope_scaling=None,
            rope_theta=500000.0,
            tie_word_embeddings=False,
            use_cache=True,
            vocab_size=32000,  # 128256,
            logits_scaling=4.0,
            # At rope backward
            # Eager produces incontiguous dq and dk
            # SDPA produces contiguous dq and incontiguous dk
            # Flash_attn produces contiguous dq and dk
            attn_implementation="sdpa",  # default value, pytorch native attention
        ),
    )

if LLAVA_AVAILABLE:
    # https://huggingface.co/llava-hf/llava-1.5-7b-hf
    MINI_MODEL_SETUPS["mini_llava"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_llava,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_llava,
        model_class=LlavaForConditionalGeneration,
        mini_model_config=LlavaConfig(
            text_config=LlamaConfig(
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=1,
                eos_token_id=2,
                hidden_act="silu",
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=2048,
                num_attention_heads=8,
                num_hidden_layers=4,
                num_key_value_heads=2,
                pretraining_tp=1,
                rope_scaling=None,
                rope_theta=500000.0,
                tie_word_embeddings=False,
                use_cache=True,
                max_position_embeddings=4096,  # llava-1.5-7b-hf
                rms_norm_eps=1e-05,  # llava-1.5-7b-hf
                vocab_size=32064,  # llava-1.5-7b-hf
                # At rope backward
                # Eager produces incontiguous dq and dk
                # SDPA produces contiguous dq and incontiguous dk
                # Flash_attn produces contiguous dq and dk
                attn_implementation="sdpa",  # default value, pytorch native attention
            ),
            vision_config=CLIPVisionConfig(
                hidden_size=1024,
                image_size=336,
                intermediate_size=2048,  # 4096
                model_type="clip_vision_model",
                num_attention_heads=4,  # 16
                num_hidden_layers=4,  # 24
                patch_size=14,
                projection_dim=768,
                vocab_size=32000,
            ),
            vocab_size=32064,
            ignore_index=-100,
            pad_token_id=4,
            image_token_index=3,
            projector_hidden_act="gelu",
            vision_feature_layer=-2,
            vision_feature_select_strategy="default",
            # At rope backward
            # Eager produces incontiguous dq and dk
            # SDPA produces contiguous dq and incontiguous dk
            # Flash_attn produces contiguous dq and dk
            attn_implementation="sdpa",  # default value, pytorch native attention
        ),
    )

if OLMO2_AVAILABLE:
    MINI_MODEL_SETUPS["mini_olmo2"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_olmo2,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_olmo2,
        model_class=Olmo2ForCausalLM,
        mini_model_config=Olmo2Config(
            bos_token_id=1,  # 128000
            eos_token_id=2,  # 128001
            pad_token_id=2,
            cross_attention_layers=None,
            dropout=0,
            hidden_act="silu",
            hidden_size=1024,  # 4096
            initializer_range=0.02,
            intermediate_size=2048,  # 14336
            max_position_embeddings=4096,
            num_attention_heads=8,  # 32
            num_hidden_layers=4,  # 40
            num_key_value_heads=2,  # 8
            rms_norm_eps=1e-5,
            rope_scaling=None,
            rope_theta=500_000,
            tie_word_embeddings=False,
            use_cache=True,
            vocab_size=32000,  # 128256,
            attn_implementation="sdpa",  # default value, pytorch native attention
        ),
    )

if OLMO3_AVAILABLE:
    MINI_MODEL_SETUPS["mini_olmo3"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_olmo3,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_olmo3,
        model_class=Olmo3ForCausalLM,
        mini_model_config=Olmo3Config(
            bos_token_id=1,  # 128000
            eos_token_id=2,  # 128001
            pad_token_id=2,
            cross_attention_layers=None,
            dropout=0,
            hidden_act="silu",
            hidden_size=1024,  # 4096
            initializer_range=0.02,
            intermediate_size=2048,  # 14336
            max_position_embeddings=4096,
            num_attention_heads=8,  # 32
            num_hidden_layers=4,  # 40
            num_key_value_heads=2,  # 8
            rms_norm_eps=1e-5,
            rope_scaling=None,
            rope_theta=500_000,
            tie_word_embeddings=False,
            use_cache=True,
            vocab_size=32000,  # 128256,
            attn_implementation="sdpa",  # default value, pytorch native attention
        ),
    )

if GLM4_AVAILABLE:
    MINI_MODEL_SETUPS["mini_glm4"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_glm4,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_glm4,
        model_class=Glm4ForCausalLM,
        mini_model_config=Glm4Config(
            bos_token_id=1,  # None
            eos_token_id=2,  # 151329, 151336, 151338
            pad_token_id=2,  # 151329
            partial_rotary_factor=0.5,
            cross_attention_layers=None,
            dropout=0,
            hidden_act="silu",
            hidden_size=1024,  # 6144
            initializer_range=0.02,
            intermediate_size=2048,  # 14336
            max_position_embeddings=4096,  # 32768
            num_attention_heads=8,  # 48
            num_hidden_layers=4,  # 61
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            rope_scaling=None,
            rope_theta=500_000,
            tie_word_embeddings=False,
            use_cache=True,
            vocab_size=32000,  # 151552
            attention_bias=True,
            attn_implementation="sdpa",  # default value, pytorch native attention
        ),
    )

if GLM4V_AVAILABLE:
    MINI_MODEL_SETUPS["mini_glm4v"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_glm4v,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_glm4v,
        model_class=Glm4vForConditionalGeneration,
        mini_model_config=Glm4vConfig(
            bos_token_id=1,  # None
            eos_token_id=2,  # 151329, 151336, 151338
            pad_token_id=2,  # 151329
            image_token_id=151343,
            video_token_id=151344,
            image_start_token_id=151339,
            image_end_token_id=151340,
            video_start_token_id=151341,
            video_end_token_id=151342,
            partial_rotary_factor=0.5,
            cross_attention_layers=None,
            dropout=0,
            hidden_act="silu",
            hidden_size=1024,  # 6144
            initializer_range=0.02,
            intermediate_size=2048,  # 14336
            max_position_embeddings=4096,  # 32768
            num_attention_heads=8,  # 48
            num_hidden_layers=4,  # 61
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            rope_scaling=None,
            rope_theta=500_000,
            tie_word_embeddings=False,
            use_cache=True,
            vocab_size=32000,  # 151552
            attention_bias=True,
            attn_implementation="sdpa",  # default value, pytorch native attention
            text_config={
                "partial_rotary_factor": 0.5,
                "hidden_act": "silu",
                "hidden_size": 1024,
                "intermediate_size": 2048,
                "max_position_embeddings": 4096,
                "num_attention_heads": 8,
                "num_hidden_layers": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5,
                "rope_scaling": {
                    "type": "default",
                    "mrope_section": [8, 12, 12],  # (temporal, height, width)
                },
                "rope_theta": 500_000,
                "vocab_size": 32000,
                "attention_bias": True,
            },
            vision_config={
                "depth": 4,  # 32
                "hidden_act": "silu",
                "hidden_size": 128,  # 1280
                "intermediate_size": 256,  # 3420
                "num_heads": 16,
                "in_chans": 3,
                "out_hidden_size": 128,  # 3584
                "patch_size": 14,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
        ),
    )
if GLM4V_MOE_AVAILABLE:
    MINI_MODEL_SETUPS["mini_glm4v_moe"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_glm4v_moe,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_glm4v_moe,
        model_class=Glm4vMoeForConditionalGeneration,
        mini_model_config=Glm4vMoeConfig(
            bos_token_id=1,  # None
            eos_token_id=2,  # 151329, 151336, 151338
            pad_token_id=2,  # 151329
            image_token_id=151343,
            video_token_id=151344,
            image_start_token_id=151339,
            image_end_token_id=151340,
            video_start_token_id=151341,
            video_end_token_id=151342,
            partial_rotary_factor=0.5,
            cross_attention_layers=None,
            dropout=0,
            hidden_act="silu",
            hidden_size=1024,  # 6144
            initializer_range=0.02,
            intermediate_size=2048,  # 14336
            max_position_embeddings=4096,  # 32768
            num_attention_heads=8,  # 48
            num_hidden_layers=4,  # 61
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            rope_scaling=None,
            rope_theta=500_000,
            tie_word_embeddings=False,
            use_cache=True,
            vocab_size=32000,  # 151552
            attention_bias=True,
            attn_implementation="sdpa",  # default value, pytorch native attention
            text_config={
                "partial_rotary_factor": 0.5,
                "hidden_act": "silu",
                "hidden_size": 1024,
                "intermediate_size": 2048,
                "max_position_embeddings": 4096,
                "num_attention_heads": 8,
                "num_hidden_layers": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5,
                "rope_scaling": {
                    "type": "default",
                    "mrope_section": [8, 12, 12],  # (temporal, height, width)
                },
                "rope_theta": 500_000,
                "vocab_size": 32000,
                "attention_bias": True,
                "attention_dropout": 0.0,
                "moe_intermediate_size": 1408,
                "num_experts_per_tok": 2,
                "n_shared_experts": 1,
                "n_routed_experts": 8,
                "routed_scaling_factor": 1.0,
                "n_group": 1,
                "topk_group": 1,
                "first_k_dense_replace": 1,
                "norm_topk_prob": True,
            },
            vision_config={
                "depth": 4,  # 32
                "hidden_act": "silu",
                "hidden_size": 128,  # 1280
                "intermediate_size": 256,  # 3420
                "num_heads": 16,
                "in_chans": 3,
                "out_hidden_size": 128,  # 3584
                "patch_size": 14,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
        ),
    )

if SMOLLM3_AVAILABLE:
    MINI_MODEL_SETUPS["mini_smollm3"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_smollm3,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_smollm3,
        model_class=SmolLM3ForCausalLM,
        mini_model_config=SmolLM3Config(
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=1,  # 128000
            eos_token_id=2,  # 128001
            pad_token_id=2,  # 128000
            hidden_act="silu",
            hidden_size=1024,  # 4096
            initializer_range=0.02,
            intermediate_size=2048,  # 14336
            max_position_embeddings=8192,
            num_attention_heads=8,  # 32
            num_hidden_layers=4,  # 32
            num_key_value_heads=2,  # 8
            pretraining_tp=1,
            rms_norm_eps=1e-5,
            rope_scaling=None,
            rope_theta=500000.0,
            tie_word_embeddings=False,
            use_cache=True,
            vocab_size=32000,  # 128256,
            # At rope backward
            # Eager produces incontiguous dq and dk
            # SDPA produces contiguous dq and incontiguous dk
            # Flash_attn produces contiguous dq and dk
            attn_implementation="sdpa",  # default value, pytorch native attention
        ),
    )

if INTERNVL_AVAILABLE:
    MINI_MODEL_SETUPS["mini_internvl"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_internvl,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_internvl,
        model_class=InternVLForConditionalGeneration,
        mini_model_config=InternVLConfig(
            text_config=Qwen2Config(
                rms_norm_eps=1e-5,
                hidden_size=256,  # 1024
                intermediate_size=1024,  # 4096
                hidden_act="silu",
                num_hidden_layers=4,  # 24
                num_attention_heads=4,  # 16
                num_key_value_heads=2,  # 16
                max_position_embeddings=4096,  # 8192
                vocab_size=32000,  # 151936
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=2,
                tie_word_embeddings=False,
            ),
            vision_config={
                "hidden_size": 256,  # 1024
                "intermediate_size": 1024,  # 4096
                "num_hidden_layers": 4,  # 24
                "num_attention_heads": 4,  # 16
            },
            image_token_id=10,
            attn_implementation="sdpa",  # default value, pytorch native attention
        ),
    )

if FALCONH1_AVAILABLE:
    MINI_MODEL_SETUPS["mini_falcon_h1"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_falcon_h1,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_falcon_h1,
        model_class=FalconH1ForCausalLM,
        mini_model_config=FalconH1Config(
            model_type="falcon_h1",
            vocab_size=32000,
            hidden_size=256,  # 4096
            num_hidden_layers=4,  # 24
            num_attention_heads=4,  # 32
            num_key_value_heads=2,  # 8
            intermediate_size=1024,  # 11008
            hidden_act="silu",
            max_position_embeddings=4096,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            mamba_d_ssm=128,  # 1024
            mamba_n_heads=16,  # 128
            mamba_d_state=32,  # 245
            mamba_d_conv=2,  # 4
        ),
    )

if QWEN3NEXT_AVAILABLE:
    MINI_MODEL_SETUPS["mini_qwen3_next"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_qwen3_next,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_qwen3_next,
        model_class=Qwen3NextForCausalLM,
        mini_model_config=Qwen3NextConfig(  # Copypaste Qwen3MoeConfig
            vocab_size=32000,
            hidden_size=896,
            intermediate_size=4864,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            hidden_act="silu",
            max_position_embeddings=32768,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            use_sliding_window=False,
            sliding_window=4096,
            max_window_layers=28,
            attention_dropout=0.0,
            decoder_sparse_step=1,
            moe_intermediate_size=768,
            num_experts_per_tok=2,
            num_experts=8,
            norm_topk_prob=False,
            output_router_logits=False,
            router_aux_loss_coef=0.001,
            # config.dtype must be set if fla installed since there's a bug in the original code (No torch.get_current_dtype())
            # https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L613
            dtype=torch.float32,
        ),
    )


if HUNYUAN_V1_AVAILABLE:
    MINI_MODEL_SETUPS["mini_hunyuan_v1"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_hunyuan_v1_dense,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_hunyuan_v1,
        model_class=HunYuanDenseV1ForCausalLM,
        mini_model_config=HunYuanDenseV1Config(
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            num_hidden_layers=4,
            hidden_size=896,
            intermediate_size=4864,
            num_attention_heads=8,
            head_dim=112,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
            max_position_embeddings=32768,
            initializer_range=0.02,
            norm_eps=1e-6,
            num_key_value_heads=2,
            rope_theta=10000.0,
            partial_rotary_factor=1.0,
            vocab_size=32000,
            use_cache=True,
            attn_implementation="sdpa",
        ),
    )

    MINI_MODEL_SETUPS["mini_hunyuan_v1_moe"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_hunyuan_v1_moe,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_hunyuan_v1_moe,
        model_class=HunYuanMoEV1ForCausalLM,
        mini_model_config=HunYuanMoEV1Config(
            hidden_act="silu",
            attention_dropout=0.0,
            num_hidden_layers=4,
            hidden_size=896,
            intermediate_size=4864,
            num_attention_heads=8,
            head_dim=112,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
            max_position_embeddings=32768,
            initializer_range=0.02,
            norm_eps=1e-6,
            num_key_value_heads=2,
            rope_theta=10000.0,
            partial_rotary_factor=1.0,
            vocab_size=32000,
            num_experts=8,
            moe_topk=2,
            use_cache=True,
            attn_implementation="sdpa",
        ),
    )


def create_model(model_name="mini_llama3"):
    """
    Create a mini version model
    The commented values are the original values
    """
    model_config = MINI_MODEL_SETUPS[model_name].mini_model_config
    model_class = MINI_MODEL_SETUPS[model_name].model_class
    return model_class(model_config)


@require_deterministic
def run_mini_model(
    model_name="mini_llama3",
    num_steps=100,
    dtype=torch.bfloat16,
    lr=1e-5,
    with_liger=False,
):
    # If we move it to the beginning of test_mini_model, the two runs are initialized with different weights.
    # This is due to RNG (Random Number Generator). The formula of RNG progression is x_(n+1) = (a * x_n + c) % m
    # Everytime RNG is used, like randomly initialzing weight, the RNG progresses to the next state.
    # Therefore, we have to reset RNG before we create the model to ensure the weight initialization started from the same RNG state.

    set_seed(42)

    revert_kwargs = {"model_config": MINI_MODEL_SETUPS[model_name]}
    if "mllama" in model_name:
        revert_kwargs["model_type"] = "causal_lm"

    if with_liger is True:
        kwargs = {
            "rope": True,
            "rms_norm": True,
        }

        if "glm4" in model_name or "llama4" in model_name:
            kwargs["rope"] = False

        model_supports_layer_norm = "qwen2_vl" in model_name
        if model_supports_layer_norm:
            kwargs["layer_norm"] = True

        if "gemma" in model_name:
            kwargs["geglu"] = True
        else:
            kwargs["swiglu"] = True

        if "llava" in model_name:
            apply_liger_kernel_to_llama(**kwargs)

        kwargs["fused_linear_cross_entropy"] = False
        kwargs["cross_entropy"] = False

        MINI_MODEL_SETUPS[model_name].liger_kernel_patch_func(**kwargs)
    else:
        MINI_MODEL_SETUPS[model_name].liger_kernel_patch_revert_func(**revert_kwargs)

    model = create_model(model_name).to(dtype).to(device)

    train_dataset = load_from_disk(DEFAULT_DATASET_PATH)
    loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=simple_collate_fn)
    loader_iter = iter(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loss_list = []

    for i in range(num_steps):
        batch = next(loader_iter).to(model.device)
        optimizer.zero_grad()
        output = model(**batch)
        output.loss.backward()
        optimizer.step()
        print(f"Step {i}, Loss: {output.loss.item()}")
        loss_list.append(output.loss.item())

    topk_logprobs = get_topk(get_logprobs(output.logits))
    MINI_MODEL_SETUPS[model_name].liger_kernel_patch_revert_func(**revert_kwargs)
    return {
        "loss": loss_list,
        "topk_logprobs": topk_logprobs.values,
        "model": model,
    }


@pytest.mark.parametrize(
    "model_name, num_steps, lr, dtype, loss_atol, loss_rtol, logprobs_atol, logprobs_rtol, param_atol, param_rtol",
    [
        pytest.param(
            "mini_llama4",
            32,
            1e-4,
            torch.float32,
            1e-8,
            1e-5,
            5e-3,
            1e-5,
            5e-3,
            1e-5,
            marks=pytest.mark.skipif(
                not LLAMA4_AVAILABLE,
                reason="Llama4 not available in this version of transformers",
            ),
        ),
        ("mini_llama3", 32, 1e-4, torch.float32, 1e-8, 2e-5, 5e-3, 1e-5, 5e-3, 1e-5),
        pytest.param(
            "mini_llava",
            32,
            1e-4,
            torch.float32,
            1e-8,
            1e-5,
            5e-3,
            1e-5,
            5e-3,
            1e-5,
            marks=pytest.mark.skipif(
                not LLAVA_AVAILABLE,
                reason="LLaVa not available in this version of transformers",
            ),
        ),
        pytest.param(
            "mini_mllama",
            32,
            1e-4,
            torch.float32,
            1e-8,
            1e-5,
            5e-3,
            1e-5,
            5e-3,
            1e-5,
            marks=pytest.mark.skipif(
                not MLLAMA_AVAILABLE,
                reason="Mllama not available in this version of transformers",
            ),
        ),
        pytest.param(
            "mini_gemma3_text",
            32,
            1e-5,
            torch.float32,
            1e-8,
            1e-4,
            5e-2,
            1e-5,
            5e-3,
            1e-5,
            marks=pytest.mark.skipif(
                not GEMMA3_AVAILABLE,
                reason="Gemma3 not available in this version of transformers",
            ),
        ),
        ("mini_qwen2", 32, 1e-4, torch.float32, 1e-8, 1e-5, 5e-3, 1e-5, 5e-3, 1e-5),
        pytest.param(
            "mini_qwen3",
            32,
            1e-4,
            torch.float32,
            1e-8,
            1e-5,
            5e-3,
            1e-5,
            5e-3,
            1e-5,
            marks=pytest.mark.skipif(
                not QWEN3_AVAILABLE,
                reason="Qwen3 not available in this version of transformers",
            ),
        ),
        pytest.param(
            "mini_qwen3_moe",
            32,
            1e-5,
            torch.float32,
            1e-8,
            1e-5,
            5e-3,
            1e-5,
            5e-3,
            1e-5,
            marks=pytest.mark.skipif(
                not QWEN3_AVAILABLE,
                reason="Qwen3 not available in this version of transformers",
            ),
        ),
        pytest.param(
            "mini_qwen2_vl",
            32,
            1e-4,
            torch.float32,
            1e-8,
            2e-5,
            5e-3,
            1e-5,
            5e-3,
            1e-5,
            marks=pytest.mark.skipif(
                not QWEN2_VL_AVAILABLE,
                reason="Qwen2-VL not available in this version of transformers",
            ),
        ),
        pytest.param(
            "mini_qwen2_5_vl",
            32,
            1e-4,
            torch.float32,
            1e-8,
            2e-5,
            5e-3,
            1e-5,
            5e-3,
            1e-5,
            marks=pytest.mark.skipif(
                not QWEN2_5_VL_AVAILABLE,
                reason="Qwen2.5-VL not available in this version of transformers",
            ),
        ),
        pytest.param(
            "mini_qwen3_vl",
            32,
            1e-4,
            torch.float32,
            1e-8,
            2e-5,
            5e-3,
            1e-5,
            5e-3,
            1e-5,
            marks=pytest.mark.skipif(
                not QWEN3_VL_AVAILABLE,
                reason="Qwen3-VL not available in this version of transformers",
            ),
        ),
        pytest.param(
            "mini_qwen3_vl_moe",
            32,
            1e-4,
            torch.float32,
            1e-8,
            2e-5,
            5e-3,
            1e-5,
            5e-3,
            1e-5,
            marks=pytest.mark.skipif(
                not QWEN3_VL_MOE_AVAILABLE,
                reason="Qwen3-VL-MoE not available in this version of transformers",
            ),
        ),
        pytest.param(
            "mini_olmo2",
            32,
            1e-4,
            torch.float32,
            1e-8,
            1e-5,
            5e-3,
            1e-5,
            5e-3,
            1e-5,
            marks=pytest.mark.skipif(
                not OLMO2_AVAILABLE,
                reason="OLMO2 not available in this version of transformers",
            ),
        ),
        pytest.param(
            "mini_olmo3",
            32,
            1e-4,
            torch.float32,
            1e-8,
            1e-5,
            5e-3,
            1e-5,
            5e-3,
            1e-5,
            marks=pytest.mark.skipif(
                not OLMO3_AVAILABLE,
                reason="OLMO3 not available in this version of transformers",
            ),
        ),
        pytest.param(
            "mini_glm4",
            32,
            1e-4,
            torch.float32,
            1e-8,
            1e-5,
            5e-3,
            1e-5,
            5e-3,
            1e-5,
            marks=pytest.mark.skipif(
                not GLM4_AVAILABLE,
                reason="Glm4 not available in this version of transformers",
            ),
        ),
        pytest.param(
            "mini_glm4v",
            32,
            1e-4,
            torch.float32,
            1e-8,
            1e-5,
            5e-3,
            1e-5,
            5e-3,
            1e-5,
            marks=pytest.mark.skipif(
                not GLM4V_AVAILABLE,
                reason="Glm4v not available in this version of transformers",
            ),
        ),
        pytest.param(
            "mini_glm4v_moe",
            32,
            1e-4,
            torch.float32,
            1e-8,
            1e-5,
            5e-3,
            1e-5,
            5e-3,
            1e-5,
            marks=pytest.mark.skipif(
                not GLM4V_MOE_AVAILABLE,
                reason="Glm4v_moe not available in this version of transformers",
            ),
        ),
        ("mini_phi3", 32, 1e-4, torch.float32, 1e-8, 1e-5, 5e-3, 1e-5, 5e-3, 1e-5),
        ("mini_mistral", 32, 1e-4, torch.float32, 1e-8, 1e-5, 5e-3, 1e-5, 5e-3, 1e-5),
        # TODO: mixtral is flaky so disable the test for now
        # ("mini_mixtral", 32, 1e-4, torch.float32, 5e-4, 1e-4, 5e-3, 1e-5, 1e-2, 1e-5),
        # Gemma 1.1 and 2 has more tolerance because currently, the kernel is not a perfect match
        ("mini_gemma1", 32, 1e-5, torch.float32, 1e-8, 1e-4, 5e-3, 1e-5, 5e-3, 1e-5),
        ("mini_gemma1.1", 32, 1e-4, torch.float32, 1e-8, 1e-4, 5e-3, 1e-5, 5e-3, 1e-5),
        ("mini_gemma2", 32, 1e-4, torch.float32, 1e-8, 1e-4, 5e-3, 1e-5, 5e-3, 1e-5),
        pytest.param(
            "mini_granite3",
            32,
            1e-4,
            torch.float32,
            1e-8,
            1e-4,
            5e-3,
            1e-5,
            5e-3,
            1e-5,
            marks=pytest.mark.skipif(
                not GRANITE_AVAILABLE,
                reason="Granite not available in this version of transformers",
            ),
        ),
        pytest.param(
            "mini_smollm3",
            32,
            1e-4,
            torch.bfloat16,
            1e-3,
            1e-2,
            1e-1,
            1e-2,
            1e-2,
            1e-2,
            marks=pytest.mark.skipif(
                not SMOLLM3_AVAILABLE,
                reason="Smollm3 not available in this version of transformers",
            ),
        ),
        pytest.param(
            "mini_internvl",
            32,
            1e-4,
            torch.float32,
            1e-8,
            1e-5,
            5e-3,
            1e-5,
            5e-3,
            1e-5,
            marks=pytest.mark.skipif(
                not INTERNVL_AVAILABLE,
                reason="InternVL not available in this version of transformers",
            ),
        ),
        pytest.param(
            "mini_falcon_h1",
            32,
            1e-5,
            torch.float32,
            1e-8,
            1e-4,
            4e-2,
            1e-5,
            5e-3,
            1e-5,
            marks=pytest.mark.skipif(
                not FALCONH1_AVAILABLE,
                reason="FalconH1 not available in this version of transformers",
            ),
        ),
        pytest.param(
            "mini_qwen3_next",
            32,
            1e-5,
            torch.float32,
            1e-8,
            1e-5,
            5e-3,
            1e-5,
            5e-3,
            1e-5,
            marks=[
                pytest.mark.skipif(
                    not QWEN3NEXT_AVAILABLE,
                    reason="Qwen3Next not available in this version of transformers",
                ),
                pytest.mark.skip(
                    reason="flash-linear-attention's ChunkGatedDeltaRuleFunction does not support float32.\n"
                    + " Torch's implementation takes too long"
                ),
            ],
        ),
        pytest.param(
            "mini_hunyuan_v1",
            32,
            1e-5,
            torch.float32,
            1e-2,
            5e-2,
            1e-1,
            1e-2,
            1e-2,
            1e-2,
            marks=[
                pytest.mark.skipif(
                    not HUNYUAN_V1_AVAILABLE,
                    reason="Hunyuan_v1 not available in this version of transformers",
                ),
            ],
        ),
        pytest.param(
            "mini_hunyuan_v1_moe",
            32,
            1e-5,
            torch.float32,
            1e-2,
            5e-2,
            1e-1,
            1e-2,
            1e-2,
            1e-2,
            marks=[
                pytest.mark.skipif(
                    not HUNYUAN_V1_AVAILABLE,
                    reason="Hunyuan_v1_moe not available in this version of transformers",
                ),
            ],
        ),
    ],
)
def test_mini_model(
    model_name,
    num_steps,
    lr,
    dtype,
    loss_atol,
    loss_rtol,
    logprobs_atol,
    logprobs_rtol,
    param_atol,
    param_rtol,
):
    # Non-liger models should be initialized and tested first to avoid the module being overridden

    expected_output = run_mini_model(model_name=model_name, num_steps=num_steps, dtype=dtype, lr=lr)

    actual_output = run_mini_model(model_name=model_name, num_steps=num_steps, dtype=dtype, lr=lr, with_liger=True)

    # Compare every step of the loss
    assert_verbose_allclose(
        torch.tensor([expected_output["loss"]]),
        torch.tensor([actual_output["loss"]]),
        atol=loss_atol,
        rtol=loss_rtol,
        extra_info="[Loss]",
    )

    # No logits are materialized
    # import pdb; pdb.set_trace()
    assert_verbose_allclose(
        expected_output["topk_logprobs"],
        actual_output["topk_logprobs"],
        atol=logprobs_atol,
        rtol=logprobs_rtol,
        extra_info="[Top K Logprobs]",
    )

    # Compare the params from the last step
    # Iterate over the model's parameters and compare them
    for expected_param, actual_param in zip(
        expected_output["model"].named_parameters(),
        actual_output["model"].named_parameters(),
    ):
        assert_verbose_allclose(
            expected_param[1],
            actual_param[1],
            atol=param_atol,
            rtol=param_rtol,
            extra_info="[Model parameters]",
        )
