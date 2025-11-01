import functools
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Ensure deterministic behavior with CuBLAS
import pytest
import torch

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

from liger_kernel.transformers import apply_liger_kernel_to_gemma3
from liger_kernel.transformers import apply_liger_kernel_to_internvl
from liger_kernel.transformers import apply_liger_kernel_to_llama4
from liger_kernel.transformers import apply_liger_kernel_to_llava
from liger_kernel.transformers import apply_liger_kernel_to_mllama
from liger_kernel.transformers import apply_liger_kernel_to_paligemma
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl_moe
from liger_kernel.transformers import apply_liger_kernel_to_smolvlm
from test.utils import FAKE_CONFIGS_PATH
from test.utils import UNTOKENIZED_DATASET_PATH
from test.utils import MiniModelConfig
from test.utils import assert_verbose_allclose
from test.utils import get_logprobs
from test.utils import get_topk
from test.utils import is_torchvision_available
from test.utils import load_image_processing_config
from test.utils import load_processor_config
from test.utils import load_tokenizer_config
from test.utils import multimodal_collate_fn
from test.utils import require_deterministic
from test.utils import revert_liger_kernel_to_gemma3
from test.utils import revert_liger_kernel_to_internvl
from test.utils import revert_liger_kernel_to_llama4
from test.utils import revert_liger_kernel_to_llava
from test.utils import revert_liger_kernel_to_mllama
from test.utils import revert_liger_kernel_to_Paligemma
from test.utils import revert_liger_kernel_to_qwen2_5_vl
from test.utils import revert_liger_kernel_to_qwen2_vl
from test.utils import revert_liger_kernel_to_qwen3_vl
from test.utils import revert_liger_kernel_to_qwen3_vl_moe
from test.utils import revert_liger_kernel_to_smolvlm2
from test.utils import set_seed
from test.utils import supports_bfloat16
from test.utils import train_bpe_tokenizer

try:
    # Qwen2-VL is only available in transformers>=4.52.4
    import transformers

    from packaging import version
    from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
    from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

    QWEN2_VL_AVAILABLE = version.parse(transformers.__version__) >= version.parse("4.52.4")
except ImportError:
    QWEN2_VL_AVAILABLE = False

try:
    # Qwen2.5-VL is only available in transformers>4.52.4
    import transformers

    from packaging import version
    from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

    QWEN2_5_VL_AVAILABLE = version.parse(transformers.__version__) >= version.parse("4.52.4")
except ImportError:
    QWEN2_5_VL_AVAILABLE = False

try:
    from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    QWEN3_VL_AVAILABLE = True
except ImportError:
    QWEN3_VL_AVAILABLE = False

try:
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeConfig
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeTextConfig
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeVisionConfig
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

    QWEN3_VL_MOE_AVAILABLE = True
except ImportError:
    QWEN3_VL_MOE_AVAILABLE = False


try:
    # Mllama is only available in transformers>=4.45.0
    from transformers.models.mllama.configuration_mllama import MllamaConfig
    from transformers.models.mllama.configuration_mllama import MllamaTextConfig
    from transformers.models.mllama.configuration_mllama import MllamaVisionConfig
    from transformers.models.mllama.image_processing_mllama import MllamaImageProcessor
    from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration
    from transformers.models.mllama.processing_mllama import MllamaProcessor

    MLLAMA_AVAILABLE = True
except ImportError:
    MLLAMA_AVAILABLE = False

try:
    from transformers import CLIPImageProcessor
    from transformers import CLIPVisionConfig
    from transformers import LlamaConfig
    from transformers.models.llava.configuration_llava import LlavaConfig
    from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
    from transformers.models.llava.processing_llava import LlavaProcessor

    from liger_kernel.transformers import apply_liger_kernel_to_llama

    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False

try:
    import transformers

    from packaging import version
    from transformers.models.gemma.configuration_gemma import GemmaConfig
    from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
    from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
    from transformers.models.paligemma.configuration_paligemma import PaliGemmaConfig
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
    from transformers.models.paligemma.processing_paligemma import PaliGemmaProcessor
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
    from transformers.models.siglip.image_processing_siglip import SiglipImageProcessor

    PALIGEMMA_AVAILABLE = version.parse(transformers.__version__) >= version.parse("4.46.0")
except ImportError:
    PALIGEMMA_AVAILABLE = False


try:
    # Gemma3 is only available in transformers>=4.50.0
    from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
    from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
    from transformers.models.gemma3.image_processing_gemma3 import Gemma3ImageProcessor
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration
    from transformers.models.gemma3.processing_gemma3 import Gemma3Processor

    GEMMA3_AVAILABLE = True
except ImportError:
    GEMMA3_AVAILABLE = False

try:
    from transformers.models.llama4.configuration_llama4 import Llama4Config
    from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
    from transformers.models.llama4.configuration_llama4 import Llama4VisionConfig
    from transformers.models.llama4.image_processing_llama4_fast import Llama4ImageProcessorFast
    from transformers.models.llama4.modeling_llama4 import Llama4ForConditionalGeneration
    from transformers.models.llama4.processing_llama4 import Llama4Processor

    LLAMA4_AVAILABLE = True

except ImportError:
    LLAMA4_AVAILABLE = False

try:
    # InternVL is only available in transformers>=4.52.1
    from transformers.models.got_ocr2.image_processing_got_ocr2_fast import GotOcr2ImageProcessorFast
    from transformers.models.internvl.configuration_internvl import InternVLConfig
    from transformers.models.internvl.modeling_internvl import InternVLForConditionalGeneration
    from transformers.models.internvl.processing_internvl import InternVLProcessor
    from transformers.models.internvl.video_processing_internvl import InternVLVideoProcessor
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

    INTERNVL_AVAILABLE = True
except ImportError:
    INTERNVL_AVAILABLE = False

try:
    # SmolVLM2 is only available in transformers>=4.50.0
    from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
    from transformers.models.smolvlm.configuration_smolvlm import SmolVLMConfig
    from transformers.models.smolvlm.image_processing_smolvlm import SmolVLMImageProcessor
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMForConditionalGeneration
    from transformers.models.smolvlm.processing_smolvlm import SmolVLMProcessor
    from transformers.models.smolvlm.video_processing_smolvlm import SmolVLMVideoProcessor

    SMOLVLM2_AVAILABLE = True
except ImportError:
    SMOLVLM2_AVAILABLE = False

try:
    from num2words import num2words  # noqa: F401

    NUM2WORDS_AVAILABLE = True
except ImportError:
    NUM2WORDS_AVAILABLE = False

from liger_kernel.utils import infer_device

device = infer_device()

torch.use_deterministic_algorithms(True)

#  Only setting torch.use_deterministic_algorithms(True) throws the following error:
#  RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`,
#  but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an
#  environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information,
#  go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

TEST_IMAGE_DIM = 64

MINI_MODEL_SETUPS = {}

if LLAMA4_AVAILABLE:
    MINI_MODEL_SETUPS["mini_llama4"] = MiniModelConfig(
        liger_kernel_patch_func=functools.partial(apply_liger_kernel_to_llama4, fused_linear_cross_entropy=False),
        liger_kernel_patch_revert_func=revert_liger_kernel_to_llama4,
        model_class=Llama4ForConditionalGeneration,
        mini_model_config=Llama4Config(
            image_token_index=8,
            vision_config=Llama4VisionConfig(
                attn_implementation_autoset=True,
                attention_dropout=0.0,
                hidden_act="gelu",
                hidden_size=512,  # 1280
                image_size=560,  # 560
                initializer_range=0.02,
                intermediate_layers_indices=[2],  # [3, 7, 15, etc...]
                intermediate_size=2048,  # 5120
                max_num_tiles=1,  # 4
                norm_eps=1e-5,
                num_attention_heads=4,  # 16
                num_channels=3,
                num_global_layers=2,  # 8
                num_hidden_layers=8,  # 32
                patch_size=280,  # 14
                supported_aspect_ratios=[[1, 1]],  # [[1, 1], [1, 2], etc... ]
                vision_output_dim=4096,  # 7680
            ),
            text_config=Llama4TextConfig(
                bos_token_id=0,
                eos_token_id=0,
                pad_token_id=0,
                cross_attention_layers=[2],  # [3, 8, 13, 18, etc...]
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
                rope_theta=500_000,
                tie_word_embeddings=False,
                use_cache=True,
                vocab_size=32000,  # 128256,
            ),
            attn_implementation="sdpa",
        ),
    )

if MLLAMA_AVAILABLE:
    MINI_MODEL_SETUPS["mini_mllama"] = MiniModelConfig(
        liger_kernel_patch_func=functools.partial(apply_liger_kernel_to_mllama, fused_linear_cross_entropy=False),
        liger_kernel_patch_revert_func=revert_liger_kernel_to_mllama,
        model_class=MllamaForConditionalGeneration,
        mini_model_config=MllamaConfig(
            vision_config=MllamaVisionConfig(
                hidden_act="gelu",
                hidden_size=512,  # 1280
                image_size=560,  # 560
                initializer_range=0.02,
                intermediate_layers_indices=[2],  # [3, 7, 15, etc...]
                intermediate_size=2048,  # 5120
                max_num_tiles=1,  # 4
                norm_eps=1e-5,
                num_attention_heads=4,  # 16
                num_channels=3,
                num_global_layers=2,  # 8
                num_hidden_layers=8,  # 32
                patch_size=140,  # 14
                supported_aspect_ratios=[[1, 1]],  # [[1, 1], [1, 2], etc... ]
                vision_output_dim=1024,  # 7680
            ),
            text_config=MllamaTextConfig(
                bos_token_id=0,
                eos_token_id=0,
                pad_token_id=0,
                cross_attention_layers=[2],  # [3, 8, 13, 18, etc...]
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
            ),
            image_token_index=1,  # NOTE: outside the vocab size
            attn_implementation="sdpa",
        ),
    )

if PALIGEMMA_AVAILABLE:
    MINI_MODEL_SETUPS["mini_paligemma"] = MiniModelConfig(
        liger_kernel_patch_func=functools.partial(apply_liger_kernel_to_paligemma, fused_linear_cross_entropy=False),
        liger_kernel_patch_revert_func=revert_liger_kernel_to_Paligemma,
        model_class=PaliGemmaForConditionalGeneration,
        mini_model_config=PaliGemmaConfig(
            vision_config=SiglipVisionConfig(
                attention_dropout=0.0,
                hidden_act="gelu_pytorch_tanh",
                hidden_size=1152,
                image_size=224,
                intermediate_size=2048,  # 4304
                layer_norm_eps=1e-06,
                num_attention_heads=4,  # 16
                num_channels=3,
                num_hidden_layers=4,  # 27
                num_image_tokens=256,
                num_positions=256,
                patch_size=14,
                projection_dim=1024,  # 2304
            ),
            text_config=GemmaConfig(
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
            image_token_index=4,  # NOTE: outside the vocab size
            attn_implementation="eager",
            vocab_size=32000,
            projection_dim=1024,
        ),
    )
    MINI_MODEL_SETUPS["mini_paligemma2"] = MiniModelConfig(
        liger_kernel_patch_func=functools.partial(apply_liger_kernel_to_paligemma, fused_linear_cross_entropy=False),
        liger_kernel_patch_revert_func=revert_liger_kernel_to_Paligemma,
        model_class=PaliGemmaForConditionalGeneration,
        mini_model_config=PaliGemmaConfig(
            vision_config=SiglipVisionConfig(
                attention_dropout=0.0,
                hidden_act="gelu_pytorch_tanh",
                hidden_size=1152,
                image_size=224,
                intermediate_size=2048,  # 4304
                layer_norm_eps=1e-06,
                num_attention_heads=4,  # 16
                num_channels=3,
                num_hidden_layers=4,  # 27
                num_image_tokens=256,
                num_positions=256,
                patch_size=14,
                projection_dim=1024,  # 2304
            ),
            text_config=Gemma2Config(
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
            image_token_index=4,  # NOTE: outside the vocab size
            attn_implementation="eager",
            vocab_size=32000,
            projection_dim=1024,
        ),
    )

if GEMMA3_AVAILABLE:
    MINI_MODEL_SETUPS["mini_gemma3"] = MiniModelConfig(
        liger_kernel_patch_func=functools.partial(apply_liger_kernel_to_gemma3, fused_linear_cross_entropy=False),
        liger_kernel_patch_revert_func=revert_liger_kernel_to_gemma3,
        model_class=Gemma3ForConditionalGeneration,
        mini_model_config=Gemma3Config(
            vision_config=SiglipVisionConfig(
                attention_dropout=0.0,
                hidden_act="gelu_pytorch_tanh",
                hidden_size=1152,
                image_size=224,
                intermediate_size=2048,  # 4304
                layer_norm_eps=1e-06,
                num_attention_heads=4,  # 16
                num_channels=3,
                num_hidden_layers=4,  # 27
                num_image_tokens=256,
                num_positions=256,
                patch_size=14,
            ).to_dict(),
            text_config=Gemma3TextConfig(
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
                tie_word_embeddings=True,
                rope_theta=10000.0,
                attention_bias=False,
                attention_dropout=0.0,
            ).to_dict(),
            image_token_index=5,  # NOTE: outside the vocab size
            boi_token_index=4,
            eoi_token_index=6,
            attn_implementation="eager",
        ),
    )

if QWEN2_VL_AVAILABLE:
    MINI_MODEL_SETUPS["mini_qwen2_vl"] = MiniModelConfig(
        liger_kernel_patch_func=functools.partial(apply_liger_kernel_to_qwen2_vl, fused_linear_cross_entropy=False),
        liger_kernel_patch_revert_func=revert_liger_kernel_to_qwen2_vl,
        model_class=Qwen2VLForConditionalGeneration,
        mini_model_config=Qwen2VLConfig(
            attention_dropout=0.0,
            # Token Ids and vocab size must match those in the tokenizer/processor
            # test/resources/fake_configs/Qwen/Qwen2-VL-7B-Instruct/tokenizer_config.json
            bos_token_id=0,
            eos_token_id=0,
            vision_start_token_id=1,
            vision_end_token_id=2,
            vision_token_id=3,
            image_token_id=4,
            video_token_id=5,
            hidden_act="silu",
            hidden_size=1024,  # 8192
            initializer_range=0.02,
            intermediate_size=1024,  # 29568
            max_position_embeddings=32768,
            max_window_layers=4,  # 80
            num_attention_heads=8,  # 64
            num_hidden_layers=4,  # 80
            num_key_value_heads=2,  # 8
            rms_norm_eps=1e-6,  # 1e-5
            rope_theta=1000000.0,
            rope_scaling=dict(
                type="mrope",
                mrope_section=[16, 24, 24],  # (temporal, height, width)
            ),
            sliding_window=4096,
            tie_word_embeddings=True,
            use_cache=False,  # True
            vocab_size=32000,  # 152064,
            use_sliding_window=False,
            vision_config={
                "depth": 4,  # 32
                "embed_dim": 128,  # 1280
                "mlp_ratio": 1,
                "num_heads": 8,  # 16
                "in_chans": 3,
                "hidden_size": 1024,  # 1536
            },
            attn_implementation="sdpa",
        ),
    )

if LLAVA_AVAILABLE:
    # https://huggingface.co/llava-hf/llava-1.5-7b-hf
    MINI_MODEL_SETUPS["mini_llava"] = MiniModelConfig(
        liger_kernel_patch_func=functools.partial(apply_liger_kernel_to_llava, fused_linear_cross_entropy=False),
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
            image_token_id=24,
            attn_implementation="sdpa",  # default value, pytorch native attention
        ),
    )

if SMOLVLM2_AVAILABLE:
    MINI_MODEL_SETUPS["mini_smolvlm2"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_smolvlm,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_smolvlm2,
        model_class=SmolVLMForConditionalGeneration,
        mini_model_config=SmolVLMConfig(
            text_config=LlamaConfig(
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=2,
                hidden_act="silu",
                hidden_size=576,  # 576 for 256M model
                initializer_range=0.041666666666666664,
                intermediate_size=1536,  # 1536 for 256M model
                max_position_embeddings=8192,
                num_attention_heads=9,  # 9 for 256M model
                num_hidden_layers=4,  # 30 -> reduced to 4 for testing
                num_key_value_heads=3,  # 3 for 256M model
                rms_norm_eps=1e-5,
                rope_theta=100000,
                tie_word_embeddings=False,
                vocab_size=49280,
            ),
            vision_config={
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 4,  # 12 -> reduced to 4 for testing
                "num_attention_heads": 12,
                "image_size": 512,
                "patch_size": 16,
            },
            image_token_id=49190,
            attn_implementation="sdpa",  # default value, pytorch native attention
        ),
    )

if QWEN2_5_VL_AVAILABLE:
    MINI_MODEL_SETUPS["mini_qwen2_5_vl"] = MiniModelConfig(
        liger_kernel_patch_func=functools.partial(apply_liger_kernel_to_qwen2_5_vl, fused_linear_cross_entropy=False),
        liger_kernel_patch_revert_func=revert_liger_kernel_to_qwen2_5_vl,
        model_class=Qwen2_5_VLForConditionalGeneration,
        mini_model_config=Qwen2_5_VLConfig(
            attention_dropout=0.0,
            # Token Ids and vocab size must match those in the tokenizer/processor
            # test/resources/fake_configs/Qwen/Qwen2-VL-7B-Instruct/tokenizer_config.json
            bos_token_id=0,
            eos_token_id=0,
            vision_start_token_id=1,
            vision_end_token_id=2,
            vision_token_id=3,
            image_token_id=4,
            video_token_id=5,
            hidden_act="silu",
            hidden_size=1024,  # 8192
            initializer_range=0.02,
            intermediate_size=1024,  # 29568
            max_position_embeddings=32768,
            max_window_layers=4,  # 80
            num_attention_heads=8,  # 64
            num_hidden_layers=4,  # 80
            num_key_value_heads=2,  # 8
            rms_norm_eps=1e-6,  # 1e-5
            rope_theta=1000000.0,
            rope_scaling=dict(
                type="mrope",
                mrope_section=[16, 24, 24],  # (temporal, height, width)
            ),
            sliding_window=4096,
            tie_word_embeddings=True,
            use_cache=False,  # True
            vocab_size=32000,  # 152064,
            use_sliding_window=False,
            vision_config={
                "depth": 4,  # 32
                "hidden_size": 128,  # 1280
                "num_heads": 16,
                "in_chans": 3,
                "out_hidden_size": 1024,
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
            attn_implementation="sdpa",
            image_token_id=4,
            video_token_id=5,
            vision_start_token_id=1,
            vision_end_token_id=2,
            tie_word_embeddings=True,
            vision_config=Qwen3VLVisionConfig(
                depth=4,
                hidden_size=256,
                hidden_act="gelu_pytorch_tanh",
                intermediate_size=512,
                num_heads=4,
                in_channels=3,
                patch_size=16,
                spatial_merge_size=2,
                temporal_patch_size=2,
                out_hidden_size=512,
                num_position_embeddings=256,
                deepstack_visual_indexes=[1, 2, 3],
                initializer_range=0.02,
            ).to_dict(),
            text_config=Qwen3VLTextConfig(
                vocab_size=32000,
                hidden_size=512,
                intermediate_size=2048,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=2,
                head_dim=64,
                hidden_act="silu",
                max_position_embeddings=32768,
                initializer_range=0.02,
                rms_norm_eps=1e-6,
                use_cache=False,
                tie_word_embeddings=True,
                rope_theta=1000000.0,
                rope_scaling=dict(
                    type="mrope",
                    mrope_section=[16, 24, 24],
                ),
                attention_dropout=0.0,
                attention_bias=False,
            ).to_dict(),
        ),
    )

if QWEN3_VL_MOE_AVAILABLE:
    MINI_MODEL_SETUPS["mini_qwen3_vl_moe"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_qwen3_vl_moe,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_qwen3_vl_moe,
        model_class=Qwen3VLMoeForConditionalGeneration,
        mini_model_config=Qwen3VLMoeConfig(
            attn_implementation="sdpa",
            image_token_id=4,
            video_token_id=5,
            vision_start_token_id=1,
            vision_end_token_id=2,
            tie_word_embeddings=True,
            vision_config=Qwen3VLMoeVisionConfig(
                depth=4,
                hidden_size=256,
                hidden_act="gelu_pytorch_tanh",
                intermediate_size=512,
                num_heads=4,
                in_channels=3,
                patch_size=16,
                spatial_merge_size=2,
                temporal_patch_size=2,
                out_hidden_size=512,
                num_position_embeddings=256,
                deepstack_visual_indexes=[1, 2, 3],
                initializer_range=0.02,
            ).to_dict(),
            text_config=Qwen3VLMoeTextConfig(
                vocab_size=32000,
                hidden_size=512,
                intermediate_size=2048,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=2,
                head_dim=64,
                hidden_act="silu",
                max_position_embeddings=32768,
                initializer_range=0.02,
                rms_norm_eps=1e-6,
                use_cache=False,
                tie_word_embeddings=True,
                rope_theta=1000000.0,
                rope_scaling=dict(
                    type="mrope",
                    mrope_section=[16, 24, 24],
                ),
                attention_dropout=0.0,
                attention_bias=False,
                decoder_sparse_step=1,
                moe_intermediate_size=1024,
                num_experts_per_tok=2,
                num_experts=4,
                mlp_only_layers=[],
            ).to_dict(),
        ),
    )


def create_processor(model_name: str):
    if model_name == "mini_qwen2_vl":
        tokenizer_config = load_tokenizer_config(
            os.path.join(FAKE_CONFIGS_PATH, "Qwen/Qwen2-VL-7B-Instruct/tokenizer_config.json")
        )
        tokenizer_base = train_bpe_tokenizer(
            [
                token.content
                for key, token in sorted(
                    tokenizer_config["added_tokens_decoder"].items(),
                    key=lambda x: int(x[0]),
                )
            ]
        )
        qwen_tokenizer = Qwen2TokenizerFast(tokenizer_object=tokenizer_base, **tokenizer_config)
        image_processor = Qwen2VLImageProcessor()
        video_processor = Qwen2VLVideoProcessor()
        return Qwen2VLProcessor(
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=qwen_tokenizer,
        )

    elif model_name == "mini_qwen2_5_vl":
        tokenizer_config = load_tokenizer_config(
            os.path.join(FAKE_CONFIGS_PATH, "Qwen/Qwen2.5-VL-7B-Instruct/tokenizer_config.json")
        )
        tokenizer_base = train_bpe_tokenizer(
            [
                token.content
                for key, token in sorted(
                    tokenizer_config["added_tokens_decoder"].items(),
                    key=lambda x: int(x[0]),
                )
            ]
        )
        qwen_tokenizer = Qwen2TokenizerFast(tokenizer_object=tokenizer_base, **tokenizer_config)
        image_processor = Qwen2VLImageProcessor()
        video_processor = Qwen2VLVideoProcessor()
        return Qwen2_5_VLProcessor(
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=qwen_tokenizer,
        )

    elif model_name in ("mini_qwen3_vl", "mini_qwen3_vl_moe"):
        tokenizer_config = load_tokenizer_config(
            os.path.join(FAKE_CONFIGS_PATH, "Qwen/Qwen3-VL-4B-Instruct/tokenizer_config.json")
        )
        tokenizer_base = train_bpe_tokenizer(
            [
                token.content
                for key, token in sorted(
                    tokenizer_config["added_tokens_decoder"].items(),
                    key=lambda x: int(x[0]),
                )
            ]
        )
        qwen_tokenizer = Qwen2TokenizerFast(tokenizer_object=tokenizer_base, **tokenizer_config)
        image_processor = Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2)
        video_processor = Qwen3VLVideoProcessor()
        return Qwen3VLProcessor(
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=qwen_tokenizer,
        )

    elif model_name == "mini_llava":
        tokenizer_config = load_tokenizer_config(
            os.path.join(
                FAKE_CONFIGS_PATH,
                "Llava/llava-1.5-7b-hf/tokenizer_config.json",
            )
        )
        image_processor_config = load_image_processing_config(
            os.path.join(
                FAKE_CONFIGS_PATH,
                "Llava/llava-1.5-7b-hf/preprocessor_config.json",
            )
        )
        processor_config = load_processor_config(
            os.path.join(
                FAKE_CONFIGS_PATH,
                "Llava/llava-1.5-7b-hf/processor_config.json",
            )
        )
        tokenizer_base = train_bpe_tokenizer(
            [
                token.content
                for key, token in sorted(
                    tokenizer_config["added_tokens_decoder"].items(),
                    key=lambda x: int(x[0]),
                )
            ]
        )

        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_base, **tokenizer_config)
        fast_tokenizer.model_input_names = ["input_ids", "attention_mask"]
        image_processor = CLIPImageProcessor(**image_processor_config)

        return LlavaProcessor(**processor_config, image_processor=image_processor, tokenizer=fast_tokenizer)

    elif model_name == "mini_internvl":
        tokenizer_config = load_tokenizer_config(
            os.path.join(FAKE_CONFIGS_PATH, "OpenGVLab/InternVL3-1B-hf/tokenizer_config.json")
        )
        tokenizer_base = train_bpe_tokenizer(
            [
                token.content
                for key, token in sorted(
                    tokenizer_config["added_tokens_decoder"].items(),
                    key=lambda x: int(x[0]),
                )
            ]
        )
        qwen_tokenizer = Qwen2TokenizerFast(tokenizer_object=tokenizer_base, **tokenizer_config)
        image_processor = GotOcr2ImageProcessorFast(
            crop_to_patches=False, min_patches=1, max_patches=12, size={"height": 448, "width": 448}
        )
        video_processor = InternVLVideoProcessor()

        # Return proper InternVL processor
        return InternVLProcessor(
            image_processor=image_processor, tokenizer=qwen_tokenizer, video_processor=video_processor
        )

    elif model_name == "mini_smolvlm2":
        tokenizer_config = load_tokenizer_config(
            os.path.join(FAKE_CONFIGS_PATH, "HuggingFaceTB/SmolVLM2-256M-Video-Instruct/tokenizer_config.json")
        )
        tokenizer_base = train_bpe_tokenizer(
            [
                token.content
                for key, token in sorted(
                    tokenizer_config["added_tokens_decoder"].items(),
                    key=lambda x: int(x[0]),
                )
            ]
        )
        gpt2_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer_base, **tokenizer_config)
        image_processor = SmolVLMImageProcessor(size={"longest_edge": 512})
        video_processor = SmolVLMVideoProcessor()

        # Return proper SmolVLM processor
        return SmolVLMProcessor(
            image_processor=image_processor, tokenizer=gpt2_tokenizer, video_processor=video_processor
        )

    elif model_name.startswith("mini_llama4"):
        tokenizer_config = load_tokenizer_config(
            os.path.join(
                FAKE_CONFIGS_PATH,
                "meta-llama/Llama-4-Scout-17B-16E-Instruct/tokenizer_config.json",
            )
        )
        tokenizer_base = train_bpe_tokenizer(
            [
                token.content
                for key, token in sorted(
                    tokenizer_config["added_tokens_decoder"].items(),
                    key=lambda x: int(x[0]),
                )
            ]
        )
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_base, **tokenizer_config)
        image_processor = Llama4ImageProcessorFast(size={"height": 560, "width": 560})
        return Llama4Processor(
            image_processor=image_processor,
            tokenizer=fast_tokenizer,
            fake_image_token="<|image|>",
            image_token="<|image|>",
        )
    elif model_name == "mini_mllama":
        tokenizer_config = load_tokenizer_config(
            os.path.join(
                FAKE_CONFIGS_PATH,
                "meta-llama/Llama-3.2-11B-Vision-Instruct/tokenizer_config.json",
            )
        )
        tokenizer_base = train_bpe_tokenizer(
            [
                token.content
                for key, token in sorted(
                    tokenizer_config["added_tokens_decoder"].items(),
                    key=lambda x: int(x[0]),
                )
            ]
        )
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_base, **tokenizer_config)
        image_processor = MllamaImageProcessor(size={"height": 560, "width": 560})
        return MllamaProcessor(image_processor=image_processor, tokenizer=fast_tokenizer)

    elif model_name.startswith("mini_paligemma"):
        tokenizer_config = load_tokenizer_config(
            os.path.join(
                FAKE_CONFIGS_PATH,
                "Google/Paligemma/paligemma-3b-pt-224/tokenizer_config.json",
            )
        )
        tokenizer_base = train_bpe_tokenizer(
            [
                token.content
                for key, token in sorted(
                    tokenizer_config["added_tokens_decoder"].items(),
                    key=lambda x: int(x[0]),
                )
            ]
        )

        fast_tokenizer = GemmaTokenizerFast(tokenizer_object=tokenizer_base, **tokenizer_config)
        image_processor = SiglipImageProcessor(size={"height": 224, "width": 224}, image_seq_length=256)
        return PaliGemmaProcessor(image_processor=image_processor, tokenizer=fast_tokenizer)

    elif model_name.startswith("mini_gemma3"):
        tokenizer_config = load_tokenizer_config(
            os.path.join(
                FAKE_CONFIGS_PATH,
                "Google/Gemma3/gemma-3-4b-it/tokenizer_config.json",
            )
        )
        tokenizer_base = train_bpe_tokenizer(
            [
                token.content
                for key, token in sorted(
                    tokenizer_config["added_tokens_decoder"].items(),
                    key=lambda x: int(x[0]),
                )
            ]
        )
        fast_tokenizer = GemmaTokenizerFast(tokenizer_object=tokenizer_base, **tokenizer_config)
        image_processor = Gemma3ImageProcessor()
        return Gemma3Processor(image_processor=image_processor, tokenizer=fast_tokenizer)

    else:
        raise ValueError(f"Processor not available for model {model_name}")


def create_multimodal_dataset(model_name: str):
    processor = create_processor(model_name)

    def generate_procedural_image(example, index):
        """Generate an image with a single row of white pixels at the index specified"""
        image = torch.zeros(3, TEST_IMAGE_DIM, TEST_IMAGE_DIM)
        image[:, index % TEST_IMAGE_DIM, :] = 255
        example["image"] = image
        return example

    def apply_chat_template(example):
        """
        Under the hood, this inserts the correct image placeholder token into the text.
        More or less this conversation format is used by HF's mllms. The fact that it is
        formatting as for IFT is not in-and-of-itself important here.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["text"]}],
            },
        ]
        example["text"] = processor.tokenizer.apply_chat_template(conversation, tokenize=False)
        return example

    def preprocess_function(examples):
        """Tokenize text, preprocess images, and generate other relevant inputs for the model."""
        if model_name == "mini_llama4":
            # Process images and text separately to avoid complex token replacement, this helped setting lower tolerance than processing them together.
            image_inputs = processor.image_processor(images=examples["image"], return_tensors="pt")
            text_inputs = processor.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            return {**text_inputs, **image_inputs}
        else:
            # For other models, use the normal processor
            return processor(
                text=examples["text"],
                images=examples["image"],
                padding="max_length",
                truncation=True,
                max_length=1024,  # longer than for text-only b/c images require quite a few tokens
                return_tensors="pt",
            )

    train_dataset = (
        load_dataset("text", data_files={"train": UNTOKENIZED_DATASET_PATH}, split="train")
        .to_iterable_dataset()  # only map examples as-needed and on-demand
        .map(generate_procedural_image, with_indices=True)
        .map(apply_chat_template)
        .map(preprocess_function, remove_columns=["text", "image"])
    )
    return train_dataset


def create_model(model_name):
    """
    Create a mini version model
    The commented values are the original values
    """
    model_config = MINI_MODEL_SETUPS[model_name].mini_model_config
    model_class = MINI_MODEL_SETUPS[model_name].model_class
    return model_class(model_config)


@require_deterministic
def run_mini_model_multimodal(
    model_name="mini_qwen2_vl",
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
    if "mllama" in model_name or "llama4" in model_name:
        revert_kwargs["model_type"] = "conditional_generation"

    if with_liger is True:
        kwargs = {
            "rope": True,
            "rms_norm": True,
            "cross_entropy": False,
        }

        if "qwen2_5_vl" not in model_name and "llava" not in model_name and "qwen3_vl" not in model_name:
            kwargs["layer_norm"] = True

        if "gemma" in model_name:
            kwargs["geglu"] = True
        else:
            kwargs["swiglu"] = True

        if "llava" in model_name:
            apply_liger_kernel_to_llama(**kwargs)

        MINI_MODEL_SETUPS[model_name].liger_kernel_patch_func(**kwargs)
    else:
        MINI_MODEL_SETUPS[model_name].liger_kernel_patch_revert_func(**revert_kwargs)

    model = create_model(model_name).to(dtype).to(device)
    model.gradient_checkpointing_enable()

    train_dataset = create_multimodal_dataset(model_name)
    loader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=multimodal_collate_fn)
    loader_iter = iter(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loss_list = []

    for i in range(num_steps):
        batch = next(loader_iter).to(model.device)
        optimizer.zero_grad()
        supports_accum = getattr(model, "_supports_accum_dtype", None)
        if supports_accum is None:
            import inspect

            params = inspect.signature(model.forward).parameters
            supports_accum = ("accum_dtype" in params) or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            setattr(model, "_supports_accum_dtype", supports_accum)

        output = model(**batch, accum_dtype=torch.float32) if supports_accum else model(**batch)
        output.loss.backward()
        optimizer.step()

        print(f"Step {i}, Loss: {output.loss.item()}")
        loss_list.append(output.loss.item())

    model.eval()
    eval_batch = next(loader_iter).to(model.device)
    if with_liger:
        eval_batch["skip_logits"] = False
    with torch.no_grad():
        eval_output = model(**eval_batch)
    print(f"Eval Loss: {eval_output.loss.item()}")
    loss_list.append(eval_output.loss.item())
    topk_logprobs = get_topk(get_logprobs(eval_output.logits))
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
            "mini_qwen2_vl",
            32,
            1e-4,
            torch.bfloat16,
            5e-2,
            5e-2,
            1e-1,
            1e-2,
            1e-2,
            1e-2,
            marks=[
                pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
                pytest.mark.skipif(
                    not QWEN2_VL_AVAILABLE,
                    reason="Qwen2-VL not available in this version of transformers",
                ),
                pytest.mark.skipif(not is_torchvision_available(), reason="Qwen2VLVideoProcessor requires torchvision"),
            ],
        ),
        pytest.param(
            "mini_llava",
            32,
            1e-5,
            torch.bfloat16,
            5e-2,
            5e-2,
            1e-1,
            1e-2,
            1e-2,
            1e-2,
            marks=[
                pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
                pytest.mark.skipif(
                    not LLAVA_AVAILABLE,
                    reason="LLaVa not available in this version of transformers",
                ),
            ],
        ),
        pytest.param(
            "mini_internvl",
            32,
            1e-5,
            torch.bfloat16,
            5e-2,
            5e-2,
            1e-1,
            1e-2,
            1e-2,
            1e-2,
            marks=[
                pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
                pytest.mark.skipif(
                    not INTERNVL_AVAILABLE,
                    reason="InternVL not available in this version of transformers",
                ),
            ],
        ),
        pytest.param(
            "mini_smolvlm2",
            32,
            1e-5,
            torch.bfloat16,
            5e-2,
            5e-2,
            1e-1,
            1e-2,
            1e-2,
            1e-2,
            marks=[
                pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
                pytest.mark.skipif(
                    not SMOLVLM2_AVAILABLE,
                    reason="SmolVLM2 not available in this version of transformers",
                ),
                pytest.mark.skipif(
                    not NUM2WORDS_AVAILABLE,
                    reason="num2words must be present to run SmolVLMProcessor",
                ),
            ],
        ),
        pytest.param(
            "mini_qwen2_5_vl",
            32,
            1e-5,
            torch.bfloat16,
            5e-2,
            5e-2,
            1e-1,
            1e-2,
            1e-2,
            1e-2,
            marks=[
                pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
                pytest.mark.skipif(
                    not QWEN2_5_VL_AVAILABLE,
                    reason="Qwen2.5-VL not available in this version of transformers",
                ),
                pytest.mark.skipif(not is_torchvision_available(), reason="Qwen2VLVideoProcessor requires torchvision"),
            ],
        ),
        pytest.param(
            "mini_qwen3_vl",
            32,
            1e-5,
            torch.bfloat16,
            5e-2,
            5e-2,
            1e-1,
            1e-2,
            1e-2,
            1e-2,
            marks=[
                pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
                pytest.mark.skipif(
                    not QWEN3_VL_AVAILABLE,
                    reason="Qwen3-VL not available in this version of transformers",
                ),
                pytest.mark.skipif(
                    not is_torchvision_available(),
                    reason="Qwen3VLVideoProcessor requires torchvision",
                ),
            ],
        ),
        pytest.param(
            "mini_qwen3_vl_moe",
            32,
            1e-5,
            torch.bfloat16,
            5e-2,
            5e-2,
            1e-1,
            1e-2,
            1e-2,
            1e-2,
            marks=[
                pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
                pytest.mark.skipif(
                    not QWEN3_VL_MOE_AVAILABLE,
                    reason="Qwen3-VL-MoE not available in this version of transformers",
                ),
                pytest.mark.skipif(
                    not is_torchvision_available(),
                    reason="Qwen3VLVideoProcessor requires torchvision",
                ),
            ],
        ),
        pytest.param(
            "mini_mllama",
            32,
            1e-5,
            torch.bfloat16,
            5e-2,
            5e-2,
            1e-1,
            1e-2,
            1e-2,
            1e-2,
            marks=[
                pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
                pytest.mark.skipif(
                    not MLLAMA_AVAILABLE,
                    reason="Mllama not available in this version of transformers",
                ),
                pytest.mark.skipif(
                    version.parse("4.51.0") > version.parse(transformers.__version__),
                    reason="MllamaForConditionalGeneration doesn't accecpt `skip_logits` kwargs",
                ),
            ],
        ),
        pytest.param(
            "mini_llama4",
            32,
            1e-5,
            torch.bfloat16,
            5e-2,
            5e-2,
            1e-1,
            1e-1,
            1e-2,
            1e-2,
            marks=[
                pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
                pytest.mark.skipif(
                    not LLAMA4_AVAILABLE,
                    reason="Llama4 not available in this version of transformers",
                ),
            ],
        ),
        pytest.param(
            "mini_paligemma",
            32,
            1e-5,
            torch.bfloat16,
            5e-2,
            5e-2,
            1e-1,
            1e-2,
            1e-2,
            1e-2,
            marks=[
                pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
                pytest.mark.skipif(
                    not PALIGEMMA_AVAILABLE,
                    reason="Paligemma not available in this version of transformers",
                ),
            ],
        ),
        pytest.param(
            "mini_paligemma2",
            32,
            1e-5,
            torch.bfloat16,
            5e-2,
            5e-2,
            1e-1,
            1e-2,
            1e-2,
            1e-2,
            marks=[
                pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
                pytest.mark.skipif(
                    not PALIGEMMA_AVAILABLE,
                    reason="Paligemma2 not available in this version of transformers",
                ),
            ],
        ),
        pytest.param(
            "mini_gemma3",
            32,
            1e-5,
            torch.bfloat16,
            5e-2,
            5e-2,
            1e-1,
            1e-1,
            1e-2,
            1e-2,
            marks=[
                pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
                pytest.mark.skipif(
                    not GEMMA3_AVAILABLE,
                    reason="Gemma3 not available in this version of transformers",
                ),
            ],
        ),
    ],
)
def test_mini_model_multimodal(
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
    expected_output = run_mini_model_multimodal(model_name=model_name, num_steps=num_steps, dtype=dtype, lr=lr)

    actual_output = run_mini_model_multimodal(
        model_name=model_name, num_steps=num_steps, dtype=dtype, lr=lr, with_liger=True
    )

    # Compare the loss of every step
    assert_verbose_allclose(
        torch.tensor([expected_output["loss"]]),
        torch.tensor([actual_output["loss"]]),
        atol=loss_atol,
        rtol=loss_rtol,
        extra_info="[Loss]",
    )

    # Compare the topk logprobs from evaluation step
    assert_verbose_allclose(
        expected_output["topk_logprobs"],
        actual_output["topk_logprobs"],
        atol=logprobs_atol,
        rtol=logprobs_rtol,
        extra_info="[Top k logprobs]",
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
