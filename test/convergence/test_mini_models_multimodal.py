import functools
import os

import pytest
import torch

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from liger_kernel.transformers import apply_liger_kernel_to_mllama
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
from test.utils import FAKE_CONFIGS_PATH
from test.utils import UNTOKENIZED_DATASET_PATH
from test.utils import MiniModelConfig
from test.utils import assert_verbose_allclose
from test.utils import load_tokenizer_config
from test.utils import multimodal_collate_fn
from test.utils import revert_liger_kernel_to_mllama
from test.utils import revert_liger_kernel_to_qwen2_vl
from test.utils import set_seed
from test.utils import supports_bfloat16
from test.utils import train_bpe_tokenizer

try:
    # Qwen2-VL is only available in transformers>=4.45.0
    from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

    QWEN2_VL_AVAILABLE = True
except ImportError:
    QWEN2_VL_AVAILABLE = False

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


def create_processor(model_name):
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
        return Qwen2VLProcessor(image_processor=image_processor, tokenizer=qwen_tokenizer)

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
    if "mllama" in model_name:
        revert_kwargs["model_type"] = "conditional_generation"

    if with_liger is True:
        kwargs = {
            "rope": True,
            "rms_norm": True,
            "cross_entropy": True,
            "layer_norm": True,
        }

        if "gemma" in model_name:
            kwargs["geglu"] = True
        else:
            kwargs["swiglu"] = True
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
        output = model(**batch)
        output.loss.backward()
        optimizer.step()

        print(f"Step {i}, Loss: {output.loss.item()}")
        loss_list.append(output.loss.item())

    MINI_MODEL_SETUPS[model_name].liger_kernel_patch_revert_func(**revert_kwargs)
    return {"loss": loss_list, "logits": output.logits, "model": model}


@pytest.mark.parametrize(
    "model_name, num_steps, lr, dtype, loss_atol, loss_rtol, logits_atol, logits_rtol, param_atol, param_rtol",
    [
        pytest.param(
            "mini_qwen2_vl",
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
                not QWEN2_VL_AVAILABLE,
                reason="Qwen2-VL not available in this version of transformers",
            ),
        ),
        pytest.param(
            "mini_qwen2_vl",
            32,
            1e-4,
            torch.bfloat16,
            1e-3,
            1e-2,
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
            ],
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
            "mini_mllama",
            32,
            1e-4,
            torch.bfloat16,
            1e-3,
            1e-2,
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
    logits_atol,
    logits_rtol,
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
    )

    # Compare the logits from the last step
    assert_verbose_allclose(
        expected_output["logits"],
        actual_output["logits"],
        atol=logits_atol,
        rtol=logits_rtol,
    )

    # Compare the params from the last step
    # Iterate over the model's parameters and compare them
    for expected_param, actual_param in zip(
        expected_output["model"].named_parameters(),
        actual_output["model"].named_parameters(),
        strict=False,
    ):
        assert_verbose_allclose(expected_param[1], actual_param[1], atol=param_atol, rtol=param_rtol)
