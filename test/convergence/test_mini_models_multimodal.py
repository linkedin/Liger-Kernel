import functools
import os
from test.utils import (
    UNTOKENIZED_DATASET_PATH,
    MiniModelConfig,
    assert_verbose_allclose,
    multimodal_collate_fn,
    revert_liger_kernel_to_qwen2_vl,
    set_seed,
    supports_bfloat16,
)

import pytest
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers.models.auto.processing_auto import AutoProcessor

from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl

try:
    # Qwen2-VL is only available in transformers>4.44.2
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VLForConditionalGeneration,
    )

    QWEN2_VL_AVAILABLE = True
except ImportError:
    QWEN2_VL_AVAILABLE = False

torch.use_deterministic_algorithms(True)

#  Only setting torch.use_deterministic_algorithms(True) throws the following error:
#  RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`,
#  but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an
#  environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information,
#  go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

TEST_IMAGE_DIM = 64

MINI_MODEL_SETUPS = {}

if QWEN2_VL_AVAILABLE:
    MINI_MODEL_SETUPS["mini_qwen2_vl"] = MiniModelConfig(
        liger_kernel_patch_func=functools.partial(
            apply_liger_kernel_to_qwen2_vl, fused_linear_cross_entropy=False
        ),
        liger_kernel_patch_revert_func=revert_liger_kernel_to_qwen2_vl,
        model_class=Qwen2VLForConditionalGeneration,
        mini_model_config=Qwen2VLConfig(
            attention_dropout=0.0,
            # Token Ids and vocab size must match those in the tokenizer/processor
            # https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/blob/main/config.json
            bos_token_id=151643,
            eos_token_id=151645,
            vision_start_token_id=151652,
            vision_end_token_id=151653,
            vision_token_id=151654,
            image_token_id=151655,
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
            vocab_size=152064,
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
        return AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
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
        example["text"] = processor.apply_chat_template(conversation, tokenize=False)
        return example

    def preprocess_function(examples):
        """Tokenize text, preprocess images, and generate other relevant inputs for the model."""
        return processor(
            text=examples["text"],
            images=examples["image"],
            padding="max_length",
            truncation=True,
            max_length=1024,  # longer than for text-only b/c images require quite a few tokens
        )

    train_dataset = (
        load_dataset(
            "text", data_files={"train": UNTOKENIZED_DATASET_PATH}, split="train"
        )
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

    if with_liger is True:
        kwargs = {
            "rms_norm": True,
            "cross_entropy": True,
        }
        model_supports_rope = "qwen2_vl" not in model_name
        if model_supports_rope:
            kwargs["rope"] = True

        model_supports_layer_norm = "qwen2_vl" in model_name
        if model_supports_layer_norm:
            kwargs["layer_norm"] = True

        if "gemma" in model_name:
            kwargs["geglu"] = True
        else:
            kwargs["swiglu"] = True
        MINI_MODEL_SETUPS[model_name].liger_kernel_patch_func(**kwargs)
    else:
        MINI_MODEL_SETUPS[model_name].liger_kernel_patch_revert_func()

    model = create_model(model_name).to(dtype).to("cuda")
    model.gradient_checkpointing_enable()

    train_dataset = create_multimodal_dataset(model_name)
    loader = DataLoader(
        train_dataset, batch_size=2, shuffle=False, collate_fn=multimodal_collate_fn
    )
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

    MINI_MODEL_SETUPS[model_name].liger_kernel_patch_revert_func()
    return {"loss": loss_list, "logits": output.logits, "model": model}


@pytest.mark.skip(
    reason="This test needs to be fixed and work without access to HF Hub"
)
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
                pytest.mark.skipif(
                    not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
                ),
                pytest.mark.skipif(
                    not QWEN2_VL_AVAILABLE,
                    reason="Qwen2-VL not available in this version of transformers",
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
    expected_output = run_mini_model_multimodal(
        model_name=model_name, num_steps=num_steps, dtype=dtype, lr=lr
    )

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
    ):
        assert_verbose_allclose(
            expected_param[1], actual_param[1], atol=param_atol, rtol=param_rtol
        )
