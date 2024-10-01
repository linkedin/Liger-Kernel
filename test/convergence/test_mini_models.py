import functools
import os
from test.utils import (
    DEFAULT_DATASET_PATH,
    MiniModelConfig,
    assert_verbose_allclose,
    revert_liger_kernel_to_gemma,
    revert_liger_kernel_to_gemma2,
    revert_liger_kernel_to_llama,
    revert_liger_kernel_to_mistral,
    revert_liger_kernel_to_mixtral,
    revert_liger_kernel_to_phi3,
    revert_liger_kernel_to_qwen2,
    set_seed,
    simple_collate_fn,
    supports_bfloat16,
)

import pytest
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers.models.gemma import GemmaConfig, GemmaForCausalLM
from transformers.models.gemma2 import Gemma2Config, Gemma2ForCausalLM
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.mistral import MistralConfig, MistralForCausalLM
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM
from transformers.models.phi3 import Phi3Config, Phi3ForCausalLM
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

from liger_kernel.transformers import (
    apply_liger_kernel_to_gemma,
    apply_liger_kernel_to_gemma2,
    apply_liger_kernel_to_llama,
    apply_liger_kernel_to_mistral,
    apply_liger_kernel_to_mixtral,
    apply_liger_kernel_to_phi3,
    apply_liger_kernel_to_qwen2,
)

torch.use_deterministic_algorithms(True)

#  Only setting torch.use_deterministic_algorithms(True) throws the following error:
#  RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`,
#  but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an
#  environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information,
#  go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

MINI_MODEL_SETUPS = {
    "mini_llama3": MiniModelConfig(
        liger_kernel_patch_func=functools.partial(
            apply_liger_kernel_to_llama, fused_linear_cross_entropy=False
        ),
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
            vocab_size=32000,  # 128256
            # At rope backward
            # Eager produces incontiguous dq and dk
            # SDPA produces contiguous dq and incontiguous dk
            # Flash_attn produces contiguous dq and dk
            attn_implementation="sdpa",  # default value, pytorch native attention
        ),
    ),
    "mini_gemma1": MiniModelConfig(
        liger_kernel_patch_func=functools.partial(
            apply_liger_kernel_to_gemma, fused_linear_cross_entropy=False
        ),
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
        liger_kernel_patch_func=functools.partial(
            apply_liger_kernel_to_gemma, fused_linear_cross_entropy=False
        ),
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
    "mini_mistral": MiniModelConfig(
        liger_kernel_patch_func=functools.partial(
            apply_liger_kernel_to_mistral, fused_linear_cross_entropy=False
        ),
        liger_kernel_patch_revert_func=revert_liger_kernel_to_mistral,
        model_class=MistralForCausalLM,
        mini_model_config=MistralConfig(
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=1024,  # 4096
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
    "mini_mixtral": MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_mixtral,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_mixtral,
        model_class=MixtralForCausalLM,
        mini_model_config=MixtralConfig(
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=1024,  # 4096
            initializer_range=0.02,
            intermediate_size=2048,  # 14336
            max_position_embeddings=32768,  # 32768
            num_attention_heads=8,  # 32
            num_experts_per_tok=2,
            num_hidden_layers=4,  # 32
            num_key_value_heads=2,  # 8
            num_local_experts=8,
            output_router_logits=False,
            rms_norm_eps=1e-5,
            rope_theta=1000000.0,
            router_aux_loss_coef=0.02,
            sliding_window=None,
            tie_word_embeddings=False,
            use_cache=True,
            vocab_size=32000,
            # At rope backward
            # Eager produces incontiguous dq and dk
            # SDPA produces contiguous dq and incontiguous dk
            # Flash_attn produces contiguous dq and dk
            attn_implementation="sdpa",  # default value, pytorch native attention
        ),
    ),
    "mini_qwen2": MiniModelConfig(
        liger_kernel_patch_func=functools.partial(
            apply_liger_kernel_to_qwen2, fused_linear_cross_entropy=False
        ),
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
        liger_kernel_patch_func=functools.partial(
            apply_liger_kernel_to_phi3, fused_linear_cross_entropy=False
        ),
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
}


def create_model(model_name="mini_llama3"):
    """
    Create a mini version model
    The commented values are the original values
    """
    model_config = MINI_MODEL_SETUPS[model_name].mini_model_config
    model_class = MINI_MODEL_SETUPS[model_name].model_class
    return model_class(model_config)


def run_mini_model(
    model_name="mini_llama3",
    num_steps=100,
    dtype=torch.bfloat16,
    lr=1e-5,
    with_liger=False,
    post_init_patching=False,
):
    # If we move it to the beginning of test_mini_model, the two runs are initialized with different weights.
    # This is due to RNG (Random Number Generator). The formula of RNG progression is x_(n+1) = (a * x_n + c) % m
    # Everytime RNG is used, like randomly initializing weight, the RNG progresses to the next state.
    # Therefore, we have to reset RNG before we create the model to ensure the weight initialization started from the same RNG state.

    set_seed(42)

    # Make sure any patches have been reverted before tests
    MINI_MODEL_SETUPS[model_name].liger_kernel_patch_revert_func()

    if with_liger is True:
        kwargs = {
            "rope": True,
            "rms_norm": True,
            "cross_entropy": True,
        }
        if "gemma" in model_name:
            kwargs["geglu"] = True
        else:
            kwargs["swiglu"] = True

        if post_init_patching:
            model = create_model(model_name).to(dtype).to("cuda")
            kwargs["model"] = model
            MINI_MODEL_SETUPS[model_name].liger_kernel_patch_func(**kwargs)
        else:
            MINI_MODEL_SETUPS[model_name].liger_kernel_patch_func(**kwargs)
            model = create_model(model_name).to(dtype).to("cuda")
    else:
        model = create_model(model_name).to(dtype).to("cuda")

    train_dataset = load_from_disk(DEFAULT_DATASET_PATH)

    loader = DataLoader(
        train_dataset, batch_size=16, shuffle=False, collate_fn=simple_collate_fn
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


@pytest.mark.parametrize(
    "model_name, num_steps, lr, dtype, loss_atol, loss_rtol, logits_atol, logits_rtol, param_atol, param_rtol",
    [
        # Gemma 1 has more tolerance because currently, the kernel is not a perfect match (casts are not done the same way)
        ("mini_gemma1", 32, 1e-4, torch.float32, 1e-8, 6e-4, 5e-3, 1e-5, 5e-3, 1e-5),
        pytest.param(
            "mini_gemma1",
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
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        ("mini_gemma1.1", 32, 1e-4, torch.float32, 1e-8, 1e-5, 5e-3, 1e-5, 5e-3, 1e-5),
        pytest.param(
            "mini_gemma1.1",
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
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        # TODO: Gemma2 tests are not passing within the tolerance range, need to investigate
        # ("mini_gemma2", 32, 1e-4, torch.float32, 1e-8, 1e-5, 5e-3, 1e-5, 5e-3, 1e-5),
        # pytest.param(
        #     "mini_gemma2",
        #     32,
        #     1e-4,
        #     torch.bfloat16,
        #     1e-3,
        #     1e-2,
        #     1e-1,
        #     1e-2,
        #     1e-2,
        #     1e-2,
        #     marks=pytest.mark.skipif(
        #         not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
        #     ),
        # ),
        ("mini_llama3", 32, 1e-4, torch.float32, 1e-8, 1e-5, 5e-3, 1e-5, 5e-3, 1e-5),
        pytest.param(
            "mini_llama3",
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
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        # TODO: torch 2.5.0 nightly breaks mixtral test, but torch 2.3.0 works fine
        # TODO: mixtral MoE structure makes the convergence flaky so disable the test for now. It needs high tol to pass.
        # ("mini_mixtral", 32, 1e-4, torch.float32, 1e-8, 1e-4, 5e-3, 1e-5, 8e-3, 1e-5),
        # ("mini_mixtral", 32, 1e-4, torch.bfloat16, 1e-8, 1e-5, 2.0, 1e-5, 1e-2, 1e-5),
        ("mini_mistral", 32, 1e-4, torch.float32, 1e-8, 1e-5, 5e-3, 1e-5, 5e-3, 1e-5),
        pytest.param(
            "mini_mistral",
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
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        ("mini_qwen2", 32, 1e-4, torch.float32, 1e-8, 1e-5, 5e-3, 1e-5, 5e-3, 1e-5),
        pytest.param(
            "mini_qwen2",
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
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        ("mini_phi3", 32, 1e-4, torch.float32, 1e-8, 1e-5, 5e-3, 1e-5, 5e-3, 1e-5),
        pytest.param(
            "mini_phi3",
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
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
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
    logits_atol,
    logits_rtol,
    param_atol,
    param_rtol,
):
    # Non-liger models should be initialized and tested first to avoid the module being overridden

    expected_output = run_mini_model(
        model_name=model_name, num_steps=num_steps, dtype=dtype, lr=lr, with_liger=False
    )

    actual_output_pre = run_mini_model(
        model_name=model_name,
        num_steps=num_steps,
        dtype=dtype,
        lr=lr,
        with_liger=True,
        post_init_patching=False,
    )

    actual_output_post = run_mini_model(
        model_name=model_name,
        num_steps=num_steps,
        dtype=dtype,
        lr=lr,
        with_liger=True,
        post_init_patching=True,
    )

    # Pre-init patching

    # Compare the loss of every step
    assert_verbose_allclose(
        torch.tensor([expected_output["loss"]]),
        torch.tensor([actual_output_pre["loss"]]),
        atol=loss_atol,
        rtol=loss_rtol,
    )

    # Compare the logits from the last step
    assert_verbose_allclose(
        expected_output["logits"],
        actual_output_pre["logits"],
        atol=logits_atol,
        rtol=logits_rtol,
    )

    # Compare the params from the last step
    # Iterate over the model's parameters and compare them
    for expected_param, actual_param in zip(
        expected_output["model"].named_parameters(),
        actual_output_pre["model"].named_parameters(),
    ):
        assert_verbose_allclose(
            expected_param[1], actual_param[1], atol=param_atol, rtol=param_rtol
        )

    # Post-init patching

    # Compare the loss of every step
    assert_verbose_allclose(
        torch.tensor([expected_output["loss"]]),
        torch.tensor([actual_output_post["loss"]]),
        atol=loss_atol,
        rtol=loss_rtol,
    )

    # Compare the logits from the last step
    assert_verbose_allclose(
        expected_output["logits"],
        actual_output_post["logits"],
        atol=logits_atol,
        rtol=logits_rtol,
    )

    # Compare the params from the last step
    # Iterate over the model's parameters and compare them
    for expected_param, actual_param in zip(
        expected_output["model"].named_parameters(),
        actual_output_post["model"].named_parameters(),
    ):
        assert_verbose_allclose(
            expected_param[1], actual_param[1], atol=param_atol, rtol=param_rtol
        )
