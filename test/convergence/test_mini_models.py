import functools
from test.utils import (
    DEFAULT_DATASET_PATH,
    MiniModelConfig,
    assert_verbose_allclose,
    set_seed,
    simple_collate_fn,
)

import pytest
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.mistral import MistralConfig, MistralForCausalLM
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM

from liger_kernel.transformers import (
    apply_liger_kernel_to_llama,
    apply_liger_kernel_to_mistral,
    apply_liger_kernel_to_mixtral,
)

MINI_MODEL_SETUPS = {
    "mini_llama3": MiniModelConfig(
        liger_kernel_patch_func=functools.partial(
            apply_liger_kernel_to_llama, fused_linear_cross_entropy=False
        ),
        model_class=LlamaForCausalLM,
        mini_model_config=LlamaConfig(
            attention_bias=False,
            attention_dropout=0.0,
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
    "mini_mistral": MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_mistral,
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
):
    # If we move it to the beginning of test_mini_model, the two runs are initialized with different weights.
    # This is due to RNG (Random Number Generator). The formula of RNG progression is x_(n+1) = (a * x_n + c) % m
    # Everytime RNG is used, like randomly initialzing weight, the RNG progresses to the next state.
    # Therefore, we have to reset RNG before we create the model to ensure the weight initialization started from the same RNG state.

    set_seed(42)

    if with_liger is True:
        MINI_MODEL_SETUPS[model_name].liger_kernel_patch_func(
            rope=True, rms_norm=True, cross_entropy=True, swiglu=True
        )

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
        output = model(**batch)
        output.loss.backward()
        optimizer.step()
        print(f"Step {i}, Loss: {output.loss.item()}")
        loss_list.append(output.loss.item())

    return {"loss": loss_list, "logits": output.logits, "model": model}


@pytest.mark.parametrize(
    "model_name, num_steps, lr, dtype, loss_atol, loss_rtol, logits_atol, logits_rtol, param_atol, param_rtol",
    [
        ("mini_llama3", 32, 1e-4, torch.float32, 1e-8, 1e-5, 1e-4, 1e-5, 2e-3, 1e-5),
        ("mini_llama3", 32, 1e-4, torch.bfloat16, 1e-8, 1e-5, 1e-1, 1e-5, 1e-2, 1e-5),
        # TODO: torch 2.5.0 nightly breaks mixtral test, but torch 2.3.0 works fine
        ("mini_mixtral", 32, 1e-4, torch.float32, 1e-8, 1e-4, 1e-3, 3e-2, 8e-3, 1e-5),
        ("mini_mixtral", 32, 1e-4, torch.bfloat16, 1e-8, 1e-5, 2.0, 1e-5, 1e-2, 1e-5),
        ("mini_mistral", 32, 1e-4, torch.float32, 1e-8, 1e-5, 5e-3, 1e-5, 5e-3, 1e-5),
        ("mini_mistral", 32, 1e-4, torch.bfloat16, 1e-8, 1e-5, 1e-1, 1e-5, 1e-2, 1e-5),
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
        model_name=model_name, num_steps=num_steps, dtype=dtype, lr=lr
    )

    actual_output = run_mini_model(
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
