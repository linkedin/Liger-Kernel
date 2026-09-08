import gc
import importlib

import torch
import torch.nn as nn

from liger_kernel.transformers import apply_liger_kernel_to_lfm2
from liger_kernel.transformers import apply_liger_kernel_to_lfm2_moe
from liger_kernel.transformers import apply_liger_kernel_to_lfm2_vl
from liger_kernel.utils import infer_device

device = infer_device()
_ORIGINAL_LAYER_NORM = nn.LayerNorm


def _text_config():
    from transformers.models.lfm2.configuration_lfm2 import Lfm2Config

    return Lfm2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        block_multiple_of=8,
        block_auto_adjust_ff_dim=False,
        layer_types=["conv", "full_attention", "conv"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        conv_bias=True,
        use_cache=False,
        attn_implementation="sdpa",
    )


def _moe_config():
    from transformers.models.lfm2_moe.configuration_lfm2_moe import Lfm2MoeConfig

    return Lfm2MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        num_dense_layers=1,
        num_experts=8,
        num_experts_per_tok=2,
        use_expert_bias=True,
        norm_topk_prob=True,
        routed_scaling_factor=1.3,
        layer_types=["conv", "full_attention", "conv"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        conv_bias=True,
        use_cache=False,
        attn_implementation="sdpa",
    )


def _vl_config():
    from transformers.models.lfm2_vl.configuration_lfm2_vl import Lfm2VlConfig

    return Lfm2VlConfig(
        text_config=_text_config().to_dict(),
        vision_config={
            "model_type": "siglip2_vision_model",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_channels": 3,
            "num_patches": 16,
            "patch_size": 2,
            "hidden_act": "gelu_pytorch_tanh",
            "vision_use_head": False,
            "attn_implementation": "sdpa",
        },
        image_token_id=127,
        projector_hidden_size=64,
        downsample_factor=2,
    )


def _reset_modules(model_kind):
    if model_kind == "lfm2":
        from transformers.models.lfm2 import modeling_lfm2

        importlib.reload(modeling_lfm2)
        return modeling_lfm2.Lfm2ForCausalLM
    if model_kind == "lfm2_moe":
        from transformers.models.lfm2_moe import modeling_lfm2_moe

        importlib.reload(modeling_lfm2_moe)
        return modeling_lfm2_moe.Lfm2MoeForCausalLM
    if model_kind == "lfm2_vl":
        from transformers.models.lfm2 import modeling_lfm2
        from transformers.models.lfm2_vl import modeling_lfm2_vl
        from transformers.models.siglip2 import modeling_siglip2

        nn.LayerNorm = _ORIGINAL_LAYER_NORM
        importlib.reload(modeling_lfm2)
        importlib.reload(modeling_siglip2)
        importlib.reload(modeling_lfm2_vl)
        return modeling_lfm2_vl.Lfm2VlForConditionalGeneration
    raise ValueError(f"unknown model kind: {model_kind}")


def _config(model_kind):
    if model_kind == "lfm2":
        return _text_config()
    if model_kind == "lfm2_moe":
        return _moe_config()
    return _vl_config()


def _batch(model_kind, dtype):
    input_ids = (torch.arange(32, device=device).reshape(2, 16) % 120) + 3
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    if model_kind == "lfm2_vl":
        input_ids = input_ids[:1]
        input_ids[:, :4] = 127
        labels = input_ids.clone()
        labels[:, :4] = -100
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
            "pixel_values": torch.linspace(-1, 1, 16 * 12, device=device, dtype=dtype).reshape(1, 16, 12),
            "spatial_shapes": torch.tensor([[4, 4]], device=device),
            "pixel_attention_mask": torch.ones(1, 16, dtype=torch.bool, device=device),
        }
    return batch


def run_lfm2_convergence(model_kind, dtype, with_liger, num_steps=2):
    model_class = _reset_modules(model_kind)
    if with_liger:
        {
            "lfm2": apply_liger_kernel_to_lfm2,
            "lfm2_moe": apply_liger_kernel_to_lfm2_moe,
            "lfm2_vl": apply_liger_kernel_to_lfm2_vl,
        }[model_kind]()

    torch.manual_seed(42)
    model = model_class(_config(model_kind)).to(device=device, dtype=dtype)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = _batch(model_kind, dtype)
    losses = []
    for _ in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().float().cpu())

    parameters = {name: parameter.detach().float().cpu() for name, parameter in model.named_parameters()}
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    _reset_modules(model_kind)
    return torch.stack(losses), parameters
