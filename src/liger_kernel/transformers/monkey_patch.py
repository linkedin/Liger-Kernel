import inspect
import logging
from functools import partial

from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.transformers.layer_norm import LigerLayerNorm
from liger_kernel.transformers.model.gemma import lce_forward as gemma_lce_forward
from liger_kernel.transformers.model.llama import lce_forward as llama_lce_forward
from liger_kernel.transformers.model.mistral import lce_forward as mistral_lce_forward
from liger_kernel.transformers.model.mixtral import lce_forward as mixtral_lce_forward
from liger_kernel.transformers.model.phi3 import lce_forward as phi3_lce_forward
from liger_kernel.transformers.model.qwen2 import lce_forward as qwen2_lce_forward
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from liger_kernel.transformers.swiglu import (
    LigerBlockSparseTop2MLP,
    LigerPhi3SwiGLUMLP,
    LigerSwiGLUMLP,
)

logger = logging.getLogger(__name__)


def apply_liger_kernel_to_llama(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Llama models (2 and 3)

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """

    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.llama import modeling_llama

    if rope:
        modeling_llama.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_llama.LlamaRMSNorm = LigerRMSNorm
    if swiglu:
        modeling_llama.LlamaMLP = LigerSwiGLUMLP
    if cross_entropy:
        modeling_llama.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        modeling_llama.LlamaForCausalLM.forward = llama_lce_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules (e.g. LlamaRMSNorm or LlamaMLP)
        config: PretrainedConfig = model.config

        if hasattr(model, "model"):
            # The case for LlamaForCausalLM or LlamaForSequenceClassification, for example
            base_model = model.model
        elif hasattr(model, "transformer"):
            # LlamaForQuestionAnswering uses "transformer" instead of "model"
            base_model = model.transformer
        else:
            # Direct LlamaModel
            base_model = model

        torch_dtype = config.torch_dtype
        if rms_norm:
            base_model.norm = LigerRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            ).to(torch_dtype)

        for decoder_layer in base_model.layers:
            if swiglu:
                decoder_layer.mlp = LigerSwiGLUMLP(config).to(torch_dtype)
            if rms_norm:
                decoder_layer.input_layernorm = LigerRMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)
                decoder_layer.post_attention_layernorm = LigerRMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)


def apply_liger_kernel_to_mistral(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Mistral models

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is True.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.mistral import modeling_mistral

    if rope:
        modeling_mistral.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_mistral.MistralRMSNorm = LigerRMSNorm
    if cross_entropy:
        modeling_mistral.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        modeling_mistral.MistralForCausalLM.forward = mistral_lce_forward
    if swiglu:
        modeling_mistral.MistralMLP = LigerSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        config: PretrainedConfig = model.config

        if hasattr(model, "model"):
            # The case for MistralForCausalLM, MistralForTokenClassification for example
            base_model = model.model
        else:
            # Direct MistralModel
            base_model = model

        torch_dtype = config.torch_dtype
        if rms_norm:
            base_model.norm = LigerRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            ).to(torch_dtype)

        for decoder_layer in base_model.layers:
            if swiglu:
                decoder_layer.mlp = LigerSwiGLUMLP(config).to(torch_dtype)
            if rms_norm:
                decoder_layer.input_layernorm = LigerRMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)
                decoder_layer.post_attention_layernorm = LigerRMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)


def apply_liger_kernel_to_mixtral(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Mixtral models

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """

    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.mixtral import modeling_mixtral

    if rope:
        modeling_mixtral.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_mixtral.MixtralRMSNorm = LigerRMSNorm
    if cross_entropy:
        modeling_mixtral.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        modeling_mixtral.MixtralForCausalLM.forward = mixtral_lce_forward
    if swiglu:
        modeling_mixtral.MixtralBlockSparseTop2MLP = LigerBlockSparseTop2MLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        config: PretrainedConfig = model.config

        if hasattr(model, "model"):
            # The case for MixtralForCausalLM, MixtralForTokenClassification for example
            base_model = model.model
        else:
            # Direct MixtralModel
            base_model = model

        torch_dtype = config.torch_dtype
        if rms_norm:
            base_model.norm = LigerRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            ).to(torch_dtype)

        for decoder_layer in base_model.layers:
            if swiglu:
                block_sparse_moe = decoder_layer.block_sparse_moe
                patched_experts = nn.ModuleList(
                    [
                        LigerBlockSparseTop2MLP(config)
                        for _ in range(block_sparse_moe.num_experts)
                    ]
                )
                decoder_layer.block_sparse_moe.experts = patched_experts.to(torch_dtype)
            if rms_norm:
                decoder_layer.input_layernorm = LigerRMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)
                decoder_layer.post_attention_layernorm = LigerRMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)


def apply_liger_kernel_to_gemma(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    geglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Gemma
    (Gemma 1 and 1.1 supported, for Gemma2 please use `apply_liger_kernel_to_gemma2` ) to make GPU go burrr.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        geglu (bool): Whether to apply Liger's GeGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.gemma import modeling_gemma

    # https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/gemma/modeling_gemma.py#L109
    LigerRMSNormForGemma = partial(
        LigerRMSNorm, offset=1.0, init_fn="zeros", casting_mode="gemma"
    )

    if rope:
        modeling_gemma.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_gemma.GemmaRMSNorm = LigerRMSNormForGemma
    if cross_entropy:
        modeling_gemma.CrossEntropyLoss = LigerCrossEntropyLoss
    if geglu:
        modeling_gemma.GemmaMLP = LigerGEGLUMLP
    if fused_linear_cross_entropy:
        modeling_gemma.GemmaForCausalLM.forward = gemma_lce_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        config: PretrainedConfig = model.config

        if hasattr(model, "model"):
            # The case for GemmaForCausalLM, GemmaForTokenClassification for example
            base_model = model.model
        else:
            # Direct GemmaModel
            base_model = model

        torch_dtype = config.torch_dtype
        if rms_norm:
            base_model.norm = LigerRMSNormForGemma(
                config.hidden_size, eps=config.rms_norm_eps
            ).to(torch_dtype)

        for decoder_layer in base_model.layers:
            if geglu:
                decoder_layer.mlp = LigerGEGLUMLP(config).to(torch_dtype)
            if rms_norm:
                decoder_layer.input_layernorm = LigerRMSNormForGemma(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)
                decoder_layer.post_attention_layernorm = LigerRMSNormForGemma(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)


def apply_liger_kernel_to_gemma2(
    rope: bool = True,
    cross_entropy: bool = True,
    rms_norm: bool = True,
    geglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Gemma2
    (for Gemma1 please use `apply_liger_kernel_to_gemma`) to make GPU go burrr.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is True.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        geglu (bool): Whether to apply Liger's GeGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    from transformers.models.gemma2 import modeling_gemma2

    LigerRMSNormForGemma2 = partial(LigerRMSNorm, offset=1.0, init_fn="zeros")
    if rope:
        modeling_gemma2.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        # https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/gemma/modeling_gemma.py#L109
        modeling_gemma2.Gemma2RMSNorm = LigerRMSNormForGemma2
    if cross_entropy:
        modeling_gemma2.CrossEntropyLoss = LigerCrossEntropyLoss
    if geglu:
        modeling_gemma2.Gemma2MLP = LigerGEGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        config: PretrainedConfig = model.config

        if hasattr(model, "model"):
            # The case for Gemma2ForCausalLM, Gemma2ForTokenClassification for example
            base_model = model.model
        else:
            # Direct Gemma2Model
            base_model = model

        torch_dtype = config.torch_dtype
        if rms_norm:
            base_model.norm = LigerRMSNormForGemma2(
                config.hidden_size, eps=config.rms_norm_eps
            ).to(torch_dtype)

        for decoder_layer in base_model.layers:
            if geglu:
                decoder_layer.mlp = LigerGEGLUMLP(config).to(torch_dtype)
            if rms_norm:
                decoder_layer.input_layernorm = LigerRMSNormForGemma2(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)
                decoder_layer.post_attention_layernorm = LigerRMSNormForGemma2(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)
                decoder_layer.pre_feedforward_layernorm = LigerRMSNormForGemma2(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)
                decoder_layer.post_feedforward_layernorm = LigerRMSNormForGemma2(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)


def apply_liger_kernel_to_qwen2(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2 models

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.qwen2 import modeling_qwen2

    if rope:
        modeling_qwen2.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_qwen2.Qwen2RMSNorm = LigerRMSNorm
    if cross_entropy:
        modeling_qwen2.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        modeling_qwen2.Qwen2ForCausalLM.forward = qwen2_lce_forward
    if swiglu:
        modeling_qwen2.Qwen2MLP = LigerSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        config: PretrainedConfig = model.config

        if hasattr(model, "model"):
            # The case for Qwen2ForCausalLM, Qwen2ForTokenClassification for example
            base_model = model.model
        else:
            # Direct Qwen2Model
            base_model = model

        torch_dtype = config.torch_dtype
        if rms_norm:
            base_model.norm = LigerRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            ).to(torch_dtype)

        for decoder_layer in base_model.layers:
            if swiglu:
                decoder_layer.mlp = LigerSwiGLUMLP(config).to(torch_dtype)
            if rms_norm:
                decoder_layer.input_layernorm = LigerRMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)
                decoder_layer.post_attention_layernorm = LigerRMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)


def apply_liger_kernel_to_qwen2_vl(
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    layer_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2-VL models.
    NOTE: Qwen2-VL is not available in transformers<=4.44.2

    Args:
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        layer_norm (bool): Whether to apply Liger's LayerNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.qwen2_vl import modeling_qwen2_vl

    from liger_kernel.transformers.model.qwen2_vl import (
        lce_forward as qwen2_vl_lce_forward,
    )

    # TODO: Support Qwen2-VL's multimodal RoPE implementation

    LigerRMSNormForQwen2VL = partial(LigerRMSNorm, init_fn="ones", casting_mode="gemma")
    if rms_norm:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L439
        modeling_qwen2_vl.Qwen2RMSNorm = LigerRMSNormForQwen2VL
    if layer_norm:
        modeling_qwen2_vl.LayerNorm = LigerLayerNorm
    if cross_entropy:
        modeling_qwen2_vl.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = qwen2_vl_lce_forward
    if swiglu:
        modeling_qwen2_vl.Qwen2MLP = LigerSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        config: PretrainedConfig = model.config

        torch_dtype = config.torch_dtype

        if hasattr(model, "model"):
            # The case for Qwen2VLForConditionalGeneration.
            base_model = model.model
        else:
            # Direct Qwen2VLModel
            base_model = model

        if hasattr(model, "visual"):
            # Patch Qwen2VisionTransformerPretrainedModel
            for vision_block in model.visual.blocks:
                if layer_norm:
                    vision_block.norm1 = LigerLayerNorm(config.embed_dim, eps=1e-6).to(
                        torch_dtype
                    )
                    vision_block.norm2 = LigerLayerNorm(config.embed_dim, eps=1e-6).to(
                        torch_dtype
                    )

        if rms_norm:
            base_model.norm = LigerRMSNormForQwen2VL(
                config.hidden_size, eps=config.rms_norm_eps
            ).to(torch_dtype)
        for decoder_layer in base_model.layers:
            if swiglu:
                decoder_layer.mlp = LigerSwiGLUMLP(config).to(torch_dtype)
            if rms_norm:
                decoder_layer.input_layernorm = LigerRMSNormForQwen2VL(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)
                decoder_layer.post_attention_layernorm = LigerRMSNormForQwen2VL(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)


def apply_liger_kernel_to_phi3(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Phi3 models.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU Phi3MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.phi3 import modeling_phi3

    if rope:
        modeling_phi3.apply_rotary_pos_emb = liger_rotary_pos_emb  # Same as Gemma
    if rms_norm:
        modeling_phi3.Phi3RMSNorm = LigerRMSNorm  # Same as Llama
    if swiglu:
        modeling_phi3.Phi3MLP = LigerPhi3SwiGLUMLP
    if cross_entropy:
        modeling_phi3.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        modeling_phi3.Phi3ForCausalLM.forward = phi3_lce_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        config: PretrainedConfig = model.config

        if hasattr(model, "model"):
            # The case for Phi3ForCausalLM, Phi3ForTokenClassification for example
            base_model = model.model
        else:
            # Direct Phi3Model
            base_model = model

        torch_dtype = config.torch_dtype
        if rms_norm:
            base_model.norm = LigerRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            ).to(torch_dtype)

        for decoder_layer in base_model.layers:
            if swiglu:
                decoder_layer.mlp = LigerPhi3SwiGLUMLP(config).to(torch_dtype)
            if rms_norm:
                decoder_layer.input_layernorm = LigerRMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)
                decoder_layer.post_attention_layernorm = LigerRMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)


# Model type corresponds to the keys defined in transformers/models/auto/modeling_auto.py
MODEL_TYPE_TO_APPLY_LIGER_FN = {
    "gemma": apply_liger_kernel_to_gemma,
    "gemma2": apply_liger_kernel_to_gemma2,
    "llama": apply_liger_kernel_to_llama,
    "mistral": apply_liger_kernel_to_mistral,
    "mixtral": apply_liger_kernel_to_mixtral,
    "qwen2": apply_liger_kernel_to_qwen2,
    "qwen2_vl": apply_liger_kernel_to_qwen2_vl,
    "phi3": apply_liger_kernel_to_phi3,
}


def _apply_liger_kernel(model_type: str, **kwargs) -> None:
    """
    Applies Liger kernels based on the specified model type. The custom
    kernels for the specified model type will be applied with the provided
    keyword arguments, otherwise the default configuration will be used.

    ** Note: Calling _apply_liger_kernel() after model initialization
    will not be able to fully patch models. This must be called before model initialization.
    If the model has already been instantiated

    Args:
        - model_type: the model types as defined in transformers/models/auto/modeling_auto.py
          and specified in the model's config.json
        - kwargs: keyword arguments that are passed to the corresponding apply_liger_kernel_to_* function.
    """
    if not model_type:
        logger.info("Model type was not provided. No Liger kernels will be applied.")
        return

    if model_type not in MODEL_TYPE_TO_APPLY_LIGER_FN.keys():
        logger.info(
            f"There are currently no Liger kernels supported for model type: {model_type}."
        )
        return

    apply_fn = MODEL_TYPE_TO_APPLY_LIGER_FN[model_type]
    apply_fn_signature = inspect.signature(apply_fn)

    # Filter out the keyword arguments that are not supported by the apply function
    applicable_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in apply_fn_signature.parameters
    }

    logger.info(
        f"Applying Liger kernels for model type: {model_type} with kwargs: {applicable_kwargs}"
    )

    # Assume this is invoked pre-model initialization, so we only need to patch transformers code
    apply_fn(**applicable_kwargs)


def _apply_liger_kernel_to_instance(model: PreTrainedModel, **kwargs) -> None:
    """
    Applies Liger kernels to the provided model instance.

    Args:
        - model: the model instance to apply Liger kernels to
        - kwargs: keyword arguments that are passed to the corresponding apply_liger_kernel_to_* function.
    """
    model_type = getattr(model, "config", None) and getattr(
        model.config, "model_type", None
    )

    if not model_type:
        logger.info(
            "Model type could not be determined from model config. No Liger kernels will be applied."
        )
        return

    if model_type not in MODEL_TYPE_TO_APPLY_LIGER_FN.keys():
        logger.info(
            f"There are currently no Liger kernels supported for model type: {model_type}."
        )
        return

    apply_fn = MODEL_TYPE_TO_APPLY_LIGER_FN[model_type]

    apply_fn_signature = inspect.signature(apply_fn)

    # Filter out the keyword arguments that are not supported by the apply function
    applicable_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in apply_fn_signature.parameters
    }

    logger.info(
        f"Applying Liger kernels to model instance with model type: {model_type} with kwargs: {applicable_kwargs}"
    )

    apply_fn(model=model, **applicable_kwargs)
