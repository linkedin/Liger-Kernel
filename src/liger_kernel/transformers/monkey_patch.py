import inspect
import logging
from functools import partial
from typing import Callable

from transformers import PreTrainedModel

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


def _bind_method_to_module(module, method_name: str, new_method: Callable):
    # Binds a new method to a module instance so that self is passed as the first argument
    module.__dict__[method_name] = new_method.__get__(module, module.__class__)


def _patch_rms_norm_module(module, offset=0.0, eps=1e-6, casting_mode="llama"):
    module.offset = offset
    module.casting_mode = casting_mode
    module.variance_epsilon = (
        getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or eps
    )
    _bind_method_to_module(module, "forward", LigerRMSNorm.forward)
    _bind_method_to_module(module, "extra_repr", LigerRMSNorm.extra_repr)


def _patch_layer_norm_module(module, eps=1e-6):
    module.variance_epsilon = (
        getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or eps
    )
    module.hidden_size = module.normalized_shape
    _bind_method_to_module(module, "forward", LigerLayerNorm.forward)
    _bind_method_to_module(module, "extra_repr", LigerLayerNorm.extra_repr)


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

        if hasattr(model, "model"):
            # The case for LlamaForCausalLM or LlamaForSequenceClassification, for example
            base_model = model.model
        elif hasattr(model, "transformer"):
            # LlamaForQuestionAnswering uses "transformer" instead of "model"
            base_model = model.transformer
        else:
            # Direct LlamaModel
            base_model = model

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _bind_method_to_module(
                    decoder_layer.mlp, "forward", LigerSwiGLUMLP.forward
                )
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


def apply_liger_kernel_to_mllama(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    layer_norm: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace MLlama models.
    NOTE: MLlama is not available in transformers<4.45.0

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

    from transformers.models.mllama import modeling_mllama
    from transformers.models.mllama.modeling_mllama import (
        MllamaForCausalLM,
        MllamaForConditionalGeneration,
        MllamaTextModel,
        MllamaVisionModel,
    )

    from liger_kernel.transformers.model.mllama import lce_forward as mllama_lce_forward

    if rope:
        modeling_mllama.apply_rotary_pos_emb = liger_rotary_pos_emb
    if layer_norm:
        modeling_mllama.nn.LayerNorm = LigerLayerNorm
    if rms_norm:
        modeling_mllama.MllamaTextRMSNorm = LigerRMSNorm
    if swiglu:
        modeling_mllama.MllamaTextMLP = LigerSwiGLUMLP
    if cross_entropy:
        modeling_mllama.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        # MllamaForConditionalGeneration uses MllamaForCausalLM under the hood
        # for the loss calculation, so we need to patch the forward method of MllamaForCausalLM
        modeling_mllama.MllamaForCausalLM.forward = mllama_lce_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        if isinstance(model, MllamaForConditionalGeneration):
            language_model: MllamaForCausalLM = model.language_model
            vision_model: MllamaVisionModel = model.vision_model
            text_model: MllamaTextModel = language_model.model
        elif isinstance(model, MllamaForCausalLM):
            text_model = model.model
            vision_model = None
        elif isinstance(model, MllamaTextModel):
            text_model = model
            vision_model = None
        else:
            raise ValueError(f"Unsupported Mllama model type: {type(model)}")

        if text_model:
            if rms_norm:
                _patch_rms_norm_module(text_model.norm)
            for decoder_layer in text_model.layers:
                if swiglu:
                    _bind_method_to_module(
                        decoder_layer.mlp, "forward", LigerSwiGLUMLP.forward
                    )
                if rms_norm:
                    _patch_rms_norm_module(decoder_layer.input_layernorm)
                    _patch_rms_norm_module(decoder_layer.post_attention_layernorm)

        if vision_model:
            _patch_layer_norm_module(vision_model.layernorm_pre)
            _patch_layer_norm_module(vision_model.layernorm_post)

            for layer in vision_model.transformer.layers:
                if layer_norm:
                    _patch_layer_norm_module(layer.input_layernorm)
                    _patch_layer_norm_module(layer.post_attention_layernorm)

            for layer in vision_model.global_transformer.layers:
                if layer_norm:
                    _patch_layer_norm_module(layer.input_layernorm)
                    _patch_layer_norm_module(layer.post_attention_layernorm)


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

        if hasattr(model, "model"):
            # The case for MistralForCausalLM, MistralForTokenClassification for example
            base_model = model.model
        else:
            # Direct MistralModel
            base_model = model

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _bind_method_to_module(
                    decoder_layer.mlp, "forward", LigerSwiGLUMLP.forward
                )
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


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

        if hasattr(model, "model"):
            # The case for MixtralForCausalLM, MixtralForTokenClassification for example
            base_model = model.model
        else:
            # Direct MixtralModel
            base_model = model

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                for expert in decoder_layer.block_sparse_moe.experts:
                    _bind_method_to_module(
                        expert, "forward", LigerBlockSparseTop2MLP.forward
                    )
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


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
    _patch_rms_norm_module_for_gemma = partial(
        _patch_rms_norm_module, casting_mode="gemma", offset=1.0
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

        if hasattr(model, "model"):
            # The case for GemmaForCausalLM, GemmaForTokenClassification for example
            base_model = model.model
        else:
            # Direct GemmaModel
            base_model = model

        if rms_norm:
            _patch_rms_norm_module_for_gemma(base_model.norm)

        for decoder_layer in base_model.layers:
            if geglu:
                _bind_method_to_module(
                    decoder_layer.mlp, "forward", LigerGEGLUMLP.forward
                )
            if rms_norm:
                _patch_rms_norm_module_for_gemma(decoder_layer.input_layernorm)
                _patch_rms_norm_module_for_gemma(decoder_layer.post_attention_layernorm)


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

    LigerRMSNormForGemma2 = partial(
        LigerRMSNorm, offset=1.0, casting_mode="gemma", init_fn="zeros"
    )
    _patch_rms_norm_module_for_gemma2 = partial(
        _patch_rms_norm_module, offset=1.0, casting_mode="gemma"
    )

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

        if hasattr(model, "model"):
            # The case for Gemma2ForCausalLM, Gemma2ForTokenClassification for example
            base_model = model.model
        else:
            # Direct Gemma2Model
            base_model = model

        if rms_norm:
            _patch_rms_norm_module_for_gemma2(base_model.norm)

        for decoder_layer in base_model.layers:
            if geglu:
                _bind_method_to_module(
                    decoder_layer.mlp, "forward", LigerGEGLUMLP.forward
                )
            if rms_norm:
                _patch_rms_norm_module_for_gemma2(decoder_layer.input_layernorm)
                _patch_rms_norm_module_for_gemma2(
                    decoder_layer.post_attention_layernorm
                )
                _patch_rms_norm_module_for_gemma2(
                    decoder_layer.pre_feedforward_layernorm
                )
                _patch_rms_norm_module_for_gemma2(
                    decoder_layer.post_feedforward_layernorm
                )


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

        if hasattr(model, "model"):
            # The case for Qwen2ForCausalLM, Qwen2ForTokenClassification for example
            base_model = model.model
        else:
            # Direct Qwen2Model
            base_model = model

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _bind_method_to_module(
                    decoder_layer.mlp, "forward", LigerSwiGLUMLP.forward
                )
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


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
    NOTE: Qwen2-VL is not available in transformers<4.45.0

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

    if rms_norm:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L439
        modeling_qwen2_vl.Qwen2RMSNorm = LigerRMSNorm
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
                    _patch_layer_norm_module(vision_block.norm1)
                    _patch_layer_norm_module(vision_block.norm2)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)
        for decoder_layer in base_model.layers:
            if swiglu:
                _bind_method_to_module(
                    decoder_layer.mlp, "forward", LigerSwiGLUMLP.forward
                )
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


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

        if hasattr(model, "model"):
            # The case for Phi3ForCausalLM, Phi3ForTokenClassification for example
            base_model = model.model
        else:
            # Direct Phi3Model
            base_model = model

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _bind_method_to_module(
                    decoder_layer.mlp, "forward", LigerPhi3SwiGLUMLP.forward
                )
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


# Model type corresponds to the keys defined in transformers/models/auto/modeling_auto.py
MODEL_TYPE_TO_APPLY_LIGER_FN = {
    "gemma": apply_liger_kernel_to_gemma,
    "gemma2": apply_liger_kernel_to_gemma2,
    "llama": apply_liger_kernel_to_llama,
    "mllama": apply_liger_kernel_to_mllama,
    "mllama_text_model": apply_liger_kernel_to_mllama,
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
