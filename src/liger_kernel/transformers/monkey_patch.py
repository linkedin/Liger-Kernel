import inspect
import logging

from functools import partial
from types import MethodType
from typing import Callable
from typing import Optional

import transformers

from packaging import version
from transformers import PreTrainedModel

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.functional import liger_cross_entropy
from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.transformers.geglu import LigerGEGLUMLPForGemma4
from liger_kernel.transformers.layer_norm import LigerLayerNorm
from liger_kernel.transformers.model.falcon_h1 import lce_forward as falcon_h1_lce_forward
from liger_kernel.transformers.model.gemma import lce_forward as gemma_lce_forward
from liger_kernel.transformers.model.gemma2 import lce_forward as gemma2_lce_forward
from liger_kernel.transformers.model.gpt_oss import lce_forward as gpt_oss_lce_forward
from liger_kernel.transformers.model.llama import lce_forward as llama_lce_forward
from liger_kernel.transformers.model.llava import lce_forward as llava_lce_forward
from liger_kernel.transformers.model.ministral import lce_forward as ministral_lce_forward
from liger_kernel.transformers.model.mistral import lce_forward as mistral_lce_forward
from liger_kernel.transformers.model.mixtral import lce_forward as mixtral_lce_forward
from liger_kernel.transformers.model.nemotron import lce_forward as nemotron_lce_forward
from liger_kernel.transformers.model.phi3 import lce_forward as phi3_lce_forward
from liger_kernel.transformers.model.qwen2 import lce_forward as qwen2_lce_forward
from liger_kernel.transformers.model.smollm3 import lce_forward as smollm3_lce_forward
from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
from liger_kernel.transformers.relu_squared import LigerReLUSquared
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from liger_kernel.transformers.rope import liger_rotary_pos_emb_vision
from liger_kernel.transformers.swiglu import LigerBlockSparseTop2MLP
from liger_kernel.transformers.swiglu import LigerExperts
from liger_kernel.transformers.swiglu import LigerPhi3SwiGLUMLP
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

try:
    import peft

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

transformer_version = version.parse(transformers.__version__)

logger = logging.getLogger(__name__)

MIN_SUPPORTED_TRANSFORMERS_VERSION = version.parse("4.52.0")
if transformer_version < MIN_SUPPORTED_TRANSFORMERS_VERSION:
    raise ImportError(
        f"liger-kernel requires transformers >= {MIN_SUPPORTED_TRANSFORMERS_VERSION}, got {transformers.__version__}. "
        "Please install an older version of liger-kernel that is compatible with your transformers version."
    )

IS_TRANSFORMERS_V5_OR_LATER = version.parse(transformers.__version__) >= version.parse("5.0.0")


def _bind_method_to_module(module, method_name: str, new_method: Callable):
    # Binds a new method to a module instance so that self is passed as the first argument
    module.__dict__[method_name] = new_method.__get__(module, module.__class__)


def _patch_rms_norm_module(module, offset=0.0, eps=1e-6, casting_mode="llama", in_place=True, row_mode=None):
    # Check if the module is a PEFT ModulesToSaveWrapper
    # If it is, we need to patch the modules_to_save.default and original_modules
    if PEFT_AVAILABLE and isinstance(module, peft.utils.other.ModulesToSaveWrapper):
        module.modules_to_save.default.offset = offset
        module.modules_to_save.default.casting_mode = casting_mode
        module.modules_to_save.default.variance_epsilon = (
            getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or eps
        )
        module.modules_to_save.default.in_place = in_place
        module.modules_to_save.default.row_mode = row_mode
        module.original_module.offset = offset
        module.original_module.casting_mode = casting_mode
        module.original_module.variance_epsilon = (
            getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or eps
        )
        module.original_module.in_place = in_place
        module.original_module.row_mode = row_mode
        _bind_method_to_module(module.modules_to_save.default, "forward", LigerRMSNorm.forward)
        _bind_method_to_module(module.modules_to_save.default, "extra_repr", LigerRMSNorm.extra_repr)
        _bind_method_to_module(module.original_module, "forward", LigerRMSNorm.forward)
        _bind_method_to_module(module.original_module, "extra_repr", LigerRMSNorm.extra_repr)
        _bind_method_to_module(module.modules_to_save.default, "_get_name", lambda self: LigerRMSNorm.__name__)
        _bind_method_to_module(module.original_module, "_get_name", lambda self: LigerRMSNorm.__name__)
    else:
        module.offset = offset
        module.casting_mode = casting_mode
        module.variance_epsilon = getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or eps
        module.in_place = in_place
        module.row_mode = row_mode
        _bind_method_to_module(module, "forward", LigerRMSNorm.forward)
        _bind_method_to_module(module, "extra_repr", LigerRMSNorm.extra_repr)
        _bind_method_to_module(module, "_get_name", lambda self: LigerRMSNorm.__name__)


def _patch_layer_norm_module(module, eps=1e-6):
    # Check if the module is a PEFT ModulesToSaveWrapper
    # If it is, we need to patch the modules_to_save.default and original_modules
    if PEFT_AVAILABLE and isinstance(module, peft.utils.other.ModulesToSaveWrapper):
        module.hidden_size = module.normalized_shape
        _bind_method_to_module(module, "forward", LigerLayerNorm.forward)
        _bind_method_to_module(module, "extra_repr", LigerLayerNorm.extra_repr)
        module.modules_to_save.default.variance_epsilon = (
            getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or eps
        )
        module.original_module.hidden_size = getattr(module, "hidden_size", None) or getattr(
            module, "normalized_shape", None
        )
        module.original_module.variance_epsilon = (
            getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or eps
        )
        module.original_module.hidden_size = getattr(module, "hidden_size", None) or getattr(
            module, "normalized_shape", None
        )
        _bind_method_to_module(module.modules_to_save.default, "forward", LigerLayerNorm.forward)
        _bind_method_to_module(module.modules_to_save.default, "extra_repr", LigerLayerNorm.extra_repr)
        _bind_method_to_module(module.original_module, "forward", LigerLayerNorm.forward)
        _bind_method_to_module(module.original_module, "extra_repr", LigerLayerNorm.extra_repr)
        _bind_method_to_module(module.modules_to_save.default, "_get_name", lambda self: LigerLayerNorm.__name__)
        _bind_method_to_module(module.original_module, "_get_name", lambda self: LigerLayerNorm.__name__)
    else:
        module.variance_epsilon = getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or eps
        module.hidden_size = getattr(module, "hidden_size", None) or getattr(module, "normalized_shape", None)
        _bind_method_to_module(module, "forward", LigerLayerNorm.forward)
        _bind_method_to_module(module, "extra_repr", LigerLayerNorm.extra_repr)
        _bind_method_to_module(module, "_get_name", lambda self: LigerLayerNorm.__name__)


def _patch_swiglu_module(module, liger_module):
    _bind_method_to_module(module, "forward", liger_module.forward)
    _bind_method_to_module(module, "_get_name", lambda self: liger_module.__name__)


def _patch_geglu_module(module):
    _bind_method_to_module(module, "forward", LigerGEGLUMLP.forward)
    _bind_method_to_module(module, "_get_name", lambda self: LigerGEGLUMLP.__name__)


def apply_liger_kernel_to_granite(
    rope: bool = True,
    cross_entropy: bool = True,
    fused_linear_cross_entropy: bool = False,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Granite 3 models

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is True.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is False.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.



    Debugging notes:
        If LigerSwiGLUMLP is OK for Llama, it should be fine for Granite, but it's not.
    """

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.granite import modeling_granite
    from transformers.models.granite.modeling_granite import GraniteModel

    if swiglu:
        modeling_granite.GraniteMLP = LigerSwiGLUMLP

    if rms_norm:
        modeling_granite.GraniteRMSNorm = LigerRMSNorm

    if rope:
        modeling_granite.apply_rotary_pos_emb = liger_rotary_pos_emb

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        raise NotImplementedError("LigerFusedLinearCrossEntropy is not available for Granite models.")
        # NOTE: Granite model `GraniteForCausalLM.forward` scales logits each
        # call, so we can't sidestep logit materialization. A bit more work
        # would be needed to add a scaling term to the `LigerFusedLinearCrossEntropyFunction`
        # for the logit output.

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules (e.g. GraniteRMSNorm or GraniteMLP)

        # get the base model from the model instance
        base_model: GraniteModel = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


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

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.llama import modeling_llama
    from transformers.models.llama.modeling_llama import LlamaModel

    if rope:
        modeling_llama.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_llama.LlamaRMSNorm = LigerRMSNorm
    if swiglu:
        modeling_llama.LlamaMLP = LigerSwiGLUMLP

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(llama_lce_forward, model)
        else:
            modeling_llama.LlamaForCausalLM.forward = llama_lce_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules (e.g. LlamaRMSNorm or LlamaMLP)

        # get the base model from the model instance
        base_model: LlamaModel = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


def apply_liger_kernel_to_smollm3(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace SmolLM3 model

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

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.smollm3 import modeling_smollm3
    from transformers.models.smollm3.modeling_smollm3 import SmolLM3Model

    if rope:
        modeling_smollm3.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_smollm3.SmolLM3RMSNorm = LigerRMSNorm
    if swiglu:
        modeling_smollm3.SmolLM3MLP = LigerSwiGLUMLP

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(smollm3_lce_forward, model)
        else:
            modeling_smollm3.SmolLM3ForCausalLM.forward = smollm3_lce_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules (e.g. SmolLM3RMSNorm or SmolLM3MLP)

        # get the base model from the model instance
        base_model: SmolLM3Model = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


def apply_liger_kernel_to_llava(
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    model: PreTrainedModel = None,
    **kwargs,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Llava models.
    Due to the characteristics of LlaVa, the model must be passed to apply Liger-Kernel's patch to other models connected to LLaVa.
    However, if an LM not supported by Liger-Kernel is connected to LLaVa, unexpected side effects may occur.
    NOTE: Llava is not available in transformers<4.36.0

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
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.llava import modeling_llava

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(llava_lce_forward, model)
        else:
            modeling_llava.LlavaForConditionalGeneration.forward = llava_lce_forward

    if model is not None:
        text_model_name, vision_model_name = model.config.text_config.model_type, model.config.vision_config.model_type
        text_liger_fn = MODEL_TYPE_TO_APPLY_LIGER_FN.get(text_model_name, None)
        vision_liger_fn = MODEL_TYPE_TO_APPLY_LIGER_FN.get(vision_model_name, None)

        kwargs = {"cross_entropy": False, "fused_linear_cross_entropy": False, **kwargs}
        if text_liger_fn:
            accept_params = inspect.signature(text_liger_fn).parameters
            remain_params = set(kwargs) - (set(accept_params) & set(kwargs))
            text_kwargs = {k: v for k, v in kwargs.items() if k not in remain_params}

            if remain_params:
                logger.warning(
                    f"These parameters are not supported by {text_model_name}. Enter the remaining {list(text_kwargs.keys())} except for {list(remain_params)}\n"
                    f"Parameters accepted by {text_model_name}: {list(accept_params.keys())}"
                )
            text_kwargs["model"] = model.model.language_model
            text_liger_fn(**text_kwargs)
        elif text_model_name not in MODEL_TYPE_TO_APPLY_LIGER_FN:
            logger.warning(f"{text_model_name} is not supported by Liger kernel.")

        if vision_liger_fn:
            accept_params = inspect.signature(vision_liger_fn).parameters
            remain_params = set(kwargs) - (set(accept_params) & set(kwargs))
            vision_kwargs = {k: v for k, v in kwargs.items() if k not in remain_params}

            if remain_params:
                logger.warning(
                    f"These parameters are not supported by {vision_model_name}. Enter the remaining {list(vision_kwargs.keys())} except for {list(remain_params)}\n"
                    f"Parameters accepted by {vision_model_name}: {list(accept_params.keys())}"
                )
            vision_kwargs["model"] = model.model.vision_tower
            vision_liger_fn(**vision_kwargs)
        elif vision_model_name not in MODEL_TYPE_TO_APPLY_LIGER_FN:
            logger.warning(f"{vision_model_name} is not supported by Liger kernel.")


def apply_liger_kernel_to_llama4(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    layer_norm: bool = True,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Llama4 models.

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
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.llama4 import modeling_llama4
    from transformers.models.llama4.modeling_llama4 import Llama4ForCausalLM
    from transformers.models.llama4.modeling_llama4 import Llama4ForConditionalGeneration
    from transformers.models.llama4.modeling_llama4 import Llama4TextModel
    from transformers.models.llama4.modeling_llama4 import Llama4VisionModel

    from liger_kernel.transformers.model.llama4 import lce_forward as llama4_lce_forward

    if rope:
        from liger_kernel.transformers.llama4_rope import apply_liger_llama4_rope_full

        apply_liger_llama4_rope_full(modeling_llama4)
    if rms_norm:
        modeling_llama4.Llama4TextRMSNorm = LigerRMSNorm
    if swiglu:
        modeling_llama4.Llama4TextMLP = LigerSwiGLUMLP

    if cross_entropy:
        modeling_llama4.CrossEntropyLoss = LigerCrossEntropyLoss

    if fused_linear_cross_entropy:
        modeling_llama4.Llama4ForCausalLM.forward = llama4_lce_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        if isinstance(model, Llama4ForConditionalGeneration):
            language_model: Llama4ForCausalLM = model.language_model
            vision_model: Llama4VisionModel = model.vision_model
            text_model: Llama4TextModel = language_model.model
        elif isinstance(model, Llama4ForCausalLM):
            text_model = model.model
            vision_model = None
        elif isinstance(model, Llama4TextModel):
            text_model = model
            vision_model = None

        else:
            raise ValueError(f"Unsupported Llama4 model type: {type(model)}")

        if text_model:
            if rms_norm:
                _patch_rms_norm_module(text_model.norm)
            for decoder_layer in text_model.layers:
                if swiglu:
                    if decoder_layer.is_moe_layer:
                        _patch_swiglu_module(decoder_layer.feed_forward.shared_expert, LigerSwiGLUMLP)
                    else:
                        _patch_swiglu_module(decoder_layer.feed_forward, LigerSwiGLUMLP)
                if rms_norm:
                    _patch_rms_norm_module(decoder_layer.input_layernorm)
                    _patch_rms_norm_module(decoder_layer.post_attention_layernorm)

        if vision_model:
            _patch_layer_norm_module(vision_model.layernorm_pre)
            _patch_layer_norm_module(vision_model.layernorm_post)

            for layer in vision_model.model.layers:
                if layer_norm:
                    _patch_layer_norm_module(layer.input_layernorm)
                    _patch_layer_norm_module(layer.post_attention_layernorm)


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

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.mllama import modeling_mllama
    from transformers.models.mllama.modeling_mllama import MllamaForCausalLM
    from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration
    from transformers.models.mllama.modeling_mllama import MllamaTextModel
    from transformers.models.mllama.modeling_mllama import MllamaVisionModel

    from liger_kernel.transformers.model.mllama import lce_forward as mllama_lce_forward

    if rope:
        modeling_mllama.apply_rotary_pos_emb = liger_rotary_pos_emb
    if layer_norm and model is None:
        modeling_mllama.nn.LayerNorm = LigerLayerNorm
    if rms_norm:
        modeling_mllama.MllamaTextRMSNorm = LigerRMSNorm
    if swiglu:
        modeling_mllama.MllamaTextMLP = LigerSwiGLUMLP
    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(mllama_lce_forward, model)
        else:
            modeling_mllama.MllamaForCausalLM.forward = mllama_lce_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        if isinstance(model, MllamaForConditionalGeneration):
            language_model: MllamaForCausalLM = model.model.language_model
            vision_model: MllamaVisionModel = model.model.vision_model
            if isinstance(language_model, MllamaForCausalLM):
                text_model: MllamaTextModel = language_model.model
            else:
                text_model = language_model
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
                    _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
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


def apply_liger_kernel_to_ministral(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Ministral models

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
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.ministral import modeling_ministral
    from transformers.models.ministral.modeling_ministral import MinistralModel

    if rope:
        modeling_ministral.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_ministral.MinistralRMSNorm = LigerRMSNorm
    if cross_entropy:
        modeling_ministral.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(ministral_lce_forward, model)
        else:
            modeling_ministral.MinistralForCausalLM.forward = ministral_lce_forward

    if swiglu:
        modeling_ministral.MinistralMLP = LigerSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: MinistralModel = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


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
        rope (bool): Whether to apply Liger's rotary position embedding. Default is False.
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
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.mistral import modeling_mistral
    from transformers.models.mistral.modeling_mistral import MistralModel

    if rope:
        modeling_mistral.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_mistral.MistralRMSNorm = LigerRMSNorm
    if cross_entropy:
        modeling_mistral.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(mistral_lce_forward, model)
        else:
            modeling_mistral.MistralForCausalLM.forward = mistral_lce_forward

    if swiglu:
        modeling_mistral.MistralMLP = LigerSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: MistralModel = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


def apply_liger_kernel_to_nemotron(
    relu_squared: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    model: PreTrainedModel = None,
    **kwargs,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Nemotron models.

    Note: NemotronLayerNorm1P (LayerNorm with +1 offset) is not currently supported by Liger kernels.
    RoPE is also not patched because Nemotron uses partial rotary embeddings
    (partial_rotary_factor=0.5) which the Liger RoPE kernel does not support.

    Args:
        relu_squared (bool): Whether to apply Liger's ReLU squared activation. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.nemotron import modeling_nemotron

    if relu_squared:
        modeling_nemotron.ACT2FN["relu2"] = LigerReLUSquared

    if cross_entropy:
        modeling_nemotron.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(nemotron_lce_forward, model)
        else:
            modeling_nemotron.NemotronForCausalLM.forward = nemotron_lce_forward

    if model is not None:
        for decoder_layer in model.model.layers:
            if relu_squared:
                decoder_layer.mlp.act_fn = LigerReLUSquared()


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

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.mixtral import modeling_mixtral
    from transformers.models.mixtral.modeling_mixtral import MixtralModel

    if rope:
        modeling_mixtral.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_mixtral.MixtralRMSNorm = LigerRMSNorm
    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(mixtral_lce_forward, model)
        else:
            modeling_mixtral.MixtralForCausalLM.forward = mixtral_lce_forward
    if swiglu:
        if IS_TRANSFORMERS_V5_OR_LATER:
            modeling_mixtral.MixtralExperts = LigerExperts
        else:
            modeling_mixtral.MixtralBlockSparseTop2MLP = LigerBlockSparseTop2MLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: MixtralModel = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                if IS_TRANSFORMERS_V5_OR_LATER:
                    _patch_swiglu_module(decoder_layer.mlp.experts, LigerExperts)
                else:
                    for expert in decoder_layer.block_sparse_moe.experts:
                        _patch_swiglu_module(expert, LigerBlockSparseTop2MLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


def apply_liger_kernel_to_pixtral(
    rope: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Pixtral vision models.

    Note: Pixtral's vision encoder does not have a cross-entropy loss, so there is no
    `fused_linear_cross_entropy` or `cross_entropy` option. The language model side of
    Pixtral uses Mistral, which can be patched separately via `apply_liger_kernel_to_mistral`.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model
            has already been loaded. Default is None.
    """
    from transformers.models.pixtral import modeling_pixtral
    from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel

    if rope:
        modeling_pixtral.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_pixtral.PixtralRMSNorm = LigerRMSNorm
    if swiglu:
        modeling_pixtral.PixtralMLP = LigerSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules.
        if isinstance(model, PixtralVisionModel):
            transformer = model.transformer
        else:
            raise ValueError(f"Unsupported Pixtral model type: {type(model)}")

        if rms_norm:
            _patch_rms_norm_module(model.ln_pre, eps=1e-5)

        for layer in transformer.layers:
            if swiglu:
                _patch_swiglu_module(layer.feed_forward, LigerSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(layer.attention_norm, eps=1e-5)
                _patch_rms_norm_module(layer.ffn_norm, eps=1e-5)


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
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.gemma import modeling_gemma
    from transformers.models.gemma.modeling_gemma import GemmaModel

    from liger_kernel.transformers.rms_norm import LigerRMSNormForGemma

    _patch_rms_norm_module_for_gemma = partial(_patch_rms_norm_module, casting_mode="gemma", offset=1.0)

    if rope:
        modeling_gemma.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_gemma.GemmaRMSNorm = LigerRMSNormForGemma
    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy
    if geglu:
        modeling_gemma.GemmaMLP = LigerGEGLUMLP
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(gemma_lce_forward, model)
        else:
            modeling_gemma.GemmaForCausalLM.forward = gemma_lce_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: GemmaModel = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module_for_gemma(base_model.norm)

        for decoder_layer in base_model.layers:
            if geglu:
                _patch_geglu_module(decoder_layer.mlp)
            if rms_norm:
                _patch_rms_norm_module_for_gemma(decoder_layer.input_layernorm)
                _patch_rms_norm_module_for_gemma(decoder_layer.post_attention_layernorm)


def apply_liger_kernel_to_gemma2(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    geglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Gemma2
    (for Gemma1 please use `apply_liger_kernel_to_gemma`) to make GPU go burrr.

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
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.gemma2 import modeling_gemma2
    from transformers.models.gemma2.modeling_gemma2 import Gemma2Model

    from liger_kernel.transformers.rms_norm import LigerRMSNormForGemma2

    _patch_rms_norm_module_for_gemma2 = partial(
        _patch_rms_norm_module, offset=1.0, casting_mode="gemma", in_place=False
    )

    if rope:
        modeling_gemma2.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        # https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/gemma/modeling_gemma.py#L109
        modeling_gemma2.Gemma2RMSNorm = LigerRMSNormForGemma2
    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(gemma2_lce_forward, model)
        else:
            modeling_gemma2.Gemma2ForCausalLM.forward = gemma2_lce_forward
    if geglu:
        modeling_gemma2.Gemma2MLP = LigerGEGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: Gemma2Model = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module_for_gemma2(base_model.norm)

        for decoder_layer in base_model.layers:
            if geglu:
                _patch_geglu_module(decoder_layer.mlp)
            if rms_norm:
                _patch_rms_norm_module_for_gemma2(decoder_layer.input_layernorm)
                _patch_rms_norm_module_for_gemma2(decoder_layer.post_attention_layernorm)
                _patch_rms_norm_module_for_gemma2(decoder_layer.pre_feedforward_layernorm)
                _patch_rms_norm_module_for_gemma2(decoder_layer.post_feedforward_layernorm)


def apply_liger_kernel_to_gemma3_text(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    geglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Gemma3

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
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.gemma3 import modeling_gemma3
    from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
    from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel

    from liger_kernel.transformers.model.gemma3 import causal_forward
    from liger_kernel.transformers.rms_norm import LigerRMSNormForGemma3

    _patch_rms_norm_module_for_gemma3 = partial(
        _patch_rms_norm_module, offset=1.0, casting_mode="gemma", in_place=False
    )

    if rope:
        modeling_gemma3.apply_rotary_pos_emb = liger_rotary_pos_emb

    if rms_norm:
        modeling_gemma3.Gemma3RMSNorm = LigerRMSNormForGemma3

    if geglu:
        modeling_gemma3.Gemma3MLP = LigerGEGLUMLP

    # Handle loss function
    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(causal_forward, model)
        else:
            modeling_gemma3.Gemma3ForCausalLM.forward = causal_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        if isinstance(model, Gemma3ForCausalLM) or isinstance(model, Gemma3TextModel):
            # get the base model from the model instance
            base_model = model.model if isinstance(model, Gemma3ForCausalLM) else model

            if rms_norm:
                _patch_rms_norm_module_for_gemma3(base_model.norm)

            for decoder_layer in base_model.layers:
                decoder_layer: Gemma3DecoderLayer
                if geglu:
                    _bind_method_to_module(decoder_layer.mlp, "forward", LigerGEGLUMLP.forward)
                if rms_norm:
                    _patch_rms_norm_module_for_gemma3(decoder_layer.input_layernorm)
                    _patch_rms_norm_module_for_gemma3(decoder_layer.post_attention_layernorm)
                    _patch_rms_norm_module_for_gemma3(decoder_layer.pre_feedforward_layernorm)
                    _patch_rms_norm_module_for_gemma3(decoder_layer.post_feedforward_layernorm)
                    _patch_rms_norm_module_for_gemma3(decoder_layer.self_attn.q_norm)
                    _patch_rms_norm_module_for_gemma3(decoder_layer.self_attn.k_norm)

        else:
            raise TypeError("The model must be Gemma3ForCausalLM.")


def apply_liger_kernel_to_gemma3(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    layer_norm: bool = True,
    rms_norm: bool = True,
    geglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Gemma3

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        layer_norm (bool): Whether to apply Liger's LayerNorm. Default is True.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        geglu (bool): Whether to apply Liger's GeGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.gemma3 import modeling_gemma3
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration
    from transformers.models.siglip import modeling_siglip
    from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel

    from liger_kernel.transformers.model.gemma3 import multimodal_forward

    _patch_rms_norm_module_for_gemma3 = partial(
        _patch_rms_norm_module, offset=1.0, casting_mode="gemma", in_place=False
    )

    if layer_norm and model is None:
        modeling_siglip.nn.LayerNorm = LigerLayerNorm

    apply_liger_kernel_to_gemma3_text(
        rope=rope, cross_entropy=False, fused_linear_cross_entropy=False, rms_norm=rms_norm, geglu=geglu
    )

    if cross_entropy:
        modeling_gemma3.nn.CrossEntropyLoss = LigerCrossEntropyLoss

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(multimodal_forward, model)
        else:
            modeling_gemma3.Gemma3ForConditionalGeneration.forward = multimodal_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        if isinstance(model, Gemma3ForConditionalGeneration):
            if isinstance(model.model.vision_tower, SiglipVisionModel):
                vision_tower = model.model.vision_tower

                _patch_layer_norm_module(vision_tower.vision_model.post_layernorm)

                for layer in vision_tower.vision_model.encoder.layers:
                    layer: SiglipEncoderLayer
                    if layer_norm:
                        _patch_layer_norm_module(layer.layer_norm1)
                        _patch_layer_norm_module(layer.layer_norm2)
            else:
                raise TypeError("The vision tower must be SiglipVisionModel")

            if rms_norm:
                _patch_rms_norm_module_for_gemma3(model.model.multi_modal_projector.mm_soft_emb_norm)

            apply_liger_kernel_to_gemma3_text(
                rope=rope,
                cross_entropy=False,
                fused_linear_cross_entropy=False,
                rms_norm=rms_norm,
                geglu=geglu,
                model=model.model.language_model,
            )

        else:
            raise TypeError("The model must be Gemma3ForConditionalGeneration.")


def apply_liger_kernel_to_gemma4_text(
    rope: bool = False,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    geglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Gemma4
    text models (Gemma4ForCausalLM / Gemma4TextForCausalLM / Gemma4TextModel).

    Primary target: Gemma 4 31B. The 31B config disables PLE
    (hidden_size_per_layer_input=0), MoE (enable_moe_block=false), KV sharing
    (num_kv_shared_layers=0), and double-wide MLP (use_double_wide_mlp=false),
    so every decoder layer is a plain (norm, attn, norm, mlp, norm) stack.

    Known limitation: rope kernel swap is a no-op on Gemma 4 — HF's
    apply_rotary_pos_emb takes a single tensor at a time, which is incompatible
    with Liger's (q, k, cos, sin) signature. HF's plain pytorch rope stays in
    place. The large training-memory win (fused linear cross-entropy, 16 GB
    logits tensor eliminated at seq 8192 / vocab 262144) is unaffected.

    Args:
        rope (bool): Currently a no-op for Gemma 4 (HF uses single-tensor
            apply_rotary_pos_emb incompatible with Liger). Default False.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default False.
        fused_linear_cross_entropy (bool): Fused linear CE for memory efficiency. Default True.
            Mutually exclusive with `cross_entropy`.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default True.
        geglu (bool): Whether to apply Liger's GeGLU MLP. Default True.
        model (PreTrainedModel): An already-instantiated model to patch in-place.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.gemma4 import modeling_gemma4
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    from liger_kernel.transformers.model.gemma4 import causal_forward
    from liger_kernel.transformers.rms_norm import LigerRMSNormForGemma4

    # The user's text-only extraction loads as Gemma4TextForCausalLM
    # (custom subclass, not in mainline HF). Grab it if present.
    Gemma4TextForCausalLM = getattr(modeling_gemma4, "Gemma4TextForCausalLM", None)

    # Gemma4RMSNorm uses ones-init, no +1 offset, fp32 compute.
    # offset=0.0 + casting_mode="gemma" is deliberate: the "gemma" path upcasts
    # to fp32 as HF does, and offset=0.0 yields ``w * x`` (no +1 bias) which
    # matches ones-init weight semantics.
    _patch_rms_norm_module_for_gemma4 = partial(
        _patch_rms_norm_module, offset=0.0, casting_mode="gemma", in_place=False
    )

    def _maybe_patch_scaled_norm(module):
        """Patch only Gemma4RMSNorm modules that carry a weight.

        Attention's ``v_norm`` is instantiated with ``with_scale=False`` — no
        weight exists, so Liger's weight-multiplying kernel cannot apply. We
        leave these as HF's scale-free RMSNorm (kernelized copies already
        swapped at the class level via LigerRMSNormForGemma4 which also
        handles with_scale=False correctly in its forward).
        """
        if module is None:
            return
        if not getattr(module, "with_scale", True):
            return
        _patch_rms_norm_module_for_gemma4(module)

    if rope:
        # HF's Gemma 4 apply_rotary_pos_emb has signature
        #   apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=2)
        # (single tensor at a time) whereas liger_rotary_pos_emb takes
        # (q, k, cos, sin, ...). Dropping it in raises
        # "TypeError: missing 1 required positional argument: 'sin'".
        # Until a Gemma-4-specific rope wrapper exists, leave HF's plain
        # pytorch rope in place. Large wins (RMSNorm, GEGLU, fused LCE)
        # still apply. Emit a single warning so callers flipping rope on
        # aren't silently ignored.
        logger.warning_once(
            "rope=True is currently a no-op for Gemma 4: HF's "
            "apply_rotary_pos_emb uses a single-tensor signature that is "
            "incompatible with liger_rotary_pos_emb. Skipping rope kernel swap."
        )

    if rms_norm:
        modeling_gemma4.Gemma4RMSNorm = LigerRMSNormForGemma4

    if geglu:
        # Gemma4TextMLP is constructed with (config, layer_idx); the wrapper
        # subclass accepts and discards layer_idx so the class-level swap
        # doesn't crash model construction.
        modeling_gemma4.Gemma4TextMLP = LigerGEGLUMLPForGemma4

    # Handle loss function
    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(causal_forward, model)
        else:
            modeling_gemma4.Gemma4ForCausalLM.forward = causal_forward
            # Also patch the user's custom text-only class if it's defined.
            if Gemma4TextForCausalLM is not None:
                Gemma4TextForCausalLM.forward = causal_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        causal_lm_types = tuple(cls for cls in (Gemma4ForCausalLM, Gemma4TextForCausalLM) if cls is not None)
        if isinstance(model, causal_lm_types) or isinstance(model, Gemma4TextModel):
            # get the base model from the model instance
            base_model = model.model if isinstance(model, causal_lm_types) else model

            if rms_norm:
                _maybe_patch_scaled_norm(base_model.norm)

            for decoder_layer in base_model.layers:
                decoder_layer: Gemma4TextDecoderLayer
                # Defensive: skip MLP rebind if a future variant flips MoE on.
                if geglu and not getattr(decoder_layer, "enable_moe_block", False):
                    _bind_method_to_module(decoder_layer.mlp, "forward", LigerGEGLUMLP.forward)
                if rms_norm:
                    _maybe_patch_scaled_norm(decoder_layer.input_layernorm)
                    _maybe_patch_scaled_norm(decoder_layer.post_attention_layernorm)
                    _maybe_patch_scaled_norm(decoder_layer.pre_feedforward_layernorm)
                    _maybe_patch_scaled_norm(decoder_layer.post_feedforward_layernorm)
                    # q_norm / k_norm exist on every 31B layer (num_kv_shared_layers=0)
                    # but stay defensive for future variants. v_norm is scale-free
                    # (with_scale=False) on all Gemma 4 variants so the helper
                    # intentionally leaves it untouched.
                    _maybe_patch_scaled_norm(getattr(decoder_layer.self_attn, "q_norm", None))
                    _maybe_patch_scaled_norm(getattr(decoder_layer.self_attn, "k_norm", None))
                    _maybe_patch_scaled_norm(getattr(decoder_layer.self_attn, "v_norm", None))
        else:
            raise TypeError("The model must be Gemma4ForCausalLM, Gemma4TextForCausalLM, or Gemma4TextModel.")


def apply_liger_kernel_to_paligemma(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    layer_norm: bool = True,
    rms_norm: bool = True,
    geglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace PaliGemma

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        layer_norm (bool): Whether to apply Liger's LayerNorm. Default is True.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        geglu (bool): Whether to apply Liger's GeGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    # PaliGemma submodules are ['vision_tower', 'multi_modal_projector', 'language_model']

    from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
    from transformers.models.gemma.modeling_gemma import GemmaModel
    from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM
    from transformers.models.gemma2.modeling_gemma2 import Gemma2Model
    from transformers.models.paligemma import modeling_paligemma
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
    from transformers.models.siglip import modeling_siglip
    from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel

    from liger_kernel.transformers.model.paligemma import lce_forward

    # The vision_tower is a SiglipVisionModel
    if layer_norm and model is None:
        modeling_siglip.nn.LayerNorm = LigerLayerNorm

    # SiglipMLP is standard FFN so LigerGEGLUMLP is not compatible
    # The multi_modal_projector is Linear, nothing to do

    # The language_model is GemmaForCausalLM or Gemma2ForCausalLM
    apply_liger_kernel_to_gemma(
        rope=rope, cross_entropy=False, fused_linear_cross_entropy=False, rms_norm=rms_norm, geglu=geglu
    )
    apply_liger_kernel_to_gemma2(
        rope=rope, cross_entropy=False, fused_linear_cross_entropy=False, rms_norm=rms_norm, geglu=geglu
    )
    # Handle loss function
    if cross_entropy:
        modeling_paligemma.nn.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(lce_forward, model)
        else:
            modeling_paligemma.PaliGemmaForConditionalGeneration.forward = lce_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        if not isinstance(model, PaliGemmaForConditionalGeneration):
            raise TypeError("model have to be of type PaliGemmaForConditionalGeneration")

        vision_tower: SiglipVisionModel = model.model.vision_tower

        _patch_layer_norm_module(vision_tower.vision_model.post_layernorm)

        for layer in vision_tower.vision_model.encoder.layers:
            layer: SiglipEncoderLayer
            if layer_norm:
                _patch_layer_norm_module(layer.layer_norm1)
                _patch_layer_norm_module(layer.layer_norm2)

        language_model = model.model.language_model

        if isinstance(language_model, (GemmaForCausalLM, GemmaModel)):
            apply_liger_kernel_to_gemma(
                rope=rope,
                cross_entropy=False,
                fused_linear_cross_entropy=False,
                rms_norm=rms_norm,
                geglu=geglu,
                model=language_model,
            )

        elif isinstance(language_model, (Gemma2ForCausalLM, Gemma2Model)):
            apply_liger_kernel_to_gemma2(
                rope=rope,
                cross_entropy=False,
                fused_linear_cross_entropy=False,
                rms_norm=rms_norm,
                geglu=geglu,
                model=language_model,
            )
        else:
            raise TypeError(
                "The language_model of a PaliGemma model must be either GemmaForCausalLM or Gemma2ForCausalLM."
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
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.qwen2 import modeling_qwen2
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

    if rope:
        modeling_qwen2.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_qwen2.Qwen2RMSNorm = LigerRMSNorm

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(qwen2_lce_forward, model)
        else:
            modeling_qwen2.Qwen2ForCausalLM.forward = qwen2_lce_forward

    if swiglu:
        modeling_qwen2.Qwen2MLP = LigerSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: Qwen2Model = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


def apply_liger_kernel_to_qwen3(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen3 models.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    from liger_kernel.transformers.model.qwen3 import lce_forward as qwen3_lce_forward

    if rope:
        modeling_qwen3.apply_rotary_pos_emb = liger_rotary_pos_emb

    if rms_norm:
        modeling_qwen3.Qwen3RMSNorm = LigerRMSNorm

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(qwen3_lce_forward, model)
        else:
            modeling_qwen3.Qwen3ForCausalLM.forward = qwen3_lce_forward

    if swiglu:
        modeling_qwen3.Qwen3MLP = LigerSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: Qwen3Model = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)
        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


def apply_liger_kernel_to_qwen3_moe(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen3 models.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.qwen3_moe import modeling_qwen3_moe
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel

    from liger_kernel.transformers.model.qwen3_moe import lce_forward as qwen3_lce_forward
    from liger_kernel.transformers.swiglu import LigerQwen3MoeSwiGLUMLP

    if rope:
        modeling_qwen3_moe.apply_rotary_pos_emb = liger_rotary_pos_emb

    if rms_norm:
        modeling_qwen3_moe.Qwen3MoeRMSNorm = LigerRMSNorm

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(qwen3_lce_forward, model)
        else:
            modeling_qwen3_moe.Qwen3MoeForCausalLM.forward = qwen3_lce_forward

    if swiglu:
        if IS_TRANSFORMERS_V5_OR_LATER:
            modeling_qwen3_moe.Qwen3MoeExperts = LigerExperts
        else:
            modeling_qwen3_moe.Qwen3MoeMLP = LigerQwen3MoeSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: Qwen3MoeModel = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)
        for decoder_layer in base_model.layers:
            if swiglu:
                if IS_TRANSFORMERS_V5_OR_LATER:
                    _patch_swiglu_module(decoder_layer.mlp.experts, LigerExperts)
                else:
                    for mlp_expert in decoder_layer.mlp.experts:
                        _patch_swiglu_module(mlp_expert, LigerQwen3MoeSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


def apply_liger_kernel_to_gpt_oss(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = False,  # Set to False by default since GPT-OSS has custom expert implementation
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace GPT-OSS models.
    NOTE: GPT-OSS is supported in transformers >= 4.55.0
    NOTE: SwiGLU patching is disabled by default for GPT-OSS as it uses a custom expert
          implementation with clamping and MXFP4 quantization.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is False.
            Note: GPT-OSS uses a custom expert implementation, so SwiGLU patching is disabled by default.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.
    """
    if version.parse(transformers.__version__) < version.parse("4.55.0"):
        logger.warning("GPT-OSS support requires transformers >= 4.55.0")
        return

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.gpt_oss import modeling_gpt_oss
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssModel

    if rope:
        modeling_gpt_oss.apply_rotary_pos_emb = liger_rotary_pos_emb

    if rms_norm:
        modeling_gpt_oss.GptOssRMSNorm = LigerRMSNorm

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(gpt_oss_lce_forward, model)
        else:
            modeling_gpt_oss.GptOssForCausalLM.forward = gpt_oss_lce_forward

    # Note: SwiGLU patching is not implemented for GPT-OSS due to custom expert implementation
    # with clamping (swiglu_limit=7.0) and MXFP4 quantization

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: GptOssModel = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)
        for decoder_layer in base_model.layers:
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


def apply_liger_kernel_to_qwen2_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    layer_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2-VL models.
    NOTE: Qwen2-VL is not supported in transformers<4.52.4

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
    if transformer_version < version.parse("4.52.4"):
        logger.warning("Qwen2-VL support is only compatible with transformers >= 4.52.4")
        return

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.qwen2_vl import modeling_qwen2_vl
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLTextModel

    from liger_kernel.transformers.model.qwen2_vl import lce_forward as qwen2_vl_lce_forward

    if rope:
        modeling_qwen2_vl.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb
    if rms_norm:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L439
        modeling_qwen2_vl.Qwen2RMSNorm = LigerRMSNorm
    if layer_norm and model is None:
        modeling_qwen2_vl.LayerNorm = LigerLayerNorm
    if cross_entropy:
        modeling_qwen2_vl.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(qwen2_vl_lce_forward, model)
        else:
            modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = qwen2_vl_lce_forward
    if swiglu:
        modeling_qwen2_vl.Qwen2MLP = LigerSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        if isinstance(model, Qwen2VLForConditionalGeneration):
            text_model: Qwen2VLTextModel = model.model.language_model
            vision_model: Qwen2VisionTransformerPretrainedModel = model.model.visual
        elif isinstance(model, Qwen2VLModel):
            text_model: Qwen2VLTextModel = model.language_model
            vision_model: Qwen2VisionTransformerPretrainedModel = model.visual
        elif isinstance(model, Qwen2VLTextModel):
            text_model: Qwen2VLTextModel = model
            vision_model = None
        else:
            # Note: Currently there's no support for patching vision model only. Feel free to raise an issue if needed.
            raise TypeError(
                f"Unsupported Qwen2VL model type. `model` must be `Qwen2VLForConditionalGeneration`, `Qwen2VLModel` or `Qwen2VLTextModel`. Got: {type(model)}"
            )

        # Patch Qwen2VisionTransformerPretrainedModel
        if vision_model is not None:
            for vision_block in vision_model.blocks:
                if layer_norm:
                    _patch_layer_norm_module(vision_block.norm1)
                    _patch_layer_norm_module(vision_block.norm2)

        # Patch Qwen2VisionTextModel
        if text_model is not None:
            if rms_norm:
                _patch_rms_norm_module(text_model.norm)
            for decoder_layer in text_model.layers:
                if swiglu:
                    _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
                if rms_norm:
                    _patch_rms_norm_module(decoder_layer.input_layernorm)
                    _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


def apply_liger_kernel_to_qwen2_5_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2.5-VL models.
    NOTE: Qwen2.5-VL is not available in transformers<4.48.2

    Args:
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
    if transformer_version < version.parse("4.52.4"):
        logger.warning("Qwen2.5-VL support is only compatible with transformers >= 4.52.4")
        return

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

    from liger_kernel.transformers.model.qwen2_5_vl import lce_forward as qwen2_5_vl_lce_forward

    if rope:
        modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb
    if rms_norm:
        modeling_qwen2_5_vl.Qwen2RMSNorm = LigerRMSNorm
    if cross_entropy:
        modeling_qwen2_5_vl.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(qwen2_5_vl_lce_forward, model)
        else:
            modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_vl_lce_forward
    if swiglu:
        modeling_qwen2_5_vl.Qwen2MLP = LigerSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        if isinstance(model, Qwen2_5_VLForConditionalGeneration):
            text_model: Qwen2_5_VLTextModel = model.model.language_model
            vision_model: Qwen2_5_VisionTransformerPretrainedModel = model.model.visual
        elif isinstance(model, Qwen2_5_VLModel):
            text_model: Qwen2_5_VLTextModel = model.language_model
            vision_model: Qwen2_5_VisionTransformerPretrainedModel = model.visual
        elif isinstance(model, Qwen2_5_VLTextModel):
            text_model: Qwen2_5_VLTextModel = model
            vision_model = None
        else:
            # Note: Currently there's no support for patching vision model only. Feel free to raise an issue if needed.
            raise TypeError(
                f"Unsupported Qwen2VL model type. `model` must be `Qwen2VLForConditionalGeneration`, `Qwen2VLModel` or `Qwen2VLTextModel`. Got: {type(model)}"
            )

        if vision_model is not None:
            # Patch Qwen2_5_VisionTransformerPretrainedModel
            for vision_block in vision_model.blocks:
                if rms_norm:
                    _patch_rms_norm_module(vision_block.norm1)
                    _patch_rms_norm_module(vision_block.norm2)

        if text_model is not None:
            if rms_norm:
                _patch_rms_norm_module(text_model.norm)
            for decoder_layer in text_model.layers:
                if swiglu:
                    _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
                if rms_norm:
                    _patch_rms_norm_module(decoder_layer.input_layernorm)
                    _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


def apply_liger_kernel_to_qwen3_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen3-VL models.

    Args:
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

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.qwen3_vl import modeling_qwen3_vl
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    from liger_kernel.transformers.model.qwen3_vl import lce_forward as qwen3_vl_lce_forward

    if rope:
        modeling_qwen3_vl.apply_rotary_pos_emb = liger_rotary_pos_emb
        modeling_qwen3_vl.apply_rotary_pos_emb_vision = liger_rotary_pos_emb_vision

    if rms_norm:
        modeling_qwen3_vl.Qwen3VLTextRMSNorm = LigerRMSNorm

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(qwen3_vl_lce_forward, model)
        else:
            modeling_qwen3_vl.Qwen3VLForConditionalGeneration.forward = qwen3_vl_lce_forward

    if swiglu:
        modeling_qwen3_vl.Qwen3VLTextMLP = LigerSwiGLUMLP

    if model is not None:
        if isinstance(model, Qwen3VLForConditionalGeneration):
            text_model: Qwen3VLTextModel = model.model.language_model
        elif isinstance(model, Qwen3VLModel):
            text_model: Qwen3VLTextModel = model.language_model
        elif isinstance(model, Qwen3VLTextModel):
            text_model = model
        else:
            raise TypeError(
                f"Unsupported Qwen3VL model type. `model` must be `Qwen3VLForConditionalGeneration`, `Qwen3VLModel` or `Qwen3VLTextModel`. Got: {type(model)}"
            )

        _patch_qwen3_vl_rms_norm = partial(_patch_rms_norm_module, offset=0.0, casting_mode="llama")

        if text_model is not None:
            if rms_norm:
                _patch_qwen3_vl_rms_norm(text_model.norm)
            for decoder_layer in text_model.layers:
                if swiglu:
                    _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
                if rms_norm:
                    _patch_qwen3_vl_rms_norm(decoder_layer.input_layernorm)
                    _patch_qwen3_vl_rms_norm(decoder_layer.post_attention_layernorm)
                    self_attn = getattr(decoder_layer, "self_attn", None)
                    if self_attn is not None:
                        if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
                            _patch_qwen3_vl_rms_norm(self_attn.q_norm)
                        if hasattr(self_attn, "k_norm") and self_attn.k_norm is not None:
                            _patch_qwen3_vl_rms_norm(self_attn.k_norm)


def apply_liger_kernel_to_qwen3_vl_moe(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen3-VL MoE models.

    Args:
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is False.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeModel
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextModel

    from liger_kernel.transformers.model.qwen3_vl_moe import lce_forward as qwen3_vl_moe_lce_forward

    if rope:
        modeling_qwen3_vl_moe.apply_rotary_pos_emb = liger_rotary_pos_emb
        modeling_qwen3_vl_moe.apply_rotary_pos_emb_vision = liger_rotary_pos_emb_vision

    if rms_norm:
        modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm = LigerRMSNorm

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(qwen3_vl_moe_lce_forward, model)
        else:
            modeling_qwen3_vl_moe.Qwen3VLMoeForConditionalGeneration.forward = qwen3_vl_moe_lce_forward

    if swiglu:
        if IS_TRANSFORMERS_V5_OR_LATER:
            modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts = LigerExperts

    if model is not None:
        if isinstance(model, Qwen3VLMoeForConditionalGeneration):
            text_model: Qwen3VLMoeTextModel = model.model.language_model
        elif isinstance(model, Qwen3VLMoeModel):
            text_model: Qwen3VLMoeTextModel = model.language_model
        elif isinstance(model, Qwen3VLMoeTextModel):
            text_model = model
        else:
            raise TypeError(
                f"Unsupported Qwen3VLMoe model type. `model` must be `Qwen3VLMoeForConditionalGeneration`, `Qwen3VLMoeModel` or `Qwen3VLMoeTextModel`. Got: {type(model)}"
            )

        _patch_qwen3_vl_moe_rms_norm = partial(_patch_rms_norm_module, offset=0.0, casting_mode="llama")

        if text_model is not None:
            if rms_norm:
                _patch_qwen3_vl_moe_rms_norm(text_model.norm)
            for decoder_layer in text_model.layers:
                if rms_norm:
                    _patch_qwen3_vl_moe_rms_norm(decoder_layer.input_layernorm)
                    _patch_qwen3_vl_moe_rms_norm(decoder_layer.post_attention_layernorm)
                    self_attn = getattr(decoder_layer, "self_attn", None)
                    if self_attn is not None:
                        if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
                            _patch_qwen3_vl_moe_rms_norm(self_attn.q_norm)
                        if hasattr(self_attn, "k_norm") and self_attn.k_norm is not None:
                            _patch_qwen3_vl_moe_rms_norm(self_attn.k_norm)
                if swiglu:
                    experts = getattr(decoder_layer.mlp, "experts", None)
                    if experts is not None:
                        if IS_TRANSFORMERS_V5_OR_LATER:
                            _patch_swiglu_module(experts, LigerExperts)


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
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.phi3 import modeling_phi3
    from transformers.models.phi3.modeling_phi3 import Phi3Model

    if rope:
        modeling_phi3.apply_rotary_pos_emb = liger_rotary_pos_emb  # Same as Gemma
    if rms_norm:
        modeling_phi3.Phi3RMSNorm = LigerRMSNorm  # Same as Llama
    if swiglu:
        modeling_phi3.Phi3MLP = LigerPhi3SwiGLUMLP
    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(phi3_lce_forward, model)
        else:
            modeling_phi3.Phi3ForCausalLM.forward = phi3_lce_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: Phi3Model = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerPhi3SwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


def apply_liger_kernel_to_olmo2(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace OLMO2 models.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU Olmo2MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.olmo2 import modeling_olmo2
    from transformers.models.olmo2.modeling_olmo2 import Olmo2Model

    from liger_kernel.transformers.model.olmo2 import lce_forward as olmo2_lce_forward
    from liger_kernel.transformers.rms_norm import LigerRMSNormForOlmo2

    if rope:
        modeling_olmo2.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_olmo2.Olmo2RMSNorm = LigerRMSNormForOlmo2
    if swiglu:
        modeling_olmo2.Olmo2MLP = LigerSwiGLUMLP
    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(olmo2_lce_forward, model)
        else:
            modeling_olmo2.Olmo2ForCausalLM.forward = olmo2_lce_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: Olmo2Model = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm, in_place=False)
                _patch_rms_norm_module(decoder_layer.post_feedforward_layernorm, in_place=False)


def apply_liger_kernel_to_olmo3(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Olmo3 models.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU to Olmo3MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.olmo3 import modeling_olmo3
    from transformers.models.olmo3.modeling_olmo3 import Olmo3Model

    from liger_kernel.transformers.model.olmo3 import lce_forward as olmo3_lce_forward
    from liger_kernel.transformers.rms_norm import LigerRMSNormForOlmo2

    # Olmo3 arch is very similar to Olmo2, so we can reuse all these components in the same way.
    if rope:
        modeling_olmo3.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_olmo3.Olmo3RMSNorm = LigerRMSNormForOlmo2  # same as olmo2
    if swiglu:
        modeling_olmo3.Olmo3MLP = LigerSwiGLUMLP
    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(olmo3_lce_forward, model)
        else:
            modeling_olmo3.Olmo3ForCausalLM.forward = olmo3_lce_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: Olmo3Model = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm, in_place=False)
                _patch_rms_norm_module(decoder_layer.post_feedforward_layernorm, in_place=False)


def apply_liger_kernel_to_glm4(
    rope: bool = False,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace GLM-4 models.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is False.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU Glm4MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.glm4 import modeling_glm4
    from transformers.models.glm4.modeling_glm4 import Glm4Model

    from liger_kernel.transformers.model.glm4 import lce_forward as glm4_lce_forward
    from liger_kernel.transformers.rms_norm import LigerRMSNormForGlm4

    if rope:
        raise NotImplementedError("liger_rotary_pos_emb is not available for Glm4 models.")
    if rms_norm:
        modeling_glm4.Glm4RMSNorm = LigerRMSNormForGlm4
    if swiglu:
        modeling_glm4.Glm4MLP = LigerPhi3SwiGLUMLP
    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(glm4_lce_forward, model)
        else:
            modeling_glm4.Glm4ForCausalLM.forward = glm4_lce_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: Glm4Model = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm, in_place=False)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerPhi3SwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm, in_place=False)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm, in_place=False)
                _patch_rms_norm_module(decoder_layer.post_self_attn_layernorm, in_place=False)
                _patch_rms_norm_module(decoder_layer.post_mlp_layernorm, in_place=False)


def apply_liger_kernel_to_glm4v(
    rope: bool = False,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace GLM-4v models.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is False.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU Glm4MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.glm4v import modeling_glm4v
    from transformers.models.glm4v.modeling_glm4v import Glm4vForConditionalGeneration
    from transformers.models.glm4v.modeling_glm4v import Glm4vModel
    from transformers.models.glm4v.modeling_glm4v import Glm4vTextModel
    from transformers.models.glm4v.modeling_glm4v import Glm4vVisionModel

    from liger_kernel.transformers.model.glm4v import lce_forward as glm4v_lce_forward
    from liger_kernel.transformers.rms_norm import LigerRMSNormForGlm4

    if rope:
        raise NotImplementedError("liger_rotary_pos_emb is not available for Glm4 models.")
    if rms_norm:
        modeling_glm4v.Glm4vRMSNorm = LigerRMSNormForGlm4
    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(glm4v_lce_forward, model)
        else:
            modeling_glm4v.Glm4vForConditionalGeneration.forward = glm4v_lce_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        if isinstance(model, Glm4vForConditionalGeneration):
            text_model: Glm4vTextModel = model.model.language_model
            vision_model: Glm4vVisionModel = model.model.visual
        elif isinstance(model, Glm4vModel):
            text_model: Glm4vTextModel = model.language_model
            vision_model: Glm4vVisionModel = model.visual
        elif isinstance(model, Glm4vTextModel):
            text_model: Glm4vTextModel = model
            vision_model = None
        else:
            # Note: Currently there's no support for patching vision model only. Feel free to raise an issue if needed.
            raise TypeError(
                f"Unsupported glm4.1v model type. `model` must be `Glm4VLForConditionalGeneration`, `Glm4vVisionModel` or `Glm4vTextModel`. Got: {type(model)}"
            )

        if vision_model is not None:
            for vision_block in vision_model.blocks:
                if rms_norm:
                    _patch_rms_norm_module(vision_block.norm1)
                    _patch_rms_norm_module(vision_block.norm2)
                if swiglu:
                    _patch_swiglu_module(vision_block.mlp, LigerSwiGLUMLP)

        if text_model is not None:
            if rms_norm:
                _patch_rms_norm_module(text_model.norm)
            for decoder_layer in text_model.layers:
                if swiglu:
                    _patch_swiglu_module(decoder_layer.mlp, LigerPhi3SwiGLUMLP)
                if rms_norm:
                    _patch_rms_norm_module(decoder_layer.input_layernorm)
                    _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
                    _patch_rms_norm_module(decoder_layer.post_self_attn_layernorm)
                    _patch_rms_norm_module(decoder_layer.post_mlp_layernorm)


def apply_liger_kernel_to_glm4v_moe(
    rope: bool = False,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace GLM4v_moe models.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is False.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLUMLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.glm4v_moe import modeling_glm4v_moe
    from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeForConditionalGeneration
    from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeModel
    from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeTextModel
    from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeVisionModel

    from liger_kernel.transformers.model.glm4v_moe import lce_forward as glm4v_moe_lce_forward
    from liger_kernel.transformers.rms_norm import LigerRMSNormForGlm4

    if rope:
        raise NotImplementedError("liger_rotary_pos_emb is not available for Glm4 models.")
    if rms_norm:
        modeling_glm4v_moe.Glm4vMoeRMSNorm = LigerRMSNormForGlm4
        modeling_glm4v_moe.Glm4vMoeTextRMSNorm = LigerRMSNormForGlm4
    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(glm4v_moe_lce_forward, model)
        else:
            modeling_glm4v_moe.Glm4vMoeForConditionalGeneration.forward = glm4v_moe_lce_forward
    if swiglu:
        if IS_TRANSFORMERS_V5_OR_LATER:
            modeling_glm4v_moe.Glm4vMoeTextNaiveMoe = LigerExperts

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        if isinstance(model, Glm4vMoeForConditionalGeneration):
            text_model: Glm4vMoeTextModel = model.model.language_model
            vision_model: Glm4vMoeVisionModel = model.model.visual
        elif isinstance(model, Glm4vMoeModel):
            text_model: Glm4vMoeTextModel = model.language_model
            vision_model: Glm4vMoeVisionModel = model.visual
        elif isinstance(model, Glm4vMoeTextModel):
            text_model: Glm4vMoeTextModel = model
            vision_model = None
        else:
            # Note: Currently there's no support for patching vision model only. Feel free to raise an issue if needed.
            raise TypeError(
                f"Unsupported glm4v_moe model type. `model` must be `Glm4vMoeForConditionalGeneration`, `Glm4vMoeVisionModel` or `Glm4vMoeTextModel`. Got: {type(model)}"
            )

        if vision_model is not None:
            _patch_rms_norm_module(vision_model.post_conv_layernorm)
            _patch_rms_norm_module(vision_model.post_layernorm)
            for vision_block in vision_model.blocks:
                if rms_norm:
                    _patch_rms_norm_module(vision_block.norm1)
                    _patch_rms_norm_module(vision_block.norm2)
                if swiglu:
                    _patch_swiglu_module(vision_block.mlp, LigerSwiGLUMLP)

        if text_model is not None:
            if rms_norm:
                _patch_rms_norm_module(text_model.norm)
            for decoder_layer in text_model.layers:
                if rms_norm:
                    _patch_rms_norm_module(decoder_layer.input_layernorm)
                    _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
                if swiglu:
                    experts = getattr(decoder_layer.mlp, "experts", None)
                    if experts is not None:
                        if IS_TRANSFORMERS_V5_OR_LATER:
                            _patch_swiglu_module(experts, LigerExperts)
                        else:
                            for expert in experts:
                                _patch_swiglu_module(expert, LigerSwiGLUMLP)
                        shared_experts = getattr(decoder_layer.mlp, "shared_experts", None)
                        if shared_experts is not None:
                            _patch_swiglu_module(shared_experts, LigerSwiGLUMLP)
                    else:
                        _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)


def apply_liger_kernel_to_internvl(
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    layer_norm: bool = True,
    model: Optional[PreTrainedModel] = None,
    **kwargs,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace InternVL models.
    Due to the characteristics of InternVL, the model must be passed to apply Liger-Kernel's patch to other models connected to InternVL.
    However, if an LM not supported by Liger-Kernel is connected to InternVL, unexpected side effects may occur.
    NOTE: InternVL is not available in transformers<4.52.1

    Args:
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        layer_norm (bool): Whether to apply Liger's LayerNorm. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )
    import torch.nn as torch_nn

    from transformers.models.internvl import modeling_internvl
    from transformers.models.internvl.modeling_internvl import InternVLForConditionalGeneration
    from transformers.models.internvl.modeling_internvl import InternVLModel
    from transformers.models.internvl.modeling_internvl import InternVLVisionLayer
    from transformers.models.internvl.modeling_internvl import InternVLVisionModel
    from transformers.models.internvl.modeling_internvl import InternVLVisionRMSNorm

    from liger_kernel.transformers.layer_norm import LigerLayerNorm
    from liger_kernel.transformers.model.internvl import lce_forward as internvl_lce_forward
    from liger_kernel.transformers.rms_norm import LigerRMSNorm

    if layer_norm and model is None:
        modeling_internvl.nn.LayerNorm = LigerLayerNorm

    if cross_entropy:
        logger.info("Apply liger cross entropy")

        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy
    if fused_linear_cross_entropy:
        modeling_internvl.InternVLForConditionalGeneration.forward = internvl_lce_forward
    if rms_norm:
        modeling_internvl.InternVLVisionRMSNorm = LigerRMSNorm

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        if isinstance(model, InternVLForConditionalGeneration):
            text_model = model.model.language_model
            vision_model: InternVLVisionModel = model.model.vision_tower
        elif isinstance(model, InternVLModel):
            text_model = model.language_model
            vision_model: InternVLVisionModel = model.vision_tower
        else:
            raise TypeError(
                f"Unsupported internvl model type. `model` must be `InternVLForConditionalGeneration`, `InternVLModel`. Got: {type(model)}"
            )

        text_model_name = model.config.text_config.model_type
        text_liger_fn = MODEL_TYPE_TO_APPLY_LIGER_FN.get(text_model_name, None)

        kwargs = {"cross_entropy": False, "fused_linear_cross_entropy": False, **kwargs} | {"rms_norm": rms_norm}
        if text_liger_fn:
            accept_params = inspect.signature(text_liger_fn).parameters
            remain_params = set(kwargs) - (set(accept_params) & set(kwargs))
            text_kwargs = {k: v for k, v in kwargs.items() if k not in remain_params}

            if remain_params:
                logger.warning(
                    f"These parameters are not supported by {text_model_name}. Enter the remaining {list(text_kwargs.keys())} except for {list(remain_params)}\n"
                    f"Parameters accepted by {text_model_name}: {list(accept_params.keys())}"
                )
            text_kwargs["model"] = text_model
            text_liger_fn(**text_kwargs)
        elif text_model_name not in MODEL_TYPE_TO_APPLY_LIGER_FN:
            logger.warning(f"{text_model_name} is not supported by Liger kernel.")

        # Patch vision model RMSNorm layers
        if rms_norm:
            for encoder_layer in vision_model.encoder.layer:
                encoder_layer: InternVLVisionLayer
                if isinstance(encoder_layer.attention.q_norm, InternVLVisionRMSNorm):
                    _patch_rms_norm_module(encoder_layer.attention.q_norm)
                if isinstance(encoder_layer.attention.k_norm, InternVLVisionRMSNorm):
                    _patch_rms_norm_module(encoder_layer.attention.k_norm)

        # Patch vision model LayerNorm layers
        if layer_norm:
            # Patch layernorm
            if isinstance(vision_model.layernorm, torch_nn.LayerNorm):
                _patch_layer_norm_module(vision_model.layernorm)

            # Patch encoder layers
            for encoder_layer in vision_model.encoder.layer:
                encoder_layer: InternVLVisionLayer
                if isinstance(encoder_layer.layernorm_before, torch_nn.LayerNorm):
                    _patch_layer_norm_module(encoder_layer.layernorm_before)
                if isinstance(encoder_layer.layernorm_after, torch_nn.LayerNorm):
                    _patch_layer_norm_module(encoder_layer.layernorm_after)


def apply_liger_kernel_to_smolvlm(
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    layer_norm: bool = True,
    model: Optional[PreTrainedModel] = None,
    **kwargs,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace SmolVLM models.
    Due to the characteristics of SmolVLM, the model must be passed to apply Liger-Kernel's patch to other models connected to SmolVLM.
    However, if an LM not supported by Liger-Kernel is connected to SmolVLM, unexpected side effects may occur.
    NOTE: SmolVLM is not available in transformers<4.50.0

    Args:
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        layer_norm (bool): Whether to apply Liger's LayerNorm. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.smolvlm import modeling_smolvlm
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMEncoderLayer
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMForConditionalGeneration
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMModel
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMVisionTransformer

    from liger_kernel.transformers.model.smolvlm import lce_forward as smolvlm_lce_forward

    # Patch LayerNorm for vision model if model is not provided (pre-initialization)
    if layer_norm and model is None:
        modeling_smolvlm.nn.LayerNorm = LigerLayerNorm

    if cross_entropy:
        logger.info("Apply liger cross entropy")

        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy
    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(smolvlm_lce_forward, model)
        else:
            modeling_smolvlm.SmolVLMForConditionalGeneration.forward = smolvlm_lce_forward
    if rms_norm:
        modeling_smolvlm.SmolVLMRMSNorm = LigerRMSNorm

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        if isinstance(model, SmolVLMForConditionalGeneration):
            text_model = model.model.text_model
            vision_model: SmolVLMVisionTransformer = model.model.vision_model
        elif isinstance(model, SmolVLMModel):
            text_model = model.text_model
            vision_model: SmolVLMVisionTransformer = model.vision_model
        else:
            raise TypeError(
                f"Unsupported smolvlm model type. `model` must be `SmolVLMForConditionalGeneration`, `SmolVLMModel`. Got: {type(model)}"
            )

        text_model_name = model.config.text_config.model_type
        text_liger_fn = MODEL_TYPE_TO_APPLY_LIGER_FN.get(text_model_name, None)

        kwargs = {"cross_entropy": False, "fused_linear_cross_entropy": False, **kwargs} | {"rms_norm": rms_norm}
        if text_liger_fn:
            accept_params = inspect.signature(text_liger_fn).parameters
            remain_params = set(kwargs) - (set(accept_params) & set(kwargs))
            text_kwargs = {k: v for k, v in kwargs.items() if k not in remain_params}

            if remain_params:
                logger.warning(
                    f"These parameters are not supported by {text_model_name}. Enter the remaining {list(text_kwargs.keys())} except for {list(remain_params)}\n"
                    f"Parameters accepted by {text_model_name}: {list(accept_params.keys())}"
                )
            text_kwargs["model"] = text_model
            text_liger_fn(**text_kwargs)
        elif text_model_name not in MODEL_TYPE_TO_APPLY_LIGER_FN:
            logger.warning(f"{text_model_name} is not supported by Liger kernel.")

        # Patch vision model LayerNorm layers
        if layer_norm:
            # Patch post_layernorm
            _patch_layer_norm_module(vision_model.post_layernorm)

            # Patch encoder layers
            for encoder_layer in vision_model.encoder.layers:
                encoder_layer: SmolVLMEncoderLayer
                _patch_layer_norm_module(encoder_layer.layer_norm1)
                _patch_layer_norm_module(encoder_layer.layer_norm2)


def apply_liger_kernel_to_falcon_h1(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Falcon-H1 models
    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is True.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is False.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is False.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.falcon_h1 import modeling_falcon_h1
    from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1Model

    from liger_kernel.transformers.swiglu import LigerFalconH1SwiGLUMLP

    if rope:
        logger.info("Apply liger rotary pos emb.")
        modeling_falcon_h1.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        logger.info("Apply liger RMSNorm")
        modeling_falcon_h1.FalconH1RMSNorm = LigerRMSNorm
    if swiglu:
        logger.info("Apply liger SwiGLU")
        modeling_falcon_h1.FalconH1MLP = LigerFalconH1SwiGLUMLP

    if cross_entropy:
        logger.info("Apply liger cross entropy")
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(falcon_h1_lce_forward, model)
        else:
            modeling_falcon_h1.FalconH1ForCausalLM.forward = falcon_h1_lce_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules (e.g. LlamaRMSNorm or LlamaMLP)

        # get the base model from the model instance
        base_model: FalconH1Model = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.final_layernorm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.feed_forward, LigerFalconH1SwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.pre_ff_layernorm)


def apply_liger_kernel_to_qwen3_next(
    rope: bool = False,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace GLM4v_moe models.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is False.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLUMLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.qwen3_next import modeling_qwen3_next
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextMLP
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextModel
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextSparseMoeBlock

    from liger_kernel.transformers.model.qwen3_next import lce_forward as qwen3_next_lce_forward
    from liger_kernel.transformers.rms_norm import LigerRMSNormForQwen3Next
    from liger_kernel.transformers.swiglu import LigerQwen3MoeSwiGLUMLP

    if rope:
        # It might enocunter nan issue
        # modeling_qwen3_next.apply_rotary_pos_emb = liger_rotary_pos_emb
        raise NotImplementedError("liger_rotary_pos_emb is not available for Qwen3Next models.")
    if rms_norm:
        modeling_qwen3_next.Qwen3NextRMSNorm = LigerRMSNormForQwen3Next
    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy
    if fused_linear_cross_entropy:
        if model is not None:
            if isinstance(model, Qwen3NextForCausalLM):
                model.forward = MethodType(qwen3_next_lce_forward, model)
            else:
                raise TypeError(
                    f" fused_linear_cross_entropy is only applicable on Qwen3NextForCausalLM. Got: {type(model)}"
                )
        else:
            modeling_qwen3_next.Qwen3NextForCausalLM.forward = qwen3_next_lce_forward
    if swiglu:
        if IS_TRANSFORMERS_V5_OR_LATER:
            modeling_qwen3_next.Qwen3NextExperts = LigerExperts
        else:
            # Qwen3MoeMLP and Qwen3NextMLP are identical, hence we reuse LigerQwen3MoeSwiGLUMLP
            modeling_qwen3_next.Qwen3NextMLP = LigerQwen3MoeSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        if isinstance(model, (Qwen3NextForCausalLM, Qwen3NextModel)):
            base_model: Qwen3NextForCausalLM = getattr(model, model.base_model_prefix, model)
        else:
            raise TypeError(
                f"Unsupported qwen3_next model type. `model` must be `Qwen3NextForCausalLM`, `Qwen3NextModel`. Got: {type(model)}"
            )

        _patch_rms_norm_module_for_qwen3_next = partial(
            _patch_rms_norm_module, offset=1.0, casting_mode="gemma", in_place=False
        )

        if rms_norm:
            _patch_rms_norm_module_for_qwen3_next(base_model.norm)

        for decoder_layer in base_model.layers:
            if rms_norm:
                _patch_rms_norm_module_for_qwen3_next(decoder_layer.input_layernorm)
                _patch_rms_norm_module_for_qwen3_next(decoder_layer.post_attention_layernorm)

            # Qwen3MoeMLP and Qwen3NextMLP are identical, hence we reuse LigerQwen3MoeSwiGLUMLP
            if swiglu:
                if isinstance(decoder_layer.mlp, Qwen3NextMLP):
                    _patch_swiglu_module(decoder_layer.mlp, LigerQwen3MoeSwiGLUMLP)
                if isinstance(decoder_layer.mlp, Qwen3NextSparseMoeBlock):
                    _patch_swiglu_module(decoder_layer.mlp.shared_expert, LigerQwen3MoeSwiGLUMLP)
                    experts = getattr(decoder_layer.mlp, "experts", None)
                    if experts is not None:
                        if IS_TRANSFORMERS_V5_OR_LATER:
                            _patch_swiglu_module(experts, LigerExperts)
                        else:
                            for expert in experts:
                                _patch_swiglu_module(expert, LigerQwen3MoeSwiGLUMLP)


def apply_liger_kernel_to_qwen3_5(
    rope: bool = False,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen3.5 dense models.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is False.
            Not yet supported for Qwen3.5 due to hybrid attention (Gated DeltaNet + Gated Attention).
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLUMLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.qwen3_5 import modeling_qwen3_5
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    except ImportError:
        Qwen3_5ForConditionalGeneration = None

    from liger_kernel.transformers.model.qwen3_5 import lce_forward as qwen3_5_lce_forward
    from liger_kernel.transformers.model.qwen3_5 import lce_forward_for_multimodal as qwen3_5_lce_forward_for_multimodal
    from liger_kernel.transformers.monkey_patch import _patch_rms_norm_module
    from liger_kernel.transformers.monkey_patch import _patch_swiglu_module
    from liger_kernel.transformers.rms_norm import LigerRMSNormForQwen3Next
    from liger_kernel.transformers.swiglu import LigerQwen3MoeSwiGLUMLP

    if rope:
        raise NotImplementedError("liger_rotary_pos_emb is not available for Qwen3_5 models.")

    if rms_norm:
        modeling_qwen3_5.Qwen3_5RMSNorm = LigerRMSNormForQwen3Next

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        from liger_kernel.transformers.cross_entropy import liger_cross_entropy

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            if isinstance(model, Qwen3_5ForCausalLM):
                model.forward = MethodType(qwen3_5_lce_forward, model)
            elif isinstance(model, Qwen3_5ForConditionalGeneration):
                model.forward = MethodType(qwen3_5_lce_forward_for_multimodal, model)
            else:
                raise TypeError(
                    f"fused_linear_cross_entropy is only applicable on Qwen3_5ForCausalLM or Qwen3_5ForConditionalGeneration. Got: {type(model)}"
                )
        else:
            modeling_qwen3_5.Qwen3_5ForCausalLM.forward = qwen3_5_lce_forward
            if Qwen3_5ForConditionalGeneration is not None:
                modeling_qwen3_5.Qwen3_5ForConditionalGeneration.forward = qwen3_5_lce_forward_for_multimodal

    if swiglu:
        modeling_qwen3_5.Qwen3_5MLP = LigerQwen3MoeSwiGLUMLP

    if model is not None:
        if isinstance(model, (Qwen3_5ForCausalLM, Qwen3_5TextModel)):
            text_model: Qwen3_5TextModel = getattr(model, model.base_model_prefix, model)
        elif Qwen3_5ForConditionalGeneration is not None and isinstance(model, Qwen3_5ForConditionalGeneration):
            text_model = model.model.language_model
        else:
            raise TypeError(f"Unsupported qwen3_5 model type. Got: {type(model)}")

        _patch_rms_norm_module_for_qwen3_5 = partial(
            _patch_rms_norm_module, offset=1.0, casting_mode="gemma", in_place=False
        )

        if rms_norm:
            _patch_rms_norm_module_for_qwen3_5(text_model.norm)

        for decoder_layer in text_model.layers:
            if rms_norm:
                _patch_rms_norm_module_for_qwen3_5(decoder_layer.input_layernorm)
                _patch_rms_norm_module_for_qwen3_5(decoder_layer.post_attention_layernorm)

            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerQwen3MoeSwiGLUMLP)


def apply_liger_kernel_to_qwen3_5_moe(
    rope: bool = False,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen3.5 MoE models.

    Supports both the text-only (`Qwen3_5MoeForCausalLM`, `Qwen3_5MoeTextModel`) and the
    multimodal (`Qwen3_5MoeForConditionalGeneration`, `Qwen3_5MoeModel`) classes. The vision
    tower itself is not patched.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is False.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLUMLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

    from liger_kernel.transformers.model.qwen3_5_moe import lce_forward as qwen3_5_moe_lce_forward
    from liger_kernel.transformers.model.qwen3_5_moe import (
        lce_forward_conditional_generation as qwen3_5_moe_conditional_generation_lce_forward,
    )
    from liger_kernel.transformers.rms_norm import LigerRMSNormForQwen3Next
    from liger_kernel.transformers.swiglu import LigerQwen3MoeSwiGLUMLP

    if rope:
        raise NotImplementedError("liger_rotary_pos_emb is not available for Qwen3_5Moe models.")
    if rms_norm:
        modeling_qwen3_5_moe.Qwen3_5MoeRMSNorm = LigerRMSNormForQwen3Next
    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy
    if fused_linear_cross_entropy:
        if model is not None:
            if isinstance(model, Qwen3_5MoeForCausalLM):
                model.forward = MethodType(qwen3_5_moe_lce_forward, model)
            elif isinstance(model, Qwen3_5MoeForConditionalGeneration):
                model.forward = MethodType(qwen3_5_moe_conditional_generation_lce_forward, model)
            # Qwen3_5MoeModel / Qwen3_5MoeTextModel have no LM head — RMSNorm / SwiGLU
            # patches below still apply, but FLCE is a no-op.
        else:
            modeling_qwen3_5_moe.Qwen3_5MoeForCausalLM.forward = qwen3_5_moe_lce_forward
            modeling_qwen3_5_moe.Qwen3_5MoeForConditionalGeneration.forward = (
                qwen3_5_moe_conditional_generation_lce_forward
            )
    if swiglu:
        modeling_qwen3_5_moe.Qwen3_5MoeExperts = LigerExperts

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        if isinstance(model, Qwen3_5MoeForConditionalGeneration):
            base_model: Qwen3_5MoeTextModel = model.model.language_model
        elif isinstance(model, Qwen3_5MoeModel):
            base_model: Qwen3_5MoeTextModel = model.language_model
        elif isinstance(model, (Qwen3_5MoeForCausalLM, Qwen3_5MoeTextModel)):
            base_model: Qwen3_5MoeTextModel = getattr(model, model.base_model_prefix, model)
        else:
            raise TypeError(
                "Unsupported qwen3_5_moe model type. `model` must be `Qwen3_5MoeForConditionalGeneration`, "
                "`Qwen3_5MoeModel`, `Qwen3_5MoeForCausalLM` or `Qwen3_5MoeTextModel`. "
                f"Got: {type(model)}"
            )

        _patch_rms_norm_module_for_qwen3_5_moe = partial(
            _patch_rms_norm_module, offset=1.0, casting_mode="gemma", in_place=False
        )

        if rms_norm:
            _patch_rms_norm_module_for_qwen3_5_moe(base_model.norm)

        for decoder_layer in base_model.layers:
            if rms_norm:
                _patch_rms_norm_module_for_qwen3_5_moe(decoder_layer.input_layernorm)
                _patch_rms_norm_module_for_qwen3_5_moe(decoder_layer.post_attention_layernorm)

            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp.shared_expert, LigerQwen3MoeSwiGLUMLP)
                experts = getattr(decoder_layer.mlp, "experts", None)
                if experts is not None:
                    _patch_swiglu_module(experts, LigerExperts)


def apply_liger_kernel_to_hunyuan_v1_dense(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Hunyuan v1 dense models.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.hunyuan_v1_dense import modeling_hunyuan_v1_dense
    from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import HunYuanDenseV1Model

    from liger_kernel.transformers.model.hunyuan_v1 import lce_forward as hunyuan_v1_lce_forward
    from liger_kernel.transformers.swiglu import LigerHunyuanV1SwiGLUMLP

    if rope:
        modeling_hunyuan_v1_dense.apply_rotary_pos_emb = liger_rotary_pos_emb

    if rms_norm:
        modeling_hunyuan_v1_dense.HunYuanDenseV1RMSNorm = LigerRMSNorm

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(hunyuan_v1_lce_forward, model)
        else:
            modeling_hunyuan_v1_dense.HunYuanDenseV1ForCausalLM.forward = hunyuan_v1_lce_forward

    if swiglu:
        modeling_hunyuan_v1_dense.HunYuanDenseV1MLP = LigerHunyuanV1SwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: HunYuanDenseV1Model = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)
        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerHunyuanV1SwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


def apply_liger_kernel_to_hunyuan_v1_moe(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen3 models.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.hunyuan_v1_moe import modeling_hunyuan_v1_moe
    from transformers.models.hunyuan_v1_moe.modeling_hunyuan_v1_moe import HunYuanMoEV1Model

    from liger_kernel.transformers.model.hunyuan_v1 import lce_forward as hunyuan_v1_moe_lce_forward
    from liger_kernel.transformers.swiglu import LigerHunyuanV1SwiGLUMLP

    if rope:
        modeling_hunyuan_v1_moe.apply_rotary_pos_emb = liger_rotary_pos_emb

    if rms_norm:
        modeling_hunyuan_v1_moe.HunYuanMoEV1RMSNorm = LigerRMSNorm

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(hunyuan_v1_moe_lce_forward, model)
        else:
            modeling_hunyuan_v1_moe.HunYuanMoEV1ForCausalLM.forward = hunyuan_v1_moe_lce_forward

    if swiglu:
        if IS_TRANSFORMERS_V5_OR_LATER:
            modeling_hunyuan_v1_moe.HunYuanMoEV1Experts = LigerExperts
        else:
            modeling_hunyuan_v1_moe.HunYuanMoEV1MLP = LigerHunyuanV1SwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: HunYuanMoEV1Model = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)
        for decoder_layer in base_model.layers:
            if swiglu:
                if IS_TRANSFORMERS_V5_OR_LATER:
                    _patch_swiglu_module(decoder_layer.mlp.experts, LigerExperts)
                else:
                    for mlp_expert in decoder_layer.mlp.experts:
                        _patch_swiglu_module(mlp_expert, LigerHunyuanV1SwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


def apply_liger_kernel_to_exaone4(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace EXAONE4 models.

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
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.exaone4 import modeling_exaone4
    from transformers.models.exaone4.modeling_exaone4 import Exaone4Model

    from liger_kernel.transformers.model.exaone4 import lce_forward as exaone4_lce_forward

    if rope:
        modeling_exaone4.apply_rotary_pos_emb = liger_rotary_pos_emb

    if rms_norm:
        # EXAONE4 requires in_place=False to avoid gradient issues
        class Exaone4LigerRMSNorm(LigerRMSNorm):
            def __init__(self, hidden_size, eps=1e-6, **kwargs):
                super().__init__(hidden_size, eps, **kwargs)
                self.in_place = False

        modeling_exaone4.Exaone4RMSNorm = Exaone4LigerRMSNorm

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(exaone4_lce_forward, model)
        else:
            modeling_exaone4.Exaone4ForCausalLM.forward = exaone4_lce_forward

    if swiglu:
        modeling_exaone4.Exaone4MLP = LigerSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: Exaone4Model = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm, in_place=False)
        for decoder_layer in base_model.layers:
            if swiglu:
                _bind_method_to_module(decoder_layer.mlp, "forward", LigerSwiGLUMLP.forward)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm, in_place=False)
                _patch_rms_norm_module(decoder_layer.post_feedforward_layernorm, in_place=False)
                _patch_rms_norm_module(decoder_layer.self_attn.q_norm, in_place=False)
                _patch_rms_norm_module(decoder_layer.self_attn.k_norm, in_place=False)


# Model type corresponds to the keys defined in transformers/models/auto/modeling_auto.py
MODEL_TYPE_TO_APPLY_LIGER_FN = {
    "gemma": apply_liger_kernel_to_gemma,
    "gemma2": apply_liger_kernel_to_gemma2,
    "gemma3_text": apply_liger_kernel_to_gemma3_text,
    "gemma3": apply_liger_kernel_to_gemma3,
    "gemma4_text": apply_liger_kernel_to_gemma4_text,
    "glm4": apply_liger_kernel_to_glm4,
    "glm4v": apply_liger_kernel_to_glm4v,
    "glm4v_moe": apply_liger_kernel_to_glm4v_moe,
    "gpt_oss": apply_liger_kernel_to_gpt_oss,
    "internvl": apply_liger_kernel_to_internvl,
    "llama": apply_liger_kernel_to_llama,
    "llama4_text": apply_liger_kernel_to_llama4,
    "llama4": apply_liger_kernel_to_llama4,
    "llava": apply_liger_kernel_to_llava,
    "granite": apply_liger_kernel_to_granite,
    "mllama": apply_liger_kernel_to_mllama,
    "mllama_text_model": apply_liger_kernel_to_mllama,
    "ministral": apply_liger_kernel_to_ministral,
    "mistral": apply_liger_kernel_to_mistral,
    "mixtral": apply_liger_kernel_to_mixtral,
    "nemotron": apply_liger_kernel_to_nemotron,
    "olmo2": apply_liger_kernel_to_olmo2,
    "pixtral": apply_liger_kernel_to_pixtral,
    "olmo3": apply_liger_kernel_to_olmo3,
    "qwen2": apply_liger_kernel_to_qwen2,
    "qwen3": apply_liger_kernel_to_qwen3,
    "qwen3_moe": apply_liger_kernel_to_qwen3_moe,
    "qwen2_vl": apply_liger_kernel_to_qwen2_vl,
    "qwen2_vl_text": apply_liger_kernel_to_qwen2_vl,
    "qwen2_5_vl": apply_liger_kernel_to_qwen2_5_vl,
    "qwen2_5_vl_text": apply_liger_kernel_to_qwen2_5_vl,
    "qwen3_next": apply_liger_kernel_to_qwen3_next,
    "qwen3_5": apply_liger_kernel_to_qwen3_5,
    "qwen3_5_text": apply_liger_kernel_to_qwen3_5,
    "qwen3_5_moe": apply_liger_kernel_to_qwen3_5_moe,
    "qwen3_5_moe_text": apply_liger_kernel_to_qwen3_5_moe,
    "qwen3_vl": apply_liger_kernel_to_qwen3_vl,
    "qwen3_vl_text": apply_liger_kernel_to_qwen3_vl,
    "qwen3_vl_moe": apply_liger_kernel_to_qwen3_vl_moe,
    "qwen3_vl_moe_text": apply_liger_kernel_to_qwen3_vl_moe,
    "smollm3": apply_liger_kernel_to_smollm3,
    "phi3": apply_liger_kernel_to_phi3,
    "paligemma": apply_liger_kernel_to_paligemma,
    "falcon_h1": apply_liger_kernel_to_falcon_h1,
    "smolvlm": apply_liger_kernel_to_smolvlm,
    "hunyuan_v1_dense": apply_liger_kernel_to_hunyuan_v1_dense,
    "hunyuan_v1_moe": apply_liger_kernel_to_hunyuan_v1_moe,
    "exaone4": apply_liger_kernel_to_exaone4,
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
        logger.info(f"There are currently no Liger kernels supported for model type: {model_type}.")
        return

    apply_fn = MODEL_TYPE_TO_APPLY_LIGER_FN[model_type]
    apply_fn_signature = inspect.signature(apply_fn)

    # Filter out the keyword arguments that are not supported by the apply function
    applicable_kwargs = {key: value for key, value in kwargs.items() if key in apply_fn_signature.parameters}

    logger.info(f"Applying Liger kernels for model type: {model_type} with kwargs: {applicable_kwargs}")

    # Assume this is invoked pre-model initialization, so we only need to patch transformers code
    apply_fn(**applicable_kwargs)


def _apply_liger_kernel_to_instance(model: PreTrainedModel, **kwargs) -> None:
    """
    Applies Liger kernels to the provided model instance.

    Args:
        - model: the model instance to apply Liger kernels to
        - kwargs: keyword arguments that are passed to the corresponding apply_liger_kernel_to_* function.
    """
    model_type = getattr(model, "config", None) and getattr(model.config, "model_type", None)

    if not model_type:
        logger.info("Model type could not be determined from model config. No Liger kernels will be applied.")
        return

    if model_type not in MODEL_TYPE_TO_APPLY_LIGER_FN.keys():
        logger.info(f"There are currently no Liger kernels supported for model type: {model_type}.")
        return

    apply_fn = MODEL_TYPE_TO_APPLY_LIGER_FN[model_type]
    apply_fn_signature = inspect.signature(apply_fn)

    # Filter out the keyword arguments that are not supported by the apply function
    applicable_kwargs = {key: value for key, value in kwargs.items() if key in apply_fn_signature.parameters}
    logger.info(
        f"Applying Liger kernels to model instance with model type: {model_type} with kwargs: {applicable_kwargs}"
    )

    apply_fn(model=model, **applicable_kwargs)
