import logging

from liger_kernel.transformers.monkey_patch import (
    apply_liger_kernel_to_gemma,
    apply_liger_kernel_to_llama,
    apply_liger_kernel_to_mistral,
    apply_liger_kernel_to_mixtral,
    apply_liger_kernel_to_phi3,
)

logger = logging.getLogger(__name__)

# Model type corresponds to the keys defined in transformers/models/auto/modeling_auto.py
MODEL_TYPE_TO_APPLY_LIGER_FN = {
    "gemma": apply_liger_kernel_to_gemma,
    "llama": apply_liger_kernel_to_llama,
    "mistral": apply_liger_kernel_to_mistral,
    "mixtral": apply_liger_kernel_to_mixtral,
    "phi3": apply_liger_kernel_to_phi3,
}


def _apply_liger_kernel(model_type: str = "", **kwargs) -> None:
    """
    Applies Liger kernels based on the specified model type. The custom
    kernels for the specified model type will be applied with the provided
    keyword arguments, otherwise the default configuration will be used.

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

    logger.info(f"Applying Liger kernels for model type: {model_type}.")
    # Apply the default combination of liger kernels available for the model
    MODEL_TYPE_TO_APPLY_LIGER_FN[model_type](**kwargs)
