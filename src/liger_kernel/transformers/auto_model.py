import inspect

from transformers import AutoConfig, AutoModelForCausalLM

from liger_kernel.transformers.monkey_patch import (
    MODEL_TYPE_TO_APPLY_LIGER_FN,
    _apply_liger_kernel,
)


def _get_model_config(model_dir, **model_init_kwargs):
    config = AutoConfig.from_pretrained(model_dir, **model_init_kwargs)
    return config


class AutoLigerKernelForCausalLM(AutoModelForCausalLM):
    """
    This class is a drop-in replacement for AutoModelForCausalLM that applies the Liger Kernel to the model
    if applicable.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model_config = _get_model_config(pretrained_model_name_or_path, **kwargs)

        # Determine the model type and apply the Liger Kernel if applicable
        # Note: _apply_liger_kernel will only pass relevant kwargs to the apply_liger_kernel_to_* function
        model_type = model_config.model_type

        _apply_liger_kernel(model_type, **kwargs)

        # Filter out kwargs that were passed to the apply_liger_* function, which will cause
        # model initialization errors otherwise
        apply_fn = MODEL_TYPE_TO_APPLY_LIGER_FN[model_type]
        apply_fn_signature = inspect.signature(apply_fn)

        applicable_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in apply_fn_signature.parameters
        }

        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **applicable_kwargs
        )
