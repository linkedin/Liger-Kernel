from transformers import AutoConfig, AutoModelForCausalLM

from liger_kernel.transformers.monkey_patch import _apply_liger_kernel


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

        # Retain only the keyword args present in the model configuration
        for k in list(kwargs.keys()):
            if k not in model_config.__dict__:
                del kwargs[k]

        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
