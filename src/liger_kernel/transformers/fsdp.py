from typing import Any
from typing import Callable

from torch.distributed.fsdp import FullyShardedDataParallel


class _FSDPForwardRedirection:
    """
    Modified based on
    https://github.com/Lightning-AI/pytorch-lightning/blob/d3f9c83d6efa4f1def36aa6c199600946cdb9117/src/lightning/pytorch/strategies/strategy.py#L601-L648
    Redirect a method call through FullyShardedDataParallel.forward so that the FSDP module's root pre-forward and
    post-forward can be properly executed around the method call.
    This is needed in cases where we call a submodule of a FSDP module. For instance, when we want to call only
    the `LlamaModel` part out of a FSDP-wrapped `LlamaForCausalLM` to get the hidden states without involving
    GPU-memory-heavy `lm_head` and cross entropy computation, doing this directly (i.e. `model.model.forward()`)
    will not work because the first `nn.Embedding` layer is not independently wrapped as a FSDP module (because of
    the transformer-based wrapping policy), and not calling it through FSDP root module forward will not all-gather
    its parameter, thus resulting in "RuntimeError: 'weight' must be 2-D" error. Similarly, if we want to call just
    the `lm_head` part of a model, we need this trick too to properly get its params all-gathered.
    """

    def __call__(
        self,
        wrapper_module: FullyShardedDataParallel,
        method: Callable,
        *args: Any,
        **kwargs: Any,
    ):
        """Reroutes a method call through the `wrapper_module`'s `forward` method.
        Args:
            wrapper_module: The module that has `original_module` wrapped.
            original_module: The module that was wrapped inside `wrapper_module`.
            method_name: The name of the method that should be called on the `original_module` after inputs get
                redirected through the `wrapper_module`'s `forward` method.
            *args: The positional arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.
            **kwargs: The keyword arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.
        """
        assert isinstance(wrapper_module, FullyShardedDataParallel)
        original_module = wrapper_module._fsdp_wrapped_module
        original_forward = original_module.forward

        def wrapped_forward(*_args: Any, **_kwargs: Any) -> Any:
            # Unpatch ourselves immediately before calling the method `method_name`
            # because itself may want to call the real `forward`
            original_module.forward = original_forward  # type: ignore[method-assign]
            # Call the actual method e.g. `.training_step(...)`
            out = method(*_args, **_kwargs)
            return out

        # Patch the original_module's forward so we can redirect the arguments back to the real method
        original_module.forward = wrapped_forward  # type: ignore[method-assign]
        wrapper_output = wrapper_module(*args, **kwargs)
        return wrapper_output
