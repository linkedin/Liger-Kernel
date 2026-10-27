"""One CV workspace slot under Triton-Ascend 3.6.0.

BiShengIR 2.0 defaults to 4 slots and overflows the 192KB UB.
"""

_INSTALLED = False


def _npu_option_fields() -> dict:
    try:
        from triton.backends.ascend.compiler import NPUOptions
    except ImportError:
        return {}
    return getattr(NPUOptions, "__dataclass_fields__", {}) or {}


def apply_ascend_compile_kwargs(kwargs: dict) -> dict:
    if "set_workspace_multibuffer" in _npu_option_fields():
        kwargs.setdefault("set_workspace_multibuffer", 1)
    return kwargs


def install_ascend_compile_kwargs() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        from triton.runtime.jit import JITFunction
    except ImportError:
        _INSTALLED = True
        return

    original_run = JITFunction.run

    def run(self, *args, **kwargs):
        apply_ascend_compile_kwargs(kwargs)
        return original_run(self, *args, **kwargs)

    JITFunction.run = run
    _INSTALLED = True
