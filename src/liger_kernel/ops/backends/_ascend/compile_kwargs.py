"""One CV workspace slot. BiShengIR 2.0 defaults to 4 and overflows the 192KB UB."""


def install_ascend_compile_kwargs() -> None:
    try:
        from triton.backends.ascend.compiler import NPUOptions
        from triton.runtime.jit import JITFunction
    except ImportError:
        return

    if "set_workspace_multibuffer" not in getattr(NPUOptions, "__dataclass_fields__", {}):
        return

    original_run = JITFunction.run
    if getattr(original_run, "_ascend_workspace_slot", False):
        return

    def run(self, *args, **kwargs):
        kwargs.setdefault("set_workspace_multibuffer", 1)
        return original_run(self, *args, **kwargs)

    run._ascend_workspace_slot = True
    JITFunction.run = run
