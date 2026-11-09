from liger_kernel.ops.backends._ascend.compile_kwargs import install_ascend_compile_kwargs
from liger_kernel.ops.backends.registry import ImplInfo
from liger_kernel.ops.backends.registry import register_impl

install_ascend_compile_kwargs()

# Ascend NPU backend — default on NPU devices.
# Future: when tilelang-ascend lands, this can be renamed to "ascend-triton"
# and a second register_impl(ImplInfo(name="ascend-tilelang", ...)) added.
register_impl(
    ImplInfo(
        name="ascend",
        devices=("npu",),
        default_devices=("npu",),
        module_path=f"{__name__}.ops",  # liger_kernel.ops.backends._ascend.ops
    )
)
