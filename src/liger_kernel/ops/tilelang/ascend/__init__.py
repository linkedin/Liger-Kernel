"""TileLang implementations for Ascend NPUs."""

from liger_kernel.ops.backends.registry import ImplInfo
from liger_kernel.ops.backends.registry import register_impl

register_impl(
    ImplInfo(
        name="ascend-tilelang",
        devices=("npu",),
        module_path=f"{__name__}.ops",
    )
)
