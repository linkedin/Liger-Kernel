"""Opt-in portable Triton implementations."""

from liger_kernel.ops.backends.registry import ImplInfo
from liger_kernel.ops.backends.registry import register_impl

register_impl(
    ImplInfo(
        name="triton",
        devices=("cuda",),
        module_path=f"{__name__}.ops",
    )
)
