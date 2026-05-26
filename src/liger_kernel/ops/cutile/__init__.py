"""
cuTile backend for Liger-Kernel.

cuTile is an optional CUDA-only DSL. It is opt-in only — users select it
explicitly via ``LIGER_KERNEL_IMPL=cutile``. It is not auto-applied on
any device (note the empty ``default_devices`` on the registration below).
"""

from liger_kernel.ops.backends.registry import ImplInfo
from liger_kernel.ops.backends.registry import register_impl

register_impl(
    ImplInfo(
        name="cutile",
        devices=("cuda",),
        module_path=f"{__name__}.ops",  # liger_kernel.ops.cutile.ops
    )
)
