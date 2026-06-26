"""
cuTeDSL backend for Liger-Kernel.

This backend is opt-in only and currently ships RMSNorm parity first.
Enable it with ``LIGER_KERNEL_IMPL=cutedsl``.
"""

from liger_kernel.ops.backends.registry import ImplInfo
from liger_kernel.ops.backends.registry import register_impl

register_impl(
    ImplInfo(
        name="cutedsl",
        devices=("cuda",),
        module_path=f"{__name__}.ops",  # liger_kernel.ops.cutedsl.ops
    )
)
