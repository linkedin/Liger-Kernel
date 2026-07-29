"""
CuTe DSL backend for Liger-Kernel.

CuTe DSL is the optional, CUDA-only Python DSL shipped with NVIDIA CUTLASS
(``import cutlass.cute``), targeting Hopper (SM90) and Blackwell (SM100/SM110).
It is opt-in only — users select it explicitly via ``LIGER_KERNEL_IMPL=cutedsl``.
It is not auto-applied on any device (note the empty ``default_devices`` on the
registration below).
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
