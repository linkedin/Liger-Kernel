"""
FlyDSL backend for Liger-Kernel.

FlyDSL is the optional, ROCm/AMD-oriented Python DSL + MLIR compiler
(``import flydsl``), targeting CDNA and RDNA GPUs. It is opt-in only —
users select it explicitly via ``LIGER_KERNEL_IMPL=flydsl``. It is not
auto-applied on any device (note the empty ``default_devices`` on the
registration below).

On ROCm builds of PyTorch the device type is still ``"cuda"`` (HIP), so
this implementation registers for ``devices=("cuda",)`` and relies on
``torch.version.hip`` / AMD hardware at runtime.
"""

from liger_kernel.ops.backends.registry import ImplInfo
from liger_kernel.ops.backends.registry import register_impl

register_impl(
    ImplInfo(
        name="flydsl",
        devices=("cuda",),
        module_path=f"{__name__}.ops",  # liger_kernel.ops.flydsl.ops
    )
)
