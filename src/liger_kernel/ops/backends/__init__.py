import importlib
import pkgutil

from liger_kernel.ops.backends.registry import VENDOR_REGISTRY  # noqa: F401
from liger_kernel.ops.backends.registry import VendorInfo  # noqa: F401
from liger_kernel.ops.backends.registry import get_vendor_for_device  # noqa: F401
from liger_kernel.ops.backends.registry import register_vendor  # noqa: F401

# Auto-import all _<vendor> subpackages to trigger registration
# Each vendor's __init__.py calls register_vendor() when imported
for _, modname, ispkg in pkgutil.iter_modules(__path__):
    if ispkg and modname.startswith("_"):
        importlib.import_module(f"{__name__}.{modname}")
