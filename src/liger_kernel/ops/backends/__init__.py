import importlib
import pkgutil

from liger_kernel.ops.backends.registry import VENDOR_REGISTRY
from liger_kernel.ops.backends.registry import VendorInfo
from liger_kernel.ops.backends.registry import get_vendor_for_device
from liger_kernel.ops.backends.registry import register_vendor

# Auto-import all _<vendor> subpackages to trigger registration
# Each vendor's __init__.py calls register_vendor() when imported
for _, modname, ispkg in pkgutil.iter_modules(__path__):
    if ispkg and modname.startswith('_'):
        importlib.import_module(f"{__name__}.{modname}")
