import importlib
import pkgutil

from liger_kernel.ops.backends.registry import VENDOR_REGISTRY

for _, module_name, _ in pkgutil.iter_modules(__path__):
    if module_name.startswith("_"):
        # This executes the __init__.py in each vendor subpackage (e.g., _nvidia/__init__.py)
        importlib.import_module(f".{module_name}", __name__)

__all__ = ["VENDOR_REGISTRY"]
