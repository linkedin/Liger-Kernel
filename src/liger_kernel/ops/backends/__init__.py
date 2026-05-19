import importlib
import pkgutil

from liger_kernel.ops.backends.registry import BACKEND_REGISTRY  # noqa: F401
from liger_kernel.ops.backends.registry import LIGER_KERNEL_BACKEND_ENV  # noqa: F401
from liger_kernel.ops.backends.registry import BackendInfo  # noqa: F401
from liger_kernel.ops.backends.registry import register_backend  # noqa: F401
from liger_kernel.ops.backends.registry import select_backend  # noqa: F401

# Auto-import all _<backend> subpackages to trigger registration.
# Each backend's __init__.py calls register_backend() when imported.
for _, modname, ispkg in pkgutil.iter_modules(__path__):
    if ispkg and modname.startswith("_"):
        importlib.import_module(f"{__name__}.{modname}")
