import importlib
import pkgutil

from liger_kernel.ops.backends.registry import IMPL_REGISTRY  # noqa: F401
from liger_kernel.ops.backends.registry import LIGER_KERNEL_IMPL_ENV  # noqa: F401
from liger_kernel.ops.backends.registry import ImplInfo  # noqa: F401
from liger_kernel.ops.backends.registry import register_impl  # noqa: F401
from liger_kernel.ops.backends.registry import select_impl  # noqa: F401

# Auto-import all _<name> subpackages to trigger registration of
# alternative-hardware backends (e.g., _ascend/). Each one calls register_impl()
# in its __init__.py.
for _, modname, ispkg in pkgutil.iter_modules(__path__):
    if ispkg and modname.startswith("_"):
        importlib.import_module(f"{__name__}.{modname}")
