import pytest
import torch

from liger_kernel.utils import infer_device
from test.utils import set_seed


@pytest.fixture(autouse=True)
def set_random_seed():
    set_seed(42)


@pytest.fixture(autouse=True)
def require_triton_apple_backend_on_mps():
    if infer_device() == "mps":
        try:
            import triton_apple_backend  # noqa: F401
        except ImportError:
            pytest.skip("triton_apple_backend is not installed")
    yield


@pytest.fixture(autouse=True)
def clear_gpu_cache():
    yield
    dev = infer_device()
    if dev == "cuda":
        torch.cuda.empty_cache()
    elif dev == "mps":
        torch.mps.empty_cache()
    elif dev == "npu":
        torch.npu.empty_cache()
    elif dev == "xpu":
        torch.xpu.empty_cache()


@pytest.fixture(autouse=True)
def reset_liger_backend_selection():
    """Isolate global backend/impl selection between tests.

    Several op tests call ``set_impl``/``set_backend`` (or set the
    ``LIGER_KERNEL_IMPL*`` env) to exercise a specific backend. Without a reset,
    that global choice leaks into every later test in the session, which routed
    unrelated ops (cross_entropy, tvd, dyt, ...) through a backend they were
    never meant to use and produced order-dependent failures. Restore the
    default (auto) selection and drop the availability memo after each test.
    """
    yield
    try:
        from liger_kernel.backends.dispatch import clear_available_cache
        from liger_kernel.backends.dispatch import set_impl

        set_impl(None)
        clear_available_cache()
    except Exception:
        pass


# Modules that call torch.use_deterministic_algorithms(True) at import. Pytest
# collection imports every test module first, so that process-global flag would
# otherwise stay on for later files. MPS scatter_reduce / index_put have no
# deterministic implementation and would raise.
_DETERMINISTIC_ALGORITHM_MODULES = frozenset(
    {
        "test.transformers.test_attn_res",
        "test.transformers.test_fused_add_rms_norm",
        "test.transformers.test_modulated_rms_norm",
        "test.transformers.test_poly_norm",
        "test.transformers.test_rms_norm",
    }
)


@pytest.fixture(autouse=True)
def isolate_deterministic_algorithms(request):
    """Re-apply each module's own deterministic setting, then clear the leak."""
    torch.use_deterministic_algorithms(request.module.__name__ in _DETERMINISTIC_ALGORITHM_MODULES)
    yield
    torch.use_deterministic_algorithms(False)
