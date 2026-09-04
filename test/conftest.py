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
