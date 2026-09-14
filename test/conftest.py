import pytest
import torch

from liger_kernel.utils import is_npu_available
from test.utils import set_seed


@pytest.fixture
def qwen4_exp_globals():
    """Restore class swaps, class attributes, and the native-forward registry together."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp

    from liger_kernel.transformers import monkey_patch

    names = (
        "Qwen4ExpTextRMSNorm",
        "Qwen4ExpTextMLP",
        "Qwen4ExpTextExperts",
        "Qwen4ExpTextNGramEmbedding",
        "Qwen4ExpTextGatedResidual",
        "Qwen4ExpTextDecoderLayer",
        "Qwen4ExpForCausalLM",
    )
    classes = {name: getattr(modeling_qwen4_exp, name) for name in names}
    attributes = {cls: dict(vars(cls)) for cls in classes.values()}
    native_classes = monkey_patch._QWEN4_EXP_NATIVE_RMS_NORM_CLASSES
    native_forward = monkey_patch._QWEN4_EXP_NATIVE_RMS_NORM_FORWARD
    try:
        yield modeling_qwen4_exp
    finally:
        for name, cls in classes.items():
            setattr(modeling_qwen4_exp, name, cls)
            saved = attributes[cls]
            for attribute in vars(cls).keys() - saved.keys():
                delattr(cls, attribute)
            for attribute, value in saved.items():
                if vars(cls).get(attribute) is not value:
                    setattr(cls, attribute, value)
        monkey_patch._QWEN4_EXP_NATIVE_RMS_NORM_CLASSES = native_classes
        monkey_patch._QWEN4_EXP_NATIVE_RMS_NORM_FORWARD = native_forward


@pytest.fixture(autouse=True)
def set_random_seed():
    set_seed(42)


@pytest.fixture(autouse=True)
def clear_gpu_cache():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif is_npu_available():
        torch.npu.empty_cache()
    elif torch.xpu.is_available():
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
