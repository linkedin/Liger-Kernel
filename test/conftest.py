import pytest
import torch

from liger_kernel.utils import is_npu_available


@pytest.fixture(autouse=True)
def clear_gpu_cache():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif is_npu_available():
        torch.npu.empty_cache()
    elif torch.xpu.is_available():
        torch.xpu.empty_cache()
