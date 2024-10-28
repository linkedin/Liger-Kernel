import pytest
import torch


@pytest.fixture(autouse=True)
def clear_cuda_cache():
    yield
    torch.cuda.empty_cache()
