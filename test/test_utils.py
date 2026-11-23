import pytest
import torch

from test.utils import require_highest_float32_matmul_precision


@pytest.mark.parametrize("initial_precision", ["highest", "high", "medium"])
@pytest.mark.parametrize("raise_in_test", [False, True])
def test_highest_float32_matmul_precision_restores_original_setting(initial_precision, raise_in_test):
    original_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision(initial_precision)

        @require_highest_float32_matmul_precision
        def run(value):
            assert torch.get_float32_matmul_precision() == "highest"
            if raise_in_test:
                raise RuntimeError("test body failed")
            return value

        if raise_in_test:
            with pytest.raises(RuntimeError, match="test body failed"):
                run(17)
        else:
            assert run(17) == 17
        assert torch.get_float32_matmul_precision() == initial_precision
    finally:
        torch.set_float32_matmul_precision(original_precision)
