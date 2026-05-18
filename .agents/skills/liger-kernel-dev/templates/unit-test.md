# Unit Test Template

## File: `test/transformers/test_{kernel}.py`

```python
import pytest
import torch

from test.utils import assert_verbose_allclose
from test.utils import infer_device
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.ops import Liger{Kernel}Function
from liger_kernel.transformers.{kernel} import Liger{Kernel}
from liger_kernel.transformers.functional import liger_{kernel}


# ---- PyTorch Reference ----
# Paste the standalone PyTorch reference from the Analyzer stage here.

class Torch{Kernel}(torch.nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...

    def forward(self, x):
        ...


set_seed(42)
device = infer_device()


# ---- Correctness Test (Module) ----

@pytest.mark.parametrize(
    "B, T, hidden_size",
    [
        (2, 8, 4096),
        (4, 16, 2048),
        (1, 1, 1023),   # Non-power-of-2
        (3, 7, 256),    # Small + prime numbers
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-1,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(),
                reason="bfloat16 not supported on this GPU",
            ),
        ),
    ],
)
def test_liger_{kernel}_correctness(B, T, hidden_size, dtype, atol, rtol):
    _input = torch.randn(B, T, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # Initialize both with same weights
    torch_layer = Torch{Kernel}(...).to(device).to(dtype)
    liger_layer = Liger{Kernel}(...).to(device).to(dtype)
    # Copy weights from torch to liger
    # liger_layer.weight.data = torch_layer.weight.data.clone()

    # Forward
    torch_output = torch_layer(x1)
    liger_output = liger_layer(x2)
    assert_verbose_allclose(torch_output, liger_output, rtol=rtol, atol=atol, extra_info="[output]")

    # Backward
    grad_output = torch.randn_like(_input)
    torch_output.backward(grad_output)
    liger_output.backward(grad_output)
    assert_verbose_allclose(x1.grad, x2.grad, rtol=rtol, atol=atol, extra_info="[input.grad]")

    # Check weight gradients if applicable
    # assert_verbose_allclose(
    #     torch_layer.weight.grad, liger_layer.weight.grad,
    #     rtol=rtol, atol=atol, extra_info="[weight.grad]"
    # )


# ---- Functional API Test ----

@pytest.mark.parametrize(
    "B, T, hidden_size",
    [
        (2, 8, 4096),
        (1, 1, 1023),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(),
                reason="bfloat16 not supported on this GPU",
            ),
        ),
    ],
)
def test_liger_{kernel}_functional(B, T, hidden_size, dtype, atol, rtol):
    _input = torch.randn(B, T, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # Test that functional and Function.apply produce identical results
    output1 = liger_{kernel}(x1, ...)
    output2 = Liger{Kernel}Function.apply(x2, ...)

    assert_verbose_allclose(output1, output2, rtol=rtol, atol=atol)

    grad_output = torch.randn_like(_input)
    output1.backward(grad_output)
    output2.backward(grad_output)
    assert_verbose_allclose(x1.grad, x2.grad, rtol=rtol, atol=atol)
```

### Key Testing Rules

1. **Always test both forward AND backward** — backward bugs are common
2. **Include non-power-of-2 shapes** — tests the mask boundary handling
3. **Parametrize dtypes with per-dtype tolerances**:
   - `float32`: `atol=1e-5, rtol=1e-5` (tight)
   - `bfloat16`: `atol=1e-1, rtol=5e-2` (loose — bf16 has limited precision)
4. **Use `set_seed(42)` at module level** for reproducibility
5. **Use `infer_device()`** not `"cuda"` — supports multi-vendor
6. **Guard bfloat16 with `supports_bfloat16()`** — skips on older GPUs
7. **Use `assert_verbose_allclose`** not `torch.testing.assert_close` — gives better error output
8. **Test weight gradients** when the kernel has learnable parameters
9. **Add `@pytest.mark.flaky(reruns=3)`** if the test is sensitive to random seeds
10. **Add extra test functions** for edge cases specific to the kernel
