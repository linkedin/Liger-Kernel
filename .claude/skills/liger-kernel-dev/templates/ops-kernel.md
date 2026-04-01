# Ops Kernel Template

## File: `src/liger_kernel/ops/{kernel}.py`

Before writing, read the reference kernel for this tier in the actual codebase.

### Import Block

```python
import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import ensure_contiguous
```

Add additional imports only as needed:
- `math` — if using `math.ceil` for row partitioning
- `from liger_kernel.ops.utils import compare_version` — if checking triton version
- `from liger_kernel.ops.utils import torch_to_triton_dtype` — if passing dtype as kernel arg
- `rsqrt` from triton libdevice — if using inverse square root (see `rms_norm.py` for the import pattern with version guards)
- `tanh` from triton libdevice — if using tanh (see `dyt.py` for the import pattern)

### Forward Kernel

```python
@triton.jit
def _{kernel}_forward_kernel(
    # Output pointers first, then input pointers
    Y_ptr, Y_stride,
    X_ptr, X_stride,
    # ... other input pointers and strides
    n_cols,
    # constexpr params for compile-time optimization
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Calculate base pointers using strides
    X_ptr += row_idx * X_stride
    Y_ptr += row_idx * Y_stride

    # Load with mask and float32 upcast for precision
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0).to(tl.float32)

    # === FORWARD COMPUTATION ===

    tl.store(Y_ptr + col_offsets, result, mask=mask)
```

### Backward Kernel

```python
@triton.jit
def _{kernel}_backward_kernel(
    # Gradient output, then gradient inputs, then saved tensors
    dY_ptr, dY_stride,
    dX_ptr, dX_stride,
    # ... saved tensors from forward
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load gradient and saved tensors
    # === BACKWARD COMPUTATION ===

    tl.store(dX_ptr + row_idx * dX_stride + col_offsets, dX_row, mask=mask)
```

### Python Wrappers

```python
def {kernel}_forward(X, ...):
    ori_shape = X.shape
    n_cols = ori_shape[-1]
    X = X.view(-1, n_cols)
    n_rows = X.shape[0]

    Y = torch.empty_like(X)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _{kernel}_forward_kernel[(n_rows,)](
        Y, Y.stride(-2),
        X, X.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return Y.view(*ori_shape)


def {kernel}_backward(dY, ...saved):
    ori_shape = dY.shape
    n_cols = ori_shape[-1]
    dY = dY.view(-1, n_cols)
    n_rows = dY.shape[0]

    dX = torch.empty_like(dY)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _{kernel}_backward_kernel[(n_rows,)](
        dY, dY.stride(-2),
        dX, dX.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return dX.view(*ori_shape)
```

### Autograd Function

```python
class Liger{Kernel}Function(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, ...params):
        Y = {kernel}_forward(X, ...params)
        ctx.save_for_backward(...)  # Save only what backward needs
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        ...saved = ctx.saved_tensors
        dX = {kernel}_backward(dY, ...saved)
        # Return one gradient per forward input, None for non-tensor params
        return dX, None, None, ...
```

### Tier-Specific Patterns

**Tier 1 (element-wise):**
- Grid: `(n_rows,)` — one program per row
- Simple 1D indexing with `col_offsets` and `mask`
- Prefer recomputation in backward over saving activations
- See `swiglu.py`: saves `a, b` but recomputes `silu(a)` in backward

**Tier 2 (reduction):**
- Forward may cache intermediate state (e.g., RSTD in RMSNorm)
- Backward may use SM-based parallelism for weight gradient reduction:
  ```python
  sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
  _dW = torch.empty((sm_count, n_cols), dtype=torch.float32, device=W.device)
  grid = (sm_count,)
  rows_per_program = math.ceil(n_rows / sm_count)
  ```
- See `rms_norm.py` for the full pattern

**Tier 3 (fused/complex):**
- Multi-pass algorithms (e.g., online softmax: pass 1 finds max+sum, pass 2 computes gradients)
- Gradient-in-forward trick: compute gradients during forward, store in input tensor
- Multiple `tl.constexpr` flags for compile-time dead code elimination
- See `cross_entropy.py` for the full pattern

### Key Rules

1. Always use `tl.program_id(0).to(tl.int64)` (prevents overflow on large tensors)
2. Always use `mask = col_offsets < n_cols` for boundary handling
3. Use `tl.constexpr` for parameters that enable compile-time optimization
4. Use stride parameters from tensors, never hardcode memory layout
5. Cast precision-sensitive ops to `tl.float32`, cast back for storage
6. Return `None` for each non-tensor parameter in backward
