# Ascend NPU UB Manager Design Document

## Overview

The UB Manager (Unified Buffer Manager) is a core component in **Liger-Kernel** responsible for managing the Unified Buffer (UB) capacity on Ascend NPUs. By automatically detecting UB capacity and providing unified tiling strategy computation, it helps Triton kernels avoid UB overflow errors while maintaining high performance.

## Design Goals

1. **Automated UB Management**: Automatically detect device UB capacity without manual configuration
2. **Unified Strategy System**: Use a single unified strategy function for all kernels, abstracting memory calculations
3. **Flexible Parameters**: Support different memory multipliers and safety margins for different kernels
4. **Easy to Use**: Simple interface that directly computes tiling results

## Architecture Design

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    UB Manager System                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐         ┌──────────────────┐          │
│  │  UBManager   │         │ Default Strategy │          │
│  │   (Singleton)│────────▶│    Function      │          │
│  └──────────────┘         └──────────────────┘          │
│         │                            │                  │
│         │                            │                  │
│         ▼                            ▼                  │
│  ┌──────────────┐         ┌──────────────────┐          │
│  │   Capacity   │         │  compute_default │          │
│  │  Detection   │         │  _tiling_strategy│          │
│  └──────────────┘         └──────────────────┘          │
│                                                         │
└─────────────────────────────────────────────────────────┘
         │                            │
         │                            │
         ▼                            ▼
┌──────────────┐         ┌──────────────────┐
│   GEGLU      │         │      ROPE        │
│   Kernel     │         │     Kernel       │
└──────────────┘         └──────────────────┘
```

### Class Diagram

```
┌─────────────────────────────────────┐
│          UBManager                  │
├─────────────────────────────────────┤
│ - _npu_model: str                   │
│ - _ub_capacity_bits: int            │
├─────────────────────────────────────┤
│ + ub_capacity_bits: int             │
│ + ub_capacity_bytes: int            │
│ + npu_model: str                    │
│ - _detect_npu_model()               │
│ - _detect_ub_capacity()             │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│   compute_default_tiling_strategy   │
├─────────────────────────────────────┤
│ + safety_margin: float               │
│ + dtype_size: int                   │
│ + memory_multiplier: float           │
│ + tiling_dims: Tuple                │
│ + unit_params: Tuple                │
├─────────────────────────────────────┤
│ Returns: Tuple (final tiling sizes)  │
└─────────────────────────────────────┘
```

## Core Functionality

### 1. UB Capacity Detection

The UB Manager detects UB capacity in the following priority order:

1. **Environment Variable**: `ASCEND_UB_CAPACITY_BITS`
2. **Device Properties**: Retrieved from `torch.npu.get_device_properties(0).ub_capacity_bits`
3. **Model Defaults**: Use predefined values based on the detected NPU model

```python
# Default UB capacity configuration
_DEFAULT_UB_CAPACITIES = {
    "Ascend910B1": 2097152,  # ~256 KB
    "Ascend910B4": 1572864,  # ~192 KB
    "default": 2097152,       # ~256 KB
}
```

### 2. Unified Strategy System

All kernels use a single unified strategy function `_default_strategy` that abstracts memory calculations:

```
Memory Formula: memory_multiplier * BLOCK_SIZE * unit_param * dtype_size * 8 bits
```

The strategy function:
- Takes UB capacity, safety margin, dtype size, memory multiplier, tiling dimensions, and unit parameters
- Calculates the maximum safe block size that fits within UB capacity
- Returns a tuple of max_safe_block_size values (one for each tiling dimension)

The `compute_default_tiling_strategy` function:
- Calls `_default_strategy` to get max_safe_block_size
- Computes desired block size using `triton.next_power_of_2(tiling_dim)`
- Returns the final result: `min(desired, max_safe)` for each dimension

### 3. Parameter Structure

The unified strategy uses the following parameters:

- **`safety_margin`**: Safety margin as a float (e.g., 0.80 for 80%). Default is 0.80.
- **`dtype_size`**: Size of data type in bytes (e.g., 2 for float16, 4 for float32)
- **`memory_multiplier`**: Memory multiplier for estimating peak memory usage
  - For GEGLU: typically 10.0 for backward, 7.0 for forward
  - For ROPE: typically 3.0
- **`tiling_dims`**: Dimensions that need tiling as tuple
  - For GEGLU: `(n_cols,)`
  - For ROPE: `(pad_n_q_head, pad_n_kv_head)`
- **`unit_params`**: Parameters related to unit length of each tile as tuple
  - All elements are multiplied together to get the final unit_param
  - For GEGLU: `()` (empty tuple, unit_param = 1)
  - For ROPE: `(pad_hd,)` (unit_param = pad_hd)
  - For kernels with multiple factors: `(factor1, factor2, ...)` (unit_param = factor1 * factor2 * ...)

### 4. Strategy Computation Flow

```
User calls compute_default_tiling_strategy()
         │
         ▼
Get UB manager instance
         │
         ▼
Validate and set defaults for dtype_size and memory_multiplier
         │
         ▼
Call _default_strategy() with:
  - ub_capacity_bits
  - safety_margin
  - dtype_size
  - memory_multiplier
  - tiling_dims
  - unit_params
         │
         ▼
_extract unit_param from unit_params (multiply all elements)
         │
         ▼
Calculate max_block_size:
  SAFE_UB_CAPACITY_BITS = ub_capacity_bits * safety_margin
  max_block_size = SAFE_UB_CAPACITY_BITS / (memory_multiplier * unit_param * dtype_size * 8)
         │
         ▼
Find largest power of 2 <= max_block_size
         │
         ▼
Return (max_safe_block_size,) * len(tiling_dims)
         │
         ▼
Compute final result:
  For each tiling_dim:
    desired = triton.next_power_of_2(tiling_dim)
    final = min(desired, max_safe_block_size)
    final = max(1, final)
         │
         ▼
Return tuple of final tiling sizes
```

## Usage Examples

### Basic Usage

```python
from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy

# GEGLU forward
strategy = compute_default_tiling_strategy(
    safety_margin=0.80,
    dtype_size=2,  # float16
    memory_multiplier=7.0,
    tiling_dims=(4096,),
    unit_params=()
)
if strategy:
    block_size = strategy[0]
    # Call kernel with block_size

# ROPE forward
strategy = compute_default_tiling_strategy(
    safety_margin=0.90,
    dtype_size=4,  # float32
    memory_multiplier=3.0,
    tiling_dims=(32, 32),
    unit_params=(128,)
)
if strategy:
    BLOCK_Q, BLOCK_K = strategy
    # Call kernel with BLOCK_Q and BLOCK_K
```

## Strategy Function Details

### `_default_strategy` Function

The core strategy function that calculates maximum safe block size:

```python
def _default_strategy(
    ub_capacity_bits: int,
    safety_margin: float,
    dtype_size: int,
    memory_multiplier: float,
    tiling_dims: Optional[Tuple],
    unit_params: Optional[Tuple],
) -> Optional[Tuple]:
    """
    Calculate maximum safe block size based on UB capacity.
    
    Memory formula: memory_multiplier * BLOCK_SIZE * unit_param * dtype_size * 8 bits
    
    Returns tuple of max_safe_block_size (power of 2) for each tiling dimension.
    """
```

**Key Steps:**
1. Extract `unit_param` from `unit_params` by multiplying all elements (defaults to 1.0 if empty)
2. Calculate `SAFE_UB_CAPACITY_BITS = ub_capacity_bits * safety_margin`
3. Solve for max_block_size: `SAFE_UB_CAPACITY_BITS / (memory_multiplier * unit_param * dtype_size * 8)`
4. Find largest power of 2 <= max_block_size
5. Return tuple with same length as `tiling_dims`

### `compute_default_tiling_strategy` Function

The public interface that computes final tiling results:

```python
def compute_default_tiling_strategy(
    safety_margin: float = 0.80,
    dtype_size: Optional[int] = None,
    memory_multiplier: Optional[float] = None,
    tiling_dims: Optional[Tuple] = None,
    unit_params: Optional[Tuple] = None,
) -> Optional[Tuple]:
    """
    Compute tiling strategy using the default strategy function.
    
    Returns final tiling sizes: min(desired, max_safe) for each dimension.
    """
```

**Key Steps:**
1. Get UB manager instance
2. Set defaults for `dtype_size` (4) and `memory_multiplier` (10.0) if not provided
3. Call `_default_strategy` to get `max_supported`
4. For each `tiling_dim`:
   - Compute `desired = triton.next_power_of_2(tiling_dim)`
   - Compute `final = min(desired, max_safe_block_size)`
   - Ensure `final >= 1`
5. Return tuple of final tiling sizes

## Memory Analysis Examples

### GEGLU Forward

```
Memory analysis:
- Inputs: a, b
- Intermediates: a_cubed, tanh_arg, tanh_result, geglu_a
- Output: c
- Total: ~7x * BLOCK_SIZE * dtype_size

Strategy:
- memory_multiplier = 7.0
- unit_params = () (unit_param = 1)
- Formula: 7.0 * BLOCK_SIZE * 1 * dtype_size * 8 bits
```

### GEGLU Backward

```
Memory analysis:
- More intermediates for gradient computation
- Total: ~10x * BLOCK_SIZE * dtype_size

Strategy:
- memory_multiplier = 10.0
- unit_params = () (unit_param = 1)
- Formula: 10.0 * BLOCK_SIZE * 1 * dtype_size * 8 bits
```

### ROPE Forward/Backward

```
Memory analysis (based on optimized ROPE kernel):
- cos_vals and sin_vals: pad_hd // 2 elements each (shared)
- In q heads loop (peak memory):
  * q_left, q_right, new_left, new_right: 2 * BLOCK_Q * pad_hd elements
- In k heads loop (peak memory):
  * k_left, k_right, new_left, new_right: 2 * BLOCK_K * pad_hd elements
- Plus shared cos/sin: pad_hd elements
- Conservative estimate: 3 * BLOCK_SIZE * pad_hd * dtype_size * 8 bits

Strategy:
- memory_multiplier = 3.0
- unit_params = (pad_hd,) (unit_param = pad_hd)
- Formula: 3.0 * BLOCK_SIZE * pad_hd * dtype_size * 8 bits
```

## Extension Guide

### Adding a New Kernel

To add tiling support for a new kernel:

1. **Analyze memory usage**:
   - Identify peak memory usage in the kernel
   - Determine memory multiplier (e.g., 7.0, 10.0, 3.0)
   - Identify unit parameters if any (e.g., head_dim for ROPE)

2. **Use `compute_default_tiling_strategy`** in your kernel:

```python
def my_kernel_forward(input):
    # Prepare parameters
    n_cols = input.shape[-1]
    dtype_size = input.element_size()
    
    # Compute strategy
    strategy = compute_default_tiling_strategy(
        safety_margin=0.80,
        dtype_size=dtype_size,
        memory_multiplier=7.0,  # Based on your memory analysis
        tiling_dims=(n_cols,),
        unit_params=()  # Or (factor1, factor2, ...) if needed
    )
    
    if strategy is not None:
        block_size = strategy[0]
    else:
        block_size = triton.next_power_of_2(n_cols)  # Fallback
    
    # Call kernel
    kernel[(grid_size,)](
        input,
        BLOCK_SIZE=block_size,
    )
```

3. **Document memory analysis** in comments:

```python
# My kernel tiling strategy:
# - Memory analysis:
#   * Input: input
#   * Intermediates: intermediate1, intermediate2
#   * Output: output
#   * Total: ~7x * BLOCK_SIZE * dtype_size
# - Uses memory_multiplier=7.0 * BLOCK_SIZE * dtype_size * 8 bits for safety
# - compute_default_tiling_strategy returns the final tiling result:
#   min(triton.next_power_of_2(n_cols), max_safe_block_size)
```

## Future Improvements

1. **Strategy Variants**: If needed, could add specialized strategy functions for specific kernels while keeping the unified interface
2. **Multi-dimensional Tiling**: Could extend to support more complex tiling patterns if needed
