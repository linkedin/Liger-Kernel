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
┌──────────────────────────────────────┐
│          UBManager                   │
├──────────────────────────────────────┤
│ - _npu_model: str                    │
│ - _ub_capacity_bits: int             │
├──────────────────────────────────────┤
│ + ub_capacity_bits: int              │
│ + ub_capacity_bytes: int             │
│ + npu_model: str                     │
│ - _detect_npu_model()                │
│ - _detect_ub_capacity()              │
│   (raises RuntimeError if fails)     │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│   compute_default_tiling_strategy    │
├──────────────────────────────────────┤
│ + safety_margin: float                │
│ + dtype_size: int                    │
│ + memory_multiplier: float            │
│ + shapes: Tuple[Tuple[int, ...], ...]│
│ + tiling_dims: Tuple                 │
├──────────────────────────────────────┤
│ Returns: Tuple[Tuple[int, ...], ...] │
│   (same structure as shapes)         │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│   _normalize_tiling_dims             │
├──────────────────────────────────────┤
│ Helper function to normalize         │
│ tiling_dim (int or tuple) to set     │
└──────────────────────────────────────┘
```

## Core Functionality

### 1. UB Capacity Detection

The UB Manager detects UB capacity in the following priority order:

1. **Environment Variable**: `ASCEND_UB_CAPACITY_BITS` (in bits)
   - If set, this value is used directly
   - Must be a positive integer representing UB capacity in bits

2. **get_soc_spec**: Query UB size from CANN's `get_soc_spec("UB_SIZE")`
   - Returns UB size in bytes
   - Automatically converted to bits (bytes * 8)
   - Requires CANN environment to be sourced (e.g., `source /usr/local/Ascend/ascend-toolkit/set_env.sh`)

3. **Error Handling**: If neither method succeeds, raises `RuntimeError` with clear instructions


```python
# Detection flow:
# 1. Check ASCEND_UB_CAPACITY_BITS env var (bits)
# 2. Try get_soc_spec("UB_SIZE") (bytes) -> convert to bits
# 3. Raise RuntimeError if both fail
```

### 2. Unified Strategy System

All kernels use a single unified strategy function `_default_strategy` that abstracts memory calculations:

```
Memory Formula: memory_multiplier * BLOCK_SIZE * unit_param * dtype_size * 8 bits
```

Where `unit_param` is automatically calculated as the product of all fixed (non-tiling) dimensions in each shape.

The strategy function:
- Takes UB capacity, safety margin, dtype size, memory multiplier, shapes, and tiling dimension specifications
- For each shape, identifies which dimensions can be tiled (from `tiling_dims`)
- Calculates `unit_param` as the product of fixed (non-tiling) dimensions
- Calculates the maximum safe block size that fits within UB capacity
- Returns a tuple of max_safe_block_size values (one for each shape)

The `compute_default_tiling_strategy` function:
- Calls `_default_strategy` to get max_safe_block_size for each shape
- For each tiling dimension, computes desired block size using `triton.next_power_of_2(original_dim)`
- Returns the final result with same structure as input shapes: tiling dimensions replaced with computed block sizes, non-tiling dimensions padded to next power of 2

### 3. Parameter Structure

The unified strategy uses the following parameters:

- **`safety_margin`**: Safety margin as a float (e.g., 0.80 for 80%). Default is 0.80.
- **`dtype_size`**: Size of data type in bytes (e.g., 2 for float16, 4 for float32)
- **`memory_multiplier`**: Memory multiplier for estimating peak memory usage
  - For GEGLU: typically 10.0 for backward, 7.0 for forward
  - For ROPE: typically 3.0
- **`shapes`**: Tuple of full shapes. Each shape is a tuple of dimension sizes.
  - For ROPE: `((n_q_head, hd), (n_kv_head, hd))`
  - For GEGLU: `((n_cols,),)`
  - Can pass original shapes (will handle padding internally) or padded shapes
- **`tiling_dims`**: Tuple specifying which dimensions can be tiled for each shape.
  - Each element can be:
    - `int`: single dimension index (e.g., `0` for first dimension)
    - `tuple of ints`: multiple dimensions that can be tiled together (non-empty)
  - For ROPE: `(0, 0)` means first dimension of each shape can be tiled
  - For GEGLU: `(0,)` means first dimension of the shape can be tiled
  - Length must match `len(shapes)`
  - Fixed dimensions (non-tiling) are automatically extracted from shapes and multiplied to get `unit_param`
  - **Validation**: Raises `ValueError` if:
    - Any `tiling_dim` is empty or invalid (e.g., empty tuple)
    - Any dimension index is out of bounds (negative or >= shape length)

### 4. Strategy Computation Flow

```
User calls compute_default_tiling_strategy()
         │
         ▼
Get UB manager instance
         │
         ▼
Validate shapes and tiling_dims (lengths must match)
         │
         ▼
Set defaults for dtype_size (4) and memory_multiplier (10.0)
         │
         ▼
Call _default_strategy() with:
  - ub_capacity_bits
  - safety_margin
  - dtype_size
  - memory_multiplier
  - shapes
  - tiling_dims
         │
         ▼
For each (shape, tiling_dim) pair:
  Normalize tiling_dim to set of dimension indices
  Validate tiling dimensions are within shape bounds
  (Raises ValueError if invalid)
         │
         ▼
  Calculate unit_param:
    unit_param = product of all non-tiling dimensions
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
Return tuple of max_safe_block_size (one per shape)
         │
         ▼
Build result with same structure as shapes:
  For each (shape, tiling_dim, max_safe):
    For each tiling dimension:
      desired = triton.next_power_of_2(original_dim)
      final = min(desired, max_safe)
      final = max(1, final)
    For each non-tiling dimension:
      pad to triton.next_power_of_2(original_dim)
         │
         ▼
Return tuple of tiled shapes
```

## Usage Examples

### Basic Usage

```python
from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy

# GEGLU forward
shapes = ((4096,),)
tile_shapes = compute_default_tiling_strategy(
    safety_margin=0.80,
    dtype_size=2,  # float16
    memory_multiplier=7.0,
    shapes=shapes,
    tiling_dims=(0,)  # First dimension can be tiled
)
if tile_shapes is not None and len(tile_shapes) > 0:
    block_size = tile_shapes[0][0]
    # Call kernel with block_size

# ROPE forward
shapes = ((32, 128), (32, 128))  # (n_q_head, hd), (n_kv_head, hd)
tile_shapes = compute_default_tiling_strategy(
    safety_margin=0.90,
    dtype_size=4,  # float32
    memory_multiplier=3.0,
    shapes=shapes,
    tiling_dims=(0, 0)  # First dimension of each shape can be tiled
)
if tile_shapes is not None and len(tile_shapes) == len(shapes):
    q_tile_shape, k_tile_shape = tile_shapes
    BLOCK_Q, _ = q_tile_shape  # Tiled dimension
    BLOCK_K, _ = k_tile_shape  # Tiled dimension
    # Call kernel with BLOCK_Q and BLOCK_K
```

## Strategy Function Details

### `_normalize_tiling_dims` Helper Function

A helper function that normalizes tiling dimension specifications:

```python
def _normalize_tiling_dims(tiling_dim: Union[int, Tuple[int, ...]]) -> set:
    """
    Normalize tiling dimension specification to a set of dimension indices.
    
    Args:
        tiling_dim: Either an int (single dimension) or tuple of ints (multiple dimensions).
    
    Returns:
        Set of dimension indices that can be tiled.
    """
```

This function handles the conversion of `tiling_dim` from either an `int` or `tuple` to a `set` for consistent processing.

### `_default_strategy` Function

The core strategy function that calculates maximum safe block size:

```python
def _default_strategy(
    ub_capacity_bits: int,
    safety_margin: float,
    dtype_size: int,
    memory_multiplier: float,
    shapes: Tuple[Tuple[int, ...], ...],
    tiling_dims: Tuple[Union[int, Tuple[int, ...]], ...],
) -> Tuple[int, ...]:
    """
    Calculate maximum safe block size based on UB capacity.
    
    Memory formula: memory_multiplier * BLOCK_SIZE * unit_param * dtype_size * 8 bits
    
    For each shape, fixed dimensions (non-tiling) are multiplied together to get unit_param.
    
    Returns:
        Tuple of max_safe_block_size (power of 2), one for each shape.
    
    Raises:
        ValueError: If any tiling_dim is empty or invalid, or if any dimension
                    index is out of bounds for the corresponding shape.
    """
```

**Key Steps:**
1. For each `(shape, tiling_dim)` pair:
   - Normalize `tiling_dim` to a set of dimension indices using `_normalize_tiling_dims`
   - Validate tiling dimensions are within shape bounds
     - Raises `ValueError` if `tiling_dim` is empty or invalid
     - Raises `ValueError` if any dimension index is out of bounds
   - Calculate `unit_param` as the product of all non-tiling dimensions
   - If all dimensions are tiling, `unit_param = 1.0`
2. Calculate `SAFE_UB_CAPACITY_BITS = ub_capacity_bits * safety_margin`
3. Solve for max_block_size: `SAFE_UB_CAPACITY_BITS / (memory_multiplier * unit_param * dtype_size * 8)`
4. Find largest power of 2 <= max_block_size
5. Return tuple with one max_safe_block_size per shape

### `compute_default_tiling_strategy` Function

The public interface that computes final tiling results:

```python
def compute_default_tiling_strategy(
    safety_margin: float = 0.80,
    dtype_size: Optional[int] = None,
    memory_multiplier: Optional[float] = None,
    shapes: Optional[Tuple[Tuple[int, ...], ...]] = None,
    tiling_dims: Optional[Tuple[Union[int, Tuple[int, ...]], ...]] = None,
) -> Optional[Tuple[Tuple[int, ...], ...]]:
    """
    Compute tiling strategy using the default strategy function.
    
    Returns tuple of tiled shapes with same structure as input shapes.
    Tiling dimensions are replaced with computed block sizes (power of 2),
    while non-tiling dimensions are padded to next power of 2.
    
    Returns:
        Tuple of tiled shapes, or None if shapes/tiling_dims are empty or
        lengths don't match.
    
    Raises:
        ValueError: If any tiling_dim is empty or invalid, or if any dimension
                    index is out of bounds for the corresponding shape.
    """
```

**Key Steps:**
1. Get UB manager instance
2. Validate `shapes` and `tiling_dims` (lengths must match, cannot be empty)
   - Returns `None` if validation fails (empty or mismatched lengths)
3. Set defaults for `dtype_size` (4) and `memory_multiplier` (10.0) if not provided
4. Call `_default_strategy` to get `max_supported` (tuple of max_safe_block_size, one per shape)
   - May raise `ValueError` if `tiling_dims` are invalid (see `_default_strategy` documentation)
5. For each `(shape, tiling_dim, max_safe)`:
   - Normalize `tiling_dim` to a set of dimension indices
   - Validate tiling dimensions are within shape bounds
     - Raises `ValueError` if `tiling_dim` is empty or invalid
     - Raises `ValueError` if any dimension index is out of bounds
   - For each tiling dimension:
     - Compute `desired = triton.next_power_of_2(original_dim)`
     - Compute `final = min(desired, max_safe)`
     - Ensure `final >= 1`
     - Replace dimension with `final`
   - For each non-tiling dimension:
     - Pad to `triton.next_power_of_2(original_dim)`
6. Return tuple of tiled shapes (same structure as input `shapes`)

## Memory Analysis Examples

### GEGLU Forward

```
Memory analysis:
- Inputs: a, b
- Intermediates: a_cubed, tanh_arg, tanh_result, geglu_a
- Output: c
- Total: ~7x * BLOCK_SIZE * dtype_size

Strategy:
- shapes: ((n_cols,),)
- tiling_dims: (0,)  # First dimension can be tiled
- Fixed dimensions: none (all dimensions are tiling)
- unit_param = 1 (product of fixed dimensions)
- memory_multiplier = 7.0
- Formula: 7.0 * BLOCK_SIZE * 1 * dtype_size * 8 bits
- Returns: ((block_size,),)
```

### GEGLU Backward

```
Memory analysis:
- More intermediates for gradient computation
- Total: ~10x * BLOCK_SIZE * dtype_size

Strategy:
- shapes: ((n_cols,),)
- tiling_dims: (0,)  # First dimension can be tiled
- Fixed dimensions: none (all dimensions are tiling)
- unit_param = 1 (product of fixed dimensions)
- memory_multiplier = 10.0
- Formula: 10.0 * BLOCK_SIZE * 1 * dtype_size * 8 bits
- Returns: ((block_size,),)
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
- shapes: ((pad_n_q_head, pad_hd), (pad_n_kv_head, pad_hd))
- tiling_dims: (0, 0)  # First dimension of each shape can be tiled
- Fixed dimensions: pad_hd (second dimension, non-tiling)
- unit_param = pad_hd (product of fixed dimensions)
- memory_multiplier = 3.0
- Formula: 3.0 * BLOCK_SIZE * pad_hd * dtype_size * 8 bits
- Returns: ((block_size_q, pad_hd), (block_size_kv, pad_hd))
```

## Extension Guide

### Adding a New Kernel

To add tiling support for a new kernel:

1. **Analyze memory usage**:
   - Identify peak memory usage in the kernel
   - Determine memory multiplier (e.g., 7.0, 10.0, 3.0)
   - Identify which dimensions can be tiled and which are fixed
   - Fixed dimensions will be automatically extracted and multiplied to get `unit_param`

2. **Use `compute_default_tiling_strategy`** in your kernel:

```python
def my_kernel_forward(input):
    # Prepare parameters
    n_cols = input.shape[-1]
    dtype_size = input.element_size()
    
    # Compute strategy
    # Example 1: Simple case (all dimensions can be tiled)
    shapes = ((n_cols,),)
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.80,
        dtype_size=dtype_size,
        memory_multiplier=7.0,  # Based on your memory analysis
        shapes=shapes,
        tiling_dims=(0,)  # First dimension can be tiled
    )
    
    if tile_shapes is not None and len(tile_shapes) > 0:
        block_size = tile_shapes[0][0]
    else:
        block_size = triton.next_power_of_2(n_cols)  # Fallback
    
    # Example 2: Multiple shapes with fixed dimensions
    # shapes = ((M, K), (K, N))
    # tiling_dims = (0, 1)  # First shape: dim 0 can be tiled, dim 1 is fixed
    #                      # Second shape: dim 0 is fixed, dim 1 can be tiled
    # Returns: ((block_M, K), (K, block_N))
    
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
# - shapes: ((n_cols,),)
# - tiling_dims: (0,) means first dimension can be tiled
# - Fixed dimensions: none (all dimensions are tiling)
# - unit_param = 1 (product of fixed dimensions)
# - Uses memory_multiplier=7.0 * BLOCK_SIZE * dtype_size * 8 bits for safety
# - compute_default_tiling_strategy returns: ((block_size,),)
#   where block_size = min(triton.next_power_of_2(n_cols), max_safe_block_size)
```

## Future Improvements

1. **Strategy Variants**: If needed, could add specialized strategy functions for specific kernels while keeping the unified interface
2. **Multi-dimensional Tiling**: Could extend to support more complex tiling patterns if needed
