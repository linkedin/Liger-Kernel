# Attribution — `_cute_lib`

The CuTe DSL utilities in this directory are adapted from **Quack**
(https://github.com/Dao-AILab/quack), Apache License 2.0,
Copyright (c) 2025 Wentao Guo, Ted Zadouri, Tri Dao.

Upstream commit pinned during extraction: **ae7a32cf5a880a2c656b7c22b8409fba74bdd3fc**
(`[Bench] Better env var for torch compile`).

## Why inline?

The Liger `nvidia-cutedsl` RMSNorm and LayerNorm backends used to import
Quack at runtime as a dependency. To keep the Liger install slim and remove
a third-party runtime requirement, we have inlined only the utilities our
kernels reach (transitive closure of the public symbols listed below). The
upstream module is otherwise unmodified beyond the import-path edits required
to make it self-contained.

## Symbol map

| Inlined file | Upstream module | Symbols kept |
|---|---|---|
| `compile_utils.py` | `quack.compile_utils` | `make_fake_tensor` |
| `dtype_map.py` | `quack.cute_dsl_utils` | `torch2cute_dtype_map` |
| `layout_utils.py` | `quack.layout_utils` | `expand` |
| `copy_utils.py` | `quack.copy_utils` | `copy`, `tiled_copy_2d`, `predicate_k`, `get_copy_atom` |
| `utils.py` | `quack.utils` | `fill_oob`, `elem_pointer`, `store_shared_remote`, `set_block_rank`, `f32x2_to_i64`, `i64_to_f32x2` |
| `reduce.py` | `quack.reduce` | `row_reduce`, `block_reduce`, `cluster_reduce`, `block_or_cluster_reduce` |
| `reduction_base.py` | `quack.reduction_base` | `ReductionBase` |
| `rmsnorm_fwd.py` | `quack.rmsnorm` | `RMSNorm` (forward kernel class), `rmsnorm_fwd`, `layernorm_fwd` |

## Symbols intentionally omitted

The upstream Quack modules ship far more functionality than Liger uses.
The following are *not* present in this inlined slice and would need to
be ported back if a future kernel ever reaches them:

- `quack.rmsnorm` — `RMSNormBackward` and `RMSNormFunction` (Liger has its
  own inline backward kernel; `RMSNormBackward` would only confuse the
  reader of the inlined slice), the persistent `.o` JIT cache
  (`@jit_cache`), the `torch.library.custom_op` wrapper
  (`quack::_rmsnorm_fwd`), and the reference implementations
  (`rmsnorm_ref` / `layernorm_ref`).
- `quack.reduce` — `online_softmax_reduce`, `sum_swap_shuffle`.
- `quack.copy_utils` — TMA helpers, gather kernels, ragged-tensor encoders,
  SMEM atom factories (`get_smem_*`), `cvt_copy`, `load_t2r`, `store`.
- `quack.layout_utils` — `transpose_view`, `select`, `permute_*`,
  MMA-accumulator reshapes, `convert_layout_zero_stride`,
  `mma_partition_*`, `copy_partition_*`.
- `quack.utils` — `make_vector`, `store_shared_remote_x4`, `fmin`, `sqrt`,
  `ceil`, `warp_prefix_sum`, `atomic_*`, `issue_clc_query_nomulticast`,
  `load_scalar_or_pointer`.
- `quack.cute_dsl_utils` — the TVM-FFI converter patch
  (`_patched_convert_single_arg`), the arch override
  (`get_device_capacity` / `_parse_arch_str`), the
  `mlir_namedtuple` decorator, and `ParamsBase`.
- `quack.cache_utils` — the persistent `.o` cache (`@jit_cache`,
  `FileLock`); replaced by a simple per-process dict cache.
