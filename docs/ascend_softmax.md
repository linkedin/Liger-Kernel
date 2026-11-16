# Ascend Softmax

This draft adds shape/dimension handling and a version-gated native launch path
for `LigerSoftmax`. The optimized path supports FP16, BF16 and FP32 forward and
backward, positive/negative `dim`, scalars, empty tensors and noncontiguous input.
Output shape/dtype are preserved and output layout is contiguous ND.

```python
from liger_kernel.transformers.softmax import LigerSoftmax
from liger_kernel.transformers.functional import liger_softmax

y = LigerSoftmax(dim=-1)(x)
z = liger_softmax(x, dim=1)
```

## Version boundary

The private generated-launcher ABI is enabled only for **triton-ascend 3.2.2 and
torch-npu 2.9.0** (local build suffixes are allowed; post releases are excluded).
The implementation is selected once at backend initialization; only the validated stack loads `_softmax_optimized.py`.
Other NPU stacks use the existing upstream Ascend kernels through the dimension
wrapper. Existing GPU kernels, including upstream's int64 indexing fixes, are
preserved. Upstream dependency pins are not downgraded.

Hardware evidence uses the task's Liger Kernel v0.8.2 baseline, CANN 9.1.0,
torch/torch-npu 2.9.0, triton-ascend 3.2.2 and Python 3.11.15. It is not validation
of all current-main integrations or its newer default NPU dependencies. Use
`pip install --no-deps -e .` inside a prepared fixed-version environment and
install Ninja separately. The native extension needs a C++17 compiler and
installed torch-npu headers/libraries.

## Kernel and host design

The kernels subtract the row maximum and use FP32 exponential/reduction
arithmetic. Backward computes `y * (dy - sum(dy * y))`. Masked row/column tails,
split reductions for irregular widths, and segmented long rows limit wasted
work and on-chip memory requirements.

A2 uses split backward reductions with 8/4/2 rows per tile for 2048/4096/8192
columns; narrow-tail forward reductions avoid power-of-two padding. A3 uses
persistent per-vector-core row loops for medium widths. For 4096 columns and
fewer than 2048 rows, it uses one-row fused tiles instead. Multi-buffer choices
are part of the measured dispatch strategy.

A C++ bridge handles allocation, native autograd and launching; numerical
computation remains in Triton. Contexts are bounded to 128 cached shapes/dtypes/
devices. Saved tensor hooks, version checks, dynamic launch hooks, graph release
and the current stream remain active. The exact torch-npu 2.9.0 stream accessor
is provided by `c10_npu::getCurrentNPUStream(device).stream(false)`.

For at least 1,048,576 elements and at most 8192 columns, first use compares
candidate tiles/buffering in two opposite-order rounds with 100 ms sampling.
A candidate replaces the default only with at least 10% measured improvement.
Temporary tuning tensors are released, input/RNG state is preserved, and later
calls reuse the selected context. Cache eviction can cause retuning. Tuning is
skipped for launch hooks, compile-only and tensor profiling. Warm up matching
shapes before NPU Graph capture. Initial compilation/tuning costs are separate
from steady-state benchmark timing.

## Validation conditions

See [validation and profiling](ascend_softmax_validation.md) and the committed
[before/after table](ascend_softmax_evidence/performance.md).

**A2's passing performance results require binding the entire benchmark process,
including PyTorch, to CPU 0–31 on this server (NPU-local NUMA node 0).** Unbound
A2 runs had failures and are also retained. This CPU list is machine-specific;
query the PCI device's `numa_node` and `local_cpulist` on another server. A3 used
default CPU scheduling. **950 is still being requested and has not been tested.**
