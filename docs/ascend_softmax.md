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
installed torch-npu headers/libraries and the CANN development headers. Source
CANN’s `set_env.sh` before use so `ASCEND_HOME_PATH` identifies the SDK.

## Kernel and host design

The kernels subtract the row maximum and use FP32 exponential/summation
arithmetic. Backward computes `y * (dy - sum(dy * y))`. Masked row/column tails,
split reductions for irregular widths, and segmented long rows limit wasted
work and on-chip memory requirements.

A2 uses split backward reductions with 8/4/2 rows per tile for 2048/4096/8192
columns and a four-row, three-part reduction for 3584 columns. Forward uses
three-part reductions at widths 3584 and 7168, with a shared reciprocal for
normalization. Narrow tails use two-part reductions with a shared reciprocal. The fused A2 forward path
reduces the maximum in the input dtype, then converts to FP32 for subtraction,
exponential and summation. Bounded offsets use int32; larger tensors retain
int64 indexing.

A3 uses two-row fused backward reductions at 2048 columns. At 4096 columns,
it uses persistent two-row tiles below 2048 rows, fused two-row tiles from 2048
to 4095 rows, and persistent one-row tiles for larger row counts. Two-row paths
use multi-buffering. Configurations are
fixed from measured cold-cache and end-to-end comparisons.

A C++ bridge handles allocation, native autograd and launching; numerical
computation remains in Triton. Contexts are bounded to 128 cached shapes/dtypes/
devices. Saved tensor hooks, version checks, dynamic launch hooks, graph release
and the current stream remain active. The exact torch-npu 2.9.0 stream accessor
is provided by `c10_npu::getCurrentNPUStream(device).stream(false)`.

The typed launcher retains Triton 3.2.2's argument packing, workspace and
profiling logic. Its native path submits the generated callback through
`OpCommand::RunOpApiV2`, avoiding construction of a generic `OpCommand` for each
launch. The callback bridge is passed explicitly rather than installed as a
mutable process-global callback. Python hook dispatch retains the original
backend path. The unprofiled native entry keeps its invariant kernel name and
empty tensor metadata in static storage, reducing callback capture costs.
The bridge passes the cached kernel name by reference, avoiding a string
allocation at the RunOpApiV2 boundary. Both extension builders use the installed
PyTorch C++ ABI. Profiling-enabled launches retain per-call tensor metadata.

Dimension validation and axis movement run in the existing native entry,
reducing per-forward Python metadata calls while preserving scalar, arbitrary-axis
and noncontiguous-input behavior.

Output and gradient allocation call torch-npu's exported ND allocator directly.
Backward checks Python launch hooks under the GIL and, after initialization,
runs the typed C++ entry outside that scope. Hook-enabled launches retain the
backend's Python dispatch. Concurrent stream and composed-autograd tests cover
these paths.

Launch configurations are fixed; first use compiles the matching kernels and
native extension without online tuning. The extension cache name includes its
source and framework versions to prevent reuse of an outdated binary. Warm up
matching shapes before NPU Graph capture. Compilation costs are separate from
steady-state benchmark timing.

## Validation conditions

See [validation and profiling](ascend_softmax_validation.md) and the committed
[current before/after table](ascend_softmax_evidence/dimcache-before-after.csv).

The current optimization runs use `CPU_AFFINITY_CONF=2,npu0:8-31` on both
servers. The original benchmark runs directly, with the same initialization
and affinity settings for Liger and PyTorch. External observations confirm the
main/autograd thread on CPU 8, the task-queue thread on CPU 9 and the release
thread on CPU 10. Do not combine this setting with `taskset`: pre-existing
process affinity can prevent torch-npu's automatic thread binding from taking
effect. Earlier `taskset` trials and their failures remain in the evidence.

These CPU numbers are specific to the test servers. See the validation report
for each revision's results and scheduling conditions. 950 validation is pending
server availability.
