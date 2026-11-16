# Ascend ReLU squared

This change provides fused Triton-Ascend forward and backward kernels for
`LigerReLUSquared`, with FP16, BF16 and FP32 support. The reference is
`torch.square(torch.relu(x))`.

## Supported optimization stack

The native launcher bridge uses private generated-launcher interfaces. It is
therefore enabled only with **triton-ascend 3.2.2 and torch-npu 2.9.0** (a local
version suffix is allowed; a post release is not assumed compatible). Other
versions retain the existing upstream ReLU squared implementation. The C++
extension and torch-npu imports are lazy and are not required on CUDA or XPU.

The hardware measurements used CANN 9.1.0, PyTorch 2.9.0, Python 3.11.15 and a
Liger Kernel v0.8.2 source baseline. Current upstream `main` declares a newer
NPU dependency stack; this PR does not downgrade those dependencies or claim
that the native bridge has been validated against them. Install an existing
fixed-version task environment with `pip install --no-deps -e .` and install
`ninja>=1.10` separately when using that environment.

## Implementation

The NPU kernels flatten contiguous elements, use 8192-element blocks, mask the
last block and calculate offsets in int64. Forward explicitly propagates NaNs.
Backward preserves the reference low-precision rounding order and nonpositive
input gradient gating. Scalars, empty inputs, noncontiguous inputs and gradients
are covered by the NPU regression tests.

The C++ bridge handles allocation, autograd saved variables, version checks and
launching. All numerical computation remains in Triton. A bounded 128-entry LRU
cache retains compiled launch contexts, not request tensors or stream pointers.
Each call reads the current stream; the bridge preserves dynamic launch hooks,
saved tensor hooks, storage lifetime checks and graph replay after warmup.

The fixed torch-npu 2.9.0 release lacks the Python NoWait stream getter. The bridge
uses `c10_npu::getCurrentNPUStream(device).stream(false)` and links against the
installed torch-npu headers and library. The generated Triton launcher retains
its task-queue submission logic. First use compiles both the extension and Triton
kernels; this startup cost is excluded by the benchmark's normal warmup.

## Validation and remaining work

See [the validation report](ascend_relu_squared_validation.md) for A2/A3 raw CSVs,
logs, profiling summaries and commands. **The 950 server is still being requested;
there are no 950 correctness or performance results.** The PR remains a draft
pending platform coverage and upstream review of the integration strategy.
