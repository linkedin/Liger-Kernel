# `device_map="auto"` and Triton Device Context Investigation

## Context

Liger RMSNorm can fail in multi-GPU inference/training flows that use
HuggingFace `device_map="auto"`. The suspected failure mode is:

1. `device_map="auto"` dispatches different model modules to different CUDA
   devices.
2. A Liger patched module receives tensors that live on a non-current CUDA
   device, for example `cuda:1`.
3. The Python process' active CUDA device can still be `cuda:0`.
4. Liger launches a Triton kernel directly with `kernel[grid](...)` and no
   surrounding device context guard.
5. Triton uses the active device to select the launch stream, compilation
   target, and kernel cache entry, instead of inferring the target device from
   tensor arguments.

The practical result is that PyTorch allocations can be on the correct tensor
device while the Triton compile/launch path targets the wrong active device.

This is not specific to RMSNorm conceptually. RMSNorm is a visible reproducer
because it is commonly used in patched HF models, but the launch pattern exists
across many Liger ops.

## Evidence

### Related Kernels-Community Patch

HuggingFace `kernels-community` PR 1029 applies the same narrow fix to its
vendored Liger RMSNorm implementation:

- PR: https://github.com/huggingface/kernels-community/pull/1029/changes
- Adds `device_context(device)` in `liger_kernels/utils.py`.
- Imports that helper in `liger_kernels/rms_norm.py`.
- Wraps both RMSNorm forward launch branches with `with device_context(X.device):`.
- Wraps both RMSNorm backward launch branches with `with device_context(X.device):`.

The helper in that PR is backend-generic:

```python
@contextmanager
def device_context(device: torch.device):
    """Context manager that sets the active device for any backend (cuda, xpu, etc.)."""
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "device"):
        with backend.device(device):
            yield
    else:
        yield
```

That pattern is directly applicable to this repository's default RMSNorm
implementation. It is also preferable to writing CUDA-only guards at each call
site because it keeps XPU/NPU-capable code paths structurally consistent.

### Hub Reproducer

A small hub-based repro demonstrates the underlying issue even when RMSNorm is
not the clearest local trigger. The repro loads Liger kernels through
`kernels.LayerRepository`, runs each kernel first on `cuda:0`, then on
`cuda:1`, and compares the output to a torch reference.

The order matters: calling the primary device first populates Triton compiled
or context caches on device 0. The subsequent call uses tensors on `cuda:1`
while the active device/cache state can still be tied to `cuda:0`.

Observed/expected cases from the repro:

- `LigerRMSNorm`: released v2 can fail or behave inconsistently on `cuda:1`;
  PR 1029 fixes the RMSNorm path.
- `LigerSiLUMul`: released v2 fails on `cuda:1`; PR 1029 is expected to keep
  failing until the same device-context guard is propagated to SwiGLU.
- `LigerGELUMul`: released v2 fails on `cuda:1`; PR 1029 is expected to keep
  failing until the same guard is propagated to GeGLU.

This is important because it confirms the bug is not only about RMSNorm logic.
RMSNorm is one affected host launch path; SwiGLU and GeGLU show the same
active-device problem for simpler elementwise Triton kernels.

### Triton Runtime Behavior

In the local environment, `triton.runtime.jit.JITFunction.run` starts by
querying the active device and stream:

```python
device = driver.active.get_current_device()
stream = driver.active.get_current_stream(device)
```

It then uses that `device` to select the device cache and compilation target:

```python
kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
```

This confirms that the active device is a real input to compile and launch.
The launch call itself does not derive the target device from the tensors passed
to the Triton kernel.

### RMSNorm Launch Sites

The default Triton RMSNorm implementation allocates outputs on `X.device`, but
launches kernels without a context guard.

Forward:

- `src/liger_kernel/ops/rms_norm.py:454`
- `src/liger_kernel/ops/rms_norm.py:475`

Backward:

- `src/liger_kernel/ops/rms_norm.py:535`
- `src/liger_kernel/ops/rms_norm.py:562`

Relevant pattern:

```python
Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

_rms_norm_forward_kernel[(n_rows,)](...)
```

The output tensors are correctly allocated on `X.device`, but the Triton launch
uses whatever CUDA device is active at the time.

### Transformers Behavior

The sibling `../transformers` checkout was inspected for RMSNorm and
`device_map="auto"` behavior.

HF RMSNorm implementations, such as Llama and Gemma RMSNorm, are plain torch
operations. They follow input and parameter tensor devices naturally and do not
need an explicit Triton launch guard.

This makes the Liger issue visible only after replacing HF modules with Liger
Triton-backed modules. The HF dispatcher can execute a module whose parameters
and inputs are on `cuda:1` without necessarily changing the process-global
active CUDA device to `cuda:1`.

### Existing Liger Helpers

`src/liger_kernel/ops/utils.py` currently provides shared helpers such as
`ensure_contiguous`, `calculate_settings`, `get_amp_custom_fwd_bwd`, and
device-type inference utilities. There is no shared helper that temporarily
sets the active device for kernel launches.

`ensure_contiguous` only makes tensor arguments contiguous before autograd
function calls. It does not affect the active device.

### Scope Survey

A launch-site survey under `src/liger_kernel/ops`, excluding the Ascend backend,
found 89 direct launch sites across 32 files. Examples include:

- `rms_norm.py`
- `fused_add_rms_norm.py`
- `modulated_rms_norm.py`
- `poly_norm.py`
- `layer_norm.py`
- `swiglu.py`
- `geglu.py`
- `cross_entropy.py`
- `fused_linear_cross_entropy.py`
- `fused_linear_jsd.py`
- `jsd.py`
- `kl_div.py`
- `rope.py`
- `llama4_rope.py`
- `qwen2vl_mrope.py`
- `softmax.py`
- `sparsemax.py`
- `group_norm.py`
- `fused_moe.py`
- `fused_neighborhood_attention.py`
- `multi_token_attention.py`
- `mhc.py`
- `experimental/embedding.py`
- `experimental/mm_int8int2.py`

The broad pattern is direct host-side launch:

```python
some_triton_kernel[grid](...)
```

without:

```python
with device_context(tensor.device):
    some_triton_kernel[grid](...)
```

CuTe/Cutile paths should also be audited. Some of them call
`torch.cuda.current_stream()` or compile/launch through their own host APIs.
Those paths have the same class of risk if stream acquisition happens while the
wrong CUDA device is active.

## Impact Assessment

### Confirmed High-Risk Areas

RMSNorm is high risk because:

- it is commonly monkey-patched into HF models;
- it is used repeatedly throughout transformer decoder layers;
- `device_map="auto"` commonly shards decoder layers across multiple GPUs;
- the implementation has unguarded forward and backward Triton launches.

SwiGLU and GeGLU are also confirmed high-risk areas because the hub repro shows
the same non-primary-device failure pattern for their Liger autograd functions.
They are common transformer MLP components and should be patched immediately
after RMSNorm.

### Likely Affected Areas

Any Liger op that launches a Triton kernel while tensors may live on a
non-current CUDA device is likely affected. This includes both model layers and
loss kernels.

For `device_map="auto"` inference, the most visible affected ops are model
components used in patched transformer blocks, such as RMSNorm, LayerNorm,
SwiGLU/GEGLU, RoPE, and fused add RMSNorm.

For training, backward launches and loss kernels are also affected.

### Backends To Treat Separately

The Ascend backend should not be blindly swept into a CUDA-focused fix. NPU
launch semantics and context APIs are different and should be audited on their
own terms.

The same applies to XPU: the helper can support `torch.xpu.device`, but test
coverage should be backend-specific.

## Plan Of Action

### 1. Add A Shared Device Context Helper

Add a small helper to `src/liger_kernel/ops/utils.py`.

Proposed API:

```python
from contextlib import contextmanager


@contextmanager
def device_context(device):
    device = torch.device(device)
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "device"):
        with backend.device(device):
            yield
    else:
        yield
```

Open design questions:

- Whether to accept only `torch.device`/tensor devices or also tensors directly.
- Whether NPU support should use the generic backend lookup immediately or be
  handled after a backend-specific audit.
- Whether to include a stricter assertion that all tensor arguments expected to
  be colocated are on the same device.

### 2. Patch RMSNorm First

Wrap all default Triton RMSNorm launch sites with the helper.

Forward:

```python
with device_context(X.device):
    _rms_norm_forward_kernel[(n_rows,)](...)
```

and:

```python
with device_context(X.device):
    _block_rms_norm_forward_kernel[(triton.cdiv(n_rows, BLOCK_ROW),)](...)
```

Backward:

```python
with device_context(X.device):
    _rms_norm_backward_kernel[grid](...)
```

and:

```python
with device_context(X.device):
    _block_rms_norm_backward_kernel[grid](...)
```

Use `X.device` as the canonical device because `X` is the saved input and the
main tensor all RMSNorm work is based on. Outputs are allocated from `X.device`,
and `W` is expected to be colocated with the hidden dimension for the patched
module.

### 3. Add A Focused Multi-GPU Regression Test

Add a test that directly reproduces the active-device mismatch without needing
a full HF model:

1. Skip unless `torch.cuda.device_count() >= 2`.
2. Set the active device to `cuda:0`.
3. Create RMSNorm input and weight tensors on `cuda:1`.
4. Call Liger RMSNorm forward and backward.
5. Compare output and gradients to a torch/HF reference.
6. Assert output and gradients remain on `cuda:1`.

This should be separate from the existing DTensor RMSNorm test. The current
DTensor test is about sharding behavior and is marked `xfail` unless many GPUs
are available. The new test only needs two GPUs and targets this specific bug.

### 4. Patch SwiGLU And GeGLU Next

After RMSNorm, patch the two activation-multiply kernels from the repro:

- `src/liger_kernel/ops/swiglu.py`
- `src/liger_kernel/ops/geglu.py`

Use the same pattern:

```python
with device_context(a.device):
    _swiglu_forward_kernel[(n_rows,)](...)
```

and equivalent guards around backward and tiled launch branches.

These are good second targets because they are simpler than normalization ops
and demonstrate that the fix is general rather than RMSNorm-specific.

### 5. Sweep Other Default Triton Ops

After RMSNorm is fixed and covered, apply the same helper around all remaining
default Triton launch sites under `src/liger_kernel/ops`, excluding vendor
backend directories until they are audited separately.

Use a consistent rule:

- guard the launch with the primary input tensor's device;
- when an op allocates an output first and input ownership is ambiguous, guard
  with the output device;
- keep all launches in a function under the same guard when they are known to
  operate on the same device.

Candidate files from the initial survey:

- `src/liger_kernel/ops/attn_res.py`
- `src/liger_kernel/ops/cross_entropy.py`
- `src/liger_kernel/ops/dyt.py`
- `src/liger_kernel/ops/experimental/embedding.py`
- `src/liger_kernel/ops/experimental/mm_int8int2.py`
- `src/liger_kernel/ops/fused_add_rms_norm.py`
- `src/liger_kernel/ops/fused_linear_cross_entropy.py`
- `src/liger_kernel/ops/fused_linear_jsd.py`
- `src/liger_kernel/ops/fused_moe.py`
- `src/liger_kernel/ops/fused_neighborhood_attention.py`
- `src/liger_kernel/ops/geglu.py`
- `src/liger_kernel/ops/group_norm.py`
- `src/liger_kernel/ops/grpo_loss.py`
- `src/liger_kernel/ops/jsd.py`
- `src/liger_kernel/ops/kl_div.py`
- `src/liger_kernel/ops/layer_norm.py`
- `src/liger_kernel/ops/llama4_rope.py`
- `src/liger_kernel/ops/mhc.py`
- `src/liger_kernel/ops/modulated_rms_norm.py`
- `src/liger_kernel/ops/multi_token_attention.py`
- `src/liger_kernel/ops/poly_norm.py`
- `src/liger_kernel/ops/qwen2vl_mrope.py`
- `src/liger_kernel/ops/relu_squared.py`
- `src/liger_kernel/ops/rope.py`
- `src/liger_kernel/ops/softmax.py`
- `src/liger_kernel/ops/sparsemax.py`
- `src/liger_kernel/ops/swiglu.py`
- `src/liger_kernel/ops/tvd.py`
- `src/liger_kernel/ops/vocab_parallel_cross_entropy.py`

### 6. Audit CuTe/Cutile Paths

Audit non-standard launch paths separately:

- `src/liger_kernel/ops/cutedsl/ops/rms_norm.py`
- `src/liger_kernel/ops/cutedsl/ops/cross_entropy.py`
- `src/liger_kernel/ops/cutile/ops/*`

These paths often acquire a stream through `torch.cuda.current_stream()` before
launch. They need the device context active before stream acquisition, not only
around the final launch call.

### 7. Validate

Minimum validation after the RMSNorm patch:

```bash
pytest test/transformers/test_rms_norm.py
make checkstyle
```

Recommended validation after the broader sweep:

```bash
pytest test/transformers/test_rms_norm.py
pytest test/transformers/test_fused_add_rms_norm.py
pytest test/transformers/test_layer_norm.py
pytest test/transformers/test_swiglu.py
pytest test/transformers/test_geglu.py
pytest test/transformers/test_cross_entropy.py
pytest test/transformers/test_rope.py
make checkstyle
```

Multi-GPU validation should run on a host with at least two CUDA devices. The
most important regression is: tensors on `cuda:1`, active device `cuda:0`, Liger
kernel executes and returns results on `cuda:1`.

## Suggested Implementation Order

1. Add `device_context` helper and tests for the helper where practical.
2. Patch RMSNorm only.
3. Add the two-GPU RMSNorm mismatch regression test.
4. Patch SwiGLU and GeGLU, using the hub repro as a confirmation case.
5. Run RMSNorm, SwiGLU, GeGLU tests and checkstyle.
6. Sweep high-use transformer block ops: fused add RMSNorm, LayerNorm, RoPE.
7. Sweep loss kernels and remaining default ops.
8. Audit CuTe/Cutile and vendor backend launch paths.
9. Run broader unit tests and at least one end-to-end HF `device_map="auto"`
   smoke test with Liger monkey patching enabled.

## End-To-End Smoke Test Idea

Use a small HF model that can be loaded with `device_map="auto"` across two
GPUs, enable the relevant Liger monkey patch, and run one forward pass with
inputs moved through the normal HF pathway.

The test should not depend on a large public model for unit CI. For local or
nightly multi-GPU validation, a tiny model with enough layers to split across
devices is sufficient.

Expected properties:

- no invalid device access;
- no Triton compile/launch against the wrong CUDA device;
- outputs match the unpatched HF model within expected tolerances;
- the test still passes when active CUDA device is intentionally set to a
  different GPU before calling the model.
