# Ascend ReLU squared validation

Measured on November 16, 2026 (Asia/Shanghai). Related task:
[triton-ascend #1803](https://github.com/triton-lang/triton-ascend/issues/1803).
The companion delivery is [GitCode draft PR #800](https://gitcode.com/Ascend/triton-ascend-kernels/merge_requests/800).

## Scope and provenance

The original hardware evidence uses Liger Kernel v0.8.2, CANN 9.1.0,
triton-ascend 3.2.2, torch/torch-npu 2.9.0 (no post suffix), Python 3.11.15,
GCC 13.3.0 on A2 and GCC 12.3.1 on A3. The distribution version is 3.2.2 while
`triton.__version__` reports 3.2.0; both are recorded in the environment logs.

The original implementation started at user commit
`1540851853e2dbd63b4d57061d240c4a0e178177`, followed by the fixed-version stream
compatibility correction. This PR ports those changes onto current upstream
`main` and adds a version guard to preserve the existing path on other NPU
stacks. Original measurements must not be presented as full-suite validation of
current upstream main or its newer default dependencies.

**950 hardware is still being requested. No 950 result or full task acceptance
is claimed.** Upstream merge is also required by the task maintainer.

## Follow-up validation of this upstream port

The final operator, native bridge and two new test files were overlaid onto an
isolated copy of the v0.8.2 task environment. Their hashes match this PR and are
recorded in each platform's `upstream-port-sha256.txt`. This checks the updated
operator and dispatch guard, not all current-main integration or dependencies.

| Platform | Tests | Fresh benchmark configurations passed | Maximum ratio |
| --- | ---: | ---: | ---: |
| A2 | 77 passed | 33/33 | 0.919713 |
| A3 | 77 passed | 31/33 | 1.411221 |

The additional six tests exercise supported, unsupported, post-release and
missing-package dispatch cases. All test/benchmark processes exited zero and
all expected speed/memory pairs were produced. However, the fresh A3
`deepseek_v2_lite` forward ratio is **1.411221**, and backward ratio is
**1.026218**. These are performance failures and remain open Draft blockers.
They are not superseded by the historical passing rounds below. See
[the follow-up summary](ascend_relu_squared_evidence/upstream_port_summary.json)
and the `upstream_port` CSVs / `upstream-port` logs in each platform directory.

## Historical A2/A3 task-environment results

| Platform | Device | Original plus NPU tests | Round 1 maximum Liger/Torch ratio | Round 2 maximum ratio |
| --- | --- | ---: | ---: | ---: |
| A2 | Ascend910B3 | 71 passed | 0.930121 | 0.931844 |
| A3 | Ascend910_9382 | 71 passed | 0.972192 | 0.947694 |

Each round has 33 speed configurations: token_length and model_config sweeps,
each covering forward, backward and full. All 132 platform/round/configuration
comparisons satisfy ratio <= 1.0. Each platform/round also produces all 11 full
memory measurements. The 18 original tests and 53 Ascend regressions pass
without changing the original assertions or tolerances. Original test and
benchmark source hashes are included.

- [Complete before/after comparison CSV](ascend_relu_squared_evidence/comparison.csv)
- [Per-sweep acceptance summary](ascend_relu_squared_evidence/acceptance_summary.json)
- [A2 evidence](ascend_relu_squared_evidence/A2/): both rounds of raw CSVs, tests,
  environment, source hashes and profiling summary.
- [A3 evidence](ascend_relu_squared_evidence/A3/): equivalent independent results.

The original benchmark and its configuration are unchanged. The default benchmark
dtype is BF16; FP16 and FP32 correctness are covered by tests. Its `full` mode
includes random gradient generation, so it is not a bare-kernel timing.
Intermediate performance results are retained: the missing Python NoWait getter initially
failed on the required torch-npu release; using its waiting getter then left one
A3 backward case at ratio 1.075760. The final C++ stream binding resolved that
case in both complete rounds.

## Profiling

The standalone profiler uses `[2048, 4096]` BF16 inputs and pre-generated gradients,
warms up outside tracing, and records ten forward/backward iterations. Profiling
is separate from acceptance timing. Raw traces are available in the companion
GitCode PR; the per-kernel and CPU-event summaries are committed here.

| Event (mean microseconds) | A2 original | A2 optimized | A3 original | A3 optimized |
| --- | ---: | ---: | ---: | ---: |
| NPU forward | 25.624 | 17.164 | 21.957 | 15.149 |
| NPU backward | 34.046 | 35.750 | 29.477 | 30.779 |
| CPU backward autograd event | 231.473 | 63.476 | 191.243 | 30.581 |

PyTorch's reference full path uses six device kernels; the fused implementation
uses one forward and one backward kernel, reducing intermediate traffic and
launches. Flattened tiling improves forward device time. Backward device time
does not decrease in this representative trace; its end-to-end improvement also
comes from lower host dispatch/autograd overhead. CPU and device event times
must not be added because execution is asynchronous.

In the intermediate A3 implementation, the waiting stream getter measured
68.447 us in the backward CPU event, versus 30.581 us with the final binding,
while device time was nearly unchanged (30.807 vs 30.779 us). This supports the
host-wait reduction diagnosis; final performance conclusions use the full rounds.

## Reproduction

Activate the fixed task environment and load the CANN environment first.
From the source checkout:

```bash
export MAX_JOBS=2
export PYTHONPATH="$PWD/src:$PWD"
python -m pytest -q test/transformers/test_relu_squared.py test/transformers/test_relu_squared_ascend.py test/transformers/test_relu_squared_dispatch.py
for round in 1 2; do
  for sweep in token_length model_config; do
    LIGER_BENCH_TARGET="relu_review_${round}_${sweep}" python benchmark/scripts/benchmark_relu_squared.py --sweep-mode "$sweep" --overwrite
  done
done
python benchmark/scripts/profile_relu_squared_ascend.py --provider liger --output results/profile_liger
python benchmark/scripts/profile_relu_squared_ascend.py --provider torch --output results/profile_torch
make checkstyle
```

The task spells the original test path as `tests/test_relu_squared.py`; v0.8.2's
actual path is `test/transformers/test_relu_squared.py`. No replacement of its
assertions or tolerances is necessary. Full repository `make test`, model
convergence tests, and CUDA/XPU hardware regressions are not claimed by these
operator-specific NPU results.
