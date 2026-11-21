# Ascend Softmax validation and profiling

Related task: [triton-ascend #1802](https://github.com/triton-lang/triton-ascend/issues/1802).
Measurements below were collected on November 16, 2026 (Asia/Shanghai).
This is a Draft; 950 hardware is still being requested and full task acceptance
is not claimed.

## Current submitted optimization checkpoint

The common-source `dimcache-port` checkpoint passes **143 tests on each of A2
and A3**: 24 unchanged original tests and 119 supplementary cases. Two complete
runs of the original benchmark produced:

| Platform | Round 1 | Round 2 | Runtime thread affinity |
| --- | ---: | ---: | --- |
| A2 | 32/33 | 32/33 | Main/autograd: CPU 8; queue: CPU 9; release: CPU 10 |
| A3 | 33/33 | 33/33 | Main/autograd: CPU 8; queue: CPU 9; release: CPU 10 |

Each round contains 33 speed pairs and 11 memory pairs; all benchmark commands
exit 0. A3's maximum Liger/PyTorch median latency ratios are 0.988784 and 0.987453.
A2 has two remaining comparisons:

| Round | Case | Mode | Liger/PyTorch median ratio |
| --- | --- | --- | ---: |
| 1 | Llama2-7B | forward | 1.010522 |
| 2 | B*T=1024 | forward | 1.048751 |

Both platforms use `CPU_AFFINITY_CONF=2,npu0:8-31` without `taskset`. The external
`record_affinity.py` monitor records actual thread affinity without modifying the
benchmark. Earlier runs combining this variable with `taskset` restricted all
threads to the specified CPU set; they did not establish fine-grained runtime
thread binding. A2 CPU 0–31 is NPU-local; A3 topology reports no NPU CPU locality.

The implementation now caches the launch name across the RunOpApiV2 bridge and
handles dimension validation and movement in the existing native entry. These
changes reduce per-call host work. Kernel changes include A2 input-dtype maximum
reduction followed by FP32 exponentiation and accumulation, shared reciprocals,
bounded int32/int64 offsets, and an A3 fused backward tile for medium row counts.
A3's actual profiler path also completes successfully. LSE compilation and
single-buffer trials showed no stable benefit and were not adopted.

The six-file source hashes match between both servers and the local checkout.
The published `9aab386` implementation was also benchmarked under the same CPU
thread binding: A2 passed 30/33 and A3 passed 26/33. Baseline source hashes were
checked against Git objects, and the task benchmark/test hashes remain unchanged.

Raw CSVs, test logs, exit codes, source hashes and affinity observations are
retained under `dimcache-port` / `dimcache_port` in
[A2 evidence](ascend_softmax_evidence/A2/) and
[A3 evidence](ascend_softmax_evidence/A3/). The
[all-trial summary](ascend_softmax_evidence/optimization_summary.json) retains
both passing and failing runs; [additional diagnostics](ascend_softmax_evidence/DIAGNOSTICS.md)
record rejected experiments. Intermediate raw files are preserved in each
platform's `optimization-trials.tar.gz`, indexed by `optimization-trials-manifest.json`. A2 optimization remains in progress. This checkpoint is submitted for preliminary
review in Draft PR #1635 with the remaining performance comparisons listed above.

The fixed software stack and v0.8.2 operator-overlay validation scope are
specified below. 950 validation is pending server availability.

## Historical fixed-environment evidence

The task implementation is commit `79353a8b7d72b61dfec479dede28da72ef6f9d63`,
based on the user's `feat/ascend-softmax` implementation and Liger Kernel v0.8.2.
Its optimized Python source was the starting point for `_softmax_optimized.py`
in this upstream port. Subsequent experiments modify this source. The version-dispatch facade and dimension-aware legacy
fallback are new; historical data is not full validation of this PR's integration
with current main or its newer NPU dependency stack.

| Platform | Device | Tests | Round 1 max ratio | Round 2 max ratio | CPU affinity |
| --- | --- | ---: | ---: | ---: | --- |
| A2 | Ascend910B3 | 123 passed | 0.994920 | 0.998461 | CPU 0–31, NUMA node 0 |
| A3 | Ascend910_9382 | 123 passed | 0.982446 | 0.981740 | Default |

Each round contains 33 forward/backward/full speed pairs and 11 full-memory
pairs. All 132 final platform/round/speed comparisons satisfy Liger/PyTorch
median ratio <= 1. The tests consist of 24 unmodified original cases and 99
supplementary cases. Original assertions, tolerances and benchmark configurations
are unchanged. Benchmark dtype is BF16; FP16/FP32 correctness is covered by tests.
The original `full` mode includes random-gradient generation.

Environment: CANN 9.1.0, torch 2.9.0, torch-npu 2.9.0 (no post suffix),
triton-ascend distribution 3.2.2 and Python 3.11.15. The installed Triton module
reports `3.2.0`; the distribution metadata reports `3.2.2`. Both are recorded.

Evidence includes the [complete before/after table](ascend_softmax_evidence/performance.md),
[all-stage comparison CSV](ascend_softmax_evidence/comparison.csv),
[all-stage summary](ascend_softmax_evidence/benchmark_summary.json),
and per-platform raw final CSVs, pytest logs, exit codes, source hashes, environment
logs and profiling trace archives in [A2](ascend_softmax_evidence/A2/) and
[A3](ascend_softmax_evidence/A3/). Intermediate stages in the comparison are not
acceptance results. Diagnostic warmup/context probes do not replace the official
benchmark. A2 unbound `calibrated` CSVs are included to show the remaining failures
without affinity; passing bound results do not establish unbound acceptance.

## Published upstream-port checkpoint (`9aab386`)

The final operator files were overlaid onto the same prepared v0.8.2 task
environment, rather than installing all of current main. This checks the changed
operator and fallback integration on the fixed stack, not the newer upstream
NPU dependencies. Both platforms passed **130 tests** (24 original and 106
supplementary). The original benchmark script and assertions were unchanged.
The recorded source hashes identify that published checkpoint.

| Platform | Final port speed pairs passing | Maximum Liger/PyTorch ratio | Condition |
| --- | ---: | ---: | --- |
| A2 | 33/33 | 0.987295 | Whole-process CPU 0–31 affinity |
| A3 | 32/33 | 1.059487 | Default scheduling |

Each platform also produced 11 full-memory pairs. The unresolved A3 case is
`token_length`, backward, `B*T=1024`: Liger is approximately **5.95% slower** than
PyTorch. It blocks a claim of complete performance acceptance for this port.
Historical passing rounds above do not supersede this result.

The first port used a per-call dispatch wrapper: A2 passed 33/33 and A3 passed
30/33, with failing backward cases `B*T=1024`, `qwen2.5_7b`, and
`deepseek_v2_lite`. The final port selects the implementation at backend module
initialization. Both runs are retained as `upstream_port` and `upstream_port2`
CSVs, logs and exit codes; [the summary](ascend_softmax_evidence/upstream_port_summary.json)
contains every ratio. These are successive code revisions, not two repeatability
runs of the final code. Further A3 optimization/repeatability, 950 validation,
and validation on current upstream dependencies remain open before acceptance.

## Profiling and root causes

Profiling uses `[2048, 4096]` BF16 inputs, pre-generated gradients, warmup outside
tracing and ten active iterations. A2 upstream and optimized profiles both use
CPU 0–31; A3 profiles use default scheduling. Official benchmark timings do not
include profiling.

| Platform / implementation | NPU forward us | NPU backward us | CPU backward node us |
| --- | ---: | ---: | ---: |
| A2 / v0.8.2 | 35.572 | 29.662 | 235.067 |
| A2 / optimized | 30.558 | 28.850 | 48.897 |
| A2 / PyTorch | 60.800 | 57.116 | 36.284 |
| A3 / v0.8.2 | 30.621 | 26.075 | 161.970 |
| A3 / optimized | 26.539 | 35.663 | 26.952 |
| A3 / PyTorch | 54.776 | 52.128 | 25.785 |

CPU-node events are `LigerSoftmaxFunctionBackward`, `LigerSoftmaxBackward` and
`SoftmaxBackward0`; nested engine events are not added again. CPU and NPU
execution overlap, so these columns must not be summed. For this representative
A3 shape, device backward time increases; lower host dispatch/autograd overhead
provides the main improvement. End-to-end conclusions use the complete benchmark.

The changes address Python launch/metadata overhead with bounded native contexts,
wasted padded work with split reductions, and per-core work distribution with
fused or persistent tiles. FP32 reductions preserve accuracy. Multi-buffer and
sub-block binding experiments were measured separately; compilation-failing
candidates and FMA rewrites without observed benefit were excluded. That historical revision used a 100 ms tuning window and a 10% replacement
threshold. The submitted implementation uses fixed configurations. NUMA-local CPU binding is an explicit A2 measurement condition.

With an existing disk compiler cache, first complete forward/backward calls for
the representative shape measured 1081.546 ms on A2 and 1084.367 ms on A3;
subsequent single synchronized wall-clock observations were 0.328 and 0.241 ms.
First-call extra peak allocation was 251,660,288 bytes versus 33,555,456 bytes
when reusing the context. Startup includes extension loading, tuning and timer
scratch memory; these are not cold-disk compilation measurements or substitutes
for the official steady-state benchmark.

## Reproduction

In the prepared task environment, load CANN and run from the checkout:

```bash
export MAX_JOBS=2
export PYTHONPATH="$PWD/src:$PWD"
export CPU_AFFINITY_CONF=2,npu0:8-31
export TASK_HARNESS_ROOT=/workspace/softmax1802/user
python -m pytest -q test/transformers/test_softmax.py test/transformers/test_softmax_ascend.py
for round in 1 2; do
  for sweep in token_length model_config; do
    LIGER_BENCH_TARGET="softmax_review_${round}_${sweep}" python "$TASK_HARNESS_ROOT/benchmark/scripts/benchmark_softmax.py" --sweep-mode "$sweep" --overwrite
  done
done
python benchmark/scripts/profile_softmax_ascend.py --provider liger --output results/profile_liger
python benchmark/scripts/profile_softmax_ascend.py --provider torch --output results/profile_torch
make checkstyle
```

These are the recorded A2/A3 affinity settings. Do not add `taskset`; the runtime
binds the main/autograd, queue and release threads to CPUs 8, 9 and 10 respectively.
Both providers and the published baseline use the same settings. CPU numbers
must be selected for the actual server topology when reproducing elsewhere.
`TASK_HARNESS_ROOT` points to the unchanged task checkout on the validation
servers; its benchmark/configuration/test hashes are recorded in
[task_harness_sha256.json](ascend_softmax_evidence/task_harness_sha256.json).
The prepared overlay contains the same original 24 tests plus the supplementary
Ascend cases. Current main's benchmark utilities differ from this pinned harness.

The issue's `tests/test_softmax.py` corresponds to v0.8.2's actual
`test/transformers/test_softmax.py`. Full-repository `make test`, model convergence,
CUDA/XPU hardware validation, the newer upstream NPU stack and 950 remain outside
these operator-specific fixed-environment results.
