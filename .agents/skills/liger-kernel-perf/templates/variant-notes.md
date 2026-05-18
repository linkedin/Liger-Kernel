# Variant {variant_number}: {variant_title}

> Kernel: `{kernel}`
> Date: {date}
> Parent variant: {parent_variant}

## Hypothesis

**Target bottleneck:** {target_bottleneck}

**Why this change should help:**

{hypothesis_reasoning}

**References from prior variants:**

{prior_variant_references}

## Changes

{changes_list}

## Expected Outcome

**Predicted improvement:** {predicted_improvement}

**Reasoning:** {prediction_reasoning}

## Actual Results

### Speed (median ms)

| Metric | x_value | Baseline (v0) | This Variant (v{variant_number}) | Delta % |
|--------|---------|---------------|----------------------------------|---------|
| Speed forward | {x_val_1} | {baseline_fwd_1} | {variant_fwd_1} | {delta_fwd_1}% |
| Speed forward | {x_val_2} | {baseline_fwd_2} | {variant_fwd_2} | {delta_fwd_2}% |
| Speed backward | {x_val_1} | {baseline_bwd_1} | {variant_bwd_1} | {delta_bwd_1}% |
| Speed backward | {x_val_2} | {baseline_bwd_2} | {variant_bwd_2} | {delta_bwd_2}% |
| Speed full | {x_val_1} | {baseline_full_1} | {variant_full_1} | {delta_full_1}% |
| Speed full | {x_val_2} | {baseline_full_2} | {variant_full_2} | {delta_full_2}% |

### Memory (median MB)

| Metric | x_value | Baseline (v0) | This Variant (v{variant_number}) | Delta % |
|--------|---------|---------------|----------------------------------|---------|
| Memory full | {x_val_1} | {baseline_mem_1} | {variant_mem_1} | {delta_mem_1}% |
| Memory full | {x_val_2} | {baseline_mem_2} | {variant_mem_2} | {delta_mem_2}% |

## Guardrail Checks

| Check | Threshold | Result | Status |
|-------|-----------|--------|--------|
| Non-target metric regression | <5% worse | {nontarget_result} | {nontarget_status} |
| Cross-pass regression | <10% on one pass to marginally improve other | {crosspass_result} | {crosspass_status} |
| Smoke test (single shape, float32, fwd+bwd) | Pass | {smoke_result} | {smoke_status} |

## Verdict

**{verdict}**

{verdict_explanation}

## Learnings

{learnings}

---

*Variant code: `optimization/{kernel}/{kernel}_v{variant_number}.py`*
*Benchmark data: `optimization/{kernel}/benchmarks/v{variant_number}_results.csv`*
