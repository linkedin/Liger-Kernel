# Variant Comparison: {kernel}

> GPU: {gpu_name} ({gpu_arch})
> Optimization Goal: {optimization_goal}
> Total Variants Tried: {total_variants}
> Date: {date}

## Master Comparison Table

Values shown are medians at x_value = {representative_x_val}.

| Variant | Strategy | Speed Fwd (ms) | Speed Bwd (ms) | Speed Full (ms) | Memory Full (MB) | Verdict |
|---------|----------|-----------------|-----------------|------------------|-------------------|---------|
| v0 (baseline) | -- | {v0_fwd} | {v0_bwd} | {v0_full} | {v0_mem} | BASELINE |
| v1 | {v1_strategy} | {v1_fwd} | {v1_bwd} | {v1_full} | {v1_mem} | {v1_verdict} |
| v2 | {v2_strategy} | {v2_fwd} | {v2_bwd} | {v2_full} | {v2_mem} | {v2_verdict} |
| v3 | {v3_strategy} | {v3_fwd} | {v3_bwd} | {v3_full} | {v3_mem} | {v3_verdict} |
| ... | ... | ... | ... | ... | ... | ... |

### Legend

- Improvements over baseline are marked with a `-` prefix in delta (lower is better for speed/memory)
- Regressions over baseline are marked with a `+` prefix in delta
- ACCEPTED: meets guardrails and improves target metric
- REJECTED: fails guardrails or regresses target metric
- MIXED: improves some metrics but regresses others within acceptable bounds

## Per-x-value Breakdown (Top 3 Variants)

### Baseline (v0) vs {winner_variant} (winner) vs {runner_up_variant} (runner-up)

#### Speed Forward (ms)

| x_value | v0 (baseline) | {winner_variant} | Delta % | {runner_up_variant} | Delta % |
|---------|---------------|-------------------|---------|----------------------|---------|
| {x_val_1} | {v0_fwd_1} | {winner_fwd_1} | {winner_fwd_delta_1}% | {runner_fwd_1} | {runner_fwd_delta_1}% |
| {x_val_2} | {v0_fwd_2} | {winner_fwd_2} | {winner_fwd_delta_2}% | {runner_fwd_2} | {runner_fwd_delta_2}% |
| {x_val_3} | {v0_fwd_3} | {winner_fwd_3} | {winner_fwd_delta_3}% | {runner_fwd_3} | {runner_fwd_delta_3}% |
| ... | ... | ... | ... | ... | ... |

#### Speed Backward (ms)

| x_value | v0 (baseline) | {winner_variant} | Delta % | {runner_up_variant} | Delta % |
|---------|---------------|-------------------|---------|----------------------|---------|
| {x_val_1} | {v0_bwd_1} | {winner_bwd_1} | {winner_bwd_delta_1}% | {runner_bwd_1} | {runner_bwd_delta_1}% |
| {x_val_2} | {v0_bwd_2} | {winner_bwd_2} | {winner_bwd_delta_2}% | {runner_bwd_2} | {runner_bwd_delta_2}% |
| {x_val_3} | {v0_bwd_3} | {winner_bwd_3} | {winner_bwd_delta_3}% | {runner_bwd_3} | {runner_bwd_delta_3}% |
| ... | ... | ... | ... | ... | ... |

#### Speed Full (ms)

| x_value | v0 (baseline) | {winner_variant} | Delta % | {runner_up_variant} | Delta % |
|---------|---------------|-------------------|---------|----------------------|---------|
| {x_val_1} | {v0_full_1} | {winner_full_1} | {winner_full_delta_1}% | {runner_full_1} | {runner_full_delta_1}% |
| {x_val_2} | {v0_full_2} | {winner_full_2} | {winner_full_delta_2}% | {runner_full_2} | {runner_full_delta_2}% |
| {x_val_3} | {v0_full_3} | {winner_full_3} | {winner_full_delta_3}% | {runner_full_3} | {runner_full_delta_3}% |
| ... | ... | ... | ... | ... | ... |

#### Memory Full (MB)

| x_value | v0 (baseline) | {winner_variant} | Delta % | {runner_up_variant} | Delta % |
|---------|---------------|-------------------|---------|----------------------|---------|
| {x_val_1} | {v0_mem_1} | {winner_mem_1} | {winner_mem_delta_1}% | {runner_mem_1} | {runner_mem_delta_1}% |
| {x_val_2} | {v0_mem_2} | {winner_mem_2} | {winner_mem_delta_2}% | {runner_mem_2} | {runner_mem_delta_2}% |
| {x_val_3} | {v0_mem_3} | {winner_mem_3} | {winner_mem_delta_3}% | {runner_mem_3} | {runner_mem_delta_3}% |
| ... | ... | ... | ... | ... | ... |

## Winner Declaration

**Winner: {winner_variant} -- {winner_title}**

**Justification:**

{winner_justification}

## Cumulative Learnings

### What Worked

{what_worked}

### What Did Not Work

{what_did_not_work}

### Why

{cumulative_reasoning}

---

*Comparison saved to `optimization/{kernel}/comparison.md`*
