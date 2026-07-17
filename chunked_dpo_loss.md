# Chunked DPO Loss: (batch × sequence)-level chunking with in-place gradient accumulation

This document explains the design and rationale behind `LigerChunkedLinearPreferenceBase` /
`LigerChunkedLinearDPOLoss` and the `ChunkMatmul` autograd function
(commits `b442602` and `cf8d269`).

## Problem

The upstream `LigerFusedLinearPreferenceBase` chunks the DPO loss along the **batch**
dimension over chosen/rejected pairs. Each chunk therefore still contains the full
sequence length, and its logits are of shape `(chunk_B, T, V)`.

In the typical DPO regime — long sequences, small effective batch — a "chunk" is
essentially the whole batch, so batch-level chunking degenerates into gradient
accumulation and does **not** reduce the peak memory footprint at the end of the
forward path (the logit / log-softmax part).

The FLCE trick (chunk over `B·T` and backprop each chunk immediately) does not transfer
naively to DPO: cross-entropy is a linear sum over tokens, but the DPO loss is
**nonlinear in the sequence-level log-probabilities**, so the loss — and therefore
backward — cannot start until every chunk's contribution to the sequence logps exists.

## Design

### Forward: chunk at (batch × sequence) granularity

`LigerChunkedLinearPreferenceBase.forward` flattens chosen and rejected inputs to
`(B/2 · T, H)`, splits them into fixed-size token chunks, and for each chunk computes:

- `logits_chunk = ChunkMatmul.apply(input_chunk, weight)` (lm-head projection)
- `log_probs_chunk = F.log_softmax(logits_chunk, dim=-1, dtype=torch.float32)`
- gathered per-token logps (and optionally the NLL loss term)

Per-token logps are collected across chunks, reassembled with `cat → view → sum(-1)`
into per-sequence logps, and the preference loss is computed once on that tiny
sequence-level graph.

Memory consequence:

- **Retained until backward:** each chunk's logits (this floor is unavoidable — DPO
  needs all sequence logps before backward can begin).
- **Transient at any moment:** only *one chunk's* fp32 log-softmax working set
  (`O(chunk · V)`), instead of the full-batch fp32 pipeline (`O(B · T · V)`).

The `dtype=torch.float32` argument (instead of `logits_chunk.float()`) additionally
avoids materializing a separate full fp32 copy of the chunk logits as a standalone
graph tensor.

### Backward: `ChunkMatmul` — streamed, in-place weight-gradient accumulation

Backward walks the chunks in reverse. With a plain autograd matmul, **each chunk's
backward would allocate a fresh `(V × H)` weight gradient** before `AccumulateGrad`
sums it into `weight.grad`. At e.g. V=128K, H=4096, fp32 that is ~2 GB of transient
allocation *per chunk*.

`ChunkMatmul` takes manual ownership of this gradient:

```python
class ChunkMatmul(torch.autograd.Function):
    buf: dict[str, torch.Tensor | int] = {"count": 0}

    @staticmethod
    def forward(ctx, x, weight):
        if weight.requires_grad:
            ctx.save_for_backward(x, weight)
            ChunkMatmul.buf["count"] += 1      # one backward call is owed
        return x.matmul(weight.T)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        grad_x = grad_out.matmul(weight)        # stateless, per chunk, as usual
        if "grad" not in ChunkMatmul.buf:
            ChunkMatmul.buf["grad"] = torch.zeros_like(weight)
        ChunkMatmul.buf["grad"].addmm_(          # fused multiply-accumulate,
            grad_out.view(-1, grad_out.shape[-1]).T,   # no intermediate (V, H) tensor
            x.view(-1, x.shape[-1]),
        )
        ChunkMatmul.buf["count"] -= 1
        if ChunkMatmul.buf["count"] == 0:       # last chunk's backward:
            grad_w = ChunkMatmul.buf["grad"]    # hand the completed sum to autograd
            del ChunkMatmul.buf["grad"]         # reset for the next step
        else:
            grad_w = None                       # earlier chunks contribute nothing
        return grad_x, grad_w
```

Key points:

- `grad_x` is pure and identical to vanilla autograd — failure modes of the buffer
  logic are confined to the weight gradient.
- All chunks accumulate into **one** buffer via `addmm_`; only the *last* backward
  (detected by the reference count reaching zero) returns the sum, so autograd
  performs a single accumulation into `weight.grad` and nothing is double-counted.
- At the end of a normally completed step, `count == 0` and the buffer is deleted:
  the state at the start of step N+1 is identical to step 1, so a one-step gradient
  equivalence check extends to the whole steady-state training loop by induction.
- `save_for_backward(x, weight)` costs no extra memory: `x` is a live activation and
  `weight` is a parameter.

### Assumptions / limitations

`ChunkMatmul` is a correct *protocol for this training loop*, not a general operator:

- One trainable weight per accumulation cycle (the buffer is keyed by nothing).
  The reference model's weight must be frozen (`requires_grad=False`) so the ref
  path never touches the counter.
- Every forward with grad enabled must be matched by exactly one backward before the
  next cycle. A forward whose backward never runs (validation with grad enabled, an
  exception between forward and backward) leaves `count > 0` and silently corrupts
  subsequent weight gradients. A cheap guard:

  ```python
  assert ChunkMatmul.buf["count"] == 0  # at the top of forward
  ```
- Single-threaded, single-stream execution; no `retain_graph` / double backward.

## Result

The fp32 log-softmax pipeline is executed strictly chunk-by-chunk in both forward
and backward: no full-batch `(B·T, V)` fp32 tensor is ever allocated, and lm-head
gradients are streamed into one in-place accumulator. Per-chunk fp32 outputs are
not retained for backward: under `torch.compile` (the default, `compiled=True`),
the partitioner saves only the bf16 logits per chunk and rematerializes the fp32
log-softmax on demand during backward, so total fp32 residency is bounded to a
single chunk at all times. The memory that scales with
`B · T · V` drops from ~10 bytes per logit element in the baseline (bf16 logits +
fp32 upcast + retained fp32 log-probs, plus their fp32 gradient as a backward
transient) to ~2 bytes (retained bf16 logits only); all remaining fp32 terms are
`O(chunk · V)` and independent of batch and sequence length. The reduction comes
from two sources:

1. the full-batch fp32 log-softmax pipeline is replaced by a per-chunk transient, and
2. per-chunk `(V × H)` weight-gradient allocations in backward are replaced by a
   single in-place accumulator.

The remaining floor is the retained per-chunk logits in forward (visible as a
"staircase" in a memory snapshot); removing it would require a two-pass /
recomputation design, since DPO forces all sequence logps to exist before backward.
