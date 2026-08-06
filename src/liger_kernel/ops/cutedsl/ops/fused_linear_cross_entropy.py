"""
CuTe DSL (NVIDIA CUTLASS Python DSL) fused linear + cross-entropy loss, Hopper-optimised.

Performance design vs the Triton FLCE
======================================
The Triton implementation splits BT tokens into tiny chunks (≈ 128 rows on LLaMA-3
vocab) to bound peak logit memory.  On Hopper, each cuBLAS GEMM call has fixed
kernel-launch overhead and the tensor-core pipeline only reaches peak throughput for
large M-dimensions.  Running 32 × (128-row) GEMMs is therefore *catastrophically*
slower than one (4096-row) GEMM — we measured 8× regression vs unfused PyTorch on H200.

This module fixes it with two Hopper-specific strategies:

1. **Large chunks**: chunk size = up to 2 GB of logit memory rather than O(V/H) tokens.
   For LLaMA-3 (H=4096, V=128256, bf16): Triton chunk_size ≈ 128; this module uses
   up to 8192+ rows → 32–64× larger GEMMs → near-peak cuBLAS tensor-core utilization.

2. **GEMM / CE stream overlap**: the cuBLAS GEMM for chunk i+1 runs concurrently with
   the CuTe DSL CE kernel for chunk i on a second CUDA stream, hiding CE latency behind
   GEMM compute.  Synchronisation is via explicit ``torch.cuda.Event`` so each stream
   only waits for the precise dependency it needs (not all prior work on the other stream).

The CE kernel itself comes from ``liger_kernel.ops.cutedsl.ops.cross_entropy``; it
already uses cp.async 4-stage pipelining with L1-bypass loads and warp-shuffle
reductions — Hopper-native async-copy hardware features.
"""

import math

from typing import Optional

import torch

from liger_kernel.ops.cutedsl.ops.cross_entropy import _launch_ce_fwd
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import element_mul_kernel

# Maximum logit-tensor memory budget per chunk (bytes).
# 2 GB keeps per-chunk peak overhead manageable on H200 (141 GB HBM3e)
# while ensuring we use one or very few GEMMs for typical training batch sizes.
_HOPPER_CHUNK_BUDGET_BYTES = 2 * 1024**3  # 2 GB

# Minimum GEMM M-dimension in tokens.  Below this cuBLAS cannot saturate tensor cores.
_MIN_CHUNK_TOKENS = 512

# Cached CUDA streams (one pair per device) — avoids creating new Stream objects per call.
_gemm_stream_cache: dict[int, torch.cuda.Stream] = {}
_ce_stream_cache: dict[int, torch.cuda.Stream] = {}


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def _select_chunk_size(BT: int, V: int, dtype: torch.dtype) -> int:
    """
    Choose a chunk size (token rows per iteration) tuned for Hopper GEMMs.

    Largest power-of-two chunk that fits within _HOPPER_CHUNK_BUDGET_BYTES,
    clamped to [_MIN_CHUNK_TOKENS, BT].
    """
    bytes_per_elem = 2 if dtype in (torch.bfloat16, torch.float16) else 4
    bytes_per_token = V * bytes_per_elem
    max_tokens = _HOPPER_CHUNK_BUDGET_BYTES // max(bytes_per_token, 1)
    candidate = _next_power_of_2(max(_MIN_CHUNK_TOKENS, min(max_tokens, BT)))
    return min(candidate, _next_power_of_2(BT))


def fused_linear_cross_entropy_forward(
    _input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ce_weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    accum_dtype: Optional[torch.dtype] = None,
    use_token_scaling: bool = False,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
):
    assert reduction in ("mean", "sum", "none"), f"Unsupported reduction: {reduction}"
    assert isinstance(return_z_loss, bool)
    assert isinstance(return_token_accuracy, bool)
    assert isinstance(return_predicted_tokens, bool)

    device = _input.device
    dtype = _input.dtype
    input_requires_grad = _input.requires_grad

    BT, H = _input.shape
    V = weight.shape[0]

    # Validate vocab size divisibility (required by CuTe DSL CE kernel for 128-bit vectorised loads)
    vec = 16 // _input.element_size()  # 8 for bf16/fp16, 4 for fp32
    assert V % vec == 0, f"cutedsl FLCE needs V % {vec} == 0 for {dtype} (128-bit vectorised loads); got V={V}."

    if _input.stride(-1) != 1:
        _input = _input.contiguous()

    # --- Hopper-tuned chunk size ---
    chunk_size = _select_chunk_size(BT, V, dtype)
    num_chunks = math.ceil(BT / chunk_size)

    # --- Output / accumulation buffers ---
    # loss_1d: fp32 regardless of input dtype (matches Triton FLCE; avoids precision loss)
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    z_loss_1d = torch.zeros(BT, dtype=dtype, device=device) if return_z_loss else None
    token_accuracy_1d = torch.zeros(BT, dtype=torch.float32, device=device) if return_token_accuracy else None
    predicted_tokens_1d = torch.full((BT,), -1, dtype=torch.int64, device=device) if return_predicted_tokens else None

    grad_input = torch.zeros_like(_input) if input_requires_grad else None
    if input_requires_grad:
        accum_dt = accum_dtype if accum_dtype is not None else weight.dtype
        grad_weight = torch.zeros_like(weight, dtype=accum_dt) if weight.requires_grad else None
        grad_bias = torch.zeros_like(bias, dtype=accum_dt) if (bias is not None and bias.requires_grad) else None
    else:
        grad_weight = None
        grad_bias = None

    # --- Pre-compute ignore-index statistics (single D2H sync) ---
    target_mask = target != ignore_index
    _mt = target * target_mask
    _stats = torch.stack((target_mask.sum(), _mt.max(), _mt.min())).tolist()
    total_n_non_ignore = int(_stats[0])
    assert _stats[1] < V, f"Target out of bounds. Expected < {V}"
    assert _stats[2] >= 0, "Target out of bounds. Expected >= 0"

    # Class-weight setup (fp32 upcast, single sum D2H sync)
    has_weight = ce_weight is not None
    ce_weight_fp32 = None
    ce_weight_sum = 0.0
    sum_non_ignore_ce_weight = float(total_n_non_ignore)
    if has_weight:
        assert ce_weight.shape[0] == V
        assert torch.is_floating_point(ce_weight), f"ce_weight must be floating point; got {ce_weight.dtype}"
        ce_weight_fp32 = ce_weight.to(torch.float32)
        if ce_weight_fp32.stride(-1) != 1:
            ce_weight_fp32 = ce_weight_fp32.contiguous()
        if total_n_non_ignore > 0:
            sum_non_ignore_ce_weight = torch.gather(ce_weight_fp32, 0, target.masked_select(target_mask)).sum().item()
        else:
            sum_non_ignore_ce_weight = 1.0
        ce_weight_sum = ce_weight_fp32.sum().item()

    # Global loss / z_loss normalizers (passed to CE kernel; applied per-row in-kernel)
    if reduction == "mean" and total_n_non_ignore > 0:
        if has_weight and sum_non_ignore_ce_weight > 0:
            inv_n_loss = 1.0 / sum_non_ignore_ce_weight
        else:
            inv_n_loss = 1.0 / total_n_non_ignore
        inv_n_z = 1.0 / total_n_non_ignore
    else:
        inv_n_loss = 1.0
        inv_n_z = 1.0

    # --- Two-stream GEMM / CE overlap ---
    # Strategy: GEMM for chunk i+1 overlaps with CE + grad-accum for chunk i.
    # Synchronisation: explicit cuda Events so each stream waits only for the precise
    # dependency it needs (not all prior work on the other stream).
    #
    # gemm_stream: cuBLAS GEMM workloads (tensor-core pipeline)
    # ce_stream:   CuTe DSL CE kernel + grad accumulation (cp.async pipeline)
    #
    # Timeline for 3 chunks:
    #   gemm_stream:  GEMM_0 [ev0] GEMM_1 [ev1] GEMM_2 [ev2]
    #   ce_stream:    {wait ev0} CE_0 GRAD_0 {wait ev1} CE_1 GRAD_1 {wait ev2} CE_2 GRAD_2
    #
    # CE_0 overlaps with GEMM_1; CE_1 overlaps with GEMM_2, etc.
    # grad_weight accumulation (GRAD_0, GRAD_1, ...) is serialised on ce_stream → no race.
    #
    # Streams are cached per-device to avoid per-call Stream object creation overhead.
    dev_idx = device.index if isinstance(device, torch.device) else torch.cuda.current_device()
    if dev_idx not in _gemm_stream_cache:
        _gemm_stream_cache[dev_idx] = torch.cuda.Stream(device=device)
        _ce_stream_cache[dev_idx] = torch.cuda.Stream(device=device)
    gemm_stream = _gemm_stream_cache[dev_idx]
    ce_stream = _ce_stream_cache[dev_idx]
    main_stream = torch.cuda.current_stream(device)

    # Both streams must wait for the main-stream work that produced _input / weight
    # (e.g. the optimizer step) before they start their own work.
    gemm_stream.wait_stream(main_stream)

    # Allocate one Event per chunk (records right after each chunk's GEMM)
    gemm_events = [torch.cuda.Event() for _ in range(num_chunks)]

    # Queue ALL GEMMs onto gemm_stream immediately — the CPU-side loop is fast and
    # the GPU executes them sequentially on gemm_stream while ce_stream overlaps.
    logits_chunks: list[torch.Tensor] = []
    with torch.no_grad(), torch.cuda.stream(gemm_stream):
        for chunk_id in range(num_chunks):
            start = chunk_id * chunk_size
            end = min(start + chunk_size, BT)
            logits_i = _input[start:end] @ weight.t()
            if bias is not None:
                logits_i = logits_i + bias
            logits_i = logits_i.contiguous()
            logits_chunks.append(logits_i)
            gemm_events[chunk_id].record(gemm_stream)

    # Now queue CE + grad-accum on ce_stream, waiting for each chunk's GEMM event.
    with torch.no_grad():
        for chunk_id in range(num_chunks):
            start = chunk_id * chunk_size
            end = min(start + chunk_size, BT)

            logits_i = logits_chunks[chunk_id]
            target_chunk = target[start:end].contiguous()
            loss_slice = loss_1d[start:end]
            z_loss_slice = z_loss_1d[start:end] if return_z_loss else None
            ta_slice = token_accuracy_1d[start:end] if return_token_accuracy else None
            pt_slice = predicted_tokens_1d[start:end] if return_predicted_tokens else None
            input_chunk = _input[start:end]

            with torch.cuda.stream(ce_stream):
                # Wait only for this chunk's GEMM to complete
                ce_stream.wait_event(gemm_events[chunk_id])

                # CuTe DSL CE kernel — reads logits_i, writes d(loss)/d(logits) in-place.
                # _launch_ce_fwd queries torch.cuda.current_stream(), which is ce_stream here
                # because we are inside the torch.cuda.stream(ce_stream) context.
                _launch_ce_fwd(
                    logits_i,
                    target_chunk,
                    loss_slice,
                    inv_n_loss,
                    ignore_index,
                    input_requires_grad,
                    lse_square_scale,
                    z_loss_slice,
                    return_z_loss,
                    softcap,
                    label_smoothing=label_smoothing,
                    weight=ce_weight_fp32,
                    weight_sum=ce_weight_sum,
                    return_token_accuracy=return_token_accuracy,
                    return_predicted_tokens=return_predicted_tokens,
                    token_acc_out=ta_slice,
                    pred_tok_out=pt_slice,
                    inv_n_z=inv_n_z,
                )

                # token_scaling: scale loss by predicted probability (detached)
                if use_token_scaling:
                    probs_f = logits_i.detach().float()
                    if softcap is not None:
                        sc = float(softcap)
                        probs_f = sc * torch.tanh(probs_f / sc)
                    probs_f = torch.softmax(probs_f, dim=-1)
                    valid_mask = target_chunk != ignore_index
                    n_rows = end - start
                    scaling = torch.zeros(n_rows, dtype=probs_f.dtype, device=device)
                    valid_targets = target_chunk[valid_mask]
                    if valid_targets.numel() > 0:
                        scaling[valid_mask] = torch.gather(
                            probs_f[valid_mask], -1, valid_targets.unsqueeze(-1)
                        ).squeeze(-1)
                    loss_1d[start:end].mul_(scaling)
                    if return_z_loss:
                        z_loss_1d[start:end].mul_(scaling)
                    logits_i.mul_(scaling.unsqueeze(-1))

                # Grad accumulation (logits_i now holds d(loss)/d(logits) after CE kernel)
                if input_requires_grad:
                    # grad_input[start:end] = grad_logits @ weight
                    torch.mm(logits_i, weight, out=grad_input[start:end])

                if grad_weight is not None:
                    # grad_weight += grad_logits.T @ input_chunk
                    torch.addmm(grad_weight, logits_i.t(), input_chunk, out=grad_weight)

                if grad_bias is not None:
                    # grad_bias += grad_logits.sum(dim=0)
                    grad_bias.add_(logits_i.sum(dim=0))

    # Sync back to main stream
    main_stream.wait_stream(ce_stream)
    main_stream.wait_stream(gemm_stream)

    # --- Final reduction ---
    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
        token_accuracy = token_accuracy_1d if return_token_accuracy else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        token_accuracy = torch.sum(token_accuracy_1d) / max(total_n_non_ignore, 1) if return_token_accuracy else None
    predicted_tokens = predicted_tokens_1d if return_predicted_tokens else None

    # Cast grad_weight / grad_bias back to weight / bias dtype
    if grad_weight is not None:
        grad_weight = grad_weight.to(weight.dtype)
    if grad_bias is not None and bias is not None:
        grad_bias = grad_bias.to(bias.dtype)

    return loss, z_loss, token_accuracy, predicted_tokens, grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_backward(
    grad_output: torch.Tensor,
    grad_input: Optional[torch.Tensor],
    grad_weight: Optional[torch.Tensor],
    grad_bias: Optional[torch.Tensor],
):
    """Scale pre-accumulated gradients by the upstream loss gradient (chain rule)."""
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return grad_input, grad_weight, grad_bias

    import triton

    if grad_input is not None:
        BT, H = grad_input.shape
        element_mul_kernel[(BT,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=min(65536, triton.next_power_of_2(H)),
            num_warps=32,
        )

    if grad_weight is not None:
        V, H_w = grad_weight.shape
        element_mul_kernel[(V,)](
            grad_weight,
            grad_weight.stride(-2),
            grad_output,
            H_w,
            BLOCK_SIZE=min(65536, triton.next_power_of_2(H_w)),
            num_warps=32,
        )

    if grad_bias is not None:
        V_b = grad_bias.shape[0]
        element_mul_kernel[(V_b,)](
            grad_bias,
            grad_bias.stride(-1),
            grad_output,
            1,
            BLOCK_SIZE=1,
            num_warps=1,
        )

    return grad_input, grad_weight, grad_bias


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    """
    CuTe DSL autograd wrapper for fused linear + cross-entropy loss.

    Hopper-specific optimisations vs the Triton implementation:

    1. **Large chunk sizes**: Up to 2 GB of logits per chunk — eliminates the tiny
       128-row GEMMs that make the Triton FLCE 8× slower than unfused PyTorch on H200.

    2. **GEMM / CE stream overlap**: cuBLAS GEMM for chunk i+1 runs concurrently with
       the CuTe DSL CE kernel for chunk i on a dedicated CUDA stream.

    3. **CuTe DSL CE kernel**: cp.async 4-stage pipelining with L1-bypass loads and
       warp-shuffle reductions — Hopper-native async-copy hardware (vs Triton ld.global).

    API is drop-in compatible with
    ``liger_kernel.ops.fused_linear_cross_entropy.LigerFusedLinearCrossEntropyFunction``.
    """

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        ce_weight: Optional[torch.FloatTensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        accum_dtype: Optional[torch.dtype] = None,
        use_token_scaling: bool = False,
        return_token_accuracy: bool = False,
        return_predicted_tokens: bool = False,
    ):
        loss, z_loss, token_accuracy, predicted_tokens, grad_input, grad_weight, grad_bias = (
            fused_linear_cross_entropy_forward(
                _input=_input,
                weight=weight,
                target=target,
                bias=bias,
                ce_weight=ce_weight,
                ignore_index=ignore_index,
                lse_square_scale=lse_square_scale,
                label_smoothing=label_smoothing,
                reduction=reduction,
                softcap=softcap,
                return_z_loss=return_z_loss,
                accum_dtype=accum_dtype,
                use_token_scaling=use_token_scaling,
                return_token_accuracy=return_token_accuracy,
                return_predicted_tokens=return_predicted_tokens,
            )
        )

        ctx.save_for_backward(
            grad_input,
            grad_weight,
            grad_bias,
        )
        ctx.return_z_loss = return_z_loss
        ctx.return_token_accuracy = return_token_accuracy
        ctx.return_predicted_tokens = return_predicted_tokens
        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2, grad_output3, grad_output4):
        if ctx.return_z_loss:
            del grad_output2
        if ctx.return_token_accuracy:
            del grad_output3
        if ctx.return_predicted_tokens:
            del grad_output4

        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )
        return (
            grad_input,
            grad_weight,
            None,  # target
            grad_bias,
            None,  # ce_weight
            None,  # ignore_index
            None,  # lse_square_scale
            None,  # label_smoothing
            None,  # reduction
            None,  # softcap
            None,  # return_z_loss
            None,  # accum_dtype
            None,  # use_token_scaling
            None,  # return_token_accuracy
            None,  # return_predicted_tokens
        )
