import torch

from packaging.version import Version

from liger_kernel.ops.cutedsl.ops.cross_entropy import _launch_ce_fwd
from liger_kernel.ops.cutedsl.ops.utils import _next_power_of_2
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd

# grad_weight accumulation uses torch.addmm(..., out_dtype=fp32, out=grad_weight) to accumulate
# bf16/fp16 operands into an fp32 buffer without a [V, H] fp32 temp — identical to the upstream
# Triton FLCE (PR #1239). out_dtype was added to torch.addmm in torch 2.8.0; earlier versions fall
# back to mm().float(). Kept byte-for-byte in sync with the Triton path so the ONLY difference
# between the two FLCE backends is the CE kernel.
_TORCH_VERSION = Version(torch.__version__.split("+")[0])
_ADDMM_SUPPORTS_OUT_DTYPE = _TORCH_VERSION >= Version("2.8.0")

_UNSUPPORTED = "cutedsl FLCE: {feat} is not supported."


def _cdiv(a, b):
    return (a + b - 1) // b


# =============================================================================
# Forward
# =============================================================================
def fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    ce_weight=None,
    bias=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
    return_z_loss=False,
    accum_dtype=None,
    use_token_scaling=False,
    return_token_accuracy=False,
    return_predicted_tokens=False,
):
    """CuTe DSL FLCE forward.

    Returns (loss, z_loss, token_accuracy, predicted_tokens, grad_input,
    grad_weight, grad_bias). Matches
    ``liger_kernel.ops.fused_linear_cross_entropy.fused_linear_cross_entropy_forward``.
    """
    # Every CE feature (ce_weight / softcap / label_smoothing / z_loss / token_accuracy /
    # predicted_tokens) is plumbed straight through to the CuTe DSL CE kernel below, which
    # already implements each one (validated by the standalone CE suite). token_scaling is
    # an FLCE-level transform (pure torch, around the kernel). The ONE feature the fused
    # design genuinely cannot support is reduction='none' WITH grad (guarded just below).
    assert reduction in ("mean", "sum", "none"), f"Unsupported reduction: {reduction}"

    device = _input.device
    input_requires_grad = _input.requires_grad
    if reduction == "none" and input_requires_grad:
        # The fused design accumulates grad_weight/grad_bias over tokens during the
        # forward pass, so backward can't re-weight them by a per-token upstream grad.
        # Refuse loudly rather than crash (the (BT,) grad_output mis-broadcasts against
        # (BT, H)) or silently return a wrong grad_weight (the Triton path scales every
        # grad by the scalar grad_output[0], itself incorrect for per-token 'none').
        raise NotImplementedError(_UNSUPPORTED.format(feat="reduction='none' with grad"))

    # inputs: (BT, H); per-chunk materialized logits: (chunk_size, V).
    BT, H = _input.shape
    V = weight.shape[0]
    # Mirror the CE kernel's divisibility contract (cross_entropy_forward): 128-bit
    # vectorized loads need V % vec == 0, vec = 16 // element_size (8 bf16 / 4 fp32). The
    # CE kernel predicates its 256-thread tail, so no stronger multiple is required. Fail
    # fast here with a dtype-aware message instead of letting the inner CE assert fire
    # mid-loop. (vec from _input.dtype == the logits dtype on the common path.)
    vec = 16 // _input.element_size()
    assert V % vec == 0, (
        f"cutedsl FLCE requires V % {vec} == 0 for {_input.dtype} logits "
        f"(the CE kernel's 128-bit vectorized loads); got V={V}."
    )

    # ---- chunk sizing (memory-minimal, identical to the upstream Triton FLCE) ----
    # Partition the BT tokens so the transient (chunk_size, V) logit tile matches the
    # (BT, H) input footprint: inc_factor = ceil(V/H), chunk_size = next_pow2(ceil(BT/inc_factor)).
    # This is the conservative upstream rule — no free-memory-dependent chunk growth — so the
    # chunk count (hence grad accumulation order) is deterministic and the peak transient is
    # bounded by construction. Keeps the CuTe DSL FLCE apples-to-apples with the Triton path.
    inc_factor = _cdiv(V, H)
    chunk_size = _next_power_of_2(_cdiv(BT, inc_factor))
    num_chunks = _cdiv(BT, chunk_size)

    grad_input = torch.empty_like(_input, device=device)  # fully overwritten per-chunk below

    # fp32 (or accum_dtype) accumulators for the weight / bias gradients.
    if input_requires_grad:
        gw_dtype = accum_dtype if accum_dtype is not None else weight.dtype
        grad_weight = torch.zeros_like(weight, dtype=gw_dtype, device=device) if weight.requires_grad else None
        if bias is not None:
            gb_dtype = accum_dtype if accum_dtype is not None else bias.dtype
            grad_bias = torch.zeros_like(bias, dtype=gb_dtype, device=device)
        else:
            grad_bias = None
    else:
        grad_weight = None
        grad_bias = None

    # fp32 loss accumulator, matching the Triton path exactly. Safe now that the CE kernel's
    # compile cache keys on loss.dtype: FLCE's fp32 loss and the standalone CE's input-dtype
    # loss compile to separate kernels instead of colliding.
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    # Aux per-row buffers (one slice handed to each chunk's CE launch, written in place).
    #   z_loss: fp32 (NOT _input.dtype like Triton) -> z matches the fp32 loss buffer, so the
    #     CE kernel never hits an untested loss.dtype != z.dtype compile combo, and the summed
    #     z_loss is more accurate. Within the bf16 z_loss tolerance vs the Triton oracle.
    #   token_accuracy: fp32 per-row 1.0/0.0; predicted_tokens: int64 argmax, -1 for ignored.
    z_loss_1d = torch.zeros(BT, dtype=torch.float32, device=device) if return_z_loss else None
    token_accuracy_1d = torch.zeros(BT, dtype=torch.float32, device=device) if return_token_accuracy else None
    predicted_tokens_1d = torch.full((BT,), -1, dtype=torch.int64, device=device) if return_predicted_tokens else None

    # Global non-ignored token count -> ONE normalizer applied to EVERY chunk, so each chunk's
    # loss/grad come out already mean-normalized (matches Triton, which passes the totals to
    # every per-chunk CE launch).
    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()
    assert (target * target_mask).max() < V, f"Target out of bounds. Expected < {V}"
    assert (target * target_mask).min() >= 0, "Target out of bounds. Expected >= 0"

    # Class weight: the mean denominator becomes the summed weight of non-ignored targets
    # (sum_non_ignore_weight) instead of the count; weight_sum (full-vector sum) feeds the
    # weighted label-smoothing term. The kernel reads weight as fp32 -> upcast here (exact
    # for fp32 weights). Mirrors the standalone CE forward.
    total_sum_non_ignore_ce_weight = float(total_n_non_ignore)
    ce_weight_sum = 0.0
    if ce_weight is not None:
        assert ce_weight.shape[0] == V, f"ce_weight must be a Tensor of size V={V}. Got: {tuple(ce_weight.shape)}"
        assert torch.is_floating_point(ce_weight), f"ce_weight must be floating point. Got: {ce_weight.dtype}"
        ce_weight = ce_weight.to(torch.float32)
        if ce_weight.stride(-1) != 1:
            ce_weight = ce_weight.contiguous()
        total_sum_non_ignore_ce_weight = torch.gather(ce_weight, 0, target.masked_select(target_mask)).sum().item()
        ce_weight_sum = ce_weight.sum().item()

    # mean -> 1/N per-row in-kernel; sum/none -> 1.0 (unnormalized); 1.0 when all-ignored
    # (avoids /0). The main loss/grad normalize by sum_non_ignore_weight when weighted; z_loss
    # is never weight-scaled, so it always uses the plain non-ignored count.
    if reduction == "mean" and total_n_non_ignore > 0:
        if ce_weight is not None and total_sum_non_ignore_ce_weight > 0:
            inv_n_loss = 1.0 / total_sum_non_ignore_ce_weight
        else:
            inv_n_loss = 1.0 / total_n_non_ignore
        inv_n_z = 1.0 / total_n_non_ignore
    else:
        inv_n_loss = 1.0
        inv_n_z = 1.0

    if target.stride(-1) != 1:
        target = target.contiguous()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]  # (chunk, H)

        # logits in the original precision (cuBLAS), exactly like the Triton path.
        logits_chunk = _input_chunk @ weight.t()  # (chunk, V)
        if bias is not None:
            # in-place add avoids a second (chunk, V) temp; fall back to out-of-place
            # when dtypes differ (autocast: bf16 matmul + fp32 bias).
            if logits_chunk.dtype == bias.dtype:
                logits_chunk += bias
            else:
                logits_chunk = logits_chunk + bias

        target_chunk = target[start_idx:end_idx]  # (chunk,)

        # Token scaling: detached softmax prob of the target token, computed on the
        # (softcapped) logits BEFORE the CE kernel overwrites logits_chunk with the gradient.
        # Ignored rows get a 0 factor. Pure FLCE-level transform (the kernel is unaware).
        if use_token_scaling:
            logits_for_softmax = logits_chunk.detach().clone()
            if softcap is not None:
                logits_for_softmax = softcap * torch.tanh(logits_for_softmax / softcap)
            probs = torch.softmax(logits_for_softmax, dim=-1)
            valid_mask = target_chunk != ignore_index
            pred_probs = torch.zeros_like(target_chunk, dtype=probs.dtype)
            if valid_mask.any():
                valid_targets = target_chunk[valid_mask]
                pred_probs[valid_mask] = torch.gather(probs[valid_mask], -1, valid_targets.unsqueeze(-1)).squeeze(-1)
            scaling_factors = pred_probs.detach()  # (chunk,)

        loss_1d_slice = loss_1d[start_idx:end_idx]  # (chunk,), fp32
        z_loss_1d_slice = z_loss_1d[start_idx:end_idx] if return_z_loss else None
        token_accuracy_1d_slice = token_accuracy_1d[start_idx:end_idx] if return_token_accuracy else None
        predicted_tokens_1d_slice = predicted_tokens_1d[start_idx:end_idx] if return_predicted_tokens else None

        # CE kernel needs the row contiguous (it slices mX[row, None]); target 1D contiguous.
        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        # CuTe DSL CE kernel: per-row loss (+ z_loss / token_accuracy / predicted_tokens) and
        # the in-place gradient over logits_chunk. has_grad gates the gradient pass (mirrors
        # Triton's HAS_GRADIENTS=input_requires_grad). Every advanced feature is a pass-through:
        # the kernel bakes the flags into its compile key and implements each term itself.
        _launch_ce_fwd(
            logits_chunk,
            target_chunk,
            loss_1d_slice,
            inv_n_loss,
            ignore_index,
            input_requires_grad,
            lse_square_scale,
            z_loss_1d_slice,
            return_z_loss,
            softcap,
            label_smoothing=label_smoothing,
            weight=ce_weight,
            weight_sum=ce_weight_sum,
            return_token_accuracy=return_token_accuracy,
            return_predicted_tokens=return_predicted_tokens,
            token_acc_out=token_accuracy_1d_slice,
            pred_tok_out=predicted_tokens_1d_slice,
            inv_n_z=inv_n_z,
        )

        # Apply token scaling to the per-row loss / z_loss (out-of-place -> write back).
        if use_token_scaling:
            loss_1d[start_idx:end_idx] = loss_1d_slice * scaling_factors
            if return_z_loss:
                z_loss_1d[start_idx:end_idx] = z_loss_1d_slice * scaling_factors

        grad_logits_chunk = logits_chunk  # (chunk, V): the in-place CE gradient
        # ... and to the gradient, so grad_input/grad_weight reflect the scaled loss.
        if use_token_scaling:
            grad_logits_chunk = grad_logits_chunk * scaling_factors.unsqueeze(-1)

        if input_requires_grad:
            grad_input[start_idx:end_idx] = grad_logits_chunk @ weight

        if grad_weight is not None:
            # Mirror the upstream Triton FLCE grad_weight accumulation EXACTLY (PR #1239): use
            # torch.addmm(out_dtype=fp32) to accumulate bf16/fp16 operands into an fp32 grad_weight
            # without materializing a [V, H] fp32 temp; otherwise the original mm().float(). Keeping
            # this identical to the Triton path means the ONLY difference between the two FLCE
            # backends is the CE kernel.
            grad_logits_t = grad_logits_chunk.t()
            if (
                _ADDMM_SUPPORTS_OUT_DTYPE
                and grad_weight.device.type == "cuda"
                and grad_weight.dtype == torch.float32
                and grad_logits_t.dtype in (torch.float16, torch.bfloat16)
            ):
                # Unlike torch.mm, torch.addmm's out_dtype path does not participate in
                # autocast operand casting, so under AMP (fp32 params, no bias) _input_chunk
                # can stay fp32 while grad_logits is the autocast dtype. addmm requires mat1
                # and mat2 to share a dtype, so align _input_chunk before accumulating.
                input_chunk = _input_chunk
                if input_chunk.dtype != grad_logits_t.dtype:
                    input_chunk = input_chunk.to(grad_logits_t.dtype)
                torch.addmm(
                    grad_weight,
                    grad_logits_t,
                    input_chunk,
                    out_dtype=torch.float32,
                    out=grad_weight,
                )
            else:
                grad_weight += torch.mm(grad_logits_chunk.t(), _input_chunk).float()

        if grad_bias is not None:
            torch.add(
                input=grad_bias,
                other=grad_logits_chunk.sum(dim=0),
                out=grad_bias,
                alpha=1.0,
            )

    # Reduce the per-row buffers. reduction='none' returns the per-token loss vector; mean/sum
    # sum it (the per-row 1/N normalizer already applied the mean). token_accuracy always
    # reduces to the mean over non-ignored tokens for mean/sum (matches Triton + standalone CE);
    # predicted_tokens is always the per-row vector. (none+grad is refused above, so the 'none'
    # branch here is forward-only.)
    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
        token_accuracy = token_accuracy_1d if return_token_accuracy else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        token_accuracy = torch.sum(token_accuracy_1d) / total_n_non_ignore if return_token_accuracy else None
    predicted_tokens = predicted_tokens_1d if return_predicted_tokens else None

    # Cast accumulators back to the parameter dtype.
    grad_weight = grad_weight.to(weight.dtype) if grad_weight is not None else None
    grad_bias = grad_bias.to(bias.dtype) if grad_bias is not None else None

    return loss, z_loss, token_accuracy, predicted_tokens, grad_input, grad_weight, grad_bias


# =============================================================================
# Backward
# =============================================================================
def fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight, grad_bias):
    """Scale the saved grads by ``grad_output`` (chain rule from upstream)."""
    # FLCE is usually the last layer -> grad_output == 1.0; skip the scaling.
    if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        # reduction='none'+grad is refused in forward, so grad_output is a scalar here.
        # Cast each product back to its tensor's dtype: the summed loss may be a
        # higher-precision scalar, so multiplying would otherwise promote e.g. bf16 grads
        # to fp32 and autograd would reject the dtype mismatch against the bf16 inputs.
        # Fresh tensors (not in-place) also avoid the autograd anomalies the Triton path
        # sidesteps with its custom element_mul kernel (which preserves dtype in place).
        grad_input = (grad_input * grad_output).to(grad_input.dtype)
        if grad_weight is not None:
            grad_weight = (grad_weight * grad_output).to(grad_weight.dtype)
        if grad_bias is not None:
            grad_bias = (grad_bias * grad_output).to(grad_bias.dtype)
    return grad_input, grad_weight, grad_bias


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    """
    CuTe DSL autograd wrapper for Fused-Linear-Cross-Entropy.

    Signature-compatible with
    ``liger_kernel.ops.fused_linear_cross_entropy.LigerFusedLinearCrossEntropyFunction``.
    """

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ce_weight=None,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        softcap=None,
        return_z_loss: bool = False,
        accum_dtype=None,
        use_token_scaling: bool = False,
        return_token_accuracy: bool = False,
        return_predicted_tokens: bool = False,
    ):
        # Memory-minimal chunking bounds the transient (chunk_size, V) logit tile to the
        # (BT, H) input footprint by construction, so no OOM-retry / chunk-growth fallback is
        # needed — call the forward directly (matches the upstream Triton FLCE control flow).
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
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if grad_bias is not None else None,
        )
        ctx.return_z_loss = return_z_loss
        ctx.return_token_accuracy = return_token_accuracy
        ctx.return_predicted_tokens = return_predicted_tokens
        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2, grad_output3, grad_output4):
        if ctx.return_z_loss:
            del grad_output2  # z_loss is only for logging
        if ctx.return_token_accuracy:
            del grad_output3  # token_accuracy is only for metrics
        if ctx.return_predicted_tokens:
            del grad_output4  # predicted_tokens is only for metrics
        (grad_input, grad_weight, grad_bias) = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )
        return (
            grad_input,
            grad_weight,
            None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # use_token_scaling
            None,  # return_token_accuracy
            None,  # return_predicted_tokens
        )
