import operator

import cutlass
import cutlass.cute as cute
import torch

from liger_kernel.ops.cutedsl.ops._sm100_gemm import K_ALIGNMENT
from liger_kernel.ops.cutedsl.ops._sm100_gemm import run_epilogue_gemm
from liger_kernel.ops.cutedsl.ops.cross_entropy import _launch_ce_fwd
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import compare_version
from liger_kernel.utils import infer_device_arch

_SUPPORTS_OUT_DTYPE = compare_version("torch", operator.ge, "2.8.0")
_FORWARD_GRAD_MIN_CHUNK = 512
_FORWARD_GRAD_MAX_CHUNK = 1024


@cute.jit
def _identity_epilogue(accumulator, output):
    output_dtype = output.element_type
    for element in cutlass.range_constexpr(cute.size(accumulator)):
        output[element] = accumulator[element].to(output_dtype)


def _forward_grad_chunk_size(total_tokens):
    target = max(_FORWARD_GRAD_MIN_CHUNK, total_tokens // 4)
    power_of_two_target = 1 << (target.bit_length() - 1)
    return min(total_tokens, _FORWARD_GRAD_MAX_CHUNK, power_of_two_target)


def _mm_out(out, lhs, rhs):
    if out.dtype == lhs.dtype:
        torch.mm(lhs, rhs, out=out)
    elif _SUPPORTS_OUT_DTYPE:
        torch.mm(lhs, rhs, out_dtype=out.dtype, out=out)
    else:
        torch.mm(lhs.float(), rhs.float(), out=out)


def _accum_grad_weight(grad_weight, dlogits_t, x_chunk):
    if grad_weight.dtype == torch.float32 and dlogits_t.dtype in (torch.float16, torch.bfloat16):
        if _SUPPORTS_OUT_DTYPE:
            torch.addmm(grad_weight, dlogits_t, x_chunk, out_dtype=torch.float32, out=grad_weight)
        else:
            grad_weight.addmm_(dlogits_t.float(), x_chunk.float())
    else:
        torch.addmm(grad_weight, dlogits_t, x_chunk, out=grad_weight)


def _target_probability_from_logits(logits, target, ignore_index, softcap, vocab_chunk=8192):
    """Compute detached target probabilities from existing logits with bounded fp32 workspace."""
    row_lse = torch.full((logits.shape[0],), -torch.inf, device=logits.device, dtype=torch.float32)
    for start in range(0, logits.shape[1], vocab_chunk):
        block = logits[:, start : start + vocab_chunk].float()
        if softcap not in (None, 0.0):
            block = softcap * torch.tanh(block / softcap)
        row_lse = torch.logaddexp(row_lse, torch.logsumexp(block, dim=-1))
    valid = target != ignore_index
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    target_logits = logits.gather(1, safe_target[:, None]).squeeze(1).float()
    if softcap not in (None, 0.0):
        target_logits = softcap * torch.tanh(target_logits / softcap)
    return torch.where(valid, torch.exp(target_logits - row_lse), torch.zeros_like(row_lse))


def _native_forward(
    X,
    W,
    target,
    bias,
    ce_weight,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    reduction,
    softcap,
    return_z_loss,
    accum_dtype,
    use_token_scaling,
    return_token_accuracy,
    return_predicted_tokens,
    needs_grad,
):
    """Compute every native output and gradient from one chunked logits pass."""
    BT, H = X.shape
    V = W.shape[0]
    chunk = _forward_grad_chunk_size(BT)
    vector_width = 16 // X.element_size()
    storage_width = ((V + vector_width - 1) // vector_width) * vector_width
    logits_storage = torch.empty(chunk, storage_width, device=X.device, dtype=X.dtype)
    ce_weight_kernel = torch.nn.functional.pad(ce_weight, (0, storage_width - V)) if ce_weight is not None else None
    loss_1d = torch.zeros(BT, device=X.device, dtype=torch.float32)
    z_loss_1d = torch.zeros(BT, device=X.device, dtype=X.dtype) if return_z_loss else None
    token_acc_1d = torch.zeros(BT, device=X.device, dtype=torch.float32) if return_token_accuracy else None
    predicted_tokens = torch.full((BT,), -1, device=X.device, dtype=torch.int64) if return_predicted_tokens else None

    valid = target != ignore_index
    n_non_ignore = int(valid.sum().item())
    n = max(n_non_ignore, 1)
    inv_n_z = 1.0 / n if reduction == "mean" else 1.0
    weight_sum = 0.0
    if ce_weight is not None:
        denominator = ce_weight[target[valid]].sum() if n_non_ignore else ce_weight.new_ones(())
        inv_n_loss = torch.reciprocal(denominator).item() if reduction == "mean" else 1.0
        weight_sum = ce_weight.sum().item()
    else:
        inv_n_loss = inv_n_z

    grad_input = torch.empty(BT, H, device=X.device, dtype=X.dtype) if needs_grad else None
    grad_weight_dtype = accum_dtype if accum_dtype is not None else W.dtype
    grad_weight = torch.empty(V, H, device=W.device, dtype=grad_weight_dtype) if needs_grad else None
    grad_bias_acc = torch.zeros(V, device=X.device, dtype=torch.float32) if needs_grad and bias is not None else None
    effective_softcap = softcap if softcap not in (None, 0.0) else None

    for start in range(0, BT, chunk):
        end = min(start + chunk, BT)
        rows = end - start
        Xc = X[start:end]
        ce_logits = logits_storage[:rows]
        dlogits = ce_logits[:, :V]
        run_epilogue_gemm(Xc, W, dlogits, _identity_epilogue)
        if bias is not None:
            dlogits.add_(bias)
        if storage_width != V:
            ce_logits[:, V:].zero_()
        token_scale = (
            _target_probability_from_logits(
                dlogits,
                target[start:end],
                ignore_index,
                effective_softcap,
            )
            if use_token_scaling
            else None
        )
        _launch_ce_fwd(
            ce_logits,
            target[start:end],
            loss_1d[start:end],
            inv_n_loss,
            ignore_index,
            needs_grad,
            lse_square_scale,
            z_loss_1d[start:end] if return_z_loss else None,
            return_z_loss,
            effective_softcap,
            label_smoothing=label_smoothing,
            weight=ce_weight_kernel,
            weight_sum=weight_sum,
            return_token_accuracy=return_token_accuracy,
            return_predicted_tokens=return_predicted_tokens,
            token_acc_out=token_acc_1d[start:end] if return_token_accuracy else None,
            pred_tok_out=predicted_tokens[start:end] if return_predicted_tokens else None,
            inv_n_z=inv_n_z,
            logical_vocab_size=V,
        )
        if token_scale is not None:
            loss_1d[start:end].mul_(token_scale)
            if return_z_loss:
                z_loss_1d[start:end].mul_(token_scale)
            if needs_grad:
                dlogits.mul_(token_scale[:, None])
        if needs_grad:
            _mm_out(grad_input[start:end], dlogits, W)
            if start == 0:
                _mm_out(grad_weight, dlogits.t(), Xc)
            else:
                _accum_grad_weight(grad_weight, dlogits.t(), Xc)
            if grad_bias_acc is not None:
                grad_bias_acc.add_(dlogits.sum(0, dtype=torch.float32))

    loss = loss_1d.sum()
    z_loss = z_loss_1d.sum() if return_z_loss else None
    token_accuracy = token_acc_1d.sum() / n_non_ignore if return_token_accuracy else None
    if grad_weight is not None:
        grad_weight = grad_weight.to(W.dtype)
    grad_bias = grad_bias_acc.to(bias.dtype) if grad_bias_acc is not None else None
    return loss, z_loss, token_accuracy, predicted_tokens, grad_input, grad_weight, grad_bias


def _validate_inputs(_input, weight, target, bias, ce_weight, reduction, ignore_index):
    if _input.ndim != 2 or weight.ndim != 2:
        raise ValueError(f"_input and weight must be 2D, got {_input.shape} and {weight.shape}.")
    if target.ndim != 1 or target.shape[0] != _input.shape[0]:
        raise ValueError(f"target must have shape ({_input.shape[0]},), got {target.shape}.")
    if _input.shape[1] != weight.shape[1]:
        raise ValueError(f"Input and weight hidden dimensions must match, got {_input.shape[1]} and {weight.shape[1]}.")
    if _input.shape[0] == 0 or _input.shape[1] == 0 or weight.shape[0] == 0:
        raise ValueError("FLCE requires non-empty token, hidden, and vocabulary dimensions.")
    if _input.device != weight.device or _input.device != target.device:
        raise ValueError(
            f"_input, weight, and target must share a device, got {_input.device}, {weight.device}, and {target.device}."
        )
    if _input.dtype != weight.dtype:
        raise TypeError(f"_input and weight must share a dtype, got {_input.dtype} and {weight.dtype}.")
    if not torch.is_floating_point(_input):
        raise TypeError(f"_input and weight must have a floating-point dtype, got {_input.dtype}.")
    if target.dtype != torch.long:
        raise TypeError(f"target must have dtype torch.long, got {target.dtype}.")
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise ValueError(f"bias must have shape ({weight.shape[0]},), got {bias.shape}.")
        if bias.device != _input.device or not torch.is_floating_point(bias):
            raise TypeError("bias must be a floating-point tensor on the same device as _input.")
    if ce_weight is not None:
        if ce_weight.ndim != 1 or ce_weight.shape[0] != weight.shape[0]:
            raise ValueError(f"ce_weight must have shape ({weight.shape[0]},), got {ce_weight.shape}.")
        if ce_weight.device != _input.device or not torch.is_floating_point(ce_weight):
            raise TypeError("ce_weight must be a floating-point tensor on the same device as _input.")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be one of 'mean', 'sum', or 'none', got {reduction!r}.")
    valid_targets = target[target != ignore_index]
    if valid_targets.numel() > 0:
        if valid_targets.max() >= weight.shape[0]:
            raise AssertionError(f"Target out of bounds. Expected < {weight.shape[0]}")
        if valid_targets.min() < 0:
            raise AssertionError("Target out of bounds. Expected >= 0")


def _native_sm100_supported(tensor):
    if tensor.device.type != "cuda":
        return False
    device_id = tensor.device.index if tensor.device.index is not None else torch.cuda.current_device()
    return infer_device_arch(device_id) == "blackwell" and torch.cuda.get_device_capability(tensor.device) == (10, 0)


class _FlceState:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


def fused_linear_cross_entropy_forward(
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
    return_z_loss=False,
    accum_dtype=None,
    use_token_scaling=False,
    return_token_accuracy=False,
    return_predicted_tokens=False,
):
    """Run the native SM100 CuTe DSL path."""
    ctx = _FlceState()
    _validate_inputs(_input, weight, target, bias, ce_weight, reduction, ignore_index)
    needs_grad = _input.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)

    if (
        not _native_sm100_supported(_input)
        or _input.dtype not in (torch.bfloat16, torch.float16)
        or reduction not in ("mean", "sum")
    ):
        raise RuntimeError(
            "Native CuTe DSL FLCE requires exact SM100 hardware, FP16/BF16 input and weight, and mean/sum reduction."
        )

    h_orig = None
    if weight.shape[1] % K_ALIGNMENT != 0:
        pad = (-weight.shape[1]) % K_ALIGNMENT
        _input = torch.nn.functional.pad(_input, (0, pad))
        weight = torch.nn.functional.pad(weight, (0, pad))
        h_orig = weight.shape[1] - pad
    ctx.h_orig = h_orig

    X = _input.detach().contiguous()
    W = weight.detach().contiguous()
    native_bias = bias.detach().contiguous() if bias is not None else None
    native_ce_weight = ce_weight.detach().float().contiguous() if ce_weight is not None else None
    native_kwargs = {
        "ignore_index": ignore_index,
        "lse_square_scale": lse_square_scale,
        "label_smoothing": label_smoothing,
        "reduction": reduction,
        "softcap": softcap,
        "return_z_loss": return_z_loss,
        "accum_dtype": accum_dtype,
        "use_token_scaling": use_token_scaling,
        "return_token_accuracy": return_token_accuracy,
        "return_predicted_tokens": return_predicted_tokens,
    }
    loss, z_out, acc_out, pred_out, gX, gW, gb = _native_forward(
        X,
        W,
        target,
        native_bias,
        native_ce_weight,
        needs_grad=needs_grad,
        **native_kwargs,
    )
    if needs_grad:
        ctx._native = True
        ctx._native_gradients_consumed = False
        ctx._repeated_grad_input = None
        ctx._repeated_grad_weight = None
        ctx._repeated_grad_bias = None
        ctx.grad_input = gX
        ctx.grad_weight = gW
        ctx.grad_bias = gb
        ctx.save_for_backward(X, W, target)
        ctx.native_bias = native_bias
        ctx.native_ce_weight = native_ce_weight
        ctx.native_kwargs = native_kwargs
    return loss, z_out, acc_out, pred_out, ctx


def _scale_gradients(grads, grad_output, in_place):
    if torch.equal(grad_output, torch.ones((), device=grad_output.device, dtype=grad_output.dtype)):
        return grads
    if in_place:
        for grad in grads:
            if grad is not None:
                grad.mul_(grad_output)
        return grads
    return tuple(grad * grad_output if grad is not None else None for grad in grads)


def _flce_backward(ctx, grad_output):
    if not getattr(ctx, "_native", False):
        raise RuntimeError("FLCE backward state is missing its native CuTe DSL marker.")
    handoff = not ctx._native_gradients_consumed
    if handoff:
        grads = (ctx.grad_input, ctx.grad_weight, ctx.grad_bias)
        ctx.grad_input = None
        ctx.grad_weight = None
        ctx.grad_bias = None
        ctx._native_gradients_consumed = True
    else:
        if ctx._repeated_grad_input is None:
            X, W, target = ctx.saved_tensors
            _, _, _, _, gX, gW, gb = _native_forward(
                X,
                W,
                target,
                ctx.native_bias,
                ctx.native_ce_weight,
                needs_grad=True,
                **ctx.native_kwargs,
            )
            ctx._repeated_grad_input = gX
            ctx._repeated_grad_weight = gW
            ctx._repeated_grad_bias = gb
        grads = (ctx._repeated_grad_input, ctx._repeated_grad_weight, ctx._repeated_grad_bias)
    gX, gW, gb = _scale_gradients(grads, grad_output, in_place=handoff)
    return (gX, gW, None, gb, None, None, None, None, None, None, None, None, None, None, None)


def fused_linear_cross_entropy_backward(ctx, grad_output):
    grads = _flce_backward(ctx, grad_output)
    h_orig = getattr(ctx, "h_orig", None)
    if h_orig is not None:
        grads = (grads[0][:, :h_orig].contiguous(), grads[1][:, :h_orig].contiguous(), *grads[2:])
    return grads


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
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
        return_z_loss=False,
        accum_dtype=None,
        use_token_scaling=False,
        return_token_accuracy=False,
        return_predicted_tokens=False,
    ):
        loss, z_loss, token_accuracy, predicted_tokens, state = fused_linear_cross_entropy_forward(
            _input,
            weight,
            target,
            bias,
            ce_weight,
            ignore_index,
            lse_square_scale,
            label_smoothing,
            reduction,
            softcap,
            return_z_loss,
            accum_dtype,
            use_token_scaling,
            return_token_accuracy,
            return_predicted_tokens,
        )
        if getattr(state, "saved_tensors", None) is not None:
            ctx.save_for_backward(*state.saved_tensors)
        for key, value in vars(state).items():
            if key != "saved_tensors":
                setattr(ctx, key, value)
        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2=None, grad_output3=None, grad_output4=None):
        return fused_linear_cross_entropy_backward(ctx, grad_output)
