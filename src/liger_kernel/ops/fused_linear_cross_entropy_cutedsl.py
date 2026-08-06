import torch

from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_backward as _triton_backward
from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward as _triton_forward
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd


class _TritonFallbackFunction(torch.autograd.Function):
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
        needs_grad = _input.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
        defer_none = reduction == "none" and needs_grad
        with torch.no_grad():
            loss, z_loss, token_accuracy, predicted_tokens, grad_input, grad_weight, grad_bias = _triton_forward(
                _input=_input,
                weight=weight,
                target=target,
                ce_weight=ce_weight,
                bias=bias,
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
                compute_gradients=False if defer_none else needs_grad,
                weight_requires_grad=weight.requires_grad,
            )
        ctx.defer_none = defer_none
        if defer_none:
            ctx.save_for_backward(_input.detach(), weight.detach(), target)
            ctx.bias = bias.detach() if bias is not None else None
            ctx.forward_kwargs = {
                "ce_weight": ce_weight,
                "ignore_index": ignore_index,
                "lse_square_scale": lse_square_scale,
                "label_smoothing": label_smoothing,
                "reduction": reduction,
                "softcap": softcap,
                "accum_dtype": accum_dtype,
                "use_token_scaling": use_token_scaling,
            }
            ctx.weight_requires_grad = weight.requires_grad
        else:
            ctx.grad_input = grad_input
            ctx.grad_weight = grad_weight
            ctx.grad_bias = grad_bias
        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2=None, grad_output3=None, grad_output4=None):
        if ctx.defer_none:
            _input, weight, target = ctx.saved_tensors
            _, _, _, _, grad_input, grad_weight, grad_bias = _triton_forward(
                _input=_input,
                weight=weight,
                target=target,
                bias=ctx.bias,
                token_grad_output=grad_output,
                compute_gradients=True,
                weight_requires_grad=ctx.weight_requires_grad,
                **ctx.forward_kwargs,
            )
        else:
            grad_input, grad_weight, grad_bias = _triton_backward(
                grad_output,
                ctx.grad_input,
                ctx.grad_weight,
                ctx.grad_bias,
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
            None,
            None,
            None,
        )


class LigerFusedLinearCrossEntropyFunction:
    """Dispatch CuTe DSL-supported inputs natively and all other inputs to Triton."""

    @staticmethod
    def apply(
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
        from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction as native_function,
        )
        from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import _validate_inputs
        from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import supports_fused_linear_cross_entropy

        if supports_fused_linear_cross_entropy(_input, weight, bias, reduction):
            function = native_function
        else:
            _validate_inputs(_input, weight, target, bias, ce_weight, reduction, ignore_index)
            function = _TritonFallbackFunction
        return function.apply(
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
