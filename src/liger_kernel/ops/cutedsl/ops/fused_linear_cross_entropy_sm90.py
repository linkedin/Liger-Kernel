"""Hopper (SM90) CuTe DSL **fused linear cross entropy**.

This operator combines the classifier projection ``Z = X @ W.T``, softmax
cross entropy, and input gradient in the autograd *forward*. The weight
gradient is deferred to *backward* so its WGMMA epilogue can apply the upstream
gradient without a separate full-tensor scale. It is a distinct implementation targeted at the Llama-scale shape
``M=4096, H=4096, V=128256`` (BF16).

Design (three logical GEMMs, "gradient-in-forward")
---------------------------------------------------
The forward runs, in order:

1. **logits GEMM** ``logits[M, V] = X[M, H] @ W[V, H].T`` -- a WGMMA GEMM that
   writes the raw BF16 logits to HBM exactly once (they are reused as the dZ
   scratch, so no separate logits buffer survives).
2. **CE + dZ** -- an eight-CTA one-read cluster kernel reads the logits with
   aligned 128-bit copies, emits the per-token NLL ``nll[M]`` and overwrites the
   logits *in place* with
   ``dZ = (softmax(logits) - onehot(target)) * row_scale`` (BF16), where
   ``row_scale`` folds the reduction (``1/N`` for ``mean``, ``1`` for ``sum``).
   Unsupported vector-cluster shapes use the safe partitioned fallback.
3. **dX GEMM** ``dX[M, H] = dZ @ W`` runs in forward. **dW GEMM**
   ``dW[V, H] = dZ.T @ X`` runs in backward. Both use the same WGMMA primitive
   on transposed operand *views*, and logits are never recomputed.

The forward returns the reduced scalar loss and saves ``dX``, ``dZ``, and
``X``. Backward scales ``dX`` and computes the scaled ``dW``.

Supported semantics (deliberately narrow MVP)
---------------------------------------------
* Hopper (compute capability 9.0) only.
* BF16 ``input[M, H]`` and ``weight[V, H]``, both contiguous; int64
  ``target[M]``.
* ``reduction`` in ``{"mean", "sum"}``; ``ignore_index`` honoured.
* **No** bias, class weight, label smoothing, ``softcap``, ``lse_square_scale``
  / z-loss, token-scaling or accuracy/metric outputs -- every one of these is
  *rejected* with a clear error rather than silently ignored.

Shape guards
------------
``H`` is padded up to ``TILE_N`` internally (a contraction-dim pad is exact and
the padded gradient columns are sliced off).  ``M`` must be a multiple of
``TILE_M`` (128) and ``V`` a multiple of ``TILE_N`` (256); both hold for the
representative shape.  Other ``M``/``V`` are rejected -- ragged token/vocab
tiling is left to a future version.

Measured performance (H200, ``M=4096, H=4096, V=128256``, BF16, mean)
---------------------------------------------------------------------
Full forward+backward, interleaved CUDA-event medians (see
``optimization/bench_flce_sm90.py``):

* this kernel: ~18.06 ms hot median
* QuACK 0.6.1 ``chunked_linear_cross_entropy``: ~17.33 ms hot median
* Triton Liger FLCE: ~141 ms

The CuTe DSL path is about 7.8x faster than the existing Triton path and remains
about 4.2% slower than QuACK at this shape.
"""

import cuda.bindings.driver as cuda
import torch

from liger_kernel.ops.cutedsl.ops._fused_linear_cross_entropy_backward_sm90 import flce_dw
from liger_kernel.ops.cutedsl.ops._fused_linear_cross_entropy_backward_sm90 import flce_dx
from liger_kernel.ops.cutedsl.ops._fused_linear_cross_entropy_forward_sm90 import TILE_M
from liger_kernel.ops.cutedsl.ops._fused_linear_cross_entropy_forward_sm90 import TILE_N
from liger_kernel.ops.cutedsl.ops._fused_linear_cross_entropy_forward_sm90 import cross_entropy_dz
from liger_kernel.ops.cutedsl.ops._fused_linear_cross_entropy_forward_sm90 import tile_gemm
from liger_kernel.ops.cutedsl.ops.utils import ensure_cuda_context
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd


def _reject_unsupported(
    bias, ce_weight, label_smoothing, softcap, lse_square_scale, use_token_scaling, return_z_loss, return_token_accuracy
):
    if bias is not None:
        raise NotImplementedError("CuTe DSL FLCE (SM90) does not support bias")
    if ce_weight is not None:
        raise NotImplementedError("CuTe DSL FLCE (SM90) does not support class weights")
    if label_smoothing:
        raise NotImplementedError("CuTe DSL FLCE (SM90) does not support label smoothing")
    if softcap is not None:
        raise NotImplementedError("CuTe DSL FLCE (SM90) does not support logit softcapping")
    if lse_square_scale:
        raise NotImplementedError("CuTe DSL FLCE (SM90) does not support z-loss / lse_square_scale")
    if use_token_scaling:
        raise NotImplementedError("CuTe DSL FLCE (SM90) does not support token scaling")
    if return_z_loss:
        raise NotImplementedError("CuTe DSL FLCE (SM90) does not return z-loss")
    if return_token_accuracy:
        raise NotImplementedError("CuTe DSL FLCE (SM90) does not return token accuracy")


def _validate(_input, weight, target, reduction):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(_input.device) != (9, 0):
        raise RuntimeError("CuTe DSL FLCE requires a Hopper (compute capability 9.0) GPU")
    if _input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("CuTe DSL FLCE supports BF16 input and weight only")
    if target.dtype != torch.int64:
        raise TypeError("target must be an int64 tensor")
    if _input.ndim != 2 or weight.ndim != 2 or target.ndim != 1:
        raise ValueError("expected input[M, H], weight[V, H] and target[M]")
    if _input.shape[0] != target.shape[0] or _input.shape[1] != weight.shape[1]:
        raise ValueError(
            f"input {tuple(_input.shape)}, weight {tuple(weight.shape)} and target "
            f"{tuple(target.shape)} shapes are incompatible"
        )
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
    m = _input.shape[0]
    v = weight.shape[0]
    if m % TILE_M != 0:
        raise NotImplementedError(f"CuTe DSL FLCE (MVP) requires M to be a multiple of {TILE_M}; got M={m}")
    if v % TILE_N != 0:
        raise NotImplementedError(f"CuTe DSL FLCE (MVP) requires V to be a multiple of {TILE_N}; got V={v}")


def _pad_hidden(x, weight):
    h = x.shape[1]
    padded = (h + TILE_N - 1) // TILE_N * TILE_N
    if padded == h:
        return x.contiguous(), weight.contiguous(), h
    x_p = torch.nn.functional.pad(x.contiguous(), (0, padded - h))
    w_p = torch.nn.functional.pad(weight.contiguous(), (0, padded - h))
    return x_p, w_p, h


def fused_linear_cross_entropy_forward_sm90(
    _input,
    weight,
    target,
    ignore_index=-100,
    reduction="mean",
    defer_dw=False,
):
    """Run the three GEMMs + CE and return ``(loss, dx, dw)`` in caller dtypes.

    ``dx`` / ``dw`` are the gradients of ``loss`` (reduction already folded in);
    the autograd wrapper scales them by the scalar upstream gradient.
    """
    with torch.cuda.device(_input.device):
        ensure_cuda_context()
        x_pad, w_pad, hidden = _pad_hidden(_input, weight)
        target = target.contiguous()
        m, h_pad = x_pad.shape
        v = w_pad.shape[0]

        if reduction == "mean":
            loss_scale = (target != ignore_index).sum().clamp_min(1).float().reciprocal()
        else:
            loss_scale = None

        stream = cuda.CUstream(torch.cuda.current_stream(_input.device).cuda_stream)

        # 1) logits = X @ W.T  (BF16, saved to HBM; reused in place as dZ)
        logits = torch.empty(m, v, device=x_pad.device, dtype=torch.bfloat16)
        tile_gemm(x_pad, w_pad, logits, a_leading=1, b_leading=1, stream=stream)

        # 2) CE + dZ (overwrites `logits` in place)
        nll = torch.empty(m, device=x_pad.device, dtype=torch.float32)
        cross_entropy_dz(logits, target, nll, 1.0, ignore_index, stream)
        dz = logits

        # 3) dX = dZ @ W ; dW = dZ.T @ X
        dx = flce_dx(
            dz,
            w_pad,
            out_dtype=_input.dtype,
            output_scale=loss_scale,
        )
        dx = dx[:, :hidden].contiguous()

        loss = nll.sum()
        if loss_scale is not None:
            loss = loss * loss_scale
        if defer_dw:
            return loss, dx, dz, x_pad, loss_scale, hidden

        dw = flce_dw(
            dz,
            x_pad,
            out_dtype=weight.dtype,
            output_scale=loss_scale,
        )
        dw = dw[:, :hidden].contiguous()
        return loss, dx, dw


class LigerFusedLinearCrossEntropySM90Function(torch.autograd.Function):
    """``apply(input, weight, target, ignore_index=-100, reduction="mean")``.

    Forward returns the scalar loss; backward scales the forward-computed input
    gradient and computes the weight gradient.
    """

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        ignore_index=-100,
        reduction="mean",
        bias=None,
        ce_weight=None,
        label_smoothing=0.0,
        softcap=None,
        lse_square_scale=0.0,
        use_token_scaling=False,
        return_z_loss=False,
        return_token_accuracy=False,
    ):
        _reject_unsupported(
            bias,
            ce_weight,
            label_smoothing,
            softcap,
            lse_square_scale,
            use_token_scaling,
            return_z_loss,
            return_token_accuracy,
        )
        _validate(_input, weight, target, reduction)
        loss, dx, dz, x_padded, loss_scale, hidden = fused_linear_cross_entropy_forward_sm90(
            _input,
            weight,
            target,
            ignore_index=ignore_index,
            reduction=reduction,
            defer_dw=True,
        )
        saved_loss_scale = (
            loss_scale if loss_scale is not None else torch.empty(0, device=_input.device, dtype=torch.float32)
        )
        ctx.save_for_backward(dx, dz, x_padded, saved_loss_scale)
        ctx.has_loss_scale = loss_scale is not None
        ctx.hidden = hidden
        ctx.weight_dtype = weight.dtype
        return loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output):
        dx, dz, x_padded, loss_scale = ctx.saved_tensors
        dx = dx * grad_output
        output_scale = grad_output * loss_scale if ctx.has_loss_scale else grad_output
        dw = flce_dw(
            dz,
            x_padded,
            out_dtype=ctx.weight_dtype,
            output_scale=output_scale,
        )
        dw = dw[:, : ctx.hidden].contiguous()
        return dx, dw, None, None, None, None, None, None, None, None, None, None, None


def liger_fused_linear_cross_entropy_sm90(_input, weight, target, ignore_index=-100, reduction="mean"):
    """Functional entry point returning the reduced scalar loss."""
    return LigerFusedLinearCrossEntropySM90Function.apply(_input, weight, target, ignore_index, reduction)


__all__ = [
    "LigerFusedLinearCrossEntropySM90Function",
    "fused_linear_cross_entropy_forward_sm90",
    "liger_fused_linear_cross_entropy_sm90",
]
