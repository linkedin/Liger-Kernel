from typing import Any
from typing import Callable
from typing import Tuple

import torch

from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.triton.mhc import mhc_mm_norm_bwd
from liger_kernel.triton.mhc import mhc_mm_norm_fwd
from liger_kernel.triton.mhc import mhc_post_res_bwd
from liger_kernel.triton.mhc import mhc_post_res_fwd
from liger_kernel.triton.mhc import mhc_pre_bwd
from liger_kernel.triton.mhc import mhc_pre_fwd
from liger_kernel.triton.mhc import mhc_sinkhorn_bwd
from liger_kernel.triton.mhc import mhc_split_sinkhorn_fwd


def _flatten_tokens(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Flattens leading dimensions so x becomes [N, HC, C].
    Returns (x_flat, outer_shape).
    """
    assert x.dim() >= 3, "x must be [..., HC, C]"
    outer = tuple(x.shape[:-2])
    hc, c = x.shape[-2], x.shape[-1]
    n = 1
    for d in outer:
        n *= int(d)
    return x.contiguous().view(n, hc, c), outer


def _unflatten_tokens(y: torch.Tensor, outer: Tuple[int, ...]) -> torch.Tensor:
    return y.view(*outer, *y.shape[1:])


class LigerMHCCoeffsFunction(torch.autograd.Function):
    """
    Autograd function for mHC coefficient computation.

    Memory/Compute Trade-off:
        When gradients are needed, Sinkhorn iteration history (hist) is saved
        during forward to avoid recomputation in backward. This increases
        memory usage by O(N * tmax * HC^2) but reduces backward compute.
    """

    @staticmethod
    @ensure_contiguous
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,  # [..., HC, C] bf16/fp16 (or fp32 if allow_fp32)
        phi: torch.Tensor,  # [HC*C, M]
        b: torch.Tensor,  # [M]
        alpha_pre: torch.Tensor,  # scalar
        alpha_post: torch.Tensor,  # scalar
        alpha_res: torch.Tensor,  # scalar
        allow_fp32: bool,
        tmax: int,
        rms_eps: float,
        pre_eps: float,
        sinkhorn_eps: float,
        post_mult: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if allow_fp32:
            assert x.dtype in (
                torch.bfloat16,
                torch.float16,
                torch.float32,
            ), "x should be BF16/FP16/FP32 when allow_fp32=True"
        else:
            assert x.dtype in (torch.bfloat16, torch.float16), "x should be BF16/FP16 (set allow_fp32=True for FP32)"
        x_flat, outer = _flatten_tokens(x)
        N, HC, C = x_flat.shape
        K = HC * C
        x_mat = x_flat.view(N, K)

        assert phi.dim() == 2 and phi.shape[0] == K, f"phi must be [HC*C, M], got {tuple(phi.shape)}"
        M = int(phi.shape[1])
        assert b.shape == (M,), f"b must be [M], got {tuple(b.shape)}"

        # (1) fused coeff matmul + norm
        mix, invr = mhc_mm_norm_fwd(x_mat, phi, eps=float(rms_eps))

        # (2) split + sigmoid + sinkhorn
        need_hist = any(ctx.needs_input_grad)
        if need_hist:
            h_pre, h_post, h_res, hist = mhc_split_sinkhorn_fwd(
                mix,
                b,
                alpha_pre,
                alpha_post,
                alpha_res,
                tmax=int(tmax),
                pre_eps=float(pre_eps),
                sinkhorn_eps=float(sinkhorn_eps),
                post_mult=float(post_mult),
                return_hist=True,
            )
        else:
            h_pre, h_post, h_res = mhc_split_sinkhorn_fwd(
                mix,
                b,
                alpha_pre,
                alpha_post,
                alpha_res,
                tmax=int(tmax),
                pre_eps=float(pre_eps),
                sinkhorn_eps=float(sinkhorn_eps),
                post_mult=float(post_mult),
            )
            hist = None

        # Save for backward
        if hist is not None:
            ctx.save_for_backward(x_mat, phi, b, mix, invr, alpha_pre, alpha_post, alpha_res, hist)
        else:
            ctx.save_for_backward(x_mat, phi, b, mix, invr, alpha_pre, alpha_post, alpha_res)
        ctx.meta = (
            outer,
            HC,
            C,
            int(tmax),
            float(rms_eps),
            float(pre_eps),
            float(sinkhorn_eps),
            float(post_mult),
            hist is not None,
        )

        return (
            _unflatten_tokens(h_pre, outer),
            _unflatten_tokens(h_post, outer),
            _unflatten_tokens(h_res, outer),
        )

    @staticmethod
    @ensure_contiguous
    def backward(
        ctx: Any,
        grad_h_pre: torch.Tensor | None,
        grad_h_post: torch.Tensor | None,
        grad_h_res: torch.Tensor | None,
    ):
        saved = ctx.saved_tensors
        outer, HC, C, tmax, rms_eps, pre_eps, sinkhorn_eps, post_mult, has_hist = ctx.meta
        if has_hist:
            x_mat, phi, b, mix, invr, alpha_pre, alpha_post, alpha_res, hist = saved
        else:
            x_mat, phi, b, mix, invr, alpha_pre, alpha_post, alpha_res = saved
            hist = None
        N = x_mat.shape[0]
        M = mix.shape[1]
        assert M == HC * HC + 2 * HC

        need_pre = grad_h_pre is not None
        need_post = grad_h_post is not None
        need_res = grad_h_res is not None

        # flatten grads (None -> zeros)
        if need_pre:
            gh_pre = grad_h_pre.contiguous().view(N, HC).to(torch.float32)
        else:
            gh_pre = torch.zeros((N, HC), device=mix.device, dtype=torch.float32)
        if need_post:
            gh_post = grad_h_post.contiguous().view(N, HC).to(torch.float32)
        else:
            gh_post = torch.zeros((N, HC), device=mix.device, dtype=torch.float32)
        if need_res:
            gh_res = grad_h_res.contiguous().view(N, HC, HC).to(torch.float32)
        else:
            gh_res = torch.zeros((N, HC, HC), device=mix.device, dtype=torch.float32)

        # --- Sinkhorn backward -> grad logits for residual matrix
        if need_res:
            grad_res_logits = mhc_sinkhorn_bwd(
                mix,
                b,
                alpha_res,
                gh_res,
                tmax=tmax,
                sinkhorn_eps=sinkhorn_eps,
                hist=hist,
            )  # [N, HC, HC] fp32
        else:
            grad_res_logits = gh_res

        # --- Pre/post derivatives (sigmoid)
        mix_pre = mix[:, :HC]
        mix_post = mix[:, HC : 2 * HC]
        mix_res = mix[:, 2 * HC :]

        b_pre = b[:HC]
        b_post = b[HC : 2 * HC]
        if need_pre:
            pre_logits = mix_pre * alpha_pre + b_pre
            pre_sig = torch.sigmoid(pre_logits)
            grad_pre_logits = gh_pre * (pre_sig * (1.0 - pre_sig))  # [N,HC]
        else:
            grad_pre_logits = gh_pre

        if need_post:
            post_logits = mix_post * alpha_post + b_post
            post_sig = torch.sigmoid(post_logits)
            grad_post_logits = gh_post * (post_mult * post_sig * (1.0 - post_sig))  # [N,HC]
        else:
            grad_post_logits = gh_post

        grad_res_logits_flat = grad_res_logits.reshape(N, HC * HC)

        # --- Grad w.r.t mix
        grad_mix = torch.empty_like(mix)
        grad_mix[:, :HC] = grad_pre_logits * alpha_pre
        grad_mix[:, HC : 2 * HC] = grad_post_logits * alpha_post
        grad_mix[:, 2 * HC :] = grad_res_logits_flat * alpha_res

        # --- Grad w.r.t b
        grad_b = torch.zeros_like(b, dtype=torch.float32)
        if need_pre:
            grad_b[:HC] = grad_pre_logits.sum(dim=0)
        if need_post:
            grad_b[HC : 2 * HC] = grad_post_logits.sum(dim=0)
        if need_res:
            grad_b[2 * HC :] = grad_res_logits_flat.sum(dim=0)

        # --- Grad w.r.t alphas
        if need_pre:
            grad_alpha_pre = (grad_pre_logits * mix_pre).sum()
        else:
            grad_alpha_pre = torch.zeros((), device=mix.device, dtype=torch.float32)
        if need_post:
            grad_alpha_post = (grad_post_logits * mix_post).sum()
        else:
            grad_alpha_post = torch.zeros((), device=mix.device, dtype=torch.float32)
        if need_res:
            grad_alpha_res = (grad_res_logits_flat * mix_res).sum()
        else:
            grad_alpha_res = torch.zeros((), device=mix.device, dtype=torch.float32)

        # --- Grad w.r.t x and phi via fused mm+norm backward
        grad_x_mat, grad_phi = mhc_mm_norm_bwd(
            x_mat,
            phi,
            mix,
            invr,
            grad_mix,
        )

        grad_x = grad_x_mat.view(N, HC, C)
        grad_x = _unflatten_tokens(grad_x, outer)

        # Return grads for each forward input
        return (
            grad_x,  # x
            grad_phi,  # phi
            grad_b,  # b
            grad_alpha_pre,  # alpha_pre
            grad_alpha_post,  # alpha_post
            grad_alpha_res,  # alpha_res
            None,  # allow_fp32
            None,
            None,
            None,
            None,
            None,  # config scalars
        )


def liger_mhc_coeffs(
    x: torch.Tensor,
    phi: torch.Tensor,
    b: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    *,
    allow_fp32: bool = False,
    tmax: int = 20,
    rms_eps: float = 1e-6,
    pre_eps: float = 0.0,
    sinkhorn_eps: float = 1e-6,
    post_mult: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute mHC coefficients (h_pre, h_post, h_res).

    Args:
        x: [..., HC, C] BF16/FP16 (or FP32 if allow_fp32=True)
        phi: [HC*C, HC*HC + 2*HC]
        b: [HC*HC + 2*HC]
        alpha_pre/post/res: scalars (FP32 tensors recommended)
        HC: number of residual streams (n in the paper)
        allow_fp32: if True, allow FP32 input and compute/output in FP32
    Returns:
        h_pre: [..., HC] FP32
        h_post: [..., HC] FP32
        h_res: [..., HC, HC] FP32
    """
    assert x.is_cuda, "CUDA only"
    if allow_fp32:
        assert x.dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ), "x should be BF16/FP16/FP32 when allow_fp32=True"
    else:
        assert x.dtype in (torch.bfloat16, torch.float16), "x should be BF16/FP16 (set allow_fp32=True for FP32)"
    assert phi.is_cuda and b.is_cuda
    return LigerMHCCoeffsFunction.apply(
        x,
        phi,
        b,
        alpha_pre,
        alpha_post,
        alpha_res,
        allow_fp32,
        int(tmax),
        float(rms_eps),
        float(pre_eps),
        float(sinkhorn_eps),
        float(post_mult),
    )


class LigerMHCPreFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx: Any, x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        x_flat, outer = _flatten_tokens(x)
        h_pre_flat = h_pre.contiguous().view(x_flat.shape[0], x_flat.shape[1]).to(torch.float32)
        out = mhc_pre_fwd(x_flat, h_pre_flat)  # [N,C] fp32
        ctx.save_for_backward(x_flat, h_pre_flat)
        ctx.outer = outer
        out = out.to(x_flat.dtype)
        return _unflatten_tokens(out, outer)

    @staticmethod
    @ensure_contiguous
    def backward(ctx: Any, grad_out: torch.Tensor):
        x_flat, h_pre_flat = ctx.saved_tensors
        outer = ctx.outer
        N, HC, C = x_flat.shape
        go = grad_out.contiguous().view(N, C).to(torch.float32)
        grad_x, grad_h = mhc_pre_bwd(x_flat, h_pre_flat, go)
        grad_x = grad_x.to(x_flat.dtype)
        return _unflatten_tokens(grad_x.view(N, HC, C), outer), _unflatten_tokens(grad_h.view(N, HC), outer)


def liger_mhc_pre(x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
    """
    Apply H_pre: x_in = sum_i h_pre[i] * x[i]
    """
    return LigerMHCPreFunction.apply(x, h_pre)


class LigerMHCPostResFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx: Any, x: torch.Tensor, f_out: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor
    ) -> torch.Tensor:
        x_flat, outer = _flatten_tokens(x)
        N, HC, C = x_flat.shape
        f_flat = f_out.contiguous().view(N, C)
        h_post_flat = h_post.contiguous().view(N, HC).to(torch.float32)
        h_res_flat = h_res.contiguous().view(N, HC, HC).to(torch.float32)
        out = mhc_post_res_fwd(x_flat, f_flat, h_post_flat, h_res_flat)  # [N,HC,C] fp32
        ctx.save_for_backward(x_flat, f_flat, h_post_flat, h_res_flat)
        ctx.outer = outer
        out = out.to(x_flat.dtype)
        return _unflatten_tokens(out, outer)

    @staticmethod
    @ensure_contiguous
    def backward(ctx: Any, grad_out: torch.Tensor):
        x_flat, f_flat, h_post_flat, h_res_flat = ctx.saved_tensors
        outer = ctx.outer
        N, HC, C = x_flat.shape
        go = grad_out.contiguous().view(N, HC, C).to(torch.float32)

        grad_x, grad_f, grad_hpost, grad_hres = mhc_post_res_bwd(x_flat, f_flat, h_post_flat, h_res_flat, go)

        return (
            _unflatten_tokens(grad_x.to(x_flat.dtype).view(N, HC, C), outer),
            _unflatten_tokens(grad_f.to(f_flat.dtype).view(N, C), outer),
            _unflatten_tokens(grad_hpost.view(N, HC), outer),
            _unflatten_tokens(grad_hres.view(N, HC, HC), outer),
        )


def liger_mhc_post_res(x: torch.Tensor, f_out: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor) -> torch.Tensor:
    """
    Apply H_res and H_post:

      x_out[o] = sum_i h_res[o,i] * x[i] + h_post[o] * f_out

    Shapes:
      x: [..., HC, C]
      f_out: [..., C]
      h_post: [..., HC]
      h_res: [..., HC, HC]
    """
    return LigerMHCPostResFunction.apply(x, f_out, h_post, h_res)


def liger_mhc_apply(
    x: torch.Tensor,
    f_out: torch.Tensor,
    h_pre: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    *,
    return_x_in: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience API: apply coefficients to get x_in and x_out.

    Args:
        x: [..., HC, C]
        f_out: [..., C]
        h_pre/h_post/h_res: coefficients from liger_mhc_coeffs
        return_x_in: if True, returns (x_out, x_in)
    """
    x_in = liger_mhc_pre(x, h_pre)
    x_out = liger_mhc_post_res(x, f_out, h_post, h_res)
    if return_x_in:
        return x_out, x_in
    return x_out


def liger_mhc_forward(
    x: torch.Tensor,
    layer: Callable[[torch.Tensor], torch.Tensor],
    phi: torch.Tensor,
    b: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    *,
    allow_fp32: bool = False,
    tmax: int = 20,
    rms_eps: float = 1e-6,
    pre_eps: float = 0.0,
    sinkhorn_eps: float = 1e-6,
    post_mult: float = 2.0,
    return_coeffs: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    High-level helper: compute coeffs, apply pre, run layer, then apply post+res.
    """
    h_pre, h_post, h_res = liger_mhc_coeffs(
        x,
        phi,
        b,
        alpha_pre,
        alpha_post,
        alpha_res,
        allow_fp32=allow_fp32,
        tmax=tmax,
        rms_eps=rms_eps,
        pre_eps=pre_eps,
        sinkhorn_eps=sinkhorn_eps,
        post_mult=post_mult,
    )
    x_in = liger_mhc_pre(x, h_pre)
    layer_dtype = x_in.dtype
    if hasattr(layer, "parameters"):
        try:
            layer_dtype = next(layer.parameters()).dtype  # type: ignore[arg-type]
        except StopIteration:
            layer_dtype = x_in.dtype
    if x_in.dtype != layer_dtype:
        x_in = x_in.to(layer_dtype)
    f_out = layer(x_in)
    x_out = liger_mhc_post_res(x, f_out, h_post, h_res)
    if return_coeffs:
        return x_out, (h_pre, h_post, h_res)
    return x_out


def mhc_coeffs(
    x: torch.Tensor,
    phi: torch.Tensor,
    b: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    *,
    allow_fp32: bool = False,
    tmax: int = 20,
    rms_eps: float = 1e-6,
    pre_eps: float = 0.0,
    sinkhorn_eps: float = 1e-6,
    post_mult: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward-compatible alias for liger_mhc_coeffs.
    """
    return liger_mhc_coeffs(
        x,
        phi,
        b,
        alpha_pre,
        alpha_post,
        alpha_res,
        allow_fp32=allow_fp32,
        tmax=tmax,
        rms_eps=rms_eps,
        pre_eps=pre_eps,
        sinkhorn_eps=sinkhorn_eps,
        post_mult=post_mult,
    )


def mhc_pre(x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
    """
    Backward-compatible alias for liger_mhc_pre.
    """
    return liger_mhc_pre(x, h_pre)


def mhc_post_res(x: torch.Tensor, f_out: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor) -> torch.Tensor:
    """
    Backward-compatible alias for liger_mhc_post_res.
    """
    return liger_mhc_post_res(x, f_out, h_post, h_res)


def mhc_apply(
    x: torch.Tensor,
    f_out: torch.Tensor,
    h_pre: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    *,
    return_x_in: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward-compatible alias for liger_mhc_apply.
    """
    return liger_mhc_apply(
        x,
        f_out,
        h_pre,
        h_post,
        h_res,
        return_x_in=return_x_in,
    )


def mhc_forward(
    x: torch.Tensor,
    layer: Callable[[torch.Tensor], torch.Tensor],
    phi: torch.Tensor,
    b: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    *,
    allow_fp32: bool = False,
    tmax: int = 20,
    rms_eps: float = 1e-6,
    pre_eps: float = 0.0,
    sinkhorn_eps: float = 1e-6,
    post_mult: float = 2.0,
    return_coeffs: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Backward-compatible alias for liger_mhc_forward.
    """
    return liger_mhc_forward(
        x,
        layer,
        phi,
        b,
        alpha_pre,
        alpha_post,
        alpha_res,
        allow_fp32=allow_fp32,
        tmax=tmax,
        rms_eps=rms_eps,
        pre_eps=pre_eps,
        sinkhorn_eps=sinkhorn_eps,
        post_mult=post_mult,
        return_coeffs=return_coeffs,
    )
