import warnings

import torch
import torch.nn as nn

from liger_kernel.transformers.functional import liger_mhc_coeffs
from liger_kernel.transformers.functional import liger_mhc_post_res
from liger_kernel.transformers.functional import liger_mhc_pre


class LigerMHC(nn.Module):
    """
    Manifold-Constrained Hyper-Connections (mHC) wrapper.

    Wraps an arbitrary layer ``F: [..., C] -> [..., C]`` with multiple residual
    streams, following the mHC architecture (arXiv:2512.24880). The input is a
    multi-stream tensor of shape ``[..., HC, C]`` where ``HC`` is the number of
    residual streams.

    The forward pass performs:

    1. **Coefficients** -- Compute data-dependent routing coefficients
       (``h_pre``, ``h_post``, ``h_res``) via a fused matmul + RMS
       normalization + Sinkhorn-Knopp iterations.
    2. **Pre-aggregate** -- ``x_in = sum_i h_pre[i] * x[i]``
       (shape: ``[..., C]``)
    3. **Layer** -- ``f_out = layer(x_in)``  (shape: ``[..., C]``)
    4. **Post + residual** --
       ``x_out[o] = sum_i h_res[o,i] * x[i] + h_post[o] * f_out``
       (shape: ``[..., HC, C]``)

    Args:
        layer: The module applied to the aggregated single-stream input.
            Must accept ``[..., C]`` and return ``[..., C]``. Common choices
            include ``nn.Linear``, attention layers, or MLP blocks.
        hc: Number of residual streams (called *n* in the original paper).
            Recommended range: [2, 16]. Larger values increase register
            pressure and Triton compile time.
        c: Per-stream channel dimension.
        tmax: Maximum Sinkhorn-Knopp iterations for doubly stochastic
            normalization of ``h_res``. Default: 20.
        rms_eps: Epsilon for RMS normalization of the projection.
            Default: 1e-6.
        pre_eps: Additive epsilon for ``h_pre`` after sigmoid. Default: 0.0.
        sinkhorn_eps: Epsilon added during Sinkhorn normalization.
            Default: 1e-6.
        post_mult: Scaling factor for ``h_post`` after sigmoid. Default: 2.0.
        phi_dtype: Dtype for the projection matrix ``phi``. Using float16 or
            bfloat16 enables Tensor Core acceleration. Default: torch.float16.
        allow_fp32: If True, accept FP32 input tensors. Note that FP32 mode
            does **not** use Tensor Cores and will be slower. Default: False.

    Learnable Parameters:
        - **phi** ``[HC*C, HC*HC + 2*HC]`` -- Projection matrix for computing
          routing coefficients from flattened stream tokens.
        - **b** ``[HC*HC + 2*HC]`` -- Bias for routing logits (float32).
        - **alpha_pre** (scalar) -- Scales pre-routing logits before sigmoid.
        - **alpha_post** (scalar) -- Scales post-routing logits before sigmoid.
        - **alpha_res** (scalar) -- Scales residual logits before Sinkhorn.

    Example::

        import torch
        import torch.nn as nn
        from liger_kernel.transformers import LigerMHC

        # Wrap a linear layer with 4 residual streams of dimension 256
        layer = nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
        mhc = LigerMHC(layer, hc=4, c=256, phi_dtype=torch.bfloat16).cuda()

        # Input: [batch, seq_len, num_streams, channels]
        x = torch.randn(2, 128, 4, 256, device="cuda", dtype=torch.bfloat16)
        out = mhc(x)  # shape: [2, 128, 4, 256]

        # In a transformer block (pseudocode):
        # x = mhc_attn(x)   # attention wrapped in LigerMHC
        # x = mhc_ffn(x)    # FFN wrapped in LigerMHC
    """

    def __init__(
        self,
        layer: nn.Module,
        *,
        hc: int,
        c: int,
        tmax: int = 20,
        rms_eps: float = 1e-6,
        pre_eps: float = 0.0,
        sinkhorn_eps: float = 1e-6,
        post_mult: float = 2.0,
        phi_dtype: torch.dtype = torch.float16,
        allow_fp32: bool = False,
    ):
        super().__init__()
        self.layer = layer
        # hc: number of residual streams (n in the paper)
        self.hc = int(hc)
        self.c = int(c)

        if hc > 16:
            warnings.warn(
                f"hc={hc} exceeds recommended range [2, 16]. "
                "Large values may cause register pressure and increased compile time.",
                stacklevel=2,
            )
        self.tmax = int(tmax)
        self.rms_eps = float(rms_eps)
        self.pre_eps = float(pre_eps)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.post_mult = float(post_mult)
        self.allow_fp32 = bool(allow_fp32)

        m = hc * hc + 2 * hc
        k = hc * c

        try:
            layer_device = next(self.layer.parameters()).device
        except StopIteration:
            layer_device = torch.device("cpu")

        # Note: for best speed, keep phi in BF16/FP16 to enable tensor-core matmul in Triton.
        self.phi = nn.Parameter(torch.randn(k, m, dtype=phi_dtype, device=layer_device) * 0.02)
        self.b = nn.Parameter(torch.zeros(m, dtype=torch.float32, device=layer_device))
        self.alpha_pre = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=layer_device))
        self.alpha_post = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=layer_device))
        self.alpha_res = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=layer_device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., HC, C] (BF16/FP16 recommended; FP32 allowed if allow_fp32=True)
        returns: [..., HC, C]
        """
        if x.shape[-2] != self.hc or x.shape[-1] != self.c:
            raise ValueError(f"Expected x.shape[-2:]=[{self.hc}, {self.c}], got {list(x.shape[-2:])}")

        h_pre, h_post, h_res = liger_mhc_coeffs(
            x,
            self.phi,
            self.b,
            self.alpha_pre,
            self.alpha_post,
            self.alpha_res,
            allow_fp32=self.allow_fp32,
            tmax=self.tmax,
            rms_eps=self.rms_eps,
            pre_eps=self.pre_eps,
            sinkhorn_eps=self.sinkhorn_eps,
            post_mult=self.post_mult,
        )
        x_in = liger_mhc_pre(x, h_pre)  # [..., C]
        layer_dtype = x_in.dtype
        for param in self.layer.parameters(recurse=True):
            layer_dtype = param.dtype
            break
        if x_in.dtype != layer_dtype:
            x_in = x_in.to(layer_dtype)
        f_out = self.layer(x_in)  # [..., C]
        x_out = liger_mhc_post_res(x, f_out, h_post, h_res)  # [..., HC, C]
        return x_out

    def extra_repr(self) -> str:
        return f"hc={self.hc}, c={self.c}, tmax={self.tmax}"
