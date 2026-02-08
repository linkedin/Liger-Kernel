import warnings

import torch
import torch.nn as nn

from liger_kernel.ops.mhc import liger_mhc_coeffs
from liger_kernel.ops.mhc import liger_mhc_post_res
from liger_kernel.ops.mhc import liger_mhc_pre


class LigerMHC(nn.Module):
    """
    Wraps a layer F: [..., C] -> [..., C] with mHC residual streams: [..., HC, C].

    Args:
        layer: module applied to the aggregated stream input
        hc: number of residual streams (n in the paper)
        c: per-stream channel dimension
        allow_fp32: if True, accept FP32 input. Note that FP32 mode does NOT
            use Tensor Cores and will be slower than BF16/FP16. Use only when
            FP32 precision is strictly required.

    Note:
        HC (number of residual streams) is recommended to be in range [2, 16].
        Larger values may cause register pressure and increased compile time
        due to Triton loop unrolling.
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

        layer_device = None
        for param in self.layer.parameters(recurse=True):
            layer_device = param.device
            break
        if layer_device is None:
            for buf in self.layer.buffers(recurse=True):
                layer_device = buf.device
                break
        if layer_device is None:
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
