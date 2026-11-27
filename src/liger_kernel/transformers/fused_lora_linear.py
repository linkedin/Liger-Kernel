import math

import torch
import torch.nn as nn

from liger_kernel.transformers.functional import liger_fused_lora_linear


class LigerFusedLoRALinear(nn.Module):
    """LoRA-adapted linear layer with a frozen base weight.

    Mathematically equivalent to a PEFT LoRA ``nn.Linear`` whose ``lora_A`` is
    ``(R, in_features)``, ``lora_B`` is ``(out_features, R)`` and scaling is
    ``lora_alpha / R``. The base GEMM stays on cuBLAS; the LoRA epilogues are
    fused (see ``ops/fused_lora_linear.py``).

    Dropout is applied outside the fused op, matching PEFT: it runs on the
    input before both projections, and autograd handles its backward.
    """

    def __init__(
        self,
        in_features,
        out_features,
        r=16,
        lora_alpha=16,
        bias=False,
        lora_dropout=0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError(f"r must be positive, got {r}")
        if not 0.0 <= lora_dropout < 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1), got {lora_dropout}")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.scaling = lora_alpha / r

        self.base_layer = nn.Linear(in_features, out_features, bias=bias, **factory_kwargs)
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(r, in_features, **factory_kwargs))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, **factory_kwargs))
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        if self.lora_dropout > 0.0 and self.training:
            x = torch.nn.functional.dropout(x, p=self.lora_dropout, training=True)
        return liger_fused_lora_linear(
            x,
            self.base_layer.weight,
            self.lora_A,
            self.lora_B,
            self.base_layer.bias,
            self.scaling,
        )

    @classmethod
    def from_linear(cls, linear, r=16, lora_alpha=16, lora_dropout=0.0):
        """Build a frozen-base LoRA layer from an existing ``nn.Linear``."""
        layer = cls(
            linear.in_features,
            linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            bias=linear.bias is not None,
            lora_dropout=lora_dropout,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            layer.base_layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.base_layer.bias.copy_(linear.bias)
        return layer

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, lora_alpha={self.lora_alpha}, "
            f"lora_dropout={self.lora_dropout}, bias={self.base_layer.bias is not None}"
        )
