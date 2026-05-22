import torch
import torch.nn as nn

from liger_kernel.ops import LigerRMSNormFunction


class LigerRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size,
        eps=1e-6,
        offset=0.0,
        casting_mode="llama",
        init_fn="ones",
        in_place=True,
        row_mode=None,
        elementwise_affine=True,
    ):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.variance_epsilon, self.offset, self.casting_mode, self.in_place, self.row_mode = (
            eps,
            offset,
            casting_mode,
            in_place,
            row_mode,
        )

    def forward(self, hidden_states):
        return LigerRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            self.offset,
            self.casting_mode,
            self.in_place,
            self.row_mode,
        )

    def extra_repr(self):
        return f"weight_shape={tuple(self.weight.shape) if self.weight is not None else None}, eps={self.variance_epsilon}, offset={self.offset}, in_place={self.in_place}, row_mode={self.row_mode}"


class LigerRMSNormForGemma(LigerRMSNorm):
    def __init__(
        self, hidden_size, eps=1e-6, offset=1.0, casting_mode="gemma", init_fn="zeros", in_place=True, row_mode=None
    ):
        super().__init__(hidden_size, eps, offset, casting_mode, init_fn, in_place, row_mode)


class LigerRMSNormForGemma2(LigerRMSNorm):
    def __init__(
        self, hidden_size, eps=1e-6, offset=1.0, casting_mode="gemma", init_fn="zeros", in_place=False, row_mode=None
    ):
        super().__init__(hidden_size, eps, offset, casting_mode, init_fn, in_place, row_mode)


class LigerRMSNormForGemma3(LigerRMSNorm):
    """Gemma3RMSNorm has a dim argument not hidden_size used in q_norm and k_norm."""

    def __init__(self, dim, eps=0.000001, offset=1.0, casting_mode="gemma", init_fn="zeros", in_place=False):
        super().__init__(dim, eps, offset, casting_mode, init_fn, in_place)


class LigerRMSNormForGemma4(LigerRMSNorm):
    """Gemma4RMSNorm inherits Gemma3nRMSNorm (not Gemma3RMSNorm); reusing
    LigerRMSNormForGemma3 here would silently diverge training because
    Gemma3's subclass applies ``(1 + w) * x`` semantics via the +1 offset.

    Gemma4RMSNorm semantics (see transformers.models.gemma4.modeling_gemma4):
      - weight initialized to ones (not zeros, unlike Gemma3)
      - no (1 + weight) offset — scales by weight directly
      - fp32 compute, cast back to input dtype
      - ``with_scale=False`` variant has NO weight parameter and is used for
        ``v_norm`` on attention (scale-free RMS normalization).

    When ``with_scale=False`` the Liger kernel has no weight to multiply by,
    so we fall back to a plain torch implementation that matches HF exactly.
    """

    def __init__(
        self,
        dim,
        eps=1e-6,
        offset=0.0,
        casting_mode="gemma",
        init_fn="ones",
        in_place=False,
        with_scale=True,
    ):
        super().__init__(dim, eps, offset, casting_mode, init_fn, in_place, elementwise_affine=with_scale)
        self.with_scale = with_scale

    def forward(self, hidden_states):
        if not self.with_scale:
            # Mirrors HF's Gemma4RMSNorm forward for the with_scale=False case:
            # scale-free RMS normalization with fp32 compute, cast back to input dtype.
            input_dtype = hidden_states.dtype
            x = hidden_states.float()
            mean_sq = x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon
            return (x * torch.pow(mean_sq, -0.5)).to(input_dtype)
        return super().forward(hidden_states)


class LigerRMSNormForOlmo2(LigerRMSNorm):
    def __init__(
        self, hidden_size, eps=1e-6, offset=0.0, casting_mode="llama", init_fn="ones", in_place=False, row_mode=None
    ):
        super().__init__(hidden_size, eps, offset, casting_mode, init_fn, in_place, row_mode)


class LigerRMSNormForGlm4(LigerRMSNorm):
    def __init__(
        self, hidden_size, eps=1e-6, offset=0.0, casting_mode="llama", init_fn="ones", in_place=False, row_mode=None
    ):
        super().__init__(hidden_size, eps, offset, casting_mode, init_fn, in_place, row_mode)


class LigerRMSNormForQwen3Next(LigerRMSNorm):
    def __init__(
        self, hidden_size, eps=1e-6, offset=1.0, casting_mode="gemma", init_fn="zeros", in_place=False, row_mode=None
    ):
        super().__init__(hidden_size, eps, offset, casting_mode, init_fn, in_place, row_mode)
