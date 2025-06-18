import torch
import torch.nn as nn

from liger_kernel.ops.rms_norm import LigerRMSNormFunction


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
    ):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.weight = nn.Parameter(torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size))
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
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, offset={self.offset}, in_place={self.in_place}, row_mode={self.row_mode}"


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
