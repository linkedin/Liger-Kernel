import inspect

import torch
import torch.nn as nn

from liger_kernel.ops import LigerRMSNormFunction


def _liger_rms_norm_supports_grouped() -> bool:
    """Whether the selected backend Function exposes the grouped RMSNorm ABI."""
    try:
        return "n_groups" in inspect.signature(LigerRMSNormFunction.forward).parameters
    except (AttributeError, TypeError, ValueError):
        return False


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
        group_size=None,
    ):
        super().__init__()
        self._liger_rms_norm_patched = True
        self._liger_rms_norm_supports_grouped = _liger_rms_norm_supports_grouped()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size))
        else:
            self.register_parameter("weight", None)
        if group_size is not None:
            if not isinstance(group_size, int) or group_size <= 0:
                raise ValueError(f"group_size must be a positive integer, got {group_size}.")
            if self.weight is None:
                raise ValueError("group_size requires elementwise_affine=True.")
            if hidden_size % group_size != 0:
                raise ValueError(f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size}).")
        self.group_size = group_size
        self.variance_epsilon, self.offset, self.casting_mode, self.in_place, self.row_mode = (
            eps,
            offset,
            casting_mode,
            in_place,
            row_mode,
        )

    def forward(self, hidden_states):
        # This method is also bound onto foreign RMSNorm instances during monkey patching, so
        # grouped metadata must be resolved from plain attributes with safe fallbacks.
        group_size = getattr(self, "group_size", None)
        weight = getattr(self, "weight", None)
        n_groups = None
        if group_size is not None:
            if weight is None:
                raise ValueError("group_size requires an elementwise-affine weight.")
            if not isinstance(group_size, int) or group_size <= 0:
                raise ValueError(f"group_size must be a positive integer, got {group_size}.")
            if weight.numel() % group_size != 0:
                raise ValueError(f"weight size ({weight.numel()}) must be divisible by group_size ({group_size}).")
            n_groups = weight.numel() // group_size
        eps = getattr(self, "eps", None)
        if eps is None:
            eps = self.variance_epsilon
        if n_groups is None:
            return LigerRMSNormFunction.apply(
                hidden_states,
                self.weight,
                eps,
                self.offset,
                self.casting_mode,
                self.in_place,
                self.row_mode,
            )
        if not self._liger_rms_norm_supports_grouped:
            raise RuntimeError("The selected RMSNorm backend does not support grouped RMSNorm.")
        return LigerRMSNormFunction.apply(
            hidden_states,
            self.weight,
            eps,
            self.offset,
            self.casting_mode,
            self.in_place,
            self.row_mode,
            n_groups,
        )

    def extra_repr(self):
        eps = getattr(self, "eps", None)
        if eps is None:
            eps = self.variance_epsilon
        return (
            f"weight_shape={tuple(self.weight.shape) if self.weight is not None else None}, "
            f"eps={eps}, offset={self.offset}, in_place={self.in_place}, "
            f"row_mode={self.row_mode}, group_size={getattr(self, 'group_size', None)}"
        )


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


class LigerRMSNormForMuseGlimmer(LigerRMSNorm):
    """MuseGlimmerRMSNorm semantics (see transformers.models.muse_glimmer.modeling_muse_glimmer):

      - weight initialized to ones, applied directly (no ``(1 + w)`` offset)
      - fp32 compute, cast back to input dtype (gemma-style casting)
      - ``with_scale=False`` variant has NO weight parameter and is used for ``qk_norm``,
        the embedding ``embed_norm`` and ``perception_emb_norm``.

    When ``with_scale=False`` the Liger kernel has no weight to multiply by, so we fall back
    to a plain torch implementation that matches HF exactly.
    """

    def __init__(
        self,
        dim=None,
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
            # Mirrors HF's MuseGlimmerRMSNorm forward for the with_scale=False case:
            # scale-free RMS normalization with fp32 compute, cast back to input dtype.
            input_dtype = hidden_states.dtype
            x = hidden_states.float()
            mean_sq = x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon
            return (x * torch.pow(mean_sq, -0.5)).to(input_dtype)
        return super().forward(hidden_states)


class LigerRMSNormForMuseGlimmerTextCentered(LigerRMSNorm):
    """MuseGlimmerTextCenteredRMSNorm scales by ``(1 + weight)`` with zero-initialized weights,
    computing in fp32 and casting back — i.e. Gemma semantics. ``in_place=False`` because each
    decoder layer chains ``pre_feedforward_layernorm`` and ``post_feedforward_layernorm`` around
    a residual connection.
    """

    def __init__(
        self, hidden_size, eps=1e-6, offset=1.0, casting_mode="gemma", init_fn="zeros", in_place=False, row_mode=None
    ):
        super().__init__(hidden_size, eps, offset, casting_mode, init_fn, in_place, row_mode)


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


class LigerRMSNormForQwen4Exp(LigerRMSNorm):
    """Drop-in replacement for ``transformers.models.qwen4_exp.Qwen4ExpTextRMSNorm``.

    Matches the HF signature ``__init__(dim, group_size=None, eps=1e-6)`` so it can be used as a
    class-level swap. Semantics of the HF module:
      - weight is initialized to zeros and applied as ``(1 + w)`` (offset=1.0)
      - everything (norm + weight multiply) is computed in fp32, then cast back to the input dtype
        (casting_mode="gemma")
    ``group_size`` selects grouped RMSNorm: the last dim (``hc_count * hidden_size`` for hyper
    connections / PLE norms) is split into ``dim // group_size`` groups that are each normalized
    independently against their own weight slice.
    """

    def __init__(self, dim, group_size=None, eps=1e-6):
        super().__init__(
            dim,
            eps,
            offset=1.0,
            casting_mode="gemma",
            init_fn="zeros",
            in_place=False,
            row_mode=None,
            elementwise_affine=True,
            group_size=group_size,
        )

    @property
    def eps(self):
        return self.variance_epsilon

    @eps.setter
    def eps(self, value):
        self.variance_epsilon = value


def liger_qwen4_exp_rms_norm_forward(self, hidden_states):
    """Use native HF grouped RMSNorm when the selected backend only supports the historical ABI."""
    if self.group_size is not None and not self._liger_rms_norm_supports_grouped:
        return self._liger_qwen4_exp_native_rms_norm_forward(hidden_states)
    return LigerRMSNorm.forward(self, hidden_states)
