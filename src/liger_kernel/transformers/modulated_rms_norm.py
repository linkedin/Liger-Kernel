import torch
import torch.nn as nn

from liger_kernel.ops import LigerModulatedRMSNormFunction


class LigerModulatedRMSNorm(nn.Module):
    """
    Fused modulated RMSNorm module: ``y = (1 + scale) * RMSNorm(x) + shift``.

    Semantics of ``eps``, ``offset``, ``casting_mode`` and ``in_place`` mirror
    :class:`liger_kernel.transformers.LigerRMSNorm`. ``scale`` and the optional
    ``shift`` follow the broadcast rules documented on
    :class:`liger_kernel.ops.LigerModulatedRMSNormFunction`.

    Args:
        hidden_size: trailing dimension over which RMS is computed.
        eps: epsilon added to mean-square inside ``rsqrt``.
        offset: constant added to the weight, e.g. ``1.0`` for Gemma-style norms.
        casting_mode: ``"llama"`` / ``"gemma"`` / ``"none"`` — see ``LigerRMSNormFunction``.
        init_fn: ``"ones"`` or ``"zeros"`` for the (optional) weight initialization.
        in_place: whether the backward kernel writes ``dX`` over ``dY`` to save memory.
        elementwise_affine: if False, no learnable weight is allocated.
    """

    def __init__(
        self,
        hidden_size,
        eps=1e-6,
        offset=0.0,
        casting_mode="llama",
        init_fn="ones",
        in_place=True,
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
        self.variance_epsilon = eps
        self.offset = offset
        self.casting_mode = casting_mode
        self.in_place = in_place

    def forward(self, hidden_states, scale, shift=None):
        return LigerModulatedRMSNormFunction.apply(
            hidden_states,
            self.weight,
            scale,
            shift,
            self.variance_epsilon,
            self.offset,
            self.casting_mode,
            self.in_place,
        )

    def extra_repr(self):
        weight_shape = tuple(self.weight.shape) if self.weight is not None else None
        return (
            f"weight_shape={weight_shape}, eps={self.variance_epsilon}, offset={self.offset}, "
            f"casting_mode={self.casting_mode}, in_place={self.in_place}"
        )
