import torch
import torch.nn as nn

from liger_kernel.functional import layer_norm as _functional_layer_norm
from liger_kernel.ops import LigerLayerNormFunction  # noqa: F401  (kept for back-compat imports)


class LigerLayerNorm(nn.Module):
    """LayerNorm layer with pluggable kernel implementation.

    The ``impl`` and ``mode`` kwargs select which kernel runs. By default the
    best available implementation for the current device is chosen; see
    ``liger_kernel.backends`` for details. ``backend=`` is accepted as a legacy
    back-compat alias for ``impl=``.
    """

    def __init__(
        self,
        hidden_size,
        eps=1e-6,
        bias=False,
        init_fn="ones",
        impl=None,
        mode=None,
        backend=None,  # legacy back-compat alias for impl=
    ):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size))
        self.bias = nn.Parameter(torch.randn(hidden_size) if bias else torch.zeros(hidden_size))
        self.variance_epsilon = eps
        if impl is None and backend is not None:
            impl = backend
        self.impl = impl
        self.mode = mode

    @property
    def backend(self):
        """Legacy back-compat alias for :attr:`impl`."""
        return self.impl

    @backend.setter
    def backend(self, value):
        self.impl = value

    def forward(self, hidden_states):
        return _functional_layer_norm(
            hidden_states,
            self.weight,
            self.bias,
            self.variance_epsilon,
            impl=getattr(self, "impl", None),
            mode=getattr(self, "mode", None),
        )

    def extra_repr(self):
        return (
            f"{self.hidden_size}, eps={self.eps}, "
            f"impl={getattr(self, 'impl', None)!r}, "
            f"mode={getattr(self, 'mode', None)!r}"
        )
