"""Re-usable test utilities for Liger Kernel's multi-DSL backends.

These helpers are shipped inside the package (rather than under ``test/``) so
that downstream users and external backend authors can validate their own
``register_op``-registered impls with the same correctness contract Liger uses
in CI.

Public surface
--------------
- :func:`assert_op_correctness` — fwd/bwd vs reference correctness driver.
- :func:`iter_test_shapes` — canonical shape generator.
- :func:`pytorch_reference_layer_norm` — ground-truth LayerNorm.
- :func:`pytorch_reference_rms_norm` — ground-truth RMSNorm.
- :func:`requires_cuda` — capability-aware ``pytest.mark.skipif`` wrapper.
- :func:`sum_based_dw_close` — fallback dW comparator used when element-wise
  comparison fails but the *summed* gradient still matches within tolerance.
"""

from liger_kernel.testing._op_helpers import assert_op_correctness
from liger_kernel.testing._op_helpers import iter_test_shapes
from liger_kernel.testing._op_helpers import pytorch_reference_layer_norm
from liger_kernel.testing._op_helpers import pytorch_reference_rms_norm
from liger_kernel.testing._op_helpers import requires_cuda
from liger_kernel.testing._op_helpers import sum_based_dw_close

__all__ = [
    "assert_op_correctness",
    "iter_test_shapes",
    "pytorch_reference_layer_norm",
    "pytorch_reference_rms_norm",
    "requires_cuda",
    "sum_based_dw_close",
]
