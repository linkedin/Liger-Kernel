"""Regression test for the chunked fused-linear JSD ignore-label bug.

Before the fix, `_jsd_loss_and_grad_cutedsl` threaded labels on
`n_non_ignore < BT`, comparing the GLOBAL non-ignore count with the per-chunk
BT. Multi-chunk batches with a few ignored tokens (global_n_non_ignore >=
chunk_BT) silently dropped the label stream, so ignored rows leaked loss and
gradient. This test forces >1 chunk and asserts ignored rows contribute zero
input-gradient for every registered JSD impl.
"""

import pytest
import torch

from liger_kernel.ops.fused_linear_jsd import fused_linear_jsd_forward

IGN = -100

_IMPLS = ["nvidia-triton"]
try:  # add cutedsl only when its JSD backend is registered in this build
    import liger_kernel.functional  # noqa: F401  (runs declare_op_locations)

    from liger_kernel.backends.dispatch import resolve_impl

    resolve_impl("jsd_loss_and_grad", requested="nvidia-cutedsl")
    _IMPLS.append("nvidia-cutedsl")
except Exception:  # pragma: no cover  (not registered / runtime missing)
    pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("impl", _IMPLS)
def test_chunked_jsd_ignored_rows_zero_grad(impl):
    torch.manual_seed(0)
    dev = "cuda"
    # Large V forces the chunked forward to use >1 chunk under the memory budget.
    BT, H, V = 2048, 1024, 64000
    student_input = torch.randn(BT, H, device=dev, dtype=torch.float32) * 0.1
    teacher_input = torch.randn(BT, H, device=dev, dtype=torch.float32) * 0.1
    student_weight = torch.randn(V, H, device=dev, dtype=torch.float32) * 0.02
    teacher_weight = torch.randn(V, H, device=dev, dtype=torch.float32) * 0.02

    shift_labels = torch.randint(0, V, (BT,), device=dev)
    ignore_mask = torch.zeros(BT, dtype=torch.bool, device=dev)
    ignore_mask[::7] = True  # scattered ignores across chunks
    shift_labels[ignore_mask] = IGN

    si = student_input.clone().requires_grad_(True)
    sw = student_weight.clone().requires_grad_(True)
    ret = fused_linear_jsd_forward(
        si,
        sw,
        teacher_input,
        teacher_weight,
        shift_labels,
        0.5,
        IGN,
        True,
        1.0,
        jsd_impl=impl,
    )
    grad_input = ret[1]
    max_ignored_grad = float(grad_input[ignore_mask].abs().max())
    assert max_ignored_grad == 0.0, (
        f"[{impl}] ignored rows leaked input-gradient (max={max_ignored_grad:.3e}); "
        f"the chunked label stream was dropped"
    )
