from importlib import import_module
from importlib.util import find_spec

import pytest


@pytest.mark.parametrize(
    "module_name, kernel_name, parameter_name",
    [
        ("layer_norm", "_layer_norm_forward_row", "n_rows"),
        ("layer_norm", "_layer_norm_forward_fused_2d", "n_rows"),
        ("layer_norm", "_layer_norm_forward_tiled", "n_rows"),
        ("layer_norm", "_layer_norm_backward_rows", "n_rows"),
        ("layer_norm", "_layer_norm_backward_tiled", "n_rows"),
        ("rms_norm", "_rms_norm_forward_row", "n_rows"),
        ("rms_norm", "_rms_norm_forward_fused_2d", "n_rows"),
        ("rms_norm", "_rms_norm_forward_tiled", "n_rows"),
        ("rms_norm", "_rms_norm_backward_row", "n_rows"),
        ("rms_norm", "_rms_norm_backward_fused_2d", "n_rows"),
        ("rms_norm", "_rms_norm_backward_tiled", "n_rows"),
        ("softmax", "_softmax_single_block_forward_kernel", "n_rows"),
        ("softmax", "_softmax_multi_block_forward_kernel", "n_rows"),
        ("softmax", "_softmax_single_block_backward_kernel", "n_rows"),
        ("softmax", "_softmax_multi_block_backward_kernel", "n_rows"),
        ("sparsemax", "_sparsemax_forward_kernel", "n_rows"),
        ("sparsemax", "_sparsemax_forward_tiled_kernel", "n_rows"),
        ("sparsemax", "_sparsemax_backward_kernel", "n_rows"),
        ("sparsemax", "_sparsemax_backward_tiled_kernel", "n_rows"),
        ("poly_norm", "_poly_norm_backward_kernel_npu", "n_rows"),
        ("poly_norm", "_poly_norm_forward_kernel_npu", "n_rows"),
        ("poly_norm", "_poly_norm_forward_kernel_no_tiling", "n_rows"),
        ("poly_norm", "_poly_norm_backward_kernel_no_tiling", "n_rows"),
        ("dyt", "_dyt_fwd_kernel", "M"),
        ("dyt", "_dyt_bwd_kernel", "M"),
        ("jsd", "_jsd_kernel", "n_rows"),
        ("fused_linear_jsd", "_element_mul_kernel", "n_rows"),
        ("tvd", "_tv_distance_kernel", "total_rows"),
    ],
)
def test_ascend_runtime_rows(module_name, kernel_name, parameter_name):
    if find_spec("triton.language.extra.cann") is None:
        pytest.skip("requires Triton-Ascend CANN extensions")
    module = import_module(f"liger_kernel.ops.backends._ascend.ops.{module_name}")
    kernel = getattr(module, kernel_name)
    assert kernel.fn.__module__ == module.__name__
    assert kernel.fn.__name__ == kernel_name
    parameter = next(param for param in kernel.params if param.name == parameter_name)
    assert not parameter.is_constexpr
