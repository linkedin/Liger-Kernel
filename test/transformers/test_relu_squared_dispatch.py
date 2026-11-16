"""Keep private Ascend launcher ABI use confined to the validated stack."""

from importlib.metadata import PackageNotFoundError

import pytest

from liger_kernel.ops import relu_squared


@pytest.mark.parametrize(
    "triton_version,torch_npu_version,expected",
    [
        ("3.2.2", "2.9.0", True),
        ("3.2.2", "2.9.0+local", True),
        ("3.2.2", "2.9.0.post1", False),
        ("3.6.0", "2.10.0.post7", False),
        ("3.2.1", "2.9.0", False),
        (None, "2.9.0", False),
    ],
)
def test_native_stack_dispatch(monkeypatch, triton_version, torch_npu_version, expected):
    def package_version(name):
        value = {"triton-ascend": triton_version, "torch-npu": torch_npu_version}[name]
        if value is None:
            raise PackageNotFoundError(name)
        return value

    supported = relu_squared._ascend_relu_squared_native_supported
    supported.cache_clear()
    try:
        monkeypatch.setattr(relu_squared, "version", package_version)
        assert supported() is expected
    finally:
        supported.cache_clear()
