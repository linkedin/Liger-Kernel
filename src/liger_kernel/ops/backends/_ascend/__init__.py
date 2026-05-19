from liger_kernel.ops.backends.registry import BackendInfo
from liger_kernel.ops.backends.registry import register_backend

# Ascend NPU backend — default on NPU devices.
register_backend(BackendInfo(name="ascend", devices=("npu",), default_devices=("npu",)))
