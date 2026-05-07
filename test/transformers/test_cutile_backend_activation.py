import os
import subprocess
import sys

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_python(script: str):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in [str(_REPO_ROOT / "src"), str(_REPO_ROOT), env.get("PYTHONPATH", "")] if path
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_cutile_adapter_import_is_optional_without_activation():
    result = _run_python(
        """
import os
os.environ.pop("CUTILE_BACKEND", None)

from liger_kernel.ops.backends._cutile import ops as cutile_ops
import liger_kernel.ops as ops

assert hasattr(cutile_ops, "TILEGYM_AVAILABLE")
assert ops.LigerJSDFunction.__module__ == "liger_kernel.ops.jsd"
assert ops.LigerFusedLinearJSDFunction.__module__ == "liger_kernel.ops.fused_linear_jsd"
"""
    )

    assert result.returncode == 0, result.stderr


def test_cutile_backend_activation_is_explicit_on_cuda():
    result = _run_python(
        """
import os
import torch

os.environ["CUTILE_BACKEND"] = "cutile"
torch.cuda.is_available = lambda: True

try:
    import liger_kernel.ops as ops
except ImportError as exc:
    assert "tilegym cutile backend is not available" in str(exc)
else:
    assert ops.LigerJSDFunction.__module__ == "tilegym.suites.liger.cutile.jsd"
    assert ops.LigerFusedLinearJSDFunction.__module__ == "tilegym.suites.liger.cutile.fused_linear_jsd"
"""
    )

    assert result.returncode == 0, result.stderr
