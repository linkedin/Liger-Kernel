import os
import subprocess
import sys
import textwrap

from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile backend requires CUDA")
@pytest.mark.skipif(
    os.environ.get("LIGER_KERNEL_IMPL", "").strip().lower() != "cutile",
    reason="cuTile backend selection test requires LIGER_KERNEL_IMPL=cutile",
)
def test_liger_kernel_impl_cutile_selects_cutile_jsd_function():
    repo_root = Path(__file__).resolve().parents[2]
    pythonpath = os.pathsep.join(
        [
            str(repo_root / "src"),
            str(repo_root),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    env = {
        **os.environ,
        "LIGER_KERNEL_IMPL": "cutile",
        "PYTHONPATH": pythonpath,
    }
    script = textwrap.dedent(
        """
        from liger_kernel.transformers.jsd import LigerJSDFunction

        module_name = LigerJSDFunction.__module__
        expected_prefix = "liger_kernel.ops.cutile."
        if not module_name.startswith(expected_prefix):
            raise AssertionError(
                f"Expected cuTile LigerJSDFunction from {expected_prefix}, got {module_name}"
            )
        """
    )

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=env,
        cwd=repo_root,
    )
