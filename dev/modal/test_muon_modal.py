from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parent.parent.parent
REMOTE_ROOT_PATH = "/root/liger-kernel"
PYTHON_VERSION = "3.12"

image = modal.Image.debian_slim(python_version=PYTHON_VERSION).pip_install("uv")
app = modal.App("liger_muon_test", image=image)

# mount local repo to remote container
repo = image.add_local_dir(ROOT_PATH, remote_path=REMOTE_ROOT_PATH)


@app.function(gpu="H100!", image=repo, timeout=15 * 60)
def run_muon_test_h100():
    import os
    import subprocess

    print("=== Installing Liger-Kernel with dev and cutedsl dependencies on H100 ===")
    subprocess.run(
        ["uv pip install -e '.[dev,cutedsl]' --system"],
        check=True,
        shell=True,
        cwd=REMOTE_ROOT_PATH,
    )

    print("=== Running Newton-Schulz and LigerMuon tests on H100 ===")
    env = os.environ.copy()
    env["LIGER_KERNEL_IMPL"] = "cutedsl"
    result = subprocess.run(
        ["python -m pytest test/cutedsl/test_newton_schulz.py -v -s"],
        shell=True,
        cwd=REMOTE_ROOT_PATH,
        env=env,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:\n", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Tests failed with exit code {result.returncode}")
    print("=== All tests passed on H100! ===")


@app.local_entrypoint()
def main():
    run_muon_test_h100.remote()
