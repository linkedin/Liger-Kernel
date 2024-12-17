from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parent.parent.parent
REMOTE_ROOT_PATH = "/root/liger-kernel"
PYTHON_VERSION = "3.12"

image = modal.Image.debian_slim(python_version=PYTHON_VERSION).pip_install("uv")

app = modal.App("liger_tests_bwd", image=image)

# mount: add local files to the remote container
repo = modal.Mount.from_local_dir(ROOT_PATH, remote_path=REMOTE_ROOT_PATH)


@app.function(gpu="A10G", mounts=[repo], timeout=60 * 15)
def liger_bwd_tests():
    import subprocess

    subprocess.run(
        ["uv pip install -e '.[dev]' --system"],
        check=True,
        shell=True,
        cwd=REMOTE_ROOT_PATH,
    )
    # force install transformers==4.44.2
    subprocess.run(
        ["uv pip install transformers==4.44.2 --system"],
        check=True,
        shell=True,
        cwd=REMOTE_ROOT_PATH,
    )
    subprocess.run(["make test"], check=True, shell=True, cwd=REMOTE_ROOT_PATH)
    subprocess.run(["make test-convergence"], check=True, shell=True, cwd=REMOTE_ROOT_PATH)
