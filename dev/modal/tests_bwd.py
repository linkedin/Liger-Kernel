import os
from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parent.parent.parent

# REBUILD_IMAGE is an environment variable that is set to "true" in the nightly build
REBUILD_IMAGE = os.getenv("REBUILD_IMAGE") is not None

# tests_bwd is to ensure the backward compatibility of liger with older transformers
image = (
    modal.Image.debian_slim()
    .pip_install_from_pyproject(
        ROOT_PATH / "pyproject.toml",
        optional_dependencies=["dev"],
        force_build=REBUILD_IMAGE,
    )
    .pip_install("transformers==4.44.2", force_build=REBUILD_IMAGE)
)

app = modal.App("liger_tests", image=image)

# mount: add local files to the remote container
repo = modal.Mount.from_local_dir(ROOT_PATH, remote_path="/root/liger-kernel")


@app.function(gpu="A10G", mounts=[repo], timeout=60 * 10)
def liger_tests_bwd():
    import subprocess

    subprocess.run(["pip", "install", "-e", "."], check=True, cwd="/root/liger-kernel")
    subprocess.run(["make", "test"], check=True, cwd="/root/liger-kernel")
    subprocess.run(["make", "test-convergence"], check=True, cwd="/root/liger-kernel")
