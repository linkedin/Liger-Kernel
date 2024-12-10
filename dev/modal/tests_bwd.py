import os
from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parent.parent.parent
REMOTE_ROOT_PATH = "/root/liger-kernel"

# REBUILD_IMAGE is an environment variable that is set to "true" in the nightly build
REBUILD_IMAGE = os.getenv("REBUILD_IMAGE") is not None

# tests_bwd is to ensure the backward compatibility of liger with older transformers
image = (
    modal.Image.debian_slim()
    .workdir(REMOTE_ROOT_PATH)
    .run_commands(["pip install -e '.[dev]'"], force_build=REBUILD_IMAGE)
    .pip_install("transformers==4.44.2", force_build=REBUILD_IMAGE)
)

app = modal.App("liger_tests", image=image)

# mount: add local files to the remote container
repo = modal.Mount.from_local_dir(ROOT_PATH, remote_path=REMOTE_ROOT_PATH)


@app.function(gpu="A10G", mounts=[repo], timeout=60 * 10)
def liger_tests_bwd():
    import subprocess

    subprocess.run(["make", "test"], check=True, cwd=REMOTE_ROOT_PATH)
    subprocess.run(["make", "test-convergence"], check=True, cwd=REMOTE_ROOT_PATH)
