from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parent.parent.parent

image = modal.Image.debian_slim().pip_install_from_pyproject(
    ROOT_PATH / "pyproject.toml", optional_dependencies=["dev"]
)

app = modal.App("liger_tests", image=image)

# mount: add local files to the remote container
repo = modal.Mount.from_local_dir(ROOT_PATH, remote_path="/root/liger-kernel")


@app.function(gpu="A10G", mounts=[repo], timeout=60 * 10)
def liger_tests():
    import subprocess

    subprocess.run(["pip", "install", "-e", "."], check=True, cwd="/root/liger-kernel")
    subprocess.run(["make", "test"], check=True, cwd="/root/liger-kernel")
    subprocess.run(["make", "test-convergence"], check=True, cwd="/root/liger-kernel")
