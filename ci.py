from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parent # .parent

image = (
    modal.Image.debian_slim()
)

app = modal.App("ci-testing", image=image)

# mount: add local files to the remote container
repo = modal.Mount.from_local_dir(ROOT_PATH, remote_path="/root/liger-kernel")


@app.function(gpu="A10G", mounts=[repo])
def pytest():
    import subprocess
    subprocess.run(["pip", "install", "-e", ".[dev]"], check=True, cwd="/root/liger-kernel")
    subprocess.run(["make", "test"], check=True, cwd="/root/liger-kernel")