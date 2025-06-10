from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parent.parent.parent
REMOTE_ROOT_PATH = "/root/liger-kernel"
PYTHON_VERSION = "3.12"

image = modal.Image.debian_slim(python_version=PYTHON_VERSION).pip_install("uv")

app = modal.App("liger_benchmarks", image=image)

# mount: add local files to the remote container
repo = image.add_local_dir(ROOT_PATH, remote_path=REMOTE_ROOT_PATH)


@app.function(gpu="A10G", image=repo, timeout=60 * 45)
def liger_benchmarks():
    import subprocess

    subprocess.run(
        ["uv pip install -e '.[dev]' --system"],
        check=True,
        shell=True,
        cwd=REMOTE_ROOT_PATH,
    )
    subprocess.run(["python benchmark/scripts/benchmark_kto_loss.py"], check=True, shell=True, cwd=REMOTE_ROOT_PATH)
