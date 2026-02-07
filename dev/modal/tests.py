from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parent.parent.parent
REMOTE_ROOT_PATH = "/root/liger-kernel"
PYTHON_VERSION = "3.12"

OLDEST_SUPPORTED_TRANSFORMERS_V4_VERSION = "4.49.0"

image = modal.Image.debian_slim(python_version=PYTHON_VERSION).pip_install("uv")

app = modal.App("liger_tests", image=image)

# mount: add local files to the remote container
repo = image.add_local_dir(ROOT_PATH, remote_path=REMOTE_ROOT_PATH)


@app.function(gpu="H100!", image=repo, timeout=90 * 60)
def liger_correctness_tests():
    import subprocess

    subprocess.run(
        ["uv pip install -e '.[dev]' --system"],
        check=True,
        shell=True,
        cwd=REMOTE_ROOT_PATH,
    )
    subprocess.run(["make test-convergence"], check=True, shell=True, cwd=REMOTE_ROOT_PATH)


@app.function(gpu="H100!", image=repo, timeout=90 * 60)
def liger_convergence_tests():
    import subprocess

    subprocess.run(
        ["uv pip install -e '.[dev]' --system"],
        check=True,
        shell=True,
        cwd=REMOTE_ROOT_PATH,
    )
    subprocess.run(["make test-convergence"], check=True, shell=True, cwd=REMOTE_ROOT_PATH)


oldest_v4_app = modal.App("liger_oldest_v4_tests", image=image)  # 4.49.0


@oldest_v4_app.function(gpu="H100!", image=repo, timeout=90 * 60)
def liger_oldest_v4_correctness_tests():
    import subprocess

    subprocess.run(
        ["uv pip install -e '.[dev]' --system"],
        check=True,
        shell=True,
        cwd=REMOTE_ROOT_PATH,
    )
    subprocess.run(
        [f"uv pip install 'transformers=={OLDEST_SUPPORTED_TRANSFORMERS_V4_VERSION}' --system"],
        check=True,
        shell=True,
        cwd=REMOTE_ROOT_PATH,
    )
    subprocess.run(["make test"], check=True, shell=True, cwd=REMOTE_ROOT_PATH)


@oldest_v4_app.function(gpu="H100!", image=repo, timeout=90 * 60)
def liger_oldest_v4_convergence_tests():
    import subprocess

    subprocess.run(
        ["uv pip install -e '.[dev]' --system"],
        check=True,
        shell=True,
        cwd=REMOTE_ROOT_PATH,
    )
    subprocess.run(
        [f"uv pip install 'transformers=={OLDEST_SUPPORTED_TRANSFORMERS_V4_VERSION}' --system"],
        check=True,
        shell=True,
        cwd=REMOTE_ROOT_PATH,
    )
    subprocess.run(["make test-convergence"], check=True, shell=True, cwd=REMOTE_ROOT_PATH)


latest_v4_app = modal.App("liger_latest_v4_tests", image=image)


@latest_v4_app.function(gpu="H100!", image=repo, timeout=90 * 60)
def liger_latest_v4_correctness_tests():
    import subprocess

    subprocess.run(
        ["uv pip install -e '.[dev]' --system"],
        check=True,
        shell=True,
        cwd=REMOTE_ROOT_PATH,
    )
    subprocess.run(
        [f"uv pip install 'transformers>={OLDEST_SUPPORTED_TRANSFORMERS_V4_VERSION}, <5.0.0' --system"],
        check=True,
        shell=True,
        cwd=REMOTE_ROOT_PATH,
    )
    subprocess.run(["make test"], check=True, shell=True, cwd=REMOTE_ROOT_PATH)


@latest_v4_app.function(gpu="H100!", image=repo, timeout=90 * 60)
def liger_latest_v4_convergence_tests():
    import subprocess

    subprocess.run(
        ["uv pip install -e '.[dev]' --system"],
        check=True,
        shell=True,
        cwd=REMOTE_ROOT_PATH,
    )
    subprocess.run(
        [f"uv pip install 'transformers>={OLDEST_SUPPORTED_TRANSFORMERS_V4_VERSION}, <5.0.0' --system"],
        check=True,
        shell=True,
        cwd=REMOTE_ROOT_PATH,
    )
    subprocess.run(["make test-convergence"], check=True, shell=True, cwd=REMOTE_ROOT_PATH)
