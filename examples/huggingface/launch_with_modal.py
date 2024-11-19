"""
launch_with_modal.py

This tool is designed to launch scripts using Modal.

It sets up the necessary environment, including GPU resources and python dependencies,
and executes the specified training script remotely.

### Setup and Usage
```bash
pip install modal
modal setup  # authenticate with Modal
export HF_TOKEN="your_huggingface_token"  # if using a gated model such as llama3
modal run launch_with_modal.py --script "run_qwen2_vl.sh"
```

### Caveats
This tool is intended as an easy on-ramp to using Liger-Kernel for fine-tuning LLMs and
VLMs - it is a reproducible way to run benchmarks and example scripts. However, it is not
the best way to develop a model on Modal, as it re-downloads the model and dataset each
time it is run. For iterative development, consider using `modal.Volume` to cache the
model and dataset between runs.
"""

import modal
from modal import gpu
from modal.exception import InvalidError

TWO_HOURS = 2 * 60 * 60
SIXTEEN_GB = 16 * 1024

app = modal.App("liger-example")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .copy_local_dir(".", "/root")
)

try:
    hf_token_secret = modal.Secret.from_local_environ(["HF_TOKEN"])
    secrets = (hf_token_secret,)
except InvalidError:
    print("HF_TOKEN not found in local environment. Skipping secret creation.")
    secrets = ()


@app.function(
    gpu=gpu.A100(count=4, size="80GB"),
    image=image,
    timeout=TWO_HOURS,
    memory=SIXTEEN_GB,
    # secrets=secrets,
)
def launch_script(script: str):
    import os
    import subprocess

    script_path = f"/root/{script}"
    os.chmod(script_path, 0o755)  # make script executable

    print(f"Running script: {script_path}")
    subprocess.run([script_path], check=True, cwd="/root", env=os.environ.copy())


@app.local_entrypoint()
def main(script: str):
    """
    Launch a script remotely on modal.
    ```bash
    export HF_TOKEN="your_huggingface_token"  # if using a gated model such as llama3
    modal run --detach launch_with_modal.py --script "run_qwen2_vl.sh"
    ```
    """
    launch_script.remote(script=script)
