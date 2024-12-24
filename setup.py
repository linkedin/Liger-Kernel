# setup.py

import subprocess

from typing import Literal

from setuptools import setup


def get_default_dependencies():
    """Determine the appropriate dependencies based on detected hardware."""
    platform = get_platform()

    if platform in ["cuda", "cpu"]:
        return [
            "torch>=2.1.2",
            "triton>=2.3.1",
        ]
    elif platform == "rocm":
        return [
            "torch>=2.6.0.dev",
            "triton>=3.0.0",
        ]


def get_optional_dependencies():
    """Get optional dependency groups."""
    return {
        "dev": [
            "transformers>=4.44.2",
            "matplotlib>=3.7.2",
            "flake8>=4.0.1.1",
            "black>=24.4.2",
            "isort>=5.13.2",
            "pytest>=7.1.2",
            "pytest-xdist",
            "pytest-rerunfailures",
            "datasets>=2.19.2",
            "seaborn",
        ]
    }


# TODO: add intel XPU
def get_platform() -> Literal["cuda", "rocm", "cpu"]:
    """
    Detect whether the system has NVIDIA or AMD GPU without torch dependency.
    """
    # Try nvidia-smi first
    try:
        subprocess.run(["nvidia-smi"], check=True)
        print("NVIDIA GPU detected")
        return "cuda"
    except (subprocess.SubprocessError, FileNotFoundError):
        # If nvidia-smi fails, check for ROCm
        try:
            subprocess.run(["rocm-smi"], check=True)
            print("ROCm GPU detected")
            return "rocm"
        except (subprocess.SubprocessError, FileNotFoundError):
            print("No GPU detected")
            return "cpu"


setup(
    name="liger_kernel",
    package_dir={"": "src"},
    packages=["liger_kernel"],
    install_requires=get_default_dependencies(),
    extras_require=get_optional_dependencies(),
)
