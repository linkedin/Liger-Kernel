# setup.py

import subprocess
from typing import Literal


def get_default_dependencies():
    """Determine the appropriate dependencies based on detected hardware."""
    platform = _get_platform()

    if platform in ["cuda", "cpu"]:
        return [
            "torch>=2.1.2",
            "triton>=2.3.1",
        ]
    elif platform == "rocm":
        return [
            "torch>=2.6.0.dev",
            "setuptools-scm>=8",
            "torchvision>=0.20.0.dev",
            "triton>=3.0.0",
        ]


def get_optional_dependencies():
    """Get optional dependency groups."""
    platform = _get_platform()

    if platform in ["cuda", "cpu"]:
        return {
            "dev": [
                "transformers>=4.44.2",
                "trl>=0.11.0",
                "matplotlib>=3.7.2",
                "flake8>=4.0.1.1",
                "black>=24.4.2",
                "isort>=5.13.2",
                "pytest>=7.1.2",
                "pytest-xdist",
                "pytest-rerunfailures",
                "datasets>=2.19.2",
                "torchvision>=0.16.2",
                "seaborn",
            ],
            "transformers": ["transformers>=4.44.2"],
        }
    elif platform == "rocm":
        return {
            "dev": [
                "setuptools-scm>=8",
                "torchvision>=0.20.0.dev",
            ],
            "transformers": ["transformers>=4.44.2"],
        }


def _get_platform() -> Literal["cuda", "rocm", "cpu"]:
    """
    Detect whether the system has NVIDIA or AMD GPU without torch dependency.
    """
    # Try nvidia-smi first
    try:
        subprocess.run(["nvidia-smi"], check=True, capture_output=True)
        return "cuda"
    except (subprocess.SubprocessError, FileNotFoundError):
        # If nvidia-smi fails, check for ROCm
        try:
            subprocess.run(["rocm-smi"], check=True, capture_output=True)
            return "rocm"
        except (subprocess.SubprocessError, FileNotFoundError):
            return "cpu"
