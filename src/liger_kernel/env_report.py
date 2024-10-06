import platform
import sys

def print_env_report():
    """
    Prints a report of the environment. Useful for debugging and reproducibility.
    Usage:
    ```
    python -m liger_kernel.env_report
    ```
    """
    print("Environment Report:")
    print("-------------------")
    print(f"Operating System: {platform.platform()}")
    print(f"Python version: {sys.version.split()[0]}")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        cuda_version = (
            torch.version.cuda if torch.cuda.is_available() else "Not available"
        )
        print(f"CUDA version: {cuda_version}")
    except ImportError:
        print("PyTorch: Not installed")
        print("CUDA version: Unable to query")

    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers: Not installed")

try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        print("TensorFlow: Not installed")
        
    try:
        from pkg_resources import working_set
        installed_packages = sorted(["{}=={}".format(i.key, i.version) for i in working_set])
        print("Installed Packages:")
        for pkg in installed_packages:
            print(pkg)
    except ImportError:
        print("Unable to list installed packages")

if __name__ == "__main__":
    print_env_report()
