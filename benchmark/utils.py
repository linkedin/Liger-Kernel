import os
import time
from typing import Callable

import torch


def _test_memory(func: Callable, _iter: int = 10) -> float:
    total_mem = []

    for _ in range(_iter):
        torch.cuda.memory.reset_peak_memory_stats()
        func()
        mem = torch.cuda.max_memory_allocated()
        total_mem.append(mem)

    return sum(total_mem) / len(total_mem)


def get_current_file_directory() -> str:
    """
    Returns the directory path of the current Python file.
    """
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the directory path of the current file
    return os.path.dirname(current_file_path)


def get_gpu_dir_name():
    """
    Returns the current GPU name, formatted to serve as a directory name
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        return gpu_name.lower().replace(" ", "_")
    else:
        raise Exception("Benchmarks can only be run on GPU.")


def create_output_dir(benchmark_type: str) -> str:
    """
    Create and return the output directory path for the specified benchmark. For example:
    "<current_dir>/cross_entropy_speed/nvidia_h100_80gb_hbm3"
    """
    curr_dir = get_current_file_directory()
    gpu_dir_name = get_gpu_dir_name()
    output_dir = os.path.join(curr_dir, benchmark_type, gpu_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def sleep(seconds):
    def decorator(function):
        def wrapper(*args, **kwargs):
            time.sleep(seconds)
            return function(*args, **kwargs)

        return wrapper

    return decorator


def _print_memory_banner():
    print("**************************************")
    print("*     BENCHMARKING GPU MEMORY        *")
    print("**************************************")


def _print_speed_banner():
    print("**************************************")
    print("*        BENCHMARKING SPEED          *")
    print("**************************************")
