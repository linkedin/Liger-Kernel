import os
import time
from typing import Callable, List, Optional

import torch

QUANTILES = [0.5, 0.2, 0.8]


def _test_memory(
    func: Callable,
    _iter: int = 10,
    quantiles: Optional[List[float]] = None,
    return_mode="mean",
) -> float:
    assert return_mode in ["min", "max", "mean", "median"]
    total_mem = []

    for _ in range(_iter):
        torch.cuda.memory.reset_peak_memory_stats()
        func()
        mem = torch.cuda.max_memory_allocated()
        total_mem.append(mem)

    total_mem = torch.tensor(total_mem, dtype=torch.float)
    if quantiles is not None:
        quantiles_data = torch.quantile(
            total_mem, torch.tensor(quantiles, dtype=torch.float)
        ).tolist()
        if len(quantiles_data) == 1:
            quantiles_data = quantiles_data[0]
        return quantiles_data
    return getattr(torch, return_mode)(total_mem).item()


def get_current_file_directory() -> str:
    """
    Returns the directory path of the current Python file.
    """
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the directory path of the current file
    return os.path.dirname(current_file_path)


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
