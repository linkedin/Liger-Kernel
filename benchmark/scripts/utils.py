import argparse
import csv
import json
import os
import time

from collections import OrderedDict
from dataclasses import asdict
from dataclasses import dataclass
from importlib.metadata import version
from itertools import zip_longest
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch

from liger_kernel.utils import infer_device

device = infer_device()

LIGER_KERNEL_VERSION = version("liger-kernel")

QUANTILES = [0.5, 0.2, 0.8]


@dataclass
class SingleBenchmarkRunInput:
    x: Union[int, float]
    kernel_provider: str
    kernel_operation_mode: Optional[str] = ""
    extra_benchmark_config: Optional[Dict[str, Any]] = None


@dataclass
class SingleBenchmarkRunOutput:
    # 20th percentile
    y_20: float
    # 50th percentile (median)
    y_50: float
    # 80th percentile
    y_80: float


@dataclass
class BenchmarkData:
    """
    BenchmarkData is a dataclass to store the benchmark data for a a completed benchmark
    run on all x-values for a given kernel/kernel operation mode/metric/extra_benchmark_config
    """

    kernel_name: str
    kernel_provider: str
    metric_name: str
    metric_unit: str
    gpu_name: str
    x_name: str
    x_label: str
    x_values: List[float]
    y_values_50: List[float]
    y_values_20: List[float]
    y_values_80: List[float]
    timestamp: str
    kernel_operation_mode: Optional[str] = None
    extra_benchmark_config_str: Optional[str] = None
    liger_version: str = LIGER_KERNEL_VERSION


@dataclass
class BenchmarkDataCSVRow:
    # The ordering of field names here will be the order of columns in the CSV
    kernel_name: str
    kernel_provider: str
    kernel_operation_mode: Union[str, None]
    metric_name: str
    metric_unit: str
    x_name: str
    x_label: str
    x_value: float
    y_value_50: float
    y_value_20: float
    y_value_80: float
    extra_benchmark_config_str: Union[str, None]
    gpu_name: str
    timestamp: str
    liger_version: str


def _test_memory(
    func: Callable,
    _iter: int = 10,
    quantiles: Optional[List[float]] = None,
    return_mode="mean",
) -> float:
    assert return_mode in ["min", "max", "mean", "median"]
    total_mem = []

    for _ in range(_iter):
        getattr(torch, device).memory.reset_peak_memory_stats()
        func()
        # Convert to MB
        mem = getattr(torch, device).max_memory_allocated() / 2**20
        total_mem.append(mem)

    total_mem = torch.tensor(total_mem, dtype=torch.float)
    if quantiles is not None:
        quantiles_data = torch.quantile(total_mem, torch.tensor(quantiles, dtype=torch.float)).tolist()
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


def _print_benchmarking_banner(metric_name: str, kernel_name: str):
    print("**************************************")
    print(f"     BENCHMARKING {metric_name.upper()} for {kernel_name.upper()}")
    print("**************************************")


def get_formatted_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def get_gpu_name():
    """
    Returns the current GPU name, formatted to serve as a directory name
    """
    torch_device = getattr(torch, device)
    if torch_device.is_available():
        gpu_name = torch_device.get_device_name(torch_device.current_device())
        return gpu_name
    else:
        raise Exception("Benchmarks can only be run on GPU.")


def update_benchmark_data_csv(
    benchmark_data_list: List[BenchmarkData],
    filename: str = "all_benchmark_data.csv",
    overwrite: bool = True,
):
    """
    Update the CSV file with the new benchmark data. If the file does not exist, create it.
    If an entry already exists for the benchmark, then overwrite it if `overwrite` is True.
    """

    def create_unique_key(row):
        # This unique key is used to determine if a benchmark run already exists in the CSV
        # If the key is the same, then the benchmark run already exists and will optionally
        # be overwritten. Otherwise, it is considered a new benchmark run and appended.
        return (
            row["kernel_name"],
            row["kernel_provider"],
            row["kernel_operation_mode"] if row["kernel_operation_mode"] else "",
            row["metric_name"],
            row["x_name"],
            str(row["x_value"]),
            (row["extra_benchmark_config_str"] if row["extra_benchmark_config_str"] else ""),
            row["gpu_name"],
        )

    fieldnames = BenchmarkDataCSVRow.__annotations__.keys()

    # Make filename path relative to current file
    filename_abs_path = os.path.join(get_current_file_directory(), "../data", filename)
    file_exists = os.path.isfile(filename_abs_path)

    # Read existing data into a list of dicts
    existing_data = []
    if file_exists:
        with open(filename_abs_path, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                existing_data.append(row)

    existing_data_dict = OrderedDict((create_unique_key(row), row) for row in existing_data)

    for benchmark_data in benchmark_data_list:
        benchmark_data_dict = asdict(benchmark_data)
        x_values = benchmark_data_dict.pop("x_values")
        y_values_50 = benchmark_data_dict.pop("y_values_50")
        y_values_20 = benchmark_data_dict.pop("y_values_20")
        y_values_80 = benchmark_data_dict.pop("y_values_80")

        # Need to convert benchmark_data into multiple rows based on x_values and y_values
        for x_value, y_value_50, y_value_20, y_value_80 in zip_longest(x_values, y_values_50, y_values_20, y_values_80):
            row = BenchmarkDataCSVRow(
                x_value=x_value,
                y_value_50=y_value_50,
                y_value_20=y_value_20,
                y_value_80=y_value_80,
                **benchmark_data_dict,
            )
            row_dict = asdict(row)

            row_key = create_unique_key(row_dict)

            if row_key in existing_data_dict:
                if overwrite:
                    # If overwriting, update the row
                    existing_data_dict[row_key] = row_dict
                else:
                    # If not overwriting, skip this row
                    pass
            else:
                existing_data_dict[row_key] = row_dict

    with open(filename_abs_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in existing_data_dict.values():
            writer.writerow(row)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.dtype):
            return str(obj)
        return super().default(self, obj)


def print_benchmark_data(benchmark_data_list: List[BenchmarkData]) -> str:
    print("********** Benchmark Data **********")
    formatted_list = [obj.__dict__ for obj in benchmark_data_list]
    print(json.dumps(formatted_list, indent=2))


def run_benchmarks(
    bench_test_fn: Callable,
    kernel_name: str,
    metric_name: str,
    metric_unit: str,
    x_name: str,
    x_label: str,
    x_values: List[Union[float, int]],
    kernel_providers: List[str],
    kernel_operation_modes: Optional[List[str]] = [None],
    extra_benchmark_configs: Optional[List[Dict[str, Any]]] = None,
    overwrite: bool = False,
):
    """
    Run benchmarks given a bench_test_fn that takes in a SingleBenchmarkRunInput as input and
    saves data to the CSV file.

    Args:
        - bench_test_fn: The benchmark test function to run. This function should take in a
            SingleBenchmarkRunInput as input and return a SingleBenchmarkRunOutput.
        - kernel_name: The name of the kernel being benchmarked (e.g. "swiglu")
        - metric_name: The name of the metric being benchmarked (e.g. "speed" or "memory")
        - metric_unit: The unit of the metric being benchmarked (e.g. "ms" or "MB")
        - x_name: The name of the x-axis (e.g. "T" for sequence length)
        - x_label: The label of the x-axis (e.g. "sequence length")
        - x_values: The list of x-values to run the benchmark on (e.g. [2**i for i in range(10, 14)])
        - kernel_providers: The list of kernel providers to run the benchmark on (e.g. ["liger", "huggingface"])
        - kernel_operation_modes: The list of kernel operation modes to run the benchmark on (e.g. ["full", "backward"])
        - extra_benchmark_configs: The list of extra benchmark configurations to run the benchmark on.
        - overwrite: Whether to overwrite the existing benchmark data entry if it already exists.
    """

    assert len(kernel_operation_modes) >= 1
    assert len(kernel_providers) >= 1

    _print_benchmarking_banner(metric_name=metric_name, kernel_name=kernel_name)

    gpu_name = get_gpu_name()
    benchmark_data_list = []
    for extra_benchmark_config in extra_benchmark_configs:
        for kernel_operation_mode in kernel_operation_modes:
            for kernel_provider in kernel_providers:
                y_values_50 = []
                y_values_20 = []
                y_values_80 = []

                for x in x_values:
                    single_benchmark_run_input = SingleBenchmarkRunInput(
                        x=x,
                        kernel_provider=kernel_provider,
                        kernel_operation_mode=kernel_operation_mode,
                        extra_benchmark_config=extra_benchmark_config,
                    )
                    benchmark_result: SingleBenchmarkRunOutput = bench_test_fn(single_benchmark_run_input)
                    y_values_50.append(benchmark_result.y_50)
                    y_values_20.append(benchmark_result.y_20)
                    y_values_80.append(benchmark_result.y_80)

                benchmark_run_data = BenchmarkData(
                    kernel_name=kernel_name,
                    kernel_operation_mode=kernel_operation_mode,
                    kernel_provider=kernel_provider,
                    metric_name=metric_name,
                    metric_unit=metric_unit,
                    gpu_name=gpu_name,
                    x_name=x_name,
                    x_label=x_label,
                    x_values=x_values,
                    y_values_50=y_values_50,
                    y_values_20=y_values_20,
                    y_values_80=y_values_80,
                    extra_benchmark_config_str=json.dumps(extra_benchmark_config, cls=CustomEncoder),
                    timestamp=get_formatted_time(),
                    liger_version=LIGER_KERNEL_VERSION,
                )

                benchmark_data_list.append(benchmark_run_data)

    print_benchmark_data(benchmark_data_list)

    update_benchmark_data_csv(benchmark_data_list=benchmark_data_list, overwrite=overwrite)


def parse_benchmark_script_args():
    parser = argparse.ArgumentParser(description="Benchmarking script for Liger-Kernel")

    # Add an optional --overwrite flag
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Flag to overwrite existing benchmark data with current run.",
    )

    args = parser.parse_args()
    return args
