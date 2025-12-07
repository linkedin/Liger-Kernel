import json
import os
import sys

from argparse import ArgumentParser
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/all_benchmark_data.csv"))
VISUALIZATIONS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "visualizations/"))


@dataclass
class VisualizationsConfig:
    """
    Configuration for the visualizations script.

    Args:
        kernel_name (str): Kernel name to benchmark. (Will run `scripts/benchmark_{kernel_name}.py`)
        metric_name (str): Metric name to visualize (speed/memory)
        kernel_operation_mode (str): Kernel operation mode to visualize (forward/backward/full). Defaults to "full"
        extra_config_filter (str, optional): A string to filter extra_benchmark_config.
                                            Can be a substring to match or a 'key=value' pair (e.g., "'H': 4096").
                                            Defaults to None, which means the first available config will be used if multiple exist.
        gpu_filter (str, optional): A string to filter GPU names. If multiple GPUs are found,
                                    the most recent one matching the filter is used. Defaults to None.
        display (bool): Display the visualization. Defaults to False
        overwrite (bool): Overwrite existing visualization, if none exist this flag has no effect as ones are always created and saved. Defaults to False

    """

    kernel_name: str
    metric_name: str
    kernel_operation_mode: str = "full"
    extra_config_filter: str | None = None
    gpu_filter: str | None = None
    display: bool = False
    overwrite: bool = False


def parse_args() -> VisualizationsConfig:
    """Parse command line arguments into a configuration object.

    Returns:
        VisualizationsConfig: Configuration object for the visualizations script.
    """
    parser = ArgumentParser()
    parser.add_argument("--kernel-name", type=str, required=True, help="Kernel name to benchmark")
    parser.add_argument(
        "--metric-name",
        type=str,
        required=True,
        help="Metric name to visualize (speed/memory)",
    )
    parser.add_argument(
        "--kernel-operation-mode",
        type=str,
        nargs="*",
        default=None,
        help="Kernel operation modes to visualize (forward/backward/full). If not provided, generate for all available modes.",
    )
    parser.add_argument(
        "--extra-config-filter",
        type=str,
        default=None,
        help="A string to filter extra_benchmark_config. "
        "Can be a substring to match or a JSON-like 'key=value' pair (e.g., \"'H': 4096\" or \"H=4096\" for simple cases). "
        "Defaults to None (first available config if multiple exist).",
    )
    parser.add_argument(
        "--gpu-filter",
        type=str,
        default=None,
        help="A string to filter GPU names. If multiple GPUs are found, the most recent one matching the filter is used.",
    )
    parser.add_argument("--display", action="store_true", help="Display the visualization")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing visualization, if none exist this flag has no effect as one are always created",
    )

    args = parser.parse_args()
    return args


def extra_config_filter(df: pd.DataFrame, config: VisualizationsConfig) -> pd.DataFrame:
    """Filters the dataframe based on the extra_benchmark_config using the provided filter string.

    Args:
        df (pd.DataFrame): The dataframe to filter.
        config (VisualizationsConfig): Configuration object for the visualizations script.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    unique_extra_configs_str = df["extra_benchmark_config_str"].unique()
    selected_extra_config_str = None

    if config.extra_config_filter:
        matched_configs = []
        try:
            if "=" in config.extra_config_filter:
                key_filter, value_filter = config.extra_config_filter.split("=", 1)
                for cfg_str in unique_extra_configs_str:
                    cfg_json = json.loads(cfg_str)
                    if str(cfg_json.get(key_filter.strip("'\" "))) == value_filter.strip("'\" "):
                        matched_configs.append(cfg_str)
            if not matched_configs:
                matched_configs = [
                    cfg_str for cfg_str in unique_extra_configs_str if config.extra_config_filter in cfg_str
                ]
        except Exception as e:
            print(
                f"Note: Could not parse extra_config_filter '{config.extra_config_filter}' as key=value ({e}), using substring match."
            )
            matched_configs = [cfg_str for cfg_str in unique_extra_configs_str if config.extra_config_filter in cfg_str]

        if matched_configs:
            if len(matched_configs) > 1:
                print(
                    f"Warning: Multiple extra_benchmark_configs match filter '{config.extra_config_filter}': {matched_configs}. "
                    f"Using the first one: {matched_configs[0]}"
                )
            selected_extra_config_str = matched_configs[0]
        else:
            print(
                f"Warning: No extra_benchmark_config matches filter '{config.extra_config_filter}'. "
                f"Available configs for {config.kernel_name} ({config.metric_name}, {config.kernel_operation_mode}): {list(unique_extra_configs_str)}"
            )
            if len(unique_extra_configs_str) > 0:
                selected_extra_config_str = unique_extra_configs_str[0]
                print(f"Defaulting to the first available extra_benchmark_config: {selected_extra_config_str}")
            else:
                raise ValueError("No extra_benchmark_config available to select after failed filter attempt.")

    elif len(unique_extra_configs_str) > 1:
        selected_extra_config_str = unique_extra_configs_str[0]
        print(
            f"Warning: Multiple extra_benchmark_configs found for {config.kernel_name} ({config.metric_name}, {config.kernel_operation_mode})."
        )
        print(f"Defaulting to use: {selected_extra_config_str}")
        print(f"Available configs: {list(unique_extra_configs_str)}")
        print(
            "Use the --extra-config-filter argument to select a specific one "
            "(e.g., --extra-config-filter \"'H': 4096\" or a substring like \"'seq_len': 512\")."
        )
    elif len(unique_extra_configs_str) == 1:
        selected_extra_config_str = unique_extra_configs_str[0]
        print(f"Using unique extra_benchmark_config: {selected_extra_config_str}")

    if selected_extra_config_str:
        filtered_df = df[df["extra_benchmark_config_str"] == selected_extra_config_str]
    else:
        print("Warning: Could not select an extra_benchmark_config. Using data from initial filter if any.")
        filtered_df = df

    return filtered_df


def gpu_name_filter(df: pd.DataFrame, config: VisualizationsConfig) -> pd.DataFrame:
    """Filters the dataframe based on the GPU name using the provided filter string.

    Args:
        df (pd.DataFrame): The dataframe to filter.
        config (VisualizationsConfig): Configuration object for the visualizations script.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    gpu_grouped = df.groupby(["gpu_name"])

    if config.gpu_filter:
        try:
            matched_gpu_names = [gpu_name for gpu_name in gpu_grouped.groups.keys() if config.gpu_filter in gpu_name]
            if matched_gpu_names:
                selected_gpu_name = matched_gpu_names[-1]
                if len(matched_gpu_names) > 1:
                    print(
                        f"Warning: Multiple GPU names match filter '{config.gpu_filter}' for ({config.metric_name}, {config.kernel_operation_mode})"
                    )
                    print(f"Deafulting to use: {selected_gpu_name}.")
                    print(f"Available GPU names: {matched_gpu_names}.")
                    print(
                        "Use a more specific --gpu-filter argument to select a specific one."
                        "(e.g., --gpu-filter 'A100' or 'H100')."
                    )

            else:
                latest_gpu_name = list(gpu_grouped.groups.keys())[-1]
                selected_gpu_name = latest_gpu_name
                print(
                    f"Warning: No GPU names match filter '{config.gpu_filter}'. "
                    f"Available GPU names: {list(gpu_grouped.groups.keys())}. "
                    f"Picking the most recent one."
                )

        except Exception as e:
            print(
                f"Warning: Error occurred while filtering GPU names with filter '{config.gpu_filter}': {e}. "
                f"Picking the most recent one."
            )
            selected_gpu_name = list(gpu_grouped.groups.keys())[-1]

        final_filtered_df = gpu_grouped.get_group((selected_gpu_name,))
    else:
        if len(gpu_grouped.groups) == 1:
            final_filtered_df = df
        else:
            available_keys = list(gpu_grouped.groups.keys())
            latest_gpu_name = available_keys[-1]
            print(f"Collected data from more than 1 gpu. Picking the most recent data on {latest_gpu_name}")
            print(f"Available GPU names: {available_keys}.")
            print(
                f"Use the --gpu-filter argument to select a specific one. (e.g., --gpu-filter '{available_keys[0]}')."
            )
            final_filtered_df = gpu_grouped.get_group((latest_gpu_name,))
    return final_filtered_df


def load_data(config: VisualizationsConfig) -> pd.DataFrame:
    """Loads the benchmark data from the CSV file and filters it based on the configuration.

    Args:
        config (VisualizationsConfig): Configuration object for the visualizations script.

    Raises:
        ValueError: If no data is found for the given filters.

    Returns:
        pd.DataFrame: Filtered benchmark dataframe.
    """
    df = pd.read_csv(DATA_PATH)
    df["extra_benchmark_config"] = df["extra_benchmark_config_str"].apply(json.loads)

    base_filtered_df = df[
        (df["kernel_name"] == config.kernel_name)
        & (df["metric_name"] == config.metric_name)
        & (df["kernel_operation_mode"] == config.kernel_operation_mode)
    ]

    if base_filtered_df.empty:
        raise ValueError(
            f"No data found for kernel_name='{config.kernel_name}', "
            f"metric_name='{config.metric_name}', "
            f"kernel_operation_mode='{config.kernel_operation_mode}'."
        )

    config_filtered_df = extra_config_filter(base_filtered_df, config)
    print(config_filtered_df["extra_benchmark_config_str"].unique())
    print(config_filtered_df["gpu_name"].unique())
    final_filtered_df = gpu_name_filter(config_filtered_df, config)

    if final_filtered_df.empty:
        raise ValueError(
            f"No data found after attempting to filter with extra_benchmark_config and gpu_name for "
            f"kernel_name='{config.kernel_name}', "
            f"metric_name='{config.metric_name}', "
            f"kernel_operation_mode='{config.kernel_operation_mode}'."
        )

    return final_filtered_df


def plot_data(df: pd.DataFrame, config: VisualizationsConfig):
    """Plots the benchmark data, saving the result if needed.

    Args:
        df (pd.DataFrame): Filtered benchmark dataframe.
        config (VisualizationsConfig): Configuration object for the visualizations script.
    """
    for col in ["y_value_20", "y_value_50", "y_value_80"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    xlabel = df["x_label"].iloc[0]
    ylabel = f"{config.metric_name} ({df['metric_unit'].iloc[0]})"
    # Sort by "kernel_provider" to ensure consistent color assignment
    df = df.sort_values(by="kernel_provider")

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    try:
        ax = sns.lineplot(
            data=df,
            x="x_value",
            y="y_value_50",
            hue="kernel_provider",
            marker="o",
            palette="tab10",
            errorbar=("ci", None),
        )
    except Exception:
        ax = sns.lineplot(
            data=df,
            x="x_value",
            y="y_value_50",
            hue="kernel_provider",
            marker="o",
            palette="tab10",
            errorbar=None,
        )

    # Seaborn can't plot pre-computed error bars, so we need to do it manually
    lines = ax.get_lines()
    colors = [line.get_color() for line in lines]

    for (_, group_data), color in zip(df.groupby("kernel_provider"), colors):
        # for i, row in group_data.iterrows():
        y_error_lower = group_data["y_value_50"] - group_data["y_value_20"]
        y_error_upper = group_data["y_value_80"] - group_data["y_value_50"]
        y_error = [y_error_lower, y_error_upper]

        plt.errorbar(
            group_data["x_value"],
            group_data["y_value_50"],
            yerr=y_error,
            fmt="o",
            color=color,
            capsize=5,
        )
    plt.legend(title="Kernel Provider")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    out_path = os.path.join(
        VISUALIZATIONS_PATH,
        f"{config.kernel_name}_{config.metric_name}_{config.kernel_operation_mode}.png",
    )

    if config.display:
        plt.show()
    if config.overwrite or not os.path.exists(
        out_path
    ):  # Save the plot if it doesn't exist or if we want to overwrite it
        os.makedirs(VISUALIZATIONS_PATH, exist_ok=True)
        plt.savefig(out_path)
    plt.close()


def main():
    args = parse_args()
    all_df = pd.read_csv(DATA_PATH)
    all_df["extra_benchmark_config"] = all_df["extra_benchmark_config_str"].apply(json.loads)

    if args.metric_name == "memory":
        modes = ["full"]
    elif args.kernel_operation_mode:
        modes = args.kernel_operation_mode
    else:
        filtered = all_df[(all_df["kernel_name"] == args.kernel_name) & (all_df["metric_name"] == args.metric_name)]
        modes = filtered["kernel_operation_mode"].unique().tolist()
        if not modes:
            print(f"No data found for kernel '{args.kernel_name}' and metric '{args.metric_name}'.", file=sys.stderr)
            sys.exit(1)

    for mode in modes:
        config = VisualizationsConfig(
            kernel_name=args.kernel_name,
            metric_name=args.metric_name,
            kernel_operation_mode=mode,
            extra_config_filter=args.extra_config_filter,
            gpu_filter=args.gpu_filter,
            display=args.display,
            overwrite=args.overwrite,
        )
        df = load_data(config)
        plot_data(df, config)


if __name__ == "__main__":
    main()
