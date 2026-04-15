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


# Map --sweep-mode values to the x_name used in benchmark CSV data.
# "model_config" sweeps always write x_name="model_config"; token-length
# sweeps use kernel-specific names (e.g. "T"), so we match them by exclusion.
SWEEP_MODE_X_NAME = "model_config"


@dataclass
class VisualizationsConfig:
    """
    Configuration for the visualizations script.

    Args:
        kernel_name (str): Kernel name to benchmark. (Will run `scripts/benchmark_{kernel_name}.py`)
        metric_name (str): Metric name to visualize (speed/memory)
        kernel_operation_mode (str): Kernel operation mode to visualize (forward/backward/full). Defaults to "full"
        sweep_mode (str, optional): Sweep mode to filter data. "token_length" selects
            token/sequence-length sweep data; "model_config" selects model-configuration
            sweep data. When None, all data is considered (legacy behaviour).
        extra_config_filter (str, optional): A string to filter extra_benchmark_config.
                                            Can be a substring to match or a 'key=value' pair (e.g., "'H': 4096").
                                            Defaults to None, which means the first available config will be used if multiple exist.
        display (bool): Display the visualization. Defaults to False
        overwrite (bool): Overwrite existing visualization, if none exist this flag has no effect as ones are always created and saved. Defaults to False

    """

    kernel_name: str
    metric_name: str
    kernel_operation_mode: str = "full"
    sweep_mode: str = "token_length"
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
        "--sweep-mode",
        type=str,
        choices=["token_length", "model_config"],
        default="token_length",
        help="Sweep mode used when running the benchmark. "
        "'token_length' selects token/sequence-length sweep data (default); "
        "'model_config' selects model-configuration sweep data.",
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
        help="Filter by GPU name. When multiple devices are present, selects "
        "the matching GPU (uses most recent match if multiple found). "
        "If omitted, the most recent device is used automatically.",
    )
    parser.add_argument("--display", action="store_true", help="Display the visualization")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing visualization, if none exist this flag has no effect as one are always created",
    )

    args = parser.parse_args()
    return args


def gpu_name_filter(df: pd.DataFrame, gpu_filter: str | None = None) -> pd.DataFrame:
    """Filter benchmark data by GPU name when multiple devices are present.

    Args:
        df: Pre-filtered benchmark dataframe.
        gpu_filter: Optional GPU name substring to match. If provided, selects
            the matching GPU (uses most recent if multiple match). If None,
            automatically picks the most recent device.

    Returns:
        pd.DataFrame: Dataframe filtered to a single GPU.
    """
    if "gpu_name" not in df.columns or df.empty:
        return df

    unique_gpus = df["gpu_name"].unique()
    if len(unique_gpus) <= 1:
        return df

    if gpu_filter:
        matched = [g for g in unique_gpus if gpu_filter in g]
        if matched:
            if len(matched) > 1:
                # Multiple matches — pick the most recent
                matched_df = df[df["gpu_name"].isin(matched)]
                selected = matched_df.sort_values("timestamp", ascending=False)["gpu_name"].iloc[0]
                print(
                    f"Warning: Multiple GPUs match filter '{gpu_filter}': {matched}. "
                    f"Using the most recent: '{selected}'."
                )
            else:
                selected = matched[0]
        else:
            # No match — fall back to most recent GPU
            selected = df.sort_values("timestamp", ascending=False)["gpu_name"].iloc[0]
            print(
                f"Warning: No GPU matches filter '{gpu_filter}'. "
                f"Available GPUs: {list(unique_gpus)}. "
                f"Falling back to most recent device: '{selected}'."
            )
    else:
        # No filter provided — pick the most recent device
        selected = df.sort_values("timestamp", ascending=False)["gpu_name"].iloc[0]
        print(
            f"Warning: Data contains entries from multiple devices: {list(unique_gpus)}. "
            f"Using data from the most recent device: '{selected}'. "
            f"Use --gpu-filter to select a specific device."
        )

    return df[df["gpu_name"] == selected]


def extra_config_filter(df: pd.DataFrame, config: VisualizationsConfig) -> pd.DataFrame:
    """Filter benchmark data by extra_benchmark_config.

    Args:
        df: Pre-filtered benchmark dataframe (already filtered by kernel, metric, etc.).
        config: Visualization configuration with optional extra_config_filter.

    Returns:
        pd.DataFrame: Dataframe filtered to a single extra_benchmark_config.
    """
    unique_extra_configs_str = df["extra_benchmark_config_str"].unique()
    selected_extra_config_str = None

    if len(unique_extra_configs_str) == 0:
        print(
            "Warning: No extra_benchmark_config found for the initial filters. "
            "Proceeding with all data from initial filter."
        )
        return df

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
        result_df = df[df["extra_benchmark_config_str"] == selected_extra_config_str]
    else:
        print("Warning: Could not select an extra_benchmark_config. Using data from initial filter if any.")
        result_df = df

    if result_df.empty:
        raise ValueError(
            f"No data found after attempting to filter by extra_benchmark_config. "
            f"Selected/Defaulted extra_config_str: {selected_extra_config_str}"
            if selected_extra_config_str
            else "No specific extra_config was selected."
        )

    print(
        f"Plotting data for extra_benchmark_config: {json.loads(selected_extra_config_str if selected_extra_config_str else '{}')}"
    )
    return result_df


def load_data(config: VisualizationsConfig) -> pd.DataFrame:
    """Loads the benchmark data from the CSV file and filters it based on the configuration.

    Applies filters in order: kernel/metric/mode → sweep mode → GPU → extra config.

    Args:
        config (VisualizationsConfig): Configuration object for the visualizations script.

    Raises:
        ValueError: If no data is found for the given filters.

    Returns:
        pd.DataFrame: Filtered benchmark dataframe.
    """
    df = pd.read_csv(DATA_PATH)
    df["extra_benchmark_config"] = df["extra_benchmark_config_str"].apply(json.loads)

    mask = (
        (df["kernel_name"] == config.kernel_name)
        & (df["metric_name"] == config.metric_name)
        & (df["kernel_operation_mode"] == config.kernel_operation_mode)
    )

    # Filter by sweep mode early, before extra_benchmark_config resolution.
    if config.sweep_mode == "model_config":
        mask = mask & (df["x_name"] == SWEEP_MODE_X_NAME)
    elif config.sweep_mode == "token_length":
        mask = mask & (df["x_name"] != SWEEP_MODE_X_NAME)

    base_filtered_df = df[mask]

    if base_filtered_df.empty:
        raise ValueError(
            f"No data found for kernel_name='{config.kernel_name}', "
            f"metric_name='{config.metric_name}', "
            f"kernel_operation_mode='{config.kernel_operation_mode}'."
        )

    # Apply GPU filter, then extra config filter
    base_filtered_df = gpu_name_filter(base_filtered_df, config.gpu_filter)
    return extra_config_filter(base_filtered_df, config)


def plot_data(df: pd.DataFrame, config: VisualizationsConfig):
    """Plots the benchmark data, saving the result if needed.

    Args:
        df (pd.DataFrame): Filtered benchmark dataframe.
        config (VisualizationsConfig): Configuration object for the visualizations script.
    """
    for col in ["y_value_20", "y_value_50", "y_value_80"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert x_value to numeric where possible so matplotlib uses a real
    # numeric axis (proper proportional spacing).  String x_values (e.g.
    # model names) stay as-is and will be treated as categorical (evenly spaced).
    x_numeric = pd.to_numeric(df["x_value"], errors="coerce")
    is_numeric_x = x_numeric.notna().all()
    if is_numeric_x:
        df["x_value"] = x_numeric

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

    # For numeric x axes, show tick labels only at actual data points
    if is_numeric_x:
        tick_values = sorted(df["x_value"].unique())
        ax.set_xticks(tick_values)
        ax.set_xticklabels([str(int(v)) if v == int(v) else str(v) for v in tick_values])

    # Seaborn can't plot pre-computed error bars, so we need to do it manually
    lines = ax.get_lines()
    colors = [line.get_color() for line in lines]

    for (_, group_data), color in zip(df.groupby("kernel_provider"), colors):
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

    sweep_suffix = f"_{config.sweep_mode}" if config.sweep_mode else ""
    out_path = os.path.join(
        VISUALIZATIONS_PATH,
        f"{config.kernel_name}_{config.metric_name}_{config.kernel_operation_mode}{sweep_suffix}.png",
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
            sweep_mode=args.sweep_mode,
            extra_config_filter=args.extra_config_filter,
            gpu_filter=args.gpu_filter,
            display=args.display,
            overwrite=args.overwrite,
        )
        df = load_data(config)
        plot_data(df, config)


if __name__ == "__main__":
    main()
