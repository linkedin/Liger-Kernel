import json
import os
import sys
from argparse import ArgumentParser
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = os.path.join(os.path.dirname(__file__), "data/all_benchmark_data.csv")
VISUALIZATIONS_PATH = os.path.join(os.path.dirname(__file__), "visualizations/")


@dataclass
class VisualizationsConfig:
    """
    Configuration for the visualizations script.

    Args:
        kernel_name (str): Kernel name to benchmark. (Will run `scripts/benchmark_{kernel_name}.py`)
        metric_name (str): Metric name to visualize (speed/memory)
        kernel_operation_mode (str): Kernel operation mode to visualize (forward/backward/full). Defaults to "full"
        display (bool): Display the visualization. Defaults to False
        overwrite (bool): Overwrite existing visualization, if none exist this flag has no effect as ones are always created and saved. Defaults to False

    """

    kernel_name: str
    metric_name: str
    kernel_operation_mode: str = "full"
    display: bool = False
    overwrite: bool = False


def get_available_options():
    csv_path = os.path.join(os.path.dirname(__file__), DATA_PATH)
    df = pd.read_csv(csv_path)
    return {
        "kernel_name": df["kernel_name"].unique().tolist(),
        "metric_name": df["metric_name"].unique().tolist(),
        "kernel_operation_mode": df["kernel_operation_mode"].unique().tolist(),
    }


def parse_args() -> VisualizationsConfig:
    """Parse command line arguments into a configuration object.

    Returns:
        VisualizationsConfig: Configuration object for the visualizations script.
    """
    available_options = get_available_options()

    parser = ArgumentParser(description="Visualize benchmark data", add_help=False)
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help message and exit"
    )
    parser.add_argument(
        "--kernel-name",
        type=str,
        help=f"Kernel name to benchmark. Options: {', '.join(available_options['kernel_name'])}",
    )
    parser.add_argument(
        "--metric-name",
        type=str,
        help=f"Metric name to visualize. Options: {', '.join(available_options['metric_name'])}",
    )
    parser.add_argument(
        "--kernel-operation-mode",
        type=str,
        help=f"Kernel operation mode to visualize. Options: {', '.join(available_options['kernel_operation_mode'])}",
    )
    parser.add_argument(
        "--display", action="store_true", help="Display the visualization"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing visualization, if none exist this flag has no effect as one are always created",
    )

    args = parser.parse_args()

    if args.help or len(sys.argv) == 1:
        parser.print_help()
        print("\nAvailable options:")
        for arg, options in available_options.items():
            print(f"  {arg}: {', '.join(options)}")
        sys.exit(0)

    if not all([args.kernel_name, args.metric_name, args.kernel_operation_mode]):
        parser.error(
            "--kernel-name, --metric-name, and --kernel-operation-mode are required arguments"
        )

    if args.kernel_name not in available_options["kernel_name"]:
        parser.error(
            f"Invalid kernel name. Choose from: {', '.join(available_options['kernel_name'])}"
        )
    if args.metric_name not in available_options["metric_name"]:
        parser.error(
            f"Invalid metric name. Choose from: {', '.join(available_options['metric_name'])}"
        )
    if args.kernel_operation_mode not in available_options["kernel_operation_mode"]:
        parser.error(
            f"Invalid kernel operation mode. Choose from: {', '.join(available_options['kernel_operation_mode'])}"
        )

    args_dict = vars(args)
    args_dict.pop("help", None)

    return VisualizationsConfig(**args_dict)


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

    filtered_df = df[
        (df["kernel_name"] == config.kernel_name)
        & (df["metric_name"] == config.metric_name)
        & (df["kernel_operation_mode"] == config.kernel_operation_mode)
        # Use this to filter by extra benchmark configuration property
        # & (data['extra_benchmark_config'].apply(lambda x: x.get('H') == 4096))
        # FIXME: maybe add a way to filter using some configuration, except of hardcoding it
    ]

    if filtered_df.empty:
        raise ValueError("No data found for the given filters")

    return filtered_df


def plot_data(df: pd.DataFrame, config: VisualizationsConfig):
    """Plots the benchmark data, saving the result if needed.

    Args:
        df (pd.DataFrame): Filtered benchmark dataframe.
        config (VisualizationsConfig): Configuration object for the visualizations script.
    """
    xlabel = df["x_label"].iloc[0]
    ylabel = f"{config.metric_name} ({df['metric_unit'].iloc[0]})"
    # Sort by "kernel_provider" to ensure consistent color assignment
    df = df.sort_values(by="kernel_provider")

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
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
        VISUALIZATIONS_PATH, f"{config.kernel_name}_{config.metric_name}.png"
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
    config = parse_args()
    df = load_data(config)
    plot_data(df, config)


if __name__ == "__main__":
    main()
