import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from argparse import ArgumentParser
from dataclasses import dataclass

DATA_PATH = "data/all_benchmark_data.csv"
VISUALIZATIONS_PATH = "visualizations/"


@dataclass
class VisualizationsConfig:
    """
    Configuration for the visualizations script.
    """

    kernel_name: str
    metric_name: str
    kernel_operation_mode: str = "full"
    display: bool = False
    overwrite: bool = False


def parse_args() -> VisualizationsConfig:
    parser = ArgumentParser()
    parser.add_argument(
        "--kernel-name", type=str, required=True, help="Kernel name to benchmark"
    )
    parser.add_argument(
        "--metric-name",
        type=str,
        required=True,
        help="Metric name to visualize (speed/memory)",
    )
    parser.add_argument(
        "--kernel-operation-mode",
        type=str,
        required=True,
        help="Kernel operation mode to visualize (forward/backward/full)",
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

    return VisualizationsConfig(**dict(args._get_kwargs()))


def load_data(config: VisualizationsConfig) -> pd.DataFrame:
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
        errorbar=("ci", None),
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
    if config.overwrite or not os.path.exists(out_path):
        os.makedirs(VISUALIZATIONS_PATH, exist_ok=True)
        plt.savefig(out_path)
    plt.close()


def main():
    config = parse_args()
    df = load_data(config)
    plot_data(df, config)


if __name__ == "__main__":
    main()
