"""
Functions for plotting data from the statistics pipeline.

Author: Christopher Millward
"""
from matplotlib.ticker import StrMethodFormatter
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import ptitprince as pt
from scipy import stats
import seaborn as sns
import numpy as np
from typing import Literal

from config import CUMULATIVE_MOTION_RAINCLOUD_PATH, OPERATED_CUMULATIVE_MOTION_HEATMAP_PATH
from modules.statistics import get_only_one_sided_participants, create_cumulative_totals_dataframe
from schema import ParticipantDetails


# -------------------------------------------------------------------
# Set Defaults
# -------------------------------------------------------------------
sns.set_theme(
    style="whitegrid",
    context="paper",
)
fig_size = (7.16, 5.0)  # 1 column width in inches
palette = "cividis"  # color palette for plots
dpi = 600  # dots per inch for the plots
transparent = False  # whether to save plots with transparent background
titles = True

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------


def _stack_heatmaps(
    data: list[ParticipantDetails],
    side: Literal["operated", "non_operated"],
    value: Literal["cumulative_motion", "elevation", "POE", "IR_ER"],
    motion_type: Literal["humerothoracic", "glenohumeral"],
) -> np.ndarray:
    """
    Stack heatmaps from a list of ParticipantDetails objects.

    Args:
        data (list[ParticipantDetails]): List of ParticipantDetails objects.
        side (Literal["operated", "non_operated"]): The side of the body to consider.
        value (Literal["cumulative_motion", "elevation", "POE", "IR_ER"]): The value to extract from the heatmap.
        motion_type (Literal["humerothoracic", "glenohumeral"]): The type of motion to consider.

    Returns:
        np.ndarray: A stacked array of heatmaps.
    """
    heatmaps = []

    for participant in data:
        side_data = getattr(participant, side)[0]
        motion_data = getattr(side_data, motion_type)
        heatmap = getattr(motion_data.heatmap, value)

        heatmaps.append(heatmap)

    return np.stack(heatmaps, axis=0)


# -------------------------------------------------------------------
# Plotting Functions
# -------------------------------------------------------------------
def plot_raincloud(
        data: list[ParticipantDetails],
        out_path: Path,
        fig_size: tuple[float, float] = fig_size,
        palette: str = palette,
        transparent: bool = transparent,
        dpi: int = 600,
        titles: bool = titles

) -> None:
    """Plot a raincloud plot of the cumulative totals.

    Args:
        data (list[ParticipantDetails]): List of ParticipantDetails objects.
        out_path (str): Path to save the plot.
        fig_size (tuple): Size of the figure.
    """
    # Prep data
    one_sided_participants = get_only_one_sided_participants(data)
    cumulative_totals = create_cumulative_totals_dataframe(
        one_sided_participants)
    df_long = pd.melt(
        cumulative_totals,
        id_vars=["participant"],
        var_name="side",
        value_name="total",
    )

    # calculate p-value for paired t-test
    t_stat, p_value = stats.ttest_rel(
        cumulative_totals['Operated'],
        cumulative_totals['Non-operated']
    )

    # Make raincloud
    # set the theme for the plot
    fig, ax = plt.subplots(figsize=fig_size)

    # Overlay paired observations first
    for participant, group in df_long.groupby("participant"):
        if len(group) == 2:
            x = [
                group.loc[group["side"] == "Operated", "total"].iloc[0],
                group.loc[group["side"] == "Non-operated", "total"].iloc[0],
            ]

            ax.plot(
                x,
                [0, 1],
                color="gray",
                alpha=0.25,
                linewidth=0.75,
                zorder=0,
            )

    # Create raincloud plot
    pt.RainCloud(
        data=df_long,
        x="side",  # type: ignore
        y="total",  # type: ignore
        jitter=False,
        palette=palette,
        hue="side",  # type: ignore
        bw=0.2,
        width_viol=0.7,
        orient="h",
        ax=ax,
    )

    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2e}"))
    ax.set_xlim(left=5e3)  # I manually set this after seeing the plot.

    ax.set_xlabel("Cumulative Humerothoracic Rotation (rad)")
    ax.set_ylabel("Arm")
    if titles:
        ax.set_title(
            "Distribution of Cumulative Humerothoracic Rotation "
            "in Operated vs. Non-Operated Sides"
        )

    # Annotate p-value
    annotation_x_pos = 2.2e4
    x_min = 5e3
    x_range = annotation_x_pos - x_min
    x_bracket = annotation_x_pos + 0.05 * x_range
    bracket_width = 0.02 * x_range

    # Draw bracket
    ax.plot(
        [x_bracket, x_bracket],
        [0, 1],
        color="black",
        linewidth=1,
    )

    ax.plot(
        [x_bracket - bracket_width, x_bracket],
        [0, 0],
        color="black",
        linewidth=1,
    )

    ax.plot(
        [x_bracket - bracket_width, x_bracket],
        [1, 1],
        color="black",
        linewidth=1,
    )

    # Add p-value text
    ax.text(
        x_bracket + 0.01 * x_range,
        0.5,
        f"$p = {p_value:.2f}$",
        ha="left",
        va="center",
        fontsize=8,
    )

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight",
                transparent=transparent)


def plot_heatmap(
    data: list[ParticipantDetails],
    side: Literal["operated", "non_operated"],
    value: Literal["cumulative_motion", "elevation", "POE", "IR_ER"],
    motion_type: Literal["humerothoracic", "glenohumeral"],
    out_path: Path,
    fig_size: tuple[float, float] = fig_size,
    palette: str = palette,
    transparent: bool = transparent,
    dpi: int = dpi,
    titles: bool = titles
) -> None:

    # Prep the data
    stacked_heatmaps = _stack_heatmaps(
        data=data,
        side=side,
        value=value,
        motion_type=motion_type
    )
    mean_heatmap = np.mean(stacked_heatmaps, axis=0)
    # Use ddof=1 for sample standard deviation
    std_heatmap = np.std(stacked_heatmaps, axis=0, ddof=1)

    # plot the heatmap with mean and std
    root_heatmap = getattr(getattr(data[0], side)[0], motion_type).heatmap
    x_min = 0
    x_max = root_heatmap.poe_range_end
    y_min = 0
    y_max = root_heatmap.elevation_range_end
    step = root_heatmap.bin_width

    # Create bin-center coordinates
    x = np.arange(x_min, x_max, step)
    y = np.arange(y_min, y_max, step)

    # Scale values by 10^2 for display
    mean_scaled = mean_heatmap / 100
    std_scaled = std_heatmap / 100

    # Create annotation strings: "mean (std)"
    annotations = np.empty(mean_heatmap.shape, dtype=object)

    for i in range(mean_heatmap.shape[0]):
        for j in range(mean_heatmap.shape[1]):
            annotations[i, j] = (
                f"{mean_scaled[i, j]:.1f}\n"
                f"({std_scaled[i, j]:.1f})"
            )

    # Create heatmap
    fig, ax = plt.subplots(figsize=tuple(x*1.4 for x in fig_size))

    sns.heatmap(
        mean_scaled,
        annot=annotations,
        fmt="",
        cmap=palette,
        xticklabels=x,
        yticklabels=y,
        cbar_kws={"label": r"Mean Rotation (rad) $\times 10^2$"},
        ax=ax,
    )

    ax.invert_yaxis()
    ax.set_xlabel("Plane of Elevation (deg)")
    ax.set_ylabel("Elevation (deg)")
    if titles:
        ax.set_title(
            f"Mean (SD) {motion_type.capitalize()} Rotation Observed in RTSA Shoulders "
            f"in Each Range of Elevation and Plane of Elevation"
        )

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight",
                transparent=transparent)


def create_and_save_all_figures(
    data: list[ParticipantDetails],
    fig_size: tuple[float, float] = fig_size,
    palette: str = palette,
    transparent: bool = transparent,
    titles: bool = titles,
    dpi=dpi
):
    """Plot all figures for the statistics pipeline.

    Args:
        data (list[ParticipantDetails]): List of ParticipantDetails objects.
        out_dir (str): Directory to save the plots.
        fig_size (tuple): Size of the figure.
        palette (str): Color palette for the plots.
    """
    plot_raincloud(
        data=data,
        out_path=CUMULATIVE_MOTION_RAINCLOUD_PATH,
        fig_size=fig_size,
        palette=palette,
        transparent=transparent,
        dpi=dpi,
        titles=titles
    )

    plot_heatmap(
        data=data,
        side="operated",
        value="cumulative_motion",
        motion_type="humerothoracic",
        out_path=OPERATED_CUMULATIVE_MOTION_HEATMAP_PATH,
        fig_size=fig_size,
        palette=palette,
        transparent=transparent,
        dpi=dpi,
        titles=titles
    )
