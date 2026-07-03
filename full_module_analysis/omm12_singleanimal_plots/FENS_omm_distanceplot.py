import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

try:
    import seaborn as sns
except ImportError:
    sns = None


PIXEL_PER_CM = 36.39
MM_PER_INCH = 25.4

"""
Construct a Figure panel for a poster using matplotlib
The panel size is 229,146 mm * 90 mm
Two plots are generated: The plot is on dark background, therefore axis and text needs to be white
The plots (A, B) are placed in a 1x2 row of the panel size, each plot is equal sized
For each plot, I want to be able to adjust the following parameters:
- scatter point colors
- scatter point size
- x and y axis line thickness
- font size of x and y labels and ticks
- distance between plots
- tick length of x and y ticks
- top, bottom, left and right margins of the panel


- no titles
- no panel letters A or B
"""


path_germfree = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\FENS\data_distance\germfree"
germfree_files = glob.glob(os.path.join(path_germfree, "*.csv"))

path_omm12 = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\FENS\data_distance\omm12"
omm12_files = glob.glob(os.path.join(path_omm12, "*.csv"))

PANEL_WIDTH_MM = 229.146
PANEL_HEIGHT_MM = 90

PLOT_STYLE = {
    "scatter_colors": ("white", "0.55"),
    "scatter_size": 30,
    "axis_line_width": 2.835,
    "tick_length": 2.835 * 2,
    "label_font_size": 14,
    "tick_font_size": 14,
}

PLOT_LAYOUT = {
    "left_margin_mm": 20,
    "right_margin_mm": 7,
    "bottom_margin_mm": 13,
    "top_margin_mm": 7,
    "plot_spacing_mm": 22,
}

COMPARISONS = (
    {
        "groups": ("group A", "group B"),
        "values": (
            np.array([138.21, 109.97, 78.98, 169.84, 169.14, 135.44]),
            np.array([5.38, 24.68, 11.92, 101.24, 61.72, 40.29]),
        ),
        "ylabel": "distance [m]",
        "ylim": (0, 180),
    },
    {
        "groups": ("group C", "group D"),
        "values": (
            np.array([41.06, 26.72, 42.8, 41.59, 33.31, 39.77]),
            np.array([45.08, 45.93, 40.85, 67.73, 78.33, 63.78]),
        ),
        "ylabel": "distance [m]",
        "ylim": (0, 90),
    },
)

OUTPUT_PATH = r"C:\Users\Fabian\Desktop\Transfer\FENS\distance.svg"


def mm_to_inches(value):
    return value / MM_PER_INCH


def mm_to_figure_fraction(value_mm, figure_size_mm):
    return value_mm / figure_size_mm


def _value_for_index(value, index):
    if isinstance(value, (list, tuple, np.ndarray)):
        return value[index % len(value)]
    return value


def despine_axis(ax):
    if sns is not None:
        sns.despine(ax=ax)
        return

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def style_dark_axis(ax, style):
    ax.set_facecolor("black")
    ax.tick_params(
        axis="both",
        colors="white",
        labelsize=style["tick_font_size"],
        width=style["axis_line_width"],
        length=style["tick_length"],
    )

    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_linewidth(style["axis_line_width"])

    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")


def apply_panel_layout(fig, plot_spacing_mm, layout):
    left = mm_to_figure_fraction(layout["left_margin_mm"], PANEL_WIDTH_MM)
    right = 1 - mm_to_figure_fraction(layout["right_margin_mm"], PANEL_WIDTH_MM)
    bottom = mm_to_figure_fraction(layout["bottom_margin_mm"], PANEL_HEIGHT_MM)
    top = 1 - mm_to_figure_fraction(layout["top_margin_mm"], PANEL_HEIGHT_MM)

    inner_width_mm = PANEL_WIDTH_MM - layout["left_margin_mm"] - layout["right_margin_mm"]
    plot_width_mm = (inner_width_mm - plot_spacing_mm) / 2
    if plot_width_mm <= 0:
        raise ValueError("plot_spacing_mm is too large for the selected panel width and margins.")

    wspace = plot_spacing_mm / plot_width_mm
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace)


def assert_panel_content_in_bounds(fig, tolerance_px=0.5):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    content_box = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)

    outside = (
        content_box.x0 < figure_box.x0 - tolerance_px
        or content_box.y0 < figure_box.y0 - tolerance_px
        or content_box.x1 > figure_box.x1 + tolerance_px
        or content_box.y1 > figure_box.y1 + tolerance_px
    )
    if outside:
        left_overrun = max(figure_box.x0 - content_box.x0, 0)
        bottom_overrun = max(figure_box.y0 - content_box.y0, 0)
        right_overrun = max(content_box.x1 - figure_box.x1, 0)
        top_overrun = max(content_box.y1 - figure_box.y1, 0)
        raise ValueError(
            "Figure content extends outside the panel boundary. "
            "Increase the margins in PLOT_LAYOUT, reduce font sizes, or reduce plot_spacing_mm. "
            "Overrun in pixels: "
            f"left={left_overrun:.1f}, bottom={bottom_overrun:.1f}, "
            f"right={right_overrun:.1f}, top={top_overrun:.1f}."
        )


def collect_distances_from_files(germfree_files, omm12_files):
    germfree_data = []
    omm12_data = []

    for germfree_file, omm12_file in zip(germfree_files, omm12_files):
        germfree_df = pd.read_csv(germfree_file, header=[0, 1, 2, 3, 4, 5], index_col=0)
        omm12_df = pd.read_csv(omm12_file, header=[0, 1, 2, 3, 4, 5], index_col=0)

        for individual in ["mouse_1", "mouse_2", "mouse_3"]:
            germfree_dist = germfree_df.loc[
                :, pd.IndexSlice[:, :, :, :, "mice_cumdists", individual]
            ].to_numpy()
            omm12_dist = omm12_df.loc[
                :, pd.IndexSlice[:, :, :, :, "mice_cumdists", individual]
            ].to_numpy()

            germfree_data.append(round(np.nanmax(germfree_dist) / PIXEL_PER_CM / 100, 2))
            omm12_data.append(round(np.nanmax(omm12_dist) / PIXEL_PER_CM / 100, 2))

    return np.array(germfree_data, dtype=float), np.array(omm12_data, dtype=float)


def print_comparison_stats(comparisons):
    for index, comparison in enumerate(comparisons, start=1):
        group1, group2 = comparison["values"]
        stat, p = mannwhitneyu(x=group1, y=group2)
        print(f"Comparison {index} p:", p)


def plot_unpaired_distance(ax, values, group_labels, ylabel, style, ylim=None):
    x_positions = np.arange(len(values))

    for group_index, group_values in enumerate(values):
        color = _value_for_index(style["scatter_colors"], group_index)
        ax.scatter(
            np.full(group_values.shape, x_positions[group_index]),
            group_values,
            s=style["scatter_size"],
            color=color,
            edgecolors="none",
            zorder=2,
        )

    ax.set_ylabel(ylabel, fontsize=style["label_font_size"])
    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_labels)
    ax.set_xlim(-0.45, len(values) - 0.55)

    if ylim is not None:
        ax.set_ylim(*ylim)

    despine_axis(ax)
    style_dark_axis(ax, style)


def build_fens_panel(comparisons=COMPARISONS, style=None, layout=None, plot_spacing_mm=None):
    if style is None:
        style = PLOT_STYLE
    if layout is None:
        layout = PLOT_LAYOUT
    if plot_spacing_mm is None:
        plot_spacing_mm = layout["plot_spacing_mm"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(mm_to_inches(PANEL_WIDTH_MM), mm_to_inches(PANEL_HEIGHT_MM)),
        facecolor="black",
    )
    apply_panel_layout(fig, plot_spacing_mm, layout)

    for ax, comparison in zip(axes, comparisons):
        plot_unpaired_distance(
            ax=ax,
            values=comparison["values"],
            group_labels=comparison["groups"],
            ylabel=comparison["ylabel"],
            style=style,
            ylim=comparison["ylim"],
        )

    return fig, axes


if __name__ == "__main__":
    if germfree_files and omm12_files:
        germfree_data, omm12_data = collect_distances_from_files(germfree_files, omm12_files)
        print(germfree_data.tolist(), omm12_data.tolist())
        print(np.mean(germfree_data), np.mean(omm12_data))

    print_comparison_stats(COMPARISONS)
    fig, axes = build_fens_panel()
    assert_panel_content_in_bounds(fig)
    plt.show()
    fig.savefig(OUTPUT_PATH)
