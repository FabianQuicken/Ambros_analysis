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


"""
Construct a Figure panel for a poster using matplotlib
The panel size is 191,854 mm * 90 mm
Three plots are generated: The plot is on dark background, therefore axis and text needs to be white
The plots (A, B, C) are placed in a 1x3 row of the panel size, each plot is equal sized
For each plot, I want to be able to adjust the following parameters:
- scatter point colors
- scatter point size
- x and y axis line thickness
- font size of x and y labels and ticks
- distance between plots
- tick length of x and y ticks
- top, bottom, left and right margins of the panel

Each metric in the list plot defines one of the panels
- each plot has a title (face investigation, body investigation, anogenital investigation)
- each plot ylim is (0, 100)
"""


path_males = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\FENS\data_social\males"
male_files = glob.glob(os.path.join(path_males, "*.csv"))

path_females = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\FENS\data\females"
female_files = glob.glob(os.path.join(path_females, "*.csv"))

individuals = ["mouse_1", "mouse_2", "mouse_3"]
plot = ["face_inv", "body_inv", "anogenital_inv"]
ylims = [40, 5, 5]

MM_PER_INCH = 25.4
PANEL_WIDTH_MM = 191.854
PANEL_HEIGHT_MM = 90

METRIC_TITLES = {
    "face_inv": "face",
    "body_inv": "body",
    "anogenital_inv": "anogenital",
}

PLOT_STYLE = {
    "scatter_colors": ("white", "0.55"),
    "scatter_size": 30,
    "axis_line_width": 2.835,
    "tick_length": 2.835 * 2,
    "label_font_size": 14,
    "tick_font_size": 14,
    "title_font_size": 14,
    "panel_label_font_size": 4,
    "panel_label_x": 0.02,
    "panel_label_y": 0.98,
}

PLOT_LAYOUT = {
    "left_margin_mm": 18,
    "right_margin_mm": 0,
    "bottom_margin_mm": 13,
    "top_margin_mm": 9,
    "plot_spacing_mm": 14,
}

OUTPUT_PATH = r"C:\Users\Fabian\Desktop\Transfer\FENS\social_behavior.svg"


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

    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")


def apply_panel_layout(fig, plot_spacing_mm, layout):
    left = mm_to_figure_fraction(layout["left_margin_mm"], PANEL_WIDTH_MM)
    right = 1 - mm_to_figure_fraction(layout["right_margin_mm"], PANEL_WIDTH_MM)
    bottom = mm_to_figure_fraction(layout["bottom_margin_mm"], PANEL_HEIGHT_MM)
    top = 1 - mm_to_figure_fraction(layout["top_margin_mm"], PANEL_HEIGHT_MM)

    inner_width_mm = PANEL_WIDTH_MM - layout["left_margin_mm"] - layout["right_margin_mm"]
    plot_width_mm = (inner_width_mm - plot_spacing_mm * 2) / 3
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


def metric_values(files, metric):
    values = []

    for file in files:
        df = pd.read_csv(file, header=[0, 1, 2, 3, 4, 5], index_col=0)

        metric_data = df.loc[:, pd.IndexSlice[:, :, :, :, metric, :]].to_numpy()
        sum_inds = np.nansum(metric_data, axis=1)
        total = np.nansum(sum_inds, axis=0)

        # Normalize to time with at least two mice, with three present mice counting double.
        mice_present = df.loc[:, pd.IndexSlice[:, :, :, :, "mice_presence", :]].to_numpy()
        sum_present = np.nansum(mice_present, axis=1)
        pair_presence = np.maximum(sum_present - 1, 0)
        pairtime = np.nansum(pair_presence)

        if pairtime < 1800:
            continue

        values.append(total / pairtime * 100)

    return np.array(values, dtype=float)


def collect_metric_data(metrics):
    metric_data = {}

    for metric in metrics:
        female_data = metric_values(female_files, metric)
        male_data = metric_values(male_files, metric)
        metric_data[metric] = {
            "female": female_data,
            "male": male_data,
        }

        print(female_data.tolist(), male_data.tolist())
        print(f"Is there a difference in {metric} between females and males?")
        stat, p = mannwhitneyu(x=female_data, y=male_data)
        print("p:", p)

    return metric_data


def plot_unpaired_metric(ax, female_data, male_data, title, style, ylim=(0, 100)):
    x_positions = np.array([0, 1])
    data_by_group = (female_data, male_data)

    for group_index, group_data in enumerate(data_by_group):
        color = _value_for_index(style["scatter_colors"], group_index)
        ax.scatter(
            np.full(group_data.shape, x_positions[group_index]),
            group_data,
            s=style["scatter_size"],
            color=color,
            edgecolors="none",
            zorder=2,
        )

    ax.set_title(title, fontsize=style["title_font_size"], pad=7)
    ax.set_ylabel("investigation [%]", fontsize=style["label_font_size"])
    ax.set_xticks(x_positions)
    ax.set_xticklabels(("female", "male"))
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(*ylim)
    despine_axis(ax)
    style_dark_axis(ax, style)


def add_panel_label(ax, label, style):
    ax.text(
        style["panel_label_x"],
        style["panel_label_y"],
        label,
        transform=ax.transAxes,
        color="white",
        fontsize=style["panel_label_font_size"],
        fontweight="bold",
        va="top",
        ha="left",
    )


def build_fens_panel(metric_data, style=None, layout=None, plot_spacing_mm=None):
    if style is None:
        style = PLOT_STYLE
    if layout is None:
        layout = PLOT_LAYOUT
    if plot_spacing_mm is None:
        plot_spacing_mm = layout["plot_spacing_mm"]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(mm_to_inches(PANEL_WIDTH_MM), mm_to_inches(PANEL_HEIGHT_MM)),
        facecolor="black",
    )
    apply_panel_layout(fig, plot_spacing_mm, layout)

    for ax, metric, ylim in zip(axes, plot, ylims):
        plot_unpaired_metric(
            ax=ax,
            female_data=metric_data[metric]["female"],
            male_data=metric_data[metric]["male"],
            title=METRIC_TITLES[metric],
            style=style,
            ylim=(0, ylim)
        )


    return fig, axes


if __name__ == "__main__":
    #data = collect_metric_data(plot)
    data = {"face_inv": {"female": np.array([14.827403171124276, 16.028268941433048, 13.380715876006166, 17.093881059056653, 13.523084217388728, 19.17972916779664]), "male": np.array([24.492208849675972, 26.272837136560746, 21.9343155447249, 35.036292573981015, 18.623129741237143])},
            "body_inv": {"female": np.array([3.165131331118797, 2.7723715267712947, 2.219558143517726, 3.702878424943697, 2.4476975048894687, 3.422465693984922]), "male": np.array([1.6140006469387527, 1.4075495841330774, 3.5025747650666608, 2.5590917550716545, 2.1228748763155525])},
            "anogenital_inv": {"female": np.array([2.171158521968426, 1.3071624773972697, 3.0142147628018496, 1.3870872758815525, 1.7987317015349966, 3.075337636274882]), "male": np.array([1.6140006469387527, 1.4075495841330774, 3.5025747650666608, 2.5590917550716545, 2.1228748763155525])}
            }
    fig, axes = build_fens_panel(data)
    assert_panel_content_in_bounds(fig)
    plt.show()
    fig.savefig(OUTPUT_PATH)
