from scipy.stats import wilcoxon
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

try:
    import seaborn as sns
except ImportError:
    sns = None

"""
Construct a Figure panel for a poster using matplotlib
The panel size is 371,333 mm * 200 mm
Fourteen plots are generated: The plot is on dark background, therefore axis and text needs to be white
This is the panel structure:
- 2 main rows of similar size, each row has 4 columns. Column 1 has the width of 2/5th of the panel, column 2-4 each have the width of 1/5th of the panel
- row1 column1 contains 4 heatmaps in 2x2 grid, column2 contains 1 scatter plot, column3 contains 1 scatter plot, column4 contains 1 scatter plot
- row2 column1 contains 4 heatmaps in 2x2 grid, column2 contains 1 scatter plot, column3 contains 1 scatter plot, column4 contains 1 scatter plot

For each plot, I want to be able to adjust the following parameters:
- scatter point colors
- scatter point size
- color of lines connecting the scatter data points
- thickness of lines connecting the scatter data points
- x and y axis line thickness
- font size of all text (ticks, labels, etc)
- distance between plots
- tick length of x and y ticks
- top, bottom, left and right margins of the panel
"""

MM_PER_INCH = 25.4
PANEL_WIDTH_MM = 411.433
PANEL_HEIGHT_MM = 205.235

DEFAULT_HEATMAP_BINS = (25, 15)

PROJECT_PATH = r"Z:\n2023_odor_related_behavior"

OUTPUT_PATH = r"Z:\n2023_odor_related_behavior\other\Reisen\Paris ECRO 2026\Poster\odor_preference.svg"

PLOT_STYLE = {
    "scatter_colors": "white",
    "scatter_size": 30,
    "line_color": "white",
    "line_width": 2.835 / 2,
    "axis_line_width": 2.835,
    "tick_length": 2.835 * 2,
    "font_size": 14,
    "label_font_size": 16,
    "title_font_size": 14,
    "heatmap_title_font_size": 14,
}

PLOT_LAYOUT = {
    "left_margin_mm": 0,
    "right_margin_mm": 0,
    "bottom_margin_mm": 8,
    "top_margin_mm": 5,
    "column_spacing_mm": 24,
    "row_spacing_mm": 24,
    "heatmap_spacing_mm": 4,
}


def module_path(folder, mouse, date, module):
    return os.path.join(PROJECT_PATH, folder, mouse, date, module)


# # # female data # # #

# heatmap data stim
# mouse 17 day2 top1 vs day1 top1
m_17_d2_stim_path = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\female_mice_male_stimuli\mouse_17\2025_04_15\top1\dlc_files"
m_17_d1_stim_path = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\female_mice_male_stimuli\mouse_17\2025_04_14\top1\dlc_files"

# heatmap data con
# mouse 17 day2 top2 vs day1 top2
m_17_d2_con_path = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\female_mice_male_stimuli\mouse_17\2025_04_15\top2\dlc_files"
m_17_d1_con_path = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\female_mice_male_stimuli\mouse_17\2025_04_14\top2\dlc_files"

# discrimination data stim: (exp1 stim - hab stim) / (exp1 stim + hab stim) and (exp2 stim - recall stim) / (exp2 stim + recall stim)
# and con (exp1 con - hab con) / (exp1 con + hab con) and (exp2 con - recall con) / (exp2 con + recall con)
f_stim_disc = [0.96, 0.17, 0.40, 0.53, 0.27, 0.49, 0.38, -0.06, 0.91, 0.24, 0.27, 0.28, 0.53, 0.09, -0.27, 0.33, 0.82, 0.09]
f_con_disc = [0.89, 0.12, -0.10, -0.15, -0.15, -0.08, -0.01, 0.19, 0.77, -0.10, 0.10, -0.50, 0.22, 0.23, -0.99, -0.07, 0.62, -0.11]

# mean visit time data
f_stim_mean_visit_time = [12.9, 20.5, 22.8, 19, 16.3, 33.9, 21.9, 17.4, 24.4, 30.1, 19.6, 19.9, 34.3, 25.3, 11.9, 17, 16.1, 27.8]
f_con_mean_visit_time = [14.1, 22.8, 12.5, 12.4, 14.3, 23.4, 14.3, 32.7, 17.2, 25.6, 12.8, 17.4, 2.1, 19.8, 9.4, 9, 13.4, 13.9]

# number of visits data
f_stim_n_visits = [388, 291, 285, 388, 362, 274, 267, 223, 139, 155, 312, 237, 60, 227, 351, 321, 372, 241]
f_con_n_visits = [389, 194, 256, 271, 129, 165, 200, 212, 263, 63, 262, 264, 3, 255, 382, 290, 175, 146]


# # # male data # # #

# heatmap data stim
# mouse 73 day2 top1 vs day1 top1
m_73_d2_stim_path = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\male_mice_female_stimuli\mouse_73\2025_04_07\top1"
m_73_d1_stim_path = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\male_mice_female_stimuli\mouse_73\2025_04_06\top1"

# heatmap data con
# mouse 73 day2 top2 vs day1 top2
m_73_d2_con_path = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\male_mice_female_stimuli\mouse_73\2025_04_07\top2"
m_73_d1_con_path = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\male_mice_female_stimuli\mouse_73\2025_04_06\top2"

# discrimination data stim: (exp1 stim - hab stim) / (exp1 stim + hab stim) and (exp2 stim - recall stim) / (exp2 stim + recall stim)
# and con (exp1 con - hab con) / (exp1 con + hab con) and (exp2 con - recall con) / (exp2 con + recall con)
m_stim_disc = [0.73, 0.09, 0.58, 0.20, 0.15, 0.30, 0.00, 0.17, 0.27, 0.06, 0.77, 0.27, 0.97, 0.93]
m_con_disc = [-0.16, 0.20, 0.72, -0.15, 0.15, 0.26, 0.31, -0.05, -0.02, 0.11, 0.59, -0.03, 0.50, 0.48]

# mean visit time data
m_stim_mean_visit_time = [50.17, 62.67, 44.24, 54.97, 41.09, 45.18, 23.77, 24.61, 40.1, 32.3, 28.1, 25.4, 27.7, 39.9]
m_con_mean_visit_time = [20.87, 24.63, 24.11, 29.53, 22.72, 34.67, 24.79, 19.88, 37.8, 38.9, 15.8, 23, 4.6, 7.4]

# number of visits data
m_stim_n_visits = [154, 136, 187, 117, 170, 167, 205, 269, 147, 147, 252, 210, 214, 97]
m_con_n_visits = [134, 207, 149, 99, 158, 176, 233, 228, 129, 161, 149, 176, 94, 67]


FEMALE_HEATMAPS = [
    (m_17_d1_con_path, "baseline preference"),
    (m_17_d2_con_path, "control preference"),
    (m_17_d1_stim_path, "baseline preference"),
    (m_17_d2_stim_path, "stimulus preference"),
]

MALE_HEATMAPS = [
    (m_73_d1_con_path, "baseline preference"),
    (m_73_d2_con_path, "control preference"),
    (m_73_d1_stim_path, "baseline preference"),
    (m_73_d2_stim_path, "stimulus preference"),
]


def mm_to_inches(value):
    return value / MM_PER_INCH


def mm_to_figure_fraction(value_mm, figure_size_mm):
    return value_mm / figure_size_mm


def get_heatmap_data(path, likelihood_threshold=0.6):
    file_list = glob.glob(os.path.join(path, "*.csv"))
    file_list.sort()
    if not file_list:
        raise FileNotFoundError(f"No CSV files found in {path}")

    dfs = []
    for file in tqdm(file_list):
        df = pd.read_csv(file, header=[0, 1, 2])
        lh = df.loc[:, (slice(None), "nose", "likelihood")].to_numpy().ravel()
        df.loc[lh < likelihood_threshold, (slice(None), "nose", ["x", "y", "likelihood"])] = np.nan
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def extract_nose_coordinates(df):
    x_values = df.loc[:, (slice(None), "nose", "x")].to_numpy().ravel()
    y_values = df.loc[:, (slice(None), "nose", "y")].to_numpy().ravel()
    valid = np.isfinite(x_values) & np.isfinite(y_values) & (x_values != 0) & (y_values != 0)
    return x_values[valid], y_values[valid]


def heatmap_counts(x_values, y_values, bins, heatmap_range):
    if len(x_values) == 0:
        return np.zeros(bins)

    return np.histogram2d(
        x_values,
        y_values,
        bins=bins,
        range=heatmap_range,
    )[0]


def load_heatmap_group(heatmap_specs, bins=DEFAULT_HEATMAP_BINS):
    coordinate_sets = []
    for path, title in heatmap_specs:
        df = get_heatmap_data(path)
        x_values, y_values = extract_nose_coordinates(df)
        coordinate_sets.append((x_values, y_values, title))

    all_x = np.concatenate([x_values for x_values, _y_values, _title in coordinate_sets])
    all_y = np.concatenate([y_values for _x_values, y_values, _title in coordinate_sets])
    if len(all_x) == 0 or len(all_y) == 0:
        heatmap_range = [[0, 1], [0, 1]]
    else:
        heatmap_range = [[np.nanmin(all_x), np.nanmax(all_x)], [np.nanmin(all_y), np.nanmax(all_y)]]
        if heatmap_range[0][0] == heatmap_range[0][1]:
            heatmap_range[0][1] += 1
        if heatmap_range[1][0] == heatmap_range[1][1]:
            heatmap_range[1][1] += 1

    heatmaps = []
    extent = [
        heatmap_range[0][0],
        heatmap_range[0][1],
        heatmap_range[1][0],
        heatmap_range[1][1],
    ]
    for x_values, y_values, title in coordinate_sets:
        heatmap = heatmap_counts(x_values, y_values, bins=bins, heatmap_range=heatmap_range)
        heatmaps.append((heatmap, title, extent))
    return heatmaps


def _value_for_index(value, index):
    if isinstance(value, (list, tuple, np.ndarray)):
        return value[index % len(value)]
    return value


def draw_shortened_line(ax, x_values, y_values, color, line_width, gap=0.08):
    for start_index in range(len(x_values) - 1):
        x0 = x_values[start_index]
        x1 = x_values[start_index + 1]
        y0 = y_values[start_index]
        y1 = y_values[start_index + 1]
        dx = x1 - x0
        ax.plot(
            [x0 + gap * dx, x1 - gap * dx],
            [y0 + gap * (y1 - y0), y1 - gap * (y1 - y0)],
            color=color,
            linewidth=line_width,
            solid_capstyle="round",
            zorder=1,
        )


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
        labelsize=style["font_size"],
        width=style["axis_line_width"],
        length=style["tick_length"],
    )

    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_linewidth(style["axis_line_width"])

    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")


def plot_paired_scatter(ax, stim_values, con_values, y_label, style, ylim=None):
    values = np.column_stack([stim_values, con_values])
    x_values = np.array([0, 1])

    for animal_index, y_values in enumerate(values):
        point_color = _value_for_index(style["scatter_colors"], animal_index)
        line_color = _value_for_index(style["line_color"], animal_index)
        draw_shortened_line(
            ax=ax,
            x_values=x_values,
            y_values=y_values,
            color=line_color,
            line_width=style["line_width"],
        )
        ax.scatter(
            x_values,
            y_values,
            s=style["scatter_size"],
            color=point_color,
            edgecolors="none",
            zorder=2,
        )

    ax.set_ylabel(y_label, fontsize=style["label_font_size"])
    ax.set_xticks(x_values)
    ax.set_xticklabels(("Stimulus", "Control"))
    ax.set_xlim(-0.35, 1.35)
    if ylim is None:
        y_min = np.nanmin(values)
        y_max = np.nanmax(values)
        padding = max((y_max - y_min) * 0.12, 1 if y_max > 10 else 0.05)
        ax.set_ylim(y_min - padding, y_max + padding)
    else:
        ax.set_ylim(*ylim)

    despine_axis(ax)
    style_dark_axis(ax, style)


def plot_heatmap_grid(fig, parent_spec, heatmaps, style, layout):
    inner = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=parent_spec,
        wspace=layout["heatmap_spacing_mm"] / 50,
        hspace=layout["heatmap_spacing_mm"] / 30,
    )
    max_count = max(np.nanmax(heatmap) for heatmap, _title, _extent in heatmaps)
    if max_count <= 0:
        max_count = 1

    axes = []
    for index, (heatmap, title, extent) in enumerate(heatmaps):
        ax = fig.add_subplot(inner[index // 2, index % 2])
        ax.imshow(
            heatmap.T,
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=max_count,
            extent=extent,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=style["heatmap_title_font_size"], pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("black")
        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_linewidth(style["axis_line_width"] / 2)
        ax.title.set_color("white")
        axes.append(ax)

    return axes


def print_wilcoxon_results():
    stats = [
        ("female stimulus vs control preference score", f_stim_disc, f_con_disc),
        ("female stimulus vs control mean visit time", f_stim_mean_visit_time, f_con_mean_visit_time),
        ("female stimulus vs control number of visits", f_stim_n_visits, f_con_n_visits),
        ("male stimulus vs control preference score", m_stim_disc, m_con_disc),
        ("male stimulus vs control mean visit time", m_stim_mean_visit_time, m_con_mean_visit_time),
        ("male stimulus vs control number of visits", m_stim_n_visits, m_con_n_visits),
    ]
    for label, stim, con in stats:
        stat, p = wilcoxon(x=stim, y=con, alternative="greater")
        print(f"{label}: W={stat:.3f}, p={p:.4g}")


def apply_panel_layout(fig, layout):
    left = mm_to_figure_fraction(layout["left_margin_mm"], PANEL_WIDTH_MM)
    right = 1 - mm_to_figure_fraction(layout["right_margin_mm"], PANEL_WIDTH_MM)
    bottom = mm_to_figure_fraction(layout["bottom_margin_mm"], PANEL_HEIGHT_MM)
    top = 1 - mm_to_figure_fraction(layout["top_margin_mm"], PANEL_HEIGHT_MM)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


def build_fens_panel(style=None, layout=None, heatmap_bins=DEFAULT_HEATMAP_BINS):
    if style is None:
        style = PLOT_STYLE
    if layout is None:
        layout = PLOT_LAYOUT

    fig = plt.figure(
        figsize=(mm_to_inches(PANEL_WIDTH_MM), mm_to_inches(PANEL_HEIGHT_MM)),
        facecolor="black",
    )
    apply_panel_layout(fig, layout)

    outer = gridspec.GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[2, 1, 1, 1],
        height_ratios=[1, 1],
        wspace=layout["column_spacing_mm"] / 50,
        hspace=layout["row_spacing_mm"] / 80,
    )

    female_heatmaps = load_heatmap_group(FEMALE_HEATMAPS, bins=heatmap_bins)
    male_heatmaps = load_heatmap_group(MALE_HEATMAPS, bins=heatmap_bins)

    plot_heatmap_grid(fig, outer[0, 0], female_heatmaps, style, layout)
    plot_heatmap_grid(fig, outer[1, 0], male_heatmaps, style, layout)

    female_axes = [fig.add_subplot(outer[0, col]) for col in range(1, 4)]
    male_axes = [fig.add_subplot(outer[1, col]) for col in range(1, 4)]

    plot_paired_scatter(
        female_axes[0],
        f_stim_disc,
        f_con_disc,
        "preference score",
        style,
        ylim=(-1.1, 1.1),
    )
    plot_paired_scatter(
        female_axes[1],
        f_stim_mean_visit_time,
        f_con_mean_visit_time,
        "mean visit length [s]",
        style,
    )
    plot_paired_scatter(
        female_axes[2],
        f_stim_n_visits,
        f_con_n_visits,
        "number of visits [n]",
        style,
    )

    plot_paired_scatter(
        male_axes[0],
        m_stim_disc,
        m_con_disc,
        "preference score",
        style,
        ylim=(-1.1, 1.1),
    )
    plot_paired_scatter(
        male_axes[1],
        m_stim_mean_visit_time,
        m_con_mean_visit_time,
        "mean visit length [s]",
        style,
    )
    plot_paired_scatter(
        male_axes[2],
        m_stim_n_visits,
        m_con_n_visits,
        "number of visits [n]",
        style,
    )

    return fig
"Control habituation"

if __name__ == "__main__":
    print_wilcoxon_results()
    fig = build_fens_panel()
    if OUTPUT_PATH:
        fig.savefig(OUTPUT_PATH, format="svg", facecolor=fig.get_facecolor())
    plt.show()

