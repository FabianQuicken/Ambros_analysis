import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None




"""
Construct a Figure panel for a poster using matplotlib
The panel size is 371,333 mm * 100 mm
Three plots are generated: The plot is on dark background, therefore axis and text needs to be white
The plots (A, B, C) are placed in a 1x3 row of the panel size, each plot is equal sized
For each plot, I want to be able to adjust the following parameters:
- scatter point colors
- scatter point size
- color of lines connecting the scatter data points
- thickness of lines connecting the scatter data points
- x and y axis line thickness
- font size of x and y labels and ticks
- distance between plots
- tick length of x and y ticks
- top, bottom, left and right margins of the panel


"""


# data for A and B in investigation time [frames]

mouse_125 = {'day1': {'stim_dish': 619, 'con_dish': 496},
             'day2': {'stim_dish': 774, 'con_dish': 1053},
             'day3': {'stim_dish': 1970, 'con_dish': 763}}

"""
mouse_122 = {'day1': {'stim_dish': 1121, 'con_dish': 1964},
             'day2': {'stim_dish': 1180, 'con_dish': 1190},
            'day3': {'stim_dish': 2723, 'con_dish': 3753}}
"""
mouse_121 = {'day1': {'stim_dish': 17, 'con_dish': 516},
             'day2': {'stim_dish': 1282, 'con_dish': 1242},
             'day3': {'stim_dish': 1294, 'con_dish': 1019}}

mouse_109 = {'day1': {'stim_dish': 135, 'con_dish': 183},
             'day2': {'stim_dish': 770, 'con_dish': 1382},
             'day3': {'stim_dish': 640, 'con_dish': 391}}

mouse_36 = {'day1': {'stim_dish': 838, 'con_dish': 810},
             'day2': {'stim_dish': 679, 'con_dish': 1580},
             'day3': {'stim_dish': 1133, 'con_dish': 1085}}

mouse_38 = {'day1': {'stim_dish': 397, 'con_dish': 273},
             'day2': {'stim_dish': 647, 'con_dish': 593},
             'day3': {'stim_dish': 2258, 'con_dish': 287}}

mouse_135 = {'day1': {'stim_dish': 1080, 'con_dish': 460},
             'day2': {'stim_dish': 1810, 'con_dish': 916},
             'day3': {'stim_dish': 805, 'con_dish': 456}}

mouse_137 = {'day1': {'stim_dish': 836, 'con_dish': 269},
             'day2': {'stim_dish': 1116, 'con_dish': 909},
             'day3': {'stim_dish': 1397, 'con_dish': 1006}}

# Data for C 

m125_p_corrected = {'day2': 0.047, 'day3': 0.042}
#m122_p_corrected = {'day2': 0.132, 'day3': -0.104}
m121_p_corrected = {'day2': 0.038, 'day3': 0.049}
m109_p_corrected = {'day2': 0.027, 'day3': 0.128}

#m135_p_corrected = {'day2': 0.029, 'day3': -0.074}
m137_p_corrected = {'day2': -0.053, 'day3': 0.064}
m36_p_corrected = {'day2': -0.034, 'day3': 0.046}
m38_p_corrected = {'day2': -0.080, 'day3': 0.009}

"""
Plot A:
Investigation of stimulus dish
- one tick for each day on x axis, seconds on y axis
- scatter plot for each day
- connect data of single days with lines, that are in between days bot dont touch the scatter points
- same color for each scatter point and line ("white" as default)
- plot title: "stimulus dish investigation
- y axis label: "time [s]"
- x axis labels: "Habituation", "Conditioning", "Recall" (relates to day1, day2 and day3 in data)
"""

"""
Plot B:
Investigation of control dish
- one tick for each day on x axis, seconds on y axis
- scatter plot for each day
- connect data of single days with lines, that are in between days bot dont touch the scatter points
- same color for each scatter point and line ("white" as default)
- plot title: "control dish investigation
- y axis label: "time [s]"
- x axis labels: "Habituation", "Conditioning", "Recall" (relates to day1, day2 and day3 in data)
"""

"""
Plot C:
Baseline corrected module preference
- one tick for day2 and day3 on x axis, baseline corrected preference on y axis
- scatter plot for each day
- connect data of single days with lines, that are in between days bot dont touch the scatter points
- plot title: "stimulus module preference
- y axis label: "preference score"
- x axis labels: "Conditioning", "Recall" (relates to day2 and day3 in data)
- add a dotted line, horizontally on y = 0 with the same line thickness as the plot axes
"""


FPS = 30
MM_PER_INCH = 25.4

PANEL_WIDTH_MM = 371.333
PANEL_HEIGHT_MM = 100

DAYS = ("day1", "day2", "day3")
DAY_LABELS = ("Habituation", "Conditioning", "Recall")
PREFERENCE_DAYS = ("day2", "day3")
PREFERENCE_DAY_LABELS = ("Conditioning", "Recall")

MICE_DATA = {
    "109": mouse_109,
    "121": mouse_121,
#    "122": mouse_122,
    "125": mouse_125,
#    "135": mouse_135,
    "137": mouse_137,
    "36": mouse_36,
    "38": mouse_38,
}

PREFERENCE_DATA = {
    "109": m109_p_corrected,
    "121": m121_p_corrected,
#    "122": m122_p_corrected,
    "125": m125_p_corrected,
#    "135": m135_p_corrected,
    "137": m137_p_corrected,
    "36": m36_p_corrected,
    "38": m38_p_corrected,
}

PLOT_STYLE = {
    "scatter_colors": "white",
    "scatter_size": 30,
    "line_color": "white",
    "line_width": 2.835/2,
    "axis_line_width": 2.835,
    "tick_length": 2.835*2,
    "label_font_size": 16,
    "tick_font_size": 14,
    "title_font_size": 18,
    "panel_label_font_size": 20,
}

PLOT_LAYOUT = {
    "left_margin_mm": 20,
    "right_margin_mm": 0,
    "bottom_margin_mm": 8,
    "top_margin_mm": 8,
    "plot_spacing_mm": 24,
}


def mm_to_inches(value):
    return value / MM_PER_INCH


def mm_to_figure_fraction(value_mm, figure_size_mm):
    return value_mm / figure_size_mm


def dish_values_seconds(mice_data, metric_key, days=DAYS, fps=FPS):
    return np.array(
        [
            [mouse_data[day][metric_key] / fps for day in days]
            for mouse_data in mice_data.values()
        ],
        dtype=float,
    )


def preference_values(preference_data, days=PREFERENCE_DAYS):
    return np.array(
        [
            [mouse_data[day] for day in days]
            for mouse_data in preference_data.values()
        ],
        dtype=float,
    )


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


def despine_axis(ax):
    if sns is not None:
        sns.despine(ax=ax)
        return

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_dish_investigation(ax, mice_data, metric_key, title, style, ylim=100):
    x_values = np.arange(len(DAYS))
    values_seconds = dish_values_seconds(mice_data, metric_key)

    for mouse_index, y_values in enumerate(values_seconds):
        point_color = _value_for_index(style["scatter_colors"], mouse_index)
        line_color = _value_for_index(style["line_color"], mouse_index)
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

    ax.set_title(title, fontsize=style["title_font_size"], pad=7)
    #ax.set_xlabel("day", fontsize=style["label_font_size"])
    ax.set_ylabel("time [s]", fontsize=style["label_font_size"])
    ax.set_xticks(x_values)
    ax.set_xticklabels(DAY_LABELS)
    ax.set_xlim(-0.35, len(DAYS) - 0.65)
    ax.set_ylim(0, ylim)
    despine_axis(ax)
    style_dark_axis(ax, style)


def plot_module_preference(ax, preference_data, title, style, ylim=(-0.1, 0.2)):
    x_values = np.arange(len(PREFERENCE_DAYS))
    values = preference_values(preference_data)

    for mouse_index, y_values in enumerate(values):
        point_color = _value_for_index(style["scatter_colors"], mouse_index)
        line_color = _value_for_index(style["line_color"], mouse_index)
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

    ax.axhline(
        0,
        color="white",
        linestyle=":",
        linewidth=style["axis_line_width"],
        zorder=0,
    )
    ax.set_title(title, fontsize=style["title_font_size"], pad=7)
    ax.set_ylabel("preference score", fontsize=style["label_font_size"])
    ax.set_xticks(x_values)
    ax.set_xticklabels(PREFERENCE_DAY_LABELS)
    ax.set_xlim(-0.35, len(PREFERENCE_DAYS) - 0.65)

    if ylim is None:
        y_min = min(np.nanmin(values), 0)
        y_max = max(np.nanmax(values), 0)
        padding = max((y_max - y_min) * 0.12, 0.02)
        ax.set_ylim(y_min - padding, y_max + padding)
    else:
        ax.set_ylim(*ylim)

    despine_axis(ax)
    style_dark_axis(ax, style)


def add_panel_label(ax, label, style):
    ax.text(
        -0.18,
        1.08,
        label,
        transform=ax.transAxes,
        color="white",
        fontsize=style["panel_label_font_size"],
        fontweight="bold",
        va="top",
        ha="left",
    )


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
        raise ValueError(
            "Figure content extends outside the panel boundary. "
            "Increase the margins in PLOT_LAYOUT, reduce font sizes, or reduce plot_spacing_mm."
        )


def build_fens_panel(style=None, layout=None, plot_spacing_mm=None):
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

    plot_dish_investigation(
        axes[0],
        MICE_DATA,
        "stim_dish",
        "stimulus dish investigation",
        style,
    )
    plot_dish_investigation(
        axes[1],
        MICE_DATA,
        "con_dish",
        "control dish investigation",
        style,
    )
    plot_module_preference(
        axes[2],
        PREFERENCE_DATA,
        "stimulus module preference",
        style,
    )

    #for label, ax in zip(("A", "B", "C"), axes):
    #    add_panel_label(ax, label, style)

    return fig, axes


if __name__ == "__main__":
    fig, axes = build_fens_panel()
    assert_panel_content_in_bounds(fig)
    plt.show()
    fig.savefig(r"Z:\n2023_odor_related_behavior\other\Reisen\Barcelona FENS 2026\Poster\Abbildungen\3_chamber.svg")
