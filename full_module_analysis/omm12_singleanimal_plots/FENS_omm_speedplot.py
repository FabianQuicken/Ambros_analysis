import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    sns = None


PIXEL_PER_CM = 36.39
FPS = 30
MM_PER_INCH = 25.4

path_speedtrace = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\FENS\data_speed\single_mouse_datatest_omm12prop_32_35_37_females_top1.csv"
full_hab_path = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\FENS\data_speed\single_mouse_datatest_omm12prop_32_35_37_females_hab.csv"

OUTPUT_PATH = r"C:\Users\Fabian\Desktop\Transfer\FENS\speed.svg"

EXAMPLE_INDIVIDUAL = "mouse_2"
EXAMPLE_START_FRAME = 155000
EXAMPLE_STOP_FRAME = 156300
SPEED_EVENTS = np.array(
    [
        155203 - EXAMPLE_START_FRAME,
        155205 - EXAMPLE_START_FRAME,
        155282 - EXAMPLE_START_FRAME,
        155319 - EXAMPLE_START_FRAME,
        155462 - EXAMPLE_START_FRAME,
        155614 - EXAMPLE_START_FRAME,
        155931 - EXAMPLE_START_FRAME,
        156031 - EXAMPLE_START_FRAME,
        156170 - EXAMPLE_START_FRAME,
        157421 - EXAMPLE_START_FRAME,
    ],
    dtype=int,
)

"""

Row 1 Column 1: Speed trace example
- y axis in cm/s
- x axis in frames
Row 2 Column 1: Acceleration trace, each speed event is highlighted with a magenta square at the top y position of the graph at the respective x position
- y axis in cm/s^2
- x axis in frames
"""

"""
Row 1 Column 2: Full speed trace
- y axis in cm/s
- x axis in minutes (FPS is 30, use it to transform the x values)

Row 2 Column 2:
- Full acceleration trace
- y axis in cm/s^2
- x axis in minutes (FPS is 30, use it to transform the x values)
"""

"""
Construct a Figure panel for a poster using matplotlib
The panel size is 371,333 mm * 110 mm
4 plots are generated: The plot is on dark background, therefore axis and text needs to be white
The plots are placed in a 2x2 row of the panel size, each plot is equal sized
For each plot, I want to be able to adjust the following parameters:
- line thickness of the plot
- x and y axis line thickness
- font size of x and y labels and ticks
- distance between plots
- tick length of x and y ticks
- top, bottom, left and right margins of the panel


- no titles
- no panel letters A or B
"""

PANEL_WIDTH_MM = 371.333
PANEL_HEIGHT_MM = 110

PLOT_STYLE = {
    "trace_color": "white",
    "event_color": "magenta",
    "event_marker_size": 36,
    "line_width": 1.4,
    "axis_line_width": 2.835,
    "tick_length": 2.835 * 2,
    "label_font_size": 14,
    "tick_font_size": 14,
}

PLOT_LAYOUT = {
    "left_margin_mm": 30,
    "right_margin_mm": 7,
    "bottom_margin_mm": 15,
    "top_margin_mm": 7,
    "horizontal_spacing_mm": 22,
    "vertical_spacing_mm": 16,
}


def mm_to_inches(value):
    return value / MM_PER_INCH


def mm_to_figure_fraction(value_mm, figure_size_mm):
    return value_mm / figure_size_mm


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


def apply_panel_layout(fig, layout):
    left = mm_to_figure_fraction(layout["left_margin_mm"], PANEL_WIDTH_MM)
    right = 1 - mm_to_figure_fraction(layout["right_margin_mm"], PANEL_WIDTH_MM)
    bottom = mm_to_figure_fraction(layout["bottom_margin_mm"], PANEL_HEIGHT_MM)
    top = 1 - mm_to_figure_fraction(layout["top_margin_mm"], PANEL_HEIGHT_MM)

    inner_width_mm = PANEL_WIDTH_MM - layout["left_margin_mm"] - layout["right_margin_mm"]
    inner_height_mm = PANEL_HEIGHT_MM - layout["top_margin_mm"] - layout["bottom_margin_mm"]
    plot_width_mm = (inner_width_mm - layout["horizontal_spacing_mm"]) / 2
    plot_height_mm = (inner_height_mm - layout["vertical_spacing_mm"]) / 2

    if plot_width_mm <= 0:
        raise ValueError("horizontal_spacing_mm is too large for the selected panel width and margins.")
    if plot_height_mm <= 0:
        raise ValueError("vertical_spacing_mm is too large for the selected panel height and margins.")

    wspace = layout["horizontal_spacing_mm"] / plot_width_mm
    hspace = layout["vertical_spacing_mm"] / plot_height_mm
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)


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
            "Increase the margins in PLOT_LAYOUT, reduce font sizes, or reduce plot spacing. "
            "Overrun in pixels: "
            f"left={left_overrun:.1f}, bottom={bottom_overrun:.1f}, "
            f"right={right_overrun:.1f}, top={top_overrun:.1f}."
        )


def _metric_array(df, metric, individual):
    return df.loc[:, pd.IndexSlice[:, :, :, :, metric, individual]].to_numpy().squeeze()


def load_example_trace(
    csv_path,
    individual=EXAMPLE_INDIVIDUAL,
    start_frame=EXAMPLE_START_FRAME,
    stop_frame=EXAMPLE_STOP_FRAME,
):
    df = pd.read_csv(csv_path, header=[0, 1, 2, 3, 4, 5], index_col=0)
    speed = _metric_array(df, "mice_distances", individual) / PIXEL_PER_CM * FPS
    acceleration = _metric_array(df, "mice_accelerations", individual)

    return {
        "speed": speed[start_frame:stop_frame],
        "acceleration": acceleration[start_frame:stop_frame],
        "events": SPEED_EVENTS,
    }


def load_full_speed_trace(csv_path, moving_average_window=60):
    df = pd.read_csv(csv_path, header=[0, 1, 2, 3, 4, 5], index_col=0)

    speed = df.loc[:, pd.IndexSlice[:, :, :, :, "mice_distances", :]].to_numpy()
    speed = np.nansum(speed, axis=1) / PIXEL_PER_CM * FPS
    speed = speed.squeeze()

    if moving_average_window is not None and moving_average_window > 1:
        kernel = np.ones(moving_average_window) / moving_average_window
        speed = np.convolve(speed, kernel, mode="same")

    return speed


def load_full_acceleration_trace(csv_path, moving_average_window=None):
    df = pd.read_csv(csv_path, header=[0, 1, 2, 3, 4, 5], index_col=0)

    acceleration = df.loc[:, pd.IndexSlice[:, :, :, :, "mice_accelerations", :]].to_numpy()
    acceleration = np.nansum(acceleration, axis=1).squeeze()

    if moving_average_window is not None and moving_average_window > 1:
        kernel = np.ones(moving_average_window) / moving_average_window
        acceleration = np.convolve(acceleration, kernel, mode="same")

    return acceleration


def plot_trace(ax, y_values, x_values, ylabel, xlabel, style):
    ax.plot(
        x_values,
        y_values,
        color=style["trace_color"],
        linewidth=style["line_width"],
    )
    ax.set_ylabel(ylabel, fontsize=style["label_font_size"])
    ax.set_xlabel(xlabel, fontsize=style["label_font_size"])
    despine_axis(ax)
    style_dark_axis(ax, style)


def plot_acceleration_trace(ax, acceleration, events, style):
    x_values = np.arange(acceleration.size)
    plot_trace(
        ax=ax,
        y_values=acceleration,
        x_values=x_values,
        ylabel="acceleration [cm/s2]",
        xlabel="frames",
        style=style,
    )

    valid_events = events[(events >= 0) & (events < acceleration.size)]
    if valid_events.size == 0:
        return

    y_min, y_max = ax.get_ylim()
    event_y = y_max - 0.04 * (y_max - y_min)
    ax.scatter(
        valid_events,
        np.full(valid_events.shape, event_y),
        marker="s",
        s=style["event_marker_size"],
        color=style["event_color"],
        edgecolors="none",
        zorder=3,
    )


def plot_blank_axis(ax):
    ax.set_facecolor("black")
    ax.axis("off")


def build_fens_panel(
    example_trace,
    full_speed_trace=None,
    full_acceleration_trace=None,
    style=None,
    layout=None,
):
    if style is None:
        style = PLOT_STYLE
    if layout is None:
        layout = PLOT_LAYOUT

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(mm_to_inches(PANEL_WIDTH_MM), mm_to_inches(PANEL_HEIGHT_MM)),
        facecolor="black",
    )
    apply_panel_layout(fig, layout)

    example_x = np.arange(example_trace["speed"].size)
    plot_trace(
        ax=axes[0, 0],
        y_values=example_trace["speed"],
        x_values=example_x,
        ylabel="speed [cm/s]",
        xlabel="frames",
        style=style,
    )
    plot_acceleration_trace(
        ax=axes[1, 0],
        acceleration=example_trace["acceleration"],
        events=example_trace["events"],
        style=style,
    )

    if full_speed_trace is None:
        plot_blank_axis(axes[0, 1])
    else:
        minutes = np.arange(full_speed_trace.size) / FPS / 60
        plot_trace(
            ax=axes[0, 1],
            y_values=full_speed_trace,
            x_values=minutes,
            ylabel="speed [cm/s]",
            xlabel="time [min]",
            style=style,
        )

    if full_acceleration_trace is None:
        plot_blank_axis(axes[1, 1])
    else:
        minutes = np.arange(full_acceleration_trace.size) / FPS / 60
        plot_trace(
            ax=axes[1, 1],
            y_values=full_acceleration_trace,
            x_values=minutes,
            ylabel="acceleration [cm/s2]",
            xlabel="time [min]",
            style=style,
        )

    return fig, axes


if __name__ == "__main__":
    example_trace = load_example_trace(path_speedtrace)
    full_speed_trace = None
    full_acceleration_trace = None

    if full_hab_path and os.path.exists(full_hab_path):
        full_speed_trace = load_full_speed_trace(full_hab_path)
        full_acceleration_trace = load_full_acceleration_trace(full_hab_path)

    fig, axes = build_fens_panel(example_trace, full_speed_trace, full_acceleration_trace)
    assert_panel_content_in_bounds(fig)
    plt.show()
    fig.savefig(OUTPUT_PATH)
