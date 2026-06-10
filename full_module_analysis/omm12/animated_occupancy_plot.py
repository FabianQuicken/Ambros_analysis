from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgb


def create_animated_occupancy_plot(
    x_coords,
    y_coords,
    rectangle_coords,
    binnumber,
    windowsize,
    normalizemode,
    original_image_size=None,
    experimentlength=None,
    savefolder=".",
    filename="animated_occupancy.mp4",
    barcolor="#4C72B0",
    stylemode="light",
    plotsize=(12, 6),
    edge_distance_coloring=True,
    fps=10,
    dpi=200,
    create_svg=False,
    svg_filename=None,
    svg_line_width=2,
    svg_raster_color="#ff0000",
    svg_font_size=16,
    svg_font_color="#ff0000",
):
    """
    Create an animated occupancy bar plot for rectangular spatial bins.

    Parameters
    ----------
    x_coords, y_coords : array-like
        One x/y coordinate per video frame.
    rectangle_coords : list of tuple
        Four (x, y) coordinates spanning an axis-aligned rectangle.
    binnumber : int
        Number of bins per rectangle side. The plot contains binnumber ** 2 bars.
    windowsize : int
        Animation step size in coordinate frames. Each animation frame shows
        cumulative occupancy from the beginning up to that timestamp.
    normalizemode : {"realtime", "fulltime"}
        "realtime" normalizes to the shown timestamp. "fulltime" normalizes to
        the complete experiment length, so bars add up to 100% at the end.
    original_image_size : tuple, optional
        Original image size in pixels as (width, height). Required when
        create_svg is True.
    experimentlength : int, optional
        Limits the x/y arrays to the first experimentlength frames.
    savefolder : str or Path
        Folder where the animation is saved.
    filename : str
        Output filename. Supported extensions are .mp4 and .gif.
    barcolor : matplotlib color, default "#4C72B0"
        Bar color used when edge_distance_coloring is False.
    stylemode : {"light", "dark"}, default "light"
        Plot background style.
    plotsize : tuple, default (12, 6)
        Matplotlib figure size.
    edge_distance_coloring : bool, default False
        If True, bins with equal distance to the rectangle edge share a color.
        Outer bins are brightest and inner bins become progressively darker.
    fps : int, default 10
        Playback frames per second for the saved animation.
    dpi : int, default 200
        Output resolution.
    create_svg : bool, default False
        If True, writes a transparent SVG bin overlay to savefolder.
    svg_filename : str, optional
        Output filename for the SVG overlay. Defaults to the animation filename
        stem plus "_bins.svg".
    svg_line_width : float, default 2
        Stroke width for the SVG bin raster.
    svg_raster_color : str, default "#ff0000"
        Stroke color for the SVG bin raster.
    svg_font_size : float, default 16
        Font size for the SVG bin labels.
    svg_font_color : str, default "#ff0000"
        Font color for the SVG bin labels.

    Returns
    -------
    tuple
        (fig, ax, anim, savepath)
    """
    if normalizemode not in ("realtime", "fulltime"):
        raise ValueError("normalizemode must be 'realtime' or 'fulltime'.")

    if stylemode not in ("light", "dark"):
        raise ValueError("stylemode must be 'light' or 'dark'.")

    if int(binnumber) != binnumber or binnumber < 1:
        raise ValueError("binnumber must be a positive integer.")

    if int(windowsize) != windowsize or windowsize < 1:
        raise ValueError("windowsize must be a positive integer.")

    binnumber = int(binnumber)
    windowsize = int(windowsize)

    x = np.asarray(x_coords, dtype=float)
    y = np.asarray(y_coords, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x_coords and y_coords must have the same shape.")

    if experimentlength is not None:
        if int(experimentlength) != experimentlength or experimentlength < 1:
            raise ValueError("experimentlength must be a positive integer.")
        experimentlength = min(int(experimentlength), x.size)
        x = x[:experimentlength]
        y = y[:experimentlength]

    if x.size == 0:
        raise ValueError("x_coords and y_coords must contain at least one point.")

    xmin, xmax, ymin, ymax = _rectangle_bounds(rectangle_coords)
    ordered_bins = _clockwise_bins_outside_to_inside(binnumber)
    bin_indices = _assign_points_to_bins(x, y, xmin, xmax, ymin, ymax, binnumber)

    n_bins = binnumber * binnumber
    cumulative_counts = np.zeros((x.size + 1, n_bins), dtype=int)
    valid_bins = bin_indices >= 0
    if np.any(valid_bins):
        frame_indices = np.flatnonzero(valid_bins) + 1
        np.add.at(cumulative_counts, (frame_indices, bin_indices[valid_bins]), 1)
    cumulative_counts = np.cumsum(cumulative_counts, axis=0)

    frame_ends = list(range(windowsize, x.size + 1, windowsize))
    if not frame_ends or frame_ends[-1] != x.size:
        frame_ends.append(x.size)

    total_in_bins = cumulative_counts[-1].sum()
    fulltime_denominator = total_in_bins if total_in_bins > 0 else x.size

    facecolor = "#111111" if stylemode == "dark" else "white"
    textcolor = "white" if stylemode == "dark" else "black"
    gridcolor = "#444444" if stylemode == "dark" else "#dddddd"

    fig, ax = plt.subplots(figsize=plotsize)
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    x_positions = np.arange(n_bins)
    colors = _edge_distance_colors(ordered_bins, binnumber, barcolor) if edge_distance_coloring else barcolor
    bars = ax.bar(x_positions, np.zeros(n_bins), color=colors, edgecolor=textcolor, linewidth=0.6)

    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, n_bins - 0.5)
    ax.set_ylabel("Occupancy (%)", color=textcolor)
    ax.set_xlabel("Bins ordered outside to inside", color=textcolor)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(i + 1) for i in range(n_bins)], rotation=90 if n_bins > 30 else 0)
    ax.tick_params(axis="both", colors=textcolor)
    ax.yaxis.grid(True, color=gridcolor, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(textcolor)

    title = ax.set_title("", color=textcolor)
    fig.tight_layout()

    def update(frame_end):
        ordered_counts = cumulative_counts[frame_end, ordered_bins]
        if normalizemode == "realtime":
            denominator = ordered_counts.sum()
        else:
            denominator = fulltime_denominator

        percentages = np.zeros_like(ordered_counts, dtype=float)
        if denominator > 0:
            percentages = ordered_counts / denominator * 100

        for bar, height in zip(bars, percentages):
            bar.set_height(height)

        title.set_text(f"Occupancy up to frame {frame_end} / {x.size}")
        return (*bars, title)

    anim = FuncAnimation(
        fig,
        update,
        frames=frame_ends,
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )

    savepath = Path(savefolder) / filename
    savepath.parent.mkdir(parents=True, exist_ok=True)
    suffix = savepath.suffix.lower()
    if suffix not in (".mp4", ".gif"):
        raise ValueError("Unsupported animation format. Use .mp4 or .gif.")

    if create_svg:
        if original_image_size is None:
            raise ValueError("original_image_size is required when create_svg is True.")
        if svg_filename is None:
            svg_filename = f"{savepath.stem}_bins.svg"
        create_occupancy_bins_svg(
            rectangle_coords=rectangle_coords,
            binnumber=binnumber,
            original_image_size=original_image_size,
            savepath=savepath.parent / svg_filename,
            line_width=svg_line_width,
            raster_color=svg_raster_color,
            font_size=svg_font_size,
            font_color=svg_font_color,
        )

    if suffix == ".mp4":
        anim.save(savepath, writer="ffmpeg", fps=fps, dpi=dpi)
    else:
        anim.save(savepath, writer="pillow", fps=fps, dpi=dpi)

    plt.close(fig)
    return fig, ax, anim, savepath


def create_occupancy_bins_svg(
    rectangle_coords,
    binnumber,
    original_image_size,
    savepath,
    line_width=2,
    raster_color="#ff0000",
    font_size=16,
    font_color="#ff0000",
):
    """
    Create a transparent SVG overlay showing the occupancy bin raster.

    The SVG canvas matches original_image_size. Bin geometry is derived from
    rectangle_coords and binnumber in the same way as the occupancy assignment.
    """
    if int(binnumber) != binnumber or binnumber < 1:
        raise ValueError("binnumber must be a positive integer.")
    binnumber = int(binnumber)

    image_width, image_height = _validate_original_image_size(original_image_size)
    xmin, xmax, ymin, ymax = _rectangle_bounds(rectangle_coords)

    line_width = float(line_width)
    font_size = float(font_size)
    if line_width < 0:
        raise ValueError("line_width must be non-negative.")
    if font_size < 0:
        raise ValueError("font_size must be non-negative.")

    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)

    bin_width = (xmax - xmin) / binnumber
    bin_height = (ymax - ymin) / binnumber
    map_y = _svg_y_mapper(ymin, ymax, image_height)
    escaped_raster_color = _escape_xml_attr(raster_color)
    escaped_font_color = _escape_xml_attr(font_color)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{image_width:g}" '
            f'height="{image_height:g}" viewBox="0 0 {image_width:g} {image_height:g}">'
        ),
        (
            f'  <g fill="none" stroke="{escaped_raster_color}" '
            f'stroke-width="{line_width:g}" vector-effect="non-scaling-stroke">'
        ),
    ]

    for col in range(binnumber + 1):
        x = xmin + col * bin_width
        lines.append(
            f'    <line x1="{x:g}" y1="{map_y(ymin):g}" '
            f'x2="{x:g}" y2="{map_y(ymax):g}" />'
        )
    for row in range(binnumber + 1):
        y = ymin + row * bin_height
        lines.append(f'    <line x1="{xmin:g}" y1="{map_y(y):g}" x2="{xmax:g}" y2="{map_y(y):g}" />')

    lines.extend(
        [
            "  </g>",
            (
                f'  <g fill="{escaped_font_color}" font-size="{font_size:g}" '
                'font-family="Arial, Helvetica, sans-serif" text-anchor="middle" '
                'dominant-baseline="central">'
            ),
        ]
    )

    ordered_bins = _clockwise_bins_outside_to_inside(binnumber)
    bin_numbers = np.empty(binnumber * binnumber, dtype=int)
    bin_numbers[ordered_bins] = np.arange(1, binnumber * binnumber + 1)
    for row in range(binnumber):
        for col in range(binnumber):
            bin_index = row * binnumber + col
            x = xmin + (col + 0.5) * bin_width
            y = ymin + (row + 0.5) * bin_height
            lines.append(f'    <text x="{x:g}" y="{map_y(y):g}">{bin_numbers[bin_index]}</text>')

    lines.extend(["  </g>", "</svg>", ""])
    savepath.write_text("\n".join(lines), encoding="utf-8")
    return savepath


def _rectangle_bounds(rectangle_coords):
    coords = np.asarray(rectangle_coords, dtype=float)
    if coords.shape != (4, 2):
        raise ValueError("rectangle_coords must be a list of four (x, y) tuples.")

    xmin = np.min(coords[:, 0])
    xmax = np.max(coords[:, 0])
    ymin = np.min(coords[:, 1])
    ymax = np.max(coords[:, 1])
    if xmin == xmax or ymin == ymax:
        raise ValueError("rectangle_coords must span a rectangle with non-zero width and height.")

    return xmin, xmax, ymin, ymax


def _validate_original_image_size(original_image_size):
    try:
        width, height = original_image_size
    except (TypeError, ValueError) as exc:
        raise ValueError("original_image_size must be a (width, height) tuple.") from exc

    width = float(width)
    height = float(height)
    if width <= 0 or height <= 0:
        raise ValueError("original_image_size values must be positive.")

    return width, height


def _svg_y_mapper(ymin, ymax, image_height):
    if ymin < 0 and ymax <= 0 and abs(ymin) <= image_height and abs(ymax) <= image_height:
        return lambda y: -y
    return lambda y: y


def _escape_xml_attr(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _assign_points_to_bins(x, y, xmin, xmax, ymin, ymax, binnumber):
    in_rectangle = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    bin_indices = np.full(x.shape, -1, dtype=int)

    x_fraction = (x[in_rectangle] - xmin) / (xmax - xmin)
    y_fraction = (y[in_rectangle] - ymin) / (ymax - ymin)
    x_bins = np.minimum((x_fraction * binnumber).astype(int), binnumber - 1)
    y_bins = np.minimum((y_fraction * binnumber).astype(int), binnumber - 1)

    bin_indices[in_rectangle] = y_bins * binnumber + x_bins
    return bin_indices


def _clockwise_bins_outside_to_inside(binnumber):
    ordered = []
    # rectangle contains multiple layers dependen on the binnumber:
    # a binnumber of 5 leads to 3 layers, e.g. three "rings" of bins 
    for layer in range((binnumber + 1) // 2):
        top = layer
        bottom = binnumber - layer - 1
        left = layer
        right = binnumber - layer - 1

        for col in range(left, right + 1):
            ordered.append(top * binnumber + col)

        for row in range(top + 1, bottom + 1):
            ordered.append(row * binnumber + right)

        if bottom > top:
            for col in range(right - 1, left - 1, -1):
                ordered.append(bottom * binnumber + col)

        if right > left:
            for row in range(bottom - 1, top, -1):
                ordered.append(row * binnumber + left)

    return np.asarray(ordered, dtype=int)


def _edge_distance_colors(ordered_bins, binnumber, basecolor):
    rgb = np.asarray(to_rgb(basecolor))
    colors = []
    max_distance = max(1, (binnumber - 1) // 2)
    for bin_index in ordered_bins:
        row, col = divmod(int(bin_index), binnumber)
        distance = min(row, col, binnumber - 1 - row, binnumber - 1 - col)
        brightness = 1.0 - 0.55 * (distance / max_distance)
        colors.append(tuple(np.clip(rgb * brightness, 0, 1)))
    return colors
