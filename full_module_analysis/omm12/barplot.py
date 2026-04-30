from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_barplot(
    data,
    colormode="group",
    plotsize=(8, 6),
    fontsize=12,
    colors=None,
    savepath=None,
    scatterdata=True,
    scattercolors=None,
    scattermarkers=None,
    stylemode="light",
):
    """
    Create one or multiple barplots from nested mean/SD data.

    Expected data shape:
        {
            "plot1": {
                "group1": {"mean": 5, "sd": 1, "values": [4, 5, 6]},
                "group2": {"mean": 7, "sd": 1.5, "values": [6, 7, 8]},
            },
            "plot2": {
                "group1": {"mean": 3, "sd": 0.5, "values": [2.5, 3, 3.5]},
                "group2": {"mean": 4, "sd": 0.8, "values": [3.2, 4, 4.8]},
            },
        }

    Parameters
    ----------
    data : dict
        Nested dictionary with plot names as first level and group names as
        second level. Each group needs a "mean" value and can optionally have
        "sd" and "values".
    colormode : {"group", "plot"}, default "group"
        If "group", equal group names use equal colors across subplots.
        If "plot", all bars inside one subplot use that subplot's color.
    plotsize : tuple, default (8, 6)
        Figure size passed to matplotlib.
    fontsize : int or float, default 12
        Base font size for labels, ticks, and titles.
    colors : dict or list, optional
        Colors for bars. Dictionaries can be keyed by group name or plot name,
        depending on colormode. Lists are cycled in order.
    savepath : str or Path, optional
        File path where the figure is saved. If omitted, only the figure/axes
        are returned.
    scatterdata : bool or dict, default True
        If True, overlay the "values" from data. If a dictionary is provided,
        it is read as scatterdata[plot_name][group_name].
    scattercolors : dict or list, optional
        Colors for scatter points. Defaults to matching the bar colors.
    scattermarkers : dict or list, optional
        Markers for scatter points. Dictionaries can be keyed by group or plot.
    stylemode : {"light", "dark"}, default "light"
        Controls figure background, axis, and text colors.

    Returns
    -------
    tuple
        (fig, axes), where axes is always a 1D numpy array.
    """
    if colormode not in ("group", "plot"):
        raise ValueError("colormode must be 'group' or 'plot'.")

    if stylemode not in ("light", "dark"):
        raise ValueError("stylemode must be 'light' or 'dark'.")

    if not data:
        raise ValueError("data must contain at least one plot.")

    plot_names = list(data.keys())
    group_names = _unique_group_names(data)

    fig, axes = plt.subplots(1, len(plot_names), figsize=plotsize, squeeze=False)
    axes = axes.ravel()

    facecolor = "#111111" if stylemode == "dark" else "white"
    textcolor = "white" if stylemode == "dark" else "black"
    gridcolor = "#444444" if stylemode == "dark" else "#dddddd"

    fig.patch.set_facecolor(facecolor)

    for plot_index, (ax, plot_name) in enumerate(zip(axes, plot_names)):
        groups = data[plot_name]
        x_positions = np.arange(len(groups))
        bar_colors = [
            _resolve_color(
                colors,
                group_name,
                plot_name,
                group_names.index(group_name),
                plot_index,
                colormode,
            )
            for group_name in groups
        ]

        means = [_as_float(values.get("mean", np.nan)) for values in groups.values()]
        sds = [_as_float(values.get("sd", 0)) for values in groups.values()]

        ax.bar(
            x_positions,
            means,
            yerr=sds,
            color=bar_colors,
            edgecolor=textcolor,
            linewidth=1,
            capsize=5,
            error_kw={"ecolor": textcolor, "elinewidth": 1.2, "capthick": 1.2},
        )

        if scatterdata:
            for group_index, (group_name, group_values) in enumerate(groups.items()):
                values = _get_scatter_values(scatterdata, plot_name, group_name, group_values)
                if values is None:
                    continue

                values = np.asarray(values, dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue

                offsets = _scatter_offsets(values.size)
                marker = _resolve_marker(
                    scattermarkers,
                    group_name,
                    plot_name,
                    group_index,
                    plot_index,
                    colormode,
                )
                scatter_color = _resolve_color(
                    scattercolors,
                    group_name,
                    plot_name,
                    group_index,
                    plot_index,
                    colormode,
                    fallback=bar_colors[group_index],
                )

                ax.scatter(
                    np.full(values.size, x_positions[group_index]) + offsets,
                    values,
                    color=scatter_color,
                    edgecolor=textcolor,
                    marker=marker,
                    s=45,
                    zorder=3,
                    alpha=0.9,
                )

        ax.set_title(str(plot_name), fontsize=fontsize + 2, color=textcolor)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(list(groups.keys()), rotation=30, ha="right", fontsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize, colors=textcolor)
        ax.tick_params(axis="x", colors=textcolor)
        ax.set_facecolor(facecolor)
        ax.yaxis.grid(True, color=gridcolor, linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_color(textcolor)

    fig.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")

    plt.show()

    return fig, axes


def _unique_group_names(data):
    group_names = []
    for groups in data.values():
        for group_name in groups:
            if group_name not in group_names:
                group_names.append(group_name)
    return group_names


def _resolve_color(
    colors,
    group_name,
    plot_name,
    group_index,
    plot_index,
    colormode,
    fallback=None,
):
    if colors is None:
        if fallback is not None:
            return fallback
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        index = group_index if colormode == "group" else plot_index
        return color_cycle[index % len(color_cycle)]

    if isinstance(colors, dict):
        key = group_name if colormode == "group" else plot_name
        return colors.get(key, fallback or _resolve_color(None, group_name, plot_name, group_index, plot_index, colormode))

    index = group_index if colormode == "group" else plot_index
    return colors[index % len(colors)]


def _resolve_marker(markers, group_name, plot_name, group_index, plot_index, colormode):
    default_markers = ["o", "^", "s", "D", "v", "P", "X"]

    if markers is None:
        index = group_index if colormode == "group" else plot_index
        return default_markers[index % len(default_markers)]

    if isinstance(markers, dict):
        key = group_name if colormode == "group" else plot_name
        return markers.get(key, default_markers[0])

    index = group_index if colormode == "group" else plot_index
    return markers[index % len(markers)]


def _get_scatter_values(scatterdata, plot_name, group_name, group_values):
    if isinstance(scatterdata, dict):
        return scatterdata.get(plot_name, {}).get(group_name)

    return group_values.get("values")


def _scatter_offsets(n_values):
    if n_values == 1:
        return np.array([0.0])
    return np.linspace(-0.12, 0.12, n_values)


def _as_float(value):
    if value is None:
        return np.nan
    return float(value)
