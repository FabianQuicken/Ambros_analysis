
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def visitplot(
    ydata,
    xdata,
    y_logtransform=True,
    x_transform=1,
    colors=None,
    markers=None,
    stylemode="light",
    savepath=None,
    ylabel=None,
    condition="top2",
    plotsize=(8, 6),
    fontsize=12,
    xlabel=None,
    xlim=None,
    ylim=None,
):
    """
    Expected data shape:
        {
            "condition1": {
                "group1": {"mean": 5, "sd": 1, "values": [4, 5, 6]},
                "group2": {"mean": 7, "sd": 1.5, "values": [6, 7, 8]},
            },
            "condition2": {...},
        }

    ydata contains the y values and xdata contains the matching x positions.
    Only one condition is plotted. Each finite x/y pair is drawn as one dot.
    """
    if stylemode not in ("light", "dark"):
        raise ValueError("stylemode must be 'light' or 'dark'.")

    if not ydata:
        raise ValueError("ydata must contain at least one condition.")

    if not xdata:
        raise ValueError("xdata must contain at least one condition.")

    if condition not in ydata:
        raise ValueError(f"condition {condition!r} is missing from ydata.")

    if condition not in xdata:
        raise ValueError(f"condition {condition!r} is missing from xdata.")

    groups = ydata[condition]
    x_groups = xdata[condition]
    if not groups:
        raise ValueError(f"condition {condition!r} must contain at least one group.")

    fig, ax = plt.subplots(1, 1, figsize=plotsize)

    facecolor = "#111111" if stylemode == "dark" else "white"
    textcolor = "white" if stylemode == "dark" else "black"
    gridcolor = "#444444" if stylemode == "dark" else "#dddddd"

    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    plotted_groups = []
    for group_index, (group_name, group_values) in enumerate(groups.items()):
        if group_name not in x_groups:
            raise ValueError(f"group {group_name!r} is missing from xdata[{condition!r}].")

        x_values = _clean_values(_get_values(x_groups[group_name])) * x_transform
        y_values = _clean_values(_get_values(group_values))
        x_values, y_values = _paired_values(x_values, y_values)

        if y_logtransform:
            x_values, y_values = _log10_y_values(x_values, y_values)

        if x_values.size == 0:
            continue

        color = _resolve_color(colors, group_name, group_index)
        marker = _resolve_marker(markers, group_name, group_index)

        ax.scatter(
            x_values,
            y_values,
            color=color,
            edgecolor=textcolor,
            marker=marker,
            s=45,
            alpha=0.9,
            label=group_name,
            zorder=3,
        )
        plotted_groups.append(group_name)

    ax.set_title(str(condition), fontsize=fontsize + 2, color=textcolor)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize, color=textcolor)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, color=textcolor)
    elif y_logtransform:
        ax.set_ylabel("log10(values)", fontsize=fontsize, color=textcolor)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(axis="both", labelsize=fontsize, colors=textcolor)
    ax.yaxis.grid(True, color=gridcolor, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.xaxis.grid(True, color=gridcolor, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(textcolor)

    if plotted_groups:
        legend = ax.legend(fontsize=fontsize)
        if legend is not None:
            legend.get_frame().set_facecolor(facecolor)
            legend.get_frame().set_edgecolor(textcolor)
            for text in legend.get_texts():
                text.set_color(textcolor)

    fig.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")

    return fig, ax


def _get_values(group_values):
    if isinstance(group_values, dict):
        return group_values.get("values", [])

    return group_values


def _clean_values(values):
    if values is None:
        return np.array([])

    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def _paired_values(x_values, y_values):
    pair_count = min(x_values.size, y_values.size)
    if pair_count == 0:
        return np.array([]), np.array([])

    return x_values[:pair_count], y_values[:pair_count]


def _log10_y_values(x_values, y_values):
    positive_mask = y_values > 0
    return x_values[positive_mask], np.log10(y_values[positive_mask]+1)


def _resolve_color(colors, group_name, group_index):
    if colors is None:
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        return color_cycle[group_index % len(color_cycle)]

    if isinstance(colors, dict):
        return colors.get(group_name, _resolve_color(None, group_name, group_index))

    return colors[group_index % len(colors)]


def _resolve_marker(markers, group_name, group_index):
    default_markers = ["o", "^", "s", "D", "v", "P", "X"]

    if markers is None:
        return default_markers[group_index % len(default_markers)]

    if isinstance(markers, dict):
        return markers.get(group_name, default_markers[group_index % len(default_markers)])

    return markers[group_index % len(markers)]
