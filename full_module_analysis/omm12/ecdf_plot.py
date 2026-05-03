from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_ecdf(
    data,
    colors=None,
    markers=None,
    stylemode="light",
    savepath=None,
    condition="top2",
    plotsize=(8, 6),
    fontsize=12,
    xlabel=None,
    ylabel="ECDF",
    xlim=None,
    ylim=(0, 1),
    linewidth=2,
):
    """
    Plot ECDF curves from create_data_dic-style data.

    Expected data shape:
        {
            "condition1": {
                "group1": {"mean": 5, "sd": 1, "values": [4, 5, 6]},
                "group2": {"mean": 7, "sd": 1.5, "values": [6, 7, 8]},
            },
            "condition2": {...},
        }

    Only one condition is plotted. Each group in that condition is drawn as one
    ECDF curve based on its finite "values".
    """
    if stylemode not in ("light", "dark"):
        raise ValueError("stylemode must be 'light' or 'dark'.")

    if not data:
        raise ValueError("data must contain at least one condition.")

    if condition not in data:
        raise ValueError(f"condition {condition!r} is missing from data.")

    groups = data[condition]
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
        values = _clean_values(_get_values(group_values))
        if values.size == 0:
            continue

        x_values, y_values = _ecdf_values(values)
        color = _resolve_color(colors, group_name, group_index)
        marker = _resolve_marker(markers, group_name, group_index)

        ax.step(
            x_values,
            y_values,
            where="post",
            color=color,
            linewidth=linewidth,
            marker=marker,
            markersize=4,
            label=group_name,
        )
        plotted_groups.append(group_name)

    ax.set_title(str(condition), fontsize=fontsize + 2, color=textcolor)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize, color=textcolor)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, color=textcolor)

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


def _ecdf_values(values):
    x_values = np.sort(values)
    y_values = np.arange(1, x_values.size + 1) / x_values.size
    return x_values, y_values


def _resolve_color(colors, group_name, group_index):
    if colors is None:
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        return color_cycle[group_index % len(color_cycle)]

    if isinstance(colors, dict):
        return colors.get(group_name, _resolve_color(None, group_name, group_index))

    return colors[group_index % len(colors)]


def _resolve_marker(markers, group_name, group_index):
    default_markers = [None, None, None, None, None, None, None]

    if markers is None:
        return default_markers[group_index % len(default_markers)]

    if isinstance(markers, dict):
        return markers.get(group_name, default_markers[group_index % len(default_markers)])

    return markers[group_index % len(markers)]
