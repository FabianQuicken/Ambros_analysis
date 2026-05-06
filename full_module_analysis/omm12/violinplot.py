from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_violinplot(
    data,
    colormode="group",
    plotsize=(8, 6),
    fontsize=12,
    colors=None,
    savepath=None,
    scatterdata=True,
    scattercolors=None,
    scattermarkers=None,
    ylim=None,
    ylabel=None,
    stylemode="light",
    showmeans=True,
    showmedians=True,
):
    """
    Create one or multiple violin plots from create_data_dic-style data.

    Expected data shape:
        {
            "plot1": {
                "group1": {"mean": 5, "sd": 1, "values": [4, 5, 6]},
                "group2": {"mean": 7, "sd": 1.5, "values": [6, 7, 8]},
            },
            "plot2": {...},
        }
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
        x_positions = np.arange(1, len(groups) + 1)
        values_by_group = [
            _clean_values(_get_values(group_values))
            for group_values in groups.values()
        ]
        violin_colors = [
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

        non_empty_positions = []
        non_empty_values = []
        non_empty_colors = []
        for position, values, color in zip(x_positions, values_by_group, violin_colors):
            if values.size > 0:
                non_empty_positions.append(position)
                non_empty_values.append(values)
                non_empty_colors.append(color)

        if non_empty_values:
            violin_parts = ax.violinplot(
                non_empty_values,
                positions=non_empty_positions,
                widths=0.75,
                showmeans=showmeans,
                showmedians=showmedians,
                showextrema=False,
            )
            _style_violin_parts(violin_parts, non_empty_colors, textcolor)

        if scatterdata:
            for group_index, (group_name, group_values) in enumerate(groups.items()):
                values = _clean_values(_get_scatter_values(scatterdata, plot_name, group_name, group_values))
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
                    fallback=violin_colors[group_index],
                )

                ax.scatter(
                    np.full(values.size, x_positions[group_index]) + offsets,
                    values,
                    color=scatter_color,
                    edgecolor=textcolor,
                    marker=marker,
                    s=35,
                    zorder=3,
                    alpha=0.9,
                )

        ax.set_title(str(plot_name), fontsize=fontsize + 2, color=textcolor)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(list(groups.keys()), rotation=30, ha="right", fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize, colors=textcolor)
        ax.set_facecolor(facecolor)
        ax.yaxis.grid(True, color=gridcolor, linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        sns.despine()

        if ylim is not None:
            ax.set_ylim(ylim)
        elif any(values.size > 0 for values in values_by_group):
            all_values = np.concatenate([values for values in values_by_group if values.size > 0])
            top = np.nanmax(all_values) * 1.15 if np.nanmax(all_values) > 0 else 1
            ax.set_ylim(bottom=0, top=top)

        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=fontsize, color=textcolor)

        for spine in ax.spines.values():
            spine.set_color(textcolor)

    fig.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")

    return fig, axes


def _style_violin_parts(violin_parts, colors, textcolor):
    for body, color in zip(violin_parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(textcolor)
        body.set_alpha(0.75)
        body.set_linewidth(1)

    for key in ("cmeans", "cmedians"):
        if key in violin_parts:
            violin_parts[key].set_color(textcolor)
            violin_parts[key].set_linewidth(1.5)


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
        return colors.get(
            key,
            fallback
            or _resolve_color(None, group_name, plot_name, group_index, plot_index, colormode),
        )

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


def _get_values(group_values):
    return group_values.get("values", [])


def _get_scatter_values(scatterdata, plot_name, group_name, group_values):
    if isinstance(scatterdata, dict):
        return scatterdata.get(plot_name, {}).get(group_name, [])

    return group_values.get("values", [])


def _clean_values(values):
    if values is None:
        return np.array([])

    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def _scatter_offsets(n_values):
    if n_values == 1:
        return np.array([0.0])
    return np.linspace(-0.12, 0.12, n_values)
