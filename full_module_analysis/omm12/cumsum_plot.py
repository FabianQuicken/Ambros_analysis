from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_cumsum(
    data,
    colors=None,
    plotsize=None,
    fontsize=12,
    savepath=None,
    ylim=(0, 1),
    ylabel=None,
    stylemode="light",
    plot_individual_curves=False,
):
    """
    Plot cumulative-sum arrays from create_data_dic output.

    Expected data shape:
        {
            "condition1": {
                "group1": {"values": [array1, array2, ...]},
                "group2": {"values": [array1, array2, ...]},
            },
            "condition2": {...},
        }

    Each condition is plotted as one subplot below the previous one. By default,
    every group is plotted as the mean over arrays with the SD shown as a
    transparent area. Set plot_individual_curves=True to plot every array as a
    separate line without mean or SD.
    """
    if stylemode not in ("light", "dark"):
        raise ValueError("stylemode must be 'light' or 'dark'.")

    if not data:
        raise ValueError("data must contain at least one condition.")

    condition_names = list(data.keys())
    group_names = _unique_group_names(data)
    if plotsize is None:
        plotsize = (10, 5 * len(condition_names))

    fig, axes = plt.subplots(len(condition_names), 1, figsize=plotsize, squeeze=False)
    axes = axes.ravel()

    facecolor = "#111111" if stylemode == "dark" else "white"
    textcolor = "white" if stylemode == "dark" else "black"
    gridcolor = "#444444" if stylemode == "dark" else "#dddddd"

    fig.patch.set_facecolor(facecolor)

    for ax, condition in zip(axes, condition_names):
        condition_data = data[condition]

        for group_index, group in enumerate(condition_data):
            values = condition_data[group].get("values", [])
            stacked_values = _stack_arrays(values)
            if stacked_values.size == 0:
                continue

            mean_values = np.nanmean(stacked_values, axis=0)
            sd_values = np.nanstd(stacked_values, axis=0)
            x_values = np.arange(len(mean_values))
            color = _resolve_color(colors, group, group_index, group_names)

            if plot_individual_curves:
                for value_index, value in enumerate(stacked_values):
                    ax.plot(
                        np.arange(len(value)),
                        value,
                        color=color,
                        linewidth=1.5,
                        alpha=0.85,
                        label=group if value_index == 0 else None,
                    )
            else:
                ax.plot(
                    x_values,
                    mean_values,
                    color=color,
                    linewidth=3,
                    label=group,
                )
                ax.fill_between(
                    x_values,
                    mean_values - sd_values,
                    mean_values + sd_values,
                    color=color,
                    alpha=0.75,
                    linewidth=0,
                )

        ax.set_title(str(condition), fontsize=fontsize + 2, color=textcolor)
        ax.set_xlabel("frame", fontsize=fontsize, color=textcolor)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=fontsize, color=textcolor)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.tick_params(axis="both", labelsize=fontsize, colors=textcolor)
        ax.set_facecolor(facecolor)
        ax.yaxis.grid(True, color=gridcolor, linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        sns.despine()

        for spine in ax.spines.values():
            spine.set_color(textcolor)

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

    return fig, axes


def _stack_arrays(values):
    arrays = []
    for value in values:
        array = np.asarray(value, dtype=float).ravel()
        if array.size > 0:
            arrays.append(array)

    if not arrays:
        return np.empty((0, 0))

    max_length = max(len(array) for array in arrays)
    stacked_values = np.full((len(arrays), max_length), np.nan)

    for index, array in enumerate(arrays):
        stacked_values[index, : len(array)] = array

    return stacked_values


def _unique_group_names(data):
    group_names = []
    for condition_data in data.values():
        for group_name in condition_data:
            if group_name not in group_names:
                group_names.append(group_name)
    return group_names


def _resolve_color(colors, group, group_index, group_names):
    if colors is None:
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        return color_cycle[group_index % len(color_cycle)]

    if isinstance(colors, dict):
        return colors.get(group, _resolve_color(None, group, group_names.index(group), group_names))

    return colors[group_index % len(colors)]
