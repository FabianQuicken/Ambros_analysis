from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


HAB_MAX_DATAPOINTS = 54000


def plot_cumsum(
    data,
    colors=None,
    figsize=None,
    plotsize=None,
    fontsize=12,
    savepath=None,
    ylim=(0, 1),
    ylabel=None,
    xlabel=None,
    fps=None,
    x_time_unit="frames",
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
    separate line without mean or SD. For the "hab" condition, only the first
    54000 datapoints are plotted.

    Parameters
    ----------
    figsize : tuple, optional
        Figure size in inches, passed to matplotlib. If omitted, the previous
        plotsize argument is used for backwards compatibility.
    xlabel : str, optional
        Label for the x-axis. If omitted, the label is inferred from
        x_time_unit.
    fps : float, optional
        Frames per second. If given, x-values are converted from frames to
        seconds or minutes depending on x_time_unit.
    x_time_unit : {"frames", "seconds", "minutes"}, optional
        Unit for the x-axis. Use "seconds" or "minutes" together with fps to
        rescale the x-axis from frame number to time.
    """
    if stylemode not in ("light", "dark"):
        raise ValueError("stylemode must be 'light' or 'dark'.")

    if x_time_unit not in ("frames", "seconds", "minutes"):
        raise ValueError("x_time_unit must be 'frames', 'seconds', or 'minutes'.")

    if x_time_unit in ("seconds", "minutes") and fps is None:
        raise ValueError("fps must be provided when x_time_unit is 'seconds' or 'minutes'.")

    if fps is not None and fps <= 0:
        raise ValueError("fps must be greater than 0.")

    if not data:
        raise ValueError("data must contain at least one condition.")

    condition_names = list(data.keys())
    group_names = _unique_group_names(data)
    if figsize is not None and plotsize is not None:
        raise ValueError("Use either figsize or plotsize, not both.")
    if figsize is None:
        figsize = plotsize
    if figsize is None:
        figsize = (10, 5 * len(condition_names))

    fig, axes = plt.subplots(len(condition_names), 1, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    facecolor = "#111111" if stylemode == "dark" else "white"
    textcolor = "white" if stylemode == "dark" else "black"
    gridcolor = "#444444" if stylemode == "dark" else "#dddddd"

    fig.patch.set_facecolor(facecolor)
    x_axis_label = xlabel
    if x_axis_label is None:
        x_axis_label = {
            "frames": "frame",
            "seconds": "Seconds",
            "minutes": "Minutes",
        }[x_time_unit]

    for ax, condition in zip(axes, condition_names):

        condition_data = data[condition]


        for group_index, group in enumerate(condition_data):
            values = condition_data[group].get("values", [])
            stacked_values = _stack_arrays(values)
            if condition == "hab":
                stacked_values = stacked_values[:, :HAB_MAX_DATAPOINTS]
            if stacked_values.size == 0:
                continue

            mean_values = np.nanmean(stacked_values, axis=0)
            sd_values = np.nanstd(stacked_values, axis=0)
            x_values = _x_values(len(mean_values), fps, x_time_unit)
            color = _resolve_color(colors, group, group_index, group_names)

            if plot_individual_curves:
                for value_index, value in enumerate(stacked_values):
                    ax.plot(
                        _x_values(len(value), fps, x_time_unit),
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
        ax.set_xlabel(x_axis_label, fontsize=fontsize, color=textcolor)
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


def _x_values(length, fps, x_time_unit):
    x_values = np.arange(length)
    if x_time_unit == "seconds":
        return x_values / fps
    if x_time_unit == "minutes":
        return x_values / fps / 60
    return x_values


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
