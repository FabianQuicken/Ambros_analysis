from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rasterplot(
    data,
    names,
    fps,
    x_time_unit="frames",
    color=None,
    stylemode="dark",
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
    Plot event-index arrays as one raster row per name.

    Each array contains the frame indices of its events. Event positions are
    shown as frames, seconds, or minutes according to x_time_unit.
    """
    if stylemode not in ("light", "dark"):
        raise ValueError("stylemode must be 'light' or 'dark'.")

    if x_time_unit not in ("frames", "seconds", "minutes"):
        raise ValueError("x_time_unit must be 'frames', 'seconds', or 'minutes'.")

    if x_time_unit in ("seconds", "minutes") and fps is None:
        raise ValueError("fps must be provided when x_time_unit is 'seconds' or 'minutes'.")

    if fps is not None and fps <= 0:
        raise ValueError("fps must be greater than 0.")

    if len(data) != len(names):
        raise ValueError("data and names must have the same length.")

    if not data:
        raise ValueError("data must contain at least one event array.")

    facecolor = "#111111" if stylemode == "dark" else "white"
    textcolor = "white" if stylemode == "dark" else "black"
    gridcolor = "#444444" if stylemode == "dark" else "#dddddd"

    event_positions = []
    for values in data:
        positions = np.asarray(values).ravel()
        event_positions.append(_x_values(positions, fps, x_time_unit))

    fig, ax = plt.subplots(1, 1, figsize=plotsize)
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    eventcolor = textcolor if color is None else color
    ax.eventplot(event_positions, color=eventcolor, linelengths=0.8)
    ax.set_yticks(np.arange(len(names)), labels=names, fontsize=fontsize)
    ax.set_title(str(condition), fontsize=fontsize + 2, color=textcolor)

    if xlabel is None:
        xlabel = {
            "frames": "Frame",
            "seconds": "Seconds",
            "minutes": "Minutes",
        }[x_time_unit]
    ax.set_xlabel(xlabel, fontsize=fontsize, color=textcolor)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, color=textcolor)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(axis="both", labelsize=fontsize, colors=textcolor)
    ax.xaxis.grid(True, color=gridcolor, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(textcolor)

    fig.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")

    return fig, ax


def _x_values(values, fps, x_time_unit):
    if x_time_unit == "seconds":
        return values / fps
    if x_time_unit == "minutes":
        return values / fps / 60
    return values
