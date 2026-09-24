"""Plot framewise visible-mouse counts for each condition and sex."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from matplotlib.patches import Patch

from prepare_data import create_data_dic


CSV_FOLDER = Path(r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest")
SAVE_FOLDER = CSV_FOLDER.parent / "Analysis4"
INDIVIDUALS = ["mouse_1", "mouse_2", "mouse_3"]
COLORS = {"germfree": "#D9D9D9", "germfreeprop": "#CCE6BB", "omm12": "#C0DEFC", "omm12prop": "#bef49d", "ommpgol": "#E58DF1"}


def visible_counts(traces):
    """Return recording x frame counts from raw_traces in groups of three.

    Missing samples stay NaN, rather than being counted as absent mice.
    Each consecutive triple contains the three mice from the same mouse_ids.
    """
    if not traces or len(traces) % 3:
        raise ValueError("Expected three mouse traces per recording.")
    recordings = []
    for start in range(0, len(traces), 3):
        mice = [np.asarray(trace, dtype=float) for trace in traces[start:start + 3]]
        mice = [trace[:, None] if trace.ndim == 1 else trace for trace in mice]
        if any(trace.ndim != 2 or trace.shape != mice[0].shape for trace in mice):
            raise ValueError("Mouse traces within a recording must have matching shapes.")
        values = np.stack(mice)
        if np.any(~(np.isnan(values) | (values == 0) | (values == 1))):
            raise ValueError("mice_presence must contain only 0, 1, or NaN.")
        recordings.extend(np.sum(values, axis=0).T)
    if not recordings or max(map(len, recordings)) == 0:
        raise ValueError("No presence samples found.")
    counts = np.full((len(recordings), max(map(len, recordings))), np.nan)
    for row, values in enumerate(recordings):
        counts[row, :len(values)] = values
    return counts


def plot_n_mice_visible(
    data, colors=COLORS, condition="hab", sex="female", fps=30,
    plotsize=(12, 6), fontsize=12, savepath=None, target_count=None,
):
    """One group band containing one strip per recording; white means zero.

    Intensity increases in four discrete steps to the group's usual color.
    Gray marks missing samples. Time is elapsed recording time in minutes.
    With target_count=1, 2, or 3, only that exact count is colored; all other
    observed counts are white, and missing samples remain gray.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and greater than zero.")
    if target_count is not None and target_count not in (1, 2, 3):
        raise ValueError("target_count must be None, 1, 2, or 3.")
    groups = list(colors)
    counts = {group: visible_counts(data[condition][group]["values"]) for group in groups}
    fig, ax = plt.subplots(figsize=plotsize)
    max_time = 0
    for row, group in enumerate(groups):
        base = np.asarray(to_rgb(colors[group]))
        shades = [1 - (1 - base) * level / 3 for level in range(4)]
        if target_count is not None:
            shades = [base if level == target_count else np.ones(3) for level in range(4)]
        cmap = ListedColormap(shades)
        cmap.set_bad("#888888")
        duration = counts[group].shape[1] / fps / 60
        max_time = max(max_time, duration)
        ax.imshow(
            counts[group], cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 4), 4),
            extent=(0, duration, row + 0.38, row - 0.38),
            aspect="auto", interpolation="nearest", rasterized=True,
        )
        ax.add_patch(plt.Rectangle((0, row - 0.38), duration, 0.76,
                                  fill=False, edgecolor="#aaaaaa", linewidth=0.5))
    ax.set(xlim=(0, max_time), ylim=(len(groups) - 0.5, -0.5),
           xlabel="Time (minutes)", ylabel="Group", title=f"Visible mice — {condition}, {sex}s")
    ax.set_yticks(range(len(groups)), labels=groups)
    if target_count is not None:
        animal_label = "mouse" if target_count == 1 else "mice"
        ax.set_title(f"Exactly {target_count} {animal_label} visible - {condition}, {sex}s")
    ax.tick_params(labelsize=fontsize)
    for label in (ax.xaxis.label, ax.yaxis.label, ax.title):
        label.set_fontsize(fontsize)
    handles = [Patch(facecolor=str(1 - level * 0.25), edgecolor="#aaaaaa", label=str(level))
               for level in range(4)]
    legend_title = "Visible mice (intensity within each group color)"
    if target_count is not None:
        handles = [
            Patch(facecolor="white", edgecolor="#aaaaaa", label="Other counts (including 0)"),
            Patch(facecolor="#666666", label=f"Exactly {target_count} (group color)"),
        ]
        legend_title = "Visible mice"
    handles.append(Patch(facecolor="#888888", label="Missing"))
    ax.legend(handles=handles, title=legend_title,
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5, frameon=False)
    fig.text(0.5, 0.01, "Each strip within a group band represents one recording.",
             ha="center", fontsize=fontsize - 2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight", facecolor="white")
    return fig, ax


def main():
    for sex in ("female", "male"):
        data = {}
        for group in COLORS:
            group_data = create_data_dic(
                CSV_FOLDER, INDIVIDUALS, sex, group, "mice_presence",
                data_extraction_mode="raw_traces",
            )
            for condition, values in group_data.items():
                data.setdefault(condition, {}).update(values)
        for condition in ("hab", "top1", "top2"):
            fig, _ = plot_n_mice_visible(
                data, condition=condition, sex=sex,
                savepath=SAVE_FOLDER / f"n_mice_visible_{condition}_{sex}s.pdf",
            )
            plt.close(fig)
            for target_count in (1, 2, 3):
                fig, _ = plot_n_mice_visible(
                    data, condition=condition, sex=sex, target_count=target_count,
                    savepath=SAVE_FOLDER / f"n_mice_visible_exactly_{target_count}_{condition}_{sex}s.pdf",
                )
                plt.close(fig)


if __name__ == "__main__":
    main()
