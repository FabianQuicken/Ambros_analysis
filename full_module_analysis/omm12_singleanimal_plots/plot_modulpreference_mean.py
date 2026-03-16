import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

FPS = 30

df = pd.read_csv(
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\single_mouse_datatest_mice_presence.csv",
    header=[0, 1, 2, 3, 4, 5],
    index_col=0
)

idx = pd.IndexSlice
n_frames = 216_000

group_colors = {
    "germfree": "black",
    "germfreeprop": "brown",
    "omm12": "darkorange",
    "omm12prop": "forestgreen",
    "ommpgol": "darkviolet"
}

sex_layout_scatter = {
    "males": "o",
    "females": "^"
}

sex_layout_line = {
    "males": "solid",
    "females": "dotted"
}

groups = df.columns.get_level_values("group").unique()
ids = df.columns.get_level_values("mouse_ids").unique()
sexes = df.columns.get_level_values("sex").unique()
conditions = df.columns.get_level_values("condition").unique()
metrics = df.columns.get_level_values("metric").unique()
individuals = df.columns.get_level_values("individual").unique()

subgroup1 = ["germfree", "omm12"]
subgroup2 = ["germfree", "germfreeprop"]
subgroup3 = ["omm12", "omm12prop"]
subgroup4 = ["omm12", "ommpgol"]

subgroups = [subgroup1, subgroup2, subgroup3, subgroup4]

for subgroup in subgroups:

    plt.figure(figsize=(12, 6))
    safename = "modulpref_" + "_".join(subgroup) + ".jpg"
    title = " vs. ".join(subgroup)

    for grp in subgroup:
        for sex in sexes:
            d = df.loc[:n_frames-1, idx[grp, :, sex, :, :, :]]

            all_data = []

            for mice_id in d.columns.get_level_values("mouse_ids").unique():
                for ind in individuals:

                    top1 = d.loc[:, idx[:, mice_id, :, "top1", "mice_presence", ind]].values.ravel()
                    top2 = d.loc[:, idx[:, mice_id, :, "top2", "mice_presence", ind]].values.ravel()

                    denom = np.nansum(top1) + np.nansum(top2)

                    if denom == 0:
                        data = np.full(n_frames, np.nan, dtype=float)
                    else:
                        data = (np.nancumsum(top1) - np.nancumsum(top2)) / denom

                    all_data.append(data)

            if len(all_data) > 0:
                all_data = np.vstack(all_data)

                mean_data = np.nanmean(all_data, axis=0)
                valid_n = np.sum(~np.isnan(all_data), axis=0)
                sem_data = np.nanstd(all_data, axis=0) / np.sqrt(valid_n)

                time_minutes = np.arange(n_frames) / (FPS * 60)
                color = group_colors[grp]
                layout = sex_layout_line[sex]

                plt.plot(
                    time_minutes,
                    mean_data,
                    color=color,
                    linestyle=layout,
                    linewidth=2.5
                )

                plt.fill_between(
                    time_minutes,
                    mean_data - sem_data,
                    mean_data + sem_data,
                    color=color,
                    alpha=0.2
                )

    plt.xlabel("min")
    plt.ylabel("Preference (top1-top2 / top1+top2)")
    plt.ylim(-1, 1)
    plt.xlim(0, n_frames / (FPS * 60))
    plt.title(title)

    group_handles = [
        mlines.Line2D([], [], color=group_colors[group], linestyle='solid', label=group)
        for group in subgroup
    ]

    sex_handles = [
        mlines.Line2D([], [], color='black', linestyle=sex_layout_line[sex], label=sex)
        for sex in sex_layout_line
    ]

    plt.legend(handles=group_handles + sex_handles)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis3\single_mice",
            safename
        ),
        dpi=300
    )
    # plt.show()
    plt.close()