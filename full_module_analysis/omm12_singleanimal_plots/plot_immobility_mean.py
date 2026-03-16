import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

FPS = 30

df = pd.read_csv(
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\single_mouse_datatest_mice_immobile.csv",
    header=[0,1,2,3,4,5],
    index_col=0
)
df_presence = pd.read_csv(
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\single_mouse_datatest_mice_presence.csv",
    header=[0,1,2,3,4,5],
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

x_positions = {
    "hab": 0,
    "top1": 1,
    "top2": 2
}

sex_offsets = {
    "males": -0.08,
    "females": 0.08
}

for subgroup in subgroups:

    n_groups = len(subgroup)
    spread = 0.15

    group_offsets = np.linspace(-spread, spread, n_groups)

    offset_map = {
        grp: offset
        for grp, offset in zip(subgroup, group_offsets)
    }

    plt.figure(figsize=(6, 6))

    title = " vs. ".join(subgroup)

    # males oder females
    sexes_to_plot = ["females"]
    # Falls beide:
    #sexes_to_plot = ["males", "females"]

    safename = "immobility_" + "_".join(subgroup) + "_" + "_".join(sexes_to_plot) + ".jpg"

    for cond in conditions:
        if cond == "hab":
            n_frames = 54_000
        else:
            n_frames = 216_000

        for grp in subgroup:
            for sex in sexes_to_plot:
                d = df.loc[:n_frames-1, idx[grp, :, sex, cond, :]]
                d_presence = df_presence.loc[:n_frames-1, idx[grp, :, sex, cond, :]]

                # hier sammeln wir alle Werte dieser grp/sex/cond-Kombination
                amounts_this_group = []

                for j, mice_id in enumerate(d.columns.get_level_values("mouse_ids").unique()):
                    for i, ind in enumerate(individuals):

                        immobile = d.loc[:, idx[:, mice_id, :, :, "mice_immobile", ind]].values.ravel()
                        presence = d_presence.loc[:, idx[:, mice_id, :, :, "mice_presence", ind]].values.ravel()

                        # robust gegen komische gespeicherte Werte
                        immobile = np.where(immobile == 1, 1, 0)
                        presence = np.where(presence == 1, 1, 0)

                        immobile_total = np.nansum(immobile)
                        presence_total = np.nansum(presence)

                        min_presence_minutes = 5
                        min_presence_frames = min_presence_minutes * 60 * FPS

                        if presence_total < min_presence_frames:
                            amount = np.nan
                        elif presence_total == 0:
                            amount = np.nan
                        else:
                            amount = 100*immobile_total / presence_total

                        if len(sexes_to_plot) > 1:
                            x = x_positions[cond] + offset_map[grp] + sex_offsets[sex]
                        else:
                            x = x_positions[cond] + offset_map[grp]

                        plt.scatter(
                            x,
                            amount,
                            color=group_colors[grp],
                            marker=sex_layout_scatter[sex],
                            s=70,
                            alpha=0.9
                        )

                        if presence_total < min_presence_frames:
                            plt.text(x, 0, "low track", color="red", fontsize=7)

                        if not np.isnan(amount):
                            amounts_this_group.append(amount)

                        if len(sexes_to_plot) < 2 and not np.isnan(amount):
                            plt.text(
                                x + 0.1,
                                amount,
                                "m" + str(j+1) + "." + str(i+1),
                                color=group_colors[grp],
                                fontsize=8,
                                va="center",
                                ha="left"
                            )

                # Mean pro grp/sex/cond als roter Marker
                if len(amounts_this_group) > 0:
                    mean_amount = np.nanmean(amounts_this_group)

                    if len(sexes_to_plot) > 1:
                        x_mean = x_positions[cond] + offset_map[grp] + sex_offsets[sex]
                    else:
                        x_mean = x_positions[cond] + offset_map[grp]

                    plt.scatter(
                        x_mean,
                        mean_amount,
                        color="red",
                        marker=sex_layout_scatter[sex],
                        s=90,
                        edgecolor="black",
                        linewidth=1.2,
                        zorder=10
                    )

    plt.xticks(
        [x_positions["hab"], x_positions["top1"], x_positions["top2"]],
        ["hab", "top1", "top2"]
    )
    plt.xlim(-0.5, 2.5)
    plt.ylim(0, 100)
    plt.ylabel("Immobile time / Presence time [%]")
    plt.title(title)

    group_handles = [
        mlines.Line2D([], [], color=group_colors[group], marker='o', linestyle='None', markersize=8, label=group)
        for group in subgroup
    ]

    sex_handles = [
        mlines.Line2D([], [], color='black', marker=sex_layout_scatter[sex], linestyle='None', markersize=8, label=sex)
        for sex in sex_layout_scatter
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