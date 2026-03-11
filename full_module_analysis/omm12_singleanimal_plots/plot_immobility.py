import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

FPS = 30

df = pd.read_csv(r"Z:\n2023_odor_related_behavior\2025_omm_mice\single_mouse_datatest_mice_immobile.csv", header=[0,1,2,3,4,5], index_col=0)
df_presence = pd.read_csv(r"Z:\n2023_odor_related_behavior\2025_omm_mice\single_mouse_datatest_mice_presence.csv", header=[0,1,2,3,4,5], index_col=0)

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
    "females":"dotted"
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
    safename = "immobility_" + "_".join(subgroup) + ".jpg"

    for cond in conditions:
        if cond == "hab":
            n_frames = 54_000
        else:
            n_frames = 216_000

        for grp in subgroup:
            for sex in sexes:
                d = df.loc[:n_frames-1, idx[grp, :, sex, cond, :]]
                d_presence = df_presence.loc[:n_frames-1, idx[grp, :, sex, cond, :]]

                for j, mice_id in enumerate(d.columns.get_level_values("mouse_ids").unique()):

                    for i, ind in enumerate(individuals):

                        immobile = d.loc[:, idx[:, mice_id, :, :, "mice_immobile", ind]].values.ravel()
                        presence = d_presence.loc[:, idx[:, mice_id, :, :, "mice_presence", ind]].values.ravel()

                        # es gibt in den Daten bei einigen Spalten einen einzelnen Wert -9*10^18 anstatt einer 0 oder 1, vermutlich durch das speichern
                        # als joblib und dann wieder einlesen und dann als speichern als csv - dieser Wert wird einmal raus gefiltert
                        immobile = np.where((immobile) == 1, 1, 0)
                        presence = np.where(presence == 1, 1, 0)


                        # total social investigation
                        immobile_total = np.nansum(immobile)
                        presence_total = np.nansum(presence)

                        

                        amount = immobile_total / presence_total

                        print(cond, grp, mice_id, ind, sex, immobile_total, presence_total)

                        x = x_positions[cond] + offset_map[grp]

                        plt.scatter(
                            x,
                            amount,
                            color=group_colors[grp],
                            marker=sex_layout_scatter[sex],
                            s=70,
                            alpha=0.9
                        )

                        plt.text(
                                x + 0.1,   # kleiner Offset nach rechts
                                amount,
                                "m"+str(j+1)+"."+str(i+1),
                                color=group_colors[grp],
                                fontsize=8,
                                va="center",
                                ha="left"
                            )

    plt.xticks(
        [x_positions["hab"], x_positions["top1"], x_positions["top2"]],
        ["hab", "top1", "top2"]
    )
    plt.xlim(-0.5, 2.5)
    plt.ylim(0, 1)
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
    plt.savefig(os.path.join(r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis3\single_mice", safename), dpi=300)
    #plt.show()
    plt.close()