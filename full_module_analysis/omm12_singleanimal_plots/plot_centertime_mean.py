import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
from tqdm import tqdm

FPS = 30

#df = pd.read_csv(
#    r"Z:\n2023_odor_related_behavior\2025_omm_mice\single_mouse_datatest_mice_in_center.csv",
#    header=[0,1,2,3,4,5],
#    index_col=0
#)
df = pd.read_csv(r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\single_mouse_datatest_mice_in_center.csv", header=[0,1,2,3,4,5], index_col=0)
df_presence = pd.read_csv(r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\single_mouse_datatest_mice_presence.csv", header=[0,1,2,3,4,5], index_col=0)
#df_presence = pd.read_csv(
#    r"Z:\n2023_odor_related_behavior\2025_omm_mice\single_mouse_datatest_mice_presence.csv",
#    header=[0,1,2,3,4,5],
#    index_col=0
#)

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

# -----------------------------------
# Einstellungen
# -----------------------------------
plot_minutes = 120              # z.B. nur erste 5 Minuten plotten
center_occupancy = False
center_preference = True        # falls True: data = 2 * occupancy - 1

if center_occupancy and center_preference:
    raise ValueError("Bitte nur einen Modus aktivieren: center_occupancy ODER center_preference.")

for subgroup in subgroups:

    for cond in tqdm(conditions):
        if cond == "hab":
            n_frames = 54_000
        else:
            n_frames = 216_000

        plt.figure(figsize=(12, 6))
        safename = "centerpref_" + cond + "_"
        title = cond + " "

        for grp_idx, grp in enumerate(subgroup):
            title += grp
            safename += grp + "_"
            if grp_idx < len(subgroup) - 1:
                title += " vs. "

            for sex in ["males"]:
                d = df.loc[:n_frames-1, idx[grp, :, sex, cond, :]]
                d_presence = df_presence.loc[:n_frames-1, idx[grp, :, sex, cond, :]]

                all_data = []

                for mice_id in d.columns.get_level_values("mouse_ids").unique():
                    for ind in individuals:

                        centertime = d.loc[:, idx[:, mice_id, :, :, "mice_in_center", ind]].values.ravel()
                        presence = d_presence.loc[:, idx[:, mice_id, :, :, "mice_presence", ind]].values.ravel()

                        # robust gegen kaputte Werte aus CSV
                        centertime = np.where(centertime == 1, 1, 0)
                        presence = np.where(presence == 1, 1, 0)

                        center_cum = np.nancumsum(centertime)
                        presence_cum = np.nancumsum(presence)

                        if center_occupancy:
                            data = np.divide(
                                center_cum,
                                presence_cum,
                                out=np.full_like(center_cum, np.nan, dtype=float),
                                where=presence_cum != 0
                            )

                        elif center_preference:
                            prop_center = np.divide(
                                center_cum,
                                presence_cum,
                                out=np.full_like(center_cum, np.nan, dtype=float),
                                where=presence_cum != 0
                            )
                            data = 2 * prop_center - 1

                        all_data.append(data)

                if len(all_data) > 0:
                    all_data = np.vstack(all_data)

                    mean_data = np.nanmean(all_data, axis=0)

                    valid_n = np.sum(~np.isnan(all_data), axis=0)
                    sem_data = np.divide(
                        np.nanstd(all_data, axis=0),
                        np.sqrt(valid_n),
                        out=np.full_like(mean_data, np.nan, dtype=float),
                        where=valid_n > 0
                    )

                    time_minutes = np.arange(n_frames) / (FPS * 60)
                    max_plot_frames = min(int(plot_minutes * 60 * FPS), n_frames)

                    color = group_colors[grp]
                    layout = sex_layout_line[sex]

                    plt.plot(
                        time_minutes[:max_plot_frames],
                        mean_data[:max_plot_frames],
                        color=color,
                        linestyle=layout,
                        linewidth=2.5
                    )

                    plt.fill_between(
                        time_minutes[:max_plot_frames],
                        mean_data[:max_plot_frames] - sem_data[:max_plot_frames],
                        mean_data[:max_plot_frames] + sem_data[:max_plot_frames],
                        color=color,
                        alpha=0.2
                    )

        plt.xlabel("min")

        if center_occupancy:
            plt.ylabel("Center Occupancy")
            plt.ylim(0, 1)
            safename += "occupancy_"
        elif center_preference:
            plt.ylabel("Center Preference")
            plt.ylim(-1, 1)
            safename += "preference_"

        plt.xlim(0, plot_minutes)
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

        safename += "_females.jpg"
        safepath = r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis3\single_mice"
        safepath = r"C:\Users\Fabian\Desktop\Transfer\Analysis3\single_mice"
        plt.savefig(
            os.path.join(
                safepath,
                safename
            ),
            dpi=300
        )
        # plt.show()
        plt.close()