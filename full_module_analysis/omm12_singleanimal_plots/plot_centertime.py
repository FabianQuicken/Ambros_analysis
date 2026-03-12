import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
from tqdm import tqdm

FPS = 30

df = pd.read_csv(r"Z:\n2023_odor_related_behavior\2025_omm_mice\single_mouse_datatest_mice_in_center.csv", header=[0,1,2,3,4,5], index_col=0)
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

for subgroup in subgroups:

    for cond in tqdm(conditions):
        if cond == "hab":
            n_frames = 54000
        else:
            n_frames = 216_000

        plt.figure(figsize=(12, 6))
        safename = r"centerpref_" + cond + "_"
        title = cond + " "
        
        for i, grp in enumerate(subgroup):
            title += grp 
            safename += grp + "_"
            if i < len(subgroup)-1:
                title += " vs. "
            for sex in sexes:
                d = df.loc[:n_frames-1, idx[grp, :, sex, cond, :]]
                d_presence = df_presence.loc[:n_frames-1, idx[grp, :, sex, cond, :]]

                for j, mice_id in enumerate(d.columns.get_level_values("mouse_ids").unique()):

                        for i, ind in enumerate(individuals):

                            centertime = d.loc[:, idx[:, mice_id, :, :, "mice_in_center", ind]].values.ravel()
                            presence = d_presence.loc[:, idx[:, mice_id, :, :, "mice_presence", ind]].values.ravel()

                            # es gibt in den Daten bei einigen Spalten einen einzelnen Wert -9*10^18 anstatt einer 0 oder 1, vermutlich durch das speichern
                            # als joblib und dann wieder einlesen und dann als speichern als csv - dieser Wert wird einmal raus gefiltert
                            #centertime = np.where((centertime) == 1, 1, 0)
                            #presence = np.where(presence == 1, 1, 0)


                            # total social investigation
                            center_cum = np.nancumsum(centertime)
                            presence_cum = np.nancumsum(presence)

                            

                            center_occupancy = False
                            if center_occupancy:
                                data = np.divide(
                                    center_cum,
                                    presence_cum,
                                    out=np.zeros_like(center_cum, dtype=float),
                                    where=presence_cum != 0
                                )

                            center_preference = True
                            if center_preference:
                                prop_center = np.divide(
                                    center_cum,
                                    presence_cum,
                                    out=np.zeros_like(center_cum, dtype=float),
                                    where=presence_cum != 0
                                )

                                data = 2 * prop_center - 1

                            time_minutes = np.arange(n_frames) / (FPS * 60)
                            color = group_colors[grp]
                            layout = sex_layout_line[sex]

                            # entweder dists oder dists_norm plotten, jenachdem was man will
                            plt.plot(time_minutes[0:9000], data[0:9000], color=color, linestyle=layout)

                            # --- Label ans Ende der Kurve ---
                            valid = np.isfinite(data)
                            if np.any(valid):
                                last_idx = np.where(valid)[0][9000]
                                x_end = time_minutes[last_idx]
                                y_end = data[last_idx]

                                plt.text(
                                    x_end + 0.1,   # kleiner Offset nach rechts
                                    y_end,
                                    "m"+str(j+1)+"."+str(i+1),
                                    color=color,
                                    fontsize=8,
                                    va="center",
                                    ha="left"
                                )

        plt.xlabel("min")
        plt.ylabel("Center Preference")
            

        plt.ylim(-1,1)


                
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
        safename += ".jpg"

        plt.savefig(os.path.join(r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis3\single_mice", safename), dpi=300)
        #plt.show()
        plt.close()