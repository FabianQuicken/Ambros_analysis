# wie modulpref, nur normalisiert auf die aktuelle experimentdauer, nicht die gesamtdauer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

FPS = 30

#df = pd.read_csv(r"Z:\n2023_odor_related_behavior\2025_omm_mice\single_mouse_datatest_mice_presence.csv", header=[0,1,2,3,4,5], index_col=0)
df = pd.read_csv(r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\single_mouse_datatest_mice_presence.csv", header=[0,1,2,3,4,5], index_col=0)

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
    "males": "circle",
    "females": "triangle"
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

# OPTIONAL: Um Plotlänge zu begrenzen
#n_frames = 27000

for subgroup in subgroups:

    plt.figure(figsize=(12, 6))
    safename = "modulpref_relative_" + "_".join(subgroup) + "_females.jpg"
    title = " "

    for i, grp in enumerate(subgroup):
        title += grp
        if i < len(subgroup)-1:
            title += " vs. "

        for sex in ["females"]:   # oder ["females"], falls du nur females willst
            d = df.loc[:n_frames-1, idx[grp, :, sex, :, :, :]]

            all_data = []   # <-- hier sammeln wir alle individuellen Kurven

            for j, mice_id in enumerate(d.columns.get_level_values("mouse_ids").unique()):
                for k, ind in enumerate(individuals):

                    top1 = d.loc[:, idx[:, mice_id, :, "top1", "mice_presence", ind]].values.ravel()
                    top2 = d.loc[:, idx[:, mice_id, :, "top2", "mice_presence", ind]].values.ravel()

                    denom = np.nancumsum(top1) + np.nancumsum(top2)

                    data = np.divide(
                        np.nancumsum(top1) - np.nancumsum(top2),
                        denom,
                        out=np.full_like(denom, np.nan, dtype=float),
                        where=denom != 0
                    )

                    all_data.append(data)

            # nur wenn überhaupt Daten vorhanden sind
            if len(all_data) > 0:
                all_data = np.vstack(all_data)
                mean_data = np.nanmean(all_data, axis=0)
                sem_data = np.nanstd(all_data, axis=0) / np.sqrt(np.sum(~np.isnan(all_data), axis=0))

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
    plt.ylabel("Preference at timepoint (top1-top2 / top1+top2)")
        

    plt.ylim(-1.1, 1.1)

            
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

    # optional etwas Platz rechts schaffen für die Labels
    safepath = r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis3\single_mice"
    safepath = r"C:\Users\Fabian\Desktop\Transfer\Analysis3\single_mice\modulpref_relative_mean"
    plt.savefig(os.path.join(safepath, safename), dpi=300)
    #plt.show() 


