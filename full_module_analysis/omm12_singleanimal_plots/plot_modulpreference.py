import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

FPS = 30

df = pd.read_csv(r"Z:\n2023_odor_related_behavior\2025_omm_mice\single_mouse_datatest_mice_presence.csv", header=[0,1,2,3,4,5], index_col=0)

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



for subgroup in subgroups:
        

        safename = ""



        plt.figure(figsize=(12, 6))
        safename = r"\modulpref_"
        title = " "
        
        for i, grp in enumerate(subgroup):
            title += grp 
            safename += grp + "_"
            if i < len(subgroup)-1:
                title += " vs. "
            for sex in ["females"]:
                d = df.loc[:n_frames-1, idx[grp, :, sex, :, :, :]]

                for mice_id in d.columns.get_level_values("mouse_ids").unique():

                    for ind in individuals:

                        top1 = d.loc[:, idx[:, mice_id, :, "top1", "mice_presence", ind]].values.ravel()
                        top2 = d.loc[:, idx[:, mice_id, :, "top2", "mice_presence", ind]].values.ravel()
 

                        data = (np.nancumsum(top1) - np.nancumsum(top2)) / (np.nansum(top1) + np.nansum(top2))

                        time_minutes = np.arange(n_frames) / (FPS * 60)
                        color = group_colors[grp]
                        layout = sex_layout_line[sex]

                        # entweder dists oder dists_norm plotten, jenachdem was man will
                        plt.plot(time_minutes, data, color=color, linestyle=layout)

                        # --- Label ans Ende der Kurve ---
                        valid = np.isfinite(data)
                        if np.any(valid):
                            last_idx = np.where(valid)[0][-1]
                            x_end = time_minutes[last_idx]
                            y_end = data[last_idx]

                            plt.text(
                                x_end + 0.1,   # kleiner Offset nach rechts
                                y_end,
                                ind,
                                color=color,
                                fontsize=8,
                                va="center",
                                ha="left"
                            )
                    

        plt.xlabel("min")
        plt.ylabel("Preference (top1-top2 / top1+top2)")
        

        plt.ylim(-1, 1)

            
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
        plt.xlim(0, n_frames / (FPS * 60) + 10)
        plt.savefig(r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis3\single_mice" + safename + ".jpg")
        #plt.show() 




