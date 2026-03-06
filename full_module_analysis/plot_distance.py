import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from config import FPS, PIXEL_PER_CM

df = pd.read_csv(r"Z:\n2023_odor_related_behavior\2025_omm_mice\data.csv", header=[0,1,2,3,4], index_col=0)



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



subgroup1 = ["germfree", "omm12"]
subgroup2 = ["germfree", "germfreeprop"]
subgroup3 = ["omm12", "omm12prop", "ommpgol"]

subgroups = [subgroup1, subgroup2, subgroup3]


for subgroup in subgroups:
    safename = ""
    for cond in conditions:
        if cond == "hab":
            n_frames = 54000
        else:
            n_frames = 216_000

        plt.figure(figsize=(12, 6))
        safename = r"\distance_" + cond + "_"
        title = cond + " "
        
        for i, grp in enumerate(subgroup):
            title += grp 
            safename += grp + "_"
            if i < len(subgroup)-1:
                title += " vs. "
            for sex in sexes:
                d = df.loc[:n_frames-1, idx[grp, :, sex, cond, :]]

                for mice_id in d.columns.get_level_values("mouse_ids").unique():

                    dists = d.loc[:, idx[:, mice_id, :, :, "mice_distances"]].values.ravel()
                    m_per_frame = d.loc[:, idx[:, mice_id, :, :, "mice_per_frame"]].values.ravel()

                    # change px -> m
                    dists = dists / (PIXEL_PER_CM*100)

                    # distances divided by animals present per frame
                    dists_norm = np.divide(
                        dists,
                        m_per_frame,
                        out=np.full_like(dists, np.nan, dtype=float),
                        where=m_per_frame != 0
                    )

                    time_minutes = np.arange(n_frames) / (FPS * 60)
                    color = group_colors[grp]
                    layout = sex_layout_line[sex]

                    # entweder dists oder dists_norm plotten, jenachdem was man will
                    plt.plot(time_minutes, np.nancumsum(dists), color=color, linestyle=layout)
                    

        plt.xlabel("min")
        plt.ylabel("Dist [m]")
        
        if cond == "hab":
            plt.ylim(0,300)
        else:
            plt.ylim(0, 600)

            
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
        plt.savefig(r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis3" + safename + ".jpg")
        #plt.show()            

            





