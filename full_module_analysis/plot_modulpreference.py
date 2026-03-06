import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from config import FPS

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

# subsets
germfree = df.loc[:n_frames-1, idx["germfree", :, :, :, :]]
germfreeprop = df.loc[:n_frames-1, idx["germfreeprop", :, :, :, :]]
omm12 = df.loc[:n_frames-1, idx["omm12", :, :, :, :]]
ommprop = df.loc[:n_frames-1, idx["omm12prop", :, :, :, :]]
ommpgol = df.loc[:n_frames-1, idx["ommpgol", :, :, :, :]]

# subsets male
germfree_male = df.loc[:n_frames-1, idx["germfree", :, "males", :, :]]
germfreeprop_male = df.loc[:n_frames-1, idx["germfreeprop", :, "males", :, :]]
omm12_male = df.loc[:n_frames-1, idx["omm12", :, "males", :, :]]
ommprop_male = df.loc[:n_frames-1, idx["omm12prop", :, "males", :, :]]
ommpgol_male = df.loc[:n_frames-1, idx["ommpgol", :, "males", :, :]]

# subsets female
germfree_female = df.loc[:n_frames-1, idx["germfree", :, "females", :, :]]
germfreeprop_female = df.loc[:n_frames-1, idx["germfreeprop", :, "females", :, :]]
omm12_female = df.loc[:n_frames-1, idx["omm12", :, "females", :, :]]
ommprop_female = df.loc[:n_frames-1, idx["omm12prop", :, "females", :, :]]
ommpgol_female = df.loc[:n_frames-1, idx["ommpgol", :, "females", :, :]]

groups = germfree.columns.get_level_values("group").unique()
ids = germfree.columns.get_level_values("mouse_ids").unique()
sexes = germfree.columns.get_level_values("sex").unique()
conditions = germfree.columns.get_level_values("condition").unique()
metrics = germfree.columns.get_level_values("metric").unique()

plt.figure(figsize=(12,6))

for d in [germfree_male, germfree_female, omm12_male, omm12_female]:
    
    for mid in d.columns.get_level_values("mouse_ids").unique():
        

        group = d.columns.get_level_values("group").unique()[0]
        sex = d.columns.get_level_values("sex").unique()[0]
        
        color = group_colors[group]
        layout = sex_layout_line[sex]

        top1 = d.loc[:, idx[:, mid, :, "top1", "mice_per_frame"]].values.ravel()
        top2 = d.loc[:, idx[:, mid, :, "top2", "mice_per_frame"]].values.ravel()



        # falls mehrere Spalten existieren → zusammenfassen
        #top1 = np.nansum(top1, axis=1)
        #top2 = np.nansum(top2, axis=1)

        diff = (np.nancumsum(top1) - np.nancumsum(top2)) / (np.nansum(top1) + np.nansum(top2))

        time_minutes = np.arange(n_frames) / (FPS * 60)

        plt.plot(time_minutes, diff, color=color, linestyle=layout)

plt.xlabel("min")
plt.ylabel("Preference (top1-top2 / top1+top2)")


plt.title("germfree vs omm12")
subgroups = ["germfree", "omm12"]
group_handles = [
    mlines.Line2D([], [], color=color, linestyle='solid', label=group)
    for group, color in group_colors.items()
]

sex_handles = [
    mlines.Line2D([], [], color='black', linestyle=sex_layout_line[sex], label=sex)
    for sex in sex_layout_line
]

plt.legend(handles=group_handles + sex_handles)
plt.ylim(-1, 1)
plt.savefig(r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis3" + r"\modulpref_germfree_omm12.jpg")
plt.show()


plt.figure(figsize=(12,6))

for d in [omm12_male, omm12_female, ommprop_male, ommprop_female, ommpgol_male, ommpgol_female]:
    
    for mid in d.columns.get_level_values("mouse_ids").unique():
        

        group = d.columns.get_level_values("group").unique()[0]
        sex = d.columns.get_level_values("sex").unique()[0]
        
        color = group_colors[group]
        layout = sex_layout_line[sex]

        top1 = d.loc[:, idx[:, mid, :, "top1", "mice_per_frame"]].values.ravel()
        top2 = d.loc[:, idx[:, mid, :, "top2", "mice_per_frame"]].values.ravel()



        # falls mehrere Spalten existieren → zusammenfassen
        #top1 = np.nansum(top1, axis=1)
        #top2 = np.nansum(top2, axis=1)

        diff = (np.nancumsum(top1) - np.nancumsum(top2)) / (np.nansum(top1) + np.nansum(top2))
        time_minutes = np.arange(n_frames) / (FPS * 60)

        plt.plot(time_minutes, diff, color=color, linestyle=layout)

plt.xlabel("min")
plt.ylabel("Preference (top1-top2 / top1+top2)")


plt.title("omm12 vs omm12 + prop vs ommpgol")
subgroups = ["germfree", "omm12"]
group_handles = [
    mlines.Line2D([], [], color=color, linestyle='solid', label=group)
    for group, color in group_colors.items()
]

sex_handles = [
    mlines.Line2D([], [], color='black', linestyle=sex_layout_line[sex], label=sex)
    for sex in sex_layout_line
]

plt.legend(handles=group_handles + sex_handles)
plt.ylim(-1, 1)
plt.savefig(r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis3" + r"\modulpref_omm12_omm12prop_ommpgol.jpg")
plt.show()


plt.figure(figsize=(12,6))

for d in [germfree_male, germfree_female, germfreeprop_male, germfreeprop_female]:
    
    for mid in d.columns.get_level_values("mouse_ids").unique():
        

        group = d.columns.get_level_values("group").unique()[0]
        sex = d.columns.get_level_values("sex").unique()[0]
        
        color = group_colors[group]
        layout = sex_layout_line[sex]

        top1 = d.loc[:, idx[:, mid, :, "top1", "mice_per_frame"]].values.ravel()
        top2 = d.loc[:, idx[:, mid, :, "top2", "mice_per_frame"]].values.ravel()



        # falls mehrere Spalten existieren → zusammenfassen
        #top1 = np.nansum(top1, axis=1)
        #top2 = np.nansum(top2, axis=1)

        diff = (np.nancumsum(top1) - np.nancumsum(top2)) / (np.nansum(top1) + np.nansum(top2))

        

        time_minutes = np.arange(n_frames) / (FPS * 60)

        plt.plot(time_minutes, diff, color=color, linestyle=layout)

plt.xlabel("min")
plt.ylabel("Preference (top1-top2 / top1+top2)")


plt.title("germfree vs germfreeprop")
subgroups = ["germfree", "omm12"]
group_handles = [
    mlines.Line2D([], [], color=color, linestyle='solid', label=group)
    for group, color in group_colors.items()
]

sex_handles = [
    mlines.Line2D([], [], color='black', linestyle=sex_layout_line[sex], label=sex)
    for sex in sex_layout_line
]

plt.legend(handles=group_handles + sex_handles)
plt.ylim(-1, 1)
plt.savefig(r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis3" + r"\modulpref_germfree_germfreeprop.jpg")
plt.show()
