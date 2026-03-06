import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from config import FPS, PIXEL_PER_CM
import os

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



subgroup1 = ["germfree", "omm12"]
subgroup2 = ["germfree", "germfreeprop"]
subgroup3 = ["omm12", "omm12prop", "ommpgol"]

subgroups = [subgroup1, subgroup2, subgroup3]

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
    safename = "social_inv_pair_time_" + "_".join(subgroup) + ".jpg"

    for cond in conditions:
        if cond == "hab":
            n_frames = 54_000
        else:
            n_frames = 216_000

        for grp in subgroup:
            for sex in sexes:
                d = df.loc[:n_frames-1, idx[grp, :, sex, cond, :]]

                for mice_id in d.columns.get_level_values("mouse_ids").unique():

                    face_inv = d.loc[:, idx[:, mice_id, :, :, "all_face"]].to_numpy().squeeze()
                    body_inv = d.loc[:, idx[:, mice_id, :, :, "all_body"]].to_numpy().squeeze()
                    anogenital_inv = d.loc[:, idx[:, mice_id, :, :, "all_anogenital"]].to_numpy().squeeze()
                    m_per_frame = d.loc[:, idx[:, mice_id, :, :, "mice_per_frame"]].to_numpy().squeeze()

