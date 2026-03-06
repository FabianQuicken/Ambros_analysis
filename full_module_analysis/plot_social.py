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

                    # mice per frame -> pair time
                    pairs_per_frame = np.divide(
                        m_per_frame * (m_per_frame - 1),
                        2,
                        out=np.zeros_like(m_per_frame, dtype=float),
                        where=~np.isnan(m_per_frame)
                    )

                    # total social investigation
                    social_inv = face_inv + body_inv + anogenital_inv

                    # normalized social investigation per pair-time
                    denom = np.nansum(pairs_per_frame)
                    if denom == 0:
                        social_inv_pair_time = 0
                    else:
                        social_inv_pair_time = 100 * np.nansum(social_inv) / denom

                    x = x_positions[cond] + offset_map[grp]

                    pairtime = np.nansum(pairs_per_frame) / FPS / 60
                    pairtime = np.nansum(pairs_per_frame)

                    if pairtime < 20:
                        social_inv_pair_time = 0

                    plt.scatter(
                        x,
                        social_inv_pair_time,
                        color=group_colors[grp],
                        marker=sex_layout_scatter[sex],
                        s=70,
                        alpha=0.9
                    )

                    if pairtime < 18000:   # z.B. weniger als 20 Minuten Pairtime
                        plt.text(
                            x + 0.03,
                            social_inv_pair_time,
                            f"{pairtime:.1f}",
                            fontsize=8,
                            color="red"
                        )

    plt.xticks(
        [x_positions["hab"], x_positions["top1"], x_positions["top2"]],
        ["hab", "top1", "top2"]
    )
    plt.xlim(-0.5, 2.5)
    plt.ylim(-2, 50)
    plt.ylabel("Social investigation / pair time [%]")
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
    plt.savefig(os.path.join(r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis3", safename), dpi=300)
    plt.show()
    plt.close()


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
    safename = "faceinv_to_totalinv_" + "_".join(subgroup) + ".jpg"

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

                    # mice per frame -> pair time
                    pairs_per_frame = np.divide(
                        m_per_frame * (m_per_frame - 1),
                        2,
                        out=np.zeros_like(m_per_frame, dtype=float),
                        where=~np.isnan(m_per_frame)
                    )

                    pairtime = np.nansum(pairs_per_frame) / FPS / 60

                    # total social investigation
                    social_inv = face_inv + body_inv + anogenital_inv

                    fraction = 100 * np.nansum(face_inv) / np.nansum(social_inv)
                    if pairtime < 10:
                        fraction = 0

                    x = x_positions[cond] + offset_map[grp]

                    plt.scatter(
                        x,
                        fraction,
                        color=group_colors[grp],
                        marker=sex_layout_scatter[sex],
                        s=70,
                        alpha=0.9
                    )


    plt.xticks(
        [x_positions["hab"], x_positions["top1"], x_positions["top2"]],
        ["hab", "top1", "top2"]
    )
    plt.xlim(-0.5, 2.5)
    plt.ylim(-2, 100)
    plt.ylabel("Face investigation / Social Investigation [%]")
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
    plt.savefig(os.path.join(r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis3", safename), dpi=300)
    #plt.show()
    plt.close()