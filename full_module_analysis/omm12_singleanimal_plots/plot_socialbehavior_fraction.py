import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

FPS = 30

df = pd.read_csv(r"Z:\n2023_odor_related_behavior\2025_omm_mice\single_mouse_datatest_social_behavior.csv", header=[0,1,2,3,4,5], index_col=0)
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
    safename = "anogenitalinv_to_totalinv_" + "_".join(subgroup) + ".jpg"

    for cond in conditions:
        if cond == "hab":
            n_frames = 54_000
        else:
            n_frames = 216_000

        for grp in subgroup:
            #sexes = ["males"]
            for sex in sexes:
                d = df.loc[:n_frames-1, idx[grp, :, sex, cond, :, :]]
                d_presence = df_presence.loc[:n_frames-1, idx[grp, :, sex, cond, :, :]]

                # Hier sammeln wir alle fractions dieser grp/sex/cond-Kombination
                fractions_this_group = []

                for j, mice_id in enumerate(d.columns.get_level_values("mouse_ids").unique()):
                    for i, ind in enumerate(individuals):

                        face_inv = d.loc[:, idx[:, mice_id, :, :, "face_inv", ind]].to_numpy().squeeze()
                        body_inv = d.loc[:, idx[:, mice_id, :, :, "body_inv", ind]].to_numpy().squeeze()
                        anogenital_inv = d.loc[:, idx[:, mice_id, :, :, "anogenital_inv", ind]].to_numpy().squeeze()
                        m_presence = d_presence.loc[:, idx[:, mice_id, :, :, "mice_presence", ind]].to_numpy().squeeze()
                        m_per_frame = d_presence.loc[:, idx[:, mice_id, :, :, "mice_presence", :]].to_numpy()
                        mice_count = np.nansum(m_per_frame, axis=1)

                        # pair time nur für frames, in denen die Maus selbst auch drin ist
                        mask = (~np.isnan(m_presence)) & (m_presence > 0)
                        mice_count = mice_count * mask

                        pairs_per_frame = np.divide(
                            mice_count * (mice_count - 1),
                            2,
                            out=np.zeros_like(mice_count, dtype=float),
                            where=~np.isnan(mice_count)
                        )
                        pairtime = np.nansum(pairs_per_frame) / FPS / 60

                        social_inv = face_inv + body_inv + anogenital_inv

                        if np.nansum(social_inv) == 0:
                            fraction = np.nan
                        else:
                            fraction = 100 * np.nansum(anogenital_inv) / np.nansum(social_inv)

                        if pairtime < 10:
                            fraction = np.nan

                        if len(sexes) > 1:
                            x = x_positions[cond] + offset_map[grp] + sex_offsets[sex]
                        else:
                            x = x_positions[cond] + offset_map[grp]
                        plt.scatter(
                            x,
                            fraction,
                            color=group_colors[grp],
                            marker=sex_layout_scatter[sex],
                            s=70,
                            alpha=0.9
                        )

                        # nur gültige Werte für Mean sammeln
                        if not np.isnan(fraction):
                            fractions_this_group.append(fraction)

                        if pairtime < 10:
                            plt.text(
                                x + 0.03,
                                0 if np.isnan(fraction) else fraction,
                                f"{pairtime:.1f}",
                                fontsize=8,
                                color="red"
                            )
                        elif len(sexes) < 2:
                            plt.text(
                                x + 0.1,
                                fraction,
                                "m" + str(j+1) + "." + str(i+1),
                                color=group_colors[grp],
                                fontsize=8,
                                va="center",
                                ha="left"
                            )

                # Mean pro grp/sex/cond als roter Marker
                if len(fractions_this_group) > 0:
                    mean_fraction = np.nanmean(fractions_this_group)

                    plt.scatter(
                        x,
                        mean_fraction,
                        color="red",
                        marker=sex_layout_scatter[sex],   # Kreis für males, Dreieck für females
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
    plt.ylim(-2, 100)
    plt.ylabel("Anogenital investigation / Social Investigation [%]")
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
