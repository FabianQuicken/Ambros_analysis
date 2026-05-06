import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from config import FPS, PIXEL_PER_CM

df = pd.read_csv(
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\data.csv",
    header=[0, 1, 2, 3, 4],
    index_col=0
)

idx = pd.IndexSlice

group_colors = {
    "germfree": "black",
    "germfreeprop": "brown",
    "omm12": "darkorange",
    "omm12prop": "forestgreen",
    "ommpgol": "darkviolet"
}

groups = df.columns.get_level_values("group").unique()
sexes = df.columns.get_level_values("sex").unique()
conditions = df.columns.get_level_values("condition").unique()

subgroup1 = ["germfree", "omm12"]
subgroup2 = ["germfree", "germfreeprop"]
subgroup3 = ["omm12", "ommpgol"]

subgroups = [subgroup3]

save_dir = r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis4"

for subgroup in subgroups:

    for cond in conditions:

        if cond == "hab":
            n_frames = 54_000
        else:
            n_frames = 216_000

        time_minutes = np.arange(n_frames) / (FPS * 60)

        for sex in sexes:

            plt.figure(figsize=(12, 6))

            title = f"{cond} {sex}: " + " vs. ".join(subgroup)
            safename = rf"\centerpref_{cond}_{sex}_" + "_".join(subgroup) + ".jpg"

            for grp in subgroup:

                mouse_curves = []

                try:
                    d = df.loc[:n_frames - 1, idx[grp, :, sex, cond, :]]
                except KeyError:
                    print(f"Skipping missing combination: {grp}, {sex}, {cond}")
                    continue

                mouse_ids = d.columns.get_level_values("mouse_ids").unique()

                for mouse_id in mouse_ids:

                    center_per_frame = d.loc[
                        :, idx[:, mouse_id, :, :, "center_per_frame"]
                    ].values.ravel()

                    m_per_frame = d.loc[
                        :, idx[:, mouse_id, :, :, "mice_per_frame"]
                    ].values.ravel()

                    center_cum = np.nancumsum(center_per_frame)
                    presence_cum = np.nancumsum(m_per_frame)

                    prop_center = np.divide(
                        center_cum,
                        presence_cum,
                        out=np.full_like(center_cum, np.nan, dtype=float),
                        where=presence_cum != 0
                    )

                    diff = 2 * prop_center - 1

                    mouse_curves.append(diff)

                if len(mouse_curves) == 0:
                    continue

                mouse_curves = np.vstack(mouse_curves)
                mean_curve = np.nanmean(mouse_curves, axis=0)
                sd_curve = np.nanstd(mouse_curves, axis=0)

                color = group_colors[grp]

                plt.plot(
                    time_minutes,
                    mean_curve,
                    color=color,
                    linewidth=2,
                    label=f"{grp} mean"
                )

                plt.fill_between(
                    time_minutes,
                    mean_curve - sd_curve,
                    mean_curve + sd_curve,
                    color=color,
                    alpha=0.2,
                    linewidth=0
                )

            plt.xlabel("min")
            plt.ylabel("Center Preference")
            plt.ylim(-1, 1)
            plt.title(title)

            group_handles = [
                mlines.Line2D(
                    [],
                    [],
                    color=group_colors[group],
                    linestyle="solid",
                    linewidth=2,
                    label=group
                )
                for group in subgroup
            ]

            sd_handle = mlines.Line2D(
                [],
                [],
                color="gray",
                linewidth=8,
                alpha=0.2,
                label="± SD"
            )

            plt.legend(handles=group_handles + [sd_handle])
            plt.tight_layout()
            plt.savefig(save_dir + safename, dpi=300)
            plt.close()