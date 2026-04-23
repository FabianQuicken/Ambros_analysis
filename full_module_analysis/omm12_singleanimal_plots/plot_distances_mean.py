import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
from tqdm import tqdm

FPS = 30
PIXEL_PER_CM = 36.39

def clean_cumdist_tail(arr, max_step=200, negative_tol=0):
    """
    Clean trailing artifacts in cumulative-distance arrays.

    Rules
    -----
    1. Keep only the first contiguous finite block.
       If a NaN appears after valid data started, everything from that NaN onward is set to NaN.
    2. Optionally stop at the first implausible step:
       - negative step smaller than -negative_tol
       - step larger than max_step (if max_step is not None)

    Parameters
    ----------
    arr : array-like
        1D cumulative distance array.
    max_step : float or None, optional
        Maximum allowed positive increment between consecutive samples.
        Use the unit of `arr` (px if raw, m if already converted).
        If None, large positive steps are not checked.
    negative_tol : float, optional
        Small tolerance for negative steps.

    Returns
    -------
    np.ndarray
        Cleaned array with same length as input, trailing artifacts replaced by NaN.
    """
    arr = np.asarray(arr, dtype=float).copy()

    if arr.size == 0:
        return arr

    finite = np.isfinite(arr)
    if not finite.any():
        return arr

    first_valid = np.argmax(finite)

    # 1) Sobald nach Start des gültigen Blocks ein NaN kommt -> Rest maskieren
    finite_after_start = finite[first_valid:]
    first_invalid_rel = np.where(~finite_after_start)[0]
    if first_invalid_rel.size > 0:
        cut_idx = first_valid + first_invalid_rel[0]
        arr[cut_idx:] = np.nan
        return arr

    # 2) Optional: unplausible Sprünge prüfen
    diffs = np.diff(arr[first_valid:])

    bad = diffs < -negative_tol
    if max_step is not None:
        bad |= diffs > max_step

    bad_idx = np.where(bad)[0]
    if bad_idx.size > 0:
        cut_idx = first_valid + bad_idx[0] + 1
        arr[cut_idx:] = np.nan

    return arr

df = pd.read_csv(
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\single_mouse_datatest_mice_cumdists.csv",
    header=[0,1,2,3,4,5],
    index_col=0
)
df_presence = pd.read_csv(
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\single_mouse_datatest_mice_presence.csv",
    header=[0,1,2,3,4,5],
    index_col=0
)

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

save_dir = r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis3\single_mice"

# hier wählen
sexes_to_plot = ["females"]   # z.B. ["males"], ["females"] oder ["males", "females"]
plot_sd = True


def safe_mean_sd(arr_2d):
    """
    arr_2d shape: (n_individuals, n_frames)
    Returns mean and sd over axis=0 with NaN handling.
    """
    mean_arr = np.nanmean(arr_2d, axis=0)
    sd_arr = np.nanstd(arr_2d, axis=0)
    return mean_arr, sd_arr


def get_condition_frames(cond):
    if cond == "hab":
        return 54_000
    return 216_000


# --------------------------------------------------
# 1) EIN PLOT PRO CONDITION
# --------------------------------------------------
for subgroup in subgroups:

    for cond in tqdm(conditions):
        n_frames = get_condition_frames(cond)

        plt.figure(figsize=(12, 6))
        safename = "cumdist_mean_sd_" + cond + "_"
        title = cond + " "

        for grp_idx, grp in enumerate(subgroup):
            title += grp
            safename += grp + "_"
            if grp_idx < len(subgroup) - 1:
                title += " vs. "

            for sex in sexes_to_plot:
                d = df.loc[:n_frames-1, idx[grp, :, sex, cond, :]] 
                d_presence = df_presence.loc[:n_frames-1, idx[grp, :, sex, cond, :]]

                all_dists = []

                for mice_id in d.columns.get_level_values("mouse_ids").unique():
                    for ind in individuals:

                        dist_sel = d.loc[:, idx[:, mice_id, :, :, "mice_cumdists", ind]]
                        pres_sel = d_presence.loc[:, idx[:, mice_id, :, :, "mice_presence", ind]]

                        if dist_sel.shape[1] == 0 or pres_sel.shape[1] == 0:
                            continue

                        dists = dist_sel.values.ravel().astype(float)
                        if cond == "top2":
                            dists = dists[0:198000]
                            n_frames = 198000
                                
                        dists = dists / (PIXEL_PER_CM * 100)  # px -> m

                        presence = pres_sel.values.ravel()

                        # optional presence mask:
                        # dists[(presence == 0) & (dists == 0)] = np.nan

                        # robust gegen Artefakte am Ende
                        if np.isfinite(dists).any():
                            dists[-1] = np.nanmax(dists)

                        all_dists.append(dists)

                if len(all_dists) == 0:
                    continue

                all_dists = np.vstack(all_dists)
                mean_dists, sd_dists = safe_mean_sd(all_dists)

                time_minutes = np.arange(n_frames) / (FPS * 60)
                color = group_colors[grp]
                layout = sex_layout_line[sex]

                plt.plot(
                    time_minutes,
                    mean_dists,
                    color=color,
                    linestyle=layout,
                    linewidth=2.5
                )

                if plot_sd:
                    plt.fill_between(
                        time_minutes,
                        mean_dists - sd_dists,
                        mean_dists + sd_dists,
                        color=color,
                        alpha=0.2
                    )

        plt.xlabel("min")
        plt.ylabel("Dist [m]")
        plt.title(title)

        group_handles = [
            mlines.Line2D([], [], color=group_colors[group], linestyle='solid', label=group)
            for group in subgroup
        ]

        sex_handles = [
            mlines.Line2D([], [], color='black', linestyle=sex_layout_line[sex], label=sex)
            for sex in sexes_to_plot
        ]

        sd_handle = []
        if plot_sd:
            sd_handle = [
                mlines.Line2D([], [], color='gray', linestyle='None', marker='s', markersize=8, alpha=0.4, label='± SD')
            ]

        plt.legend(handles=group_handles + sex_handles + sd_handle)
        plt.tight_layout()

        safename += "_".join(sexes_to_plot) + ".jpg"
        savepath = os.path.join(save_dir, safename)
        #plt.savefig(savepath, dpi=300)
        plt.show()
        plt.close()




