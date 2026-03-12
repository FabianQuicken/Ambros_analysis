import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
from tqdm import tqdm
from scipy.optimize import curve_fit

FPS = 30
PIXEL_PER_CM = 36.39

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


def exp_rise(t, y0, A, tau):
    """
    Exponential rise to plateau:
    y(t) = y0 + A * (1 - exp(-t/tau))
    """
    return y0 + A * (1 - np.exp(-t / tau))


fit_hab = True
plot_tau_text = True

save_dir = r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis3\single_mice"

for subgroup in subgroups:

    for cond in tqdm(conditions):
        if cond == "hab":
            n_frames = 54_000
        else:
            n_frames = 216_000

        plt.figure(figsize=(12, 6))
        safename = "cumdist_" + cond + "_"
        title = cond + " "

        for grp_idx, grp in enumerate(subgroup):
            title += grp
            safename += grp + "_"
            if grp_idx < len(subgroup) - 1:
                title += " vs. "

            for sex in ["males"]:
                d = df.loc[:n_frames-1, idx[grp, :, sex, cond, :]]
                d_presence = df_presence.loc[:n_frames-1, idx[grp, :, sex, cond, :]]

                for mouse_idx, mice_id in enumerate(d.columns.get_level_values("mouse_ids").unique()):

                    for ind_idx, ind in enumerate(individuals):

                        dist_sel = d.loc[:, idx[:, mice_id, :, :, "mice_cumdists", ind]]
                        pres_sel = d_presence.loc[:, idx[:, mice_id, :, :, "mice_presence", ind]]

                        if dist_sel.shape[1] == 0 or pres_sel.shape[1] == 0:
                            continue

                        dists = dist_sel.values.ravel().astype(float)
                        dists = dists / (PIXEL_PER_CM * 100)  # px -> m

                        presence = pres_sel.values.ravel()

                        # Optional: Presence-basierte Artefaktmaske
                        # dists[(presence == 0) & (dists == 0)] = np.nan

                        # Robust gegen NaNs / Artefakte am Ende
                        if np.isfinite(dists).any():
                            dists[-1] = np.nanmax(dists)

                        time_minutes = np.arange(n_frames) / (FPS * 60)
                        color = group_colors[grp]
                        layout = sex_layout_line[sex]

                        # Rohdaten plotten
                        plt.plot(time_minutes, dists, color=color, linestyle=layout, alpha=0.9)

                        # Exponential fit nur für hab
                        if cond == "hab" and fit_hab:
                            valid = np.isfinite(time_minutes) & np.isfinite(dists)

                            t_fit = time_minutes[valid]
                            y_fit = dists[valid]

                            # Mindestens ein paar Punkte nötig
                            if len(t_fit) > 10:
                                try:
                                    y0_guess = 0
                                    A_guess = max(y_fit[-1] - y_fit[0], 1e-6)
                                    tau_guess = max(t_fit[-1] / 3, 1e-3)

                                    

                                    popt, pcov = curve_fit(
                                        exp_rise,
                                        t_fit,
                                        y_fit,
                                        p0=[y0_guess, A_guess, tau_guess],
                                        bounds=(
                                            [-np.inf, 0, 1e-6],
                                            [ np.inf, np.inf, np.inf]
                                        ),
                                        maxfev=20000
                                    )

                                    y0_hat, A_hat, tau_hat = popt
                                    y_hat = exp_rise(t_fit, y0_hat, A_hat, tau_hat)

                                    plt.plot(
                                        t_fit,
                                        y_hat,
                                        color=color,
                                        linestyle="--",
                                        linewidth=2,
                                        alpha=0.9
                                    )

                                    if plot_tau_text:
                                        # Text ungefähr bei 70% der Fit-Zeit
                                        text_idx = int(tau_hat*FPS * 60)
                                        text_idx = min(text_idx, len(t_fit) - 1)

                                        plt.text(
                                            t_fit[text_idx],
                                            y_hat[text_idx],
                                            f"τ={tau_hat:.2f} min",
                                            color=color,
                                            fontsize=8,
                                            ha="left",
                                            va="bottom"
                                        )

                                except Exception as e:
                                    print(f"Fit failed for {grp}, {sex}, {mice_id}, {ind}: {e}")

                        # Label ans Ende der Kurve
                        valid = np.isfinite(dists)
                        if np.any(valid):
                            last_idx = np.where(valid)[0][-1]
                            x_end = time_minutes[last_idx]
                            y_end = dists[last_idx]

                            plt.text(
                                x_end + 0.1,
                                y_end,
                                f"m{mouse_idx+1}.{ind_idx+1}",
                                color=color,
                                fontsize=8,
                                va="center",
                                ha="left"
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
            for sex in sex_layout_line
        ]

        fit_handle = []
        if cond == "hab" and fit_hab:
            fit_handle = [
                mlines.Line2D([], [], color='black', linestyle='--', label='exp. fit')
            ]

        plt.legend(handles=group_handles + sex_handles + fit_handle)
        plt.tight_layout()

        safename += "females.jpg"
        savepath = os.path.join(save_dir, safename)
        #plt.savefig(savepath, dpi=300)
        plt.show()
        plt.close()