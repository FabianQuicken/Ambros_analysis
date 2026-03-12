import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
from tqdm import tqdm
from scipy.optimize import curve_fit

FPS = 30
PIXEL_PER_CM = 36.39

cond = "hab"
n_frames = 54000
sexes_to_plot = ["females"]

save_dir = r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis3\single_mice"

bin_seconds = 10

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

group_colors = {
    "germfree": "black",
    "germfreeprop": "brown",
    "omm12": "darkorange",
    "omm12prop": "forestgreen",
    "ommpgol": "darkviolet"
}

sex_layout_line = {
    "males": "solid",
    "females": "dotted"
}

individuals = df.columns.get_level_values("individual").unique()

subgroups = [
    ["germfree","omm12"],
    ["germfree","germfreeprop"],
    ["omm12","omm12prop"],
    ["omm12","ommpgol"]
]


def exp_decay(t, c, A, tau):
    return c + A * np.exp(-t / tau)


def compute_speed(cumdist):
    speed = np.diff(cumdist, prepend=cumdist[0]) * FPS
    speed[speed < 0] = np.nan
    return speed


def bin_timeseries(values):

    bin_size = bin_seconds * FPS
    n_complete = len(values) // bin_size

    trimmed = values[:n_complete * bin_size]
    reshaped = trimmed.reshape(n_complete, bin_size)

    binned = np.nanmean(reshaped, axis=1)

    centers = (np.arange(n_complete) * bin_size + bin_size/2) / (FPS*60)

    return centers, binned


for subgroup in subgroups:

    plt.figure(figsize=(12,6))

    title = cond + " "
    safename = "speed_decay_mean_" + cond + "_"

    tau_values = []


    for grp_i, grp in enumerate(subgroup):

        title += grp
        safename += grp + "_"

        if grp_i < len(subgroup)-1:
            title += " vs. "

        for sex in sexes_to_plot:

            d = df.loc[:n_frames-1, idx[grp, :, sex, cond, :]]
            d_presence = df_presence.loc[:n_frames-1, idx[grp, :, sex, cond, :]]

            mouse_ids = d.columns.get_level_values("mouse_ids").unique()

            
            for mouse_i, mice_id in enumerate(mouse_ids):

                all_speeds = []
                
                for ind in individuals:

                    dist_sel = d.loc[:, idx[:, mice_id, :, :, "mice_cumdists", ind]]

                    if dist_sel.shape[1] == 0:
                        continue

                    cumdist = dist_sel.values.ravel().astype(float)

                    cumdist = cumdist / (PIXEL_PER_CM*100)

                    speed = compute_speed(cumdist)

                    all_speeds.append(speed)

                if len(all_speeds) == 0:
                    continue

                all_speeds = np.vstack(all_speeds)

                mean_speed = np.nanmean(all_speeds, axis=0)

                t, speed_plot = bin_timeseries(mean_speed)

                color = group_colors[grp]
                layout = sex_layout_line[sex]

                plt.plot(
                    t,
                    speed_plot,
                    color=color,
                    linestyle=layout,
                    linewidth=2
                )

                valid = np.isfinite(t) & np.isfinite(speed_plot)

                t_fit = t[valid]
                y_fit = speed_plot[valid]

                if len(t_fit) > 5:

                    c_guess = np.nanmedian(y_fit[-5:])
                    A_guess = max(y_fit[0]-c_guess,1e-6)
                    tau_guess = t_fit[-1]/3

                    popt,_ = curve_fit(
                        exp_decay,
                        t_fit,
                        y_fit,
                        p0=[c_guess,A_guess,tau_guess],
                        bounds=([0,0,1e-6],[np.inf,np.inf,np.inf]),
                        maxfev=20000
                    )

                    c_hat, A_hat, tau_hat = popt

                    y_hat = exp_decay(t_fit, c_hat, A_hat, tau_hat)

                    # Fitlinie
                    plt.plot(
                        t_fit,
                        y_hat,
                        color=color,
                        linestyle="--",
                        linewidth=2
                    )

                    # y Wert der Kurve bei tau
                    y_tau = exp_decay(tau_hat, c_hat, A_hat, tau_hat)


                    plt.plot(
                        [tau_hat, tau_hat],
                        [0, y_tau],
                        linestyle=":",
                        color=color,
                        linewidth=1.5
                    )

                    # Punkt auf der Kurve
                    plt.scatter(
                        tau_hat,
                        y_tau,
                        color=color,
                        s=40,
                        zorder=5
                    )

                    # Tau Values
                    tau_values.append(
                        (f"m{mouse_i+1}", tau_hat, color)
                    )

    plt.xlabel("Time [min]")
    plt.ylabel("Speed [m/s]")
    plt.title(title)

    group_handles = [
        mlines.Line2D([],[],color=group_colors[g],linestyle='solid',label=g)
        for g in subgroup
    ]

    plt.legend(handles=group_handles)

    # sortiere tau Werte
    tau_values = sorted(tau_values, key=lambda x: x[1])

    # Position unter der Legende
    start_y = -0.18
    line_spacing = 0.06

    for i, (label, tau_val, color) in enumerate(tau_values):

        y_pos = start_y - i * line_spacing

        plt.gca().text(
            0.02,
            y_pos,
            f"{label}   τ = {tau_val:.2f} min",
            transform=plt.gca().transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=dict(
                facecolor="white",
                edgecolor=color,
                boxstyle="round,pad=0.25",
                alpha=0.9
            )
        )

    plt.tight_layout(rect=[0,0.15,1,1])

    safename += "females.jpg"

    plt.savefig(os.path.join(save_dir,safename),dpi=300)
    plt.show()

    plt.close()