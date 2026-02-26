from main_multi_animal import multi_animal_main
import matplotlib.pyplot as plt
from config import PIXEL_PER_CM, FPS
import numpy as np

def violinplot_mean_sem(
    data,
    labels,
    colors=None,
    *,
    y_label="Value",
    x_label="Group",
    title="",
    savefig="",
    violin_alpha=0.6,
    show_stats=True,
    stat_fontsize=9,
    stat_offset=0.15  # Abstand rechts von der Violine
):
    """
    Violin plot per group with mean ± SEM overlay and optional
    text annotation of n, mean, and SEM next to each violin.
    """

    if len(data) != len(labels):
        raise ValueError("data and labels must have same length")

    n_groups = len(data)

    cleaned = []
    means = np.zeros(n_groups)
    sems  = np.zeros(n_groups)
    ns    = np.zeros(n_groups, dtype=int)

    # Daten vorbereiten
    for i, arr in enumerate(data):
        arr = np.asarray(arr, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]

        ns[i] = len(arr)
        means[i] = np.mean(arr) if ns[i] > 0 else np.nan
        sems[i] = (np.std(arr, ddof=1) / np.sqrt(ns[i])) if ns[i] > 1 else np.nan

        cleaned.append(arr)

    positions = np.arange(1, n_groups + 1)

    fig, ax = plt.subplots(figsize=(max(6, 1.3*n_groups), 4))

    vp = ax.violinplot(
        cleaned,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.8
    )

    # Farben setzen
    if colors is None:
        colors = [None] * n_groups

    for i, body in enumerate(vp["bodies"]):
        if colors[i] is not None:
            body.set_facecolor(colors[i])
            body.set_edgecolor(colors[i])
        body.set_alpha(violin_alpha)
        body.set_linewidth(1.2)

    # Mean ± SEM Overlay
    for i, x in enumerate(positions):

        if ns[i] == 0:
            continue

        ax.errorbar(
            x, means[i],
            yerr=sems[i],
            fmt="none",
            ecolor="black",
            elinewidth=1.6,
            capsize=6,
            capthick=1.6,
            zorder=3
        )

        ax.scatter(
            [x], [means[i]],
            color="black",
            s=40,
            zorder=4
        )

        # --- Statistik Text ---
        if show_stats:
            stat_text = (
                f"n = {ns[i]}\n"
                f"mean = {means[i]:.2f}\n"
                f"SEM = {sems[i]:.2f}"
            )

            ax.text(
                x + stat_offset,
                means[i],
                stat_text,
                va="center",
                fontsize=stat_fontsize
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if title:
        ax.set_title(title)

    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    if savefig:
        plt.savefig(savefig, format="jpg")

    #plt.show()

def lineplot(data, colors, labels, x_label = "Time[s]", y_label = "Distance [cm]", savefig=""):


    

    # gemeinsame minimale Länge bestimmen
    min_len = min(len(arr) for arr in data)

    time_sec = np.arange(min_len) / FPS

    plt.figure(figsize=(8, 4))

    used_labels = set()

    for arr, color, label in zip(data, colors, labels):

        # Label nur einmal pro Gruppe anzeigen
        if label not in used_labels:
            plt.plot(time_sec, arr[:min_len], color=color, label=label)
            used_labels.add(label)
        else:
            plt.plot(time_sec, arr[:min_len], color=color)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    if savefig:
         plt.savefig(savefig, format="jpg")
    #plt.show()

def scatterplot(
    data,
    colors,
    labels,
    x_label="X",
    y_label="Y",
    savefig="",
    alpha=0.7,
    point_size=15,
    size_by_count=True,
    x_bin=5,
    y_bin=None,
    size_scale=1.0,
    size_transform="sqrt",
    show_group_stats=True,
    draw_mean_line=True
):

    if len(data) != len(colors) or len(data) != len(labels):
        raise ValueError("data, colors, labels must have same length")

    plt.figure(figsize=(8, 5))
    used_labels = set()

    group_stats = []

    for (x, y), color, label in zip(data, colors, labels):

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        # -------- GROUP STATISTICS --------
        n = len(y)
        mean_visit = np.sum(y) / n if n > 0 else np.nan
        sd_visit = np.std(y, ddof=1) if n > 1 else np.nan

        group_stats.append((label, mean_visit, sd_visit, n))

        # -------- SIZE ENCODING --------
        if not size_by_count:
            if label not in used_labels:
                plt.scatter(x, y, color=color, label=label, alpha=alpha, s=point_size)
                used_labels.add(label)
            else:
                plt.scatter(x, y, color=color, alpha=alpha, s=point_size)
        else:
            if x_bin is not None:
                xq = np.round(x / x_bin) * x_bin
            else:
                xq = x

            if y_bin is not None:
                yq = np.round(y / y_bin) * y_bin
            else:
                yq = y

            coords = np.column_stack([xq, yq])
            uniq, counts = np.unique(coords, axis=0, return_counts=True)

            xu = uniq[:, 0]
            yu = uniq[:, 1]

            if size_transform == "linear":
                sizes = point_size * counts
            elif size_transform == "sqrt":
                sizes = point_size * np.sqrt(counts)
            elif size_transform == "log":
                sizes = point_size * np.log1p(counts)
            else:
                raise ValueError("size_transform must be one of: 'linear', 'sqrt', 'log'")

            sizes = sizes * size_scale

            if label not in used_labels:
                plt.scatter(xu, yu, color=color, label=label, alpha=alpha, s=sizes)
                used_labels.add(label)
            else:
                plt.scatter(xu, yu, color=color, alpha=alpha, s=sizes)

        # Optional: mean line
        if draw_mean_line and n > 0:
            plt.axhline(mean_visit, color=color, linestyle="--", alpha=0.4)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.legend()
    plt.tight_layout()

    # ---------- STAT TEXT ----------
    if show_group_stats:
        stat_text = ""
        for label, mean_visit, sd_visit, n in group_stats:
            stat_text += (
                f"{label}\n"
                f"n = {n}\n"
                f"mean = {mean_visit:.3f}\n"
                f"SD = {sd_visit:.3f}\n\n"
            )

        plt.gcf().text(
            0.98, 0.98,
            stat_text.strip(),
            ha="right",
            va="top",
            fontsize=9
        )

    if savefig:
        plt.savefig(savefig, format="jpg")

    #plt.show()

top1 = r"\top1"
top2 = r"\top2"
hab = r"\top1\hab"


germfree = [
            r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\germfree\females_30_45_46",
            r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\germfree\females_68_69_70",
            r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\germfree\males_38_47_53",
            r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\germfree\males_53_55_61"
]

germfreeprop = [
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\germfreeprop\females_37_44_55",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\germfreeprop\females_52_56_62",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\germfreeprop\males_34_38_42"
]

omm12 = [
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\omm12\females_31_36_59",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\omm12\females_54_57_60",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\omm12\males_41_43_58",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\omm12\males_83_86_71"
]

omm12prop = [
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\omm12prop\females_32_35_37",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\omm12prop\females_75_78_82",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\omm12prop\males_60_64_66",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\omm12prop\males_73_74_77"
]

ommpgol = [
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\ommpgol\females_33_47_48",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\ommpgol\females_72_76_79",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\ommpgol\males_41_44_51",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\ommpgol\males_80_81_87"
]


testpaths_grp1 = [
    r"C:\Users\quicken\Code\Ambros_analysis\code_test\ommfull\grp1\f1",
    r"C:\Users\quicken\Code\Ambros_analysis\code_test\ommfull\grp1\f2",
    r"C:\Users\quicken\Code\Ambros_analysis\code_test\ommfull\grp1\m1",
    r"C:\Users\quicken\Code\Ambros_analysis\code_test\ommfull\grp1\m2"
]
testpaths_grp2 = [
    r"C:\Users\quicken\Code\Ambros_analysis\code_test\ommfull\grp2\f1",
    r"C:\Users\quicken\Code\Ambros_analysis\code_test\ommfull\grp2\f2",
    r"C:\Users\quicken\Code\Ambros_analysis\code_test\ommfull\grp2\m1",
    r"C:\Users\quicken\Code\Ambros_analysis\code_test\ommfull\grp2\m2"
]



groups = [testpaths_grp1, testpaths_grp2]
groups = [germfree, germfreeprop, omm12, omm12prop, ommpgol]
test_colors = ["cyan", "red", "blue", "black", "magenta"]
groupnames = ["germfree", "germfreeprop", "omm12", "omm12prop", "ommpgol"]




mode = top1
cumdists = []
cumpresence = []
start_len_visits = [] # list of tuples [(startarr, lenarr)]
all_thetas = []
all_accelerations = []
colors = []
labels = []
for j, group in enumerate(groups):
        accelerations_group = []
        theta_group = []
        timepoints = []
        lens = []
        for path in group:
            full_path = path + mode
            dic = multi_animal_main(full_path)

            cumdists.append(dic["cumdist"] / PIXEL_PER_CM)  # direkt in cm
            mice_per_frame = dic["mice_per_frame"] / FPS
            cumpresence.append(np.nancumsum(mice_per_frame))
            colors.append(test_colors[j])
            labels.append(groupnames[j])
            visits = dic["visits"] # liste mit visits
            
            for visit in visits:
                tp = visit[0]/FPS
                l = visit[1]/FPS
                timepoints.append(tp)
                lens.append(l)
            
            thetas = dic["thetas"] # n_ind, frames

            for e in thetas:
                theta_group += e
            
            accelerations = dic["accelerations"] # all accelerations as list of values
            # nur positive werte behalten, die über gewissem trh liegen (da sonst immobile)
            positive_acc = []
            for acc in accelerations:
                if acc > 0.5:
                    positive_acc.append(acc)        

            accelerations_group += positive_acc

        all_thetas.append(theta_group)
        all_accelerations.append(accelerations_group)

        start_len_visits.append((timepoints, lens))
        scatterplot(data=[start_len_visits[j]],
                    colors=[test_colors[j]],
                    labels=[groupnames[j]],
                    savefig=r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis" + mode + groupnames[j] + r".jpg",
                    x_label="Time [s]",
                    y_label="Visitduration [s]")

savepath = r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis" + mode
lineplot(cumdists, colors, labels, savefig=savepath+"cumdists.jpg")
lineplot(cumpresence, colors, labels, y_label = "Time present [s]", savefig=savepath+r"cumpresence.jpg")
violinplot_mean_sem(all_thetas,labels=groupnames,colors=test_colors, title="Thetas", savefig=savepath+r"thetas.jpg")
violinplot_mean_sem(all_accelerations,labels=groupnames,colors=test_colors, title="Accelerations", savefig=savepath+r"acc.jpg")