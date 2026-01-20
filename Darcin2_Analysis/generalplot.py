import matplotlib.pyplot as plt
import numpy as np

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _default_mouse_colors(mice):
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if not prop_cycle:
        prop_cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]
    return {m: prop_cycle[i % len(prop_cycle)] for i, m in enumerate(mice)}

def plot_days_one_metric(
    mice_data: dict,
    mice_order: list,
    metric_key: str,                 # z.B. "stim_dish" oder "con_dish"
    days=("day1", "day2", "day3"),
    title: str = None,
    ylabel: str = "Investigation time (frames)",
    show: bool = True,
    save_as: str = None,
    ax=None,
):
    """
    Funktion 1 (generalisiert):
    Plottet Tag1-3 für EINE Bedingung (z.B. nur stim_dish oder nur con_dish).
    Pro Maus: Punkte je Tag + Linie (Trend über Tage).
    """

    x = np.arange(len(days))
    colors = _default_mouse_colors(mice_order)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    for mouse in mice_order:
        if mouse not in mice_data:
            raise KeyError(f"Mouse '{mouse}' not found in mice_data keys: {list(mice_data.keys())}")

        y = []
        for d in days:
            if d not in mice_data[mouse]:
                raise KeyError(f"Day '{d}' not found for mouse '{mouse}'")
            if metric_key not in mice_data[mouse][d]:
                raise KeyError(f"Key '{metric_key}' not found for mouse '{mouse}' on '{d}'")
            y.append(_to_float(mice_data[mouse][d][metric_key]))

        ax.plot(x, y, marker="o", linewidth=1.5, label=f"Mouse {mouse}", color=colors[mouse])

    ax.set_xticks(x)
    ax.set_xticklabels(days)
    ax.set_ylabel(ylabel)
    ax.set_title(title if title else f"{metric_key} across days")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, ncols=2)

    if save_as:
        fig.savefig(save_as, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_each_day_two_metrics(
    mice_data: dict,
    mice_order: list,
    key_a: str = "stim_dish",
    key_b: str = "con_dish",
    days=("day1", "day2", "day3"),
    title_prefix: str = "Stimulus vs Control (Investigation)",
    ylabel: str = "Time spent (Fraction)",
    save_as: str = None,
    show: bool = True,
):
    """
    Funktion 2 (generalisiert):
    Für jeden Tag ein eigener Plot.
    In jedem Plot: zwei Bedingungen (key_a vs key_b) pro Maus als zwei Punkte,
    verbunden durch eine Linie (Unterschied innerhalb des Tages).
    """

    colors = _default_mouse_colors(mice_order)

    figs_axes = []
    for d in days:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))

        x = np.array([0, 1])
        ax.set_xticks(x)
        ax.set_xticklabels([key_a, key_b])

        for mouse in mice_order:
            if mouse not in mice_data:
                raise KeyError(f"Mouse '{mouse}' not found in mice_data keys: {list(mice_data.keys())}")
            if d not in mice_data[mouse]:
                raise KeyError(f"Day '{d}' not found for mouse '{mouse}'")

            day_dict = mice_data[mouse][d]
            if key_a not in day_dict or key_b not in day_dict:
                raise KeyError(f"Mouse '{mouse}' on '{d}' missing '{key_a}' or '{key_b}'")

            y_a = _to_float(day_dict[key_a])
            y_b = _to_float(day_dict[key_b])

            ax.plot(x, [y_a, y_b], marker="o", linewidth=1.5, color=colors[mouse], label=f"Mouse {mouse}")

        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix} – {d}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(frameon=False, ncols=2)

        figs_axes.append((fig, ax))

        if save_as:
            fig.savefig(save_as + "/" + d + "timespent_stimvscon.svg", dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return figs_axes

stim_inv = []

# die Daten stammen aus der main.py mit likelihood filter_value > 0.8
# und zeigen die Zeit pro Modul anteilig zur gesamtzeit des jeweiligen Versuchs
# d.h. stim_modul + con_modul + maus auf keiner Kamera = 1
# es gibt vermutlich leichte Abweichungen durch die likelihood filterung + interpolation
# interpolation ist auf 30frames begrenzt (größere misstracking zeiträume bleiben leer)
mouse_125 = {'day1': {'stim_modul': np.float64(0.216), 'con_modul': np.float64(0.212)},
            'day2': {'stim_modul': np.float64(0.28), 'con_modul': np.float64(0.229)}, 
            'day3': {'stim_modul': np.float64(0.292), 'con_modul': np.float64(0.246)}}

mouse_122 = {'day1': {'stim_modul': np.float64(0.282), 'con_modul': np.float64(0.358)},
            'day2': {'stim_modul': np.float64(0.283), 'con_modul': np.float64(0.227)},
            'day3': {'stim_modul': np.float64(0.257), 'con_modul': np.float64(0.437)}}

mouse_121 = {'day1': {'stim_modul': np.float64(0.09), 'con_modul': np.float64(0.128)},
            'day2': {'stim_modul': np.float64(0.267), 'con_modul': np.float64(0.267)},
            'day3': {'stim_modul': np.float64(0.243), 'con_modul': np.float64(0.232)}}

mouse_109 = {'day1': {'stim_modul': np.float64(0.031), 'con_modul': np.float64(0.15)},
            'day2': {'stim_modul': np.float64(0.206), 'con_modul': np.float64(0.298)},
            'day3': {'stim_modul': np.float64(0.194), 'con_modul': np.float64(0.185)}}


# wir beziehen den habituation day auf die Tage 2 und 3 ein
# dafür berechnen wir zuerst den: 
# Präferenzscore P: P(day) = stim_modul(day) - con_modul(day)
m125_p = {'day1': 0.004, 'day2': 0.051, 'day3': 0.046}
m122_p = {'day1': -0.076, 'day2': 0.056, 'day3': -0.18}
m121_p = {'day1': -0.038, 'day2': 0.000, 'day3': 0.011}
m109_p = {'day1': -0.119, 'day2': -0.092, 'day3': 0.009}

# nun machen wir eine baseline korrektur und berechnen die differenz von Tag 2 und Tag 3 mit Tag 1 P(day) - P(day1):
m125_p_corrected = {'day2': 0.047, 'day3': 0.042}
m122_p_corrected = {'day2': 0.132, 'day3': -0.104}
m121_p_corrected = {'day2': 0.038, 'day3': 0.049}
m109_p_corrected = {'day2': 0.027, 'day3': 0.128}


mice_data = {
    "109": mouse_109,
    "121": mouse_121,
    "122": mouse_122,
    "125": mouse_125,
}



mice_order = ["109", "121", "122", "125"]
exp_path = r"Z:\n2023_odor_related_behavior\2025_darcin\Darcin2\raw"

# ---- Deine handberechneten Daten zusammenführen ----
p = {
    "125": m125_p,
    "122": m122_p,
    "121": m121_p,
    "109": m109_p,
}

dp = {
    "125": m125_p_corrected,
    "122": m122_p_corrected,
    "121": m121_p_corrected,
    "109": m109_p_corrected,
}

mice_order = ["109", "121", "122", "125"]

def _default_mouse_colors(mice):
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ["C0","C1","C2","C3"])
    return {m: colors[i % len(colors)] for i, m in enumerate(mice)}

colors = _default_mouse_colors(mice_order)

# =========================
# Plot A: baseline-korrigierte ΔP (Day2 & Day3)
# =========================
days_dp = ["day2", "day3"]
x = np.arange(len(days_dp))

fig, ax = plt.subplots(figsize=(6.5, 4.5))

for m in mice_order:
    y = [float(dp[m][d]) for d in days_dp]
    ax.plot(x, y, marker="o", linewidth=1.5, color=colors[m], label=f"Mouse {m}")

# Gruppenmittel als Referenz (optional)
#group_mean = [np.mean([float(dp[m][d]) for m in mice_order]) for d in days_dp]
#group_sd   = [np.std([float(dp[m][d]) for m in mice_order], ddof=1) for d in days_dp]
#ax.plot(x, group_mean, linewidth=3)                 # bewusst ohne Farbsetzung
#ax.errorbar(x, group_mean, yerr=group_sd, capsize=4, linewidth=0)

ax.axhline(0, linewidth=1, color='black', ls='--')
ax.set_xticks(x)
ax.set_xticklabels(["Day2", "Day3"])
ax.set_ylabel("ΔP = (stim - con) - baseline (day1)")
ax.set_title("Baseline-corrected preference shift")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(frameon=False, ncols=2)
fig.savefig(exp_path + "/modul_preferenceindex_baselinecorrected.svg", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# Plot B: Roh-Präferenz P (Day1–Day3)
# =========================
days_p = ["day1", "day2", "day3"]
x = np.arange(len(days_p))

fig, ax = plt.subplots(figsize=(6.5, 4.5))

for m in mice_order:
    y = [float(p[m][d]) for d in days_p]
    ax.plot(x, y, marker="o", linewidth=1.5, color=colors[m], label=f"Mouse {m}")

ax.axhline(0, linewidth=1, color='black', ls='--')
ax.set_xticks(x)
ax.set_xticklabels(["Day1", "Day2", "Day3"])
ax.set_ylabel("P = stim - con")
ax.set_title("Raw preference score across days")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(frameon=False, ncols=2)
fig.savefig(exp_path + "/modul_preferenceindex.svg", dpi=300, bbox_inches="tight")
plt.show()

# Funktion 1: über Tage, nur Stimulus
#plot_days_one_side(mice_data, mice_order, side="stim_modul", title="Stimulus chamber (Day1–Day3)", save_as=exp_path+"/time_spent_stim.svg")

# Funktion 1: über Tage, nur Control
#plot_days_one_side(mice_data, mice_order, side="con_modul", title="Control chamber (Day1–Day3)", save_as=exp_path+"/time_spent_con.svg")

# Funktion 2: pro Tag ein Plot mit Stim vs Control verbunden
#plot_each_day_stim_vs_con(mice_data, mice_order, save_as=exp_path)
"""

mouse_125 = {'day1': {'stim_dish': 619, 'con_dish': 496},
             'day2': {'stim_dish': 774, 'con_dish': 1053},
             'day3': {'stim_dish': 1970, 'con_dish': 763}}

mouse_122 = {'day1': {'stim_dish': 1121, 'con_dish': 1964},
             'day2': {'stim_dish': 1180, 'con_dish': 1190},
            'day3': {'stim_dish': 2723, 'con_dish': 3753}}

mouse_121 = {'day1': {'stim_dish': 17, 'con_dish': 516},
             'day2': {'stim_dish': 1282, 'con_dish': 1242},
             'day3': {'stim_dish': 1294, 'con_dish': 1019}}

mouse_109 = {'day1': {'stim_dish': 135, 'con_dish': 183},
             'day2': {'stim_dish': 770, 'con_dish': 1382},
             'day3': {'stim_dish': 640, 'con_dish': 391}}

mice_data = {
    "109": mouse_109,
    "121": mouse_121,
    "122": mouse_122,
    "125": mouse_125,
}
"""
"""
# Funktion 1: nur Stimulus-Dish über die Tage
plot_days_one_metric(
    mice_data, mice_order,
    metric_key="stim_dish",
    title="Investigation: Stimulus dish (Day1–Day3)",
    save_as=exp_path+"/invtime_stim.svg"
)

# Funktion 1: nur Control-Dish über die Tage
plot_days_one_metric(
    mice_data, mice_order,
    metric_key="con_dish",
    title="Investigation: Control dish (Day1–Day3)",
    save_as=exp_path+"/invtime_con.svg"
)

# Funktion 2: pro Tag Stim vs Control (verbunden)
plot_each_day_two_metrics(
    mice_data, mice_order,
    key_a="stim_modul", key_b="con_modul",
    title_prefix="Time spent: Stim vs Control",
    save_as=exp_path
)
"""
